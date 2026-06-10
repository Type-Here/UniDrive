#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
new_orchestrator.py — experimental control authority
----------------------------------------------------
Subclass of `orchestrator.Orchestrator` that keeps all the proven machinery
(remap, drift correction, pure-pursuit `_carrot`/`_pursuit_twist`, callbacks,
publishers, in-place junction spin) and replaces ONLY the per-tick decision
logic with two new ideas:

  1. Disagreement-driven blend.  Instead of a binary alpha that ignores the map
     until the robot is on top of a junction node, the lane/map blend weight is
     driven by the *cosine similarity* `c` between where lane detection wants to
     go (`l`) and where the map wants to go (`m`).  The map starts pulling toward
     a turn as soon as the two headings diverge — well before the node.

  2. GPS-style replan.  When the robot drifts far off the planned path, instead
     of dragging it cross-country back to the nearest point of the OLD path
     (pure-pursuit), the orchestrator republishes a goal
     `[closest_map_node, end_id]` to `/waypoint_manager/goal`, which re-runs
     Dijkstra from the current position to the same destination.

Blend convention here (note: INVERSE of the parent's `alpha`):
    out = a_lane * lane + (1 - a_lane) * map
    a_lane = 1 -> pure lane,   a_lane = 0 -> pure map.

Lane vector `l` is a proxy reconstructed from the lane controller's `angular.z`
(a steering rate), since the lane node does not publish a heading:
    theta_l = clamp(angular.z / map_max_angular, -1, 1) * proxy_max
    theta_m = angle_diff(heading_to_next, robot_yaw)
    c       = cos(theta_l - theta_m)
`c` only *selects* a_lane; the blend itself stays on the scalar angular.z.

States
------
  IDLE · NAVIGATING · JUNCTION · ROUNDABOUT · FALLBACK · EMERGENCY_STOP · DONE

This file is standalone: it imports the parent read-only (the parent only spins
up a ROS node under its own `__main__`) and touches no other module.  Run it in
place of `orchestrator.py` (never both at once — same node name).
"""

from __future__ import print_function
import math
from collections import namedtuple

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Int32MultiArray, String  # noqa: F401 (String kept for parity)

from orchestrator import Orchestrator, angle_diff, clamp


# Per-tick snapshot passed to the state handlers (immutable, no re-locking).
_Ctx = namedtuple(
    "_Ctx",
    "rx ry ryaw dist is_junction heading_nxt next_id cur_node_id cur_path_idx "
    "path_ids lane_cmd lane_state lane_fresh lane_usable in_roundabout "
    "theta_m c odom_yaw lane_info")


def decide_blend(lane_angular, theta_m, dist, in_junction,
                 map_max_w, proxy_max, junction_influence_radius,
                 turn_full_rad):
    """Pure decision helper — no ROS, unit-testable.

    Returns (a_lane, c):
      a_lane : weight on the lane command in  out = a_lane*lane + (1-a_lane)*map
      c      : cosine similarity between the lane proxy vector and the map vector

    On open road a_lane = 1 (pure lane; map influence comes from FALLBACK/REPLAN).
    During a junction *approach* the map is blended in, distance-gated over
    junction_influence_radius, by how strongly the map wants to turn OR how much
    the lane disagrees with it:

        approach = (infl - dist) / infl                     far=0 .. near=1
        pull     = max( |theta_m| / turn_full ,  1 - c )    how much the map wins
        a_lane   = 1 - approach * pull

    The MAGNITUDE term (|theta_m|/turn_full) is the fix for a sharp ~90deg
    crossway: there lane and map still *agree in direction* (c stays ~0.5), so the
    old cosine-only pull barely engaged and the lane drove straight through. By also
    pulling in proportion to how hard the map wants to turn, the robot arcs into the
    turn progressively — and because it is gated on `dist <= infl` (now wide), it no
    longer depends on `dist` ever reaching `junction_radius` (which a left-offset
    mapped position can keep it from doing, so the in-place spin never armed).
    The DISAGREEMENT term (1 - c) is preserved for the original "steering too soon"
    case (gentle turns where lane and map point different ways).
    """
    mw = map_max_w if map_max_w > 1e-6 else 1.0
    theta_l = clamp(lane_angular / mw, -1.0, 1.0) * proxy_max
    c = math.cos(theta_l - theta_m)
    if in_junction:
        infl = junction_influence_radius
        approach = clamp((infl - dist) / infl, 0.0, 1.0) if infl > 1e-6 else 1.0
        tf       = turn_full_rad if turn_full_rad > 1e-6 else 1.0
        pull     = max(clamp(abs(theta_m) / tf, 0.0, 1.0), 1.0 - clamp(c, 0.0, 1.0))
        a_lane   = 1.0 - approach * pull
    else:
        a_lane = 1.0
    return a_lane, c


def _polar_arc(center, a0, r0, a1, r1, res):
    """Dense points along a polar arc from (a0,r0) to (a1,r1) about `center`.

    Angle is interpolated along the SHORT direction; radius is interpolated
    linearly, so the arc keeps each endpoint's true distance from the center
    (an outlying node -> a locally flatter arc).  Excludes the end point (the
    next segment / final append provides it).  py2-safe.
    """
    cx, cy = center
    da = a1 - a0
    while da >  math.pi: da -= 2.0 * math.pi
    while da < -math.pi: da += 2.0 * math.pi
    arc   = abs(da) * max(0.5 * (r0 + r1), 1e-3)
    steps = max(2, int(math.ceil(arc / max(res, 1e-3))))
    pts = []
    for s in range(steps):
        f = s / float(steps)
        a = a0 + da * f
        r = r0 + (r1 - r0) * f
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def radial_ring_curve(center, nodes, res):
    """Dense polyline through ring `nodes` via per-segment polar interpolation.

    Pure function (no ROS), py2-safe.  `nodes` are (x,y) in traversal order.
    Because each node keeps its own radius about `center`, an outlying node
    (e.g. the higher east node) yields a flatter local arc while a tighter node
    yields a sharper one — the "smooth radial" behaviour.  Passes through every
    node exactly (polar(node) about any centre reproduces the node).
    """
    cx, cy = center
    polar = [(math.atan2(y - cy, x - cx), math.hypot(x - cx, y - cy))
             for (x, y) in nodes]
    out = []
    for i in range(len(nodes) - 1):
        a0, r0 = polar[i]
        a1, r1 = polar[i + 1]
        out.extend(_polar_arc(center, a0, r0, a1, r1, res))
    out.append((nodes[-1][0], nodes[-1][1]))
    return out


def _densify(p, q, res):
    """Dense points from p to q (excludes q); for straight lead-in/out segments."""
    d     = math.hypot(q[0] - p[0], q[1] - p[1])
    steps = max(1, int(math.ceil(d / max(res, 1e-3))))
    return [(p[0] + (q[0] - p[0]) * s / float(steps),
             p[1] + (q[1] - p[1]) * s / float(steps)) for s in range(steps)]


class NewOrchestrator(Orchestrator):

    # Extra states beyond the parent's IDLE/NAVIGATING/JUNCTION/FALLBACK/DONE.
    ROUNDABOUT     = "ROUNDABOUT"
    EMERGENCY_STOP = "EMERGENCY_STOP"   # terminal failsafe halt (ends navigation)

    def __init__(self):
        super(NewOrchestrator, self).__init__()

        ns = "orchestrator/"
        def rp(k, d):
            return rospy.get_param(ns + k, d)

        # Off-route threshold t (m).  >= t -> FALLBACK ; >= 2t -> REPLAN.
        self._offroute_t   = float(rp("offroute_t", 0.40))
        # Consecutive conflict ticks (c < 0) before EMERGENCY_STOP (debounce).
        self._conflict_ticks = int(rp("conflict_ticks", 12))
        # Conflict stop only counts while the robot is genuinely ON the planned
        # path. Off-route (e.g. after cutting a corner) the bearing-to-next-node is
        # large, so c collapses to cos(theta_m) and a straight, recovering lane reads
        # as a >90deg "conflict" even though nothing is wrong-way — a recovery case
        # FALLBACK owns. So only count the conflict when the cross-track to the path
        # is small enough that the map heading is a trustworthy "this is the way".
        self._conflict_onpath_m = float(rp("conflict_onpath_m", 0.35))
        # angular.z -> implied heading proxy: full steer maps to this many deg.
        self._proxy_max    = math.radians(float(rp("proxy_max_deg", 80.0)))
        # Distance over which the map progressively blends the robot INTO a junction
        # turn. This approach-blend is what actually curves the robot before the
        # in-place spin arms at junction_radius. It is bounded ABOVE by the length of
        # the segment leading into the junction node: too wide and the blend starts
        # pulling toward the post-junction heading while the robot is still maneuvering
        # the PREVIOUS node (e.g. node 6 is fed by the 0.75 m 5->6 segment with an 18deg
        # bend at node 5 — 0.90 reached back past node 5 onto the opposite straight).
        # The window WIDTH is not what makes the turn engage — the magnitude term in
        # decide_blend is: within 0.50 m |theta_m| is already large so the pull is
        # strong, where the old cosine-only blend gave ~1 weak tick. Keep < ~0.75 m.
        self._junction_influence_radius = float(rp("junction_influence_radius", 0.50))
        # Map heading error at which the map fully takes over the approach blend (the
        # magnitude term: |theta_m| >= this -> pull = 1). Sized below a 90deg crossway
        # so a real turn pulls hard while a near-straight junction barely pulls.
        self._junc_turn_full = math.radians(float(rp("junction_turn_full_deg", 50.0)))
        # Junction anti-cut guardrail (camera-frame, NAVIGATING approach only): while
        # the line on the INSIDE of the turn is still confidently seen near the robot
        # centre, scale the turn command toward straight so the robot waits until that
        # line clears (the intersection opens) before committing — your "go straight a
        # bit until the left lane is no longer a problem". Returns scale 1.0 (no damp)
        # unless a turn is actually intended and the inside line is close.
        self._junc_lane_correct = bool (rp("junction_lane_correct", True))
        self._junc_inside_clear = float(rp("junction_inside_clear", 0.30))
        self._junc_lane_gain    = float(rp("junction_lane_gain",    1.0))
        self._junc_lane_floor   = float(rp("junction_lane_floor",   0.0))
        # Lateral map re-centering (idea 3): GPS-style. When the camera is confidently
        # centred on a two-line lane on a STRAIGHT, the robot sits on the lane
        # centreline = the map edge, so nudge the remap translation perpendicular onto
        # the edge (EMA). Removes the lateral drift that skips junctions / inflates
        # dist. Heavily gated to straights (off in turns/junctions/roundabout) so it
        # can't pull a sparse-node curve onto a chord; lateral only, so along-track
        # node advancement is untouched. Propagates to WM via /remap_transform.
        self._lat_correct        = bool (rp("lateral_correct_enable", True))
        # Run only every N NAVIGATING ticks (drift is slow; no need at 25 Hz). The
        # per-correction step (alpha) is sized larger to compensate so net settling
        # stays useful: 0.35 every 50 ticks (~2 s) ~= 35% of the remaining offset / 2 s.
        self._lat_period         = int  (rp("lateral_correct_period", 50))
        self._lat_alpha          = float(rp("lateral_correct_alpha",  0.35))
        self._lat_centered_clear = float(rp("lateral_centered_clear", 0.15))
        self._lat_align_deg      = float(rp("lateral_align_deg",      15.0))
        self._lat_min_m          = float(rp("lateral_min_correct_m",  0.03))
        self._lat_max_m          = float(rp("lateral_max_correct_m",  0.40))
        self._lat_tick           = 0    # NAVIGATING-tick counter for the period gate
        # Min seconds between GPS-style replans (avoid spamming the planner).
        self._replan_cooldown = float(rp("replan_cooldown", 3.0))
        # Roundabout conflict is judged more leniently (lane is unreliable there).
        self._roundabout_conflict_c = float(rp("roundabout_conflict_c", -0.5))
        # Roundabout curve (item 1): follow a smooth per-segment radial arc through
        # the ring nodes instead of aiming at sparse chords (which cuts the circle).
        # Small lookahead so the carrot hugs the curve; fine sample spacing.
        self._round_lookahead   = float(rp("roundabout_lookahead_m", 0.25))
        self._round_spline_res  = float(rp("roundabout_spline_res_m", 0.03))
        # Roundabout lane guardrail (item 2): camera-frame correction, ROUNDABOUT
        # only. NOT lane-following — it only nudges away from a road edge we are
        # about to cross. If the relevant painted line isn't confidently seen, it
        # does nothing and the map-frame curve drives (e.g. the unreliable outer
        # line at the entrance is simply ignored). inner = island side, outer =
        # road edge; the side is decided from where the roundabout centre is.
        self._round_lane_correct = bool (rp("roundabout_lane_correct", True))
        self._round_outer_clear  = float(rp("roundabout_outer_clear", 0.25))
        self._round_inner_clear  = float(rp("roundabout_inner_clear", 0.20))
        self._round_lane_gain    = float(rp("roundabout_lane_gain",   0.40))
        self._round_lane_max     = float(rp("roundabout_lane_max",    0.50))
        # Roundabout off-reference failsafe: how far the robot may stray from its OWN
        # reference (the radial spline on the ring, or the path on the exit leg) before
        # it trips the SHARED stop failsafe (EMERGENCY_STOP). Debounced. Lane is
        # unreliable in the ring, so this geometric check — not a lane check — is the
        # only safety net here; without it a bad map/odom drive just runs away off-road
        # (the observed failure). Generous threshold so only a genuine excursion trips.
        self._round_offref_m     = float(rp("roundabout_offref_m", 0.40))
        self._round_offref_ticks = int  (rp("roundabout_offref_ticks", 10))
        # Off-route / REPLAN is localization-dependent. OFF by default: it must
        # never preempt a healthy lane, and with an imperfect remap a geometric
        # check can mis-fire. When enabled it only triggers a debounced REPLAN
        # (republish goal) — it never seizes steering from the lane.
        self._offroute_enable = bool(rp("offroute_enable", False))
        self._offroute_ticks  = int (rp("offroute_ticks", 10))

        self._conflict_count   = 0      # consecutive conflict ticks
        self._emergency        = False  # terminal emergency-stop latch (cleared by a new goal)
        self._offroute_count   = 0      # consecutive off-route ticks
        self._round_offref_count = 0    # consecutive off-reference ticks (roundabout failsafe)
        self._round_offset       = 0.0  # latest robot distance to its roundabout reference (m)
        self._last_replan      = rospy.Time(0)

        # Roundabout curve cache (rebuilt when the ring node set changes, e.g. REPLAN).
        self._round_spline = None       # dense [(x,y), ...] curve through ring nodes
        self._round_key    = None       # tuple of ring node ids the cache was built for
        self._round_center = None       # (cx,cy) center of the ring (for inner/outer side)
        self._round_i      = 0          # monotonic progress index along the polyline
        # Allow a little backward search so odom noise can't make progress jitter,
        # but not enough to latch onto the spatially-near exit (entry/exit are close).
        self._round_back   = max(1, int(0.10 / max(self._round_spline_res, 1e-3)))

        # Latest /lane_controller/info (camera-frame lane geometry; item 2).
        self._lane_info       = None
        self._lane_info_stamp = None
        rospy.Subscriber("/lane_controller/info", Float64MultiArray,
                         self._lane_info_cb, queue_size=1)

        # GPS-style replan goes out on the existing goal topic (no other file changes).
        self._goal_pub = rospy.Publisher(
            "/waypoint_manager/goal", Int32MultiArray, queue_size=1)

        rospy.loginfo(
            "[new_orch] ready: offroute_t=%.2fm conflict_ticks=%d "
            "proxy_max=%.0fdeg junc_infl=%.2fm replan_cd=%.1fs",
            self._offroute_t, self._conflict_ticks,
            math.degrees(self._proxy_max), self._junction_influence_radius,
            self._replan_cooldown)

    # ------------------------------------------------------------ small helpers

    def _lane_info_cb(self, msg):
        if len(msg.data) < 8:
            return
        with self._lock:
            self._lane_info       = list(msg.data)
            self._lane_info_stamp = rospy.Time.now()

    def _offpath_dist(self, rx, ry, path_ids):
        """Cross-track distance from the robot to the nearest path *segment* (m).

        Distance to the nearest node over-estimates off-route by up to half the
        node spacing on a straight; distance to the polyline does not, so this is
        the metric the off-route check uses.
        """
        if self._map is None or len(path_ids) < 1:
            return float("inf")
        if len(path_ids) == 1:
            nx, ny = self._map.node_xy(path_ids[0])
            return math.hypot(rx - nx, ry - ny)
        best = float("inf")
        ax, ay = self._map.node_xy(path_ids[0])
        for i in range(1, len(path_ids)):
            bx, by = self._map.node_xy(path_ids[i])
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            if seg2 < 1e-12:
                d = math.hypot(rx - ax, ry - ay)
            else:
                t = clamp(((rx - ax) * dx + (ry - ay) * dy) / seg2, 0.0, 1.0)
                d = math.hypot(rx - (ax + t * dx), ry - (ay + t * dy))
            if d < best:
                best = d
            ax, ay = bx, by
        return best

    def _nearest_segment_cross(self, rx, ry, path_ids):
        """Perpendicular vector from (rx,ry) to the nearest path segment (map frame).

        Returns (cross_dx, cross_dy, seg_yaw, cross_dist) for the nearest segment whose
        perpendicular foot falls WITHIN the segment, else None. `cross` points from the
        robot toward the segment line and is purely perpendicular to it (so applying it
        shifts the position laterally only — never along-track). Used by the lateral
        re-centering; foot-within-segment keeps it from grabbing a far node off the end.
        """
        if self._map is None or len(path_ids) < 2:
            return None
        best = None
        ax, ay = self._map.node_xy(int(path_ids[0]))
        for i in range(1, len(path_ids)):
            bx, by = self._map.node_xy(int(path_ids[i]))
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            if seg2 > 1e-12:
                t = ((rx - ax) * dx + (ry - ay) * dy) / seg2
                if 0.0 <= t <= 1.0:                       # perpendicular foot in segment
                    fx, fy = ax + t * dx, ay + t * dy
                    cdx, cdy = fx - rx, fy - ry           # robot -> line (perpendicular)
                    d = math.hypot(cdx, cdy)
                    if best is None or d < best[3]:
                        best = (cdx, cdy, math.atan2(dy, dx), d)
            ax, ay = bx, by
        return best

    def _apply_lateral_correction(self, C):
        """GPS-style lateral re-centering of the map frame (idea 3), straights only.

        When the camera is confidently centred-tracking a two-line lane on a STRAIGHT
        segment, the robot is on that lane's centreline = the map edge. If the mapped
        position has drifted laterally off the edge, nudge the remap translation
        perpendicular onto it (EMA) so dist/heading to the next node stay truthful and
        the junction logic fires on time (the lateral drift is what skipped node 6 and
        inflated dist). Heavily gated: NAVIGATING only, never in a junction/roundabout
        or while approaching the next node (sparse nodes in turns would pull a curve
        onto a chord), and only on a confident, centred, well-aligned two-line track.
        Lateral ONLY (perpendicular), so along-track node advancement is untouched.
        Runs only once every `lateral_correct_period` NAVIGATING ticks (drift is slow).
        """
        if not self._lat_correct:
            return
        self._lat_tick = (self._lat_tick + 1) % max(1, self._lat_period)
        if self._lat_tick != 0:
            return
        if (self._map is None or C.lane_info is None
                or self._state != self.NAVIGATING or C.in_roundabout or C.is_junction):
            return
        if C.dist <= self._junction_influence_radius:    # stay clear of the turn-in zone
            return
        li = C.lane_info
        if (C.lane_state != "TRACKING_CC" or li[2] <= 0.5 or li[3] <= 0.5
                or abs(li[6]) > self._lat_centered_clear):
            return                                       # need a confident, centred track
        seg = self._nearest_segment_cross(C.rx, C.ry, C.path_ids)
        if seg is None:
            return
        cdx, cdy, seg_yaw, cross = seg
        if abs(angle_diff(seg_yaw, C.ryaw)) > math.radians(self._lat_align_deg):
            return                                       # not driving ALONG a straight
        if cross < self._lat_min_m or cross > self._lat_max_m:
            return                                       # noise floor / broken-localization reject
        with self._lock:
            odom = self._odom_pos
        if odom is None:
            return
        # Target = the perpendicular FOOT on the segment (lateral only, never the node).
        # Solve for the exact remap translation that maps the current ODOM point onto
        # the foot, then EMA toward it — same convention as the parent's drift fix
        # (map = R(-theta)/scale * (odom - t)  =>  t = odom - scale*R(theta)*map_pt).
        # NOTE: t lives in the ODOM frame; the earlier version added a MAP-frame vector
        # straight onto t (wrong frame AND sign), which injected along-track error and
        # pushed AWAY from the path — the "thinks it's behind / never recovers" bug.
        foot_x, foot_y = C.rx + cdx, C.ry + cdy
        theta = self._remap_theta
        scale = self._remap_scale if self._remap_scale != 0.0 else 1.0
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        tx_exact = odom[0] - scale * (cos_t * foot_x - sin_t * foot_y)
        ty_exact = odom[1] - scale * (sin_t * foot_x + cos_t * foot_y)
        self._remap_tx = (1.0 - self._lat_alpha) * self._remap_tx + self._lat_alpha * tx_exact
        self._remap_ty = (1.0 - self._lat_alpha) * self._remap_ty + self._lat_alpha * ty_exact
        self._publish_remap()
        rospy.loginfo_throttle(
            1.0, "[new_orch] lateral correction: cross=%.3fm -> tx=%.4f ty=%.4f",
            cross, self._remap_tx, self._remap_ty)

    def _turn_angle(self, path_ids, idx, cur_id, next_id):
        """Absolute heading change at the current node (map edge directions)."""
        if self._map is None or idx <= 0 or idx - 1 >= len(path_ids):
            return None
        try:
            px, py   = self._map.node_xy(int(path_ids[idx - 1]))
            cx, cy   = self._map.node_xy(cur_id)
            nx2, ny2 = self._map.node_xy(next_id)
            incoming = math.atan2(cy - py, cx - px)
            outgoing = math.atan2(ny2 - cy, nx2 - cx)
            return abs(angle_diff(outgoing, incoming))
        except Exception:
            return None

    def _maybe_replan(self, rx, ry, path_ids):
        """Republish goal [closest_map_node, end_id] to trigger a fresh Dijkstra."""
        if self._map is None or not path_ids:
            return False
        now = rospy.Time.now()
        if (now - self._last_replan).to_sec() < self._replan_cooldown:
            return False
        start_id, _ = self._map.closest_node(rx, ry)
        end_id = int(path_ids[-1])
        if start_id is None or int(start_id) == end_id:
            return False
        msg = Int32MultiArray()
        msg.data = [int(start_id), end_id]
        self._goal_pub.publish(msg)
        self._last_replan = now
        rospy.logwarn("[new_orch] REPLAN: %d -> %d (off-route)", int(start_id), end_id)
        return True

    # state-entry shortcuts -------------------------------------------------

    def _enter_fallback(self, why):
        if self._state != self.FALLBACK:
            self._state          = self.FALLBACK
            self._hold_count     = 0
            self._recv_count     = 0
            self._conflict_count = 0   # off-path excursion is a recovery, not a conflict
            self._publish_orc_state(self.FALLBACK)
            rospy.loginfo("[new_orch] -> FALLBACK (%s)", why)

    def _enter_emergency_stop(self, why="lane vs map conflict"):
        """Terminal failsafe: halt the robot and END navigation (like arrival).

        The debounce upstream already absorbs transient camera/segmentation noise, so a
        trip here is a *sustained* fault that, in practice, never self-recovers. Instead
        of a recoverable hold, we stop the wheels, disable the lane controller, and
        CANCEL the active goal (empty goal -> waypoint_manager IDLE). The robot stays
        stopped and the dashboard shows EMERGENCY_STOP until a NEW goal is issued
        (cleared in _path_cb). The latch is honored at the top of _step.
        """
        self._emergency = True
        self._state     = self.EMERGENCY_STOP
        self._set_lane_enabled(False)              # stop the lane controller driving
        self._cmd_pub.publish(Twist())             # halt the wheels
        self._goal_pub.publish(Int32MultiArray())  # cancel navigation (empty goal)
        self._publish_orc_state(self.EMERGENCY_STOP)
        rospy.logwarn("[new_orch] EMERGENCY STOP: %s -- navigation halted; "
                      "issue a new goal to resume", why)

    def _path_cb(self, msg):
        # A genuinely new goal (non-empty path) clears a latched emergency and resumes.
        super(NewOrchestrator, self)._path_cb(msg)
        if self._emergency and len(msg.data) > 0:
            self._emergency = False
            self._state     = self.NAVIGATING
            rospy.loginfo("[new_orch] EMERGENCY cleared by new goal -> resuming")

    # --------------------------------------------------------------- main FSM

    def _step(self):
        # Terminal emergency stop: hold the robot halted (lane off + zero cmd) and keep
        # the dashboard informed until a NEW goal is issued (cleared in _path_cb). Sits
        # ABOVE everything — including the inactive/DONE handler — so the nav cancel we
        # publish on entry can't bounce us into IDLE and mask the EMERGENCY_STOP state.
        if self._emergency:
            self._set_lane_enabled(False)
            self._cmd_pub.publish(Twist())
            self._publish_orc_state(self.EMERGENCY_STOP)
            return

        # --- snapshot shared state under one lock (mirrors parent) ---
        with self._lock:
            pose       = self._pose
            odom_pos   = self._odom_pos
            odom_yaw   = self._odom_yaw
            lane_cmd   = self._lane_cmd
            lane_state = self._lane_state
            nav_info   = self._nav_info
            path_ids   = list(self._path_ids)
            lane_fresh = self._fresh(self._lane_cmd_stamp)
            info_fresh = self._fresh(self._nav_info_stamp)
            lane_info  = (list(self._lane_info)
                          if self._lane_info is not None
                          and self._fresh(self._lane_info_stamp) else None)

        if pose is None:
            self._cmd_pub.publish(Twist())
            return

        active = (info_fresh and nav_info is not None
                  and nav_info[self._NI_ACTIVE] > 0.5)
        if not active:
            self._handle_inactive(lane_fresh, lane_state, lane_cmd)
            return

        rx, ry, ryaw = pose
        dist         = nav_info[self._NI_DIST]
        is_junction  = nav_info[self._NI_JUNC] > 0.5
        heading_nxt  = nav_info[self._NI_HDG]
        next_id      = int(nav_info[self._NI_NEXT])
        cur_node_id  = int(nav_info[self._NI_NODE])
        cur_path_idx = int(nav_info[self._NI_IDX])

        # Roundabout window (current or next node tagged) — lane unreliable here.
        # Extended through the EXIT STUB: the first non-ring node whose path
        # predecessor is a ring node (e.g. node 23 reached from 28). Keep map/curve
        # authority across it so the lane controller can't hijack the exit drive and
        # follow the ring line off-road — the observed failure. The robot hands back
        # to the lane only once it is genuinely out (predecessor no longer a ring node).
        in_roundabout = (self._map is not None and (
            self._map.is_roundabout_node(cur_node_id) or
            (next_id >= 0 and self._map.is_roundabout_node(next_id)) or
            self._is_exit_stub(cur_node_id, cur_path_idx, path_ids)))
        if in_roundabout != self._in_roundabout:
            self._in_roundabout = in_roundabout
            rospy.loginfo("[new_orch] roundabout: %s (node=%d)",
                          "ENTER" if in_roundabout else "EXIT", cur_node_id)

        # Position drift correction — same trigger/guard as parent (NAVIGATING only).
        if cur_node_id != self._last_corrected_node:
            self._node_corrected      = False
            self._last_corrected_node = cur_node_id
        if (self._state == self.NAVIGATING and not self._node_corrected
                and dist < self._drift_trigger_r and odom_pos is not None
                and self._map is not None and lane_state not in self._LANE_BAD):
            self._apply_drift_correction(cur_node_id, odom_pos[0], odom_pos[1])
            self._node_corrected = True

        # Lane/map agreement.  c is only meaningful with a usable lane reading;
        # otherwise treat as "agreeing" so lane-loss is handled by FALLBACK, not
        # mistaken for a conflict.
        lane_usable = lane_fresh and lane_state not in self._LANE_BAD
        theta_m = angle_diff(heading_nxt, ryaw) if next_id >= 0 else 0.0
        if lane_usable and next_id >= 0:
            mw = self._map_max_w if self._map_max_w > 1e-6 else 1.0
            theta_l = clamp(lane_cmd.angular.z / mw, -1.0, 1.0) * self._proxy_max
            c = math.cos(theta_l - theta_m)
        else:
            c = 1.0

        C = _Ctx(rx, ry, ryaw, dist, is_junction, heading_nxt, next_id,
                 cur_node_id, cur_path_idx, path_ids, lane_cmd, lane_state,
                 lane_fresh, lane_usable, in_roundabout, theta_m, c, odom_yaw,
                 lane_info)

        # Heartbeat: one line/sec so a misbehavior is traceable to its inputs.
        rospy.loginfo_throttle(
            1.0, "[new_orch] st=%s node=%d->%d lane=%s junc=%d round=%d dist=%.2f "
            "hdg_err=%+.0fdeg c=%+.2f", self._state, cur_node_id, next_id,
            lane_state, int(is_junction), int(in_roundabout), dist,
            math.degrees(theta_m), c)

        # --- dispatch ---
        # (The terminal EMERGENCY_STOP is handled at the very top of _step, so it never
        # reaches here.) The roundabout pre-empts everything except a junction spin.
        if in_roundabout and self._state != self.JUNCTION:
            self._h_roundabout(C)
        elif self._state == self.JUNCTION:
            self._h_junction(C)
        elif self._state == self.FALLBACK:
            self._h_fallback(C)
        else:
            self._h_navigating(C)

    # ----------------------------------------------------------- inactive / idle

    def _handle_inactive(self, lane_fresh, lane_state, lane_cmd):
        if self._state not in (self.IDLE, self.DONE):
            rospy.loginfo("[new_orch] nav inactive -> DONE")
            self._set_lane_enabled(False)
            self._cmd_pub.publish(Twist())
            self._state          = self.DONE
            self._hold_count     = 0
            self._recv_count     = 0
            self._conflict_count = 0
            self._freerun        = False
        self._publish_orc_state(self.IDLE)
        # Freerun: pass lane commands through when the dashboard enables lane directly.
        if lane_fresh and lane_state not in self._LANE_BAD:
            self._freerun = True
            self._cmd_pub.publish(lane_cmd)
        elif self._freerun:
            self._freerun = False
            self._cmd_pub.publish(Twist())

    # ------------------------------------------------------------- NAVIGATING

    def _h_navigating(self, C):
        if self._state != self.NAVIGATING:
            self._state = self.NAVIGATING
            self._set_lane_enabled(True)
            rospy.loginfo("[new_orch] -> NAVIGATING")
        self._publish_orc_state(self.NAVIGATING)

        # GPS-style lateral re-centering of the map frame (idea 3). Gated to confident
        # straights; lateral only. Done here (lane drives on a straight, a_lane=1) so a
        # frame nudge causes no steering jerk while it makes dist/heading truthful.
        self._apply_lateral_correction(C)

        # 1) Direct conflict (lane vs map > 90 deg apart) — debounced stop.
        #    SUPPRESSED while the current node is a junction: there the map heading
        #    points at the *post-junction* node, so lane (straight) and map (into the
        #    turn) are *expected* to diverge past 90 deg. That divergence is the turn
        #    signal, not a wrong-way conflict — counting it stops the robot dead in
        #    the middle of every sharp turn (observed). A genuine wrong-way is still
        #    caught once past the junction (is_junction clears -> counting resumes).
        #    ALSO gated on being ON the path: off-route (e.g. after cutting a corner)
        #    the bearing-to-next-node is large, so c collapses to cos(theta_m) and a
        #    straight, recovering lane reads as a >90deg conflict even though nothing
        #    is wrong-way. That excursion is a recovery FALLBACK owns; counting it
        #    here terminally stopped the robot mid-recovery (observed). When off the
        #    path the map heading isn't a trustworthy "this is the way" signal, so we
        #    only judge a conflict while the cross-track to the path is small.
        on_path = self._offpath_dist(C.rx, C.ry, C.path_ids) <= self._conflict_onpath_m
        if (C.lane_usable and C.next_id >= 0 and not C.is_junction
                and on_path and C.c < 0.0):
            self._conflict_count += 1
            if self._conflict_count >= self._conflict_ticks:
                self._enter_emergency_stop()
                return
        else:
            self._conflict_count = 0

        # 2) Off-route safety net (opt-in, localization-dependent).
        #    A sustained large cross-track error means we genuinely drove off the
        #    planned route -> republish the goal so Dijkstra reroutes from here.
        #    This does NOT take steering from the lane (lane stays primary); it
        #    just refreshes the path. True lane loss is still handled in step 4.
        if self._offroute_enable:
            off = self._offpath_dist(C.rx, C.ry, C.path_ids)
            if off >= 2.0 * self._offroute_t:
                self._offroute_count += 1
                if (self._offroute_count >= self._offroute_ticks
                        and self._maybe_replan(C.rx, C.ry, C.path_ids)):
                    self._offroute_count = 0
            else:
                self._offroute_count = 0

        # 3) Sharp-junction in-place spin entry (hybrid: gentle turns stay blended).
        if self._junction_entry(C):
            self._state      = self.JUNCTION
            self._hold_count = 0
            rospy.loginfo("[new_orch] -> JUNCTION (dist=%.2fm)", C.dist)
            self._h_junction(C)
            return

        # 4) Blend weight.
        if C.next_id < 0:
            # Final segment: pure lane until the goal distance stop.
            self._hold_count = 0
            a_lane = 1.0
        elif C.lane_state in self._LANE_LOST:
            # Debounced lane loss ramps toward map, then escalates to FALLBACK.
            self._hold_count += 1
            if self._hold_count >= self._hold_ramp:
                self._enter_fallback("lane lost")
                self._cmd_pub.publish(self._pursuit_twist((C.rx, C.ry, C.ryaw), C.path_ids))
                return
            a_lane = 1.0 - (self._hold_count / float(self._hold_ramp))
        else:
            self._hold_count = 0
            # Distance-gated map pull while approaching a junction node.
            in_approach = C.is_junction and C.next_id >= 0
            a_lane, _ = decide_blend(
                C.lane_cmd.angular.z, C.theta_m, C.dist, in_approach,
                self._map_max_w, self._proxy_max, self._junction_influence_radius,
                self._junc_turn_full)

        lane_ang = C.lane_cmd.angular.z if C.lane_fresh else 0.0
        lane_lin = C.lane_cmd.linear.x  if C.lane_fresh else 0.0
        map_ang  = self._map_angular(C.heading_nxt, C.ryaw) if C.next_id >= 0 else 0.0

        blended = a_lane * lane_ang + (1.0 - a_lane) * map_ang
        # Anti-cut guardrail: hold the turn straighter until the inside line clears.
        scale   = self._junction_anti_cut_scale(C)

        twist = Twist()
        twist.linear.x  = lane_lin
        twist.angular.z = blended * scale
        # Shows when the map is overriding the lane (a_lane low), when the guardrail is
        # holding the turn (scale < 1), and the resulting drive/turn.
        rospy.loginfo_throttle(
            1.0, "[new_orch] NAV a_lane=%.2f scale=%.2f lane(v=%.2f w=%+.2f) "
            "map_w=%+.2f -> v=%.2f w=%+.2f", a_lane, scale, lane_lin, lane_ang,
            map_ang, twist.linear.x, twist.angular.z)
        self._cmd_pub.publish(twist)

    def _junction_entry(self, C):
        """Whether to start an in-place rotation at the current junction node."""
        if not (C.is_junction and C.next_id >= 0 and C.dist <= self._junc_radius):
            return False
        if C.cur_node_id == self._handled_junction or C.in_roundabout:
            return False
        ta = self._turn_angle(C.path_ids, C.cur_path_idx, C.cur_node_id, C.next_id)
        if ta is not None and ta < math.radians(self._gentle_turn_deg):
            return False  # gentle turn — let the blend handle it
        return True

    def _junction_anti_cut_scale(self, C):
        """Scale (<=1) on the junction turn command to avoid cutting the inside line.

        During a junction turn the painted line on the side we are turning TOWARD
        (the inside line) is the one we would cut by turning too soon. While that
        line is still confidently seen near the robot centre, scale the turn toward
        straight (down to `_junc_lane_floor`) so the robot drives on until the line
        clears — i.e. until the intersection opens — before committing the turn. Once
        the inside line goes invalid or moves away from centre, scale returns to 1.0
        and the blend's full turn-in takes over.

        Returns 1.0 (no damping) when: disabled, no fresh lane info, not at a junction,
        no real turn intended (map heading still ~forward), or the inside line is
        already clear / not seen. Camera-frame, NAVIGATING approach only.
        """
        if (not self._junc_lane_correct or C.lane_info is None
                or not C.is_junction or C.next_id < 0):
            return 1.0
        # Only damp once a genuine turn is intended; below this the lane is merely
        # centering on the approach and must not be held back.
        if abs(C.theta_m) < math.radians(20.0):
            return 1.0
        turn_left = C.theta_m > 0.0      # map wants left (+w) -> inside line is LEFT
        li = C.lane_info
        inside_valid = (li[2] > 0.5) if turn_left else (li[3] > 0.5)
        inside_off   =  li[4]        if turn_left else  li[5]
        if not inside_valid or abs(inside_off) >= self._junc_inside_clear:
            return 1.0                   # inside line clear (or unseen) -> commit turn
        sev = (self._junc_inside_clear - abs(inside_off)) / self._junc_inside_clear
        return clamp(1.0 - self._junc_lane_gain * sev, self._junc_lane_floor, 1.0)

    # --------------------------------------------------------------- JUNCTION

    def _h_junction(self, C):
        # In-place rotation toward next-waypoint heading (kept from parent).
        self._publish_orc_state(self.JUNCTION)
        err      = angle_diff(C.heading_nxt, C.ryaw)
        spin_ang = clamp(1.5 * err, -self._junc_spin, self._junc_spin)

        if abs(err) < math.radians(self._junc_align):
            self._handled_junction = C.cur_node_id
            self._state = self.NAVIGATING
            self._publish_orc_state(self.NAVIGATING)
            self._apply_theta_correction(C.heading_nxt, C.odom_yaw)
            rospy.loginfo("[new_orch] JUNCTION aligned (err=%.1fdeg) -> NAVIGATING",
                          math.degrees(abs(err)))
            return

        lane_ang = C.lane_cmd.angular.z if C.lane_usable else 0.0
        twist = Twist()
        twist.linear.x  = 0.0
        twist.angular.z = (1.0 - self._alpha_junc) * lane_ang + self._alpha_junc * spin_ang
        self._cmd_pub.publish(twist)

    # ------------------------------------------------------------- ROUNDABOUT

    def _is_exit_stub(self, cur_id, idx, path_ids):
        """True when the current target is the first non-ring node right after the
        ring on the path (the exit stub, e.g. 23 reached from 28).

        Keeps map/curve authority across the exit so the lane controller can't grab
        the exit drive and follow the ring line off-road. Detected exactly as the
        user described: the node is outside the roundabout but its path predecessor
        is a ring node (a 'lookahead' on the previous node)."""
        if self._map is None or idx <= 0 or idx >= len(path_ids):
            return False
        if self._map.is_roundabout_node(cur_id):
            return False
        return self._map.is_roundabout_node(int(path_ids[idx - 1]))

    def _ring_ids(self, path_ids):
        """Ordered ring node ids on the path (first..last tagged-roundabout span).

        Takes the contiguous span between the first and last roundabout-tagged
        nodes so the spline covers the whole ring portion of the route, tolerant
        of an untagged node slipping in between.
        """
        if self._map is None or not path_ids:
            return []
        ridx = [i for i, nid in enumerate(path_ids)
                if self._map.is_roundabout_node(int(nid))]
        if not ridx:
            return []
        return [int(path_ids[i]) for i in range(ridx[0], ridx[-1] + 1)]

    def _build_round_curve(self, path_ids, center):
        """Per-segment radial-arc polyline through the ring nodes (MAP frame).

        Each ring segment is a polar arc about `center` keeping each node's own
        radius, so an outlying node gives a flatter local arc and a tighter node a
        sharper one (the "smooth radial").  Straight lead-in/out segments connect
        the entry/exit neighbors so the curve joins the rest of the path.  Returns
        the dense polyline, or None for < 3 ring nodes (caller falls back to nodes).
        """
        if self._map is None or not path_ids:
            return None
        ridx = [i for i, nid in enumerate(path_ids)
                if self._map.is_roundabout_node(int(nid))]
        if not ridx:
            return None
        first, last = ridx[0], ridx[-1]
        nodes = [self._map.node_xy(int(path_ids[i])) for i in range(first, last + 1)]
        if len(nodes) < 3:
            return None
        out = []
        if first > 0:  # straight lead-in from the entry neighbor
            out.extend(_densify(self._map.node_xy(int(path_ids[first - 1])),
                                nodes[0], self._round_spline_res))
        out.extend(radial_ring_curve(center, nodes, self._round_spline_res))
        if last < len(path_ids) - 1:  # straight lead-out to the exit neighbor
            ex = self._map.node_xy(int(path_ids[last + 1]))
            out.extend(_densify(nodes[-1], ex, self._round_spline_res)[1:])
            out.append(ex)
        return out

    def _ensure_round_spline(self, path_ids):
        """Return the cached roundabout polyline + center, rebuilding on ring change."""
        key = tuple(self._ring_ids(path_ids))
        if key != self._round_key:
            self._round_key    = key
            self._round_i      = 0
            self._round_center = None
            self._round_spline = None
            if len(key) >= 3 and self._map is not None:
                pts = [self._map.node_xy(n) for n in key]
                cx  = sum(p[0] for p in pts) / len(pts)
                cy  = sum(p[1] for p in pts) / len(pts)
                self._round_center = (cx, cy)
                self._round_spline = self._build_round_curve(path_ids, (cx, cy))
                rospy.loginfo(
                    "[new_orch] roundabout curve: %d pts / %d nodes %s centre=(%.2f,%.2f)",
                    len(self._round_spline) if self._round_spline else 0,
                    len(key), list(key), cx, cy)
            else:
                rospy.logwarn_throttle(
                    5.0, "[new_orch] roundabout: %d ring node(s) (<3) — node pursuit",
                    len(key))
        return self._round_spline

    def _spline_carrot(self, rx, ry, poly):
        """Lookahead point on the dense roundabout polyline (MAP frame).

        Nearest-point search is forward-only (with a small backward slack) from the
        last progress index, so the carrot can't snap to the spatially-near exit
        portion of the ring and cut straight across.
        """
        if not poly:
            return None
        lo = max(0, self._round_i - self._round_back)
        best_i, best_d = lo, float("inf")
        for i in range(lo, len(poly)):
            px, py = poly[i]
            d = (rx - px) ** 2 + (ry - py) ** 2
            if d < best_d:
                best_d, best_i = d, i
        self._round_i = best_i
        self._round_offset = math.sqrt(best_d)  # deviation from the curve (failsafe)
        accum  = 0.0
        px, py = poly[best_i]
        for i in range(best_i + 1, len(poly)):
            nx, ny = poly[i]
            seg = math.hypot(nx - px, ny - py)
            if accum + seg >= self._round_lookahead:
                frac = (self._round_lookahead - accum) / max(seg, 1e-9)
                return (px + frac * (nx - px), py + frac * (ny - py))
            accum += seg
            px, py = nx, ny
        return poly[-1]

    def _h_roundabout(self, C):
        if self._state != self.ROUNDABOUT:
            self._state = self.ROUNDABOUT
            self._set_lane_enabled(True)  # keep lane as a sensor; output unused
            rospy.loginfo("[new_orch] -> ROUNDABOUT")
        self._publish_orc_state(self.ROUNDABOUT)

        # Strong-conflict guard (lenient threshold) — debounced stop.
        if C.lane_usable and C.next_id >= 0 and C.c < self._roundabout_conflict_c:
            self._conflict_count += 1
            if self._conflict_count >= self._conflict_ticks:
                self._enter_emergency_stop()
                return
        else:
            self._conflict_count = 0

        # Carrot source.
        #   - Ring (next target still a ring node): follow the smooth radial curve
        #     through the ring nodes (built for >=3 ring nodes).
        #   - EXIT approach (next target has LEFT the ring, e.g. 28->23->2): drop the
        #     spline and use plain pure-pursuit straight at the exit node — the user's
        #     "map, pure-pursuit, no radial between 28-23". Why: the spline's tangent
        #     at the last ring node still points AROUND the ring, so with the small
        #     lookahead the carrot holds that tangent and the robot sails PAST the
        #     exit, then snaps ~90deg toward 23 and cuts the edge off-road (observed).
        #     _carrot aims into the exit from the start (smooth turn-in) and, unlike
        #     _spline_carrot, recomputes from the nearest path node every tick — so it
        #     can never latch onto a stale curve endpoint and run away off-road.
        # Same boundary as the guardrail-off test in _round_lane_nudge (kept in sync):
        # when `next` leaves the ring, both the radial AND the guardrail give way.
        exiting = (C.next_id < 0 or (self._map is not None
                   and not self._map.is_roundabout_node(C.next_id)))
        poly = None if exiting else self._ensure_round_spline(C.path_ids)
        if poly:
            carrot = self._spline_carrot(C.rx, C.ry, poly)  # sets self._round_offset
            src    = "curve"
        else:
            carrot = self._carrot(C.rx, C.ry, C.path_ids)
            src    = "exit" if exiting else "nodes"
            # No spline here, so measure deviation against the path the pursuit follows.
            self._round_offset = self._offpath_dist(C.rx, C.ry, C.path_ids)
        if carrot is None:
            self._cmd_pub.publish(Twist())
            return

        # Off-reference failsafe -> the SINGLE shared terminal stop (EMERGENCY_STOP).
        # Lane is unreliable in the roundabout, so a geometric deviation from our own
        # reference (spline on the ring, path on the exit) is the only safety net. A
        # sustained excursion means the map/odom drive has gone wrong; halt and END the
        # run rather than keep driving off-road. Debounced (transient noise absorbed);
        # no auto-recovery — resume only by issuing a new goal from the dashboard.
        if self._round_offset > self._round_offref_m:
            self._round_offref_count += 1
            if self._round_offref_count >= self._round_offref_ticks:
                self._round_offref_count = 0
                self._enter_emergency_stop("off-reference %.2fm (roundabout)"
                                           % self._round_offset)
                return
        else:
            self._round_offref_count = 0
        heading = math.atan2(carrot[1] - C.ry, carrot[0] - C.rx)
        err     = angle_diff(heading, C.ryaw)
        base    = clamp(self._map_kp * err, -self._map_max_w, self._map_max_w)
        # Camera-frame guardrail nudge (additive; 0 unless a road edge is close).
        nudge   = self._round_lane_nudge(C)
        twist = Twist()
        twist.linear.x  = self._roundabout_speed
        twist.angular.z = clamp(base + nudge, -self._map_max_w, self._map_max_w)
        self._cmd_pub.publish(twist)
        rospy.loginfo_throttle(
            1.0, "[new_orch] ROUND src=%s carrot=(%.2f,%.2f) err=%+.0fdeg "
            "base=%+.2f nudge=%+.2f w=%+.2f", src, carrot[0], carrot[1],
            math.degrees(err), base, nudge, twist.angular.z)

    def _round_lane_nudge(self, C):
        """Camera-frame guardrail correction inside the roundabout (rad/s).

        NOT lane-following: returns 0 unless a painted line is confidently seen
        AND close enough to the robot centre that we're about to cross an edge.
          - outer (road-edge) line too close  -> nudge inward (toward the centre)
          - inner (island) line too close     -> nudge outward
        Inner/outer is decided per-tick from which side the ring center is on, so
        it works for CW or CCW travel. If the relevant line isn't seen, the map
        curve drives unaided (e.g. the unreliable outer line at the entrance).
        """
        if (not self._round_lane_correct or self._round_center is None
                or C.lane_info is None):
            return 0.0
        # Near the exit the robot must CROSS the ring's outer boundary to leave;
        # the guardrail would read that as "about to go off-road" and nudge inward,
        # blocking the exit (observed -> went off-road). Within the roundabout window
        # the next target is outside the ring only on the exit leg, so disable the
        # guardrail there and let the radial lead-out + map drive the robot out.
        if C.next_id < 0 or (self._map is not None
                             and not self._map.is_roundabout_node(C.next_id)):
            rospy.loginfo_throttle(1.0, "[new_orch] ROUND guardrail OFF (exit leg)")
            return 0.0
        li = C.lane_info
        left_valid,  left_off  = li[2] > 0.5, li[4]
        right_valid, right_off = li[3] > 0.5, li[5]

        cx, cy = self._round_center
        rel    = angle_diff(math.atan2(cy - C.ry, cx - C.rx), C.ryaw)
        center_left = rel > 0.0          # ring centre is to the robot's left
        if center_left:                  # island on the left, road edge on the right
            inner_valid, inner_off = left_valid,  left_off
            outer_valid, outer_off = right_valid, right_off
            inward = 1.0                 # turning left (+w) heads toward the center
        else:
            inner_valid, inner_off = right_valid, right_off
            outer_valid, outer_off = left_valid,  left_off
            inward = -1.0

        nudge = 0.0
        if outer_valid and abs(outer_off) < self._round_outer_clear:
            sev = (self._round_outer_clear - abs(outer_off)) / self._round_outer_clear
            nudge += inward * self._round_lane_gain * sev
        if inner_valid and abs(inner_off) < self._round_inner_clear:
            sev = (self._round_inner_clear - abs(inner_off)) / self._round_inner_clear
            nudge -= inward * self._round_lane_gain * sev
        return clamp(nudge, -self._round_lane_max, self._round_lane_max)

    # --------------------------------------------------------------- FALLBACK

    def _h_fallback(self, C):
        # Pure-pursuit on the path; recover only with optical + positional proof.
        near_path = C.dist <= self._recovery_radius
        lane_ok   = C.lane_state not in self._LANE_BAD
        if lane_ok and near_path:
            self._recv_count += 1
            if self._recv_count >= self._recovery_ticks:
                self._recv_count = 0
                self._hold_count = 0
                self._state = self.NAVIGATING
                self._publish_orc_state(self.NAVIGATING)
                rospy.loginfo("[new_orch] FALLBACK -> NAVIGATING (lane OK, dist=%.2fm)",
                              C.dist)
                self._h_navigating(C)
                return
            self._publish_orc_state(self.FALLBACK)
        else:
            self._recv_count = 0
            self._publish_orc_state(self.FALLBACK)
        self._cmd_pub.publish(self._pursuit_twist((C.rx, C.ry, C.ryaw), C.path_ids))

    # EMERGENCY_STOP is terminal: entered via _enter_emergency_stop(), held at the top
    # of _step(), and cleared only by a new goal (_path_cb). No per-tick handler / no
    # auto-recovery — a sustained fault past the debounce ends the run by design.


if __name__ == "__main__":
    try:
        NewOrchestrator().run()
    except rospy.ROSInterruptException:
        pass