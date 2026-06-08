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
  IDLE · NAVIGATING · JUNCTION · ROUNDABOUT · FALLBACK · CONFLICT_STOP · DONE

This file is standalone: it imports the parent read-only (the parent only spins
up a ROS node under its own `__main__`) and touches no other module.  Run it in
place of `orchestrator.py` (never both at once — same node name).
"""

from __future__ import print_function
import math
from collections import namedtuple

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray, String  # noqa: F401 (String kept for parity)

from orchestrator import Orchestrator, angle_diff, clamp


# Per-tick snapshot passed to the state handlers (immutable, no re-locking).
_Ctx = namedtuple(
    "_Ctx",
    "rx ry ryaw dist is_junction heading_nxt next_id cur_node_id cur_path_idx "
    "path_ids lane_cmd lane_state lane_fresh lane_usable in_roundabout "
    "theta_m c odom_yaw")


def decide_blend(lane_angular, theta_m, dist, in_junction,
                 map_max_w, proxy_max, junction_influence_radius):
    """Pure decision helper — no ROS, unit-testable.

    Returns (a_lane, c):
      a_lane : weight on the lane command in  out = a_lane*lane + (1-a_lane)*map
      c      : cosine similarity between the lane proxy vector and the map vector

    On open road a_lane = 1 (pure lane; map influence comes from FALLBACK/REPLAN).
    During a junction *approach* the weight is distance-gated: far from the node
    we stay on lane (don't cut the corner early); as we close in AND the lane
    disagrees with the map (the turn is appearing, c drops) the map takes over.
    """
    mw = map_max_w if map_max_w > 1e-6 else 1.0
    theta_l = clamp(lane_angular / mw, -1.0, 1.0) * proxy_max
    c = math.cos(theta_l - theta_m)
    if in_junction:
        infl = junction_influence_radius
        approach = clamp((infl - dist) / infl, 0.0, 1.0) if infl > 1e-6 else 1.0
        a_lane = 1.0 - approach * (1.0 - clamp(c, 0.0, 1.0))
    else:
        a_lane = 1.0
    return a_lane, c


class NewOrchestrator(Orchestrator):

    # Extra states beyond the parent's IDLE/NAVIGATING/JUNCTION/FALLBACK/DONE.
    ROUNDABOUT    = "ROUNDABOUT"
    CONFLICT_STOP = "CONFLICT_STOP"

    def __init__(self):
        super(NewOrchestrator, self).__init__()

        ns = "orchestrator/"
        def rp(k, d):
            return rospy.get_param(ns + k, d)

        # Off-route threshold t (m).  >= t -> FALLBACK ; >= 2t -> REPLAN.
        self._offroute_t   = float(rp("offroute_t", 0.40))
        # Consecutive conflict ticks (c < 0) before CONFLICT_STOP (debounce).
        self._conflict_ticks = int(rp("conflict_ticks", 12))
        # angular.z -> implied heading proxy: full steer maps to this many deg.
        self._proxy_max    = math.radians(float(rp("proxy_max_deg", 80.0)))
        # Distance over which the map progressively pulls into a junction turn.
        self._junction_influence_radius = float(rp("junction_influence_radius", 0.50))
        # Min seconds between GPS-style replans (avoid spamming the planner).
        self._replan_cooldown = float(rp("replan_cooldown", 3.0))
        # Roundabout conflict is judged more leniently (lane is unreliable there).
        self._roundabout_conflict_c = float(rp("roundabout_conflict_c", -0.5))
        # Off-route / REPLAN is localization-dependent. OFF by default: it must
        # never preempt a healthy lane, and with an imperfect remap a geometric
        # check can mis-fire. When enabled it only triggers a debounced REPLAN
        # (republish goal) — it never seizes steering from the lane.
        self._offroute_enable = bool(rp("offroute_enable", False))
        self._offroute_ticks  = int (rp("offroute_ticks", 10))

        self._conflict_count   = 0      # consecutive conflict ticks
        self._conflict_recover = 0      # consecutive recovered ticks in CONFLICT_STOP
        self._offroute_count   = 0      # consecutive off-route ticks
        self._last_replan      = rospy.Time(0)

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
            self._state      = self.FALLBACK
            self._hold_count = 0
            self._recv_count = 0
            self._publish_orc_state(self.FALLBACK)
            rospy.loginfo("[new_orch] -> FALLBACK (%s)", why)

    def _enter_conflict_stop(self):
        self._state            = self.CONFLICT_STOP
        self._conflict_recover = 0
        self._publish_orc_state(self.CONFLICT_STOP)
        rospy.logwarn("[new_orch] -> CONFLICT_STOP (lane vs map conflict)")

    # --------------------------------------------------------------- main FSM

    def _step(self):
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
        in_roundabout = (self._map is not None and (
            self._map.is_roundabout_node(cur_node_id) or
            (next_id >= 0 and self._map.is_roundabout_node(next_id))))
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
                 lane_fresh, lane_usable, in_roundabout, theta_m, c, odom_yaw)

        # Heartbeat: one line/sec so a misbehavior is traceable to its inputs.
        rospy.loginfo_throttle(
            1.0, "[new_orch] st=%s lane=%s junc=%d round=%d dist=%.2f "
            "hdg_err=%+.0fdeg c=%+.2f", self._state, lane_state,
            int(is_junction), int(in_roundabout), dist,
            math.degrees(theta_m), c)

        # --- dispatch ---
        # Roundabout pre-empts everything except an in-progress junction spin.
        if in_roundabout and self._state != self.JUNCTION:
            self._h_roundabout(C)
        elif self._state == self.CONFLICT_STOP:
            self._h_conflict_stop(C)
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

        # 1) Direct conflict (lane vs map > 90 deg apart) — debounced stop.
        if C.lane_usable and C.next_id >= 0 and C.c < 0.0:
            self._conflict_count += 1
            if self._conflict_count >= self._conflict_ticks:
                self._enter_conflict_stop()
                self._cmd_pub.publish(Twist())
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
                self._map_max_w, self._proxy_max, self._junction_influence_radius)

        lane_ang = C.lane_cmd.angular.z if C.lane_fresh else 0.0
        lane_lin = C.lane_cmd.linear.x  if C.lane_fresh else 0.0
        map_ang  = self._map_angular(C.heading_nxt, C.ryaw) if C.next_id >= 0 else 0.0

        twist = Twist()
        twist.linear.x  = lane_lin
        twist.angular.z = a_lane * lane_ang + (1.0 - a_lane) * map_ang
        # Shows when the map is overriding the lane (a_lane low) and whether the
        # output is a forward drive or a stationary spin (v~0, w large).
        rospy.loginfo_throttle(
            1.0, "[new_orch] NAV a_lane=%.2f lane(v=%.2f w=%+.2f) map_w=%+.2f "
            "-> v=%.2f w=%+.2f", a_lane, lane_lin, lane_ang, map_ang,
            twist.linear.x, twist.angular.z)
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
                self._enter_conflict_stop()
                self._cmd_pub.publish(Twist())
                return
        else:
            self._conflict_count = 0

        # Pure map following at roundabout speed.
        carrot = self._carrot(C.rx, C.ry, C.path_ids)
        if carrot is None:
            self._cmd_pub.publish(Twist())
            return
        heading = math.atan2(carrot[1] - C.ry, carrot[0] - C.rx)
        err     = angle_diff(heading, C.ryaw)
        twist = Twist()
        twist.linear.x  = self._roundabout_speed
        twist.angular.z = clamp(self._map_kp * err, -self._map_max_w, self._map_max_w)
        self._cmd_pub.publish(twist)

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

    # ----------------------------------------------------------- CONFLICT_STOP

    def _h_conflict_stop(self, C):
        # Hold a full stop until lane and map agree again for recovery_ticks.
        self._publish_orc_state(self.CONFLICT_STOP)
        if C.lane_usable and C.next_id >= 0 and C.c >= 0.0:
            self._conflict_recover += 1
            if self._conflict_recover >= self._recovery_ticks:
                self._conflict_recover = 0
                self._conflict_count   = 0
                self._state = self.NAVIGATING
                rospy.loginfo("[new_orch] CONFLICT_STOP -> NAVIGATING (agreement restored)")
                self._h_navigating(C)
                return
        else:
            self._conflict_recover = 0
        self._cmd_pub.publish(Twist())


if __name__ == "__main__":
    try:
        NewOrchestrator().run()
    except rospy.ROSInterruptException:
        pass
