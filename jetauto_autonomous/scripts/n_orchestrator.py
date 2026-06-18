#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
n_orchestrator.py — the production orchestrator (sole cmd_vel publisher)
------------------------------------------------------------------------
Self-contained: the base `Orchestrator` class (remap, drift correction,
pure-pursuit, in-place junction spin, and the IDLE/NAVIGATING/JUNCTION/FALLBACK/
DONE FSM) is defined inline below; `NewOrchestrator` extends it. Same node name
as the legacy files — run exactly one orchestrator.

On top of the original disagreement-blend design (radial roundabout,
terminal EMERGENCY_STOP), this version adds:

  1. Real lane heading.  `theta_l` comes from /lane_controller/info[1]
     (`heading_rad`, +ve bends right in image coords -> negated to standard
     CCW yaw) whenever the lane geometry is trustworthy, instead of the
     angular.z steering-rate proxy. The proxy remains the fallback and can be
     forced back with `use_lane_heading: false`.

  2. Yaw drift correction on straights.  The lateral re-centering gates
     (confident centred two-line track, aligned with a straight segment) also
     pin the robot's true map yaw to `seg_yaw + heading_rad`, so `remap_theta`
     is EMA-corrected there — closing the loop on the yaw drift that the
     translation-only correction could never remove (it only chased its
     symptom). Both this and the inherited junction theta fix re-anchor the
     translation so the frame pivots about the ROBOT, not the remap origin.

  3. FALLBACK failsafe.  FALLBACK was the one state that could run away
     forever on broken localization; it now trips the shared EMERGENCY_STOP
     on sustained cross-track excursion or timeout.

  4. Anti-cut damps only the MAP turn-in (lane centering is never scaled
     away), and a per-tick /orchestrator/diag topic separates camera/BEV
     calibration error from odom drift (lane-centred vs map cross-track
     residual).

Blend convention (INVERSE of the parent's `alpha`):
    out = a_lane * lane + (1 - a_lane) * map
    a_lane = 1 -> pure lane,   a_lane = 0 -> pure map.

    theta_l = real lane heading (or angular.z proxy fallback)
    theta_m = angle_diff(heading_to_next, robot_yaw)
    c       = cos(theta_l - theta_m)
`c` only *selects* a_lane; the blend itself stays on the scalar angular.z.

States
------
  IDLE · NAVIGATING · JUNCTION · ROUNDABOUT · FALLBACK · EMERGENCY_STOP · DONE

This file is fully self-contained: the base `Orchestrator` is defined inline
below and `NewOrchestrator` extends it; it touches no other module besides
`map_loader`.
"""

from __future__ import print_function
import json
import math
import os
import threading
from collections import namedtuple

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64MultiArray, Int32MultiArray, String

from map_loader import MapLoader
from object_detection.traffic_sign_handler import TrafficSignHandler


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def angle_diff(a, b):
    """Signed shortest angular difference (a - b) wrapped to (-pi, pi]."""
    d = a - b
    while d >  math.pi: d -= 2.0 * math.pi
    while d < -math.pi: d += 2.0 * math.pi
    return d


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# Per-tick snapshot passed to the state handlers (immutable, no re-locking).
# theta_l: lane direction in the robot frame (real heading or proxy fallback).
# offpath: cross-track distance to the planned path (m) — computed once per
#          tick, shared by the conflict gate, FALLBACK failsafe and diagnostics.
_Ctx = namedtuple(
    "_Ctx",
    "rx ry ryaw dist is_junction heading_nxt next_id cur_node_id cur_path_idx "
    "path_ids lane_cmd lane_state lane_fresh lane_usable in_roundabout "
    "theta_m theta_l c offpath odom_yaw lane_info")


def decide_blend(theta_l, theta_m, dist, in_junction,
                 junction_influence_radius, turn_full_rad):
    """Pure decision helper — no ROS, unit-testable.

    `theta_l` is the lane direction in the robot frame (standard CCW yaw),
    already resolved by the caller (real heading from /lane_controller/info,
    or the angular.z proxy fallback — see NewOrchestrator._lane_theta).

    Returns (a_lane, c):
      a_lane : weight on the lane command in  out = a_lane*lane + (1-a_lane)*map
      c      : cosine similarity between the lane vector and the map vector

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


class Orchestrator(object):

    IDLE       = "IDLE"
    NAVIGATING = "NAVIGATING"
    JUNCTION   = "JUNCTION"
    FALLBACK   = "FALLBACK"
    DONE       = "DONE"

    _LANE_BAD   = frozenset(("HOLD", "STOP", "DISABLED"))  # used for freerun / recovery gates
    _LANE_LOST  = frozenset(("HOLD", "STOP"))               # only these ramp toward FALLBACK

    # nav_info Float64MultiArray slot indices
    _NI_NODE   = 0
    _NI_NEXT   = 1
    _NI_DIST   = 2
    _NI_JUNC   = 3
    _NI_HDG    = 4
    _NI_IDX    = 5
    _NI_LEN    = 6
    _NI_ACTIVE = 7

    def __init__(self):
        rospy.init_node("orchestrator", anonymous=False)

        ns = "orchestrator/"
        def rp(k, d):
            return rospy.get_param(ns + k, d)

        self._rate_hz        = float(rp("rate_hz",            25.0))
        self._stale_timeout  = float(rp("stale_timeout",       0.5))
        # Output limiting: the base /odom integrates the COMMANDED cmd_vel open-loop
        # (no encoders; see docs/odometry_drift_analysis.md), so a command the robot
        # can't physically track becomes map drift. Clamp to the base's manual
        # envelope (driver clamps manual cmd_vel to lin<=0.2, ang<=0.5) and slew-limit
        # the change so commanded ~= executed.
        self._out_max_lin    = float(rp("output_max_linear",   0.20))   # m/s hard clamp
        self._out_max_ang    = float(rp("output_max_angular",  0.50))   # rad/s hard clamp
        # Tighter angular cap for IN-PLACE rotation (linear=0). Manual in-place spins are
        # much slower than the 0.5 general clamp; a slow pivot keeps the open-loop odom
        # faithful (less command-vs-gyro divergence to snap to at spin-stop). Applied by
        # _publish_cmd(..., in_place=True) at the spin sites; normal driving keeps 0.5.
        self._out_max_ang_ip = float(rp("output_max_angular_inplace", 0.20))  # rad/s
        self._out_acc_lin    = float(rp("output_accel_linear", 0.50))   # m/s^2 slew (0=off)
        self._out_acc_ang    = float(rp("output_accel_angular", 3.0))   # rad/s^2 slew (0=off)
        self._last_vx = self._last_vy = self._last_wz = 0.0
        self._last_cmd_t     = None
        self._hold_ramp      = int  (rp("hold_ramp_ticks",     15))
        self._recovery_ticks = int  (rp("recovery_ticks",       5))
        self._junc_radius    = float(rp("junction_radius",     0.30))
        self._junc_align     = float(rp("junction_align_deg",  22.0))
        self._junc_spin      = float(rp("junction_spin_speed",       0.40))
        # Forward creep during in-place spins (m/s). A pure pivot (linear=0) is the
        # max-wheel-scrub, least odom-faithful mecanum motion; a small forward speed
        # turns it into a rolling ARC (wheels roll, low slip) like manual/lane driving,
        # cutting the open-loop yaw drift a spin injects. 0 = pure pivot (old behavior);
        # the turn arcs wider as this grows, so tune on the robot. Applies to junction
        # spins and the roundabout exit-spin.
        self._junc_spin_creep = float(rp("junction_spin_creep",      0.0))
        self._alpha_junc     = float(rp("junction_alpha",            0.9))
        self._lookahead      = float(rp("lookahead_m",               0.50))
        self._recovery_radius = float(rp("fallback_recovery_radius", 0.50))
        self._map_speed      = float(rp("map_drive_speed",      0.04))
        self._map_kp         = float(rp("map_kp",               1.2))
        self._map_max_w      = float(rp("map_max_angular",      0.50))
        # Drift auto-correction (EMA applied at each confirmed node passage / junction)
        self._drift_pos_alpha    = float(rp("drift_pos_alpha",       0.35))
        self._drift_theta_alpha  = float(rp("drift_theta_alpha",     0.30))
        self._drift_min_m        = float(rp("drift_min_correct_m",   0.03))
        self._drift_max_m        = float(rp("drift_max_correct_m",   0.50))
        self._drift_trigger_r    = float(rp("drift_trigger_radius",  0.20))
        # Gyro-bias feed-forward: /odom yaw is the open-loop integral of a bias-prone gyro
        # with NO absolute reference (see odometry_drift_analysis.md), so a constant gyro
        # bias drifts yaw monotonically (~0.1 deg/s, measurable while STOPPED since no
        # commanded rotation masks it — but the same bias integrates while MOVING too). The
        # camera yaw fix is the only thing that re-pins it, but it only fires on steady
        # straights, so the bias accumulates uncorrected through every stop/spin/junction.
        # We cancel the constant part directly: subtract bias_rate * elapsed from the raw
        # odom yaw before anything downstream (map_yaw, theta correction) sees it, leaving
        # the camera fix only the residual (spins + BEV bias) to absorb. SIGN = the observed
        # drift rate: yaw drifting DOWN ~0.1 deg/s -> set -0.1 (deg/s). 0 = disabled.
        self._gyro_bias_rate = math.radians(float(rp("gyro_bias_rate", 0.0)))  # deg/s -> rad/s
        if self._gyro_bias_rate != 0.0:
            rospy.loginfo("[nn_orch] gyro-bias feed-forward ENABLED: %.4f deg/s",
                          math.degrees(self._gyro_bias_rate))

        # Turn-angle-aware junction: skip full rotation for gentle heading changes
        self._gentle_turn_deg   = float(rp("gentle_turn_deg",    20.0))
        # Roundabout: pure map following at this speed (lane detection unreliable)
        self._roundabout_speed  = float(rp("roundabout_drive_speed", 0.10))
        # A sharp exit turn (e.g. 28->23->2 doubles back ~160 deg) can't be driven as a
        # forward pure-pursuit arc (min radius ~ speed/max_w): the robot sweeps wide and
        # ends up perpendicular to the exit road, only partly rotated (observed at 23).
        # Above this heading error to the exit carrot, spin in place first then drive.
        self._round_exit_spin_deg = float(rp("roundabout_exit_spin_deg", 90.0))


        # Remap transform (odom -> map frame), mirrors waypoint_manager
        self._remap_theta = 0.0
        self._remap_scale = 1.0
        self._remap_tx    = 0.0
        self._remap_ty    = 0.0
        self._load_remap_params()

        # Map loader for pure-pursuit coordinate lookup
        self._map = None
        map_file  = rospy.get_param("waypoint_manager/map_file", "")
        if map_file and map_file != "/tmp/UNSET_MAP_FILE":
            try:
                rospy.loginfo("[orchestrator] loading map: %s", map_file)
                self._map = MapLoader(map_file)
            except Exception as e:
                rospy.logwarn("[orchestrator] map load failed (%s); pure-pursuit disabled", e)
        else:
            rospy.logwarn("[orchestrator] map_file not set; pure-pursuit disabled")

        # Shared state — protected by _lock (written by callbacks, read by _step)
        self._lock           = threading.Lock()
        self._pose           = None     # (x, y, yaw) in MAP frame
        self._odom_pos       = None     # raw (x, y) in ODOM frame — for drift correction
        self._odom_yaw       = 0.0      # raw yaw in ODOM frame — for theta correction
        self._lane_cmd       = Twist()
        self._lane_state     = "STOP"
        self._nav_info       = None     # latest data list
        self._path_ids       = []       # current path node IDs
        self._lane_cmd_stamp = None
        self._nav_info_stamp = None

        # FSM state — written only from _step() (no lock needed)
        self._state             = self.IDLE
        self._hold_count        = 0
        self._recv_count        = 0
        self._freerun           = False  # True while passing lane cmds through in no-nav mode
        self._handled_junction    = -1   # node ID of last completed junction; blocks re-trigger
        self._node_corrected      = False  # True once drift correction fired for current node
        self._last_corrected_node = -1     # node ID that was last drift-corrected
        self._in_roundabout       = False  # True while navigating through roundabout nodes

        # Publishers
        self._cmd_pub    = rospy.Publisher(
            "/jetauto_controller/cmd_vel", Twist, queue_size=1)
        self._enable_pub = rospy.Publisher(
            "/lane_controller/enable", Bool, queue_size=1, latch=True)
        self._state_pub      = rospy.Publisher(
            "/orchestrator/state", String, queue_size=1, latch=True)
        self._last_orc_state = ""
        self._remap_pub  = rospy.Publisher(
            "/remap_transform", Float64MultiArray, queue_size=1)

        # Subscribers
        rospy.Subscriber("/lane_controller/cmd_vel",   Twist,
                         self._lane_cmd_cb,   queue_size=1)
        rospy.Subscriber("/lane_controller/state",     String,
                         self._lane_state_cb, queue_size=1)
        rospy.Subscriber("/waypoint_manager/nav_info", Float64MultiArray,
                         self._nav_info_cb,   queue_size=1)
        rospy.Subscriber("/waypoint_manager/path",     Int32MultiArray,
                         self._path_cb,       queue_size=1)
        rospy.Subscriber("/odom",                      Odometry,
                         self._odom_cb,       queue_size=10)
        rospy.Subscriber("/remap_transform",           Float64MultiArray,
                         self._remap_cb,      queue_size=1)

        self._set_lane_enabled(False)
        rospy.loginfo(
            "[orchestrator] ready. rate=%.0fHz  hold_ramp=%d  "
            "recovery=%d  junc_align=%.0fdeg",
            self._rate_hz, self._hold_ramp,
            self._recovery_ticks, self._junc_align)

    # ------------------------------------------------------------------ remap

    def _load_remap_params(self):
        here  = os.path.dirname(os.path.abspath(__file__))
        fpath = os.path.join(here, "..", "web", "remap_params.json")
        try:
            with open(fpath) as f:
                p = json.load(f)
            self._remap_theta = float(p.get("theta", 0.0))
            self._remap_scale = float(p.get("scale", 1.0))
            self._remap_tx    = float(p.get("tx",    0.0))
            self._remap_ty    = float(p.get("ty",    0.0))
            rospy.loginfo(
                "[orchestrator] remap loaded: theta=%.3f scale=%.3f tx=%.3f ty=%.3f",
                self._remap_theta, self._remap_scale,
                self._remap_tx, self._remap_ty)
        except Exception as e:
            rospy.loginfo("[orchestrator] no remap_params.json (%s), using identity", e)

    def _remap_cb(self, msg):
        if len(msg.data) < 4:
            return
        self._remap_theta = float(msg.data[0])
        self._remap_scale = float(msg.data[1])
        self._remap_tx    = float(msg.data[2])
        self._remap_ty    = float(msg.data[3])

    def _odom_to_map(self, x, y):
        """Inverse similarity transform: map = R(-theta)/scale * (odom - t)."""
        theta = self._remap_theta
        scale = self._remap_scale if self._remap_scale != 0.0 else 1.0
        cos_t = math.cos(-theta)
        sin_t = math.sin(-theta)
        dx = x - self._remap_tx
        dy = y - self._remap_ty
        return (cos_t * dx - sin_t * dy) / scale, (sin_t * dx + cos_t * dy) / scale

    # ---------------------------------------------------------------- callbacks

    def _odom_cb(self, msg):
        p   = msg.pose.pose.position
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        # Gyro-bias feed-forward: remove the constant monotonic gyro drift from the raw
        # odom yaw before anything downstream uses it. Accumulates from the first odom msg;
        # angle_diff(.,0) re-normalizes into (-pi, pi]. No-op when gyro_bias_rate == 0.
        if self._gyro_bias_rate != 0.0:
            stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
            if self._odom_t0 is None:
                self._odom_t0 = stamp
            yaw = angle_diff(yaw - self._gyro_bias_rate * (stamp - self._odom_t0).to_sec(), 0.0)
        mx, my  = self._odom_to_map(p.x, p.y)
        map_yaw = angle_diff(yaw, self._remap_theta)
        with self._lock:
            self._pose     = (mx, my, map_yaw)
            self._odom_pos = (p.x, p.y)
            self._odom_yaw = yaw

    def _lane_cmd_cb(self, msg):
        with self._lock:
            self._lane_cmd       = msg
            self._lane_cmd_stamp = rospy.Time.now()

    def _lane_state_cb(self, msg):
        with self._lock:
            self._lane_state = msg.data

    def _nav_info_cb(self, msg):
        if len(msg.data) < 8:
            return
        with self._lock:
            self._nav_info       = list(msg.data)
            self._nav_info_stamp = rospy.Time.now()

    def _path_cb(self, msg):
        with self._lock:
            self._path_ids = list(msg.data)
        self._handled_junction    = -1   # new path resets junction history
        self._node_corrected      = False
        self._last_corrected_node = -1

    # ----------------------------------------------------------------- helpers

    def _set_lane_enabled(self, on):
        self._enable_pub.publish(Bool(data=bool(on)))

    # ----------------------------------------------- hardware cmd_vel output
    # Single guarded path to /jetauto_controller/cmd_vel. Driving commands go through
    # _publish_cmd (envelope clamp + slew limit); hard stops go through _publish_stop
    # (immediate zero, slew state reset so the next motion ramps up from rest).

    @staticmethod
    def _slew(prev, target, max_step):
        """Limit |target - prev| to max_step (0 = limiter off, clamp only)."""
        if max_step <= 0.0:
            return target
        return prev + clamp(target - prev, -max_step, max_step)

    def _publish_cmd(self, twist, in_place=False):
        """Clamp a driving Twist to the base's manual envelope and slew-limit the
        change before publishing, so the open-loop odom stays faithful to execution.

        in_place=True applies the tighter in-place angular cap (output_max_angular_inplace)
        instead of the general 0.5 — manual in-place spins are slow, and a slow pivot keeps
        the command-integrated yaw close to the gyro so less error snaps in at spin-stop."""
        now = rospy.Time.now()
        if self._last_cmd_t is None:
            dt = 1.0 / max(self._rate_hz, 1.0)
        else:
            dt = (now - self._last_cmd_t).to_sec()
            if dt <= 0.0 or dt > 0.5:          # first tick / stall guard
                dt = 1.0 / max(self._rate_hz, 1.0)
        ang_cap = self._out_max_ang_ip if in_place else self._out_max_ang
        vx = clamp(twist.linear.x,  -self._out_max_lin, self._out_max_lin)
        vy = clamp(twist.linear.y,  -self._out_max_lin, self._out_max_lin)
        wz = clamp(twist.angular.z, -ang_cap, ang_cap)
        vx = self._slew(self._last_vx, vx, self._out_acc_lin * dt)
        vy = self._slew(self._last_vy, vy, self._out_acc_lin * dt)
        wz = self._slew(self._last_wz, wz, self._out_acc_ang * dt)
        self._last_vx, self._last_vy, self._last_wz = vx, vy, wz
        self._last_cmd_t = now
        out = Twist()
        out.linear.x, out.linear.y, out.angular.z = vx, vy, wz
        self._cmd_pub.publish(out)

    def _publish_stop(self):
        """Immediate zero (no slew) for hard stops/idle; resets the slew state."""
        self._last_vx = self._last_vy = self._last_wz = 0.0
        self._last_cmd_t = rospy.Time.now()
        self._cmd_pub.publish(Twist())

    def _publish_orc_state(self, s):
        if s != self._last_orc_state:
            self._state_pub.publish(String(data=s))
            self._last_orc_state = s

    def _fresh(self, stamp):
        return (stamp is not None and
                (rospy.Time.now() - stamp).to_sec() < self._stale_timeout)

    def _map_angular(self, heading_to_next, robot_yaw):
        """P-controller toward map heading, clamped."""
        err = angle_diff(heading_to_next, robot_yaw)
        return clamp(self._map_kp * err, -self._map_max_w, self._map_max_w)

    # --------------------------------------------------- drift auto-correction

    def _publish_remap(self):
        """Broadcast current remap params so waypoint_manager stays in sync."""
        msg = Float64MultiArray()
        msg.data = [self._remap_theta, self._remap_scale,
                    self._remap_tx,    self._remap_ty]
        self._remap_pub.publish(msg)

    def _apply_drift_correction(self, node_id, ox, oy):
        """EMA-correct tx/ty using a confirmed node passage as a position fix point."""
        if self._map is None:
            return
        try:
            nx, ny = self._map.node_xy(node_id)
        except Exception:
            return
        mx, my    = self._odom_to_map(ox, oy)
        pos_error = math.hypot(mx - nx, my - ny)
        if pos_error < self._drift_min_m or pos_error > self._drift_max_m:
            return
        # Compute exact (tx, ty) that would map (ox, oy) to (nx, ny)
        theta = self._remap_theta
        scale = self._remap_scale if self._remap_scale != 0.0 else 1.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        tx_exact = ox - scale * (cos_t * nx - sin_t * ny)
        ty_exact = oy - scale * (sin_t * nx + cos_t * ny)
        self._remap_tx = ((1.0 - self._drift_pos_alpha) * self._remap_tx
                          + self._drift_pos_alpha * tx_exact)
        self._remap_ty = ((1.0 - self._drift_pos_alpha) * self._remap_ty
                          + self._drift_pos_alpha * ty_exact)
        self._publish_remap()
        rospy.loginfo(
            "[orchestrator] drift pos correction: node=%d err=%.3fm"
            "  tx=%.4f ty=%.4f", node_id, pos_error,
            self._remap_tx, self._remap_ty)

    def _apply_theta_correction(self, heading_nxt, odom_yaw):
        """EMA-correct remap_theta using a completed junction alignment as a heading fix."""
        # After JUNCTION spin, robot faces heading_nxt in map frame.
        # Exact remap_theta: odom_yaw - heading_nxt = remap_theta  =>  map_yaw = heading_nxt
        theta_exact = angle_diff(odom_yaw, heading_nxt)
        delta = angle_diff(theta_exact, self._remap_theta)
        self._remap_theta = self._remap_theta + self._drift_theta_alpha * delta
        self._publish_remap()
        rospy.loginfo(
            "[orchestrator] drift theta correction: delta=%.2fdeg  theta=%.4f",
            math.degrees(delta), self._remap_theta)

    # -------------------------------------------------------- pure-pursuit

    def _carrot(self, rx, ry, path_ids):
        """Find lookahead point on path (MAP frame)."""
        if not path_ids or self._map is None:
            return None

        # Closest node on path
        best_idx, best_dist = 0, float("inf")
        for i, nid in enumerate(path_ids):
            nx, ny = self._map.node_xy(nid)
            d = math.hypot(rx - nx, ry - ny)
            if d < best_dist:
                best_dist, best_idx = d, i

        # If robot has passed best_idx toward best_idx+1, advance
        if best_idx < len(path_ids) - 1:
            ax, ay = self._map.node_xy(path_ids[best_idx])
            bx, by = self._map.node_xy(path_ids[best_idx + 1])
            if (rx - ax) * (bx - ax) + (ry - ay) * (by - ay) > 0:
                best_idx += 1

        # Walk forward until lookahead distance is accumulated
        accum  = 0.0
        px, py = self._map.node_xy(path_ids[best_idx])
        for i in range(best_idx + 1, len(path_ids)):
            nx, ny = self._map.node_xy(path_ids[i])
            seg    = math.hypot(nx - px, ny - py)
            if accum + seg >= self._lookahead:
                frac = (self._lookahead - accum) / max(seg, 1e-9)
                return px + frac * (nx - px), py + frac * (ny - py)
            accum += seg
            px, py = nx, ny
        return self._map.node_xy(path_ids[-1])

    def _pursuit_twist(self, pose_map, path_ids):
        """Compute pure-pursuit Twist (all in MAP frame)."""
        rx, ry, ryaw = pose_map
        carrot = self._carrot(rx, ry, path_ids)
        if carrot is None:
            return Twist()
        heading = math.atan2(carrot[1] - ry, carrot[0] - rx)
        err     = angle_diff(heading, ryaw)
        t = Twist()
        t.linear.x  = self._map_speed
        t.angular.z = clamp(self._map_kp * err, -self._map_max_w, self._map_max_w)
        return t

    # The control-loop FSM (_step + per-state handlers) is provided by the
    # NewOrchestrator subclass — the only class instantiated. The base class
    # contributes the shared machinery above (remap, callbacks, drift fix,
    # pure-pursuit, _map_angular) plus run() below.

    def run(self):
        rate = rospy.Rate(self._rate_hz)
        while not rospy.is_shutdown():
            try:
                self._step()
            except Exception as e:
                rospy.logerr_throttle(2.0, "[orchestrator] step err: %s", e)
            rate.sleep()
        # Clean shutdown
        self._publish_stop()
        self._set_lane_enabled(False)


class NewOrchestrator(Orchestrator):

    # Extra states beyond the parent's IDLE/NAVIGATING/JUNCTION/FALLBACK/DONE.
    ROUNDABOUT     = "ROUNDABOUT"
    EMERGENCY_STOP = "EMERGENCY_STOP"   # terminal failsafe halt (ends navigation)
    # Object-detection overrides (transient; published but never stored in
    # self._state, so navigation resumes once the override clears). The strings
    # match the object_detection.traffic_sign_handler constants of the same name.
    TRAFFIC_STOP   = "TRAFFIC_STOP"     # held at a red light
    STOP_SIGN      = "STOP_SIGN"        # held at a stop sign

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
        # Only the FALLBACK when the real lane heading below is unavailable.
        self._proxy_max    = math.radians(float(rp("proxy_max_deg", 80.0)))
        # Use the real lane heading (/lane_controller/info[1]) as theta_l instead
        # of the angular.z proxy. The proxy conflates curvature, centering and EMA
        # lag, which is why c needed so many gates; heading_rad measures the lane
        # direction itself. Switch off to A/B against the old proxy on the robot.
        self._use_lane_heading = bool(rp("use_lane_heading", True))
        # Yaw drift correction on straights (the missing half of the lateral
        # re-centering): under the same "confidently centred on a straight" gates,
        # the robot's true map yaw is seg_yaw + heading_rad, so remap_theta gets
        # EMA-corrected there. Without it the translation-only fix forever chases
        # the lateral error that the yaw error keeps regenerating.
        self._lat_theta_alpha   = float(rp("lateral_theta_alpha",   0.18))
        # Per-fire STEP cap on the yaw fix (was a hard reject — see _apply_straight_corrections).
        self._lat_theta_max     = math.radians(float(rp("lateral_theta_max_deg", 10.0)))
        # Sanity bound: deltas above this are dropped as a bad lane/segment association;
        # within it the camera fix is allowed to correct (rate-limited), so a large
        # post-rotation drift can actually be undone instead of rejected at 10 deg.
        self._lat_theta_reject  = math.radians(float(rp("lateral_theta_reject_deg", 40.0)))
        # BEV/camera yaw bias: heading_rad (info[1]) carries a constant offset (a slightly
        # rotated BEV warp / single-line read), so on a CENTRED STRAIGHT it reads non-zero.
        # The yaw fix trusts seg_yaw + heading_rad as the robot's true map heading, so that
        # bias is baked into remap_theta every fire — one-signed, and the lever arm turns it
        # into a steady map-frame drift (observed: ~+4-5 deg th_l on a centred straight on the
        # north road dragged the mapped pose south/east). Subtract the measured constant first.
        # Measure: log /lane_controller/info[1] (heading_rad) mean on a known-straight, centred
        # segment; set this (deg, SAME sign as info[1]). 0 = off.
        self._lat_heading_bias  = math.radians(float(rp("lane_heading_bias_deg", 0.0)))
        # Max cross-track at which the YAW re-pin is still trusted. The position branch is
        # already bounded (lateral_max_correct_m); the yaw branch used to fire "regardless of
        # cross-track size", so once localization had drifted (cross=0.46m observed) it pinned
        # yaw to a segment the robot was nowhere near and amplified the error into a conflict
        # EMERGENCY_STOP. This is the missing UPPER bound; it still fires before the position
        # drifts past the noise floor (the "fire early" reasoning bounds the MINIMUM, not this).
        self._lat_theta_max_cross = float(rp("lateral_theta_max_cross_m", 0.35))
        # Post-rotation relock: an in-place spin (junction / roundabout exit) integrates
        # the COMMANDED rotation open-loop and injects tens of degrees of yaw drift that
        # map-frame alignment cannot self-correct. Arm a one-shot full camera re-pin so
        # the first confident straight lane after the spin snaps remap_theta to truth.
        self._yaw_relock_alpha  = float(rp("yaw_relock_alpha", 0.7))
        # Yaw-glitch guard: /odom integrates WHATEVER lands on /jetauto_controller/cmd_vel,
        # so a FOREIGN publisher (a stray joystick/teleop node) injecting a big angular.z
        # spins the robot and dumps yaw into odom that we never commanded — corrupting the
        # map. We can't stop the physical spin (shared topic), but we refuse to bake it into
        # remap: if the MEASURED odom yaw-rate exceeds the rate WE commanded by more than
        # this, suppress the camera yaw/lateral re-pin for a short cooldown.
        self._yaw_glitch_thresh  = float(rp("yaw_glitch_rate_thresh", 1.0))   # rad/s
        self._yaw_glitch_ticks   = int  (rp("yaw_glitch_cooldown_ticks", 15)) # ~0.5 s @30Hz
        # FALLBACK failsafe: pure-pursuit on broken localization could previously
        # run away forever (the only state with no excursion check). Sustained
        # cross-track beyond fallback_offref_m, or simply staying in FALLBACK past
        # fallback_timeout_s, trips the shared terminal EMERGENCY_STOP.
        self._fb_offref_m     = float(rp("fallback_offref_m",     0.60))
        self._fb_offref_ticks = int  (rp("fallback_offref_ticks", 25))
        self._fb_timeout_s    = float(rp("fallback_timeout_s",    60.0))
        # FALLBACK pure-pursuit has no terminal node: at the final waypoint the carrot
        # stays pinned on the last node and forward speed keeps the robot ORBITING it at
        # a radius set by odom drift. If that radius exceeds the waypoint manager's
        # arrival tolerance, arrival never fires and it circles forever (observed). So
        # count consecutive ticks spent targeting the final node within this radius; once
        # sustained, end the run terminally (a healthy run arrives in NAVIGATING, so a
        # FALLBACK arrival is always degraded / unverified localization).
        self._fb_goal_radius  = float(rp("fallback_goal_radius",  0.50))
        self._fb_goal_ticks   = int  (rp("fallback_goal_ticks",   50))
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
        # Dashed-separator lane change (e.g. node 10->11): the inside line of the
        # turn is a DASHED separator that is legal to cross. The anti-cut guardrail
        # exempts it (see _junction_anti_cut_scale); this hold additionally keeps
        # MAP authority (a_lane=0) through the crossing so the lane controller
        # cannot re-center into the original lane after the junction turn, until
        # the dashed line registers on the OPPOSITE side (= crossed) or the tick
        # budget expires. Armed on the junction approach, frozen during the spin.
        self._dash_cross_enable = bool(rp("dash_cross_enable", True))
        self._dash_cross_ticks  = int (rp("dash_cross_hold_ticks", 75))
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
        # Steadiness gate: the yaw/lateral re-pin only fires after the car has tracked a
        # confident, centred, straight lane for this many CONSECUTIVE ticks. Mid-maneuver
        # (catching a line, fresh out of FALLBACK) the lane is transiently at an angle, and
        # re-pinning then bakes that transient into remap as a false yaw error.
        self._settle_ticks       = int  (rp("lateral_settle_ticks",  15))
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
        self._estop_zero_left  = 0      # remaining active-braking ticks after an e-stop
        self._offroute_count   = 0      # consecutive off-route ticks
        self._fb_offref_count  = 0      # consecutive FALLBACK off-reference ticks
        self._fb_goal_count    = 0      # consecutive ticks orbiting the goal in FALLBACK
        self._fb_entered       = None   # rospy.Time of the last FALLBACK entry
        self._yaw_relock       = False  # one-shot: snap yaw from the camera after a spin
        self._prev_odom_yaw    = None   # last odom yaw (yaw-glitch rate estimate)
        self._prev_odom_t      = None   # rospy.Time of that sample
        self._odom_t0          = None   # rospy.Time of first odom msg (gyro-bias feed-forward origin)
        self._glitch_cooldown  = 0      # ticks left suppressing the yaw re-pin after a glitch
        self._steady_count     = 0      # consecutive steady-tracking ticks (yaw re-pin gate)
        self._last_a_lane      = 1.0    # last blend weight (diagnostics)
        self._last_scale       = 1.0    # last anti-cut scale (diagnostics)
        self._dash_hold        = 0      # remaining dashed-crossing hold ticks (0 = off)
        self._dash_left        = True   # crossing direction the hold was armed for
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

        # Object-detection override (traffic lights / STOP signs). Consumes the
        # perception node's JSON detections and, when a red light is latched or a
        # stop sign is active, seizes cmd_vel with a full stop (see _step). Set
        # traffic_light_enable=false to develop/test the driving stack on its own.
        self._obj_det = TrafficSignHandler(
            enable      = bool(rp("traffic_light_enable", True)),
            topic       = rp("traffic_light_topic", "/object_detection/drive"),
            red_label   = rp("traffic_light_red_label", "red_TL"),
            green_label = rp("traffic_light_green_label", "green_TL"),
            stop_label  = rp("stop_sign", "stop_s"),
            stop_hold_s = float(rp("stop_sign_hold", 3.0)))

        # Per-tick diagnostics (Float64MultiArray) — the camera-vs-drift separator.
        # Layout (fixed):
        #   [0] state code (0 IDLE, 1 NAVIGATING, 2 JUNCTION, 3 ROUNDABOUT,
        #                   4 FALLBACK, 5 EMERGENCY_STOP, 6 DONE)
        #   [1] cross-track to path (m)   [2] theta_m (rad)   [3] theta_l (rad)
        #   [4] c                         [5] a_lane          [6] anti-cut scale
        #   [7] lane center_offset (norm) [8] remap_theta     [9] remap_tx
        #   [10] remap_ty                 [11] dist to node   [12] round_offset (m)
        # If [7]~0 (lane says centred) while [1] stays large, the residual is
        # calibration/remap error, not driving error — log it and compare runs.
        self._DIAG_STATE = {self.IDLE: 0.0, self.NAVIGATING: 1.0,
                            self.JUNCTION: 2.0, self.ROUNDABOUT: 3.0,
                            self.FALLBACK: 4.0, self.EMERGENCY_STOP: 5.0,
                            self.DONE: 6.0}
        self._diag_pub = rospy.Publisher(
            "/orchestrator/diag", Float64MultiArray, queue_size=1)

        rospy.loginfo(
            "[nn_orch] ready: lane_heading=%s conflict_ticks=%d junc_infl=%.2fm "
            "lat_theta_alpha=%.2f fb_offref=%.2fm/%dticks/%.0fs",
            self._use_lane_heading, self._conflict_ticks,
            self._junction_influence_radius, self._lat_theta_alpha,
            self._fb_offref_m, self._fb_offref_ticks, self._fb_timeout_s)

    # ------------------------------------------------------------ small helpers

    def _lane_info_cb(self, msg):
        if len(msg.data) < 8:
            return
        with self._lock:
            self._lane_info       = list(msg.data)
            self._lane_info_stamp = rospy.Time.now()

    def _lane_theta(self, lane_cmd, lane_info):
        """Lane direction in the robot frame, standard CCW yaw (rad).

        Primary source: the REAL lane heading from /lane_controller/info[1]
        (`heading_rad`, image convention: +ve = lane bends RIGHT of robot
        forward -> NEGATED here for standard CCW yaw). Used only when the
        geometry is trustworthy: a tracking state (TRACKING_CC/SINGLE_*) with
        at least one valid line. Otherwise — or with use_lane_heading=false —
        fall back to the old angular.z steering-rate proxy. The heading
        measures the lane itself, while the proxy folds in centering effort
        and EMA lag; this is what made c need so many compensating gates.
        """
        if self._use_lane_heading and lane_info is not None:
            code = lane_info[0]
            if code in (2.0, 3.0, 4.0) and (lane_info[2] > 0.5 or lane_info[3] > 0.5):
                return -lane_info[1]
        mw = self._map_max_w if self._map_max_w > 1e-6 else 1.0
        return clamp(lane_cmd.angular.z / mw, -1.0, 1.0) * self._proxy_max

    def _publish_diag(self, C):
        """Per-tick /orchestrator/diag (layout documented at the publisher)."""
        li  = C.lane_info
        msg = Float64MultiArray()
        msg.data = [
            self._DIAG_STATE.get(self._state, -1.0),
            C.offpath, C.theta_m, C.theta_l, C.c,
            self._last_a_lane, self._last_scale,
            li[6] if li is not None else 0.0,
            self._remap_theta, self._remap_tx, self._remap_ty,
            C.dist, self._round_offset,
        ]
        self._diag_pub.publish(msg)

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

    def _apply_straight_corrections(self, C):
        """GPS-style re-centering of the map frame on straights: LATERAL + YAW.

        When the camera is confidently centred-tracking a two-line lane on a STRAIGHT
        segment, two things are simultaneously true and both pin the remap:
          - POSITION: the robot is on that lane's centreline = the map edge, so the
            remap translation is nudged perpendicular onto it (EMA). This keeps
            dist/heading to the next node truthful so the junction logic fires on
            time (lateral drift is what skipped node 6 and inflated dist).
          - YAW: the robot's true map heading is seg_yaw + heading_rad (heading_rad
            from /lane_controller/info[1] is +ve-right in image coords), so
            remap_theta is EMA-corrected toward
                theta_exact = odom_yaw - (seg_yaw + heading_rad).
            This is the half the translation-only fix was missing: a yaw error
            REGENERATES lateral error continuously, so correcting only tx/ty chases
            a moving target. The delta is correcting-toward, RATE-LIMITED to
            lateral_theta_max_deg per fire (not rejected at it — that left a >10 deg
            post-spin drift permanently uncorrectable); only an insane delta above
            lateral_theta_reject_deg is dropped as a bad association. After an in-place
            spin a one-shot relock takes the first reading as a near-full re-pin
            (yaw_relock_alpha). lateral_theta_alpha=0 disables the yaw fix.
        The translation is solved AFTER the theta update, so the frame pivots about
        the ROBOT (its mapped position lands exactly on the foot) instead of
        rotating about the remap origin — a theta-only tweak would otherwise
        translate the mapped position by delta x lever-arm.

        Heavily gated: NAVIGATING only, never in a junction/roundabout or within the
        turn-in zone, on a confident centred well-aligned track, once every
        `lateral_correct_period` NAVIGATING ticks. Both YAW and the LATERAL nudge fire
        on a single confident line (TRACKING_CC / SINGLE_L / SINGLE_R) — the narrow
        FOV rarely shows both, and on a long junction-free straight the lateral pin is
        the only thing that undoes the lever-arm map drift (it shows up as cross-track
        there). Lateral is perpendicular-foot ONLY, so along-track node advancement is
        untouched.
        """
        if not self._lat_correct:
            return
        # An uncommanded rotation just spun the robot (foreign cmd_vel / slip): the lane
        # reading and the odom yaw are momentarily inconsistent, so re-pinning now would
        # bake the glitch into remap. Skip yaw + lateral + relock until the cooldown ends
        # (the relock stays armed, so it still fires on the first CLEAN straight after).
        if self._glitch_cooldown > 0:
            return
        self._lat_tick = (self._lat_tick + 1) % max(1, self._lat_period)
        if self._lat_tick != 0:
            return
        if (self._map is None or C.lane_info is None
                or self._state != self.NAVIGATING or C.in_roundabout or C.is_junction):
            return
        # Only re-pin once the car has been steadily tracking for a while (not mid-maneuver):
        # the streak is reset by any unsteady/transient/glitch tick in _step.
        if self._steady_count < self._settle_ticks:
            return
        li = C.lane_info
        # Relaxed gate: a single confidently-tracked line (SINGLE_L/SINGLE_R) drives
        # BOTH the yaw and the lateral-position fix. The narrow FOV rarely shows both
        # lines, so requiring TRACKING_CC starved the correction — and on a long
        # junction-free straight (e.g. the east road) the lateral fix is the ONLY
        # thing that can undo the lever-arm map drift, which shows up as cross-track
        # there. With one line the centreline is inferred from the dynamic lane-width
        # estimate; the centred gate (|center_offset| <= lateral_centered_clear) plus
        # the EMA and [lateral_min/max_correct_m] bounds keep a single-line pin safe.
        # Junction/roundabout nodes are already excluded above.
        have_line = (li[2] > 0.5 or li[3] > 0.5)
        if (C.lane_state not in ("TRACKING_CC", "SINGLE_L", "SINGLE_R")
                or not have_line or abs(li[6]) > self._lat_centered_clear):
            return                                       # need a confident, centred track
        # Turn-in-zone gate: only block near the current node when the path actually
        # BENDS there. The old unconditional `dist <= junction_influence_radius`
        # starved the correction on dense maps — most segments here are 0.30-0.47 m,
        # so dist almost never exceeded 0.50 m and the fix fired ~never (sim).
        if C.dist <= self._junction_influence_radius:
            ta = self._turn_angle(C.path_ids, C.cur_path_idx, C.cur_node_id, C.next_id)
            if ta is None or ta >= math.radians(self._gentle_turn_deg):
                return                                   # bend (or unknown) ahead — stay out
        seg = self._nearest_segment_cross(C.rx, C.ry, C.path_ids)
        if seg is None:
            return
        cdx, cdy, seg_yaw, cross = seg
        # De-bias the lane heading before it is used as truth (constant BEV/single-line yaw
        # offset, see _lat_heading_bias): both the straightness gate below and yaw_true use it.
        heading = angle_diff(li[1] - self._lat_heading_bias, 0.0)
        # "Driving ALONG a straight" is judged from the LANE (|heading_rad| small):
        # the camera is immune to odom/remap drift. Gating on the MAPPED yaw here
        # (the old check) was a catch-22 — once accumulated theta error exceeded
        # the gate, the very correction that fixes theta could never fire again
        # (observed in sim: 19deg of injected drift, only 1.5deg ever absorbed).
        # The mapped yaw keeps only a LOOSE sanity bound against associating the
        # robot with a perpendicular segment of the path.
        if abs(heading) > math.radians(self._lat_align_deg):
            return                                       # lane says we're not tracking straight
        if abs(angle_diff(seg_yaw, C.ryaw)) > math.radians(45.0):
            return                                       # wrong-segment association reject
        with self._lock:
            odom = self._odom_pos
            odom_yaw = self._odom_yaw
        if odom is None:
            return

        # --- YAW: true map yaw on a centred straight = seg_yaw + heading_rad ---
        # Runs regardless of cross-track size: yaw error regenerates lateral error
        # continuously, so waiting for the position to drift past the noise floor
        # (the old gate order) let theta run away while position kept being pinned.
        dtheta = 0.0
        # Gate the YAW re-pin on cross-track: far off the mapped segment, seg_yaw is the wrong
        # heading to pin to (broken localization), so re-pinning amplifies rather than corrects.
        if self._lat_theta_alpha > 0.0 and cross <= self._lat_theta_max_cross:
            yaw_true    = seg_yaw + heading
            theta_exact = angle_diff(odom_yaw, yaw_true)
            delta       = angle_diff(theta_exact, self._remap_theta)
            # The camera is the ONLY absolute yaw reference, so a large delta is to be
            # ABSORBED, not refused: an in-place spin injects tens of degrees that map-
            # frame alignment can't fix. Reject only an insane delta (bad lane/segment
            # association) above lat_theta_reject; within it, step toward truth. Normally
            # rate-limited to lat_theta_max/fire; on a post-rotation relock, take the
            # first confident reading as a near-full re-pin (yaw_relock_alpha) so a big
            # spin drift snaps back in one straight rather than crawling 10 deg/tick.
            if abs(delta) <= self._lat_theta_reject:
                if self._yaw_relock and self._yaw_relock_alpha > 0.0:
                    self._yaw_relock = False
                    dtheta = self._yaw_relock_alpha * delta
                    rospy.loginfo("[nn_orch] yaw relock (post-spin): "
                                  "delta=%+.1fdeg", math.degrees(delta))
                else:
                    dtheta = clamp(self._lat_theta_alpha * delta,
                                   -self._lat_theta_max, self._lat_theta_max)

        # --- POSITION: perpendicular FOOT on the segment (lateral only, never the
        # node), gated by the noise floor / broken-localization bounds. Solve the
        # exact remap translation that maps the current ODOM point onto the foot
        # UNDER THE UPDATED THETA, then EMA toward it — same convention as the
        # parent's drift fix
        # (map = R(-theta)/scale * (odom - t)  =>  t = odom - scale*R(theta)*map_pt).
        # NOTE: t lives in the ODOM frame; an earlier version added a MAP-frame
        # vector straight onto t (wrong frame AND sign) — the "thinks it's behind /
        # never recovers" bug. When only YAW fires, the translation is re-anchored
        # so the CURRENT mapped pose is invariant (pivot about the robot) — a
        # theta-only change would otherwise rotate the pose about the remap origin.
        # Lateral nudge whenever the cross-track sits in the trusted band: it pins the
        # robot to the lane centreline (= the map edge), the only correction for the
        # lever-arm map drift on a long single-line straight. The centred gate above
        # plus this [min,max] band keep a single-line (width-inferred) pin honest.
        do_xy = self._lat_min_m <= cross <= self._lat_max_m
        if dtheta == 0.0 and not do_xy:
            return
        anchor_x, anchor_y = (C.rx + cdx, C.ry + cdy) if do_xy else (C.rx, C.ry)
        with self._lock:
            theta = self._remap_theta + dtheta
            scale = self._remap_scale if self._remap_scale != 0.0 else 1.0
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            tx_exact = odom[0] - scale * (cos_t * anchor_x - sin_t * anchor_y)
            ty_exact = odom[1] - scale * (sin_t * anchor_x + cos_t * anchor_y)
            a = self._lat_alpha if do_xy else 1.0   # pure re-anchor keeps pose fixed
            self._remap_theta = theta
            self._remap_tx = (1.0 - a) * self._remap_tx + a * tx_exact
            self._remap_ty = (1.0 - a) * self._remap_ty + a * ty_exact
        self._publish_remap()
        rospy.loginfo_throttle(
            1.0, "[nn_orch] straight correction: cross=%.3fm xy=%d dtheta=%+.2fdeg "
            "-> theta=%.4f tx=%.4f ty=%.4f", cross, int(do_xy), math.degrees(dtheta),
            self._remap_theta, self._remap_tx, self._remap_ty)

    def _apply_theta_correction(self, heading_nxt, odom_yaw):
        """Pivot-invariant override of the parent's post-junction theta fix.

        The parent EMA-corrects remap_theta and republishes, but rotating the
        odom->map transform about the remap ORIGIN translates every mapped point
        by (delta x lever-arm from the origin): metres of position jump for a few
        degrees of theta at this map's scale, immediately after a junction. That
        side effect is the likely reason the theta fix was tuned to 0 and yaw
        drift went uncorrected. Here the translation is re-solved so the robot's
        CURRENT mapped position is invariant: the frame pivots about the robot.
        """
        with self._lock:
            odom = self._odom_pos
            pose = self._pose
        if odom is None or pose is None:
            super(NewOrchestrator, self)._apply_theta_correction(heading_nxt, odom_yaw)
            return
        theta_exact = angle_diff(odom_yaw, heading_nxt)
        delta = angle_diff(theta_exact, self._remap_theta)
        with self._lock:
            self._remap_theta = self._remap_theta + self._drift_theta_alpha * delta
            # Re-anchor: keep odom -> (mx, my) fixed under the new theta.
            mx, my = pose[0], pose[1]
            scale = self._remap_scale if self._remap_scale != 0.0 else 1.0
            cos_t, sin_t = math.cos(self._remap_theta), math.sin(self._remap_theta)
            self._remap_tx = odom[0] - scale * (cos_t * mx - sin_t * my)
            self._remap_ty = odom[1] - scale * (sin_t * mx + cos_t * my)
        self._publish_remap()
        rospy.loginfo(
            "[nn_orch] theta correction (pivot@robot): delta=%.2fdeg theta=%.4f",
            math.degrees(delta), self._remap_theta)

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
        rospy.logwarn("[nn_orch] REPLAN: %d -> %d (off-route)", int(start_id), end_id)
        return True

    # state-entry shortcuts -------------------------------------------------

    def _enter_fallback(self, why):
        if self._state != self.FALLBACK:
            self._state          = self.FALLBACK
            self._hold_count     = 0
            self._recv_count     = 0
            self._conflict_count = 0   # off-path excursion is a recovery, not a conflict
            self._fb_offref_count = 0
            self._fb_goal_count   = 0
            self._fb_entered      = rospy.Time.now()   # for the timeout failsafe
            self._publish_orc_state(self.FALLBACK)
            rospy.loginfo("[nn_orch] -> FALLBACK (%s)", why)

    def _enter_emergency_stop(self, why="lane vs map conflict"):
        """Terminal failsafe: halt the robot and END navigation (like arrival).

        The debounce upstream already absorbs transient camera/segmentation noise, so a
        trip here is a *sustained* fault that, in practice, never self-recovers. Instead
        of a recoverable hold, we stop the wheels, disable the lane controller, and
        CANCEL the active goal (empty goal -> waypoint_manager IDLE). The robot stays
        stopped and the dashboard shows EMERGENCY_STOP until a NEW goal is issued
        (cleared in _path_cb). The latch is honored at the top of _step.
        """
        with self._lock:
            self._emergency = True
        # Brake actively for ~0.5 s, then go SILENT on cmd_vel (see _step latch).
        self._estop_zero_left = max(1, int(0.5 * self._rate_hz))
        self._state     = self.EMERGENCY_STOP
        self._set_lane_enabled(False)              # stop the lane controller driving
        self._publish_stop()                       # halt the wheels
        self._goal_pub.publish(Int32MultiArray())  # cancel navigation (empty goal)
        self._publish_orc_state(self.EMERGENCY_STOP)
        rospy.logwarn("[nn_orch] EMERGENCY STOP: %s -- navigation halted; "
                      "issue a new goal to resume", why)

    def _path_cb(self, msg):
        # A genuinely new goal (non-empty path) clears a latched emergency and resumes.
        # Latch handled under the lock: this runs on a subscriber thread while _step
        # reads the latch on the control thread (cancel vs new-goal ordering race).
        super(NewOrchestrator, self)._path_cb(msg)
        cleared = False
        with self._lock:
            if self._emergency and len(msg.data) > 0:
                self._emergency = False
                cleared = True
        if cleared:
            self._state = self.NAVIGATING
            rospy.loginfo("[nn_orch] EMERGENCY cleared by new goal -> resuming")

    # --------------------------------------------------------------- main FSM

    def _step(self):
        # Terminal emergency stop: hold the robot halted (lane off + zero cmd) and keep
        # the dashboard informed until a NEW goal is issued (cleared in _path_cb). Sits
        # ABOVE everything — including the inactive/DONE handler — so the nav cancel we
        # publish on entry can't bounce us into IDLE and mask the EMERGENCY_STOP state.
        with self._lock:
            emergency = self._emergency
        if emergency:
            # Brake actively only for a short grace window after entry, then stay
            # SILENT on cmd_vel: a permanent 25 Hz zero stream interleaves with the
            # dashboard's manual/remap commands on the same topic and the robot
            # "struggles to move" until a new goal clears the latch (observed).
            # The latch itself stays terminal — we just stop spamming zeros.
            if self._estop_zero_left > 0:
                self._estop_zero_left -= 1
                self._set_lane_enabled(False)
                self._publish_stop()
            self._publish_orc_state(self.EMERGENCY_STOP)
            return

        # --- object-detection override (traffic light / STOP sign) ---
        # Highest priority after the terminal emergency latch: while a red light
        # is latched or a stop-sign hold is active, seize the cmd_vel bus with a
        # full stop. We return BEFORE touching self._state, so navigation resumes
        # exactly where it left off once the light turns green / the hold expires.
        decision = self._obj_det.evaluate()
        if decision.stop:
            self._publish_orc_state(decision.state)   # TRAFFIC_STOP / STOP_SIGN
            self._publish_stop()                      # full stop
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

        # Flag uncommanded rotation (foreign cmd_vel publisher / gross slip) so the camera
        # re-pin won't bake a glitch-corrupted reading into remap. Updated every tick.
        self._update_yaw_glitch(odom_yaw)

        # Tracking steadiness: count consecutive ticks of confident, centred, straight,
        # glitch-free NAVIGATING tracking. The yaw/lateral re-pin requires a streak so it
        # fires only when the car is STEADILY following the road — not mid-maneuver
        # (catching a line, fresh out of FALLBACK), where the transient lane angle would
        # be baked into remap as a false yaw error. (self._state is the start-of-tick
        # state, so the first ticks after a FALLBACK/JUNCTION exit still read as unsteady.)
        steady = (self._state == self.NAVIGATING and self._glitch_cooldown == 0
                  and lane_state in ("TRACKING_CC", "SINGLE_L", "SINGLE_R")
                  and lane_info is not None and len(lane_info) >= 8
                  and (lane_info[2] > 0.5 or lane_info[3] > 0.5)
                  and abs(lane_info[6]) <= self._lat_centered_clear
                  and abs(angle_diff(lane_info[1] - self._lat_heading_bias, 0.0))
                      <= math.radians(self._lat_align_deg))
        self._steady_count = (self._steady_count + 1) if steady else 0

        if pose is None:
            # No odom yet (bring-up). Halt only if we were actually driving;
            # spamming zeros here fights manual/remap driving on the same topic.
            if self._state not in (self.IDLE, self.DONE):
                self._publish_stop()
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
            if in_roundabout:
                # Fresh traversal: a new goal over the SAME ring keeps the cached
                # spline (key = ring ids, unchanged), but the forward-only progress
                # index is still parked at the previous traversal's tail — the
                # nearest-point search then reads the distance to the FAR side of
                # the ring (~0.7 m here) and the off-reference failsafe trips as
                # soon as the spline branch engages (observed on lap 2 and on every
                # retry). Restart progress + debounce at the window edge.
                self._round_i = 0
                self._round_offref_count = 0
            rospy.loginfo("[nn_orch] roundabout: %s (node=%d)",
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

        # Lane/map agreement.  theta_l is the real lane heading when trustworthy
        # (angular.z proxy fallback — see _lane_theta).  c is only meaningful with
        # a usable lane reading; otherwise treat as "agreeing" so lane-loss is
        # handled by FALLBACK, not mistaken for a conflict.
        lane_usable = lane_fresh and lane_state not in self._LANE_BAD
        theta_m = angle_diff(heading_nxt, ryaw) if next_id >= 0 else 0.0
        theta_l = self._lane_theta(lane_cmd, lane_info) if lane_usable else 0.0
        c = math.cos(theta_l - theta_m) if (lane_usable and next_id >= 0) else 1.0

        # Cross-track to the planned path — one compute per tick, shared by the
        # conflict gate, the FALLBACK failsafe and the diagnostics.
        offpath = self._offpath_dist(rx, ry, path_ids)

        C = _Ctx(rx, ry, ryaw, dist, is_junction, heading_nxt, next_id,
                 cur_node_id, cur_path_idx, path_ids, lane_cmd, lane_state,
                 lane_fresh, lane_usable, in_roundabout, theta_m, theta_l, c,
                 offpath, odom_yaw, lane_info)

        # Heartbeat: one line/sec so a misbehavior is traceable to its inputs.
        rospy.loginfo_throttle(
            1.0, "[nn_orch] st=%s node=%d->%d lane=%s junc=%d round=%d dist=%.2f "
            "off=%.2f hdg_err=%+.0fdeg th_l=%+.0fdeg c=%+.2f", self._state,
            cur_node_id, next_id, lane_state, int(is_junction),
            int(in_roundabout), dist, offpath, math.degrees(theta_m),
            math.degrees(theta_l), c)

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

        self._publish_diag(C)

    # ----------------------------------------------------------- inactive / idle

    def _handle_inactive(self, lane_fresh, lane_state, lane_cmd):
        if self._state not in (self.IDLE, self.DONE):
            rospy.loginfo("[nn_orch] nav inactive -> DONE")
            self._set_lane_enabled(False)
            self._publish_stop()
            self._state          = self.DONE
            self._hold_count     = 0
            self._recv_count     = 0
            self._conflict_count = 0
            self._dash_hold      = 0
            self._freerun        = False
        self._publish_orc_state(self.IDLE)
        # Freerun: pass lane commands through when the dashboard enables lane directly.
        if lane_fresh and lane_state not in self._LANE_BAD:
            self._freerun = True
            self._publish_cmd(lane_cmd)
        elif self._freerun:
            self._freerun = False
            self._publish_stop()

    # ------------------------------------------------------------- NAVIGATING

    def _h_navigating(self, C):
        if self._state != self.NAVIGATING:
            self._state = self.NAVIGATING
            self._set_lane_enabled(True)
            rospy.loginfo("[nn_orch] -> NAVIGATING")
        self._publish_orc_state(self.NAVIGATING)

        # GPS-style re-centering of the map frame (lateral + yaw). Gated to confident
        # straights. Done here (lane drives on a straight, a_lane=1) so a frame nudge
        # causes no steering jerk while it makes dist/heading truthful.
        self._apply_straight_corrections(C)

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
        on_path = C.offpath <= self._conflict_onpath_m
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
            if C.offpath >= 2.0 * self._offroute_t:
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
            rospy.loginfo("[nn_orch] -> JUNCTION (dist=%.2fm)", C.dist)
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
                self._publish_cmd(self._pursuit_twist((C.rx, C.ry, C.ryaw), C.path_ids))
                return
            a_lane = 1.0 - (self._hold_count / float(self._hold_ramp))
        else:
            self._hold_count = 0
            # Distance-gated map pull while approaching a junction node.
            in_approach = C.is_junction and C.next_id >= 0
            a_lane, _ = decide_blend(
                C.theta_l, C.theta_m, C.dist, in_approach,
                self._junction_influence_radius, self._junc_turn_full)

        # Dashed-separator lane change: keep map authority through the crossing
        # (arms on the approach, persists past the spin until crossed/expired).
        a_lane = self._dash_cross_hold(C, a_lane)

        lane_ang = C.lane_cmd.angular.z if C.lane_fresh else 0.0
        lane_lin = C.lane_cmd.linear.x  if C.lane_fresh else 0.0
        map_ang  = self._map_angular(C.heading_nxt, C.ryaw) if C.next_id >= 0 else 0.0

        # Anti-cut guardrail: hold the MAP's turn-in until the inside line clears.
        # It scales ONLY the map term — scaling the whole blend also erased the
        # lane's centering correction, leaving near-zero authority exactly where
        # a command was needed (low a_lane + floor 0 -> w ~ 0 at the turn).
        scale   = self._junction_anti_cut_scale(C)
        blended = a_lane * lane_ang + (1.0 - a_lane) * map_ang * scale

        self._last_a_lane = a_lane     # diagnostics
        self._last_scale  = scale

        twist = Twist()
        twist.linear.x  = lane_lin
        twist.angular.z = blended
        # Shows when the map is overriding the lane (a_lane low), when the guardrail is
        # holding the turn (scale < 1), and the resulting drive/turn.
        rospy.loginfo_throttle(
            1.0, "[nn_orch] NAV a_lane=%.2f scale=%.2f lane(v=%.2f w=%+.2f) "
            "map_w=%+.2f -> v=%.2f w=%+.2f", a_lane, scale, lane_lin, lane_ang,
            map_ang, twist.linear.x, twist.angular.z)
        self._publish_cmd(twist)

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
        line is still validly seen ahead — i.e. the intersection has NOT opened yet —
        scale the turn toward straight (down to `_junc_lane_floor`) so the robot
        drives up to the node before committing the turn. Once the inside line goes
        invalid (the intersection mouth opens) the scale returns to 1.0 and the
        blend's full turn-in (and/or the in-place spin) takes over.

        Returns 1.0 (no damping) when: disabled, no fresh lane info, not at a junction,
        no real turn intended (map heading still ~forward), the inside line is no
        longer seen (intersection open), or it is a DASHED legal-crossing separator.
        Camera-frame, NAVIGATING approach only.
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
        # The intersection "opens" when the inside line stops being painted ahead.
        # THAT — not the offset shrinking below a small threshold — is the cue to
        # commit the turn: a correctly-centred inside line sits at ~|0.79| (a full
        # lane half-width), so the old `|inside_off| >= junc_inside_clear (0.30) ->
        # commit` test fired on EVERY normal approach and the guardrail only woke up
        # after the robot had already cut a half-lane inward (the wide node-6 cut).
        # While the inside line is still validly seen we are still in the approach
        # lane, so hold the map turn toward straight and drive up to the node; the
        # line vanishing (or the in-place spin arming) releases it.
        if not inside_valid:
            return 1.0                   # intersection opened -> commit turn
        # A DASHED inside line (info[8]/[9]) is a legally crossable separator
        # (lane change, e.g. 10->11) — never hold the turn against it.
        if len(li) >= 10 and ((li[8] > 0.5) if turn_left else (li[9] > 0.5)):
            return 1.0
        # Full hold while the inside line sits at/inside its nominal lane position;
        # ease as it recedes outward past nominal by up to junction_inside_clear (as
        # the intersection opens the line drifts toward the image edge before it goes
        # invalid), so the turn eases in smoothly rather than snapping on the drop-out.
        half_w = (li[7] / 2.0) if (len(li) >= 8 and li[7] > 0.1) else 0.80
        over   = abs(inside_off) - half_w
        sev    = clamp(1.0 - over / max(self._junc_inside_clear, 1e-3), 0.0, 1.0)
        return clamp(1.0 - self._junc_lane_gain * sev, self._junc_lane_floor, 1.0)

    def _dash_cross_hold(self, C, a_lane):
        """Map-authority hold while crossing a dashed separator (lane change).

        Armed on the junction approach when the turn points INTO a valid DASHED
        inside line (e.g. 10->11): after the junction turn the lane controller
        still sees the original lane and would re-center into it, so while the
        hold runs a_lane is forced to 0 and the map alone drives the crossing.
        The budget is topped up while the arming conditions hold and is only
        consumed by NAVIGATING ticks (the JUNCTION spin freezes it). Released
        early once the dashed line registers on the OPPOSITE side (= crossed),
        or when `dash_cross_hold_ticks` expires. Dashedness only ever PERMITS
        the crossing — the map decides whether one happens.
        """
        if not self._dash_cross_enable:
            return a_lane
        li = C.lane_info
        has_dash = li is not None and len(li) >= 10
        # (Re-)arm: junction approach with a real turn into a valid dashed line.
        if (has_dash and C.is_junction and C.next_id >= 0
                and abs(C.theta_m) >= math.radians(20.0)):
            turn_left = C.theta_m > 0.0
            inside_valid  = (li[2] > 0.5) if turn_left else (li[3] > 0.5)
            inside_dashed = (li[8] > 0.5) if turn_left else (li[9] > 0.5)
            if inside_valid and inside_dashed:
                if self._dash_hold == 0:
                    rospy.loginfo("[nn_orch] dash-cross hold armed (%s, %d ticks)",
                                  "left" if turn_left else "right",
                                  self._dash_cross_ticks)
                self._dash_hold = self._dash_cross_ticks
                self._dash_left = turn_left
        if self._dash_hold <= 0:
            return a_lane
        # Crossed: the dashed line now shows on the side we came FROM.
        if has_dash:
            crossed = ((li[3] > 0.5 and li[9] > 0.5) if self._dash_left
                       else (li[2] > 0.5 and li[8] > 0.5))
            if crossed:
                rospy.loginfo("[nn_orch] dash-cross hold released: separator crossed")
                self._dash_hold = 0
                return a_lane
        self._dash_hold -= 1
        if self._dash_hold == 0:
            rospy.logwarn("[nn_orch] dash-cross hold expired without seeing the "
                          "separator on the far side — releasing to the lane")
        return 0.0

    # --------------------------------------------------------------- JUNCTION

    def _arm_yaw_relock(self):
        """Arm a one-shot camera yaw re-pin after an in-place spin, and OPEN the
        straight-correction period gate so it fires on the next eligible NAVIGATING
        tick rather than up to lateral_correct_period (~2 s) later — a spin's open-loop
        yaw drift must be fixed promptly, before it regenerates position error."""
        self._yaw_relock = True
        self._lat_tick   = self._lat_period - 1   # next period-gate check opens

    def _update_yaw_glitch(self, odom_yaw):
        """Detect an UNcommanded rotation by comparing the measured odom yaw-rate to the
        rate we last commanded (`_last_wz`). A big mismatch means something we didn't
        command rotated the robot — a foreign cmd_vel publisher (stray joystick/teleop)
        or gross slip — so the camera re-pin must NOT trust this reading. Arms/refreshes
        `_glitch_cooldown`; decremented on clean ticks. Updated every tick (any state)."""
        now = rospy.Time.now()
        glitch = False
        if (odom_yaw is not None and self._prev_odom_yaw is not None
                and self._prev_odom_t is not None):
            dt = (now - self._prev_odom_t).to_sec()
            if 0.01 < dt < 0.5:
                rate = angle_diff(odom_yaw, self._prev_odom_yaw) / dt
                if abs(rate - self._last_wz) > self._yaw_glitch_thresh:
                    self._glitch_cooldown = self._yaw_glitch_ticks
                    glitch = True
                    rospy.logwarn_throttle(
                        1.0, "[nn_orch] yaw GLITCH: odom %.1f rad/s vs commanded %.1f "
                        "(foreign cmd_vel? slip?) -> suppressing yaw re-pin %d ticks",
                        rate, self._last_wz, self._yaw_glitch_ticks)
        self._prev_odom_yaw = odom_yaw
        self._prev_odom_t   = now
        if not glitch and self._glitch_cooldown > 0:
            self._glitch_cooldown -= 1

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
            # The map-frame alignment above is near-invariant (it pins to the drifting
            # frame); arm a camera re-pin so the first straight after the turn fixes the
            # yaw drift the open-loop spin just injected.
            self._arm_yaw_relock()
            rospy.loginfo("[nn_orch] JUNCTION aligned (err=%.1fdeg) -> NAVIGATING",
                          math.degrees(abs(err)))
            return

        lane_ang = C.lane_cmd.angular.z if C.lane_usable else 0.0
        twist = Twist()
        twist.linear.x  = self._junc_spin_creep   # 0 = pure pivot; >0 arcs (rolls -> less drift)
        twist.angular.z = (1.0 - self._alpha_junc) * lane_ang + self._alpha_junc * spin_ang
        self._publish_cmd(twist, in_place=(self._junc_spin_creep == 0.0))

    # ------------------------------------------------------------- ROUNDABOUT

    def _next_left_ring(self, C):
        """True when the next target node is outside the ring (the exit leg).

        Shared by _h_roundabout (drop the radial spline for plain pure-pursuit at
        the exit) and _round_lane_nudge (disable the guardrail so the robot can
        cross the outer boundary to leave) — the two must stay on the same boundary.
        """
        return (C.next_id < 0 or (self._map is not None
                and not self._map.is_roundabout_node(C.next_id)))

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

    def _ring_span(self, path_ids):
        """(first, last) path indices of the roundabout-tagged span, or None.

        The contiguous span between the first and last roundabout-tagged nodes, so
        callers cover the whole ring portion of the route, tolerant of an untagged
        node slipping in between. Shared by _ring_ids and _build_round_curve.
        """
        if self._map is None or not path_ids:
            return None
        ridx = [i for i, nid in enumerate(path_ids)
                if self._map.is_roundabout_node(int(nid))]
        if not ridx:
            return None
        return ridx[0], ridx[-1]

    def _ring_ids(self, path_ids):
        """Ordered ring node ids on the path (first..last tagged-roundabout span)."""
        span = self._ring_span(path_ids)
        if span is None:
            return []
        first, last = span
        return [int(path_ids[i]) for i in range(first, last + 1)]

    def _build_round_curve(self, path_ids, center):
        """Per-segment radial-arc polyline through the ring nodes (MAP frame).

        Each ring segment is a polar arc about `center` keeping each node's own
        radius, so an outlying node gives a flatter local arc and a tighter node a
        sharper one (the "smooth radial").  Straight lead-in/out segments connect
        the entry/exit neighbors so the curve joins the rest of the path.  Returns
        the dense polyline, or None for < 3 ring nodes (caller falls back to nodes).
        """
        span = self._ring_span(path_ids)
        if span is None:
            return None
        first, last = span
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
                    "[nn_orch] roundabout curve: %d pts / %d nodes %s centre=(%.2f,%.2f)",
                    len(self._round_spline) if self._round_spline else 0,
                    len(key), list(key), cx, cy)
            else:
                rospy.logwarn_throttle(
                    5.0, "[nn_orch] roundabout: %d ring node(s) (<3) — node pursuit",
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
            rospy.loginfo("[nn_orch] -> ROUNDABOUT")
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
        #   - Ring (current target IS a ring node, next still in the ring): follow
        #     the smooth radial curve through the ring nodes (built for >=3 ring
        #     nodes).
        #   - ENTRY approach (window opened because `next` is a ring node, but the
        #     robot is still on the segment INTO the ring, e.g. 30->29 with
        #     next=24): plain path pursuit + path cross-track. The spline only
        #     starts at the ring's entry neighbor (path[first-1], node 29), so on
        #     this segment its nearest point measures the ALONG-TRACK distance to
        #     that node (~0.7 m at node 30) and the off-reference failsafe tripped
        #     right there (observed). The path polyline DOES cover this segment, so
        #     it is the truthful reference; the spline takes over once `cur` is a
        #     ring node (its lead-in covers the handover segment).
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
        exiting = self._next_left_ring(C)
        on_ring = (self._map is not None
                   and self._map.is_roundabout_node(C.cur_node_id))
        poly = (self._ensure_round_spline(C.path_ids)
                if on_ring and not exiting else None)
        if poly:
            carrot = self._spline_carrot(C.rx, C.ry, poly)  # sets self._round_offset
            src    = "curve"
        else:
            carrot = self._carrot(C.rx, C.ry, C.path_ids)
            src    = "exit" if exiting else ("approach" if not on_ring else "nodes")
            # No spline here, so measure deviation against the path the pursuit follows.
            self._round_offset = self._offpath_dist(C.rx, C.ry, C.path_ids)
        if carrot is None:
            self._publish_stop()
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
        # Sharp exit turn -> spin in place first. Driving a ~160deg exit (28->23->2) as a
        # forward pure-pursuit arc sweeps wide and leaves the robot perpendicular to the
        # exit road (observed at node 23). When exiting the ring with a large heading
        # error to the exit carrot, rotate in place (junction-style) and hold forward
        # speed until roughly aligned, then let normal pursuit resume.
        if exiting and abs(err) >= math.radians(self._round_exit_spin_deg):
            out = Twist()
            out.linear.x  = self._junc_spin_creep   # 0 = pure pivot; >0 arcs (less drift)
            out.angular.z = clamp(1.5 * err, -self._junc_spin, self._junc_spin)
            self._publish_cmd(out, in_place=(self._junc_spin_creep == 0.0))
            # Same open-loop yaw drift as a junction spin: arm a camera re-pin for the
            # first straight after the exit (once the lane resumes past the ring).
            self._arm_yaw_relock()
            rospy.loginfo_throttle(
                1.0, "[nn_orch] ROUND exit spin err=%+.0fdeg", math.degrees(err))
            return
        base    = clamp(self._map_kp * err, -self._map_max_w, self._map_max_w)
        # Camera-frame guardrail nudge (additive; 0 unless a road edge is close).
        nudge   = self._round_lane_nudge(C)
        twist = Twist()
        twist.linear.x  = self._roundabout_speed
        twist.angular.z = clamp(base + nudge, -self._map_max_w, self._map_max_w)
        self._publish_cmd(twist)
        rospy.loginfo_throttle(
            1.0, "[nn_orch] ROUND src=%s carrot=(%.2f,%.2f) err=%+.0fdeg "
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
        if self._next_left_ring(C):
            rospy.loginfo_throttle(1.0, "[nn_orch] ROUND guardrail OFF (exit leg)")
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
        # Off-reference failsafe (same idea as the roundabout's): FALLBACK is map
        # pure-pursuit on a localization that already proved doubtful (the lane was
        # lost), and it previously had NO excursion check — the one state that could
        # drive off-road indefinitely. A sustained cross-track excursion from the
        # path it is itself pursuing, or simply staying in FALLBACK too long,
        # means the recovery is not working: halt terminally instead of wandering.
        if C.offpath > self._fb_offref_m:
            self._fb_offref_count += 1
            if self._fb_offref_count >= self._fb_offref_ticks:
                self._fb_offref_count = 0
                self._enter_emergency_stop("off-reference %.2fm (fallback)" % C.offpath)
                return
        else:
            self._fb_offref_count = 0
        if (self._fb_timeout_s > 0.0 and self._fb_entered is not None
                and (rospy.Time.now() - self._fb_entered).to_sec() > self._fb_timeout_s):
            self._enter_emergency_stop("fallback timeout %.0fs" % self._fb_timeout_s)
            return

        # Orbiting the goal in FALLBACK. Pure-pursuit has no terminal node: at the final
        # waypoint (next_id < 0) it keeps driving forward toward a carrot pinned on the
        # goal and CIRCLES it at a radius set by odom drift. If that radius exceeds the
        # waypoint manager's arrival tolerance, arrival never fires and it circles forever
        # (observed). Conversely, when arrival DOES fire mid-FALLBACK the run "completes"
        # while physically off-road (the broken-localization runaway). Both are the same
        # degraded case — a healthy run arrives in NAVIGATING — so count consecutive ticks
        # spent at the final node and, once sustained, end the run on the shared terminal
        # failsafe: stop the circling AND flag it rather than rubber-stamp the arrival.
        if C.next_id < 0 and C.dist <= self._fb_goal_radius:
            self._fb_goal_count += 1
            if self._fb_goal_count >= self._fb_goal_ticks:
                self._fb_goal_count = 0
                self._enter_emergency_stop("goal unreachable in FALLBACK "
                                           "(lane never recovered)")
                return
        else:
            self._fb_goal_count = 0

        # Pure-pursuit on the path; recover only with optical + positional proof.
        # "Near path" accepts EITHER proximity to the current target node OR being back
        # on the path by cross-track: after an off-road excursion the robot can be back
        # on the road yet far from the nearest node on a long segment, where the
        # node-distance test alone never re-arms NAVIGATING (observed: stayed in FALLBACK).
        near_path = (C.dist <= self._recovery_radius
                     or C.offpath <= self._recovery_radius)
        lane_ok   = C.lane_state not in self._LANE_BAD
        if lane_ok and near_path:
            self._recv_count += 1
            if self._recv_count >= self._recovery_ticks:
                self._recv_count = 0
                self._hold_count = 0
                self._state = self.NAVIGATING
                self._publish_orc_state(self.NAVIGATING)
                rospy.loginfo("[nn_orch] FALLBACK -> NAVIGATING (lane OK, dist=%.2fm)",
                              C.dist)
                self._h_navigating(C)
                return
            self._publish_orc_state(self.FALLBACK)
        else:
            self._recv_count = 0
            self._publish_orc_state(self.FALLBACK)
        self._publish_cmd(self._pursuit_twist((C.rx, C.ry, C.ryaw), C.path_ids))

    # EMERGENCY_STOP is terminal: entered via _enter_emergency_stop(), held at the top
    # of _step(), and cleared only by a new goal (_path_cb). No per-tick handler / no
    # auto-recovery — a sustained fault past the debounce ends the run by design.


if __name__ == "__main__":
    try:
        NewOrchestrator().run()
    except rospy.ROSInterruptException:
        pass