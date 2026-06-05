#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
orchestrator.py — unified control authority
--------------------------------------------
Sole publisher of /jetauto_controller/cmd_vel.

Owns all driving decisions:
  - Lane detection is primary on ALL road segments (straight and curves)
  - Handles junction rotation (disables lane, rotates to heading, re-enables)
  - Falls back to pure-pursuit when lane is lost

Alpha is 0 on normal road (pure lane always), ramps toward 1 only when
lane state is HOLD/STOP for hold_ramp_ticks consecutive ticks.

FSM states
----------
  IDLE        No active goal (nav_info inactive)
  NAVIGATING  Pure lane control (alpha=0); map used only for heading at junctions
  JUNCTION    In-place rotation toward next-waypoint heading; lane stays enabled
              at junction_alpha blend (10% hint); hands off once within junction_align_deg
  FALLBACK    Pure-pursuit on planned path; lane stays enabled as a silent sensor
  DONE        Goal reached, zero cmd

Topics
------
  IN  /lane_controller/cmd_vel    geometry_msgs/Twist
  IN  /lane_controller/state      std_msgs/String
  IN  /waypoint_manager/nav_info  std_msgs/Float64MultiArray
  IN  /waypoint_manager/path      std_msgs/Int32MultiArray
  IN  /odom                       nav_msgs/Odometry
  IN  /remap_transform            std_msgs/Float64MultiArray [theta,scale,tx,ty]
  OUT /jetauto_controller/cmd_vel geometry_msgs/Twist
  OUT /lane_controller/enable     std_msgs/Bool  (latched)
"""

from __future__ import print_function
import json
import math
import os
import threading

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64MultiArray, Int32MultiArray, String

from map_loader import MapLoader


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
        self._hold_ramp      = int  (rp("hold_ramp_ticks",     15))
        self._recovery_ticks = int  (rp("recovery_ticks",       5))
        self._junc_radius    = float(rp("junction_radius",     0.30))
        self._junc_align     = float(rp("junction_align_deg",  22.0))
        self._junc_spin      = float(rp("junction_spin_speed",       0.40))
        self._alpha_junc     = float(rp("junction_alpha",            0.9))
        self._lookahead      = float(rp("lookahead_m",               0.50))
        self._recovery_radius = float(rp("fallback_recovery_radius", 0.50))
        self._map_speed      = float(rp("map_drive_speed",      0.04))
        self._map_kp         = float(rp("map_kp",               1.2))
        self._map_max_w      = float(rp("map_max_angular",      0.80))
        # Drift auto-correction (EMA applied at each confirmed node passage / junction)
        self._drift_pos_alpha    = float(rp("drift_pos_alpha",       0.35))
        self._drift_theta_alpha  = float(rp("drift_theta_alpha",     0.30))
        self._drift_min_m        = float(rp("drift_min_correct_m",   0.03))
        self._drift_max_m        = float(rp("drift_max_correct_m",   0.50))
        self._drift_trigger_r    = float(rp("drift_trigger_radius",  0.20))
        # Turn-angle-aware junction: skip full rotation for gentle heading changes
        self._gentle_turn_deg   = float(rp("gentle_turn_deg",    20.0))

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

    # --------------------------------------------------------------- main FSM

    def _step(self):
        # Snapshot all shared state under one lock acquisition
        with self._lock:
            pose        = self._pose
            odom_pos    = self._odom_pos
            odom_yaw    = self._odom_yaw
            lane_cmd    = self._lane_cmd
            lane_state  = self._lane_state
            nav_info    = self._nav_info
            path_ids    = list(self._path_ids)
            lane_fresh  = self._fresh(self._lane_cmd_stamp)
            info_fresh  = self._fresh(self._nav_info_stamp)

        if pose is None:
            self._cmd_pub.publish(Twist())
            return

        # Determine whether navigation is currently active
        active = (info_fresh and nav_info is not None
                  and nav_info[self._NI_ACTIVE] > 0.5)

        if not active:
            if self._state not in (self.IDLE, self.DONE):
                rospy.loginfo("[orchestrator] nav inactive -> DONE")
                self._set_lane_enabled(False)
                self._cmd_pub.publish(Twist())  # one-shot stop on navigation end
                self._state      = self.DONE
                self._hold_count = 0
                self._recv_count = 0
                self._freerun    = False
            self._publish_orc_state(self.IDLE)
            # Freerun: pass lane commands through when AVVIA enables lane directly
            if lane_fresh and lane_state not in self._LANE_BAD:
                self._freerun = True
                self._cmd_pub.publish(lane_cmd)
            elif self._freerun:
                # FERMA just disabled lane — one-shot stop, then go silent
                self._freerun = False
                self._cmd_pub.publish(Twist())
            # else: already stopped; publish nothing so remap can drive uncontested
            return

        rx, ry, ryaw = pose
        dist         = nav_info[self._NI_DIST]
        is_junction  = nav_info[self._NI_JUNC] > 0.5
        heading_nxt  = nav_info[self._NI_HDG]   # MAP frame: robot → next node
        next_id      = int(nav_info[self._NI_NEXT])
        cur_node_id  = int(nav_info[self._NI_NODE])
        cur_path_idx = int(nav_info[self._NI_IDX])

        # Position drift correction — fires once per node when the robot is close
        # enough that the odom position is a reliable fix (dist < drift_trigger_radius).
        # Guarded to NAVIGATING state only: must not fire mid-junction and corrupt heading_nxt.
        # Junction nodes get heading-fixed separately via _apply_theta_correction on JUNCTION exit.
        if cur_node_id != self._last_corrected_node:
            self._node_corrected      = False
            self._last_corrected_node = cur_node_id
        if (self._state == self.NAVIGATING
                and not self._node_corrected
                and dist < self._drift_trigger_r
                and odom_pos is not None and self._map is not None
                and lane_state not in self._LANE_BAD):
            self._apply_drift_correction(cur_node_id, odom_pos[0], odom_pos[1])
            self._node_corrected = True

        # ======== JUNCTION ========
        # Re-trigger guard: skip if this junction node was already handled this path.
        # _handled_junction is cleared on every new path arrival.
        _enter_junction = (is_junction and next_id >= 0
                           and dist <= self._junc_radius
                           and cur_node_id != self._handled_junction)

        # Turn-angle gate: skip full rotation for gentle heading changes.
        # Use map edge directions (stable) rather than robot-relative heading_nxt.
        if _enter_junction and self._map is not None and cur_path_idx > 0:
            prev_id = (int(path_ids[cur_path_idx - 1])
                       if cur_path_idx - 1 < len(path_ids) else -1)
            if prev_id >= 0 and next_id >= 0:
                try:
                    px, py   = self._map.node_xy(prev_id)
                    cx, cy   = self._map.node_xy(cur_node_id)
                    nx2, ny2 = self._map.node_xy(next_id)
                    incoming   = math.atan2(cy - py, cx - px)
                    outgoing   = math.atan2(ny2 - cy, nx2 - cx)
                    turn_angle = abs(angle_diff(outgoing, incoming))
                    if turn_angle < math.radians(self._gentle_turn_deg):
                        _enter_junction = False
                        rospy.logdebug_throttle(
                            2.0, "[orchestrator] junc %d: gentle turn %.1fdeg — skip rotation",
                            cur_node_id, math.degrees(turn_angle))
                except Exception:
                    pass

        if self._state == self.JUNCTION or _enter_junction:

            if self._state != self.JUNCTION:
                self._state      = self.JUNCTION
                self._hold_count = 0
                self._recv_count = 0
                # Lane stays enabled — its angular contribution is blended at alpha_junc
                # so the robot gets a 10% hint from any visible road marking during the spin.
                rospy.loginfo("[orchestrator] -> JUNCTION (dist=%.2fm)", dist)

            self._publish_orc_state(self.JUNCTION)
            err      = angle_diff(heading_nxt, ryaw)
            spin_ang = clamp(1.5 * err, -self._junc_spin, self._junc_spin)

            if abs(err) < math.radians(self._junc_align):
                # Roughly aligned — hand off to lane immediately.
                # Lane detection now sees the new road direction and finishes the turn.
                self._handled_junction = cur_node_id
                self._state = self.NAVIGATING
                self._publish_orc_state(self.NAVIGATING)
                # Robot is facing heading_nxt — use alignment as a theta fix point.
                self._apply_theta_correction(heading_nxt, odom_yaw)
                rospy.loginfo("[orchestrator] JUNCTION aligned (err=%.1fdeg) -> NAVIGATING",
                              math.degrees(abs(err)))
                return

            # Spinning: 90% map heading, 10% lane hint (zero if lane is bad/stale)
            lane_ang = (lane_cmd.angular.z
                        if lane_fresh and lane_state not in self._LANE_BAD else 0.0)
            twist = Twist()
            twist.linear.x  = 0.0  # no forward drift during spin
            twist.angular.z = (1.0 - self._alpha_junc) * lane_ang + self._alpha_junc * spin_ang
            self._cmd_pub.publish(twist)
            return

        # ======== MAP FALLBACK ========
        # Lane stays enabled so lane_state reflects actual road visibility.
        # Recovery requires both optical signal (lane OK) and positional proximity
        # (robot close to a path node) to avoid false exits far from the road.
        if self._state == self.FALLBACK:
            near_path = dist <= self._recovery_radius
            lane_ok   = lane_state not in self._LANE_BAD
            if lane_ok and near_path:
                self._recv_count += 1
                if self._recv_count >= self._recovery_ticks:
                    self._recv_count = 0
                    self._hold_count = 0
                    self._state = self.NAVIGATING
                    self._publish_orc_state(self.NAVIGATING)
                    rospy.loginfo(
                        "[orchestrator] FALLBACK -> NAVIGATING "
                        "(lane OK, dist=%.2fm)", dist)
                    # Fall through to NAVIGATING section below
                else:
                    # Hysteresis: both conditions met but not confirmed yet
                    self._publish_orc_state(self.FALLBACK)
                    self._cmd_pub.publish(self._pursuit_twist(pose, path_ids))
                    return
            else:
                if self._recv_count > 0:
                    rospy.logdebug_throttle(
                        1.0, "[orchestrator] FALLBACK recovery reset "
                        "(lane_ok=%s near=%s dist=%.2fm)", lane_ok, near_path, dist)
                self._recv_count = 0
                self._publish_orc_state(self.FALLBACK)
                self._cmd_pub.publish(self._pursuit_twist(pose, path_ids))
                return

        # ======== NAVIGATING ========
        if self._state != self.NAVIGATING:
            self._state = self.NAVIGATING
            self._set_lane_enabled(True)
            rospy.loginfo("[orchestrator] -> NAVIGATING")
        self._publish_orc_state(self.NAVIGATING)

        # Compute dynamic alpha
        if next_id < 0:
            # Final segment: no next node — pure lane until goal distance stop
            alpha = 0.0
            self._hold_count = 0
        elif lane_state in self._LANE_LOST:
            # Only HOLD/STOP ramp toward fallback — DISABLED means lane was
            # intentionally turned off (e.g. startup, post-remap) and must not
            # count as "lane lost".
            self._hold_count += 1
            if self._hold_count >= self._hold_ramp:
                self._hold_count = 0
                self._recv_count = 0
                self._state      = self.FALLBACK
                self._publish_orc_state(self.FALLBACK)
                rospy.loginfo("[orchestrator] NAVIGATING -> FALLBACK (lane lost)")
                self._cmd_pub.publish(self._pursuit_twist(pose, path_ids))
                return
            alpha = self._hold_count / float(self._hold_ramp)
        else:
            self._hold_count = 0
            alpha = 0.0  # lane is primary; map only used during FALLBACK

        # Blend lane + map on angular.z; speed from lane (or zero if stale)
        lane_angular = lane_cmd.angular.z if lane_fresh else 0.0
        lane_linear  = lane_cmd.linear.x  if lane_fresh else 0.0
        map_ang      = self._map_angular(heading_nxt, ryaw) if next_id >= 0 else 0.0

        twist = Twist()
        twist.linear.x  = lane_linear
        twist.angular.z = (1.0 - alpha) * lane_angular + alpha * map_ang
        self._cmd_pub.publish(twist)

    def run(self):
        rate = rospy.Rate(self._rate_hz)
        while not rospy.is_shutdown():
            try:
                self._step()
            except Exception as e:
                rospy.logerr_throttle(2.0, "[orchestrator] step err: %s", e)
            rate.sleep()
        # Clean shutdown
        self._cmd_pub.publish(Twist())
        self._set_lane_enabled(False)


if __name__ == "__main__":
    try:
        Orchestrator().run()
    except rospy.ROSInterruptException:
        pass