#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
waypoint_manager_node.py
------------------------
Loads the map, receives a goal (start, end), computes the path with Dijkstra,
and drives the robot along the path.

FSM states
----------
  IDLE          no active goal
  NAVIGATING    lane_controller drives laterally; this node monitors progress
  JUNCTION      in-place rotation toward the next-waypoint heading
  MAP_FALLBACK  pure-pursuit on the planned path when lane markings are lost
  GOAL_REACHED  terminal; robot stopped
  ERROR         terminal; planning failure

Waypoint advancement
--------------------
Intermediate (non-junction) nodes are advanced via a dot-product test: when the
robot's projection onto the segment node[idx]->node[idx+1] is positive, the robot
has passed node[idx] and idx advances.  This is robust to odometry drift; the
robot never stalls on a node it has already passed.

Junction nodes are never skipped by the dot-product test.  They require the robot
to enter JUNCTION state and complete the rotation maneuver, which advances idx
explicitly upon completion.

The final node uses a distance check (waypoint_tolerance) as a stop condition.

Map-follower integration
------------------------
When lane_controller reports HOLD/STOP for hold_fallback_frames consecutive ticks,
this node switches to MAP_FALLBACK: it disables the lane controller and drives the
robot with a pure-pursuit controller along the planned path.  When the lane
recovers for lane_recovery_frames consecutive ticks, NAV resumes.

This replaces the former standalone map_follower_node.py process.

Topics
------
  IN  /odom                        nav_msgs/Odometry
  IN  /waypoint_manager/goal       std_msgs/Int32MultiArray  [start, end]
  IN  /lane_controller/state       std_msgs/String
  IN  /remap_transform             std_msgs/Float64MultiArray [theta,scale,tx,ty]
  OUT /waypoint_manager/status     std_msgs/String  (latched)
  OUT /waypoint_manager/path       std_msgs/Int32MultiArray  (latched)
  OUT /jetauto_controller/cmd_vel  geometry_msgs/Twist
  OUT /lane_controller/enable      std_msgs/Bool  (latched)
  OUT /map_follower/active         std_msgs/Bool  (latched)
"""

from __future__ import print_function
import math
import threading

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64MultiArray, String, Int32MultiArray

from map_loader import MapLoader, NODE_JUNCTION


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


class WaypointManagerNode(object):

    IDLE         = "IDLE"
    NAV          = "NAVIGATING"
    JUNCTION     = "JUNCTION"
    MAP_FALLBACK = "MAP_FALLBACK"
    DONE         = "GOAL_REACHED"
    ERROR        = "ERROR"

    _NO_LANE = frozenset(("HOLD", "STOP"))

    def __init__(self):
        rospy.init_node("waypoint_manager_node", anonymous=False)

        ns = "waypoint_manager/"
        self.map_file        = rospy.get_param(ns + "map_file")
        self.odom_topic      = rospy.get_param(ns + "odom_topic",        "/odom")
        self.goal_topic      = rospy.get_param(ns + "goal_topic",        "/waypoint_manager/goal")
        self.status_topic    = rospy.get_param(ns + "status_topic",      "/waypoint_manager/status")
        self.cmd_topic       = rospy.get_param(ns + "cmd_topic",         "/jetauto_controller/cmd_vel")
        self.enable_topic    = rospy.get_param(ns + "lane_enable_topic", "/lane_controller/enable")

        self.tol                 = float(rospy.get_param(ns + "waypoint_tolerance",  0.25))
        self.junction_speed      = float(rospy.get_param(ns + "junction_slowdown",   0.08))
        self.junction_radius     = float(rospy.get_param(ns + "junction_radius",     0.30))
        self.junction_spin_speed = float(rospy.get_param(ns + "junction_spin_speed", 0.40))
        self.rate_hz             = float(rospy.get_param(ns + "rate_hz",             25))

        # Map-follower (pure-pursuit fallback) parameters — namespace map_follower/
        mf = "map_follower/"
        self._mf_enabled     = bool (rospy.get_param(mf + "enable",               False))
        self._mf_hold_frames = int  (rospy.get_param(mf + "hold_fallback_frames", 15))
        self._mf_recv_frames = int  (rospy.get_param(mf + "lane_recovery_frames",  5))
        self._mf_lookahead   = float(rospy.get_param(mf + "lookahead_m",          0.50))
        self._mf_speed       = float(rospy.get_param(mf + "map_drive_speed",      0.08))
        self._mf_kp          = float(rospy.get_param(mf + "angular_kp",           1.2))
        self._mf_max_w       = float(rospy.get_param(mf + "max_angular_z",        0.80))

        # Load map
        rospy.loginfo("[waypoint_manager] loading map: %s", self.map_file)
        self.map = MapLoader(self.map_file)
        rospy.loginfo("[waypoint_manager] %s", self.map.stats())

        # 2D similarity transform: odom = scale * R(theta) * map + t
        # Loaded from remap_params.json at startup; updated live via /remap_transform.
        self._remap_theta = 0.0
        self._remap_scale = 1.0
        self._remap_tx    = 0.0
        self._remap_ty    = 0.0
        self._load_remap_params()

        # Shared state — lock protects fields written from ROS callbacks
        self.lock        = threading.Lock()
        self.pose        = None    # (x, y, yaw) in MAP frame
        self.path        = []      # list of node IDs in planned order
        self.idx         = 0       # index of the current target node in path
        self.state       = self.IDLE
        self._lane_state = "STOP"  # latest /lane_controller/state value

        # Step-local counters — only written in _step(), reset on new goal
        self._hold_count = 0   # consecutive HOLD/STOP ticks while in NAV
        self._recv_count = 0   # consecutive non-HOLD ticks while in MAP_FALLBACK

        # Publishers
        self.cmd_pub     = rospy.Publisher(self.cmd_topic,    Twist,  queue_size=1)
        self.enable_pub  = rospy.Publisher(self.enable_topic, Bool,   queue_size=1, latch=True)
        self.status_pub  = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.path_pub    = rospy.Publisher("/waypoint_manager/path",
                                           Int32MultiArray, queue_size=1, latch=True)
        self._active_pub = rospy.Publisher("/map_follower/active", Bool, queue_size=1, latch=True)

        # Subscribers
        rospy.Subscriber(self.odom_topic,          Odometry,          self._odom_cb,       queue_size=10)
        rospy.Subscriber(self.goal_topic,          Int32MultiArray,   self._goal_cb,       queue_size=1)
        rospy.Subscriber("/remap_transform",       Float64MultiArray, self._remap_cb,      queue_size=1)
        rospy.Subscriber("/lane_controller/state", String,            self._lane_state_cb, queue_size=1)

        self._publish_status(self.IDLE, "")
        self._set_lane_enabled(False)
        self._active_pub.publish(Bool(data=False))

        if self._mf_enabled:
            rospy.loginfo(
                "[waypoint_manager] map fallback ENABLED "
                "(lookahead=%.2fm  speed=%.2fm/s  hold=%d  recv=%d ticks)",
                self._mf_lookahead, self._mf_speed,
                self._mf_hold_frames, self._mf_recv_frames)
        else:
            rospy.logwarn("[waypoint_manager] map fallback DISABLED "
                          "(set map_follower/enable: true to activate)")

        rospy.loginfo("[waypoint_manager] ready. tol=%.2fm  junction_r=%.2fm  rate=%.0fHz",
                      self.tol, self.junction_radius, self.rate_hz)

    # ------------------------------------------------------------------ remap

    def _load_remap_params(self):
        import json as _json, os as _os
        here  = _os.path.dirname(_os.path.abspath(__file__))
        fpath = _os.path.join(here, "..", "web", "remap_params.json")
        try:
            with open(fpath) as f:
                p = _json.load(f)
            self._remap_theta = float(p.get("theta", 0.0))
            self._remap_scale = float(p.get("scale", 1.0))
            self._remap_tx    = float(p.get("tx",    0.0))
            self._remap_ty    = float(p.get("ty",    0.0))
            rospy.loginfo(
                "[waypoint_manager] remap loaded: theta=%.3f scale=%.3f tx=%.3f ty=%.3f",
                self._remap_theta, self._remap_scale, self._remap_tx, self._remap_ty)
        except Exception as e:
            rospy.loginfo("[waypoint_manager] no remap_params.json (%s), using identity", e)

    def _remap_cb(self, msg):
        if len(msg.data) < 4:
            return
        self._remap_theta = float(msg.data[0])
        self._remap_scale = float(msg.data[1])
        self._remap_tx    = float(msg.data[2])
        self._remap_ty    = float(msg.data[3])
        rospy.loginfo(
            "[waypoint_manager] remap updated: theta=%.3f scale=%.3f tx=%.3f ty=%.3f",
            self._remap_theta, self._remap_scale, self._remap_tx, self._remap_ty)

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
        mx, my = self._odom_to_map(p.x, p.y)
        # Transform yaw to map frame: map_yaw = odom_yaw - theta
        map_yaw = angle_diff(yaw, self._remap_theta)
        with self.lock:
            self.pose = (mx, my, map_yaw)

    def _lane_state_cb(self, msg):
        with self.lock:
            self._lane_state = msg.data

    def _goal_cb(self, msg):
        if len(msg.data) == 0:
            with self.lock:
                self.path  = []
                self.idx   = 0
                self.state = self.IDLE
            self._hold_count = 0
            self._recv_count = 0
            self._set_lane_enabled(False)
            self._active_pub.publish(Bool(data=False))
            self.cmd_pub.publish(Twist())
            self._publish_status(self.IDLE)
            rospy.loginfo("[waypoint_manager] navigation canceled")
            return

        if len(msg.data) < 2:
            rospy.logwarn("[waypoint_manager] malformed goal (expected [start, end])")
            return

        start, end = int(msg.data[0]), int(msg.data[1])
        try:
            path = self.map.get_path(start, end)
        except Exception as e:
            rospy.logerr("[waypoint_manager] Dijkstra failed: %s", e)
            self._publish_status(self.ERROR, str(e))
            return

        with self.lock:
            self.path  = path
            self.idx   = 0
            self.state = self.NAV
        self._hold_count = 0
        self._recv_count = 0

        m = Int32MultiArray()
        m.data = [int(x) for x in path]
        self.path_pub.publish(m)

        self._active_pub.publish(Bool(data=False))
        self._set_lane_enabled(True)
        self._publish_status(self.NAV,
                             "path=%s len=%.2fm" % (path, self.map.path_length(path)))
        rospy.loginfo("[waypoint_manager] new path %d nodes (%.2fm)",
                      len(path), self.map.path_length(path))

    # ----------------------------------------------------------------- helpers

    def _set_lane_enabled(self, on):
        self.enable_pub.publish(Bool(data=bool(on)))

    def _publish_status(self, state, info=""):
        s = state if not info else "%s | %s" % (state, info)
        self.status_pub.publish(String(data=s))

    def _stop_robot(self):
        self.cmd_pub.publish(Twist())

    # ---------------------------------------------------------------- junction

    def _handle_junction(self, pose):
        """Rotate in-place toward the next waypoint heading, then re-enable lane."""
        with self.lock:
            cur_idx = self.idx
            path    = list(self.path)

        if cur_idx + 1 >= len(path):
            return

        next_id    = path[cur_idx + 1]
        nx, ny     = self.map.node_xy(next_id)
        target_yaw = math.atan2(ny - pose[1], nx - pose[0])
        err        = angle_diff(target_yaw, pose[2])

        twist = Twist()
        if abs(err) < math.radians(8.0):
            # Aligned: creep forward and hand back to lane controller
            twist.linear.x = self.junction_speed
            self.cmd_pub.publish(twist)
            self._set_lane_enabled(True)
            self._active_pub.publish(Bool(data=False))
            with self.lock:
                self.state = self.NAV
                self.idx   = cur_idx + 1   # advance past the junction node
            self._hold_count = 0
            self._recv_count = 0
            self._publish_status(self.NAV, "exit junction -> %d" % next_id)
            return

        # Pure in-place spin — mecanum wheels avoid tire slip
        spd = self.junction_spin_speed
        twist.linear.x  = 0.0
        twist.angular.z = max(-spd, min(spd, 1.5 * err))
        self.cmd_pub.publish(twist)
        self._publish_status(self.JUNCTION,
                             "rot to %d err=%.1fdeg" % (next_id, math.degrees(err)))

    # --------------------------------------------------------- waypoint helpers

    def _advance_past_nodes(self, rx, ry, path, idx):
        """Advance idx past non-junction nodes the robot has physically passed.

        Uses a dot-product test on each segment: if the robot's projection onto
        path[idx]->path[idx+1] is positive, the robot is past path[idx] and idx
        advances.  Junction nodes are never skipped — they require the explicit
        rotation maneuver.  The final node is never skipped (handled by goal check).
        """
        while idx < len(path) - 1:
            if self.map.is_junction(path[idx]):
                break  # junctions handled by _handle_junction
            nx,  ny  = self.map.node_xy(path[idx])
            nx2, ny2 = self.map.node_xy(path[idx + 1])
            if (rx - nx) * (nx2 - nx) + (ry - ny) * (ny2 - ny) > 0:
                idx += 1
            else:
                break
        return idx

    # ---------------------------------------------------- pure-pursuit helpers

    def _pure_pursuit_twist(self, pose, path):
        rx, ry, ryaw = pose
        carrot = self._mf_carrot(rx, ry, path)
        if carrot is None:
            return Twist()
        heading = math.atan2(carrot[1] - ry, carrot[0] - rx)
        err     = angle_diff(heading, ryaw)
        t = Twist()
        t.linear.x  = self._mf_speed
        t.angular.z = max(-self._mf_max_w, min(self._mf_max_w, self._mf_kp * err))
        return t

    def _mf_carrot(self, rx, ry, path):
        """Find the carrot point exactly lookahead_m ahead of the robot on the path."""
        if not path:
            return None
        # Closest node
        best_idx, best_dist = 0, float("inf")
        for i, nid in enumerate(path):
            nx, ny = self.map.node_xy(nid)
            d = math.hypot(rx - nx, ry - ny)
            if d < best_dist:
                best_dist, best_idx = d, i
        # Overshoot correction: if robot is past best_idx toward best_idx+1, advance
        if best_idx < len(path) - 1:
            ax, ay = self.map.node_xy(path[best_idx])
            bx, by = self.map.node_xy(path[best_idx + 1])
            if (rx - ax) * (bx - ax) + (ry - ay) * (by - ay) > 0:
                best_idx += 1
        # Walk forward until lookahead distance is reached
        accum          = 0.0
        prev_x, prev_y = self.map.node_xy(path[best_idx])
        for i in range(best_idx + 1, len(path)):
            nx, ny = self.map.node_xy(path[i])
            seg = math.hypot(nx - prev_x, ny - prev_y)
            if accum + seg >= self._mf_lookahead:
                frac = (self._mf_lookahead - accum) / max(seg, 1e-9)
                return (prev_x + frac * (nx - prev_x),
                        prev_y + frac * (ny - prev_y))
            accum += seg
            prev_x, prev_y = nx, ny
        return self.map.node_xy(path[-1])

    # --------------------------------------------------------------- main loop

    def _step(self):
        with self.lock:
            pose       = self.pose
            state      = self.state
            path       = list(self.path)
            idx        = self.idx
            lane_state = self._lane_state

        if pose is None:
            return
        if state in (self.IDLE, self.DONE, self.ERROR):
            return
        if not path or idx >= len(path):
            return

        # --- Dot-product advancement ---
        # Skip past non-junction nodes the robot has already passed.
        # Not applied while in JUNCTION state (that state manages idx itself).
        if state != self.JUNCTION:
            new_idx = self._advance_past_nodes(pose[0], pose[1], path, idx)
            if new_idx > idx:
                with self.lock:
                    self.idx = new_idx
                rospy.loginfo("[waypoint_manager] passed WP(s) %s -> targeting %d",
                              path[idx:new_idx],
                              path[new_idx] if new_idx < len(path) else -1)
                idx = new_idx
                if idx >= len(path):
                    self._stop_robot()
                    self._set_lane_enabled(False)
                    self._active_pub.publish(Bool(data=False))
                    with self.lock:
                        self.state = self.DONE
                    self._publish_status(self.DONE, "last=%d" % path[-1])
                    rospy.loginfo("[waypoint_manager] GOAL reached.")
                    return

        cur_id = path[idx]
        cur_xy = self.map.node_xy(cur_id)
        dist   = math.hypot(pose[0] - cur_xy[0], pose[1] - cur_xy[1])

        # --- Goal: last waypoint (distance-based stop) ---
        if idx == len(path) - 1 and dist <= self.tol:
            self._stop_robot()
            self._set_lane_enabled(False)
            self._active_pub.publish(Bool(data=False))
            with self.lock:
                self.state = self.DONE
            self._publish_status(self.DONE, "last=%d" % cur_id)
            rospy.loginfo("[waypoint_manager] GOAL reached.")
            return

        # --- Junction maneuver ---
        # Sticky: once in JUNCTION state, keep handling until _handle_junction
        # sets state=NAV on alignment (avoids falling through if dist drifts).
        if state == self.JUNCTION or (
                self.map.is_junction(cur_id)
                and dist <= self.junction_radius
                and idx + 1 < len(path)):
            if state != self.JUNCTION:
                self._set_lane_enabled(False)
                self._active_pub.publish(Bool(data=False))
                with self.lock:
                    self.state = self.JUNCTION
                rospy.loginfo("[waypoint_manager] entering JUNCTION at node %d (d=%.2fm)",
                              cur_id, dist)
            self._handle_junction(pose)
            return

        # --- MAP_FALLBACK: pure-pursuit when lane is lost ---
        if state == self.MAP_FALLBACK:
            if lane_state not in self._NO_LANE:
                self._recv_count += 1
                if self._recv_count >= self._mf_recv_frames:
                    self._recv_count = 0
                    self._hold_count = 0
                    with self.lock:
                        self.state = self.NAV
                    self._set_lane_enabled(True)
                    self._active_pub.publish(Bool(data=False))
                    self._publish_status(self.NAV, "lane_recovered")
                    rospy.loginfo("[waypoint_manager] MAP_FALLBACK -> NAV (lane recovered)")
                    return
            else:
                self._recv_count = 0
                self.cmd_pub.publish(self._pure_pursuit_twist(pose, path))
            self._publish_status(self.MAP_FALLBACK,
                                 "wp=%d (%s) d=%.2fm idx=%d/%d" %
                                 (cur_id, self.map.node_type(cur_id),
                                  dist, idx + 1, len(path)))
            return

        # --- Normal NAV ---
        if state != self.NAV:
            with self.lock:
                self.state = self.NAV
            self._set_lane_enabled(True)

        # Check whether to activate map fallback
        if self._mf_enabled:
            if lane_state in self._NO_LANE:
                self._hold_count += 1
                if self._hold_count >= self._mf_hold_frames:
                    self._hold_count = 0
                    self._recv_count = 0
                    with self.lock:
                        self.state = self.MAP_FALLBACK
                    self._set_lane_enabled(False)
                    self._active_pub.publish(Bool(data=True))
                    self._publish_status(self.MAP_FALLBACK, "lane_lost wp=%d" % cur_id)
                    rospy.loginfo("[waypoint_manager] NAV -> MAP_FALLBACK (lane lost at wp %d)",
                                  cur_id)
                    return
            else:
                self._hold_count = 0

        self._publish_status(self.NAV,
                             "wp=%d (%s) d=%.2fm idx=%d/%d" %
                             (cur_id, self.map.node_type(cur_id),
                              dist, idx + 1, len(path)))

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            try:
                self._step()
            except Exception as e:
                rospy.logerr_throttle(2.0, "[waypoint_manager] step err: %s" % e)
            rate.sleep()
        self._stop_robot()
        self._active_pub.publish(Bool(data=False))


if __name__ == "__main__":
    try:
        WaypointManagerNode().run()
    except rospy.ROSInterruptException:
        pass