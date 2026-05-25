#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
waypoint_manager_node.py
------------------------
Loads the map, receives a goal (start, end), computes the path with Dijkstra,
and follows waypoints using odometry. Between waypoints it hands lateral control
to the lane_controller (publishing True on /lane_controller/enable). At
junctions (degree>2) it takes direct control to rotate toward the next
waypoint in the path.

Topic:
  IN  - /odom                        (nav_msgs/Odometry)
  IN  - /waypoint_manager/goal       (std_msgs/Int32MultiArray) [start, end]
  OUT - /waypoint_manager/status     (std_msgs/String)
  OUT - /jetauto_controller/cmd_vel  (geometry_msgs/Twist)  [only during junction maneuvers]
  OUT - /lane_controller/enable      (std_msgs/Bool)
  OUT - /waypoint_manager/path       (std_msgs/Int32MultiArray) current path (for dashboard)
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
    # Z-Y-X yaw from quaternion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    """Angular difference in [-pi, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


class WaypointManagerNode(object):

    # FSM states
    IDLE     = "IDLE"
    NAV      = "NAVIGATING"
    JUNCTION = "JUNCTION"
    DONE     = "GOAL_REACHED"
    ERROR    = "ERROR"

    def __init__(self):
        rospy.init_node("waypoint_manager_node", anonymous=False)

        ns = "waypoint_manager/"
        self.map_file        = rospy.get_param(ns + "map_file")
        self.odom_topic      = rospy.get_param(ns + "odom_topic", "/odom")
        self.goal_topic      = rospy.get_param(ns + "goal_topic", "/waypoint_manager/goal")
        self.status_topic    = rospy.get_param(ns + "status_topic", "/waypoint_manager/status")
        self.cmd_topic       = rospy.get_param(ns + "cmd_topic", "/jetauto_controller/cmd_vel")
        self.enable_topic    = rospy.get_param(ns + "lane_enable_topic", "/lane_controller/enable")

        self.tol             = float(rospy.get_param(ns + "waypoint_tolerance", 0.18))
        self.junction_speed  = float(rospy.get_param(ns + "junction_slowdown", 0.08))
        self.junction_radius = float(rospy.get_param(ns + "junction_radius", 0.25))
        self.rate_hz         = float(rospy.get_param(ns + "rate_hz", 10))

        # Load map
        rospy.loginfo("[waypoint_manager] loading map: %s", self.map_file)
        self.map = MapLoader(self.map_file)
        rospy.loginfo("[waypoint_manager] %s", self.map.stats())

        # 2D similarity transform: odom = scale * R(theta) * map + t
        # Loaded from remap_params.json at startup, updated live via /remap_transform.
        self._remap_theta = 0.0
        self._remap_scale = 1.0
        self._remap_tx    = 0.0
        self._remap_ty    = 0.0
        self._load_remap_params()

        # State
        self.lock = threading.Lock()
        self.pose = None      # (x, y, yaw) in MAP frame
        self.path = []        # list of node_id
        self.idx  = 0         # current waypoint index in path
        self.state = self.IDLE

        # Pub/Sub
        self.cmd_pub    = rospy.Publisher(self.cmd_topic,    Twist, queue_size=1)
        self.enable_pub = rospy.Publisher(self.enable_topic, Bool,  queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.path_pub   = rospy.Publisher("/waypoint_manager/path",
                                          Int32MultiArray, queue_size=1, latch=True)

        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=10)
        rospy.Subscriber(self.goal_topic, Int32MultiArray, self._goal_cb, queue_size=1)
        rospy.Subscriber("/remap_transform", Float64MultiArray, self._remap_cb, queue_size=1)

        self._publish_status(self.IDLE, "")
        self._set_lane_enabled(False)
        rospy.loginfo("[waypoint_manager] ready.")

    # ------------------------------------------------------------- helpers: remap
    def _load_remap_params(self):
        """Load persisted transform from ../web/remap_params.json."""
        import json as _json
        import os as _os
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
                "[waypoint_manager] remap_params loaded: theta=%.3f scale=%.3f tx=%.3f ty=%.3f",
                self._remap_theta, self._remap_scale, self._remap_tx, self._remap_ty)
        except Exception as e:
            rospy.loginfo("[waypoint_manager] no remap_params.json (%s), using identity", e)

    def _remap_cb(self, msg):
        """Live-update the similarity transform from /remap_transform [theta, scale, tx, ty]."""
        if len(msg.data) < 4:
            return
        self._remap_theta = float(msg.data[0])
        self._remap_scale = float(msg.data[1])
        self._remap_tx    = float(msg.data[2])
        self._remap_ty    = float(msg.data[3])
        rospy.loginfo(
            "[waypoint_manager] remap_transform updated: theta=%.3f scale=%.3f tx=%.3f ty=%.3f",
            self._remap_theta, self._remap_scale, self._remap_tx, self._remap_ty)

    def _odom_to_map(self, x, y):
        """Apply inverse similarity transform: map = R(-theta)/scale * (odom - t)."""
        theta = self._remap_theta
        scale = self._remap_scale if self._remap_scale != 0.0 else 1.0
        cos_t = math.cos(-theta)
        sin_t = math.sin(-theta)
        dx = x - self._remap_tx
        dy = y - self._remap_ty
        return (cos_t * dx - sin_t * dy) / scale, (sin_t * dx + cos_t * dy) / scale

    # ------------------------------------------------------------- callbacks
    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        mx, my = self._odom_to_map(p.x, p.y)
        with self.lock:
            self.pose = (mx, my, yaw)

    def _goal_cb(self, msg):
        if len(msg.data) == 0:
            # Empty goal = cancel current navigation
            with self.lock:
                self.path = []
                self.idx  = 0
            self._publish_status("IDLE")
            # Stop the robot
            stop_twist = Twist()
            self.cmd_pub.publish(stop_twist)
            rospy.loginfo("[waypoint_manager] navigation canceled from the dashboard")
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
            self.path = path
            self.idx  = 0
            self.state = self.NAV

        # Publish path for the dashboard
        m = Int32MultiArray()
        m.data = [int(x) for x in path]
        self.path_pub.publish(m)

        self._set_lane_enabled(True)
        self._publish_status(self.NAV,
                             "path=%s len=%.2fm" %
                             (path, self.map.path_length(path)))
        rospy.loginfo("[waypoint_manager] new path %d nodes (%.2f m)",
                      len(path), self.map.path_length(path))

    # ------------------------------------------------------------- helpers
    def _set_lane_enabled(self, on):
        self.enable_pub.publish(Bool(data=bool(on)))

    def _publish_status(self, state, info=""):
        s = state if not info else "%s | %s" % (state, info)
        self.status_pub.publish(String(data=s))

    def _stop_robot(self):
        self.cmd_pub.publish(Twist())

    def _current_waypoint(self):
        with self.lock:
            if not self.path or self.idx >= len(self.path):
                return None
            return self.path[self.idx]

    def _is_at(self, node_id, pose, tol):
        nx, ny = self.map.node_xy(node_id)
        return math.hypot(pose[0] - nx, pose[1] - ny) <= tol

    def _heading_to(self, node_id, pose):
        nx, ny = self.map.node_xy(node_id)
        return math.atan2(ny - pose[1], nx - pose[0])

    # ------------------------------------------------------------- junction handling
    def _handle_junction(self, pose):
        """
        When the robot is within junction_radius of a junction node,
        it takes direct control: disables the lane_controller, rotates toward the
        NEXT waypoint in the path (chosen by Dijkstra), then re-enables
        the lane controller.
        """
        with self.lock:
            cur_idx = self.idx
            path = list(self.path)

        if cur_idx + 1 >= len(path):
            return  # no next waypoint, handled by main loop

        next_id = path[cur_idx + 1]
        target_yaw = self._heading_to(next_id, pose)
        err = angle_diff(target_yaw, pose[2])

        twist = Twist()
        if abs(err) < math.radians(8.0):
            # Aligned: advance slowly and hand back control
            twist.linear.x = self.junction_speed
            self.cmd_pub.publish(twist)
            self._set_lane_enabled(True)
            with self.lock:
                self.state = self.NAV
            self._publish_status(self.NAV, "exit junction -> %d" % next_id)
            return

        # Rotate in place (with slight forward motion)
        twist.linear.x  = 0.03
        twist.angular.z = max(-1.0, min(1.0, 1.5 * err))
        self.cmd_pub.publish(twist)
        self._publish_status(self.JUNCTION,
                             "rot to %d err=%.1fdeg" % (next_id, math.degrees(err)))

    # ------------------------------------------------------------- main loop
    def _step(self):
        with self.lock:
            pose = self.pose
            state = self.state
            path = list(self.path)
            idx  = self.idx

        if pose is None:
            return
        if state == self.IDLE or state == self.DONE or state == self.ERROR:
            return
        if not path or idx >= len(path):
            return

        cur_id = path[idx]
        cur_xy = self.map.node_xy(cur_id)
        dist = math.hypot(pose[0] - cur_xy[0], pose[1] - cur_xy[1])

        # Waypoint reached?
        if dist <= self.tol:
            new_idx = idx + 1
            if new_idx >= len(path):
                # Path complete
                self._stop_robot()
                self._set_lane_enabled(False)
                with self.lock:
                    self.state = self.DONE
                self._publish_status(self.DONE, "last=%d" % cur_id)
                rospy.loginfo("[waypoint_manager] GOAL reached.")
                return
            with self.lock:
                self.idx = new_idx
            rospy.loginfo("[waypoint_manager] WP %d (%s) reached, next=%d",
                          cur_id, self.map.node_type(cur_id), path[new_idx])
            return

        # Near a junction? -> trigger junction maneuver
        # (check current waypoint: if it's a junction and we're within junction_radius)
        if self.map.is_junction(cur_id) and dist <= self.junction_radius and idx + 1 < len(path):
            with self.lock:
                self.state = self.JUNCTION
            self._set_lane_enabled(False)
            self._handle_junction(pose)
            return

        # Normal navigation: lane_controller drives, here only monitoring
        if state != self.NAV:
            with self.lock:
                self.state = self.NAV
            self._set_lane_enabled(True)
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


if __name__ == "__main__":
    try:
        WaypointManagerNode().run()
    except rospy.ROSInterruptException:
        pass
