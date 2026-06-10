#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
waypoint_manager_node.py — pure path-tracker
---------------------------------------------
Loads the map, receives goals, runs Dijkstra, tracks robot position along
the planned path via dot-product, and publishes real-time navigation info
for the orchestrator.

All driving decisions (junction rotation, map fallback, lane/map blending)
are handled by orchestrator.py.

Topics
------
  IN  /odom                       nav_msgs/Odometry
  IN  /waypoint_manager/goal      std_msgs/Int32MultiArray  [start, end]
  IN  /remap_transform            std_msgs/Float64MultiArray [theta,scale,tx,ty]
  OUT /waypoint_manager/path      std_msgs/Int32MultiArray  (latched)
  OUT /waypoint_manager/nav_info  std_msgs/Float64MultiArray (25 Hz)
      layout: [node_id, next_id, dist_to_current, is_junction,
               heading_to_next, path_idx, path_len, is_active]
  OUT /waypoint_manager/status    std_msgs/String  (latched)
"""

from __future__ import print_function
import math
import threading

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, String, Int32MultiArray

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


class WaypointManagerNode(object):

    IDLE  = "IDLE"
    NAV   = "NAVIGATING"
    DONE  = "GOAL_REACHED"
    ERROR = "ERROR"

    def __init__(self):
        rospy.init_node("waypoint_manager_node", anonymous=False)

        ns = "waypoint_manager/"
        self.map_file     = rospy.get_param(ns + "map_file")
        self.odom_topic   = rospy.get_param(ns + "odom_topic",   "/odom")
        self.goal_topic   = rospy.get_param(ns + "goal_topic",   "/waypoint_manager/goal")
        self.status_topic = rospy.get_param(ns + "status_topic", "/waypoint_manager/status")
        self.tol          = float(rospy.get_param(ns + "waypoint_tolerance", 0.25))
        self.rate_hz      = float(rospy.get_param(ns + "rate_hz", 25.0))

        rospy.loginfo("[waypoint_manager] loading map: %s", self.map_file)
        self.map = MapLoader(self.map_file)
        rospy.loginfo("[waypoint_manager] %s", self.map.stats())

        self._remap_theta = 0.0
        self._remap_scale = 1.0
        self._remap_tx    = 0.0
        self._remap_ty    = 0.0
        self._load_remap_params()

        self.lock  = threading.Lock()
        self.pose  = None   # (x, y, yaw) in MAP frame
        self.path  = []     # list of node IDs
        self.idx   = 0      # current target node index
        self.state = self.IDLE

        self.status_pub   = rospy.Publisher(self.status_topic,
                                            String, queue_size=1, latch=True)
        self.path_pub     = rospy.Publisher("/waypoint_manager/path",
                                            Int32MultiArray, queue_size=1, latch=True)
        self.nav_info_pub = rospy.Publisher("/waypoint_manager/nav_info",
                                            Float64MultiArray, queue_size=1)

        rospy.Subscriber(self.odom_topic,    Odometry,          self._odom_cb,  queue_size=10)
        rospy.Subscriber(self.goal_topic,    Int32MultiArray,   self._goal_cb,  queue_size=1)
        rospy.Subscriber("/remap_transform", Float64MultiArray, self._remap_cb, queue_size=1)

        self.status_pub.publish(String(data=self.IDLE))
        self._publish_nav_info()

        rospy.loginfo("[waypoint_manager] ready. tol=%.2fm  rate=%.0fHz",
                      self.tol, self.rate_hz)

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
        mx, my  = self._odom_to_map(p.x, p.y)
        map_yaw = angle_diff(yaw, self._remap_theta)
        with self.lock:
            self.pose = (mx, my, map_yaw)

    def _goal_cb(self, msg):
        if len(msg.data) == 0:
            with self.lock:
                self.path  = []
                self.idx   = 0
                self.state = self.IDLE
            self.status_pub.publish(String(data=self.IDLE))
            self._publish_nav_info()
            rospy.loginfo("[waypoint_manager] navigation canceled")
            return

        if len(msg.data) < 2:
            rospy.logwarn("[waypoint_manager] malformed goal (need [start, end])")
            return

        start, end = int(msg.data[0]), int(msg.data[1])
        try:
            path = self.map.get_path(start, end)
        except Exception as e:
            rospy.logerr("[waypoint_manager] Dijkstra failed: %s", e)
            self.status_pub.publish(String(data=self.ERROR))
            return

        with self.lock:
            self.path  = path
            self.idx   = 0
            self.state = self.NAV

        m = Int32MultiArray()
        m.data = [int(x) for x in path]
        self.path_pub.publish(m)
        self.status_pub.publish(String(data=self.NAV))

        rospy.loginfo("[waypoint_manager] new path %d nodes (%.2fm)",
                      len(path), self.map.path_length(path))

    # ----------------------------------------------------------------- helpers

    def _publish_nav_info(self, node_id=-1, next_id=-1, dist=0.0,
                          is_junction=False, heading_to_next=0.0,
                          path_idx=0, path_len=0, active=False):
        msg = Float64MultiArray()
        msg.data = [
            float(node_id),
            float(next_id),
            float(dist),
            1.0 if is_junction else 0.0,
            float(heading_to_next),
            float(path_idx),
            float(path_len),
            1.0 if active else 0.0,
        ]
        self.nav_info_pub.publish(msg)

    def _advance_past_nodes(self, rx, ry, path, idx):
        """Advance idx past nodes the robot has already passed (dot-product test).

        The test asks "have we traveled THROUGH node[idx]?" measured along the
        direction we ENTER it (the incoming edge prev->cur), NOT the direction we
        leave it (cur->next). The outgoing edge skips a junction prematurely on a
        sharp turn: at node 6 the outgoing 6->30 edge points west, so the test is
        `(rx-3.9)*(-0.75) > 0` i.e. `rx < 3.9` — any small westward map error (the
        blue dot mapped a little left) satisfies it and the target jumps to node 30
        BEFORE the robot reaches node 6. is_junction then never latches, the turn-in
        never arms, and the robot drives straight through (the observed failure).
        The incoming edge is the robot's approach axis, so a lateral error no longer
        triggers an advance — the test fires only once the robot has actually moved
        through the node. Junction rotation is handled by the orchestrator (it creeps
        forward after aligning), so this still advances naturally afterwards.
        """
        while idx < len(path) - 1:
            nx, ny = self.map.node_xy(path[idx])
            if idx > 0:                       # incoming edge prev -> cur (approach axis)
                px, py = self.map.node_xy(path[idx - 1])
                dx, dy = nx - px, ny - py
            else:                             # first node: no predecessor -> cur -> next
                nx2, ny2 = self.map.node_xy(path[idx + 1])
                dx, dy = nx2 - nx, ny2 - ny
            if (rx - nx) * dx + (ry - ny) * dy > 0:
                idx += 1
            else:
                break
        return idx

    # --------------------------------------------------------------- main loop

    def _step(self):
        with self.lock:
            pose  = self.pose
            state = self.state
            path  = list(self.path)
            idx   = self.idx

        if pose is None or state in (self.IDLE, self.ERROR):
            self._publish_nav_info()
            return

        if state == self.DONE or not path or idx >= len(path):
            self._publish_nav_info()
            return

        # Dot-product advancement
        new_idx = self._advance_past_nodes(pose[0], pose[1], path, idx)
        if new_idx > idx:
            with self.lock:
                self.idx = new_idx
            rospy.loginfo("[waypoint_manager] passed WP(s) %s -> targeting %d",
                          path[idx:new_idx],
                          path[new_idx] if new_idx < len(path) else -1)
            idx = new_idx
            if idx >= len(path):
                with self.lock:
                    self.state = self.DONE
                self.status_pub.publish(String(data=self.DONE))
                self._publish_nav_info()
                rospy.loginfo("[waypoint_manager] GOAL reached (end of path).")
                return

        cur_id = path[idx]
        cur_xy = self.map.node_xy(cur_id)
        dist   = math.hypot(pose[0] - cur_xy[0], pose[1] - cur_xy[1])

        # Final waypoint stop: within tolerance OR having PASSED the goal along the
        # approach (incoming edge). The "passed" test makes arrival robust to a lateral
        # map offset: the goal (node 2) is a junction, so the through-road continues
        # straight past it; a slightly-offset robot never gets dist <= tol, so without
        # this it sails through and the lane follower drives it on down the road, never
        # stopping (observed). Passing the node's perpendicular plane = arrived.
        if idx == len(path) - 1:
            reached = dist <= self.tol
            if not reached and idx > 0:
                px, py = self.map.node_xy(path[idx - 1])
                nx, ny = cur_xy
                if (pose[0] - nx) * (nx - px) + (pose[1] - ny) * (ny - py) > 0:
                    reached = True
                    rospy.loginfo("[waypoint_manager] GOAL passed (dist=%.3fm > tol).", dist)
            if reached:
                with self.lock:
                    self.state = self.DONE
                self.status_pub.publish(String(data=self.DONE))
                self._publish_nav_info()
                rospy.loginfo("[waypoint_manager] GOAL reached (dist=%.3fm).", dist)
                return

        next_id = path[idx + 1] if idx + 1 < len(path) else -1
        is_junc = self.map.is_junction(cur_id)

        # Heading from robot to next node in MAP frame
        heading_to_next = 0.0
        if next_id >= 0:
            nx, ny = self.map.node_xy(next_id)
            heading_to_next = math.atan2(ny - pose[1], nx - pose[0])

        self._publish_nav_info(
            node_id=cur_id,
            next_id=next_id,
            dist=dist,
            is_junction=is_junc,
            heading_to_next=heading_to_next,
            path_idx=idx,
            path_len=len(path),
            active=True,
        )

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            try:
                self._step()
            except Exception as e:
                rospy.logerr_throttle(2.0, "[waypoint_manager] step err: %s", e)
            rate.sleep()


if __name__ == "__main__":
    try:
        WaypointManagerNode().run()
    except rospy.ROSInterruptException:
        pass