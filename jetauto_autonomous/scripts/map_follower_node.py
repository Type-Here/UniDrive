#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
map_follower_node.py  --  Map-assisted fallback driving for JetAuto.

Automatically activates when the lane controller stays in HOLD state for
hold_fallback_frames consecutive ticks, meaning no lane markings are
visible (roundabout, bare crossroads, etc.). Uses a pure-pursuit
controller on the planned waypoint path to keep the robot on track until
lane markings reappear and the lane controller recovers.

State machine
-------------
    INACTIVE
        lane_state == HOLD  for N ticks
        AND nav == NAVIGATING  AND NOT JUNCTION
          ↓
    ACTIVATING  (publish enable=False to silence lane controller).
          ↓
    ACTIVE  (publish cmd_vel via pure pursuit)
        lane recovers (non-HOLD for M ticks)  OR  nav leaves NAVIGATING.
          ↓
    RECOVERING  (publish enable=True, hand back to lane controller).
          ↓
    INACTIVE

Arbitration note
----------------
This node uses the same /lane_controller/enable arbitration as
waypoint_manager_node.py (it disables the lane controller while active and
re-enables on recovery).  It stays passive (INACTIVE) whenever the
waypoint manager is in JUNCTION or IDLE state to avoid conflicts.

ROS interface
-------------
Subscribed:
    /lane_controller/state       std_msgs/String
    /waypoint_manager/status     std_msgs/String
    /waypoint_manager/path       std_msgs/Int32MultiArray
    /odom                        nav_msgs/Odometry

Published:
    /jetauto_controller/cmd_vel  geometry_msgs/Twist   (only while ACTIVE)
    /lane_controller/enable      std_msgs/Bool         (latched)
    /map_follower/active         std_msgs/Bool         (latched)
    /map_follower/state          std_msgs/String       (latched)

Parameters (rosparam, namespace map_follower/)
----------------------------------------------
    hold_fallback_frames  int    15      HOLD ticks before activation
    lane_recovery_frames  int    5       non-HOLD ticks before deactivation
    lookahead_m           float  0.50    pure-pursuit lookahead distance (m)
    map_drive_speed       float  0.04    linear.x while in map mode (m/s)
    angular_kp            float  1.2     P-gain: heading error -> angular.z
    max_angular_z         float  0.80    clamp (rad/s)
    rate_hz               float  10.0    control loop frequency
    odom_origin_x         float  0.0     raw odom x that corresponds to map origin
    odom_origin_y         float  0.0     raw odom y that corresponds to map origin
    map_file              str    (inherits from waypoint_manager/map_file)

Usage
-----
    # start_all.sh already loads lane_params.yaml into rosparam;
    # add map_follower_node.py to the startup script alongside the other nodes.
    python2 map_follower_node.py

Merging into main code (future)
--------------------------------
    MapFollowerCore has no ROS imports -- move it into waypoint_manager_node.py
    (or a shared module) and wire the step() call into the existing spin loop.
    The MapFollowerNode class can then be deleted.
"""

from __future__ import print_function

import math
import sys
import threading

import rospy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Int32MultiArray, String

# map_loader.py lives in the same directory
sys.path.insert(0, rospy.get_param(
    "scripts_dir",
    "/home/jetauto/jetauto_autonomous/scripts"))
from map_loader import MapLoader  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaw_from_quat(q):
    """Extract the yaw (rotation about Z) from a ROS quaternion message.

    A quaternion q = (x, y, z, w) encodes a 3-D rotation.  For a robot
    moving on a flat plane we only need the heading angle (yaw = rotation
    about the vertical Z axis).

    Using the ZYX Euler-angle convention the yaw is:

        yaw = atan2( 2*(w*z + x*y),  1 - 2*(y² + z²) )

    The two arguments of atan2 are the sine and cosine of yaw scaled by
    the same factor, so their ratio gives tan(yaw) and atan2 recovers the
    full-circle angle in [-π, π].

    Args:
        q: geometry_msgs/Quaternion (fields .x .y .z .w)

    Returns:
        float: yaw angle in radians, range [-pi, pi]
    """
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _angle_diff(a, b):
    """Compute the signed shortest angular difference (a - b), wrapped to [-pi, pi].

    Naively subtracting angles can give values outside [-π, π] (e.g. 350°
    minus 10° gives 340° instead of the geometrically equivalent -20°).
    This function always returns the shortest-arc difference, which is what
    a proportional heading controller needs to avoid spinning the wrong way.

    Args:
        a (float): first angle in radians
        b (float): second angle in radians

    Returns:
        float: (a - b) normalized to (-pi, pi]
    """
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def _clamp(v, lo, hi):
    """Clamp value v to the closed interval [lo, hi]."""
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Pure-Python core (no ROS imports -- easy to unit-test and merge elsewhere)
# ---------------------------------------------------------------------------

class MapFollowerCore(object):
    """Pure-pursuit controller that follows a list of map waypoints.

    Pure pursuit is a classic path-tracking algorithm from robotics:
    instead of trying to minimize the cross-track error at the robot's
    current position (which produces oscillation), it picks a "carrot"
    point that is a fixed look-ahead distance L ahead on the path, and
    steers toward that point.

    The steering command is:

        heading_to_carrot = atan2(carrot_y - robot_y, carrot_x - robot_x)
        heading_error     = heading_to_carrot - robot_yaw   (wrapped to [-π,π])
        angular_z         = Kp * heading_error              (clamped to ±max_w)

    A larger L gives smoother (but slower-reacting) tracking; a smaller L
    gives tighter tracking but more oscillation.  Typical choice: L equals
    1–3 times the robot's wheelbase or roughly the distance covered in 0.5 s
    at operating speed.

    Callers feed pose and path updates via set_pose() / set_path(), then
    call step() at a fixed rate.  step() returns what to publish without
    doing any I/O itself.
    """

    INACTIVE   = "INACTIVE"
    ACTIVATING = "ACTIVATING"
    ACTIVE     = "ACTIVE"
    RECOVERING = "RECOVERING"

    # Lane-controller states that count as "no lane"
    _NO_LANE_STATES = frozenset(("HOLD", "STOP"))

    def __init__(self, params, map_loader):
        p = params
        self.hold_fallback_frames = int(  p.get("hold_fallback_frames", 15))
        self.lane_recovery_frames = int(  p.get("lane_recovery_frames",  5))
        self.lookahead_m          = float(p.get("lookahead_m",          0.50))
        self.map_drive_speed      = float(p.get("map_drive_speed",      0.04))
        self.angular_kp           = float(p.get("angular_kp",           1.2))
        self.max_angular_z        = float(p.get("max_angular_z",        0.80))

        self._map    = map_loader
        self._path   = []     # list of node IDs in path order
        self._pose   = None   # (x, y, yaw) in odom frame, corrected for origin

        self._state          = self.INACTIVE
        self._hold_count     = 0
        self._recovery_count = 0

    # -- External inputs ------------------------------------------------------

    def set_path(self, path):
        """Update the planned path (list of integer node IDs)."""
        self._path = list(path)

    def set_pose(self, x, y, yaw):
        """Update the robot's current pose (already offset-corrected)."""
        self._pose = (x, y, yaw)

    # -- Main step ------------------------------------------------------------

    def step(self, lane_state, nav_status):
        """
        Advance the state machine by one tick.

        Parameters
        ----------
        lane_state : str   Latest value from /lane_controller/state
        nav_status : str   Latest value from /waypoint_manager/status

        Returns
        -------
        (state_changed, new_state, twist_or_none, enable_lane_or_none)

        twist_or_none       : Twist to publish to cmd_vel, or None
        enable_lane_or_none : bool to publish to /lane_controller/enable, or None
        """
        is_no_lane    = lane_state in self._NO_LANE_STATES
        is_navigating = nav_status.startswith("NAVIGATING")
        is_junction   = nav_status.startswith("JUNCTION")

        prev_state = self._state
        twist      = None
        enable_cmd = None

        if self._state == self.INACTIVE:
            if is_no_lane and is_navigating and not is_junction:
                self._hold_count += 1
                if self._hold_count >= self.hold_fallback_frames:
                    self._state      = self.ACTIVATING
                    self._hold_count = 0
                    self._recovery_count = 0
            else:
                self._hold_count = 0

        elif self._state == self.ACTIVATING:
            enable_cmd  = False          # silence the lane controller
            self._state = self.ACTIVE

        elif self._state == self.ACTIVE:
            # Hand back if waypoint manager reclaims control or path ends
            if not is_navigating or is_junction:
                self._state          = self.RECOVERING
                self._recovery_count = 0
            elif not self._path:
                # No known path: can't follow the map, recover safely
                self._state          = self.RECOVERING
                self._recovery_count = 0
            elif not is_no_lane:
                # Lane markings visible again
                self._recovery_count += 1
                if self._recovery_count >= self.lane_recovery_frames:
                    self._state          = self.RECOVERING
                    self._recovery_count = 0
            else:
                self._recovery_count = 0
                twist = self._pure_pursuit()

        elif self._state == self.RECOVERING:
            enable_cmd  = True           # hand control back to lane controller
            self._state = self.INACTIVE
            self._hold_count = 0

        changed = (self._state != prev_state)
        return changed, self._state, twist, enable_cmd

    # -- Pure pursuit ---------------------------------------------------------

    def _pure_pursuit(self):
        """Compute a Twist command using the pure-pursuit algorithm.

        Steps:
          1. Find the carrot point L metres ahead on the path via
             _find_lookahead().
          2. Compute the direction from the robot to the carrot:
                heading = atan2(cy - ry, cx - rx)
          3. Compute heading error (shortest-arc difference so the robot
             always turns in the geometrically correct direction):
                err = heading - robot_yaw    wrapped to [-π, π]
          4. Apply a P-controller:
                angular_z = Kp * err         clamped to ±max_angular_z
          5. Set a constant forward speed (map_drive_speed).

        The linear and angular velocities are packed into a ROS Twist and
        returned to the caller for publishing.

        Returns:
            geometry_msgs/Twist: velocity command, or a zero Twist (stop)
                if the robot pose or path are not yet available.
        """
        if self._pose is None or len(self._path) < 2:
            return Twist()

        rx, ry, ryaw = self._pose
        carrot = self._find_lookahead(rx, ry)
        if carrot is None:
            return Twist()

        cx, cy = carrot
        heading = math.atan2(cy - ry, cx - rx)
        err     = _angle_diff(heading, ryaw)

        twist           = Twist()
        twist.linear.x  = self.map_drive_speed
        twist.angular.z = _clamp(
            self.angular_kp * err,
            -self.max_angular_z,
            self.max_angular_z)
        return twist

    def _find_lookahead(self, rx, ry):
        """Find the carrot point exactly lookahead_m ahead of the robot on the path.

        Algorithm
        ---------
        1. **Nearest-node search** — iterate all nodes in the planned path
           and find the index ``best_idx`` with minimum Euclidean distance
           to the robot:
               d = sqrt((rx - nx)² + (ry - ny)²)

        2. **Overshoot correction (dot-product test)** — if the robot has
           already moved past ``best_idx`` toward the next node, the naive
           closest node would be behind the robot and we'd steer backward.
           We detect this by projecting the robot-to-node vector onto the
           node-to-next-node direction:
               dot = (rx - ax)*(bx - ax) + (ry - ay)*(by - ay)
           A positive dot product means the robot is on the "far side" of
           ``best_idx`` relative to the next node, so we advance by 1.

        3. **Lookahead walk** — starting from ``best_idx``, accumulate
           segment lengths along the path until the total exceeds
           ``lookahead_m``.  On the segment that crosses the threshold,
           linearly interpolate to land exactly at the right distance:
               frac    = (lookahead_m - accumulated_so_far) / segment_length
               carrot  = prev_point + frac * (next_point - prev_point)

           If the path ends before ``lookahead_m`` is reached, the last
           node is returned as the carrot (robot just drives to the goal).

        Args:
            rx (float): robot x position in map frame (metres)
            ry (float): robot y position in map frame (metres)

        Returns:
            tuple(float, float) or None: (x, y) carrot point, or None if
                the path is empty.
        """
        if not self._path:
            return None

        # 1. Closest node
        best_idx  = 0
        best_dist = float("inf")
        for i, nid in enumerate(self._path):
            nx, ny = self._map.node_xy(nid)
            d = math.hypot(rx - nx, ry - ny)
            if d < best_dist:
                best_dist = d
                best_idx  = i

        # 2. Advance if the robot has passed best_idx toward best_idx+1
        if best_idx < len(self._path) - 1:
            ax, ay = self._map.node_xy(self._path[best_idx])
            bx, by = self._map.node_xy(self._path[best_idx + 1])
            # Vector from node best_idx to robot, and to next node
            dot = (rx - ax) * (bx - ax) + (ry - ay) * (by - ay)
            if dot > 0:
                best_idx += 1

        # 3. Walk lookahead_m forward
        accum    = 0.0
        prev_x, prev_y = self._map.node_xy(self._path[best_idx])

        for i in range(best_idx + 1, len(self._path)):
            nx, ny = self._map.node_xy(self._path[i])
            seg = math.hypot(nx - prev_x, ny - prev_y)

            if accum + seg >= self.lookahead_m:
                # Interpolate along this segment to hit exactly lookahead_m
                remaining = self.lookahead_m - accum
                frac      = remaining / max(seg, 1e-9)
                return (prev_x + frac * (nx - prev_x),
                        prev_y + frac * (ny - prev_y))

            accum  += seg
            prev_x, prev_y = nx, ny

        # Path ended before lookahead_m: aim for the last node
        last = self._path[-1]
        return self._map.node_xy(last)

    @property
    def state(self):
        return self._state


# ---------------------------------------------------------------------------
# ROS node wrapper
# ---------------------------------------------------------------------------

class MapFollowerNode(object):

    def __init__(self):
        rospy.init_node("map_follower", anonymous=False)

        ns = "map_follower/"
        def rp(key, default):
            return rospy.get_param(ns + key, default)

        # Map (shares waypoint_manager/map_file if not overridden)
        map_file = rp("map_file",
                      rospy.get_param("waypoint_manager/map_file", ""))
        if not map_file:
            rospy.logfatal("[map_follower] map_file not set. "
                           "Configure map_follower/map_file or "
                           "waypoint_manager/map_file.")
            raise SystemExit(1)

        rospy.loginfo("[map_follower] Loading map: %s", map_file)
        ml = MapLoader(map_file)
        rospy.loginfo("[map_follower] Map loaded: %s", ml.stats())

        params = {
            "hold_fallback_frames": rp("hold_fallback_frames", 15),
            "lane_recovery_frames": rp("lane_recovery_frames",  5),
            "lookahead_m":          rp("lookahead_m",          0.50),
            "map_drive_speed":      rp("map_drive_speed",      0.04),
            "angular_kp":           rp("angular_kp",           1.2),
            "max_angular_z":        rp("max_angular_z",        0.80),
        }
        self._core = MapFollowerCore(params, ml)
        self._lock = threading.Lock()

        # Odom-to-map origin offset (set when user clicks Calibra in dashboard)
        # Defaults to 0,0 -- works when robot boots at odom origin = map node 0.
        self._odom_origin_x = float(rp("odom_origin_x", 0.0))
        self._odom_origin_y = float(rp("odom_origin_y", 0.0))

        # Topic names (overrideable via rosparam)
        cmd_topic         = rp("cmd_topic",         "/jetauto_controller/cmd_vel")
        lane_enable_topic = rp("lane_enable_topic", "/lane_controller/enable")
        active_topic      = rp("active_topic",      "/map_follower/active")
        state_topic       = rp("state_topic",       "/map_follower/state")
        odom_topic         = rp("odom_topic",          "/odom")
        path_topic         = rp("path_topic",          "/waypoint_manager/path")
        lane_state_topic   = rp("lane_state_topic",    "/lane_controller/state")
        nav_status_topic   = rp("nav_status_topic",    "/waypoint_manager/status")
        odom_origin_topic  = rp("odom_origin_topic",   "/map_follower/odom_origin")

        # Publishers
        self._cmd_pub    = rospy.Publisher(cmd_topic,         Twist,  queue_size=1)
        self._enable_pub = rospy.Publisher(lane_enable_topic, Bool,   queue_size=1, latch=True)
        self._active_pub = rospy.Publisher(active_topic,      Bool,   queue_size=1, latch=True)
        self._state_pub  = rospy.Publisher(state_topic,       String, queue_size=1, latch=True)

        # Internal state (updated by callbacks, consumed by _step)
        self._lane_state = "STOP"
        self._nav_status = "IDLE"
        self._last_published_state = MapFollowerCore.INACTIVE

        # Subscribers
        rospy.Subscriber(lane_state_topic,  String,
                         self._lane_cb,        queue_size=1)
        rospy.Subscriber(nav_status_topic,  String,
                         self._nav_cb,         queue_size=1)
        rospy.Subscriber(path_topic,        Int32MultiArray,
                         self._path_cb,        queue_size=1)
        rospy.Subscriber(odom_topic,        Odometry,
                         self._odom_cb,        queue_size=5)
        rospy.Subscriber(odom_origin_topic, Point,
                         self._odom_origin_cb, queue_size=1)

        self._rate_hz = float(rp("rate_hz", 10.0))

        self._publish_state(MapFollowerCore.INACTIVE)
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "[map_follower] Ready. "
            "lookahead=%.2f m  speed=%.2f m/s  "
            "fallback=%d ticks  recovery=%d ticks  rate=%.0f Hz",
            params["lookahead_m"],
            params["map_drive_speed"],
            params["hold_fallback_frames"],
            params["lane_recovery_frames"],
            self._rate_hz)

    # -- Callbacks -------------------------------------------------------------

    def _odom_origin_cb(self, msg):
        """Live-update the odom-to-map origin offset published by the dashboard Calibra button."""
        with self._lock:
            self._odom_origin_x = msg.x
            self._odom_origin_y = msg.y
        rospy.loginfo("[map_follower] odom_origin updated: (%.3f, %.3f)", msg.x, msg.y)

    def _lane_cb(self, msg):
        with self._lock:
            self._lane_state = msg.data

    def _nav_cb(self, msg):
        with self._lock:
            self._nav_status = msg.data

    def _path_cb(self, msg):
        with self._lock:
            self._core.set_path(list(msg.data))

    def _odom_cb(self, msg):
        p   = msg.pose.pose.position
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        with self._lock:
            # Subtract odom origin so poses are in the same frame as map coords
            self._core.set_pose(
                p.x - self._odom_origin_x,
                p.y - self._odom_origin_y,
                yaw)

    # -- Main loop -------------------------------------------------------------

    def _step(self):
        with self._lock:
            lane_state = self._lane_state
            nav_status = self._nav_status

        changed, new_state, twist, enable_cmd = self._core.step(
            lane_state, nav_status)

        if enable_cmd is not None:
            self._enable_pub.publish(Bool(data=bool(enable_cmd)))
            rospy.loginfo("[map_follower] lane_controller/enable -> %s", enable_cmd)

        if twist is not None:
            self._cmd_pub.publish(twist)

        if changed:
            self._publish_state(new_state)
            rospy.loginfo(
                "[map_follower] %s -> %s  "
                "(lane_state=%s  nav=%s)",
                self._last_published_state, new_state,
                lane_state, nav_status.split("|")[0].strip())
            self._last_published_state = new_state

    def _publish_state(self, state):
        self._state_pub.publish(String(data=state))
        self._active_pub.publish(Bool(data=(state == MapFollowerCore.ACTIVE)))

    def _on_shutdown(self):
        # Safety: always re-enable the lane controller and stop the robot.
        self._enable_pub.publish(Bool(data=True))
        self._cmd_pub.publish(Twist())
        rospy.loginfo("[map_follower] shutdown: lane_controller re-enabled, robot stopped.")

    def run(self):
        rate = rospy.Rate(self._rate_hz)
        while not rospy.is_shutdown():
            try:
                self._step()
            except Exception as e:
                rospy.logerr_throttle(2.0, "[map_follower] step error: %s", e)
            rate.sleep()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        MapFollowerNode().run()
    except rospy.ROSInterruptException:
        pass
