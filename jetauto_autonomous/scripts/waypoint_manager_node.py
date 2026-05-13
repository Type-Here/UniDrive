#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
waypoint_manager_node.py
------------------------
Carica la mappa, riceve goal (start, end), calcola percorso con Dijkstra,
segue i waypoint usando l'odometria. Tra waypoint cede il controllo laterale
al lane_controller (pubblicando True su /lane_controller/enable). Negli
incroci (degree>2) prende lui il comando per ruotare verso il waypoint
successivo del percorso.

Topic:
  IN  - /odom                        (nav_msgs/Odometry)
  IN  - /waypoint_manager/goal       (std_msgs/Int32MultiArray) [start, end]
  OUT - /waypoint_manager/status     (std_msgs/String)
  OUT - /jetauto_controller/cmd_vel  (geometry_msgs/Twist)  [solo durante manovre incrocio]
  OUT - /lane_controller/enable      (std_msgs/Bool)
  OUT - /waypoint_manager/path       (std_msgs/Int32MultiArray) percorso corrente (per dashboard)
"""

from __future__ import print_function
import math
import threading

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String, Int32MultiArray

from map_loader import MapLoader, NODE_JUNCTION


def yaw_from_quat(q):
    # Z-Y-X yaw da quaternione
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    """Differenza angolare in [-pi, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


class WaypointManagerNode(object):

    # Stati FSM
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

        # Carica mappa
        rospy.loginfo("[waypoint_manager] carico mappa: %s", self.map_file)
        self.map = MapLoader(self.map_file)
        rospy.loginfo("[waypoint_manager] %s", self.map.stats())

        # Stato
        self.lock = threading.Lock()
        self.pose = None      # (x, y, yaw)
        self.path = []        # lista di node_id
        self.idx  = 0         # indice waypoint corrente nel path
        self.state = self.IDLE

        # Pub/Sub
        self.cmd_pub    = rospy.Publisher(self.cmd_topic,    Twist, queue_size=1)
        self.enable_pub = rospy.Publisher(self.enable_topic, Bool,  queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.path_pub   = rospy.Publisher("/waypoint_manager/path",
                                          Int32MultiArray, queue_size=1, latch=True)

        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=10)
        rospy.Subscriber(self.goal_topic, Int32MultiArray, self._goal_cb, queue_size=1)

        self._publish_status(self.IDLE, "")
        self._set_lane_enabled(False)
        rospy.loginfo("[waypoint_manager] pronto.")

    # ------------------------------------------------------------- callbacks
    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        with self.lock:
            self.pose = (p.x, p.y, yaw)

    def _goal_cb(self, msg):
        if len(msg.data) == 0:
            # Goal vuoto = cancella navigazione corrente
            with self.lock:
                self.path = []
                self.idx  = 0
            self._publish_status("IDLE")
            # Ferma il robot
            stop_twist = Twist()
            self.cmd_pub.publish(stop_twist)
            rospy.loginfo("[waypoint_manager] navigazione annullata dalla dashboard")
            return
        if len(msg.data) < 2:
            rospy.logwarn("[waypoint_manager] goal malformato (serve [start, end])")
            return
        start, end = int(msg.data[0]), int(msg.data[1])
        try:
            path = self.map.get_path(start, end)
        except Exception as e:
            rospy.logerr("[waypoint_manager] Dijkstra fallito: %s", e)
            self._publish_status(self.ERROR, str(e))
            return

        with self.lock:
            self.path = path
            self.idx  = 0
            self.state = self.NAV

        # Pubblica il path per la dashboard
        m = Int32MultiArray()
        m.data = [int(x) for x in path]
        self.path_pub.publish(m)

        self._set_lane_enabled(True)
        self._publish_status(self.NAV,
                             "path=%s len=%.2fm" %
                             (path, self.map.path_length(path)))
        rospy.loginfo("[waypoint_manager] nuovo percorso %d nodi (%.2f m)",
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
        Quando il robot è dentro junction_radius da un nodo junction,
        prende il controllo: spegne il lane_controller, ruota verso il
        waypoint SUCCESSIVO del path (deciso da Dijkstra), poi riabilita
        il lane controller.
        """
        with self.lock:
            cur_idx = self.idx
            path = list(self.path)

        if cur_idx + 1 >= len(path):
            return  # niente prossimo waypoint, gestito dal loop principale

        next_id = path[cur_idx + 1]
        target_yaw = self._heading_to(next_id, pose)
        err = angle_diff(target_yaw, pose[2])

        twist = Twist()
        if abs(err) < math.radians(8.0):
            # Allineato: avanza piano e cedi
            twist.linear.x = self.junction_speed
            self.cmd_pub.publish(twist)
            self._set_lane_enabled(True)
            with self.lock:
                self.state = self.NAV
            self._publish_status(self.NAV, "exit junction -> %d" % next_id)
            return

        # Ruota in posto (con leggero avanzamento)
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

        # Waypoint raggiunto?
        if dist <= self.tol:
            new_idx = idx + 1
            if new_idx >= len(path):
                # Fine percorso
                self._stop_robot()
                self._set_lane_enabled(False)
                with self.lock:
                    self.state = self.DONE
                self._publish_status(self.DONE, "last=%d" % cur_id)
                rospy.loginfo("[waypoint_manager] GOAL raggiunto.")
                return
            with self.lock:
                self.idx = new_idx
            rospy.loginfo("[waypoint_manager] WP %d (%s) raggiunto, next=%d",
                          cur_id, self.map.node_type(cur_id), path[new_idx])
            return

        # Siamo vicini a un incrocio? -> manovra
        # (controllo sul waypoint corrente: se è junction e siamo entro junction_radius)
        if self.map.is_junction(cur_id) and dist <= self.junction_radius and idx + 1 < len(path):
            with self.lock:
                self.state = self.JUNCTION
            self._set_lane_enabled(False)
            self._handle_junction(pose)
            return

        # Navigazione normale: lane_controller pilota, qui solo monitoraggio
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
