#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lane_controller_node.py — nodo ROS per il controllo laterale JetAuto.

Tutta la logica pura (Hough, fit, steering) è in lane_core.py.
Questo file aggiunge soltanto il wiring ROS: rosparam, pub/sub, cv_bridge.

Input:  /lane_mask_bev  (use_bev=true)  o  /lane_mask  (use_bev=false)
Output: /jetauto_controller/cmd_vel  (Twist)
        /lane_debug/image            (Image, se publish_debug=true)
        /lane_controller/state       (String)
"""

from __future__ import print_function
import math
import threading

import cv2
import numpy as np
import rospy

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from lane_core import LaneControllerCore, clamp


class LaneControllerV2Node(LaneControllerCore):

    def __init__(self):
        rospy.init_node("lane_controller_v2", anonymous=False)

        ns = "lane_controller/"

        def rp(key, default):
            return rospy.get_param(ns + key, default)

        # ── Costruisce il dict params e inizializza LaneControllerCore ────────
        # use_bev è letto prima per impostare il default del topic
        use_bev = bool(rp("use_bev", True))

        params = {
            "use_bev":                  use_bev,
            "bev_scale":                rp("bev_scale",                1.0),
            "hough_roi_top_frac":       rp("hough_roi_top_frac",       0.0),
            "hough_threshold":          rp("hough_threshold",          50),
            "hough_min_line_px":        rp("hough_min_line_px",        20),
            "hough_max_gap_px":         rp("hough_max_gap_px",         40),
            "hough_min_length_px":      rp("hough_min_length_px",      20),
            "min_valid_points":         rp("min_valid_points",          0),
            "line_height_ratio":        rp("line_height_ratio",        0.8),
            "lane_width_px":            rp("lane_width_px",          280.0),
            "center_y_ratio":           rp("center_y_ratio",          0.10),
            "max_steering_angle":       rp("max_steering_angle",      48.0),
            "single_line_offset":       rp("single_line_offset",       0.0),
            "min_distance_from_center": rp("min_distance_from_center", 0.0),
            "lane_fit_mode":            rp("lane_fit_mode",          "auto"),
            "angle_smooth_alpha_base":  rp("angle_smooth_alpha_base",  0.65),
            "angle_smooth_delta_scale": rp("angle_smooth_delta_scale", 8.0),
            "max_angular_z":            rp("max_angular_z",           0.80),
            "linear_x_speed":           rp("linear_x_speed",          0.05),
            "class_lane_marking":       rp("class_lane_marking",         2),
            "class_lane_dashed":        rp("class_lane_dashed",          3),
            "lane_width_bottom_frac":   rp("lane_width_bottom_frac",  0.55),
            "no_bev_roi_top_frac":      rp("no_bev_roi_top_frac",     0.45),
            "lane_width_dynamic_enable": rp("lane_width_dynamic_enable", True),
            "lane_width_ema_alpha":      rp("lane_width_ema_alpha",      0.10),
            "lane_width_min_px":         rp("lane_width_min_px",        180.0),
            "lane_width_max_px":         rp("lane_width_max_px",        380.0),
            "lane_width_sanity_band":    rp("lane_width_sanity_band",   0.25),
            "lane_width_reset_after":    rp("lane_width_reset_after",     0),
        }
        LaneControllerCore.__init__(self, params)

        # ── Parametri ROS-only ────────────────────────────────────────────────
        _default_mask = "/lane_mask" if not use_bev else "/lane_mask_bev"
        self.mask_topic   = rp("mask_topic",   _default_mask)
        self.cmd_topic    = rp("cmd_topic",    "/jetauto_controller/cmd_vel")
        self.debug_topic  = rp("debug_topic",  "/lane_debug/image")
        self.state_topic  = rp("state_topic",  "/lane_controller/state")
        self.enable_topic = rp("enable_topic", "/lane_controller/enable")

        self.drive_mode    = rp("drive_mode",    "classic")
        self.max_linear_y  = float(rp("max_linear_y",  0.08))
        self.publish_debug = bool(rp("publish_debug", True))
        self.debug_scale   = float(rp("debug_scale",  0.5))
        self.rate_hz       = float(rp("control_rate_hz", 10.0))

        # ── Stato ROS interno ─────────────────────────────────────────────────
        self.bridge        = CvBridge()
        self.lock          = threading.Lock()
        self.latest_mask   = None
        self.enabled       = False
        self.last_state    = "STOP"

        # ── Publishers / Subscribers ─────────────────────────────────────────
        self.cmd_pub   = rospy.Publisher(self.cmd_topic,   Twist,  queue_size=1)
        self.state_pub = rospy.Publisher(self.state_topic, String, queue_size=1, latch=True)
        if self.publish_debug:
            self.debug_pub = rospy.Publisher(self.debug_topic, Image, queue_size=1)

        rospy.Subscriber(self.mask_topic,   Image, self._mask_cb,   queue_size=1, buff_size=2**20)
        rospy.Subscriber(self.enable_topic, Bool,  self._enable_cb, queue_size=1)

        rospy.loginfo("[lane_ctrl_v2] avviato. mask=%s  drive=%s  max_steer=%.1f  "
                      "use_bev=%s  bev_scale=%.1f  roi_top=%.0f%%  rate=%.0fHz",
                      self.mask_topic, self.drive_mode, self.max_steer_angle,
                      self.use_bev, self.bev_scale,
                      (self.hough_roi_top_frac if self.use_bev
                       else self.no_bev_roi_top_frac) * 100,
                      self.rate_hz)

        if self.use_bev and self.lane_width_dynamic_enable:
            rospy.loginfo("[lane_ctrl_v2] dyn lane width: ENABLED  alpha=%.2f  "
                          "range=[%.0f,%.0f]px  band=%.0f%%  reset_after=%d",
                          self.lane_width_ema_alpha, self.lane_width_min_px,
                          self.lane_width_max_px, self.lane_width_sanity_band * 100,
                          self.lane_width_reset_after)
            if (self.lane_width_max_px < self.lane_width_px * 1.1
                    or self.lane_width_min_px > self.lane_width_px * 0.9):
                rospy.logwarn("[lane_ctrl_v2] lane_width_min/max_px non comprende "
                              "lane_width_px=%.0f. Hai cambiato bev_scale senza "
                              "scalare i bound?", self.lane_width_px)

    # ── Override logging ──────────────────────────────────────────────────────

    def _warn(self, msg):
        rospy.logwarn_throttle(2.0, msg)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _mask_cb(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, "[lane_ctrl_v2] mask bridge: %s" % e)
            return
        with self.lock:
            self.latest_mask = mask

    def _enable_cb(self, msg):
        self.enabled = bool(msg.data)
        rospy.loginfo("[lane_ctrl_v2] enable=%s", self.enabled)
        if not self.enabled:
            self.cmd_pub.publish(Twist())
            self._publish_state("DISABLED")

    def _publish_state(self, s):
        if s != self.last_state:
            self.state_pub.publish(String(data=s))
            self.last_state = s

    # ── Twist output ─────────────────────────────────────────────────────────

    def _steering_to_twist(self, steering_angle):
        norm  = steering_angle / self.max_steer_angle
        twist = Twist()
        twist.linear.x = self.linear_x
        if self.drive_mode == "mecanum":
            twist.linear.y = clamp(-norm * self.max_linear_y,
                                   -self.max_linear_y, self.max_linear_y)
        else:
            twist.angular.z = clamp(-norm * self.max_angular_z,
                                    -self.max_angular_z, self.max_angular_z)
        return twist

    # ── Debug image ───────────────────────────────────────────────────────────

    def _publish_debug_image(self, info, steering, twist):
        mask       = info["mask"]
        left_line  = info["left_line"]
        right_line = info["right_line"]
        valid_l    = info["valid_l"]
        valid_r    = info["valid_r"]
        lane_center = info["lane_center"]
        center_y   = info["center_y"]
        roi_top_px = info["roi_top_px"]
        dyn_w      = info.get("dyn_lane_width")
        meas_w     = info.get("measured_lane_width")
        w_acc      = info.get("width_meas_accepted", False)

        dst_h, dst_w = mask.shape[:2]
        out = self._make_colored_mask(mask)

        if valid_l and left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, "L", (max(x1 - 15, 0), y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if valid_r and right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, "R", (x1 + 5, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cx = dst_w // 2
        cv2.line(out, (cx, 0), (cx, dst_h), (0, 0, 255), 1)
        cv2.putText(out, "SX", (cx - 28, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(out, "DX", (cx + 5,  12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if roi_top_px > 0:
            cv2.line(out, (0, roi_top_px), (dst_w - 1, roi_top_px), (0, 165, 255), 1)
            cv2.putText(out, "ROI", (3, roi_top_px - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

        if center_y is not None:
            cv2.line(out, (0, center_y), (dst_w - 1, center_y), (255, 0, 255), 1)
        if lane_center is not None and center_y is not None:
            cv2.circle(out, (int(lane_center), center_y), 6, (255, 0, 255), -1)
            if dyn_w is not None:
                half = int(dyn_w / 2)
                x_l = int(lane_center) - half
                x_r = int(lane_center) + half
                cv2.line(out, (x_l, center_y - 6), (x_l, center_y + 6), (0, 200, 255), 1)
                cv2.line(out, (x_r, center_y - 6), (x_r, center_y + 6), (0, 200, 255), 1)

        arr_cx = dst_w // 2
        arr_by = dst_h - 8
        angle_rad = math.radians(steering)
        arr_tx = int(arr_cx + 40 * math.sin(angle_rad))
        arr_ty = int(arr_by - 40 * math.cos(angle_rad))
        arr_color = ((0, 255, 0) if abs(steering) < 15
                     else (0, 255, 255) if abs(steering) < 30
                     else (0, 0, 255))
        cv2.arrowedLine(out, (arr_cx, arr_by), (arr_tx, arr_ty),
                        arr_color, 2, tipLength=0.3)
        cv2.putText(out, "%.1fdeg %s" % (steering, self.last_state),
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, arr_color, 1, cv2.LINE_AA)
        cv2.putText(out, "vx=%.2f vy=%.2f wz=%.2f" % (
                        twist.linear.x, twist.linear.y, twist.angular.z),
                    (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        if self.no_lane_count > 0:
            cv2.putText(out, "hold %d" % self.no_lane_count,
                        (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)

        dyn_str  = ("W=%.0f"  % dyn_w)  if dyn_w  is not None else "W=--"
        meas_str = ("Wm=%.0f" % meas_w) if meas_w is not None else "Wm=--"
        meas_col = (0, 255, 0) if w_acc else (0, 0, 255)
        cv2.putText(out, dyn_str,  (5, dst_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(out, meas_str, (5, dst_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, meas_col, 1, cv2.LINE_AA)

        if self.debug_scale != 1.0:
            out = cv2.resize(out, None, fx=self.debug_scale, fy=self.debug_scale,
                             interpolation=cv2.INTER_AREA)
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(out, encoding="bgr8"))
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, "[lane_ctrl_v2] debug bridge: %s" % e)

    # ── Step principale ───────────────────────────────────────────────────────

    def _step(self):
        with self.lock:
            mask = None if self.latest_mask is None else self.latest_mask.copy()

        if not self.enabled:
            self._publish_state("DISABLED")
            return

        if mask is None:
            self._publish_state("STOP")
            return

        steering, _, state, info = self.step(mask)

        self._publish_state(state)
        twist = self._steering_to_twist(steering)
        self.cmd_pub.publish(twist)

        if self.publish_debug:
            self._publish_debug_image(info, steering, twist)

    # ── Spin ──────────────────────────────────────────────────────────────────

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            try:
                self._step()
            except Exception as e:
                rospy.logerr_throttle(2.0, "[lane_ctrl_v2] step err: %s" % e)
            rate.sleep()
        self.cmd_pub.publish(Twist())


# =============================================================================
if __name__ == "__main__":
    try:
        LaneControllerV2Node().run()
    except rospy.ROSInterruptException:
        pass
