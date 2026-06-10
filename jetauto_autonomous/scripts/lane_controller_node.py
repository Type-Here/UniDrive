#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lane_controller_node.py - ROS node for JetAuto lateral control.

All pure logic (Hough, fit, steering) is in lane_core.py.
This file only adds ROS wiring: rosparam, pub/sub.

Input:  /lane_mask_bev  (BEV-warped class mask, mono8)
Output: /lane_controller/cmd_vel  (Twist, proposed — forwarded to hardware by orchestrator)
        /lane_debug/image         (Image, if publish_debug=true)
        /lane_controller/state    (String)
"""

from __future__ import print_function
import math
import threading

import cv2
import numpy as np
import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64MultiArray, String

from lane_core import LaneControllerCore, clamp


class LaneControllerV2Node(LaneControllerCore):

    # state string -> numeric code for the additive /lane_controller/info topic
    _STATE_CODE = {
        "STOP": 0.0, "HOLD": 1.0, "TRACKING_CC": 2.0,
        "SINGLE_L": 3.0, "SINGLE_R": 4.0, "DISABLED": 5.0,
    }

    def __init__(self):
        rospy.init_node("lane_controller_v2", anonymous=False)

        ns = "lane_controller/"

        def rp(key, default):
            return rospy.get_param(ns + key, default)

        # -- Build params dict and initialise LaneControllerCore -------------
        params = {
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
            "lane_width_dynamic_enable": rp("lane_width_dynamic_enable", True),
            "lane_width_ema_alpha":      rp("lane_width_ema_alpha",      0.10),
            "lane_width_min_px":         rp("lane_width_min_px",        180.0),
            "lane_width_max_px":         rp("lane_width_max_px",        380.0),
            "lane_width_sanity_band":    rp("lane_width_sanity_band",   0.25),
            "lane_width_reset_after":    rp("lane_width_reset_after",     0),
        }
        LaneControllerCore.__init__(self, params)

        # -- ROS-only parameters -----------------------------------------------
        self.mask_topic   = rp("mask_topic",   "/lane_mask_bev")
        self.cmd_topic    = rp("cmd_topic",    "/lane_controller/cmd_vel")
        self.debug_topic  = rp("debug_topic",  "/lane_debug/image")
        self.state_topic  = rp("state_topic",  "/lane_controller/state")
        self.info_topic   = rp("info_topic",   "/lane_controller/info")
        self.enable_topic = rp("enable_topic", "/lane_controller/enable")

        self.drive_mode    = rp("drive_mode",    "classic")
        self.max_linear_y  = float(rp("max_linear_y",  0.08))
        self.publish_debug = bool(rp("publish_debug", True))
        self.debug_scale   = float(rp("debug_scale",  0.5))
        self.rate_hz       = float(rp("control_rate_hz", 10.0))
        self.enable_log    = bool(rp("enable_log", True))

        # -- Internal ROS state ------------------------------------------------
        self.lock          = threading.Lock()
        self.latest_mask   = None
        self.enabled       = False
        self.last_state    = "STOP"

        # -- Publishers / Subscribers -----------------------------------------
        self.cmd_pub   = rospy.Publisher(self.cmd_topic,   Twist,  queue_size=1)
        self.state_pub = rospy.Publisher(self.state_topic, String, queue_size=1, latch=True)
        # Additive geometry topic for the orchestrator's roundabout guardrail.
        # Purely informational — the cmd_vel/state path above is unchanged.
        self.info_pub  = rospy.Publisher(self.info_topic,  Float64MultiArray, queue_size=1)
        if self.publish_debug:
            self.debug_pub = rospy.Publisher(self.debug_topic, Image, queue_size=1)

        rospy.Subscriber(self.mask_topic,   Image, self._mask_cb,   queue_size=1, buff_size=2**20)
        rospy.Subscriber(self.enable_topic, Bool,  self._enable_cb, queue_size=1)

        if self.enable_log:
            rospy.loginfo("[lane_ctrl_v2] started. mask=%s  drive=%s  max_steer=%.1f  "
                          "bev_scale=%.1f  roi_top=%.0f%%  rate=%.0fHz",
                          self.mask_topic, self.drive_mode, self.max_steer_angle,
                          self.bev_scale, self.hough_roi_top_frac * 100,
                          self.rate_hz)

        if self.lane_width_dynamic_enable:
            if self.enable_log:
                rospy.loginfo("[lane_ctrl_v2] dyn lane width: ENABLED  alpha=%.2f  "
                              "range=[%.0f,%.0f]px  band=%.0f%%  reset_after=%d",
                              self.lane_width_ema_alpha, self.lane_width_min_px,
                              self.lane_width_max_px, self.lane_width_sanity_band * 100,
                              self.lane_width_reset_after)
            if (self.lane_width_max_px < self.lane_width_px * 1.1
                    or self.lane_width_min_px > self.lane_width_px * 0.9):
                if self.enable_log:
                    rospy.logwarn("[lane_ctrl_v2] lane_width_min/max_px does not include "
                                  "lane_width_px=%.0f. Did you change bev_scale without "
                                  "scaling the bounds?", self.lane_width_px)

    # -- Override logging ------------------------------------------------------

    def _warn(self, msg):
        if self.enable_log:
            rospy.logwarn_throttle(2.0, msg)

    # -- Callbacks -------------------------------------------------------------

    def _mask_cb(self, msg):
        try:
            mask = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
            mask = mask.copy()  # np.frombuffer returns a read-only buffer
        except Exception as e:
            if self.enable_log:
                rospy.logwarn_throttle(5.0, "[lane_ctrl_v2] mask decode: %s" % e)
            return
        with self.lock:
            is_first = self.latest_mask is None
            self.latest_mask = mask
        if is_first and self.enable_log:
            rospy.loginfo("[lane_ctrl_v2] first mask received: shape=%s", mask.shape)

    def _enable_cb(self, msg):
        self.enabled = bool(msg.data)
        if self.enable_log:
            rospy.loginfo("[lane_ctrl_v2] enable=%s", self.enabled)
        if not self.enabled:
            self.cmd_pub.publish(Twist())
            self._publish_state("DISABLED")

    def _publish_state(self, s):
        if s != self.last_state:
            self.state_pub.publish(String(data=s))
            self.last_state = s

    # -- Additive lane-geometry info (orchestrator roundabout guardrail) --------

    @staticmethod
    def _x_at_y(line, y):
        """x of a fitted line [x_bot,y_bot,x_top,y_top] at row y (linear interp)."""
        if line is None:
            return None
        x1, y1, x2, y2 = line
        if y2 == y1:
            return float(x1)
        return x1 + float(y - y1) / float(y2 - y1) * (x2 - x1)

    @staticmethod
    def _lane_heading(left_line, right_line):
        """Lane forward direction vs robot straight-ahead (rad). +ve bends right.

        Averaged over whichever lines are present; 0.0 if none. This is the REAL
        lane direction the orchestrator can use instead of the angular.z proxy.
        """
        def hdg(line):
            if line is None:
                return None
            x1, y1, x2, y2 = line
            # forward = from the near (larger y) to the far (smaller y) endpoint
            if y1 >= y2:
                dx, dyf = (x2 - x1), (y1 - y2)
            else:
                dx, dyf = (x1 - x2), (y2 - y1)
            if dyf <= 1e-6:
                return None
            return math.atan2(dx, dyf)
        vals = [h for h in (hdg(left_line), hdg(right_line)) if h is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def _publish_info(self, info=None, state="STOP"):
        """Publish /lane_controller/info (Float64MultiArray). ADDITIVE / read-only.

        Layout:
          [0] state_code   [1] heading_rad
          [2] left_valid   [3] right_valid
          [4] left_offset  [5] right_offset   [6] center_offset   [7] lane_width
        Offsets are normalized to half image width: (x - cx) / (W/2), so left is
        ~negative, right ~positive, |.|~0 means the line is at the robot centre
        (about to be crossed). lane_width is (rx-lx)/(W/2) (or dyn width) or 0.
        Invalid/absent fields are 0.0; consumers must gate on the valid flags.
        """
        msg  = Float64MultiArray()
        code = self._STATE_CODE.get(state, 0.0)
        if info is None:
            msg.data = [code, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.info_pub.publish(msg)
            return
        dst_h, dst_w = info["mask"].shape[:2]
        half = max(dst_w / 2.0, 1.0)
        cy   = info["center_y"]
        vl, vr = bool(info["valid_l"]), bool(info["valid_r"])
        lx = self._x_at_y(info["left_line"],  cy) if vl else None
        rx = self._x_at_y(info["right_line"], cy) if vr else None
        lc = info.get("lane_center")
        left_off   = (lx - half) / half if lx is not None else 0.0
        right_off  = (rx - half) / half if rx is not None else 0.0
        center_off = (lc - half) / half if lc is not None else 0.0
        if lx is not None and rx is not None:
            width = (rx - lx) / half
        elif info.get("dyn_lane_width"):
            width = info["dyn_lane_width"] / half
        else:
            width = 0.0
        heading = self._lane_heading(info["left_line"] if vl else None,
                                     info["right_line"] if vr else None)
        msg.data = [code, heading,
                    1.0 if vl else 0.0, 1.0 if vr else 0.0,
                    left_off, right_off, center_off, width]
        self.info_pub.publish(msg)

    # -- Twist output ---------------------------------------------------------

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

    # -- Debug image -----------------------------------------------------------

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
        cv2.line(out, (cx, 0), (cx, dst_h), (100, 100, 100), 1)
        cv2.putText(out, "SX", (cx - 28, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        cv2.putText(out, "DX", (cx + 5,  12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
        if lane_center is not None:
            lc_x = int(lane_center)
            cv2.line(out, (lc_x, 0), (lc_x, dst_h), (0, 0, 255), 2)

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
            msg          = Image()
            msg.height   = out.shape[0]
            msg.width    = out.shape[1]
            msg.encoding = "bgr8"
            msg.step     = out.shape[1] * 3
            msg.data     = out.tobytes()
            self.debug_pub.publish(msg)
        except Exception as e:
            if self.enable_log:
                rospy.logwarn_throttle(5.0, "[lane_ctrl_v2] debug publish: %s" % e)

    # -- Principal Step -------------------------------------------------------

    def _step(self):
        with self.lock:
            mask = None if self.latest_mask is None else self.latest_mask.copy()

        if not self.enabled:
            self._publish_state("DISABLED")
            self._publish_info(None, "DISABLED")
            return

        if mask is None:
            self._publish_state("STOP")
            self._publish_info(None, "STOP")
            return

        steering, _, state, info = self.step(mask)

        self._publish_state(state)
        twist = self._steering_to_twist(steering)
        self.cmd_pub.publish(twist)
        self._publish_info(info, state)

        if self.publish_debug:
            self._publish_debug_image(info, steering, twist)

    # -- Spin ------------------------------------------------------------------

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            try:
                self._step()
            except Exception as e:
                if self.enable_log:
                    rospy.logerr_throttle(2.0, "[lane_ctrl_v2] step err: %s" % e)
            rate.sleep()
        self.cmd_pub.publish(Twist())


# =============================================================================
if __name__ == "__main__":
    try:
        LaneControllerV2Node().run()
    except rospy.ROSInterruptException:
        pass
