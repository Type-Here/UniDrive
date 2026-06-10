#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lane_core.py - pure lane controller logic, no ROS dependencies.

Importable from:
  - lane_controller_node.py  (ROS node on the robot)
  - Testing/offline_tester.py (offline testing on Mac, without ROS)

The LaneControllerCore class receives parameters as a Python dict
(keys match the names in lane_params.yaml under lane_controller:).
"""
from __future__ import print_function
import math
import warnings

import cv2
import numpy as np


# BGR colors per segmentation class (used by _make_colored_mask and debug output)
CLASS_COLORS = {
    0: (0,   0,   0),    # 0 background
    1: (180, 130,  70),  # 1 road
    2: (0,   255, 255),  # 2 lane_marking
    3: (255, 255,   0),  # 3 lane_dashed
    4: (0,   0,   255),  # 4 zebra
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class LaneControllerCore(object):
    """
    Pure lane controller logic: Hough -> fit -> steering.

    Instantiate with a params dict (see lane_params.yaml for keys).
    Default values match those in the YAML file.
    """

    def __init__(self, params):
        p = params

        # -- BEV mode ----------------------------------------------------------
        # The controller is BEV-only: it consumes the BEV-warped mask published on
        # /lane_mask_bev. The legacy raw-perspective path was removed.
        self.bev_scale = float(p.get("bev_scale", 1.0))

        # -- ROI Hough --------------------------------------------------------
        self.hough_roi_top_frac = float(p.get("hough_roi_top_frac", 0.0))

        # -- HoughLinesP ------------------------------------------------------
        # Pixel parameters in the YAML are expressed at bev_scale=1.0 and are
        # multiplied by bev_scale here; hough_threshold (votes) does not scale.
        s = self.bev_scale
        self.hough_thresh     = int(  p.get("hough_threshold",     50))
        self.hough_min_line   = max(1, int(round(float(p.get("hough_min_line_px",   20)) * s)))
        self.hough_max_gap    = max(1, int(round(float(p.get("hough_max_gap_px",    40)) * s)))
        self.hough_min_length = max(1, int(round(float(p.get("hough_min_length_px", 20)) * s)))
        self.min_valid_points = int(  p.get("min_valid_points",     0))
        self.line_height_ratio = float(p.get("line_height_ratio",  0.8))

        # -- Lane geometry -----------------------------------------------------
        self.lane_width_px            = float(p.get("lane_width_px",              280.0)) * s
        self.center_y_ratio           = float(p.get("center_y_ratio",             0.10))
        self.max_steer_angle          = float(p.get("max_steering_angle",         48.0))
        self.single_line_offset       = float(p.get("single_line_offset",          0.0)) * s
        self.min_distance_from_center = float(p.get("min_distance_from_center",    0.0)) * s

        # -- Line fit ----------------------------------------------------------
        # "linear" = line | "quadratic" = parabola | "auto" = parabola if >=6 points
        self.lane_fit_mode = p.get("lane_fit_mode", "auto")

        # -- Adaptive EMA ------------------------------------------------------
        self.alpha_base  = float(p.get("angle_smooth_alpha_base",  0.65))
        self.alpha_delta = float(p.get("angle_smooth_delta_scale", 8.0))

        # -- Velocity (used to compute angular_z in step() return value) -------
        self.max_angular_z = float(p.get("max_angular_z",  0.80))
        self.linear_x      = float(p.get("linear_x_speed", 0.05))

        # -- Class IDs --------------------------------------------------------
        self.cls_marking = int(p.get("class_lane_marking", 2))
        self.cls_dashed  = int(p.get("class_lane_dashed",  3))

        # -- Dynamic lane width calibration -----------------------------------
        self.lane_width_dynamic_enable = bool( p.get("lane_width_dynamic_enable", True))
        self.lane_width_ema_alpha      = float(p.get("lane_width_ema_alpha",      0.10))
        self.lane_width_min_px         = float(p.get("lane_width_min_px",         180.0)) * s
        self.lane_width_max_px         = float(p.get("lane_width_max_px",         380.0)) * s
        self.lane_width_sanity_band    = float(p.get("lane_width_sanity_band",    0.25))
        self.lane_width_reset_after    = int(  p.get("lane_width_reset_after",      0))

        # -- Internal state ----------------------------------------------------
        self.prev_steering = 0.0
        self.dyn_cx        = None
        self.dyn_cx_line   = None
        self.last_slope_l  = None
        self.last_slope_r  = None
        self.no_lane_count = 0
        self.state         = "STOP"

        self.dyn_lane_width         = None
        self.last_measured_width    = None
        self.last_measured_accepted = False
        self.frames_since_two_lines = 0

        # -- Optional GPU (cv2.cuda_GpuMat) - same pattern as lane_follower.py -
        self._use_cuda = (hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0)
        if self._use_cuda:
            self._gpu_mat            = cv2.cuda_GpuMat()
            self._gpu_dst            = cv2.cuda_GpuMat()   # explicit dst for morphology ops
            _k = np.ones((3, 3), np.uint8)
            self._morph_open_filter  = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_OPEN,  cv2.CV_8UC1, _k)
            self._morph_close_filter = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_CLOSE, cv2.CV_8UC1, _k)

    # -- Logging hook (override in the ROS subclass) ---------------------------

    def _warn(self, msg):
        """Override in LaneControllerV2Node to use rospy.logwarn_throttle."""
        pass

    # -- Mask utilities --------------------------------------------------------

    def _get_binary(self, mask):
        return ((mask == self.cls_marking) | (mask == self.cls_dashed)).astype(np.uint8) * 255

    def _make_colored_mask(self, mask):
        h, w = mask.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in CLASS_COLORS.items():
            colored[mask == cls_id] = color
        return colored

    # -- Pipeline Hough -------------------------------------------------------

    def _detect_hough(self, bev_binary):
        processed = None
        if self._use_cuda:
            try:
                self._gpu_mat.upload(bev_binary)
                self._morph_open_filter.apply(self._gpu_mat, self._gpu_dst)
                self._morph_close_filter.apply(self._gpu_dst, self._gpu_mat)
                processed = self._gpu_mat.download()
            except Exception:
                pass   # fall back to CPU below
        if processed is None:
            k         = np.ones((3, 3), np.uint8)
            opened    = cv2.morphologyEx(bev_binary, cv2.MORPH_OPEN,  k)
            processed = cv2.morphologyEx(opened,     cv2.MORPH_CLOSE, k)
        return cv2.HoughLinesP(
            processed, 1, np.pi / 180,
            threshold=self.hough_thresh,
            minLineLength=self.hough_min_line,
            maxLineGap=self.hough_max_gap,
        )

    def _offset_lines_y(self, lines, offset):
        if lines is None or offset == 0:
            return lines
        result = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            result.append([[x1, y1 + offset, x2, y2 + offset]])
        return np.array(result, dtype=np.int32)

    def _separate_lines(self, lines, img_cx):
        """BEV mode: classify by slope + position relative to dyn_cx_line."""
        if lines is None:
            return [], []
        if self.dyn_cx_line is None:
            self.dyn_cx_line = img_cx
        left_lines, right_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = float(y2 - y1) / float(x2 - x1)
            cx = self.dyn_cx_line
            if slope < 0 and x1 < cx and x2 < cx:
                self.last_slope_l = slope
                left_lines.append(line[0])
            elif slope > 0 and x1 > cx and x2 > cx:
                self.last_slope_r = slope
                right_lines.append(line[0])
        return left_lines, right_lines

    def _lane_width_at_y(self):
        """Estimated lane width in px (BEV): dynamic estimate if calibrated and
        enabled, otherwise the static lane_width_px fallback."""
        if self.lane_width_dynamic_enable and self.dyn_lane_width is not None:
            return self.dyn_lane_width
        return self.lane_width_px

    def _update_dyn_lane_width(self, lx, rx):
        """Update self.dyn_lane_width via EMA with sanity-check.
        Absolute bounds always applied; relative band only if already initialised.
        Returns True if the measurement was accepted into the EMA.
        """
        if lx is None or rx is None:
            return False
        measured = float(rx - lx)
        self.last_measured_width    = measured
        self.last_measured_accepted = False

        if measured < self.lane_width_min_px or measured > self.lane_width_max_px:
            return False

        if self.dyn_lane_width is not None:
            rel = abs(measured - self.dyn_lane_width) / max(self.dyn_lane_width, 1.0)
            if rel > self.lane_width_sanity_band:
                return False

        a = self.lane_width_ema_alpha
        if self.dyn_lane_width is None:
            self.dyn_lane_width = measured
        else:
            self.dyn_lane_width = a * measured + (1.0 - a) * self.dyn_lane_width
        self.last_measured_accepted = True
        return True

    def _fit_line(self, lines, dst_h):
        """
        Polynomial fit over segment endpoints for one side.
        Degree: "linear"->1  "quadratic"->2 if ≥3 points  "auto"->2 if ≥6 points.
        Returns ([x_bot, y_bot, x_top, y_top], poly) or (None, None).
        """
        if not lines:
            return None, None
        x_coords, y_coords = [], []
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        if len(x_coords) < 2:
            return None, None

        if self.lane_fit_mode == "linear":
            deg = 1
        elif self.lane_fit_mode == "quadratic":
            deg = 2 if len(x_coords) >= 3 else 1
        else:  # "auto"
            deg = 2 if len(x_coords) >= 6 else 1

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                poly = np.polyfit(y_coords, x_coords, deg)
        except Exception:
            return None, None

        if not np.all(np.isfinite(poly)):
            return None, None

        y_bot = dst_h
        y_top = int(dst_h * (1.0 - self.line_height_ratio))
        x_bot = int(np.polyval(poly, y_bot))
        x_top = int(np.polyval(poly, y_top))
        return [x_bot, y_bot, x_top, y_top], poly

    def _get_x_at_y(self, line, target_y):
        if line is None:
            return None
        x1, y1, x2, y2 = line
        if y2 == y1:
            return float(x1)
        return x1 + float(target_y - y1) / float(y2 - y1) * (x2 - x1)

    def _eval_line(self, poly, line, target_y):
        """Evaluate x at target_y: use poly if available, else linear interpolation."""
        if poly is not None:
            v = float(np.polyval(poly, target_y))
            return v if np.isfinite(v) else None
        return self._get_x_at_y(line, target_y)

    def _validate_lines(self, left_line, right_line, img_cx, dst_h):
        """Length >= hough_min_length and at least min_valid_points check-y values correct."""
        check_ys = [int(dst_h * 0.3), int(dst_h * 0.5), int(dst_h * 0.7)]

        def check(line, side):
            if line is None:
                return False
            x1, y1, x2, y2 = line
            if math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) < self.hough_min_length:
                return False
            ok = 0
            for cy in check_ys:
                lx = self._get_x_at_y(line, cy)
                if lx is None:
                    continue
                if side == 'L' and lx < img_cx - 5:
                    ok += 1
                elif side == 'R' and lx > img_cx + 5:
                    ok += 1
            return ok >= self.min_valid_points

        return check(left_line, 'L'), check(right_line, 'R')

    def _calc_steering(self, left_line, right_line, poly_l, poly_r,
                       valid_l, valid_r, dst_w, dst_h, roi_top_px):
        """
        Compute steering angle in degrees.
        1. Sample at center_y (clamped above roi_top_px).
        2. Discard lines from the wrong side or too close to center.
        3. lane_center = average of L+R, or single-line estimate with perspective width.
        4. Map normalised offset -> piecewise angle.
        5. Adaptive EMA.
        """
        img_cx   = dst_w / 2.0
        center_y = max(int(dst_h * self.center_y_ratio), roi_top_px)

        position_check_passed = True
        if valid_l:
            lx = self._get_x_at_y(left_line, center_y)
            if lx is not None and lx >= img_cx:
                valid_l = False
                self._warn("[lane_core] L line crossed center - discarded")
                position_check_passed = False
            elif lx is not None and abs(lx - img_cx) < self.min_distance_from_center:
                valid_l = False
                self._warn("[lane_core] L line too close to center - discarded")
                position_check_passed = False
        if valid_r:
            rx = self._get_x_at_y(right_line, center_y)
            if rx is not None and rx <= img_cx:
                valid_r = False
                self._warn("[lane_core] R line crossed center - discarded")
                position_check_passed = False
            elif rx is not None and abs(rx - img_cx) < self.min_distance_from_center:
                valid_r = False
                self._warn("[lane_core] R line too close to center - discarded")
                position_check_passed = False

        if not position_check_passed:
            return self.prev_steering, None, center_y

        lane_center = None
        self.last_measured_accepted = False
        if valid_l and valid_r:
            lx = self._eval_line(poly_l, left_line,  center_y)
            rx = self._eval_line(poly_r, right_line, center_y)
            if lx is not None and rx is not None:
                lane_center = 0.5 * (lx + rx)
                if self.lane_width_dynamic_enable:
                    self._update_dyn_lane_width(lx, rx)
        elif valid_l:
            lx = self._eval_line(poly_l, left_line, center_y)
            if lx is not None:
                half_w = self._lane_width_at_y() / 2.0
                lane_center = lx + half_w + self.single_line_offset
        elif valid_r:
            rx = self._eval_line(poly_r, right_line, center_y)
            if rx is not None:
                half_w = self._lane_width_at_y() / 2.0
                lane_center = rx - half_w - self.single_line_offset

        if lane_center is None:
            return self.prev_steering, None, center_y

        norm = (lane_center - img_cx) / (dst_w / 2.0)

        if abs(norm) <= 0.1:
            raw_angle = 0.0
        else:
            sign     = 1.0 if norm > 0 else -1.0
            abs_norm = abs(norm)
            mapped   = abs_norm if abs_norm < 0.5 else 0.75 + (abs_norm - 0.5) * 2.0
            raw_angle = sign * min(mapped, 1.0) * self.max_steer_angle

        raw_angle = clamp(raw_angle, -self.max_steer_angle, self.max_steer_angle)

        delta = abs(raw_angle - self.prev_steering)
        alpha = clamp(
            self.alpha_base + 0.35 * (1.0 - math.exp(-delta / self.alpha_delta)),
            self.alpha_base, 0.95,
        )
        steering = alpha * raw_angle + (1.0 - alpha) * self.prev_steering

        slope_l = self.last_slope_l if self.last_slope_l is not None else 0.0
        slope_r = self.last_slope_r if self.last_slope_r is not None else 0.0
        if steering > 30 and (slope_r < 2 or slope_l > -2):
            self.single_line_offset = 30.0
        else:
            self.single_line_offset = 0.0

        self.prev_steering = steering
        self.dyn_cx_line   = lane_center
        self.dyn_cx = lane_center if self.dyn_cx is None else 0.5 * self.dyn_cx + 0.5 * lane_center

        return steering, lane_center, center_y

    # -- Entry point ----------------------------------------------------------

    def step(self, mask):
        """
        Process a uint8 BEV mask with class IDs.
        Returns (steering_deg, angular_z, state, debug_info).

        debug_info keys:
          mask        -> processed mask (after bev_scale) for visualization
          left_line   -> [x_bot, y_bot, x_top, y_top] or None
          right_line  -> same
          valid_l     -> bool
          valid_r     -> bool
          lane_center -> float or None
          center_y    -> int (measurement height)
          roi_top_px  -> int (Hough ROI upper limit)
        """
        # Upscaling (INTER_NEAREST preserves integer class IDs)
        if self.bev_scale != 1.0:
            if self._use_cuda:
                new_w = int(mask.shape[1] * self.bev_scale)
                new_h = int(mask.shape[0] * self.bev_scale)
                self._gpu_mat.upload(mask)
                mask = cv2.cuda.resize(self._gpu_mat, (new_w, new_h),
                                       interpolation=cv2.INTER_NEAREST).download()
            else:
                mask = cv2.resize(mask, None, fx=self.bev_scale, fy=self.bev_scale,
                                  interpolation=cv2.INTER_NEAREST)

        dst_h, dst_w = mask.shape[:2]
        img_cx      = dst_w / 2.0
        roi_top_px  = int(dst_h * self.hough_roi_top_frac)

        bev_bin = self._get_binary(mask)
        bev_roi = bev_bin[roi_top_px:, :]
        lines   = self._offset_lines_y(self._detect_hough(bev_roi), roi_top_px)

        left_lines, right_lines = self._separate_lines(lines, img_cx)

        left_line,  poly_l = self._fit_line(left_lines,  dst_h)
        right_line, poly_r = self._fit_line(right_lines, dst_h)
        valid_l, valid_r   = self._validate_lines(left_line, right_line, img_cx, dst_h)

        steering, lane_center, center_y = self._calc_steering(
            left_line, right_line, poly_l, poly_r, valid_l, valid_r, dst_w, dst_h, roi_top_px)

        if lane_center is not None:
            self.no_lane_count = 0
            self.state = ("TRACKING_CC" if (valid_l and valid_r)
                          else ("SINGLE_L" if valid_l else "SINGLE_R"))
        else:
            self.no_lane_count += 1
            self.state = "HOLD"

        if self.state == "TRACKING_CC":
            self.frames_since_two_lines = 0
        else:
            self.frames_since_two_lines += 1
            if (self.lane_width_reset_after > 0
                    and self.frames_since_two_lines >= self.lane_width_reset_after):
                self.dyn_lane_width = None

        norm      = steering / self.max_steer_angle
        angular_z = clamp(-norm * self.max_angular_z, -self.max_angular_z, self.max_angular_z)

        return steering, angular_z, self.state, {
            "mask":        mask,
            "left_line":   left_line,  "right_line": right_line,
            "valid_l":     valid_l,    "valid_r":    valid_r,
            "lane_center": lane_center,"center_y":   center_y,
            "roi_top_px":  roi_top_px,
            "dyn_lane_width":      self.dyn_lane_width,
            "measured_lane_width": self.last_measured_width,
            "width_meas_accepted": self.last_measured_accepted,
        }
