#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lane_core.py — logica pura del lane controller, senza dipendenze ROS.

Importabile da:
  - lane_controller_node.py  (nodo ROS sul robot)
  - Testing/offline_tester.py (test offline su Mac, senza ROS)

La classe LaneControllerCore riceve i parametri come dict Python
(le chiavi corrispondono ai nomi in lane_params.yaml sotto lane_controller:).
"""
from __future__ import print_function
import math
import warnings

import cv2
import numpy as np


# Colori BGR per classe segmentation (usati da _make_colored_mask e dal debug)
CLASS_COLORS = {
    0: (0,   0,   0),
    1: (180, 130,  70),
    2: (0,   255, 255),
    3: (255, 255,   0),
    4: (0,   0,   255),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class LaneControllerCore(object):
    """
    Logica pura del lane controller: Hough → fit → steering.

    Instanziare con un dict params (vedere lane_params.yaml per le chiavi).
    I valori di default corrispondono a quelli del file YAML.
    """

    def __init__(self, params):
        p = params

        # ── Modalità BEV ─────────────────────────────────────────────────────
        self.use_bev   = bool( p.get("use_bev",   True))
        self.bev_scale = float(p.get("bev_scale", 1.0))

        # ── ROI Hough ────────────────────────────────────────────────────────
        self.hough_roi_top_frac = float(p.get("hough_roi_top_frac", 0.0))

        # ── HoughLinesP ──────────────────────────────────────────────────────
        self.hough_thresh     = int(  p.get("hough_threshold",     50))
        self.hough_min_line   = int(  p.get("hough_min_line_px",   20))
        self.hough_max_gap    = int(  p.get("hough_max_gap_px",    40))
        self.hough_min_length = int(  p.get("hough_min_length_px", 20))
        self.min_valid_points = int(  p.get("min_valid_points",     0))
        self.line_height_ratio = float(p.get("line_height_ratio",  0.8))

        # ── Geometria corsia ─────────────────────────────────────────────────
        self.lane_width_px            = float(p.get("lane_width_px",              280.0))
        self.center_y_ratio           = float(p.get("center_y_ratio",             0.10))
        self.max_steer_angle          = float(p.get("max_steering_angle",         48.0))
        self.single_line_offset       = float(p.get("single_line_offset",          0.0))
        self.min_distance_from_center = float(p.get("min_distance_from_center",    0.0))

        # ── Fit linee ────────────────────────────────────────────────────────
        # "linear" retta | "quadratic" parabola | "auto" parabola se >=6 punti
        self.lane_fit_mode = p.get("lane_fit_mode", "auto")

        # ── EMA adattivo ─────────────────────────────────────────────────────
        self.alpha_base  = float(p.get("angle_smooth_alpha_base",  0.65))
        self.alpha_delta = float(p.get("angle_smooth_delta_scale", 8.0))

        # ── Velocità (usate per calcolare angular_z nel return di step()) ────
        self.max_angular_z = float(p.get("max_angular_z",  0.80))
        self.linear_x      = float(p.get("linear_x_speed", 0.05))

        # ── Class IDs ────────────────────────────────────────────────────────
        self.cls_marking = int(p.get("class_lane_marking", 2))
        self.cls_dashed  = int(p.get("class_lane_dashed",  3))

        # ── Parametri no-BEV ─────────────────────────────────────────────────
        self.lane_width_bottom_frac = float(p.get("lane_width_bottom_frac", 0.55))
        self.no_bev_roi_top_frac    = float(p.get("no_bev_roi_top_frac",    0.45))

        # ── Calibrazione dinamica larghezza corsia (solo use_bev=True) ───────
        self.lane_width_dynamic_enable = bool( p.get("lane_width_dynamic_enable", True))
        self.lane_width_ema_alpha      = float(p.get("lane_width_ema_alpha",      0.10))
        self.lane_width_min_px         = float(p.get("lane_width_min_px",         180.0))
        self.lane_width_max_px         = float(p.get("lane_width_max_px",         380.0))
        self.lane_width_sanity_band    = float(p.get("lane_width_sanity_band",    0.25))
        self.lane_width_reset_after    = int(  p.get("lane_width_reset_after",      0))

        # ── Stato interno ────────────────────────────────────────────────────
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

    # ── Hook di logging (override nella sottoclasse ROS) ─────────────────────

    def _warn(self, msg):
        """Override in LaneControllerV2Node per usare rospy.logwarn_throttle."""
        pass

    # ── Utilità maschera ─────────────────────────────────────────────────────

    def _get_binary(self, mask):
        return ((mask == self.cls_marking) | (mask == self.cls_dashed)).astype(np.uint8) * 255

    def _make_colored_mask(self, mask):
        h, w = mask.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in CLASS_COLORS.items():
            colored[mask == cls_id] = color
        return colored

    # ── Pipeline Hough ───────────────────────────────────────────────────────

    def _detect_hough(self, bev_binary):
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(bev_binary, cv2.MORPH_CLOSE, kernel)
        return cv2.HoughLinesP(
            closed, 1, np.pi / 180,
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
        """BEV mode: classifica per slope + posizione rispetto a dyn_cx_line."""
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

    def _separate_lines_no_bev(self, lines, img_cx):
        """no-BEV mode: classifica per posizione x del midpoint (non usa slope)."""
        if lines is None:
            return [], []
        left_lines, right_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (x1 + x2) / 2.0 < img_cx:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])
        return left_lines, right_lines

    def _lane_width_at_y(self, y, dst_h, dst_w):
        """Larghezza corsia stimata in px a quota y.
        BEV: dyn_lane_width se calibrato e abilitato, altrimenti lane_width_px statico.
        no-BEV: modello prospettivo lineare.
        """
        if self.use_bev:
            if self.lane_width_dynamic_enable and self.dyn_lane_width is not None:
                return self.dyn_lane_width
            return self.lane_width_px
        return self.lane_width_bottom_frac * float(dst_w) * (float(y) / max(float(dst_h), 1.0))

    def _update_dyn_lane_width(self, lx, rx):
        """Aggiorna self.dyn_lane_width via EMA con sanity-check.
        Bound assoluti sempre; banda relativa solo se gia inizializzato.
        Restituisce True se la misura e' stata accettata nell'EMA.
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
        Fit polinomiale sui punti dei segmenti di un lato.
        Grado: "linear"→1  "quadratic"→2 se ≥3 punti  "auto"→2 se ≥6 punti.
        Ritorna ([x_bot, y_bot, x_top, y_top], poly) oppure (None, None).
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
        """Valuta x a target_y: usa poly se disponibile, altrimenti interpolazione lineare."""
        if poly is not None:
            v = float(np.polyval(poly, target_y))
            return v if math.isfinite(v) else None
        return self._get_x_at_y(line, target_y)

    def _validate_lines(self, left_line, right_line, img_cx, dst_h):
        """Lunghezza >= hough_min_length e almeno min_valid_points check-y corretti."""
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
        Calcola angolo di sterzata in gradi.
        1. Campiona a center_y (clamped sopra roi_top_px).
        2. Scarta linee dal lato sbagliato o troppo vicine al centro.
        3. lane_center = media L+R, o stima da singola linea con larghezza prospettica.
        4. Mappa normalizzata → angolo piecewise.
        5. EMA adattivo.
        """
        img_cx   = dst_w / 2.0
        center_y = max(int(dst_h * self.center_y_ratio), roi_top_px)

        position_check_passed = True
        if valid_l:
            lx = self._get_x_at_y(left_line, center_y)
            if lx is not None and lx >= img_cx:
                valid_l = False
                self._warn("[lane_core] L line crossed center — discarded")
                position_check_passed = False
            elif lx is not None and abs(lx - img_cx) < self.min_distance_from_center:
                valid_l = False
                self._warn("[lane_core] L line too close to center — discarded")
                position_check_passed = False
        if valid_r:
            rx = self._get_x_at_y(right_line, center_y)
            if rx is not None and rx <= img_cx:
                valid_r = False
                self._warn("[lane_core] R line crossed center — discarded")
                position_check_passed = False
            elif rx is not None and abs(rx - img_cx) < self.min_distance_from_center:
                valid_r = False
                self._warn("[lane_core] R line too close to center — discarded")
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
                if self.use_bev and self.lane_width_dynamic_enable:
                    self._update_dyn_lane_width(lx, rx)
        elif valid_l:
            lx = self._eval_line(poly_l, left_line, center_y)
            if lx is not None:
                half_w = self._lane_width_at_y(center_y, dst_h, dst_w) / 2.0
                lane_center = lx + half_w + self.single_line_offset
        elif valid_r:
            rx = self._eval_line(poly_r, right_line, center_y)
            if rx is not None:
                half_w = self._lane_width_at_y(center_y, dst_h, dst_w) / 2.0
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

        if self.use_bev:
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

    # ── Entry point ──────────────────────────────────────────────────────────

    def step(self, mask):
        """
        Processa una maschera uint8 con class IDs (BEV o raw, dipende da use_bev).
        Ritorna (steering_deg, angular_z, state, debug_info).

        debug_info keys:
          mask        → maschera processata (dopo bev_scale) per la visualizzazione
          left_line   → [x_bot, y_bot, x_top, y_top] o None
          right_line  → idem
          valid_l     → bool
          valid_r     → bool
          lane_center → float o None
          center_y    → int (quota di misura)
          roi_top_px  → int (limite superiore ROI Hough)
        """
        # Upscaling (INTER_NEAREST preserva class IDs interi)
        if self.bev_scale != 1.0:
            mask = cv2.resize(mask, None, fx=self.bev_scale, fy=self.bev_scale,
                              interpolation=cv2.INTER_NEAREST)

        dst_h, dst_w = mask.shape[:2]
        img_cx      = dst_w / 2.0
        roi_top_frac = self.hough_roi_top_frac if self.use_bev else self.no_bev_roi_top_frac
        roi_top_px  = int(dst_h * roi_top_frac)

        bev_bin = self._get_binary(mask)
        bev_roi = bev_bin[roi_top_px:, :]
        lines   = self._offset_lines_y(self._detect_hough(bev_roi), roi_top_px)

        if self.use_bev:
            left_lines, right_lines = self._separate_lines(lines, img_cx)
        else:
            left_lines, right_lines = self._separate_lines_no_bev(lines, img_cx)

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
