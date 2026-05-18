#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lane_controller_node.py  (v3)
-----------------------------
Controller laterale per JetAuto a ruote Mecanum, basato su segmentation
mask. Logica per-classe (no clustering generico):

  Classi maschera:
    1 = road
    2 = lane_marking  (qualsiasi linea continua, sx o dx)
    3 = lane_dashed   (linea tratteggiata, di solito centrale o sx)
    4 = zebra         (ignorata in questa versione)

  Casi gestiti:
    TRACKING_CC   -> 2 cluster di classe 2: sx e dx        (caso comune)
    TRACKING_DC   -> classe 3 a sx + classe 2 a dx         (tratto raro)
    SINGLE_LINE   -> solo classe 2 (un lato della corsia)  (curve strette)
    SINGLE_DASHED -> solo classe 3                         (raro)
    GRACE         -> niente per <N frame: ramp-down velocità ultimo Twist
    STOP          -> niente per >=N frame, oppure disabilitato

Modalità input (parametro input_mode):
    "camera"     -> maschera in image space (default)
    "bev_topic"  -> maschera GIÀ warpata pubblicata sul mask_topic
    "bev_warp"   -> warp interno usando bev_config.json

Compatibile Python 2.7 (Melodic) / Python 3 (Noetic).
"""

from __future__ import print_function
import threading

import numpy as np
import cv2
import rospy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge, CvBridgeError


# Colormap per debug (BGR)
CLASS_COLORS = {
    0: (0,   0,   0),
    1: (60,  60,  60),
    2: (0,   0,   255),    # lane_marking continua -> rosso
    3: (0,   255, 255),    # lane_dashed -> giallo
    4: (255, 0,   255),    # zebra -> magenta (ignorata)
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =============================================================================
class ClassDescriptor(object):
    """
    Estrae descrittori da una singola classe nella ROI.
    Può restituire:
      - None se sotto soglia
      - un singolo blob {cx, cy, slope, n}                 (linea unica)
      - due blob {left:..., right:...} se lo split sulla   (caso comune)
        mediana di x trova due cluster ben separati
    """

    def __init__(self, min_pixels_total=80, min_pixels_per_side=40,
                 min_separation_px=60):
        self.min_total = min_pixels_total
        self.min_side = min_pixels_per_side
        self.min_sep = min_separation_px

    def analyze(self, mask_roi, class_id):
        bin_mask = (mask_roi == class_id)
        n_total = int(np.count_nonzero(bin_mask))
        if n_total < self.min_total:
            return None

        ys, xs = np.nonzero(bin_mask)
        h_roi = mask_roi.shape[0]

        # Provo lo split sulla mediana di x: se ho due lati distinti,
        # le due metà avranno centroidi ben separati.
        med_x = float(np.median(xs))
        sel_l = xs <= med_x
        sel_r = ~sel_l
        n_l = int(sel_l.sum()); n_r = int(sel_r.sum())

        # Test split valido
        valid_split = False
        if n_l >= self.min_side and n_r >= self.min_side:
            cxl = float(xs[sel_l].mean())
            cxr = float(xs[sel_r].mean())
            if (cxr - cxl) >= self.min_sep:
                valid_split = True

        if valid_split:
            return {
                'mode': 'PAIR',
                'left':  self._desc(xs, ys, sel_l, h_roi),
                'right': self._desc(xs, ys, sel_r, h_roi),
            }
        else:
            return {
                'mode': 'SINGLE',
                'single': self._desc(xs, ys, np.ones_like(xs, dtype=bool), h_roi),
            }

    @staticmethod
    def _desc(xs, ys, sel, h_roi):
        xs_s = xs[sel]; ys_s = ys[sel]
        n = int(xs_s.size)
        cx = float(xs_s.mean())
        cy = float(ys_s.mean())
        slope = 0.0
        top = ys_s < (h_roi * 0.5)
        bot = ~top
        if top.any() and bot.any():
            slope = float(xs_s[top].mean() - xs_s[bot].mean())
        return (cx, cy, slope, n)


# =============================================================================
class BEVWarper(object):
    """Warp opzionale a BEV usando bev_config.json (formato compatibile col
    repo on_jetauto_scripts/utilities/calibrate_bev.py)."""

    def __init__(self, bev_config_path):
        import json
        with open(bev_config_path, "r") as f:
            cfg = json.load(f)
        self.H = np.array(cfg["homography"], dtype=np.float64)
        self.bev_w = int(cfg["bev_width"])
        self.bev_h = int(cfg["bev_height"])

    def warp(self, mask):
        return cv2.warpPerspective(
            mask, self.H, (self.bev_w, self.bev_h),
            flags=cv2.INTER_NEAREST)


# =============================================================================
class LaneControllerNode(object):

    def __init__(self):
        rospy.init_node("lane_controller_node", anonymous=False)

        ns = "lane_controller/"
        # ---- Topic ----
        self.mask_topic    = rospy.get_param(ns + "mask_topic", "/lane_mask")
        self.rgb_topic     = rospy.get_param(ns + "rgb_topic", "/depth_cam/rgb/image_raw")
        self.cmd_topic     = rospy.get_param(ns + "cmd_topic", "/jetauto_controller/cmd_vel")
        self.debug_topic   = rospy.get_param(ns + "debug_topic", "/lane_debug/image")
        self.state_topic   = rospy.get_param(ns + "state_topic", "/lane_controller/state")
        self.enable_topic  = rospy.get_param(ns + "enable_topic", "/lane_controller/enable")

        # ---- Modalità input ----
        self.input_mode = rospy.get_param(ns + "input_mode", "camera")
        self.bev_config = rospy.get_param(ns + "bev_config_path", "")

        # ---- Velocità ----
        self.linear_x      = float(rospy.get_param(ns + "linear_x_speed", 0.10))
        self.max_linear_y  = float(rospy.get_param(ns + "max_linear_y", 0.15))
        self.max_angular_z = float(rospy.get_param(ns + "max_angular_z", 1.20))

        # ---- Gain P ----
        self.Kp_lat = float(rospy.get_param(ns + "Kp_lat", 0.0030))
        self.Kp_ang = float(rospy.get_param(ns + "Kp_ang", 0.010))

        # ---- Logica corsie ----
        self.lateral_offset_px      = int(rospy.get_param(ns + "lateral_offset_px", 60))
        self.min_pixels_threshold   = int(rospy.get_param(ns + "min_pixels_threshold", 80))
        self.min_pixels_per_side    = int(rospy.get_param(ns + "min_pixels_per_side", 40))
        self.min_lr_separation      = int(rospy.get_param(ns + "min_lr_separation", 60))
        self.roi_top_fraction       = float(rospy.get_param(ns + "mask_roi_top_fraction", 0.55))
        self.smooth_alpha           = float(rospy.get_param(ns + "smooth_alpha", 0.5))
        self.single_line_mode       = rospy.get_param(ns + "single_line_mode", "spatial")

        # ---- Anti-flicker / grace period ----
        self.no_lane_grace_frames = int(rospy.get_param(ns + "no_lane_grace_frames", 3))
        self.no_lane_decay        = float(rospy.get_param(ns + "no_lane_decay", 0.5))

        # ---- Class IDs ----
        self.cls_marking = int(rospy.get_param(ns + "class_lane_marking", 2))
        self.cls_dashed  = int(rospy.get_param(ns + "class_lane_dashed", 3))

        # ---- Debug ----
        self.publish_debug = bool(rospy.get_param(ns + "publish_debug", True))
        self.debug_scale   = float(rospy.get_param(ns + "debug_scale", 0.5))
        self.rate_hz       = float(rospy.get_param(ns + "control_rate_hz", 20))

        # ---- Setup BEV ----
        self.bev_warper = None
        if self.input_mode == "bev_warp":
            if not self.bev_config:
                rospy.logfatal("[lane_controller] input_mode=bev_warp ma bev_config_path vuoto")
                raise SystemExit(1)
            self.bev_warper = BEVWarper(self.bev_config)
            rospy.loginfo("[lane_controller] BEV warper: %dx%d",
                          self.bev_warper.bev_w, self.bev_warper.bev_h)

        # In BEV niente compensazione di slope (prospettiva già rimossa)
        self.use_slope_for_angular = (self.input_mode == "camera")

        # Class descriptor (parametri condivisi sia per classe 2 sia per classe 3)
        self.descr = ClassDescriptor(
            min_pixels_total=self.min_pixels_threshold,
            min_pixels_per_side=self.min_pixels_per_side,
            min_separation_px=self.min_lr_separation)

        # Stato
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_mask = None
        self.latest_rgb  = None
        self.enabled     = False   # attende AVVIA dalla dashboard
        self.smoothed_err = 0.0
        self.last_state   = "STOP"
        self.last_twist   = Twist()       # ultimo comando valido (per grace period)
        self.no_lane_count = 0

        # Pub/Sub
        self.cmd_pub   = rospy.Publisher(self.cmd_topic,   Twist,  queue_size=1)
        self.state_pub = rospy.Publisher(self.state_topic, String, queue_size=1, latch=True)
        if self.publish_debug:
            self.debug_pub = rospy.Publisher(self.debug_topic, Image, queue_size=1)

        rospy.Subscriber(self.mask_topic, Image, self._mask_cb,
                         queue_size=1, buff_size=2**20)
        if self.publish_debug:
            rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb,
                             queue_size=1, buff_size=2**22)
        rospy.Subscriber(self.enable_topic, Bool, self._enable_cb, queue_size=1)

        rospy.loginfo("[lane_controller] avviato. input_mode=%s mask=%s grace=%d",
                      self.input_mode, self.mask_topic, self.no_lane_grace_frames)

    # ------------------------------------------------------------- callbacks
    def _mask_cb(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, "[lane_controller] mask bridge: %s" % e)
            return
        if self.bev_warper is not None:
            mask = self.bev_warper.warp(mask)
        with self.lock:
            self.latest_mask = mask

    def _rgb_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, "[lane_controller] rgb bridge: %s" % e)
            return
        with self.lock:
            self.latest_rgb = img

    def _enable_cb(self, msg):
        self.enabled = bool(msg.data)
        rospy.loginfo("[lane_controller] enable=%s", self.enabled)
        if not self.enabled:
            self.cmd_pub.publish(Twist())
            self._publish_state("DISABLED")

    def _publish_state(self, s):
        if s != self.last_state:
            self.state_pub.publish(String(data=s))
            self.last_state = s

    # ------------------------------------------------------------- helpers logica
    def _decide_target(self, info_marking, info_dashed, img_cx):
        """
        Restituisce (state, target_x, slope, dbg_extras) in funzione di cosa
        è stato visto dei due tipi di linea.
        dbg_extras: dict con info da disegnare nel debug (lato linea, ecc.)
        """
        m = info_marking; d = info_dashed
        extras = {'side': None}

        # Numero di "lati" rilevati per ciascuna classe
        m_pair  = m is not None and m['mode'] == 'PAIR'
        m_single = m is not None and m['mode'] == 'SINGLE'
        d_present = d is not None  # PAIR o SINGLE per dashed

        # CASO 1: due continue (tipico) -> centra tra le due
        if m_pair:
            l = m['left']; r = m['right']
            target_x = 0.5 * (l[0] + r[0])
            slope = 0.5 * (l[2] + r[2])
            return "TRACKING_CC", target_x, slope, extras

        # CASO 2: dashed-sx + continua-dx (tratto raro a doppia corsia)
        # condizione: ho una continua singola sul lato destro dell'immagine
        # E ho una dashed sul lato sinistro.
        if m_single and d_present:
            ms = m['single']
            ds = d['single'] if d['mode'] == 'SINGLE' else d.get('left', d.get('right'))
            # Verifico che la continua sia a destra del centro
            # e la dashed a sinistra del centro
            if ms[0] > img_cx and ds is not None and ds[0] < img_cx:
                target_x = 0.5 * (ms[0] + ds[0])
                slope = 0.5 * (ms[2] + ds[2])
                extras['used_dashed'] = True
                return "TRACKING_DC", target_x, slope, extras

        # CASO 3: solo continua singola (curva stretta -> linea esterna)
        if m_single:
            ms = m['single']
            slope = ms[2]
            cx_line = ms[0]
            target_x, side = self._apply_single_offset(cx_line, img_cx,
                                                       self.single_line_mode,
                                                       self.lateral_offset_px)
            extras['side'] = side
            return "SINGLE_LINE", target_x, slope, extras

        # CASO 4: solo dashed (raro)
        if d_present:
            ds = d['single'] if d['mode'] == 'SINGLE' else d.get('left') or d.get('right')
            if ds is not None:
                slope = ds[2]
                cx_line = ds[0]
                # Per la dashed l'offset è opposto: di solito è centrale,
                # quindi se la vedo a sinistra del centro -> sto a destra di essa,
                # se a destra -> sto a sinistra.
                target_x, side = self._apply_single_offset(cx_line, img_cx,
                                                           "spatial_dashed",
                                                           self.lateral_offset_px)
                extras['side'] = side
                extras['used_dashed'] = True
                return "SINGLE_DASHED", target_x, slope, extras

        # CASO 5: niente
        return "NONE", None, 0.0, extras

    @staticmethod
    def _apply_single_offset(cx_line, img_cx, mode, offset_px):
        """Determina target_x dato un cluster singolo e la modalità."""
        if mode == "fixed_right":
            return cx_line - offset_px, 'R'
        if mode == "fixed_left":
            return cx_line + offset_px, 'L'
        if mode == "spatial_dashed":
            # dashed centrale: se a sx tieniti a destra di essa, e viceversa
            if cx_line < img_cx:
                return cx_line + offset_px, 'L'
            else:
                return cx_line - offset_px, 'R'
        # default "spatial" per la continua singola:
        # cluster a sx -> è bordo SX -> target a destra (+offset)
        # cluster a dx -> è bordo DX -> target a sinistra (-offset)
        if cx_line < img_cx:
            return cx_line + offset_px, 'L'
        else:
            return cx_line - offset_px, 'R'

    # ------------------------------------------------------------- step
    def _step(self):
        with self.lock:
            mask = None if self.latest_mask is None else self.latest_mask.copy()
            rgb  = None if self.latest_rgb  is None else self.latest_rgb.copy()

        if not self.enabled:
            self._publish_state("DISABLED")
            return

        if mask is None:
            # Non abbiamo ancora ricevuto mask
            self._publish_state("STOP")
            return

        H, W = mask.shape[:2]
        roi_y0 = int(H * self.roi_top_fraction)
        roi = mask[roi_y0:, :]
        img_cx = W * 0.5

        info_marking = self.descr.analyze(roi, self.cls_marking)
        info_dashed  = self.descr.analyze(roi, self.cls_dashed)

        state, target_x, slope, extras = self._decide_target(
            info_marking, info_dashed, img_cx)

        twist = Twist()

        if target_x is not None:
            # Reset contatore: linee viste
            self.no_lane_count = 0

            err_px = target_x - img_cx
            a = self.smooth_alpha
            self.smoothed_err = a * self.smoothed_err + (1.0 - a) * err_px

            lin_y = -self.Kp_lat * self.smoothed_err
            ang_z = -self.Kp_ang * slope if self.use_slope_for_angular else 0.0

            twist.linear.x  = self.linear_x
            twist.linear.y  = clamp(lin_y, -self.max_linear_y, self.max_linear_y)
            twist.angular.z = clamp(ang_z, -self.max_angular_z, self.max_angular_z)

            self.last_twist = twist
            self.cmd_pub.publish(twist)
            self._publish_state(state)
        else:
            # Nessuna linea: applico grace period
            self.no_lane_count += 1
            if self.no_lane_count <= self.no_lane_grace_frames:
                # Decay: ripeto l'ultimo Twist con velocità ridotta
                decay = self.no_lane_decay
                twist.linear.x  = self.last_twist.linear.x  * decay
                twist.linear.y  = self.last_twist.linear.y  * decay
                twist.angular.z = self.last_twist.angular.z * decay
                self.cmd_pub.publish(twist)
                self._publish_state("GRACE")
            else:
                # STOP definitivo
                self.smoothed_err = 0.0
                self.last_twist = Twist()
                self.cmd_pub.publish(Twist())
                self._publish_state("STOP")

        if self.publish_debug:
            self._publish_debug(mask, rgb, roi_y0,
                                info_marking, info_dashed,
                                target_x, state, twist, extras)

    # ------------------------------------------------------------- debug overlay
    def _publish_debug(self, mask, rgb, roi_y0,
                       info_marking, info_dashed,
                       target_x, state, twist, extras):
        H, W = mask.shape[:2]

        if rgb is not None:
            rgb_resized = cv2.resize(rgb, (W, H)) if rgb.shape[:2] != (H, W) else rgb.copy()
            base = rgb_resized
        else:
            base = cv2.cvtColor((mask * 50).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        overlay = np.zeros_like(base)
        for cid, color in CLASS_COLORS.items():
            if cid == 0:
                continue
            overlay[mask == cid] = color
        out = cv2.addWeighted(base, 0.65, overlay, 0.35, 0)

        cv2.line(out, (0, roi_y0), (W - 1, roi_y0), (255, 255, 255), 1)
        cv2.line(out, (W // 2, roi_y0), (W // 2, H - 1), (180, 180, 180), 1)

        # Disegna i cluster di classe 2 (continua) in rosso
        self._draw_class_info(out, info_marking, roi_y0, (0, 0, 255), 'C')
        # Disegna i cluster di classe 3 (dashed) in giallo
        self._draw_class_info(out, info_dashed, roi_y0, (0, 255, 255), 'D')

        if target_x is not None:
            tx = int(target_x); ty = (roi_y0 + H) // 2
            cv2.arrowedLine(out, (W // 2, H - 10), (tx, ty),
                            (0, 255, 0), 2, tipLength=0.2)
            cv2.circle(out, (tx, ty), 7, (0, 255, 0), 2)

        col = {
            "TRACKING_CC":   (0, 255, 0),
            "TRACKING_DC":   (0, 255, 200),
            "SINGLE_LINE":   (0, 200, 255),
            "SINGLE_DASHED": (0, 255, 255),
            "GRACE":         (0, 165, 255),
            "STOP":          (0, 0, 255),
            "DISABLED":      (160, 160, 160),
            "NONE":          (0, 0, 255),
        }.get(state, (255, 255, 255))
        cv2.putText(out, "STATE: %s [%s]" % (state, self.input_mode),
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.putText(out, "vx=%.2f vy=%.2f wz=%.2f" %
                    (twist.linear.x, twist.linear.y, twist.angular.z),
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if self.no_lane_count > 0:
            cv2.putText(out, "no_lane=%d/%d" %
                        (self.no_lane_count, self.no_lane_grace_frames),
                        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 165, 255), 1, cv2.LINE_AA)

        if self.debug_scale != 1.0:
            out = cv2.resize(out, None, fx=self.debug_scale, fy=self.debug_scale,
                             interpolation=cv2.INTER_AREA)
        try:
            msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
            self.debug_pub.publish(msg)
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, "[lane_controller] debug bridge: %s" % e)

    @staticmethod
    def _draw_class_info(img, info, roi_y0, color, prefix):
        if info is None:
            return
        if info['mode'] == 'PAIR':
            for label, key in (('L', 'left'), ('R', 'right')):
                d = info[key]
                cx = int(d[0]); cy = int(d[1]) + roi_y0
                cv2.circle(img, (cx, cy), 6, color, -1)
                cv2.putText(img, "%s%s" % (prefix, label), (cx + 8, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        elif info['mode'] == 'SINGLE':
            d = info['single']
            cx = int(d[0]); cy = int(d[1]) + roi_y0
            cv2.circle(img, (cx, cy), 6, color, -1)
            cv2.putText(img, prefix, (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------- spin
    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            try:
                self._step()
            except Exception as e:
                rospy.logerr_throttle(2.0, "[lane_controller] step err: %s" % e)
            rate.sleep()
        self.cmd_pub.publish(Twist())


if __name__ == "__main__":
    try:
        LaneControllerNode().run()
    except rospy.ROSInterruptException:
        pass
