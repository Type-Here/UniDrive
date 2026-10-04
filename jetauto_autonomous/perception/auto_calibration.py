"""
BEV auto-calibration: find the four lane corners in a segmentation mask and
build the perspective warp that turns the mask into a bird's-eye view.

The calibration is split into three independent steps so that a caller can
preview a candidate calibration before committing to it (see
``bev_calibration_session.py``):

    compute()      pure -- scans the mask, returns a CalibrationResult.
                   Touches no instance state and writes nothing.
    apply_points() commits a result into this instance (invalidates the
                   cached warp matrix).
    save()         persists the committed calibration to JSON (atomic).

``calibrate()`` is the original one-shot convenience wrapper that chains all
three; it is kept so existing callers (offline tester, legacy lane_follower)
work unchanged.
"""
import json
import os
from collections import namedtuple
from typing import Optional, Tuple, Union

import numpy as np
import cv2


# Result of a calibration attempt. ``ok`` False means nothing usable was found
# and ``reason`` carries a human-readable explanation (surfaced in the UI).
CalibrationResult = namedtuple(
    "CalibrationResult", ["ok", "src_points", "angle", "reason"])


class AutoCalibration:
    def __init__(self, top_line:int, bottom_line:Union[int, None],
                 last_src_pts: np.float32 = None,
                 calib_angle:float = 0.0,
                 save_path="calibration.json"):
        self.top_line = int(top_line)
        self.bottom_line = int(bottom_line) if bottom_line is not None else None
        self.calibration_angle = calib_angle
        self.max_angle = np.deg2rad(70.0)
        self._last_src_pts = last_src_pts
        self.save_path = save_path
        self._cached_M = None        # cached perspective matrix, invalidated on recalibration
        self._cached_mask_size = None

        # Pre-allocated GPU source buffer for BEV warp (reused every frame)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self._gpu_mask_src = cv2.cuda_GpuMat()
            self._use_cuda = True
        else:
            self._use_cuda = False

    # -- Calibration steps -----------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._last_src_pts is not None

    @property
    def src_points(self) -> Optional[np.ndarray]:
        return self._last_src_pts

    def compute(self, segm_output: np.ndarray, lane_label: int) -> CalibrationResult:
        """
        Scan the segmentation mask for the lane corners and derive the warp
        source points, WITHOUT modifying this instance and WITHOUT saving.

        Looks at two rows (``top_line`` and ``bottom_line``): the leftmost and
        rightmost pixel carrying ``lane_label`` on each row give the four
        corners TL, TR, BR, BL.
        """
        # Here we suppose that top-left mask image is point (0,0)
        mask_w = segm_output.shape[1]
        mask_h = segm_output.shape[0]

        if not (0 <= self.top_line < mask_h):
            return CalibrationResult(
                False, None, 0.0,
                "top_line (%d) out of bounds for a %d-row mask"
                % (self.top_line, mask_h))

        bottom = self.bottom_line if self.bottom_line is not None else mask_h - 1
        if not (self.top_line < bottom < mask_h):
            return CalibrationResult(
                False, None, 0.0,
                "bottom_line (%d) out of bounds for a %d-row mask"
                % (bottom, mask_h))

        # Find TL, TR points: use top_line height and search for
        # the first pixel and the last pixel with lane_label
        # in that row inside the mask image (array)
        row = segm_output[self.top_line]
        lane_pixels = np.where(row == lane_label)[0]  # get x indices of lane pixels in the top line
        if lane_pixels.size == 0:
            return CalibrationResult(
                False, None, 0.0,
                "no lane pixels on the top line (row %d): aim the camera at a "
                "stretch of road where both lane markings are visible"
                % self.top_line)

        tl = lane_pixels[0]  # leftmost lane pixel
        tr = lane_pixels[-1] # rightmost lane pixel

        bottom_row = segm_output[bottom]
        lane_pixels_bottom = np.where(bottom_row == lane_label)[0]
        if lane_pixels_bottom.size == 0:
            return CalibrationResult(
                False, None, 0.0,
                "no lane pixels on the bottom line (row %d): aim the camera at "
                "a stretch of road where both lane markings are visible"
                % bottom)

        bl = lane_pixels_bottom[0]
        br = lane_pixels_bottom[-1]

        # Enlarge of 10 pixels the points if possible
        tl = max(0, tl - 10)
        tr = min(mask_w - 1, tr + 10)
        bl = max(0, bl - 10)
        br = min(mask_w - 1, br + 10)

        src_pts = np.float32([
            [tl, self.top_line],
            [tr, self.top_line],
            [br, bottom],
            [bl, bottom],
        ])

        # Calculate the angle to warp image based on tl, tr, bl, br point in order to get them aligned vertically
        # We can use the average of the angles between (tl, bl) and (tr, br)
        dy = bottom - self.top_line
        angle_tl_bl = np.arctan2(dy, bl - tl)
        angle_tr_br = np.arctan2(dy, br - tr)
        angle = (angle_tl_bl + angle_tr_br) / 2
        angle = float(np.clip(angle, -self.max_angle, self.max_angle))

        return CalibrationResult(True, src_pts, angle, "")

    def apply_points(self, src_points: np.ndarray, angle: float) -> None:
        """Commit a calibration into this instance (used from the next frame on)."""
        self._last_src_pts = np.float32(src_points)
        self.calibration_angle = float(angle)
        self._cached_M = None  # invalidate cached matrix on recalibration

    def save(self, path: str = None) -> str:
        """
        Persist the committed calibration as JSON. Written to a temporary file
        and renamed, so a crash mid-write cannot leave a truncated file behind.
        Returns the path written.
        """
        if self._last_src_pts is None:
            raise ValueError("nothing to save: no calibration has been applied")
        target = path or self.save_path
        payload = {
            "src_points": np.asarray(self._last_src_pts).tolist(),
            "calibration_angle": self.calibration_angle,
        }
        tmp = target + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, target)
        return target

    def calibrate(self, segm_output:np.ndarray, lane_label:int) -> float:
        """
        One-shot calibration: compute, commit and save. Returns the calibration
        angle, or 0.0 if no lane corners could be found (in which case nothing
        is committed and nothing is written).
        """
        result = self.compute(segm_output, lane_label)
        if not result.ok:
            print("Warning: " + result.reason)
            return 0.0
        self.apply_points(result.src_points, result.angle)
        self.save()
        return result.angle

    # -- Warp ------------------------------------------------------------------

    def _compute_warp_points(self, segm_output: np.ndarray,
                             src_pts: np.ndarray = None):
        mask_w = segm_output.shape[1]
        mask_h = segm_output.shape[0]
        bottom = self.bottom_line if self.bottom_line is not None else mask_h - 1
        if not (0 <= self.top_line < bottom < mask_h):
            return None
        if src_pts is None:
            src_pts = self._last_src_pts
        if src_pts is None:
            return None
        # dst_pts span the full output height so warpPerspective stretches directly
        # to the final size -- no separate crop+resize needed
        dst_pts = np.float32([
            [0.0, 0.0],
            [mask_w - 1.0, 0.0],
            [mask_w - 1.0, mask_h - 1.0],
            [0.0, mask_h - 1.0],
        ])
        return bottom, np.float32(src_pts), dst_pts

    @staticmethod
    def _colorize_mask(mask_u8: np.ndarray) -> np.ndarray:
        colors = np.array([
            [0, 0, 0],
            [180, 130, 70],
            [0, 255, 255],
            [255, 255, 0],
            [0, 0, 255],
        ], dtype=np.uint8)
        return colors[mask_u8.clip(0, 4)]

    def render_debug(self, segm_output: np.ndarray, src_pts: np.ndarray = None
                     ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Build the two calibration debug images in memory (BGR):

            points  colorized mask with the four source corners marked
            warp    the BEV those corners produce (None if not calibrated)

        ``src_pts`` lets a caller preview a *candidate* calibration without
        committing it; defaults to the committed one.
        """
        mask_u8 = segm_output.astype(np.uint8, copy=False)
        mask_h, mask_w = mask_u8.shape[:2]
        colored = self._colorize_mask(mask_u8)
        pts = self._compute_warp_points(mask_u8, src_pts)
        if pts is None:
            return colored, None
        bottom, src, dst = pts
        vis = colored.copy()
        for (x, y) in src:
            cv2.circle(vis, (int(x), int(y)), 5, (255, 255, 255), -1)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            colored, M, (mask_w, mask_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return vis, warped

    def save_debug(self, segm_output: np.ndarray, prefix: str = "lane_calibration") -> None:
        points_img, warp_img = self.render_debug(segm_output)
        cv2.imwrite(prefix + "_points.jpg", points_img)
        if warp_img is not None:
            cv2.imwrite(prefix + "_warp.jpg", warp_img)

    def make_bev(self, segm_output:np.ndarray) -> np.ndarray:
        """
        For each mask apply a perspective transform to warp the image to a bird's eye view,
        using the calibration angle computed in calibrate().
        Returns only the warped image between top_line and bottom_line parameters from init
        :param segm_output: Segmentation Mask from Model
        :return: Warped and cropped image
        """
        # Here we suppose that top-left mask image is point (0,0)
        mask_w = segm_output.shape[1]
        mask_h = segm_output.shape[0]

        pts = self._compute_warp_points(segm_output)
        if pts is None:
            # Uncalibrated fallback: plain crop of the near region, stretched
            # back to the full mask size. Keeping the output shape constant
            # matters -- downstream consumers (lane_controller) size their ROI
            # and pixel thresholds off it.
            bottom = self.bottom_line if self.bottom_line is not None else mask_h - 1
            crop = segm_output[self.top_line:bottom, :]
            if crop.size == 0:
                return segm_output.astype(np.uint8, copy=False)
            return cv2.resize(crop.astype(np.uint8, copy=False), (mask_w, mask_h),
                              interpolation=cv2.INTER_NEAREST)

        bottom, src_pts, dst_pts = pts

        cur_size = (segm_output.shape[1], segm_output.shape[0])
        if self._cached_M is None or self._cached_mask_size != cur_size:
            self._cached_M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self._cached_mask_size = cur_size
        M = self._cached_M

        mask_u8 = segm_output.astype(np.uint8, copy=False)

        if self._use_cuda:
            self._gpu_mask_src.upload(mask_u8)
            return cv2.cuda.warpPerspective(
                self._gpu_mask_src, M, (mask_w, mask_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0).download()

        return cv2.warpPerspective(
            mask_u8, M, (mask_w, mask_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
