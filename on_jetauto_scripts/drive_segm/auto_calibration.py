from typing import Union

import numpy as np
import cv2

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

    def calibrate(self, segm_output:np.ndarray, lane_label:int) -> float:
        """
        Given the segmentation output, compute the average x position of the lane pixels
        in the masked area. This can be used to estimate the horizontal offset of the lane.
        """
        # Here we suppose that top-left mask image is point (0,0)
        mask_w = segm_output.shape[1]
        mask_h = segm_output.shape[0]

        if not (0 <= self.top_line < mask_h):
            print("Warning: top_line is out of bounds.")
            return 0.0

        bottom = self.bottom_line if self.bottom_line is not None else mask_h - 1
        if not (self.top_line < bottom < mask_h):
            print("Warning: bottom_line is out of bounds.")
            return 0.0

        # Find TL, TR points: use top_line height and search for
        # the first pixel and the last pixel with lane_label
        # in that row inside the mask image (array)
        row = segm_output[self.top_line]
        lane_pixels = np.where(row == lane_label)[0]  # get x indices of lane pixels in the top line
        if lane_pixels.size == 0:
            print("Warning: no lane pixels found in the top line.")
            return 0.0

        tl = lane_pixels[0]  # leftmost lane pixel
        tr = lane_pixels[-1] # rightmost lane pixel

        bottom_row = segm_output[bottom]
        lane_pixels_bottom = np.where(bottom_row == lane_label)[0]
        if lane_pixels_bottom.size == 0:
            print("Warning: no lane pixels found in the bottom line.")
            return 0.0
        bl = lane_pixels_bottom[0]
        br = lane_pixels_bottom[-1]

        # Enlarge of 10 pixels the points if possible
        tl = max(0, tl - 10)
        tr = min(mask_w - 1, tr + 10)
        bl = max(0, bl - 10)
        br = min(mask_w - 1, br + 10)

        # Store last calibration points for BEV warp
        self._last_src_pts = np.float32([
            [tl, self.top_line],
            [tr, self.top_line],
            [br, bottom],
            [bl, bottom],
        ])
        self._cached_M = None  # invalidate cached matrix on recalibration

        # Calculate the angle to warp image based on tl, tr, bl, br point in order to get them aligned vertically
        # We can use the average of the angles between (tl, bl) and (tr, br)
        dy = bottom - self.top_line
        angle_tl_bl = np.arctan2(dy, bl - tl)
        angle_tr_br = np.arctan2(dy, br - tr)
        angle = (angle_tl_bl + angle_tr_br) / 2
        angle = float(np.clip(angle, -self.max_angle, self.max_angle))
        self.calibration_angle = angle

        # Save src points and angle in json file as
        # points and angle names
        with open(self.save_path, "w") as f:
            import json
            json.dump({
                "src_points": self._last_src_pts.tolist(),
                "calibration_angle": self.calibration_angle,
            }, f, indent=2)


        return angle


    def _compute_warp_points(self, segm_output: np.ndarray):
        mask_w = segm_output.shape[1]
        mask_h = segm_output.shape[0]
        bottom = self.bottom_line if self.bottom_line is not None else mask_h - 1
        if not (0 <= self.top_line < bottom < mask_h):
            return None
        if self._last_src_pts is None:
            return None
        src_pts = self._last_src_pts
        # dst_pts span the full output height so warpPerspective stretches directly
        # to the final size -- no separate crop+resize needed
        dst_pts = np.float32([
            [0.0, 0.0],
            [mask_w - 1.0, 0.0],
            [mask_w - 1.0, mask_h - 1.0],
            [0.0, mask_h - 1.0],
        ])
        return bottom, src_pts, dst_pts

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

    def save_debug(self, segm_output: np.ndarray, prefix: str = "lane_calibration") -> None:
        mask_u8 = segm_output.astype(np.uint8, copy=False)
        mask_h, mask_w = mask_u8.shape[:2]
        colored = self._colorize_mask(mask_u8)
        pts = self._compute_warp_points(mask_u8)
        if pts is None:
            cv2.imwrite(prefix + "_points.jpg", colored)
            return
        bottom, src_pts, dst_pts = pts
        vis = colored.copy()
        for (x, y) in src_pts:
            cv2.circle(vis, (int(x), int(y)), 5, (255, 255, 255), -1)
        cv2.imwrite(prefix + "_points.jpg", vis)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            colored, M, (mask_w, mask_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(prefix + "_warp.jpg", warped)

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
            bottom = self.bottom_line if self.bottom_line is not None else mask_h - 1
            return segm_output[self.top_line:bottom, :]

        bottom, src_pts, dst_pts = pts

        cur_size = (segm_output.shape[1], segm_output.shape[0])
        if self._cached_M is None or self._cached_mask_size != cur_size:
            self._cached_M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self._cached_mask_size = cur_size
        M = self._cached_M

        mask_u8 = segm_output.astype(np.uint8, copy=False)

        # Single warpPerspective fills the full output -- no crop or resize needed
        return cv2.warpPerspective(
            mask_u8, M, (mask_w, mask_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
