from typing import Union

import numpy as np
import cv2

class AutoCalibration:
    def __init__(self, top_line:int, bottom_line:Union[int, None]):
        self.top_line = int(top_line)
        self.bottom_line = int(bottom_line) if bottom_line is not None else None
        self.calibration_angle = 0.0
        self.max_angle = np.deg2rad(40.0)
        self._last_src_pts = None

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

        # Calculate the angle to warp image based on tl, tr, bl, br point in order to get them aligned vertically
        # We can use the average of the angles between (tl, bl) and (tr, br)
        dy = bottom - self.top_line
        angle_tl_bl = np.arctan2(dy, bl - tl)
        angle_tr_br = np.arctan2(dy, br - tr)
        angle = (angle_tl_bl + angle_tr_br) / 2
        angle = float(np.clip(angle, -self.max_angle, self.max_angle))
        self.calibration_angle = angle
        return angle


    def _compute_warp_points(self, segm_output: np.ndarray):
        mask_w = segm_output.shape[1]
        mask_h = segm_output.shape[0]
        bottom = self.bottom_line if self.bottom_line is not None else mask_h - 1
        if not (0 <= self.top_line < bottom < mask_h):
            return None
        if self._last_src_pts is None:
            return None
        span = bottom - self.top_line
        src_pts = self._last_src_pts
        dst_pts = np.float32([
            [0.0, 0.0],
            [mask_w - 1.0, 0.0],
            [mask_w - 1.0, span],
            [0.0, span],
        ])
        return bottom, span, src_pts, dst_pts

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
        bottom, span, src_pts, dst_pts = pts
        # Draw points on the colored mask
        vis = colored.copy()
        for (x, y) in src_pts:
            cv2.circle(vis, (int(x), int(y)), 5, (255, 255, 255), -1)
        cv2.imwrite(prefix + "_points.jpg", vis)
        # Warp the colored mask for inspection
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            colored, M, (mask_w, mask_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Stretch to original mask height (same as make_bev output)
        stretched = cv2.resize(warped[self.top_line:bottom, :], (mask_w, mask_h),
                               interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(prefix + "_warp.jpg", stretched)

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

        bottom, span, src_pts, dst_pts = pts

        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Warp the image using the perspective transform
        mask_u8 = segm_output.astype(np.uint8, copy=False)

        warped = cv2.warpPerspective(
            mask_u8, M, (mask_w, mask_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Crop the warped image to the area between top_line and bottom_line
        crop_height = self.bottom_line - self.top_line
        mask_cropped = warped[:crop_height, :]

        # Stretch image to original mask height
        stretched = cv2.resize(mask_cropped, (mask_w, mask_h), interpolation=cv2.INTER_NEAREST)

        return stretched
