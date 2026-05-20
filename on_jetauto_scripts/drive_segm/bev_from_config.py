import json

import cv2
import numpy as np

MODEL_H = 256
MODEL_W = 640
SRC_IMAGE_WIDTH = 640
SRC_IMAGE_HEIGHT = 480


class BEVProcessor:
    """Legacy BEV processor that uses bev_config.json homography."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            cfg = json.load(f)

        self.bev_w = int(cfg["bev_width"])
        self.bev_h = int(cfg["bev_height"])
        self.crop_top_frac = float(cfg["crop_top_frac"])
        self.px_per_m = float(cfg["pixels_per_metre"])

        H = np.array(cfg["homography"], dtype=np.float64)

        src_w = int(cfg.get("src_image_width", SRC_IMAGE_WIDTH))
        src_h = int(cfg.get("src_image_height", SRC_IMAGE_HEIGHT))
        crop_w = int(cfg.get("cropped_width", src_w))
        crop_h = int(cfg.get("cropped_height", int(src_h * (1.0 - self.crop_top_frac))))

        if crop_w <= 0 or crop_h <= 0:
            crop_w, crop_h = MODEL_W, MODEL_H

        scale_x = float(crop_w) / float(MODEL_W)
        scale_y = float(crop_h) / float(MODEL_H)
        S = np.array([
            [scale_x, 0.0, 0.0],
            [0.0, scale_y, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self.H = H @ S

    def mask_to_bev(self, mask: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(
            mask.astype(np.uint8), self.H,
            (self.bev_w, self.bev_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)

