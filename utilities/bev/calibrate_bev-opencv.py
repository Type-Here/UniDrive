#!/usr/bin/env python3
"""
calibrate_bev.py -- Interactive BEV homography calibration tool.

How to use:
    1. Place a physical rectangle of known dimensions on the track floor,
       fully visible in the lower portion of the camera image.
       Good options: an A4 sheet (21.0 x 29.7 cm), 4 tape markers,
       or any flat rectangle you can measure with a ruler.

    2. Pick any camera frame where the rectangle is clearly visible.
       The image must NOT be cropped -- use the original full-size frame.

    3. Run this tool:
           python3 calibrate_bev.py [image.jpg]

    4. An interactive window opens. Click the 4 corners of the physical
       rectangle IN THIS ORDER:
           1 (top-left) --> 2 (top-right) --> 3 (bottom-right) --> 4 (bottom-left)
       The "top" of the rectangle is the side farther from the camera
       (higher up in the image, smaller in perspective).

    5. Enter the real-world dimensions when prompted (in metres).

    6. The tool shows a preview of the BEV transformation and saves
       bev_config.json with the homography matrix and BEV parameters.

Output:
    bev_config.json  -- loaded by the lane controller at runtime

Usage:
    python3 calibrate_bev.py [image.jpg] [--out bev_config.json] [--crop-top 0.45]

Options:
    --out PATH         Output JSON file (default: bev_config.json)
    --crop-top FRAC    Same crop fraction used during training (default: 0.45)
                       The tool applies the same crop so pixel coordinates match
                       what the model actually sees.
    --bev-width  W     Width  of the BEV output image in pixels (default: 400)
    --bev-height H     Height of the BEV output image in pixels (default: 400)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from config import ROOT_DIR


# -- Interactive point picker --------------------------------------------------

class PointPicker:
    """
    Opens an OpenCV window and lets the user click exactly 4 points.
    Points are drawn as numbered circles as they are clicked.
    """

    LABELS = ["1 top-left", "2 top-right", "3 bottom-right", "4 bottom-left"]
    COLORS = [
        (0, 255, 0),    # green
        (0, 200, 255),  # yellow
        (0, 100, 255),  # orange
        (255, 0, 100),  # purple
    ]

    def __init__(self, image: np.ndarray):
        self.image  = image.copy()
        self.canvas = image.copy()
        self.points = []
        self.done   = False

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            idx   = len(self.points) - 1
            color = self.COLORS[idx]
            # Draw circle and label
            cv2.circle(self.canvas, (x, y), 8, color, -1)
            cv2.circle(self.canvas, (x, y), 8, (255, 255, 255), 2)
            cv2.putText(self.canvas, self.LABELS[idx],
                        (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 2, cv2.LINE_AA)
            # Draw connecting lines once we have >1 point
            if len(self.points) > 1:
                cv2.line(self.canvas,
                         self.points[-2], self.points[-1],
                         (255, 255, 255), 1, cv2.LINE_AA)
            if len(self.points) == 4:
                cv2.line(self.canvas,
                         self.points[-1], self.points[0],
                         (255, 255, 255), 1, cv2.LINE_AA)
                self.done = True

        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            # Right click to undo last point
            self.points.pop()
            self.canvas = self.image.copy()
            for i, pt in enumerate(self.points):
                c = self.COLORS[i]
                cv2.circle(self.canvas, pt, 8, c, -1)
                cv2.circle(self.canvas, pt, 8, (255, 255, 255), 2)
                cv2.putText(self.canvas, self.LABELS[i],
                            (pt[0] + 12, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            c, 2, cv2.LINE_AA)
            if len(self.points) > 1:
                for i in range(1, len(self.points)):
                    cv2.line(self.canvas,
                             self.points[i-1], self.points[i],
                             (255, 255, 255), 1, cv2.LINE_AA)
            self.done = False

    def _draw_instructions(self):
        instructions = [
            "Click 4 corners of the physical rectangle:",
            "  1. top-left (far side, left)",
            "  2. top-right (far side, right)",
            "  3. bottom-right (near side, right)",
            "  4. bottom-left (near side, left)",
            "Right-click to undo last point.",
            "Press ENTER to confirm, ESC to cancel.",
        ]
        y0 = 25
        for line in instructions:
            cv2.putText(self.canvas, line, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)
            y0 += 22

    def run(self) -> list:
        win = "BEV Calibration -- click 4 corners (right-click to undo, ENTER to confirm)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 900, 600)
        cv2.setMouseCallback(win, self._mouse_cb)

        while True:
            display = self.canvas.copy()
            self._draw_instructions()

            # Status line at bottom
            remaining = 4 - len(self.points)
            status = (f"Click {remaining} more point(s)."
                      if remaining > 0 else
                      "All 4 points selected. Press ENTER to confirm.")
            h = display.shape[0]
            cv2.putText(display, status, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 100) if self.done else (255, 200, 0),
                        2, cv2.LINE_AA)

            cv2.imshow(win, display)
            key = cv2.waitKey(20) & 0xFF

            if key == 13 and self.done:   # ENTER
                break
            elif key == 27:               # ESC
                print("  Cancelled.")
                cv2.destroyAllWindows()
                sys.exit(0)

        cv2.destroyAllWindows()
        return self.points


# -- Homography computation ----------------------------------------------------

def compute_homography(src_pts: list, real_w: float, real_h: float,
                       bev_w: int, bev_h: int) -> np.ndarray:
    """
    Compute the homography matrix H that maps src_pts (image pixel coords)
    to a top-down BEV image of size (bev_w x bev_h).

    The 4 source points correspond to the corners of a rectangle of
    real_w x real_h metres on the floor.

    The destination rectangle is centred in the BEV image, scaled so that
    real_w metres maps to bev_w * 0.8 pixels (80% of BEV width).

    Args:
        src_pts:  list of 4 (x,y) pixel tuples in image: TL, TR, BR, BL
        real_w:   real-world width  of the rectangle in metres
        real_h:   real-world height of the rectangle in metres
        bev_w:    BEV output image width  in pixels
        bev_h:    BEV output image height in pixels

    Returns:
        H  -- 3x3 homography matrix (float64)
        pixels_per_metre  -- scale factor for the BEV image
    """
    # Scale: fit the larger dimension into 80% of the BEV image
    scale = min(bev_w * 0.8 / real_w, bev_h * 0.8 / real_h)  # px/m

    rect_w_px = real_w * scale
    rect_h_px = real_h * scale

    # Centre the rectangle in the BEV image
    cx = bev_w / 2
    cy = bev_h / 2

    # Destination points: TL, TR, BR, BL (same order as src_pts)
    dst_pts = np.float32([
        [cx - rect_w_px / 2, cy - rect_h_px / 2],   # TL
        [cx + rect_w_px / 2, cy - rect_h_px / 2],   # TR
        [cx + rect_w_px / 2, cy + rect_h_px / 2],   # BR
        [cx - rect_w_px / 2, cy + rect_h_px / 2],   # BL
    ])

    src = np.float32(src_pts)
    H, status = cv2.findHomography(src, dst_pts, cv2.RANSAC, 5.0)

    return H, float(scale)


# -- BEV preview ---------------------------------------------------------------

def make_bev_preview(image: np.ndarray, H: np.ndarray,
                     bev_w: int, bev_h: int) -> np.ndarray:
    return cv2.warpPerspective(image, H, (bev_w, bev_h))


def save_preview(original: np.ndarray, bev: np.ndarray,
                 src_pts: list, out_path: Path):
    """
    Save a side-by-side image: annotated original | BEV result.
    """
    # Draw polygon on original
    orig_ann = original.copy()
    pts = np.array(src_pts, dtype=np.int32)
    cv2.polylines(orig_ann, [pts], True, (0, 255, 0), 2)
    labels = ["1-TL", "2-TR", "3-BR", "4-BL"]
    colors = [(0,255,0),(0,200,255),(0,100,255),(255,0,100)]
    for i, (pt, lab, col) in enumerate(zip(src_pts, labels, colors)):
        cv2.circle(orig_ann, pt, 8, col, -1)
        cv2.putText(orig_ann, lab, (pt[0]+10, pt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    # Resize both to same height for hstack
    h = max(orig_ann.shape[0], bev.shape[0])
    def pad_h(img, target_h):
        if img.shape[0] < target_h:
            pad = np.zeros((target_h - img.shape[0], img.shape[1], 3),
                           dtype=np.uint8)
            return np.vstack([img, pad])
        return img

    panel = np.hstack([pad_h(orig_ann, h), pad_h(bev, h)])
    cv2.imwrite(str(out_path), panel)
    print(f"  Preview    : {out_path}")


# -- Save config ---------------------------------------------------------------

def save_config(H: np.ndarray, scale: float,
                src_pts: list, real_w: float, real_h: float,
                bev_w: int, bev_h: int, crop_top_frac: float,
                out_path: Path):
    """
    Save BEV calibration config as JSON.

    The homography H maps from the CROPPED image (after applying crop_top_frac)
    to the BEV image. The lane controller must apply the same crop before
    calling warpPerspective.
    """
    config = {
        "homography":       H.tolist(),
        "pixels_per_metre": scale,
        "bev_width":        bev_w,
        "bev_height":       bev_h,
        "crop_top_frac":    crop_top_frac,
        "src_points_px":    src_pts,
        "real_rect_m":      {"width": real_w, "height": real_h},
        "notes": (
            "H maps from cropped image pixels to BEV pixels. "
            "Apply the same crop_top_frac before warpPerspective. "
            "pixels_per_metre is the BEV scale factor."
        )
    }
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {out_path}")


# -- Main ----------------------------------------------------------------------

def main():

    bev_folder = Path(ROOT_DIR) / "artifacts" / "bev"
    bev_default_in = bev_folder / "reference"
    bev_save = bev_folder / "bev_config.json"

    # Check if default input reference exists and if it's jpg or png
    # Replace "reference.*" * with correct extension:
    if bev_default_in.with_suffix(".jpg").exists():
        bev_default_in = bev_default_in.with_suffix(".jpg")
    elif bev_default_in.with_suffix(".png").exists():
        bev_default_in = bev_default_in.with_suffix(".png")
    else:
        print(f"WARNING: No default reference image found at {bev_default_in.with_suffix('.jpg')} or {bev_default_in.with_suffix('.png')}")
        bev_default_in = None

    parser = argparse.ArgumentParser(
        description="Interactive BEV homography calibration")
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help=(
            "Camera frame to use for calibration. "
            "If omitted, uses artifacts/bev/reference.jpg or .png when available."
        ),
    )
    parser.add_argument("--out",        default=f"{bev_save}",)
    parser.add_argument("--crop-top",   type=float, default=0.45,
                        dest="crop_top",
                        help="Same crop fraction used in training (default: 0.45)")
    parser.add_argument("--bev-width",  type=int, default=400,
                        dest="bev_width")
    parser.add_argument("--bev-height", type=int, default=400,
                        dest="bev_height")
    args = parser.parse_args()

    img_path = Path(args.image) if args.image else bev_default_in
    if img_path is None:
        print("ERROR: no input image provided and no default reference image found.")
        print("       Provide an image path, or place reference.jpg/reference.png in artifacts/bev/.")
        sys.exit(1)
    if not img_path.exists():
        print(f"ERROR: input image not found: {img_path}")
        sys.exit(1)
    out_path = Path(args.out)

    # Load image
    img_full = cv2.imread(str(img_path))
    if img_full is None:
        print(f"ERROR: cannot read {img_path}")
        sys.exit(1)

    h_full, w_full = img_full.shape[:2]

    # Apply the same crop used during training
    crop_top_px = int(h_full * args.crop_top)
    img_cropped = img_full[crop_top_px:, :]

    print(f"\n  Image      : {img_path}  ({w_full}x{h_full})")
    print(f"  Crop top   : {crop_top_px}px  ({args.crop_top*100:.0f}%)")
    print(f"  Cropped    : {img_cropped.shape[1]}x{img_cropped.shape[0]}")
    print(f"  BEV size   : {args.bev_width}x{args.bev_height}")
    print()
    print("  INSTRUCTIONS:")
    print("  1. Place a physical rectangle of KNOWN dimensions on the track floor.")
    print("     It must be fully visible in the image.")
    print("     Example: A4 sheet = 0.210 x 0.297 m")
    print()
    print("  2. In the window that opens, click the 4 corners in order:")
    print("     1=top-left  2=top-right  3=bottom-right  4=bottom-left")
    print("     'top' = the side farther from the camera (smaller in perspective)")
    print()
    print("  3. Press ENTER to confirm, right-click to undo last point.")
    print()

    # Open picker on cropped image
    picker = PointPicker(img_cropped)
    src_pts = picker.run()

    if len(src_pts) != 4:
        print("ERROR: need exactly 4 points")
        sys.exit(1)

    print(f"  Points selected (in cropped image):")
    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
    for label, pt in zip(labels, src_pts):
        print(f"    {label:<14}: ({pt[0]}, {pt[1]})")

    # Get real-world dimensions
    print()
    print("  Enter the REAL dimensions of the rectangle you clicked:")
    try:
        real_w = float(input("  Width  (metres, left-right):  "))
        real_h = float(input("  Height (metres, near-far):    "))
    except (ValueError, EOFError):
        print("  ERROR: invalid input")
        sys.exit(1)

    if real_w <= 0 or real_h <= 0:
        print("  ERROR: dimensions must be positive")
        sys.exit(1)

    # Compute homography
    H, scale = compute_homography(
        src_pts, real_w, real_h, args.bev_width, args.bev_height)

    print(f"\n  Homography computed.")
    print(f"  Scale: {scale:.1f} pixels/metre")

    # BEV preview
    bev = make_bev_preview(img_cropped, H, args.bev_width, args.bev_height)

    preview_path = out_path.with_name(out_path.stem + "_preview.jpg")
    save_preview(img_cropped, bev, src_pts, preview_path)

    # Show preview in window briefly
    cv2.imshow("BEV result -- press any key to close", bev)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # Save config
    save_config(H, scale, src_pts, real_w, real_h,
                args.bev_width, args.bev_height,
                args.crop_top, out_path)

    print()
    print("  Next steps:")
    print(f"    1. Check the preview: {preview_path}")
    print(f"       The BEV should show the rectangle as a square/rectangle")
    print(f"       from above, with straight lines.")
    print(f"    2. If the BEV looks wrong, re-run and click more carefully.")
    print(f"    3. Copy {out_path} to the robot:")
    print(f"       scp {out_path} jetauto@<ip>:~/")
    print()


if __name__ == "__main__":
    main()