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

    4. An interactive window opens. Left-click to place corners in order:
           1=top-left  2=top-right  3=bottom-right  4=bottom-left
       Right-click to undo the last point.
       Press ENTER or close the window to confirm once 4 points are placed.

    5. Enter the real-world dimensions when prompted (in metres).

    6. The tool saves bev_config.json and a side-by-side preview image.

Usage:
    python3 calibrate_bev.py [image.jpg] [--out bev_config.json] [--crop-top 0.45]

Options:
    --out PATH         Output JSON file (default: bev_config.json)
    --crop-top FRAC    Same crop fraction used during training (default: 0.45)
    --bev-width  W     BEV output width  in pixels (default: 400)
    --bev-height H     BEV output height in pixels (default: 400)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# -- Point picker --------------------------------------------------------------

class PointPicker:
    """
    Matplotlib-based interactive point picker.
    Does NOT require opencv highgui / GTK -- works on headless systems.

    Controls:
        Left-click   -- place a corner (up to 4)
        Right-click  -- undo the last placed corner
        ENTER        -- confirm selection (only if 4 points placed)
        Escape       -- cancel and exit
    """

    LABELS = ["1-TL", "2-TR", "3-BR", "4-BL"]
    COLORS = ["lime", "yellow", "orange", "violet"]

    def __init__(self, image_bgr: np.ndarray):
        self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.points    = []
        self.confirmed = False

    def run(self) -> list:
        import matplotlib
        for backend in ("TkAgg", "Qt5Agg", "Qt4Agg", "WXAgg", "GTK3Agg"):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue

        import matplotlib.pyplot as plt

        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        self.ax.imshow(self.image_rgb)
        self.ax.axis("off")
        self._update_title()

        # Keep references to drawn artists so we can remove them on undo
        self._artists = []   # list of lists, one per point

        self.fig.canvas.mpl_connect("button_press_event",  self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",     self._on_key)
        self.fig.canvas.mpl_connect("close_event",         self._on_close)

        plt.tight_layout()
        plt.show()

        if not self.confirmed:
            print("  Cancelled.")
            sys.exit(0)

        return self.points

    # -- Event handlers --------------------------------------------------------

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:          # left click -- add point
            if len(self.points) >= 4:
                return
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            self._add_point(x, y)

        elif event.button == 3:        # right click -- undo
            self._undo_point()

    def _on_key(self, event):
        if event.key == "enter":
            if len(self.points) == 4:
                self.confirmed = True
                import matplotlib.pyplot as plt
                plt.close(self.fig)
            else:
                print(f"  Need 4 points, only {len(self.points)} placed so far.")

        elif event.key == "escape":
            print("  Cancelled.")
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            sys.exit(0)

        elif event.key == "ctrl+z":    # extra undo shortcut
            self._undo_point()

    def _on_close(self, event):
        # Closing the window counts as confirm if 4 points are placed
        if len(self.points) == 4:
            self.confirmed = True

    # -- Drawing helpers -------------------------------------------------------

    def _add_point(self, x: int, y: int):
        idx   = len(self.points)
        color = self.COLORS[idx]
        label = self.LABELS[idx]

        artists = []

        # Dot
        sc, = self.ax.plot(x, y, "o", color=color, ms=10,
                           mec="white", mew=1.5, zorder=5)
        artists.append(sc)

        # Label
        ann = self.ax.annotate(
            label, (x, y),
            xytext=(x + 15, y - 15),
            color=color, fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=1.2))
        artists.append(ann)

        # Line from previous point
        if len(self.points) >= 1:
            px, py = self.points[-1]
            ln, = self.ax.plot([px, x], [py, y],
                               color="white", lw=1.2, alpha=0.7, zorder=4)
            artists.append(ln)

        # Closing line back to first point when placing the 4th
        if len(self.points) == 3:
            fx, fy = self.points[0]
            ln2, = self.ax.plot([x, fx], [y, fy],
                                color="white", lw=1.2, alpha=0.7, zorder=4)
            artists.append(ln2)

        self._artists.append(artists)
        self.points.append((x, y))
        self._update_title()
        self.fig.canvas.draw_idle()

    def _undo_point(self):
        if not self.points:
            return
        # Remove all artists associated with the last point
        for artist in self._artists.pop():
            artist.remove()
        self.points.pop()
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        n = len(self.points)
        if n < 4:
            next_label = self.LABELS[n]
            msg = (f"Left-click to place corner {n+1}/4: {next_label}    "
                   f"Right-click or Ctrl+Z to undo    Esc to cancel")
        else:
            msg = "4 points placed. Press ENTER or close window to confirm."
        self.ax.set_title(msg, fontsize=9,
                          color="lime" if n == 4 else "white")


# -- Homography ----------------------------------------------------------------

def compute_homography(src_pts: list, real_w: float, real_h: float,
                       bev_w: int, bev_h: int) -> tuple:
    """
    Compute H that maps src_pts (cropped image pixels) to a top-down BEV.
    The real rectangle is centred in the BEV and scaled to fill 80% of it.

    Returns (H, pixels_per_metre).
    """
    scale     = min(bev_w * 0.8 / real_w, bev_h * 0.8 / real_h)
    rect_w_px = real_w * scale
    rect_h_px = real_h * scale
    cx, cy    = bev_w / 2, bev_h / 2

    dst_pts = np.float32([
        [cx - rect_w_px / 2, cy - rect_h_px / 2],   # TL
        [cx + rect_w_px / 2, cy - rect_h_px / 2],   # TR
        [cx + rect_w_px / 2, cy + rect_h_px / 2],   # BR
        [cx - rect_w_px / 2, cy + rect_h_px / 2],   # BL
    ])

    H, _ = cv2.findHomography(np.float32(src_pts), dst_pts, cv2.RANSAC, 5.0)
    return H, float(scale)


# -- Preview and config --------------------------------------------------------

def save_preview(original_bgr: np.ndarray, bev_bgr: np.ndarray,
                 src_pts: list, out_path: Path):
    ann    = original_bgr.copy()
    pts_np = np.array(src_pts, dtype=np.int32)
    cv2.polylines(ann, [pts_np], True, (0, 255, 0), 2)

    labels = ["1-TL", "2-TR", "3-BR", "4-BL"]
    colors = [(0,255,0), (0,200,255), (0,100,255), (255,0,100)]
    for pt, lab, col in zip(src_pts, labels, colors):
        cv2.circle(ann, pt, 8, col, -1)
        cv2.putText(ann, lab, (pt[0]+10, pt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    # Pad to same height then concatenate
    h = max(ann.shape[0], bev_bgr.shape[0])
    def pad(img, th):
        if img.shape[0] < th:
            p = np.zeros((th - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, p])
        return img

    panel = np.hstack([pad(ann, h), pad(bev_bgr, h)])
    cv2.imwrite(str(out_path), panel)
    print(f"  Preview    : {out_path}")


def save_config(H: np.ndarray, scale: float, src_pts: list,
                real_w: float, real_h: float,
                bev_w: int, bev_h: int,
                crop_top_frac: float,
                src_w: int, src_h: int, crop_w: int, crop_h: int,
                out_path: Path):
    config = {
        "homography":       H.tolist(),
        "pixels_per_metre": scale,
        "bev_width":        bev_w,
        "bev_height":       bev_h,
        "crop_top_frac":    crop_top_frac,
        "src_image_width":  int(src_w),
        "src_image_height": int(src_h),
        "cropped_width":    int(crop_w),
        "cropped_height":   int(crop_h),
        "src_points_px":    src_pts,
        "real_rect_m":      {"width": real_w, "height": real_h},
        "notes": (
            "H maps from cropped-image pixels to BEV pixels. "
            "Apply crop_top_frac before cv2.warpPerspective. "
            "pixels_per_metre gives the BEV scale. "
            "cropped_width/height describe the crop used for calibration."
        )
    }
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config     : {out_path}")


# -- Main ----------------------------------------------------------------------

def main():
    # Default paths -- adapt to your project layout
    default_ref = None
    for ext in (".jpg", ".jpeg", ".png"):
        p = Path("artifacts/bev/reference" + ext)
        if p.exists():
            default_ref = p
            break

    parser = argparse.ArgumentParser(
        description="Interactive BEV homography calibration")
    parser.add_argument("image", nargs="?", default=None,
                        help="Input frame (omit to use artifacts/bev/reference.*)")
    parser.add_argument("--out",        default="bev_config.json")
    parser.add_argument("--crop-top",   type=float, default=0.45, dest="crop_top")
    parser.add_argument("--bev-width",  type=int,   default=640,  dest="bev_width")
    parser.add_argument("--bev-height", type=int,   default=640,  dest="bev_height")
    args = parser.parse_args()

    img_path = Path(args.image) if args.image else default_ref
    if img_path is None or not img_path.exists():
        print("ERROR: no input image found. Provide a path or place")
        print("       reference.jpg / reference.png in artifacts/bev/")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and crop
    img_full = cv2.imread(str(img_path))
    if img_full is None:
        print(f"ERROR: cannot read {img_path}")
        sys.exit(1)

    h_full, w_full = img_full.shape[:2]
    crop_px        = int(h_full * args.crop_top)
    img_cropped    = img_full[crop_px:, :]
    crop_h, crop_w = img_cropped.shape[:2]

    print(f"\n  Image   : {img_path}  ({w_full}x{h_full})")
    print(f"  Crop    : top {crop_px}px removed  -> {img_cropped.shape[1]}x{img_cropped.shape[0]}")
    print(f"  BEV out : {args.bev_width}x{args.bev_height} px")
    print()
    print("  Place a rectangle of KNOWN size on the track floor.")
    print("  Example: A4 sheet = 0.210 wide x 0.297 m tall")
    print()
    print("  Controls in the window:")
    print("    Left-click  -- place corner (order: TL -> TR -> BR -> BL)")
    print("    Right-click -- undo last corner")
    print("    Ctrl+Z      -- undo last corner")
    print("    ENTER       -- confirm (needs 4 points)")
    print("    Escape      -- cancel")
    print()

    # Pick points
    picker  = PointPicker(img_cropped)
    src_pts = picker.run()

    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
    print("  Points (in cropped image):")
    for lab, pt in zip(labels, src_pts):
        print(f"    {lab:<14}: {pt}")

    # Real dimensions
    print()
    print("  Enter the real dimensions of the rectangle:")
    try:
        real_w = float(input("  Width  (m, left to right) : "))
        real_h = float(input("  Height (m, near to far)   : "))
    except (ValueError, EOFError):
        print("  ERROR: invalid input")
        sys.exit(1)

    # Compute and save
    H, scale = compute_homography(src_pts, real_w, real_h,
                                   args.bev_width, args.bev_height)
    print(f"\n  Scale : {scale:.1f} px/m")

    bev          = cv2.warpPerspective(img_cropped, H,
                                       (args.bev_width, args.bev_height))
    preview_path = out_path.with_name(out_path.stem + "_preview.jpg")
    save_preview(img_cropped, bev, src_pts, preview_path)
    save_config(H, scale, src_pts, real_w, real_h,
                args.bev_width, args.bev_height,
                args.crop_top, w_full, h_full, crop_w, crop_h,
                out_path)

    # Show BEV result
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))
        plt.title("BEV result -- close to finish")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

    print()
    print(f"  Done. Check the preview at {preview_path}")
    print(f"  Copy config to robot: scp {out_path} jetauto@<ip>:~/")
    print()


if __name__ == "__main__":
    main()