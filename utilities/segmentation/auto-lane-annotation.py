#!/usr/bin/env python3
"""
preannotate.py - Auto-generate LabelMe JSON annotations for lane markings
=========================================================================
Uses in-memory enhancement + thresholding + geometric filtering to detect
white stripes on dark road surface, then exports contours as LabelMe polygons.

The generated .json files can be opened directly in LabelMe for refinement.

Usage:
    python3 preannotate.py <image_or_folder> [options]

Arguments:
    image_or_folder     Single image or folder of images (jpg/png)

Options:
    --label NAME        Label name for annotations (default: lane_marking)
    --thresh INT        Brightness threshold 0-255 (default: 130)
    --min-area INT      Minimum blob area in pixels (default: 50)
    --horizon FLOAT     Horizon as fraction of image height 0-1 (default: auto)
    --adaptive-thresh   Use adaptive threshold instead of fixed --thresh
    --epsilon-frac F    Polygon simplification fraction (default: 0.005)
    --max-points INT    Max points per polygon (default: 40)
    --out DIR           Output directory (default: same as input)
    --preview           Also save a preview image with overlay
    --debug-temp        Save temporary preprocessing images under _debug/

Examples:
    python3 preannotate.py frames/
    python3 preannotate.py frames/ --preview
    python3 preannotate.py frames/ --thresh 190 --min-area 30 --preview
    python3 preannotate.py frame_001.jpg --horizon 0.45 --preview
"""

import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path


# -- Core segmentation ---------------------------------------------------------

def find_horizon(gray, max_frac=0.65):
    """
    Auto-detect road horizon row.
    Finds the sharpest dark transition in the vertical gradient
    (wall->road = sudden drop in brightness going top->bottom).
    """
    h, w = gray.shape
    cx   = w // 4
    row_means = gray[:, cx:3*cx].mean(axis=1)
    grad      = np.diff(row_means.astype(float))
    search    = grad[:int(h * max_frac)]
    horizon   = int(np.argmin(search))
    # Clamp between 15% and 65% of height
    return max(int(h * 0.15), min(int(h * max_frac), horizon))


def preprocess_lane_signal(img):
    """Enhance bright lane markings while reducing illumination bias."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(luminance)

    # Top-hat emphasizes bright thin structures on darker background.
    k_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    top_hat = cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, k_tophat)
    top_hat = cv2.GaussianBlur(top_hat, (5, 5), 0)
    return luminance, eq, top_hat


def segment_stripes(
    img,
    horizon_frac=None,
    thresh_bright=130,
    min_area=50,
    adaptive_thresh=False,
    return_debug=False,
):
    """
    Returns a binary mask (uint8) where 255 = lane marking.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    luminance, eq, lane_signal = preprocess_lane_signal(img)
    h, w = gray.shape

    # Horizon
    if horizon_frac is not None:
        horizon = int(h * horizon_frac)
    else:
        horizon = find_horizon(gray)

    # Road mask - only process pixels below horizon
    road_mask = np.zeros((h, w), np.uint8)
    road_mask[horizon:, :] = 255

    # Brightness thresholding on enhanced signal.
    if adaptive_thresh:
        bright = cv2.adaptiveThreshold(
            lane_signal,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            -5,
        )
    else:
        _, bright = cv2.threshold(lane_signal, thresh_bright, 255, cv2.THRESH_BINARY)

    # Local contrast: stripe pixels sit on dark background
    kernel = np.ones((15, 15), np.uint8)
    local_min = cv2.erode(eq, kernel)
    contrast = eq.astype(np.int16) - local_min.astype(np.int16)
    contrast_mask = (contrast > 35).astype(np.uint8) * 255

    # Combine all masks
    mask = cv2.bitwise_and(bright,        contrast_mask)
    mask = cv2.bitwise_and(mask,          road_mask)

    # Morphological cleanup
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # Remove small and non-stripe-like blobs.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, n_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        aspect = max(bw, bh) / max(1, min(bw, bh))
        fill_ratio = area / float(max(1, bw * bh))
        is_large_component = area >= (min_area * 8)

        if aspect < 1.4 and not is_large_component:
            continue
        if fill_ratio > 0.9 and not is_large_component:
            continue

        clean[labels == i] = 255

    if not return_debug:
        return clean, horizon

    debug = {
        "gray": gray,
        "luminance": luminance,
        "eq": eq,
        "lane_signal": lane_signal,
        "bright": bright,
        "contrast": np.clip(contrast, 0, 255).astype(np.uint8),
        "contrast_mask": contrast_mask,
        "road_mask": road_mask,
        "raw_mask": mask,
        "clean": clean,
    }
    return clean, horizon, debug


# -- Contour -> LabelMe polygon -------------------------------------------------

def mask_to_polygons(mask, epsilon_frac=0.005, max_points=40, min_poly_area=30):
    """
    Convert binary mask to list of simplified polygons.
    Each polygon is a list of [x, y] points.

    epsilon_frac: contour approximation strength (fraction of perimeter).
                  Higher = simpler polygons, fewer points.
    max_points:   hard cap on points per polygon.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_poly_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        epsilon = max(1.0, epsilon_frac * perimeter)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Gradually simplify until under the point cap.
        while len(approx) > max_points and epsilon < (0.08 * perimeter):
            epsilon *= 1.35
            approx = cv2.approxPolyDP(cnt, epsilon, True)

        pts = approx.reshape(-1, 2)
        if pts.ndim < 2 or len(pts) < 3:
            continue

        if len(pts) > max_points:
            step = max(1, len(pts) // max_points)
            pts = pts[::step]
            if len(pts) < 3:
                continue

        polygons.append([[float(p[0]), float(p[1])] for p in pts])

    return polygons


# -- LabelMe JSON builder ------------------------------------------------------

def build_labelme_json(img_path, polygons, label, img_w, img_h):
    shapes = []
    for poly in polygons:
        shapes.append({
            "label":       label,
            "points":      poly,
            "group_id":    None,
            "description": "auto",
            "shape_type":  "polygon",
            "flags":       {},
            "mask":        None
        })
    return {
        "version":     "5.3.1",
        "flags":       {},
        "shapes":      shapes,
        "imagePath":   Path(img_path).name,
        "imageData":   None,   # LabelMe loads from file, no need to embed
        "imageHeight": img_h,
        "imageWidth":  img_w,
    }


# -- Preview image -------------------------------------------------------------

def save_preview(img, mask, horizon, polygons, out_path):
    h, w = img.shape[:2]

    # Green overlay on mask
    overlay        = img.copy()
    overlay[mask > 0] = [0, 220, 0]
    blended        = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)

    # Draw polygons in cyan
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(blended, [pts], True, (255, 220, 0), 1)

    # Draw horizon line
    cv2.line(blended, (0, horizon), (w, horizon), (0, 0, 255), 1)
    cv2.putText(blended, f"horizon={horizon}px",
                (5, horizon - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 255), 1)

    # Side-by-side: original | mask | annotated
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    panel    = np.hstack([img, mask_bgr, blended])
    cv2.imwrite(str(out_path), panel)


# -- Process single image ------------------------------------------------------

def process_image(img_path, args, out_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  WARNING: cannot read {img_path} - skipping")
        return None

    h, w = img.shape[:2]

    if args.debug_temp:
        mask, horizon, debug_steps = segment_stripes(
            img,
            horizon_frac=args.horizon,
            thresh_bright=args.thresh,
            min_area=args.min_area,
            adaptive_thresh=args.adaptive_thresh,
            return_debug=True,
        )
    else:
        mask, horizon = segment_stripes(
            img,
            horizon_frac=args.horizon,
            thresh_bright=args.thresh,
            min_area=args.min_area,
            adaptive_thresh=args.adaptive_thresh,
            return_debug=False,
        )
        debug_steps = None

    polygons = mask_to_polygons(
        mask,
        epsilon_frac=args.epsilon_frac,
        max_points=args.max_points,
    )
    json_data = build_labelme_json(img_path, polygons, args.label, w, h)

    # Save JSON
    json_path = out_dir / (img_path.stem + ".json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Save preview
    if args.preview:
        prev_path = out_dir / (img_path.stem + "_preview.jpg")
        save_preview(img, mask, horizon, polygons, prev_path)

    if args.debug_temp and debug_steps is not None:
        debug_dir = out_dir / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for name, debug_img in debug_steps.items():
            debug_path = debug_dir / f"{img_path.stem}_{name}.png"
            cv2.imwrite(str(debug_path), debug_img)

    n_shapes  = len(polygons)
    n_pixels  = int(mask.sum() // 255)
    print(f"  {img_path.name:<35} "
          f"horizon={horizon:3d}px  "
          f"shapes={n_shapes:2d}  "
          f"pixels={n_pixels:6d}  "
          f"-> {json_path.name}")

    return n_shapes


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate LabelMe JSON for lane markings")
    parser.add_argument("input",
        help="Image file or folder of images")
    parser.add_argument("--label",    default="lane_marking",
        help="Annotation label (default: lane_marking)")
    parser.add_argument("--thresh",   type=int,   default=130,
        help="Brightness threshold 0-255 (default: 130)")
    parser.add_argument("--min-area", type=int,   default=50,
        dest="min_area",
        help="Min blob area in pixels (default: 50)")
    parser.add_argument("--horizon",  type=float, default=None,
        help="Horizon as fraction of height e.g. 0.45 (default: auto)")
    parser.add_argument("--adaptive-thresh", action="store_true",
        help="Use adaptive threshold on enhanced luminance signal")
    parser.add_argument("--epsilon-frac", type=float, default=0.005,
        dest="epsilon_frac",
        help="Polygon simplification fraction (default: 0.005)")
    parser.add_argument("--max-points", type=int, default=40,
        dest="max_points",
        help="Maximum points per polygon (default: 40)")
    parser.add_argument("--out",      default=None,
        help="Output directory (default: same as input)")
    parser.add_argument("--preview",  action="store_true",
        help="Save preview images (original | mask | overlay)")
    parser.add_argument("--debug-temp", action="store_true",
        dest="debug_temp",
        help="Save temporary preprocessing images in output/_debug")
    args = parser.parse_args()

    input_path = Path(args.input)

    # Collect images
    if input_path.is_dir():
        images = sorted(
            [p for p in input_path.glob("*.jpg")   if "_preview" not in p.stem] +
            [p for p in input_path.glob("*.jpeg")  if "_preview" not in p.stem] +
            [p for p in input_path.glob("*.png")   if "_preview" not in p.stem]
        )
        out_dir = Path(args.out) if args.out else input_path
    elif input_path.is_file():
        images  = [input_path]
        out_dir = Path(args.out) if args.out else input_path.parent
    else:
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Input    : {input_path}")
    print(f"  Images   : {len(images)}")
    print(f"  Output   : {out_dir}")
    print(f"  Label    : {args.label}")
    print(f"  Threshold: {args.thresh}")
    print(f"  Adaptive : {args.adaptive_thresh}")
    print(f"  Min area : {args.min_area} px")
    print(f"  Epsilon  : {args.epsilon_frac}")
    print(f"  Max pts  : {args.max_points}")
    print(f"  Horizon  : {'auto' if args.horizon is None else args.horizon}")
    print(f"  Preview  : {args.preview}")
    print(f"  Debug    : {args.debug_temp}")
    print()

    total_shapes = 0
    for img_path in images:
        n = process_image(img_path, args, out_dir)
        if n:
            total_shapes += n

    print(f"\n  Done - {len(images)} images, {total_shapes} total shapes")
    print(f"  Open in LabelMe: labelme {out_dir}")
    print()

if __name__ == "__main__":
    main()