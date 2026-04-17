#!/usr/bin/env python3
"""
preannotate.py - Auto-generate LabelMe JSON annotations for lane markings
=========================================================================
Uses brightness thresholding + local contrast to detect white stripes
on dark road surface, then exports contours as LabelMe polygon annotations.

The generated .json files can be opened directly in LabelMe for refinement.

Usage:
    python3 preannotate.py <image_or_folder> [options]

Arguments:
    image_or_folder     Single image or folder of images (jpg/png)

Options:
    --label NAME        Label name for annotations (default: lane_marking)
    --thresh INT        Brightness threshold 0-255 (default: 200)
    --min-area INT      Minimum blob area in pixels (default: 50)
    --horizon FLOAT     Horizon as fraction of image height 0-1 (default: auto)
    --out DIR           Output directory (default: same as input)
    --preview           Also save a preview image with overlay

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


def segment_stripes(img, horizon_frac=None, thresh_bright=200, min_area=50):
    """
    Returns a binary mask (uint8) where 255 = lane marking.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Horizon
    if horizon_frac is not None:
        horizon = int(h * horizon_frac)
    else:
        horizon = find_horizon(gray)

    # Road mask - only process pixels below horizon
    road_mask = np.zeros((h, w), np.uint8)
    road_mask[horizon:, :] = 255

    # Absolute brightness threshold
    _, bright = cv2.threshold(gray, thresh_bright, 255, cv2.THRESH_BINARY)

    # Local contrast: stripe pixels sit on dark background
    kernel    = np.ones((15, 15), np.uint8)
    local_min = cv2.erode(gray, kernel)
    contrast  = (gray.astype(int) - local_min.astype(int))
    contrast_mask = (contrast > 80).astype(np.uint8) * 255

    # Combine all masks
    mask = cv2.bitwise_and(bright,        contrast_mask)
    mask = cv2.bitwise_and(mask,          road_mask)

    # Morphological cleanup
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)

    # Remove small blobs
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    return clean, horizon


# -- Contour -> LabelMe polygon -------------------------------------------------

def mask_to_polygons(mask, epsilon_frac=0.005, max_points=40):
    """
    Convert binary mask to list of simplified polygons.
    Each polygon is a list of [x, y] points.

    epsilon_frac: contour approximation strength (fraction of perimeter).
                  Higher = simpler polygons, fewer points.
    max_points:   hard cap on points per polygon.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        epsilon   = epsilon_frac * perimeter
        approx    = cv2.approxPolyDP(cnt, epsilon, True)

        pts = approx.squeeze()
        if pts.ndim < 2 or len(pts) < 3:
            continue

        # Further reduce if too many points
        if len(pts) > max_points:
            step = len(pts) // max_points
            pts  = pts[::step]

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

    mask, horizon = segment_stripes(
        img,
        horizon_frac  = args.horizon,
        thresh_bright = args.thresh,
        min_area      = args.min_area,
    )

    polygons  = mask_to_polygons(mask)
    json_data = build_labelme_json(img_path, polygons, args.label, w, h)

    # Save JSON
    json_path = out_dir / (img_path.stem + ".json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Save preview
    if args.preview:
        prev_path = out_dir / (img_path.stem + "_preview.jpg")
        save_preview(img, mask, horizon, polygons, prev_path)

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
    parser.add_argument("--thresh",   type=int,   default=200,
        help="Brightness threshold 0-255 (default: 200)")
    parser.add_argument("--min-area", type=int,   default=50,
        dest="min_area",
        help="Min blob area in pixels (default: 50)")
    parser.add_argument("--horizon",  type=float, default=None,
        help="Horizon as fraction of height e.g. 0.45 (default: auto)")
    parser.add_argument("--out",      default=None,
        help="Output directory (default: same as input)")
    parser.add_argument("--preview",  action="store_true",
        help="Save preview images (original | mask | overlay)")
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
    print(f"  Min area : {args.min_area} px")
    print(f"  Horizon  : {'auto' if args.horizon is None else args.horizon}")
    print(f"  Preview  : {args.preview}")
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