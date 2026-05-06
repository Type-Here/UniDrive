#!/usr/bin/env python3
"""
1_prepare_dataset.py - Convert LabelMe JSONs to segmentation masks,
                        crop, resize and split into train/val/test.

Usage:
    python3 1_prepare_dataset.py [--config config.yaml] [--preview N]

Options:
    --config PATH    Path to config.yaml (default: config.yaml)
    --preview N      Save N random preview images (original | mask | overlay)
                     to dataset/previews/. Set to 0 to skip. Default: 5.
    --dry-run        Print stats without writing any files.

Output structure:
    data/dataset/
    ├-- train/images/   *.jpg  (cropped + resized)
    ├-- train/masks/    *.png  (uint8, values 0-4)
    ├-- val/images/
    ├-- val/masks/
    ├-- test/images/
    ├-- test/masks/
    └-- previews/       *.jpg  (original | mask colored | overlay)
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from config import ROOT_DIR, PIPELINE_CONFIG


# -- Load config ---------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# -- JSON -> mask ---------------------------------------------------------------

# Rendering priority: classes with higher value are drawn last (overwrite lower).
RENDER_ORDER = ["background", "road", "lane_marking", "lane_dashed", "zebra"]


def json_to_mask(json_path: Path, class_map: dict, h: int, w: int) -> np.ndarray:
    """
    Convert a LabelMe JSON file to a uint8 segmentation mask.

    Shapes are rendered in RENDER_ORDER so that precise classes
    (lane_marking, zebra) overwrite coarser ones (road) where they overlap.

    Returns:
        mask  - np.ndarray shape (h, w) dtype uint8, values 0..num_classes-1
    """
    with open(json_path) as f:
        data = json.load(f)

    mask = np.zeros((h, w), dtype=np.uint8)

    # Group shapes by label for ordered rendering
    shapes_by_label: dict[str, list] = {label: [] for label in RENDER_ORDER}
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label in shapes_by_label:
            shapes_by_label[label].append(shape)
        else:
            print(f"  WARNING: unknown label '{label}' in {json_path.name} - skipped")

    for label in RENDER_ORDER:
        class_id = class_map.get(label, 0)
        for shape in shapes_by_label[label]:
            pts = shape.get("points", [])
            if len(pts) < 3:
                continue
            # Convert to int32 array required by fillPoly
            poly = np.array([[round(x), round(y)] for x, y in pts],
                            dtype=np.int32)
            cv2.fillPoly(mask, [poly], color=class_id)

    return mask


# -- Preprocessing -------------------------------------------------------------

def preprocess(img: np.ndarray, mask: np.ndarray,
               crop_top_frac: float,
               out_h: int, out_w: int):
    """
    1. Crop top fraction (noise: wall, ceiling, people)
    2. Resize to model input size

    The same crop and resize is applied identically to image and mask.
    Mask uses INTER_NEAREST to preserve class IDs exactly.
    """
    h = img.shape[0]
    crop_top = math.floor(h * crop_top_frac)

    img_cropped  = img[crop_top:, :]
    mask_cropped = mask[crop_top:, :]

    img_resized  = cv2.resize(img_cropped,  (out_w, out_h),
                              interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_cropped, (out_w, out_h),
                              interpolation=cv2.INTER_NEAREST)

    return img_resized, mask_resized


# -- Preview -------------------------------------------------------------------

def make_preview(img: np.ndarray,
                 mask: np.ndarray,
                 class_colors: list) -> np.ndarray:
    """
    Return a side-by-side preview: original | coloured mask | overlay.
    """
    h, w = img.shape[:2]

    # Coloured mask
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(class_colors):
        colored[mask == cls_id] = color  # config stores RGB

    # Overlay (60% image, 40% mask)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) \
        if img.shape[2] == 3 else img
    overlay = cv2.addWeighted(img_rgb, 0.6, colored, 0.4, 0)

    panel = np.hstack([img_rgb, colored, overlay])
    return cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)


# -- Dataset stats -------------------------------------------------------------

def compute_stats(masks: list[np.ndarray], num_classes: int) -> dict:
    """Compute per-class pixel counts and frequencies."""
    counts = np.zeros(num_classes, dtype=np.int64)
    total  = 0
    for mask in masks:
        for cls_id in range(num_classes):
            counts[cls_id] += int((mask == cls_id).sum())
        total += mask.size
    freq = counts / total if total > 0 else counts
    return {"counts": counts, "frequencies": freq, "total_pixels": total}


# -- Split ---------------------------------------------------------------------

# Known illumination condition prefixes.
# Images not matching any prefix are grouped as "other".
ILLUMINATION_PREFIXES = ("normal_", "reflex_", "night_")


def split_files(files: list[Path], train_frac: float,
                val_frac: float, seed: int):
    """
    Stratified split by illumination condition.
    Files are grouped by their name prefix (normal_, reflex_, night_).
    The split fractions are applied independently within each group
    so every condition is proportionally represented in train/val/test.
    """
    rng = random.Random(seed)

    # Group files by illumination prefix
    groups: dict[str, list[Path]] = {}
    for f in files:
        matched = False
        for prefix in ILLUMINATION_PREFIXES:
            if f.name.startswith(prefix):
                groups.setdefault(prefix, []).append(f)
                matched = True
                break
        if not matched:
            groups.setdefault("other", []).append(f)

    print(f"  Illumination groups found:")
    for group, members in sorted(groups.items()):
        print(f"    {group:<12}: {len(members)} images")

    train, val, test = [], [], []

    for group, members in groups.items():
        rng.shuffle(members)
        n = len(members)
        n_train = math.floor(n * train_frac)
        n_val = math.floor(n * val_frac)
        # Ensure at least 1 image per split when group is large enough
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
        train += members[:n_train]
        val += members[n_train:n_train + n_val]
        test += members[n_train + n_val:]

    # Shuffle the final lists so groups are interleaved during training
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare lane segmentation dataset from LabelMe JSONs")
    parser.add_argument("--config",  default=f"{PIPELINE_CONFIG}")
    parser.add_argument("--preview", type=int, default=5,
                        help="Number of preview images to save (0 = skip)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing files")
    args = parser.parse_args()

    if ROOT_DIR is None:
        print("ERROR: ROOT_DIR is not set. Please set it in config.py")
        sys.exit(1)
    else:
        print(f"\n  ROOT_DIR: {ROOT_DIR}\n")
        root_dir = Path(ROOT_DIR)
    print(f"pipeline_config_file: {PIPELINE_CONFIG}\n")
    cfg = load_config(args.config)

    raw_img_dir   = root_dir / Path(cfg["paths"]["raw_images"])
    raw_label_dir = root_dir / Path(cfg["paths"]["raw_labels"])
    dataset_dir   = root_dir / Path(cfg["paths"]["dataset"])

    class_map    = cfg["classes"]
    num_classes  = cfg["num_classes"]
    class_colors = cfg["class_colors"]

    orig_h       = cfg["image"]["original_h"]
    orig_w       = cfg["image"]["original_w"]
    crop_top     = cfg["image"]["crop_top_frac"]
    out_h        = cfg["image"]["model_h"]
    out_w        = cfg["image"]["model_w"]

    train_frac   = cfg["split"]["train"]
    val_frac     = cfg["split"]["val"]
    seed         = cfg["split"]["seed"]

    # -- Discover paired files -------------------------------------------------
    json_files = sorted(raw_label_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: no JSON files found in {raw_label_dir}")
        sys.exit(1)

    # Match each JSON to its image (same stem, any image extension)
    pairs = []
    missing_imgs = []
    for jf in json_files:
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = raw_img_dir / (jf.stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            missing_imgs.append(jf.stem)
        else:
            pairs.append((img_path, jf))

    print(f"\n  Found {len(pairs)} image-JSON pairs")
    if missing_imgs:
        print(f"  WARNING: no image found for {len(missing_imgs)} JSONs:")
        for s in missing_imgs:
            print(f"    {s}")

    if not pairs:
        print("ERROR: no valid pairs found. Check paths in config.yaml")
        sys.exit(1)

    # -- Split -----------------------------------------------------------------
    img_paths = [p[0] for p in pairs]
    train_imgs, val_imgs, test_imgs = split_files(
        img_paths, train_frac, val_frac, seed)

    img_to_json = {p[0]: p[1] for p in pairs}

    splits = {
        "train": train_imgs,
        "val":   val_imgs,
        "test":  test_imgs,
    }

    print(f"\n  Split (seed={seed}):")
    for split_name, imgs in splits.items():
        print(f"    {split_name:5s}: {len(imgs):3d} images")

    crop_px = math.floor(orig_h * crop_top)
    print(f"\n  Preprocessing:")
    print(f"    Original  : {orig_w}x{orig_h}")
    print(f"    Crop top  : {crop_px}px  ({crop_top*100:.0f}%)")
    print(f"    After crop: {orig_w}x{orig_h - crop_px}")
    print(f"    Model input: {out_w}x{out_h}")

    if args.dry_run:
        print("\n  Dry run - no files written.")
        return

    # -- Create directories ----------------------------------------------------
    for split_name in splits:
        (dataset_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split_name / "masks").mkdir(parents=True,  exist_ok=True)

    preview_dir = dataset_dir / "previews"
    if args.preview > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    # Collect all masks for stats
    all_masks: dict[str, list[np.ndarray]] = {s: [] for s in splits}

    # Pick random samples for preview (across all splits)
    all_pairs = [(img, img_to_json[img], split_name)
                 for split_name, imgs in splits.items()
                 for img in imgs]
    rng = random.Random(seed)
    preview_samples = rng.sample(all_pairs, min(args.preview, len(all_pairs)))
    preview_set = {str(s[0]) for s in preview_samples}

    # -- Process images --------------------------------------------------------
    total = sum(len(v) for v in splits.values())
    done  = 0

    for split_name, img_list in splits.items():
        for img_path in img_list:
            json_path = img_to_json[img_path]

            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  WARNING: cannot read {img_path} - skipped")
                continue

            # Generate mask from JSON
            mask = json_to_mask(json_path, class_map, orig_h, orig_w)

            # Crop + resize
            img_pp, mask_pp = preprocess(img, mask, crop_top, out_h, out_w)

            # Save image
            out_img = dataset_dir / split_name / "images" / img_path.name
            cv2.imwrite(str(out_img), img_pp,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save mask as PNG (lossless, values 0-4)
            out_mask = dataset_dir / split_name / "masks" / (img_path.stem + ".png")
            cv2.imwrite(str(out_mask), mask_pp)

            # Save preview
            if str(img_path) in preview_set:
                preview = make_preview(img_pp, mask_pp, class_colors)
                cv2.imwrite(
                    str(preview_dir / (img_path.stem + "_preview.jpg")),
                    preview)

            all_masks[split_name].append(mask_pp)

            done += 1
            print(f"  [{done:3d}/{total}] {split_name:5s}  {img_path.name}",
                  end="\r")

    print(f"\n\n  All {done} images processed.")

    # -- Per-split stats -------------------------------------------------------
    class_names = {v: k for k, v in class_map.items()}
    print()
    for split_name, masks in all_masks.items():
        if not masks:
            continue
        stats = compute_stats(masks, num_classes)
        print(f"  -- {split_name} ({len(masks)} images) --")
        for cls_id in range(num_classes):
            name  = class_names.get(cls_id, str(cls_id))
            freq  = stats["frequencies"][cls_id]
            #count = stats["counts"][cls_id]
            bar   = "█" * int(freq * 40)
            print(f"    {cls_id} {name:14s}  {freq*100:5.1f}%  {bar}")
        print()

    # -- Class weight suggestion -----------------------------------------------
    all_train_masks = all_masks["train"]
    if all_train_masks:
        stats  = compute_stats(all_train_masks, num_classes)
        freqs  = stats["frequencies"]
        # Inverse frequency, normalized so median weight = 1
        inv    = np.where(freqs > 0, 1.0 / (freqs + 1e-6), 0.0)
        med    = np.median(inv[inv > 0]) if (inv > 0).any() else 1.0
        weights = np.round(inv / med, 2)
        print("  Suggested class_weights (inverse freq, median-normalised):")
        print(f"  {list(weights)}")
        print("  -> Copy this into config.yaml -> training -> class_weights")

    print(f"\n  Dataset ready at: {dataset_dir.resolve()}")
    if args.preview > 0:
        print(f"  Previews at     : {preview_dir.resolve()}")
    print()


if __name__ == "__main__":
    main()