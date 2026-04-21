#!/usr/bin/env python3
"""
2_dataset.py -- PyTorch Dataset and DataLoaders for lane segmentation.

Expects the folder structure produced by 1_prepare_dataset.py:
    data/dataset/
        train/images/   *.jpg
        train/masks/    *.png
        val/images/
        val/masks/
        test/images/
        test/masks/

Usage (standalone test):
    python3 2_dataset.py --config config.yaml
    Prints dataset stats and saves a batch preview to data/dataset/batch_preview.jpg
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import PIPELINE_CONFIG


# -- Load config ---------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# -- Augmentation pipelines ----------------------------------------------------

def get_train_transforms(cfg: dict) -> A.Compose:
    """
    Augmentation applied only during training.
    Geometric transforms are applied to both image and mask.
    Photometric transforms are applied to image only.
    """
    aug      = cfg["augmentation"]
    h        = cfg["image"]["model_h"]
    w        = cfg["image"]["model_w"]
    sh_lim   = aug["shift_limit"]
    rot_lim  = aug["rotate_limit"]

    return A.Compose([
        # Geometric -- applied to image AND mask
        A.HorizontalFlip(p=aug["hflip_prob"]),
        A.Affine(
            translate_percent={"x": (-sh_lim, sh_lim), "y": (-sh_lim, sh_lim)},
            rotate=(-rot_lim, rot_lim),
            scale=1.0,
            p=aug["shift_prob"],
        ),

        # Photometric -- image only (mask is not affected by color ops)
        A.RandomBrightnessContrast(
            brightness_limit=aug["brightness_limit"],
            contrast_limit=aug["contrast_limit"],
            p=aug["photo_prob"],
        ),
        A.HueSaturationValue(
            hue_shift_limit=aug["hue_shift"],
            sat_shift_limit=aug["sat_shift"],
            val_shift_limit=aug["val_shift"],
            p=aug["hue_prob"],
        ),
        A.GaussianBlur(
            blur_limit=(3, aug["blur_limit"]),
            p=aug["blur_prob"],
        ),
        A.GaussNoise(
            std_range=(0, aug["noise_var"]),
            p=aug["noise_prob"],
        ),
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),  # only on bottom half (road area)
            p=aug["shadow_prob"],
        ),

        # Normalise and convert to tensor
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(cfg: dict) -> A.Compose:
    """
    Validation / test transforms: normalise only, no augmentation.
    """
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# -- Dataset -------------------------------------------------------------------

class LaneDataset(Dataset):
    """
    Loads (image, mask) pairs from a split folder.

    Args:
        split_dir:  Path to train/, val/, or test/ folder.
        transform:  Albumentations Compose pipeline.
        cfg:        Full config dict (used for class count etc.).
    """

    def __init__(self, split_dir: Path, transform: A.Compose, cfg: dict):
        self.split_dir  = Path(split_dir)
        self.transform  = transform
        self.cfg        = cfg
        self.num_classes = cfg["num_classes"]

        self.image_dir = self.split_dir / "images"
        self.mask_dir  = self.split_dir / "masks"

        self.image_paths = sorted(self.image_dir.glob("*.jpg")) + \
                           sorted(self.image_dir.glob("*.jpeg")) + \
                           sorted(self.image_dir.glob("*.png"))
        self.image_paths = sorted(self.image_paths)

        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}. "
                "Run 1_prepare_dataset.py first.")

        # Verify masks exist for all images
        missing = []
        for img_path in self.image_paths:
            mask_path = self.mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                missing.append(img_path.name)
        if missing:
            raise FileNotFoundError(
                f"Missing masks for {len(missing)} images: {missing[:5]} ...")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path  = self.image_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        # Load image as RGB (albumentations expects RGB)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise IOError(f"Cannot read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load mask as single-channel uint8 (values 0..num_classes-1)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise IOError(f"Cannot read mask: {mask_path}")

        # Apply transforms
        transformed = self.transform(image=img_rgb, mask=mask)
        image = transformed["image"]          # float32 tensor C x H x W
        mask  = transformed["mask"].long()    # int64 tensor H x W

        return {
            "image":    image,
            "mask":     mask,
            "filename": img_path.name,
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights from the config.
        Returns a float tensor of shape (num_classes,).
        Falls back to uniform weights if not set in config.
        """
        weights = self.cfg["training"].get("class_weights")
        if weights is None:
            return torch.ones(self.num_classes)
        return torch.tensor(weights, dtype=torch.float32)


# -- DataLoader factory --------------------------------------------------------

def get_dataloaders(cfg: dict) -> dict[str, DataLoader]:
    """
    Build and return train, val, test DataLoaders.
    """
    dataset_dir = Path(cfg["paths"]["dataset"])
    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    train_transform = get_train_transforms(cfg)
    val_transform   = get_val_transforms(cfg)

    train_ds = LaneDataset(dataset_dir / "train", train_transform, cfg)
    val_ds   = LaneDataset(dataset_dir / "val",   val_transform,   cfg)
    test_ds  = LaneDataset(dataset_dir / "test",  val_transform,   cfg)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_dl, "val": val_dl, "test": test_dl}


# -- Visualisation helper ------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization and convert to uint8 HxWxC numpy array.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).numpy()
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)


def mask_to_color(mask: np.ndarray, class_colors: list) -> np.ndarray:
    """Convert a 2D class-index mask to an RGB color image."""
    h, w   = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(class_colors):
        colored[mask == cls_id] = color  # config stores RGB
    return colored


def save_batch_preview(batch: dict, cfg: dict, out_path: Path,
                       max_items: int = 4):
    """
    Save a grid of (image | mask | overlay) for up to max_items batch items.
    """
    class_colors = cfg["class_colors"]
    images  = batch["image"]
    masks   = batch["mask"]
    n       = min(max_items, images.shape[0])
    rows    = []

    for i in range(n):
        img_np   = denormalize(images[i])
        mask_np  = masks[i].numpy().astype(np.uint8)
        colored  = mask_to_color(mask_np, class_colors)
        overlay  = cv2.addWeighted(img_np, 0.6, colored, 0.4, 0)

        row = np.hstack([img_np, colored, overlay])
        rows.append(row)

    grid = np.vstack(rows)
    # Convert RGB to BGR for OpenCV save
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  Batch preview saved: {out_path}")


# -- Standalone test -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test dataset loading and augmentation")
    parser.add_argument("--config", default=f"{PIPELINE_CONFIG}")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_dir = Path(cfg["paths"]["dataset"])

    print(f"\n  Config      : {args.config}")
    print(f"  Dataset dir : {dataset_dir}\n")

    # Build datasets for all splits
    val_transform   = get_val_transforms(cfg)
    train_transform = get_train_transforms(cfg)

    results = {}
    for split_name, transform in [("train", train_transform),
                                   ("val",   val_transform),
                                   ("test",  val_transform)]:
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
            print(f"  {split_name:5s}: not found (skip)")
            continue
        try:
            ds = LaneDataset(split_dir, transform, cfg)
            results[split_name] = ds
            print(f"  {split_name:5s}: {len(ds)} images")
        except FileNotFoundError as e:
            print(f"  {split_name:5s}: ERROR -- {e}")

    if not results:
        print("\n  No splits found. Run 1_prepare_dataset.py first.")
        return

    # Load one batch from the largest split and verify shapes
    split_name = max(results, key=lambda k: len(results[k]))
    ds = results[split_name]
    dl = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=True)
    batch = next(iter(dl))

    print(f"\n  Batch from '{split_name}':")
    print(f"    image shape : {batch['image'].shape}  dtype={batch['image'].dtype}")
    print(f"    mask shape  : {batch['mask'].shape}   dtype={batch['mask'].dtype}")
    print(f"    mask values : {batch['mask'].unique().tolist()}")
    print(f"    filenames   : {batch['filename']}")

    # Class weights
    weights = ds.get_class_weights()
    print(f"\n  Class weights: {weights.tolist()}")

    # Save batch preview
    preview_path = dataset_dir / "batch_preview.jpg"
    save_batch_preview(batch, cfg, preview_path)

    print("\n  Dataset OK.\n")


if __name__ == "__main__":
    main()