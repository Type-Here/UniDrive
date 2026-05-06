#!/usr/bin/env python3
"""
4_evaluate.py -- Evaluate a trained SegFormer checkpoint on val or test split.

Computes per-class IoU, mean IoU, pixel accuracy and confusion matrix.
Saves prediction visualisations for manual inspection.

Usage:
    python3 4_evaluate.py --checkpoint checkpoints/best.pth [options]

Options:
    --config PATH      Path to config.yaml (default: config.yaml)
    --checkpoint PATH  Checkpoint to evaluate (default: checkpoints/best.pth)
    --split NAME       Dataset split to evaluate on: val or test (default: test)
    --save-preds N     Save N prediction images to logs/predictions/ (default: 10)
    --batch N          Override batch size for inference (default: from config)
"""

import argparse
import importlib.util as ilu
from pathlib import Path

import cv2
import matplotlib

from config import PIPELINE_CONFIG

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import SegformerForSemanticSegmentation


# -- Config --------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# -- Load dataset module -------------------------------------------------------

def _load_dataset_module(here: Path):
    for candidate in ("dataset.py", "2_dataset.py"):
        p = here / candidate
        if p.exists():
            spec = ilu.spec_from_file_location("dataset", p)
            mod  = ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(
        "Cannot find dataset.py or 2_dataset.py next to 4_evaluate.py")


# -- Model loading -------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device):
    """
    Load model architecture and weights from a checkpoint saved by 3_train.py.
    The checkpoint stores the full cfg so we can reconstruct the model exactly.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["cfg"]

    model_cfg = cfg["model"]
    num_cls   = cfg["num_classes"]
    id2label  = {int(k): v for k, v in model_cfg["id2label"].items()}
    label2id  = model_cfg["label2id"]

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_cfg["name"],
        num_labels=num_cls,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print(f"  Loaded checkpoint: epoch={ckpt.get('epoch', '?')}"
          f"  best_miou={ckpt.get('best_miou', 0.0):.4f}")
    return model, cfg


# -- Metrics -------------------------------------------------------------------

class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        p = preds.cpu().numpy().flatten().astype(np.int64)
        l = labels.cpu().numpy().flatten().astype(np.int64)
        valid = (l >= 0) & (l < self.num_classes)
        np.add.at(self.mat, (l[valid], p[valid]), 1)

    def iou_per_class(self) -> np.ndarray:
        iou = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp    = self.mat[c, c]
            fp    = self.mat[:, c].sum() - tp
            fn    = self.mat[c, :].sum() - tp
            denom = tp + fp + fn
            iou[c] = float(tp) / denom if denom > 0 else 0.0
        return iou

    def mean_iou(self, ignore_bg: bool = False) -> float:
        iou = self.iou_per_class()
        start = 1 if ignore_bg else 0
        valid = iou[start:]
        return float(np.mean(valid[valid > 0])) if (valid > 0).any() else 0.0

    def pixel_accuracy(self) -> float:
        total   = self.mat.sum()
        correct = np.diag(self.mat).sum()
        return float(correct / total) if total > 0 else 0.0

    def precision_recall_f1(self) -> tuple:
        prec = np.zeros(self.num_classes)
        rec  = np.zeros(self.num_classes)
        f1   = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = self.mat[c, c]
            fp = self.mat[:, c].sum() - tp
            fn = self.mat[c, :].sum() - tp
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec[c] = p
            rec[c]  = r
            f1[c]   = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return prec, rec, f1


# -- Inference -----------------------------------------------------------------

def predict_batch(model, images: torch.Tensor,
                  target_size: tuple, device: torch.device,
                  model_name: str) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(pixel_values=images.to(device)) \
            if model_name in ("segformer-b0", "segformer-b1") \
            else model(images.to(device))

        if model_name in ("segformer-b0", "segformer-b1"):
            logits = outputs.logits
        else:
            logits = outputs["out"]

        logits = nn.functional.interpolate(
            logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    return logits.argmax(dim=1)


# -- Visualisation -------------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406]) # mean and std values from ImageNet
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).cpu().numpy() # [C,H,W] -> [H,W,C]
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)


def mask_to_color(mask: np.ndarray, class_colors: list) -> np.ndarray:
    h, w    = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(class_colors):
        colored[mask == cls_id] = color
    return colored


def save_prediction_grid(images, masks_gt, masks_pred,
                         filenames, class_colors, out_dir: Path,
                         max_items: int = 4):
    """
    Save a grid: original | ground truth | prediction | difference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(max_items, len(images))

    for i in range(n):
        img      = denormalize(images[i])
        gt       = masks_gt[i].cpu().numpy().astype(np.uint8)
        pred     = masks_pred[i].cpu().numpy().astype(np.uint8)

        gt_col   = mask_to_color(gt,   class_colors)
        pred_col = mask_to_color(pred, class_colors)

        # Difference map: red = false positive, blue = false negative
        diff = np.zeros((*gt.shape, 3), dtype=np.uint8)
        diff[(pred != gt) & (pred != 0)] = [255, 0, 0]    # FP -- red
        diff[(pred != gt) & (gt   != 0)] = [0,   0, 255]  # FN -- blue

        row  = np.hstack([img, gt_col, pred_col, diff])
        name = Path(filenames[i]).stem
        cv2.imwrite(str(out_dir / f"{name}_pred.jpg"),
                    cv2.cvtColor(row, cv2.COLOR_RGB2BGR))


def save_confusion_matrix_plot(cm_mat: np.ndarray,
                                class_names: list, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    n = len(class_names)

    # Normalise rows (recall)
    row_sums = cm_mat.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm_mat / row_sums, 0.0)

    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix (row-normalised)")

    for r in range(n):
        for c in range(n):
            val  = cm_norm[r, c]
            text = f"{val:.2f}"
            color = "white" if val > 0.5 else "black"
            ax.text(c, r, text, ha="center", va="center",
                    color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=130)
    plt.close()
    print(f"  Confusion matrix saved: {out_path}")


def save_iou_bar_chart(iou: np.ndarray, class_names: list, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(class_names, iou, color="#4a90d9")
    ax.set_xlim(0, 1)
    ax.set_xlabel("IoU")
    ax.set_title("Per-class IoU")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, iou):
        ax.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=130)
    plt.close()
    print(f"  IoU bar chart saved  : {out_path}")


# -- Main ----------------------------------------------------------------------

def evaluate(cfg: dict, checkpoint_path: Path, split: str,
             save_preds: int, batch_override: int = None):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n  Device     : {device}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Split      : {split}\n")

    # Load model
    model, ckpt_cfg = load_model(checkpoint_path, device)

    # Use config from checkpoint so model and data always match
    # but allow path overrides from the local config
    ckpt_cfg["paths"] = cfg["paths"]

    # Dataset
    here = Path(__file__).parent
    ds_mod = _load_dataset_module(here)

    batch_size = batch_override or ckpt_cfg["training"]["batch_size"]
    val_tf     = ds_mod.get_val_transforms(ckpt_cfg)
    dataset    = ds_mod.LaneDataset(
        Path(ckpt_cfg["paths"]["dataset"]) / split,
        val_tf, ckpt_cfg)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)

    num_classes  = ckpt_cfg["num_classes"]
    class_colors = ckpt_cfg["class_colors"]
    id2label = {int(k): v for k, v in ckpt_cfg["model"]["id2label"].items()}
    class_names = [id2label[i] for i in range(num_classes)]

    cm = ConfusionMatrix(num_classes)

    # Collect first N batches for prediction visualization
    pred_images, pred_gt, pred_masks, pred_names = [], [], [], []
    collected = 0

    print(f"  Running inference on {len(dataset)} images...")

    for batch in loader:
        images    = batch["image"]
        masks_gt  = batch["mask"]
        filenames = batch["filename"]

        target_size = tuple(masks_gt.shape[-2:])
        preds = predict_batch(model, images, target_size, device,
                              ckpt_cfg["model"]["name"])

        cm.update(preds, masks_gt)

        if collected < save_preds:
            n = min(save_preds - collected, images.shape[0])
            pred_images.extend([images[i] for i in range(n)])
            pred_gt.extend([masks_gt[i] for i in range(n)])
            pred_masks.extend([preds[i] for i in range(n)])
            pred_names.extend(filenames[:n])
            collected += n

    # Metrics
    iou      = cm.iou_per_class()
    miou     = cm.mean_iou(ignore_bg=False)
    miou_nobg = cm.mean_iou(ignore_bg=True)
    px_acc   = cm.pixel_accuracy()
    prec, rec, f1 = cm.precision_recall_f1()

    # Print results
    print(f"\n  Results on '{split}' split")
    print(f"  {'Metric':<22} {'Value':>8}")
    print(f"  {'-'*32}")
    print(f"  {'mIoU (all classes)':<22} {miou:>8.4f}")
    print(f"  {'mIoU (excl. bg)':<22} {miou_nobg:>8.4f}")
    print(f"  {'Pixel accuracy':<22} {px_acc:>8.4f}")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<16} {'IoU':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*46}")
    for c, name in enumerate(class_names):
        print(f"  {name:<16} {iou[c]:>6.4f}  {prec[c]:>6.4f}"
              f"  {rec[c]:>6.4f}  {f1[c]:>6.4f}")

    # Save outputs
    log_dir = Path(cfg["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)

    if save_preds > 0 and pred_images:
        pred_dir = log_dir / "predictions"
        save_prediction_grid(pred_images, pred_gt, pred_masks,
                             pred_names, class_colors, pred_dir,
                             max_items=save_preds)
        print(f"\n  Predictions saved  : {pred_dir.resolve()}")

    save_confusion_matrix_plot(
        cm.mat, class_names, log_dir / f"confusion_{split}.png")

    save_iou_bar_chart(
        iou, class_names, log_dir / f"iou_{split}.png")

    # Save metrics to txt
    results_path = log_dir / f"results_{split}.txt"
    with open(results_path, "w") as f:
        f.write(f"Split: {split}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        f.write(f"mIoU (all): {miou:.6f}\n")
        f.write(f"mIoU (no bg): {miou_nobg:.6f}\n")
        f.write(f"Pixel accuracy: {px_acc:.6f}\n\n")
        f.write(f"{'Class':<16} {'IoU':>8}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}\n")
        for c, name in enumerate(class_names):
            f.write(f"{name:<16} {iou[c]:>8.6f}  {prec[c]:>8.6f}"
                    f"  {rec[c]:>8.6f}  {f1[c]:>8.6f}\n")
    print(f"  Results saved      : {results_path.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a SegFormer checkpoint on val or test split")
    parser.add_argument("--config",     default=f"{PIPELINE_CONFIG}")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--split",      default="test",
                        choices=["val", "test"])
    parser.add_argument("--save-preds", type=int, default=10,
                        dest="save_preds",
                        help="Number of prediction images to save (0 = skip)")
    parser.add_argument("--batch",      type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(
        cfg,
        checkpoint_path=Path(args.checkpoint),
        split=args.split,
        save_preds=args.save_preds,
        batch_override=args.batch,
    )


if __name__ == "__main__":
    main()