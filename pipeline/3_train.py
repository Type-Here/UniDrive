#!/usr/bin/env python3
"""
3_train.py -- Fine-tuning SegFormer for lane segmentation.

Expects the dataset prepared by 1_prepare_dataset.py and
the Dataset/DataLoader from 2_dataset.py.

Usage:
    python3 3_train.py [--config config.yaml] [--resume checkpoints/last.pth]

Options:
    --config PATH    Path to config.yaml (default: config.yaml)
    --resume PATH    Resume from a checkpoint file
    --model NAME     Override model name from config (e.g. nvidia/mit-b0)
    --epochs N       Override number of epochs from config
    --batch N        Override batch size from config

Outputs:
    checkpoints/best.pth   -- best val mIoU checkpoint
    checkpoints/last.pth   -- latest epoch checkpoint
    logs/train_log.csv     -- per-epoch metrics
    logs/train_log.png     -- loss + mIoU curves
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sympy.physics.units import current
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR
from transformers import SegformerForSemanticSegmentation

# Local modules -- 2_dataset.py cannot be imported directly because Python
# module names cannot start with a digit. We use importlib to load it.
import importlib.util as _ilu

from config import PIPELINE_CONFIG


def _load_dataset_module():
    _here = Path(__file__).parent
    for candidate in ("dataset.py", "2_dataset.py"):
        _p = _here / candidate
        if _p.exists():
            _spec = _ilu.spec_from_file_location("dataset", _p)
            _mod  = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            return _mod
    raise ImportError(
        "Cannot find dataset.py or 2_dataset.py in the same folder as 3_train.py")

_ds_mod          = _load_dataset_module()
get_dataloaders  = _ds_mod.get_dataloaders
LaneDataset      = _ds_mod.LaneDataset


# -- Config --------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# -- Metrics -------------------------------------------------------------------

class SegmentationMetrics:
    """
    Accumulates confusion matrix over batches and computes:
        - per-class IoU
        - mean IoU (ignoring background by default)
        - pixel accuracy
    """

    def __init__(self, num_classes: int, ignore_bg: bool = False):
        self.num_classes = num_classes
        self.ignore_bg   = ignore_bg
        self.reset()

    def reset(self):
        self.confusion = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        preds:  (N, H, W) int64 -- predicted class indices
        labels: (N, H, W) int64 -- ground truth class indices
        """
        preds  = preds.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()

        valid = (labels >= 0) & (labels < self.num_classes)
        preds  = preds[valid]
        labels = labels[valid]

        np.add.at(self.confusion,
                  (labels, preds),
                  1)

    def compute(self) -> dict:
        conf  = self.confusion
        iou   = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = conf[c, c]
            fp = conf[:, c].sum() - tp
            fn = conf[c, :].sum() - tp
            denom = tp + fp + fn
            iou[c] = tp / denom if denom > 0 else 0.0

        start = 1 if self.ignore_bg else 0
        valid_iou = iou[start:]
        mean_iou  = float(np.mean(valid_iou[valid_iou > 0])
                          if (valid_iou > 0).any() else 0.0)

        total   = conf.sum()
        correct = np.diag(conf).sum()
        pixel_acc = float(correct / total) if total > 0 else 0.0

        return {
            "mean_iou":   mean_iou,
            "per_class_iou": iou.tolist(),
            "pixel_acc":  pixel_acc,
        }


# -- Model factory -------------------------------------------------------------

def build_model(cfg: dict) -> SegformerForSemanticSegmentation:
    model_cfg = cfg["model"]
    num_cls   = cfg["num_classes"]

    id2label = {int(k): v for k, v in model_cfg["id2label"].items()}
    label2id = model_cfg["label2id"]

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_cfg["name"],
        num_labels=num_cls,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


# -- Loss ----------------------------------------------------------------------

def build_criterion(cfg: dict, device: torch.device) -> nn.CrossEntropyLoss:
    weights = cfg["training"].get("class_weights")
    if weights is not None:
        w = torch.tensor(weights, dtype=torch.float32).to(device)
    else:
        w = None
    return nn.CrossEntropyLoss(weight=w, ignore_index=255)


# -- Scheduler -----------------------------------------------------------------

def build_scheduler(optimizer, cfg: dict):
    tr      = cfg["training"]
    sched   = tr.get("scheduler", "cosine")
    epochs  = tr["epochs"]
    warmup  = tr.get("warmup_epochs", 5)

    if sched == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup,
            eta_min=1e-7,
        )
    else:
        main_scheduler = PolynomialLR(
            optimizer,
            total_iters=epochs - warmup,
            power=0.9,
        )
    return main_scheduler


# -- Checkpoint ----------------------------------------------------------------

def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model, optimizer, scaler) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


# -- Logging -------------------------------------------------------------------

class Logger:
    """Writes per-epoch metrics to a CSV and plots curves at the end."""

    FIELDS = [
        "epoch", "train_loss", "val_loss",
        "train_miou", "val_miou",
        "train_pxacc", "val_pxacc",
        "lr", "epoch_time_s",
    ]

    def __init__(self, log_dir: Path):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "train_log.csv"
        self.rows     = []

        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, row: dict):
        self.rows.append(row)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow({k: row.get(k, "") for k in self.FIELDS})

    def plot(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        epochs     = [r["epoch"]      for r in self.rows]
        train_loss = [r["train_loss"] for r in self.rows]
        val_loss   = [r["val_loss"]   for r in self.rows]
        train_iou  = [r["train_miou"] for r in self.rows]
        val_iou    = [r["val_miou"]   for r in self.rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, train_loss, label="train")
        ax1.plot(epochs, val_loss,   label="val")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, train_iou, label="train")
        ax2.plot(epochs, val_iou,   label="val")
        ax2.set_title("Mean IoU")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(str(self.log_dir / "train_log.png"), dpi=120)
        plt.close()


# -- One epoch -----------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer,
              scaler, metrics, device, cfg,
              is_train: bool) -> dict:
    """
    Run one training or validation epoch.
    Returns a dict with loss, mean_iou, pixel_acc.
    """
    model.train() if is_train else model.eval()
    metrics.reset()

    total_loss = 0.0
    n_batches  = 0
    use_amp    = cfg["training"].get("amp", True) and device.type == "cuda"

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device,  non_blocking=True)

            with autocast(device.type, enabled=use_amp):
                outputs = model(pixel_values=images)
                # SegFormer outputs logits at 1/4 resolution -- upsample to mask size
                logits  = outputs.logits
                logits  = nn.functional.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            preds = logits.argmax(dim=1)
            metrics.update(preds, masks)

            total_loss += loss.item()
            n_batches  += 1

    result = metrics.compute()
    result["loss"] = total_loss / max(n_batches, 1)
    return result


# -- Main training loop --------------------------------------------------------

def train(cfg: dict, resume_path: str = None):
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")

    # Paths
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    log_dir  = Path(cfg["paths"]["logs"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("  Loading data...")
    loaders    = get_dataloaders(cfg)
    train_dl   = loaders["train"]
    val_dl     = loaders["val"]
    num_classes = cfg["num_classes"]

    print(f"  Train batches: {len(train_dl)}"
          f"  ({len(train_dl.dataset)} images)")
    print(f"  Val batches  : {len(val_dl)}"
          f"  ({len(val_dl.dataset)} images)")

    # Model
    print(f"  Loading model: {cfg['model']['name']} ...")
    model     = build_model(cfg).to(device)
    criterion = build_criterion(cfg, device) # Loss

    # Optimizer
    tr     = cfg["training"]
    optim  = AdamW(model.parameters(),
                   lr=tr["lr"],
                   weight_decay=tr["weight_decay"])
    scaler = GradScaler(enabled=tr.get("amp", True) and device.type == "cuda")

    epochs      = tr["epochs"]
    warmup_ep   = tr.get("warmup_epochs", 5)
    patience    = tr.get("early_stopping_patience", 15)
    save_every  = tr.get("save_every", 10)
    scheduler   = build_scheduler(optim, cfg)

    metrics     = SegmentationMetrics(num_classes, ignore_bg=False)
    logger      = Logger(log_dir)

    start_epoch     = 1
    best_miou       = 0.0
    no_improve      = 0
    class_names     = {v: k for k, v in cfg["classes"].items()}

    # Resume
    if resume_path and Path(resume_path).exists():
        print(f"  Resuming from {resume_path}")
        ckpt        = load_checkpoint(Path(resume_path), model, optim, scaler)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        print(f"  Resuming at epoch {start_epoch}, best mIoU={best_miou:.4f}")

    print(f"\n  Training for {epochs} epochs "
          f"(warmup={warmup_ep}, patience={patience})\n")

    current_epoch = start_epoch

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # Warmup: linearly ramp LR from 0 to target
        if epoch <= warmup_ep:
            for pg in optim.param_groups:
                pg["lr"] = tr["lr"] * epoch / warmup_ep
        else:
            scheduler.step()

        current_lr = optim.param_groups[0]["lr"]

        # Train
        train_res = run_epoch(model, train_dl, criterion, optim,
                              scaler, metrics, device, cfg, is_train=True)

        # Validate
        val_res = run_epoch(model, val_dl, criterion, None,
                            scaler, metrics, device, cfg, is_train=False)

        epoch_time = time.time() - t0

        # Log
        row = {
            "epoch":        epoch,
            "train_loss":   round(train_res["loss"], 5),
            "val_loss":     round(val_res["loss"], 5),
            "train_miou":   round(train_res["mean_iou"], 5),
            "val_miou":     round(val_res["mean_iou"], 5),
            "train_pxacc":  round(train_res["pixel_acc"], 5),
            "val_pxacc":    round(val_res["pixel_acc"], 5),
            "lr":           round(current_lr, 8),
            "epoch_time_s": round(epoch_time, 1),
        }
        logger.log(row)

        # Print
        print(f"  Epoch {epoch:4d}/{epochs}"
              f"  loss {train_res['loss']:.4f}/{val_res['loss']:.4f}"
              f"  mIoU {train_res['mean_iou']:.4f}/{val_res['mean_iou']:.4f}"
              f"  px {val_res['pixel_acc']:.4f}"
              f"  lr {current_lr:.2e}"
              f"  {epoch_time:.0f}s")

        # Per-class IoU every 10 epochs
        if epoch % 10 == 0:
            iou = val_res["per_class_iou"]
            print("    Val IoU per class:")
            for cid, iou_val in enumerate(iou):
                name = class_names.get(cid, str(cid))
                bar  = "#" * int(iou_val * 20)
                print(f"      {cid} {name:14s} {iou_val:.4f}  {bar}")

        # Checkpoint -- best
        val_miou = val_res["mean_iou"]
        if val_miou > best_miou:
            best_miou  = val_miou
            no_improve = 0
            save_checkpoint({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optim.state_dict(),
                "scaler":    scaler.state_dict(),
                "best_miou": best_miou,
                "cfg":       cfg,
            }, ckpt_dir / "best.pth")
            print(f"    ** New best mIoU: {best_miou:.4f} -- saved best.pth")
        else:
            no_improve += 1

        # Checkpoint -- periodic
        if epoch % save_every == 0:
            save_checkpoint({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optim.state_dict(),
                "scaler":    scaler.state_dict(),
                "best_miou": best_miou,
                "cfg":       cfg,
            }, ckpt_dir / "last.pth")

        # Early stopping
        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break
        current_epoch = epoch + 1

    # Final checkpoint
    save_checkpoint({
        "epoch":     current_epoch,
        "model":     model.state_dict(),
        "optimizer": optim.state_dict(),
        "scaler":    scaler.state_dict(),
        "best_miou": best_miou,
        "cfg":       cfg,
    }, ckpt_dir / "last.pth")

    logger.plot()

    print(f"\n  Training complete.")
    print(f"  Best val mIoU : {best_miou:.4f}")
    print(f"  Checkpoints   : {ckpt_dir.resolve()}")
    print(f"  Log           : {log_dir.resolve()}\n")


# -- Entry point ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SegFormer for lane segmentation")
    parser.add_argument("--config",  default=f"{PIPELINE_CONFIG}")
    parser.add_argument("--resume",  default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--model",   default=None,
                        help="Override model name (e.g. nvidia/mit-b0)")
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--batch",   type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch:
        cfg["training"]["batch_size"] = args.batch

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()