#!/usr/bin/env python3
"""
5_export.py -- Export a trained SegFormer checkpoint to ONNX.

The exported model accepts a single batched input and returns the segmentation
mask as class indices (argmax applied inside the graph).

Usage:
    python3 5_export.py --checkpoint checkpoints/best.pth [options]

Options:
    --config PATH      Path to config.yaml (default: config.yaml)
    --checkpoint PATH  Checkpoint to export (default: checkpoints/best.pth)
    --out PATH         Output ONNX file (default: exports/model.onnx)
    --batch N          Static batch size baked into the graph (default: 1)
    --simplify         Run onnx-simplifier after export (requires onnxsim)
    --verify           Run a forward pass to verify ONNX output matches PyTorch
    --opset N          ONNX opset version (default: from config, fallback 12)

After export you can convert to TensorRT on the Jetson with:
    trtexec --onnx=exports/model.onnx \
            --saveEngine=exports/model.trt \
            --fp16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import SegformerForSemanticSegmentation

from config import PIPELINE_CONFIG


# -- Config --------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# -- Wrapper model -------------------------------------------------------------

class SegFormerExportWrapper(nn.Module):
    """
    Thin wrapper that:
      1. Runs the SegFormer backbone + head
      2. Upsamples logits to full input resolution
      3. Returns argmax class indices as int64

    This makes the ONNX graph self-contained: the consumer only needs to pass
    an image tensor and receives a segmentation mask directly.
    """

    def __init__(self, model: nn.Module, out_h: int, out_w: int):
        super().__init__()
        self.model = model
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        logits  = outputs.logits
        logits  = nn.functional.interpolate(
            logits,
            size=(self.out_h, self.out_w),
            mode="bilinear",
            align_corners=False,
        )
        return logits.argmax(dim=1)


# -- Load model from checkpoint ------------------------------------------------

def load_model(checkpoint_path: Path) -> tuple:
    """
    Returns (wrapped_model, cfg) reconstructed from the checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["cfg"]

    model_cfg = cfg["model"]
    num_cls   = cfg["num_classes"]
    id2label  = {int(k): v for k, v in model_cfg["id2label"].items()}
    label2id  = model_cfg["label2id"]

    base_model = SegformerForSemanticSegmentation.from_pretrained(
        model_cfg["name"],
        num_labels=num_cls,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    base_model.load_state_dict(ckpt["model"])
    base_model.eval()

    out_h   = cfg["image"]["model_h"]
    out_w   = cfg["image"]["model_w"]
    wrapped = SegFormerExportWrapper(base_model, out_h, out_w)
    wrapped.eval()

    print(f"  Loaded  : {model_cfg['name']}")
    print(f"  Epoch   : {ckpt.get('epoch', '?')}")
    print(f"  Val mIoU: {ckpt.get('best_miou', 0.0):.4f}")
    print(f"  Input   : 3 x {out_h} x {out_w}")
    print(f"  Classes : {num_cls}")
    return wrapped, cfg


# -- ONNX export ---------------------------------------------------------------

def export_onnx(model: nn.Module, cfg: dict,
                out_path: Path, batch_size: int, opset: int, dynamo=True):
    """
    Export the wrapped model to ONNX with dynamic batch size.
    """
    out_h = cfg["image"]["model_h"]
    out_w = cfg["image"]["model_w"]

    # batch_size is fixed to 1 for TRT 8.2.1 static batch compatibility.
    # Ignore the --batch argument when targeting TRT 8.2.1.
    if opset <= 11 and batch_size != 1:
        print(f"  Warning: opset {opset} does not support dynamic batch size.")
        print(f"  Ignoring --batch {batch_size} and using batch_size=1 for export.")
        batch_size = 1

    dummy = torch.zeros(batch_size, 3, out_h, out_w)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Exporting to ONNX (opset {opset}) ...")
    # Use legacy exporter (dynamo=False) with static batch=1 and opset<=11
    # for TensorRT 8.2.1 compatibility on Jetson Nano.
    # - opset 11: LayerNorm is decomposed into primitives (ReduceMean, Sub,
    #             Pow, Add, Sqrt, Div, Mul) which TRT 8.2.1 supports natively.
    # - static batch=1: avoids dynamic shape issues in TRT 8.2.1.
    # - dynamo=False: uses the stable trace-based exporter of PyTorch 1.10.
    if opset <= 11:
        print("  Using legacy ONNX exporter for opset <= 11 compatibility.")
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            opset_version=opset,
            input_names=["pixel_values"],
            output_names=["segmentation_mask"],
            do_constant_folding=True,
        )

    else:
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            opset_version=opset,
            input_names=["pixel_values"],
            output_names=["segmentation_mask"],
            dynamic_axes={
                "pixel_values":     {0: "batch_size"},
                "segmentation_mask":{0: "batch_size"},
            },
            do_constant_folding=True,
            dynamo=dynamo
        )

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved   : {out_path.resolve()}  ({size_mb:.1f} MB)")


# -- Optional: simplify --------------------------------------------------------

def simplify_onnx(onnx_path: Path) -> Path:
    """
    Run onnxsim to fold constants and clean up the graph.
    Saves the simplified model as <name>_simplified.onnx.
    """
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("  onnxsim not installed -- skipping simplification")
        print("  Install with: pip install onnxsim onnx")
        return onnx_path

    print("  Simplifying ONNX graph ...")
    model_onnx = onnx.load(str(onnx_path))
    model_simplified, ok = simplify(model_onnx)

    if ok:
        out = onnx_path.with_name(onnx_path.stem + "_simplified.onnx")
        onnx.save(model_simplified, str(out))
        size_mb = out.stat().st_size / 1024 / 1024
        print(f"  Simplified: {out.resolve()}  ({size_mb:.1f} MB)")
        return out
    else:
        print("  Simplification failed -- using original ONNX")
        return onnx_path


# -- Optional: verify ----------------------------------------------------------

def verify_onnx(onnx_path: Path, pytorch_model: nn.Module,
                cfg: dict, atol: float = 1e-3):
    """
    Run a random input through both the PyTorch model and the ONNX runtime,
    then check that the outputs match.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed -- skipping verification")
        print("  Install with: pip install onnxruntime  (or onnxruntime-gpu)")
        return

    out_h = cfg["image"]["model_h"]
    out_w = cfg["image"]["model_w"]

    x     = torch.randn(1, 3, out_h, out_w)

    # PyTorch output
    with torch.no_grad():
        pt_out = pytorch_model(x).cpu().numpy()

    # ONNX runtime output
    sess  = ort.InferenceSession(str(onnx_path),
                                  providers=["CPUExecutionProvider"])
    inp   = {sess.get_inputs()[0].name: x.numpy()}
    ort_out = sess.run(None, inp)[0]

    match = np.allclose(pt_out, ort_out, atol=atol)
    if match:
        print(f"  Verification: PASSED  (max diff = "
              f"{np.abs(pt_out - ort_out).max():.2e})")
    else:
        print(f"  Verification: FAILED  (max diff = "
              f"{np.abs(pt_out - ort_out).max():.2e})")
        print("  This may indicate a numerical issue with the ONNX export.")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export SegFormer checkpoint to ONNX")
    parser.add_argument("--config",     default=f"{PIPELINE_CONFIG}")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--out",        default=None,
                        help="Output ONNX path (default: exports/model.onnx)")
    parser.add_argument("--batch",      type=int, default=1,
                        help="Static batch size for the dummy input (default: 1)")
    parser.add_argument("--simplify",   action="store_true",
                        help="Run onnx-simplifier after export")
    parser.add_argument("--verify",     action="store_true",
                        help="Verify ONNX output against PyTorch")
    parser.add_argument("--opset",      type=int, default=None,
                        help="ONNX opset version (default: from config)")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    ckpt = Path(args.checkpoint)

    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}")
        sys.exit(1)

    opset = args.opset or cfg.get("export", {}).get("opset_version", 12)

    out_path = Path(args.out) if args.out else Path(cfg["paths"]["exports"]) / "model.onnx"

    print(f"\n  Checkpoint : {ckpt}")
    print(f"  Output     : {out_path}")
    print(f"  Batch size : {args.batch}")
    print(f"  Opset      : {opset}")

    model, ckpt_cfg = load_model(ckpt)

    export_onnx(model, ckpt_cfg, out_path, args.batch, opset)

    onnx_path = out_path
    if args.simplify:
        onnx_path = simplify_onnx(out_path)

    if args.verify:
        print("\n  Verifying ONNX output against PyTorch ...")
        verify_onnx(onnx_path, model, ckpt_cfg)

    print("\n  Export complete.")
    print(f"  To convert to TensorRT on the Jetson:")
    print(f"    trtexec --onnx={onnx_path} \\")
    print(f"            --saveEngine=exports/model.trt \\")
    print(f"            --fp16")
    print()


if __name__ == "__main__":
    main()