#!/usr/bin/env python3
"""
convert_tensorrt.py -- Convert ONNX model to TensorRT engine via Python API.

Works on Jetson Nano with TensorRT 8.x installed (part of JetPack).
No trtexec command needed -- uses tensorrt Python bindings directly.

Setup (on Jetson Nano):
    # TensorRT Python bindings are installed with JetPack but may need
    # to be linked into your virtualenv/conda env:
    pip install --extra-index-url https://pypi.ngc.nvidia.com tensorrt
    # OR if already installed system-wide:
    pip install pycuda

Usage:
    python3 convert_tensorrt.py --onnx model.onnx [options]

Options:
    --onnx  PATH      Input ONNX file
    --out   PATH      Output engine file (default: model.engine)
    --fp16            Enable FP16 precision (recommended for Jetson Nano)
    --int8            Enable INT8 precision (fastest, needs calibration data)
    --workspace MB    Max GPU workspace in MB (default: 512)
    --verbose         Show detailed TensorRT build log
"""

import argparse
import sys
import time
from pathlib import Path


def build_engine(onnx_path: str, engine_path: str,
                 fp16: bool, int8: bool,
                 workspace_mb: int, verbose: bool):

    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: tensorrt not found.")
        print("On Jetson Nano, TensorRT Python bindings are in:")
        print("  /usr/lib/python3/dist-packages/tensorrt/")
        print("Add to your env with:")
        print("  export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH")
        print("Or install via:")
        print("  pip install --extra-index-url https://pypi.ngc.nvidia.com tensorrt")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

    print(f"\n  TensorRT version : {trt.__version__}")
    print(f"  ONNX input       : {onnx_path}")
    print(f"  Engine output    : {engine_path}")
    print(f"  FP16             : {fp16}")
    print(f"  INT8             : {int8}")
    print(f"  Workspace        : {workspace_mb} MB")
    print()

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)
    config  = builder.create_builder_config()

    # Workspace memory (TRT 8.x API)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    # Precision flags
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled (platform supported)")
        else:
            print("  WARNING: FP16 not supported on this platform -- using FP32")

    if int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  INT8 enabled")
            print("  NOTE: INT8 without calibration uses implicit quantisation.")
            print("        For best accuracy provide a calibration dataset.")
        else:
            print("  WARNING: INT8 not supported -- falling back to FP16/FP32")

    # Parse ONNX
    print("  Parsing ONNX model...")
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())

    if not ok:
        print("  ERROR: ONNX parsing failed:")
        for i in range(parser.num_errors):
            print(f"    {parser.get_error(i)}")
        sys.exit(1)

    print(f"  Network inputs : {network.num_inputs}")
    print(f"  Network outputs: {network.num_outputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    Input  {i}: {inp.name}  {inp.shape}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    Output {i}: {out.name}  {out.shape}")

    # Build engine -- this is the slow step (can take 5-15 min on Jetson Nano)
    print()
    print("  Building TensorRT engine...")
    print("  This can take 5-15 minutes on Jetson Nano -- please wait.")
    t0 = time.time()

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  ERROR: engine build failed.")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"  Build time: {elapsed:.0f}s")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = Path(engine_path).stat().st_size / 1024 / 1024
    print(f"  Engine saved: {engine_path}  ({size_mb:.1f} MB)")


def verify_engine(engine_path: str):
    """
    Quick sanity check: load the engine and run one random inference.
    """
    print("\n  Verifying engine...")
    try:
        import tensorrt as trt
        import pycuda.autoinit   # noqa: F401
        import pycuda.driver as cuda
        import numpy as np
    except ImportError as e:
        print(f"  Skipping verification -- missing: {e}")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine  = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Find input/output shapes
    bindings  = []
    host_mems = []
    dev_mems  = []

    for binding in engine:
        shape = tuple(engine.get_binding_shape(binding))
        # Replace dynamic dims (-1) with concrete values
        shape = tuple(1 if s == -1 else s for s in shape)
        dtype = np.float32
        size  = int(np.prod(shape))

        h_mem = cuda.pagelocked_empty(size, dtype)
        d_mem = cuda.mem_alloc(h_mem.nbytes)
        bindings.append(int(d_mem))
        host_mems.append(h_mem)
        dev_mems.append(d_mem)

        is_input = engine.binding_is_input(binding)
        print(f"    {'Input ' if is_input else 'Output'}: {binding}  {shape}")

    stream = cuda.Stream()

    # Fill input with random data
    np.copyto(host_mems[0],
              np.random.randn(*host_mems[0].shape).astype(np.float32).ravel())
    cuda.memcpy_htod_async(dev_mems[0], host_mems[0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_mems[-1], dev_mems[-1], stream)
    stream.synchronize()

    print(f"  Verification OK -- output sample: {host_mems[-1][:5]}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX to TensorRT engine on Jetson Nano")
    parser.add_argument("--onnx",      required=True,
                        help="Input ONNX model path")
    parser.add_argument("--out",       default=None,
                        help="Output engine path (default: <onnx>.engine)")
    parser.add_argument("--fp16",      action="store_true",
                        help="Enable FP16 (recommended for Jetson Nano)")
    parser.add_argument("--int8",      action="store_true",
                        help="Enable INT8 (fastest, implicit quantisation)")
    parser.add_argument("--workspace", type=int, default=512,
                        help="GPU workspace in MB (default: 512)")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--verify",    action="store_true",
                        help="Run a test inference after building")
    args = parser.parse_args()

    onnx_path   = args.onnx
    engine_path = args.out or str(Path(onnx_path).with_suffix(".engine"))

    build_engine(onnx_path, engine_path,
                 fp16=args.fp16,
                 int8=args.int8,
                 workspace_mb=args.workspace,
                 verbose=args.verbose)

    if args.verify:
        verify_engine(engine_path)

    print()
    print("  Done. Test the engine with:")
    print(f"    python3 lane_follower.py --model {engine_path} --tensorrt")
    print()


if __name__ == "__main__":
    main()