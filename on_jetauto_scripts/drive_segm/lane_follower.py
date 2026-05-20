#!/usr/bin/env python3
"""
lane_follower.py -- ROS1 lane following node for JetAuto (Jetson Nano).

Uses a SegFormer segmentation model to detect lane markings in the camera
image, computes a lateral error in Bird's Eye View, and sends velocity
commands to keep the robot centred in the lane.

Architecture:
    /depth_cam/rgb/image_raw  (sensor_msgs/Image)
        |
        v
    preprocess: crop + resize to model input (640x256)
        |
        v
    SegFormer inference (ONNX or TensorRT)
        |
        v
    segmentation mask (5 classes: bg, road, lane_marking, lane_dashed, zebra)
        |
        v
    BEV warpPerspective (auto-calibration from two horizontal mask lines)
        |
        v
    lateral error computation (centroid of lane markings vs BEV center)
        |
        v
    PID controller -> angular.z
        |
        v
    /jetauto_controller/cmd_vel  (geometry_msgs/Twist)

Usage (on the robot):
    python3 lane_follower.py --model model.onnx

    # When TensorRT engine is ready:
    python3 lane_follower.py --model model.engine --tensorrt

Options:
    --model PATH       Path to .onnx or .engine model file
    --speed FLOAT      Forward speed in m/s (default: 0.15)
    --kp    FLOAT      PID proportional gain (default: 1.2)
    --ki    FLOAT      PID integral gain     (default: 0.0)
    --kd    FLOAT      PID derivative gain   (default: 0.3)
    --tensorrt         Use TensorRT engine instead of ONNX runtime
    --debug            Publish debug image on /lane_follower/debug_image
    --dry-run          Run inference but do not publish cmd_vel
    --publish-masks    Publish /lane_mask and /lane_mask_bev (mono8) for external lane_controller.
                       Default: off. Can be combined with --dry-run.
"""
import argparse
import threading
import time
from typing import Union

import cv2
import numpy as np
import rospy
# cv_bridge is not used -- raw numpy conversion avoids Python 2/3 issues
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from auto_calibration import AutoCalibration

import ctypes
import ctypes.util

# Optional debug image publisher
try:
    from sensor_msgs.msg import Image as RosImage
    HAS_ROS_IMAGE = True
except ImportError:
    HAS_ROS_IMAGE = False


# -- Configuration defaults ----------------------------------------------------

CAMERA_TOPIC  = "/depth_cam/rgb/image_raw" # Or "/astra_cam/rgb/image_raw"
CMDVEL_TOPIC  = "/jetauto_controller/cmd_vel"
DEBUG_TOPIC   = "/lane_follower/debug_image"

# Topic for the external lane_controller (Python 2.7, separate node).
# Published only if you pass --publish-masks via CLI.
LANE_MASK_TOPIC      = "/lane_mask"        # maschera in image space, mono8
LANE_MASK_BEV_TOPIC  = "/lane_mask_bev"    # maschera in BEV, mono8

# Model input size -- must match training config
MODEL_H = 256
MODEL_W = 640

# Camera WxH
SRC_IMAGE_WIDTH = 640
SRC_IMAGE_HEIGHT = 480

CROP_TOP_FRAC = 0.45

# ImageNet normalization (same as training pipeline)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Fused constants: out = pixel * NORM_SCALE + NORM_SHIFT  (single pass)
NORM_SCALE = (1.0 / (255.0 * IMAGENET_STD)).reshape(1, 1, 3).astype(np.float32)
NORM_SHIFT = (-IMAGENET_MEAN / IMAGENET_STD).reshape(1, 1, 3).astype(np.float32)

# Segmentation class ids (from config.yaml)
CLASS_ROAD         = 1
CLASS_LANE_MARKING = 2
CLASS_LANE_DASHED  = 3
CLASS_ZEBRA        = 4

CLASS_COLORS = np.array([
        [0,   0,   0],    # 0 background
        [180, 130, 70],   # 1 road
        [0,   255, 255],  # 2 lane_marking
        [255, 255,   0],  # 3 lane_dashed
        [0,   0,   255],  # 4 zebra
    ], dtype=np.uint8)

# Classes used for lateral error computation
# We use lane markings + dashed lines as the primary cue
LANE_CLASSES = [CLASS_LANE_MARKING, CLASS_LANE_DASHED]

# Safety: stop if fewer than this many lane pixels are visible in BEV
MIN_LANE_PIXELS = 50


# -- Model backends ------------------------------------------------------------

class ONNXBackend:
    """Inference via onnxruntime -- works on any machine without TensorRT."""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if self._has_cuda() else ["CPUExecutionProvider"])
        self.sess      = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        rospy.loginfo("[lane_follower] ONNX backend loaded: %s", model_path)
        rospy.loginfo("[lane_follower] Providers: %s",
                      self.sess.get_providers())

    def infer(self, img_chw: np.ndarray) -> np.ndarray:
        """
        img_chw: float32 array (3, H, W) normalised
        Returns: int64 array (H, W) with class indices
        """
        inp    = img_chw[np.newaxis]   # (1, 3, H, W)
        result = self.sess.run(None, {self.input_name: inp})
        return result[0][0]            # (H, W)

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False


class TensorRTBackend:
    """
    Inference via TensorRT using ctypes + CUDA Runtime API.
    No PyTorch needed -- saves ~1.5GB RAM on Jetson Nano.

    Uses cudaRT (libcudart.so) instead of the driver API so that
    TensorRT and our memory transfers share the same CUDA context,
    avoiding the "invalid resource handle" error that occurs when
    mixing cuCtxCreate (driver API) with TensorRT (runtime API).

    Requires only: tensorrt + numpy + libcudart.so (always on Jetson)
    """

    def __init__(self, engine_path: str):
        import tensorrt as trt

        # Load CUDA Runtime library (runtime API -- same one TensorRT uses)
        for lib_name in ("libcudart.so.10.2",
                         "libcudart.so",
                         "/usr/local/cuda/lib64/libcudart.so"):
            try:
                self.rt = ctypes.CDLL(lib_name)
                break
            except OSError:
                continue
        else:
            raise RuntimeError(
                "Cannot find libcudart.so -- check CUDA installation")

        # Load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(
                f.read())
        self.context = self.engine.create_execution_context()

        # Create CUDA stream via runtime API (cudaStreamCreate)
        # cudaStream_t is a pointer -- represent as c_void_p
        self.stream = ctypes.c_void_p()
        self._rt(self.rt.cudaStreamCreate(ctypes.byref(self.stream)))

        # Allocate host (pinned) + device buffers per binding
        self.host_bufs = []
        self.dev_bufs = []
        self.bindings = []
        self.in_idx = []
        self.out_idx = []
        self.out_shapes = []

        for i, binding in enumerate(self.engine):
            shape = tuple(max(s, 1) for s in
                          self.engine.get_binding_shape(binding))
            n = int(np.prod(shape))

            trt_dtype = self.engine.get_binding_dtype(binding)
            dtype_map = {
                trt.DataType.FLOAT: (np.float32, 4),
                trt.DataType.HALF: (np.float16, 2),
                trt.DataType.INT32: (np.int32, 4),
                trt.DataType.INT8: (np.int8, 1),
            }
            np_dtype, item_size = dtype_map.get(trt_dtype, (np.float32, 4))

            h_ptr = ctypes.c_void_p()
            self._rt(self.rt.cudaMallocHost(
                ctypes.byref(h_ptr), n * item_size))
            h_buf = np.frombuffer(
                (ctypes.c_char * (n * item_size)).from_address(h_ptr.value),
                dtype=np_dtype)

            d_ptr = ctypes.c_void_p()
            self._rt(self.rt.cudaMalloc(
                ctypes.byref(d_ptr), n * item_size))

            self.host_bufs.append((h_ptr, h_buf, item_size))
            self.dev_bufs.append(d_ptr)
            self.bindings.append(d_ptr.value)

            if self.engine.binding_is_input(binding):
                self.in_idx.append(i)
            else:
                self.out_idx.append(i)
                self.out_shapes.append(shape)

        rospy.loginfo("[lane_follower] TensorRT/cudart backend: %s", engine_path)
        rospy.loginfo("[lane_follower] Input  shape: %s",
                      tuple(self.engine.get_binding_shape(
                          list(self.engine)[self.in_idx[0]])))
        rospy.loginfo("[lane_follower] Output shape: %s", self.out_shapes[0])


    def _rt(self, result):
        """Check cudaError_t -- 0 = cudaSuccess."""
        if result != 0:
            raise RuntimeError(f"CUDA runtime error code: {result}")

    def infer(self, img_chw: np.ndarray,
              out_buf: np.ndarray = None) -> np.ndarray:
        ii = self.in_idx[0]
        oi = self.out_idx[0]

        # Copy directly in pinned buffer w/o astype/ravel
        h_in = self.host_bufs[ii][1]
        n_in = img_chw.size
        np.copyto(h_in[:n_in], img_chw.ravel())  # img_chw is already float32

        self._rt(self.rt.cudaMemcpyAsync(
            self.dev_bufs[ii], self.host_bufs[ii][0],
            n_in * 4, ctypes.c_int(1), self.stream))

        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.value)

        n_out = int(np.prod(self.out_shapes[0]))
        item_size = self.host_bufs[oi][2]
        self._rt(self.rt.cudaMemcpyAsync(
            self.host_bufs[oi][0], self.dev_bufs[oi],
            n_out * item_size, ctypes.c_int(2), self.stream))
        self._rt(self.rt.cudaStreamSynchronize(self.stream))

        # raw is already int32 inside pinned buffer -- no copy
        raw = self.host_bufs[oi][1][:n_out].reshape(self.out_shapes[0])[0]

        if out_buf is not None:
            # Cast int32->int64 in-place
            np.copyto(out_buf, raw, casting='unsafe')
            return out_buf
        else:
            return raw.astype(np.int64)

class TensorRTBackendTorch:
    """
    Inference via TensorRT using PyTorch CUDA tensors -- no pycuda needed.

    Uses torch.cuda tensors as GPU buffers and ctypes to pass pointers
    to TensorRT execute_v2. Requires only tensorrt + torch with CUDA.

    The engine file must have been compiled on the same Jetson device.
    """

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TensorRT backend requires CUDA. "
                "torch.cuda.is_available() returned False.")

        self.torch = torch
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate GPU tensors for each binding using PyTorch
        self.gpu_bufs  = []   # list of torch CUDA tensors
        self.bindings  = []   # list of data_ptr() for execute_v2
        self.in_idx    = []   # indices of input bindings
        self.out_idx   = []   # indices of output bindings
        self.out_shapes = []  # shapes of output bindings

        for i, binding in enumerate(self.engine):
            shape = tuple(self.engine.get_binding_shape(binding))
            # Replace any -1 dynamic dims with 1
            shape = tuple(max(s, 1) for s in shape)
            buf   = torch.zeros(shape, dtype=torch.float32, device="cuda")
            self.gpu_bufs.append(buf)
            self.bindings.append(buf.data_ptr())

            if self.engine.binding_is_input(binding):
                self.in_idx.append(i)
            else:
                self.out_idx.append(i)
                self.out_shapes.append(shape)

        rospy.loginfo("[lane_follower] TensorRT backend loaded: %s", engine_path)
        rospy.loginfo("[lane_follower] Input  shape: %s",
                      tuple(self.engine.get_binding_shape(
                          list(self.engine)[self.in_idx[0]])))
        rospy.loginfo("[lane_follower] Output shape: %s",
                      self.out_shapes[0])

    def infer(self, img_chw: np.ndarray) -> np.ndarray:
        """
        img_chw: float32 numpy array (3, H, W) normalised
        Returns: int64 numpy array (H, W) with class indices
        """
        torch = self.torch

        # Copy input numpy array to GPU tensor
        inp_tensor = torch.from_numpy(
            img_chw[np.newaxis].astype(np.float32)).cuda()
        self.gpu_bufs[self.in_idx[0]].copy_(inp_tensor)

        # Run inference synchronously
        self.context.execute_v2(bindings=self.bindings)

        # Copy output from GPU to CPU numpy
        out_tensor = self.gpu_bufs[self.out_idx[0]]
        out_np     = out_tensor.cpu().numpy()

        # Output is (1, H, W) -- remove batch dim and cast to int
        return out_np[0].astype(np.int64)


# -- Image preprocessing -------------------------------------------------------

def preprocess(img_rgb: np.ndarray, crop_top_frac: float) -> np.ndarray:
    """
    Crop top, resize to model input, normalize with ImageNet stats.
    Returns float32 array (3, MODEL_H, MODEL_W) ready for inference.
    """
    h = img_rgb.shape[0]
    crop_px = int(h * crop_top_frac)
    cropped = img_rgb[crop_px:, :]
    resized = cv2.resize(cropped, (MODEL_W, MODEL_H),
                         interpolation=cv2.INTER_LINEAR)
    # Input is already RGB -- normalize directly
    normalised = (resized.astype(np.float32) / 255.0
                  - IMAGENET_MEAN) / IMAGENET_STD
    return normalised.transpose(2, 0, 1)  # HWC -> CHW               # (3, H, W)

def preprocess_inplace(img_rgb: np.ndarray, crop_top_frac: float,
                        out_buf: np.ndarray):
    """
    Crop top, resize to model input, normalize with ImageNet stats.
    Writes the result in out_buf (1,3,H,W) float32.
    No allocation -- out_buf must be already allocated.
    """
    h       = img_rgb.shape[0]
    crop_px = int(h * crop_top_frac)
    # Normalize and write directly inside the pre-allocated buffer
    # out_buf shape: (3, H, W)
    resized = cv2.resize(img_rgb[crop_px:], (MODEL_W, MODEL_H),
                         interpolation=cv2.INTER_LINEAR)
    # Single fused pass: out = pixel * scale + shift  (avoids 3 separate traversals)
    dst_hwc = out_buf[0].transpose(1, 2, 0)   # view, no copy
    np.multiply(resized, NORM_SCALE, out=dst_hwc, casting='unsafe')
    dst_hwc += NORM_SHIFT

# -- BEV + lateral error -------------------------------------------------------

# bev_config-based BEV logic moved to bev_from_config.py (legacy).


# -- PID controller ------------------------------------------------------------

class PIDController:
    """
    Simple discrete PID controller for angular velocity.

    error > 0 (centroid right of center) -> steer right -> angular.z negative
    error < 0 (centroid left  of center) -> steer left  -> angular.z positive
    """

    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: float = 1.0):
        self.kp           = kp
        self.ki           = ki
        self.kd           = kd
        self.output_limit = output_limit
        self._integral    = 0.0
        self._prev_error  = 0.0
        self._prev_time   = None

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

    def compute(self, error: float) -> float:
        """
        Compute PID output given the current normalized lateral error.
        Returns angular velocity correction (rad/s).
        """
        now = time.time()
        dt  = (now - self._prev_time) if self._prev_time is not None else 0.05
        dt  = max(dt, 1e-4)

        self._integral  += error * dt
        # Anti-windup: clamp integral
        self._integral   = float(np.clip(self._integral, -2.0, 2.0))

        derivative       = (error - self._prev_error) / dt
        output           = (self.kp * error +
                            self.ki * self._integral +
                            self.kd * derivative)

        self._prev_error = error
        self._prev_time  = now

        # Negative because: error>0 means steer right = negative angular.z in ROS
        return float(np.clip(-output, -self.output_limit, self.output_limit))


# -- Debug visualisation -------------------------------------------------------

def make_debug_image(img_bgr: np.ndarray,
                     mask: np.ndarray,
                     bev_mask: np.ndarray,
                     error_norm: float,
                     n_pixels: int,
                     angular_z: float) -> np.ndarray:
    """
    Build a debug image: original | colored mask | BEV mask
    with error bar and stats overlaid.
    """
    CLASS_COLORS = np.array([
        [0,   0,   0],    # 0 background
        [180, 130, 70],   # 1 road
        [0,   255, 255],  # 2 lane_marking
        [255, 255,   0],  # 3 lane_dashed
        [0,   0,   255],  # 4 zebra
    ], dtype=np.uint8)

    h, w = img_bgr.shape[:2]

    # Colored mask overlay
    colored = CLASS_COLORS[mask.clip(0, 4)]
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, colored_bgr, 0.4, 0)

    # BEV mask
    bev_colored = CLASS_COLORS[bev_mask.clip(0, 4)]
    bev_bgr     = cv2.cvtColor(bev_colored, cv2.COLOR_RGB2BGR)
    bev_resized = cv2.resize(bev_bgr, (w, h))

    # Error bar on BEV image
    cx    = int(w / 2)
    ex    = int(cx + error_norm * (w / 2))
    cv2.line(bev_resized, (cx, h-20), (cx, h-5),  (0, 255, 0),  2)
    cv2.line(bev_resized, (cx, h-12), (ex, h-12), (0, 100, 255), 3)
    cv2.circle(bev_resized, (ex, h-12), 5, (0, 100, 255), -1)

    # Stats text
    stats = [
        f"err={error_norm:+.3f}",
        f"ang={angular_z:+.3f} rad/s",
        f"px={n_pixels}",
    ]
    for i, txt in enumerate(stats):
        cv2.putText(overlay, txt, (8, 20 + i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

    return np.hstack([overlay, bev_resized])


# -- ROS node ------------------------------------------------------------------

class LaneFollowerNode:

    def __init__(self, args):
        rospy.init_node("lane_follower", anonymous=False)

        # Load model backend
        if args.tensorrt:
            self.model = TensorRTBackend(args.model)
        else:
            self.model = ONNXBackend(args.model)

        self.crop_top_frac = CROP_TOP_FRAC

        # Auto-calibration (top half to near-bottom)
        top_line = MODEL_H - (MODEL_H // 4)
        bottom_line = MODEL_H - 10
        self.angle = 0 # Placeholder
        self.log_calibration_once = True

        self.calib_lane_label = CLASS_LANE_MARKING
        calib_path = args.calibration

        points, angle = self.check_calibration(calib_path)
        if points is None:
            self.auto_calib = AutoCalibration(top_line, bottom_line)
            self.pending_calibration = self._prompt_calibration()
            if not self.pending_calibration:
                rospy.signal_shutdown("Calibration declined")
                raise SystemExit(0)
        else:
            rospy.loginfo("[lane_follower] Loaded calibration points and angle from: %s", calib_path)
            self.auto_calib = AutoCalibration(top_line, bottom_line, last_src_pts=points, calib_angle=angle)
            self.pending_calibration = False

        # PID controller
        self.pid = PIDController(
            kp=args.kp, ki=args.ki, kd=args.kd,
            output_limit=1.5)

        self.speed   = args.speed
        self.dry_run = args.dry_run
        self.debug   = args.debug
        self.publish_masks = args.publish_masks

        # State
        self.last_error    = 0.0
        self.frames_total  = 0
        self.frames_no_lane = 0

        # Input and Mask Buffers
        self._inp_buf = np.empty((1, 3, MODEL_H, MODEL_W), dtype=np.float32)
        self._mask_buf = np.empty((MODEL_H, MODEL_W), dtype=np.int64)

        # Publishers
        if not self.dry_run:
            self.cmd_pub = rospy.Publisher(
                CMDVEL_TOPIC, Twist, queue_size=1)
        if self.debug:
            self.debug_pub = rospy.Publisher(
                DEBUG_TOPIC, Image, queue_size=1)

        # Mask publishers for external lane_controller (opt-in with --publish-masks).
        if self.publish_masks:
            self.lane_mask_pub     = rospy.Publisher(
                LANE_MASK_TOPIC, Image, queue_size=1)
            self.lane_mask_bev_pub = rospy.Publisher(
                LANE_MASK_BEV_TOPIC, Image, queue_size=1)
            rospy.loginfo(
                "[lane_follower] publish_masks ON: %s, %s",
                LANE_MASK_TOPIC, LANE_MASK_BEV_TOPIC)

        # Shared frame buffer: inference thread reads, camera callback writes
        self._frame_lock  = threading.Lock()
        self._frame_event = threading.Event()
        self._latest_frame  = None   # (img_rgb, header) tuple
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="lane_inference")
        self._inference_thread.start()

        # Subscriber -- just converts and stores latest frame, never blocks
        self.sub = rospy.Subscriber(
            CAMERA_TOPIC, Image, self.image_cb,
            queue_size=1, buff_size=2**24)

        rospy.loginfo("[lane_follower] Node started")
        rospy.loginfo("[lane_follower] Speed: %.2f m/s  Kp=%.2f Ki=%.2f Kd=%.2f",
                      self.speed, args.kp, args.ki, args.kd)
        rospy.loginfo("[lane_follower] Dry-run: %s  Debug: %s",
                      self.dry_run, self.debug)
        rospy.loginfo("[lane_follower] Publish masks: %s", self.publish_masks)
        if self.dry_run:
            rospy.logwarn("[lane_follower] DRY-RUN mode -- no cmd_vel published")

        rospy.on_shutdown(self._on_shutdown)

    def _prompt_calibration(self) -> bool:
        while True:
            try:
                resp = input("Start calibration? [y/n]: ").strip().lower()
            except EOFError:
                return False
            if resp in ("y", "yes"):
                return True
            if resp in ("n", "no"):
                return False

    @staticmethod
    def check_calibration(path) -> Union[(np.float32, np.float32), (None, None)]:
        import os
        if os.path.exists(path):
            rospy.loginfo_once("[lane_follower] Calibration found: %s", path)

            with open(path, "rb") as f:
                data = np.load(f)
                points = data.get("calib_points", None)
                angles = data.get("calib_angles", None)
                return points, angles
        #Else return None,  None
        return     None, None



    # -- Camera callback (lightweight) ----------------------------------------

    def image_cb(self, msg: Image):
        """Convert the ROS image and store it; inference runs in a separate thread."""
        try:
            n_ch = {"rgb8": 3, "bgr8": 3, "mono8": 1,
                    "rgba8": 4, "bgra8": 4}.get(msg.encoding, 3)
            img_raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, n_ch)
            rospy.loginfo_once("[lane_follower] Camera encoding: %s  shape=%s",
                               msg.encoding, img_raw.shape)

            if msg.encoding == "rgb8":
                img_rgb = img_raw[:, :, :3].copy()
            elif msg.encoding == "bgr8":
                img_rgb = img_raw[:, :, ::-1].copy()
            elif msg.encoding == "rgba8":
                img_rgb = img_raw[:, :, :3].copy()
            elif msg.encoding == "bgra8":
                img_rgb = img_raw[:, :, 2::-1].copy()
            elif msg.encoding == "mono8":
                img_rgb = np.stack([img_raw[:, :, 0]] * 3, axis=-1)
            else:
                img_rgb = img_raw[:, :, :3].copy()
        except Exception as e:
            rospy.logwarn_throttle(5, "[lane_follower] image conversion failed: %s", e)
            return

        with self._frame_lock:
            self._latest_frame = (img_rgb, msg.header)
        self._frame_event.set()

    # -- Inference loop (runs in background thread) ----------------------------

    def _inference_loop(self):
        while not rospy.is_shutdown():
            if not self._frame_event.wait(timeout=0.1):
                continue
            self._frame_event.clear()

            with self._frame_lock:
                payload = self._latest_frame
            if payload is None:
                continue
            img_rgb, header = payload

            t0 = time.time()

            # Preprocess
            preprocess_inplace(img_rgb, self.crop_top_frac, self._inp_buf)

            # Inference
            try:
                mask = self.model.infer(self._inp_buf[0])
            except Exception as e:
                rospy.logerr("[lane_follower] Inference failed: %s", e)
                self._publish_stop()
                continue

            # Calibration on first valid frame
            if self.pending_calibration:
                self.angle = self.auto_calib.calibrate(mask, self.calib_lane_label)
                self.pending_calibration = False

            # BEV transform
            bev_mask = self.auto_calib.make_bev(mask)
            bev_mask_u8 = np.clip(bev_mask, 0, 255).astype(np.uint8, copy=False)

            if self.log_calibration_once:
                self.auto_calib.save_debug(mask, prefix="lane_calibration")
                rospy.loginfo("[lane_follower] Saved calibration debug images")
                self.log_calibration_once = False

            # Publish masks for external lane_controller (opt-in)
            if self.publish_masks:
                try:
                    self.lane_mask_pub.publish(
                        self._make_mono8_msg(mask, header))
                    self.lane_mask_bev_pub.publish(
                        self._make_mono8_msg(bev_mask_u8, header))
                except Exception as e:
                    rospy.logwarn_throttle(
                        5, "[lane_follower] mask publish err: %s", e)

            # Lateral error
            error_norm, n_pixels = self._lateral_error(bev_mask_u8)

            self.frames_total += 1

            if n_pixels < MIN_LANE_PIXELS:
                self.frames_no_lane += 1
                rospy.logwarn_throttle(
                    2, "[lane_follower] Few lane pixels (%d) -- holding last error", n_pixels)
                error_norm = self.last_error * 0.5
            else:
                self.last_error = error_norm

            # PID
            angular_z = self.pid.compute(error_norm)

            # Publish cmd_vel
            if not self.dry_run:
                twist = Twist()
                twist.linear.x  = self.speed
                twist.angular.z = angular_z
                self.cmd_pub.publish(twist)

            elapsed_ms = (time.time() - t0) * 1000
            rospy.loginfo_throttle(
                1, "[lane_follower] err=%+.3f ang=%+.3f px=%d fps=%.1f no_lane=%d/%d",
                error_norm, angular_z, n_pixels,
                1000.0 / max(elapsed_ms, 1),
                self.frames_no_lane, self.frames_total)

            # Debug image
            if self.debug and self.debug_pub.get_num_connections() > 0:
                crop_px  = int(img_rgb.shape[0] * self.crop_top_frac)
                img_disp = cv2.resize(img_rgb[crop_px:], (MODEL_W, MODEL_H))
                dbg = make_debug_image(img_disp, mask, bev_mask_u8,
                                       error_norm, n_pixels, angular_z)
                try:
                    dbg_msg          = Image()
                    dbg_msg.header   = header
                    dbg_msg.height   = dbg.shape[0]
                    dbg_msg.width    = dbg.shape[1]
                    dbg_msg.encoding = "bgr8"
                    dbg_msg.step     = dbg.shape[1] * 3
                    dbg_msg.data     = dbg.tobytes()
                    self.debug_pub.publish(dbg_msg)
                except Exception:
                    pass

    def _lateral_error(self, bev_mask: np.ndarray) -> tuple:
        h, w = bev_mask.shape[:2]
        roi = bev_mask[h // 2:, :]

        lane_mask = np.zeros_like(roi, dtype=bool)
        for cls in LANE_CLASSES:
            lane_mask |= (roi == cls)

        n_pixels = int(lane_mask.sum())
        if n_pixels < MIN_LANE_PIXELS:
            return 0.0, n_pixels

        xs = np.where(lane_mask)[1].astype(np.float32)
        centroid_x = float(xs.mean())
        error_px = centroid_x - (w * 0.5)
        error_norm = error_px / (w * 0.5)

        return float(np.clip(error_norm, -1.0, 1.0)), n_pixels

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _make_mono8_msg(mask, header):
        """Convert an HxW uint8 mask to a sensor_msgs/Image mono8.
        Same raw-numpy pattern as the debug_image, no cv_bridge."""
        msg          = Image()
        msg.header   = header
        msg.height   = mask.shape[0]
        msg.width    = mask.shape[1]
        msg.encoding = "mono8"
        msg.step     = mask.shape[1]
        msg.data     = mask.astype(np.uint8).tobytes()
        return msg

    def _publish_stop(self):
        """Publish zero velocity to stop the robot safely."""
        if not self.dry_run:
            self.cmd_pub.publish(Twist())
        self.pid.reset()

    def _on_shutdown(self):
        rospy.loginfo("[lane_follower] Shutting down -- sending stop")
        self._publish_stop()

    def run(self):
        rospy.spin()


# -- Entry point ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lane following node for JetAuto (ROS1 Melodic)")
    parser.add_argument("--model",    required=True,
                        help="Path to .onnx or .engine model file")
    parser.add_argument("--speed",    type=float, default=0.15,
                        help="Forward speed m/s (default: 0.15)")
    parser.add_argument("--kp",       type=float, default=1.2)
    parser.add_argument("--ki",       type=float, default=0.0)
    parser.add_argument("--kd",       type=float, default=0.3)
    parser.add_argument("--tensorrt", action="store_true",
                        help="Use TensorRT backend instead of ONNX")
    parser.add_argument("--debug",    action="store_true",
                        help="Publish debug image on /lane_follower/debug_image")
    parser.add_argument("--publish-masks", action="store_true",
                        dest="publish_masks",
                        help="Publish /lane_mask and /lane_mask_bev (mono8) "
                             "for external lane_controller. Default: off. "
                             "Can be combined with --dry-run.")
    parser.add_argument("--dry-run",  action="store_true", dest="dry_run",
                        help="Run inference without publishing cmd_vel")
    parser.add_argument("--calibration", help="Path to calibration json file", default="calibration.json")

    # ROS passes extra args -- filter them out
    import rospy
    args = parser.parse_args(rospy.myargv()[1:])

    node = LaneFollowerNode(args)
    node.run()


if __name__ == "__main__":
    main()