#!/usr/bin/env python3
"""
lane_follower.py -- ROS1 segmentation node for JetAuto (Jetson Nano).

Runs a MobileNet (SegFormer also compatible) segmentation model on the camera
image and publishes the lane-marking mask (Bird's Eye View) for the downstream
Python 2.7 control stack. It does NOT drive: steering and cmd_vel are produced
by lane_controller_node.py + orchestrator.py.

Architecture:
    /depth_cam/rgb/image_raw  (sensor_msgs/Image)
        |
        v
    preprocess: crop + resize to model input (320x128)
        |
        v
    Model inference (ONNX or TensorRT)
        |
        v
    segmentation mask (5 classes: bg, road, lane_marking, lane_dashed, zebra)
        |
        v
    BEV warpPerspective (auto-calibration from two horizontal mask lines)
        |
        v
    /lane_mask_bev  (sensor_msgs/Image, mono8)  -> lane_controller_node.py

Usage (on the robot):
    python3 lane_follower.py --model model.onnx

    # When TensorRT engine is ready:
    python3 lane_follower.py --model model.engine --tensorrt

Options:
    --model PATH        Path to .onnx or .engine model file
    --tensorrt          Use TensorRT engine instead of ONNX runtime
    --debug             Publish debug image on /lane_follower/debug_image
    --calibration PATH  Path to calibration json (default: calibration.json)
    --max-fps FLOAT     Cap inference rate to this FPS (0 = unlimited)
    --print-debug       Print per-frame timing/state to console
"""
import argparse
import threading
import time
from typing import Union

import cv2
import numpy as np
import rospy
# cv_bridge is not used -- raw numpy conversion avoids Python 2/3 issues
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
DEBUG_TOPIC   = "/lane_follower/debug_image"

# BEV mask topic consumed by the external lane_controller (Python 2.7, separate node).
LANE_MASK_BEV_TOPIC  = "/lane_mask_bev"    # BEV-warped class mask, mono8

# Model input size -- must match training config
MODEL_H = 128
MODEL_W = 320

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

# Top Line cut of the BEV for error computation -- avoids far-away noisy pixels
TOP_LINE = MODEL_H - (MODEL_H // 3)
# Bottom Line for BEV
BOTTOM_LINE = MODEL_H - 20


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

# -- Image preprocessing -------------------------------------------------------

def preprocess_inplace(img_rgb: np.ndarray, crop_top_frac: float,
                        out_buf: np.ndarray,
                        gpu_src=None):
    """
    Crop top, resize to model input, normalize with ImageNet stats.
    Writes the result in out_buf (1,3,H,W) float32.
    If gpu_src/gpu_dst (pre-allocated cv2.cuda_GpuMat) are provided, the
    resize runs on GPU; otherwise falls back to CPU cv2.resize.
    """
    h       = img_rgb.shape[0]
    crop_px = int(h * crop_top_frac)
    cropped = img_rgb[crop_px:]

    if gpu_src is not None:
        gpu_src.upload(cropped)
        resized = cv2.cuda.resize(gpu_src, (MODEL_W, MODEL_H),
                                  interpolation=cv2.INTER_LINEAR).download()
    else:
        resized = cv2.resize(cropped, (MODEL_W, MODEL_H),
                             interpolation=cv2.INTER_LINEAR)

    dst_hwc = out_buf[0].transpose(1, 2, 0)   # view, no copy
    np.multiply(resized, NORM_SCALE, out=dst_hwc, casting='unsafe')
    dst_hwc += NORM_SHIFT

# -- BEV note ------------------------------------------------------------------

# bev_config-based BEV logic moved to bev_from_config.py (legacy).


# -- Debug visualisation -------------------------------------------------------

def make_debug_image(img_bgr: np.ndarray,
                     mask: np.ndarray,
                     bev_mask: np.ndarray) -> np.ndarray:
    """
    Build a debug image: segmentation overlay | BEV mask.
    No error bar / steering stats -- driving is handled elsewhere.
    """
    h, w = img_bgr.shape[:2]

    # Colored mask overlay on the cropped input
    colored     = CLASS_COLORS[mask.clip(0, 4)]
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    overlay     = cv2.addWeighted(img_bgr, 0.6, colored_bgr, 0.4, 0)

    # BEV mask, resized to match
    bev_colored = CLASS_COLORS[bev_mask.clip(0, 4)]
    bev_bgr     = cv2.cvtColor(bev_colored, cv2.COLOR_RGB2BGR)
    bev_resized = cv2.resize(bev_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

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
        top_line = TOP_LINE
        bottom_line = BOTTOM_LINE
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

        self.debug   = args.debug
        self.max_fps = args.max_fps
        self._min_period = (1.0 / args.max_fps) if args.max_fps > 0 else 0.0
        self.print_debug = args.print_debug

        # State
        self.frames_total  = 0

        # Input and Mask Buffers
        self._inp_buf = np.empty((1, 3, MODEL_H, MODEL_W), dtype=np.float32)
        self._mask_buf = np.empty((MODEL_H, MODEL_W), dtype=np.int64)

        # Pre-allocated GPU source buffer for preprocessing (reused every frame)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self._gpu_crop_src = cv2.cuda_GpuMat()
            rospy.loginfo("[lane_follower] CUDA resize enabled")
        else:
            self._gpu_crop_src = None
            rospy.logwarn("[lane_follower] CUDA not available -- resize on CPU")

        # Publishers -- the BEV mask is the node's only product
        self.lane_mask_bev_pub = rospy.Publisher(
            LANE_MASK_BEV_TOPIC, Image, queue_size=1)
        if self.debug:
            self.debug_pub = rospy.Publisher(
                DEBUG_TOPIC, Image, queue_size=1)

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

        rospy.loginfo("[lane_follower] Node started (segmentation-only)")
        rospy.loginfo("[lane_follower] Publishing BEV mask on %s  Debug: %s",
                      LANE_MASK_BEV_TOPIC, self.debug)

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
    def check_calibration(path):
        import os, json
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                points = np.array(data["src_points"], dtype=np.float32)
                angle  = float(data["calibration_angle"])
                rospy.loginfo("[lane_follower] Calibration loaded from: %s", path)
                return points, angle
            except Exception as e:
                rospy.logwarn("[lane_follower] Failed to load calibration %s: %s", path, e)
        return None, None



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
            preprocess_inplace(img_rgb, self.crop_top_frac, self._inp_buf,
                               self._gpu_crop_src)

            # Inference
            try:
                mask = self.model.infer(self._inp_buf[0])
            except Exception as e:
                rospy.logerr("[lane_follower] Inference failed: %s", e)
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

            # Publish the BEV mask for the external lane_controller
            try:
                self.lane_mask_bev_pub.publish(
                    self._make_mono8_msg(bev_mask_u8, header))
            except Exception as e:
                rospy.logwarn_throttle(
                    5, "[lane_follower] bev publish err: %s", e)

            self.frames_total += 1

            elapsed_ms = (time.time() - t0) * 1000
            if self.print_debug:
                rospy.loginfo_throttle(
                    1, "[lane_follower] fps=%.1f frames=%d",
                    1000.0 / max(elapsed_ms, 1), self.frames_total)
            else:
                rospy.loginfo_once("[lane_follower] Started Inference")

            if self._min_period > 0:
                remaining = self._min_period - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)

            # Debug image
            if self.debug and self.debug_pub.get_num_connections() > 0:
                crop_px  = int(img_rgb.shape[0] * self.crop_top_frac)
                img_crop = cv2.resize(img_rgb[crop_px:], (MODEL_W, MODEL_H))
                img_disp = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
                dbg = make_debug_image(img_disp, mask, bev_mask_u8)
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

    def _on_shutdown(self):
        rospy.loginfo("[lane_follower] Shutting down")

    def run(self):
        rospy.spin()


# -- Entry point ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lane following node for JetAuto (ROS1 Melodic)")
    parser.add_argument("--model",    required=True,
                        help="Path to .onnx or .engine model file")
    parser.add_argument("--tensorrt", action="store_true",
                        help="Use TensorRT backend instead of ONNX")
    parser.add_argument("--debug",    action="store_true",
                        help="Publish debug image on /lane_follower/debug_image")
    parser.add_argument("--calibration", help="Path to calibration json file", default="calibration.json")
    parser.add_argument("--max-fps", type=float, default=0.0, dest="max_fps",
                        help="Cap inference rate to this FPS (0 = unlimited)")
    parser.add_argument("--print-debug", action="store_true", dest="print_debug", default=False,
                        help="Print per-frame timing/state to console")

    # ROS passes extra args -- filter them out
    import rospy
    args = parser.parse_args(rospy.myargv()[1:])

    node = LaneFollowerNode(args)
    node.run()


if __name__ == "__main__":
    main()