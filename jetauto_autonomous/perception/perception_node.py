#!/usr/bin/env python3
"""
perception_node.py -- merged perception node for JetAuto (Jetson Orin, ROS1).

Runs two models on the SAME camera frame, sharing one camera subscriber, one
frame buffer and one CUDA context to keep the Jetson affordable:

    1. Object detection (YOLO11, TensorRT)  -- runs on the FULL frame, draws
       bounding boxes, publishes an annotated video on /object_detection/video.
    2. Lane segmentation (MobileNet/SegFormer, ONNX or TensorRT) -- runs on the
       bottom crop, produces a BEV mask on /lane_mask_bev for the external
       lane_controller, and an optional debug image.

Per frame the detection model runs first, then the same frame is cropped and the
segmentation model runs.

Both models go through the SAME pycuda-free cudart/ctypes TensorRT backend
(`TRTEngine`) so they share one CUDA context. Do NOT add pycuda here -- mixing
the pycuda driver API context with the cudart runtime API causes the
"invalid resource handle" error.

Driving logic is intentionally NOT in this node:
  * The object-detection driving FSM (the other group's `Brain`) is stripped;
    they publish detections on their own topics for the orchestrator.
  * Lane following / cmd_vel is handled by the external Python 2.7
    lane_controller_node, which consumes /lane_mask_bev.

Authorship / attribution
------------------------
The OBJECT-DETECTION half of this file is NOT our work. The YOLO11 engine
usage, the `MY_CLASS_NAMES` map and the `preprocess_detection` /
`postprocess_detections` / `draw_detections` helpers are derived from the
TensorRT detection publisher written by another exam group (source:
`yolo_publisher.py`). We only adapted their code to share our pycuda-free
cudart `TRTEngine`, camera subscriber and frame buffer. All credit for the
object-detection model and its pre/post-processing belongs to them.

The lane-segmentation half and the merge/integration are ours.

  Object detection authors (other group): <FILL IN NAMES / GITHUB HANDLES>

Usage (on the robot, conda Python 3 env):
    # Both models
    python3 perception_node.py --mode both \
        --det-model yolo11s_320x320.engine \
        --seg-model model.engine --seg-tensorrt --debug

    # Detection only
    python3 perception_node.py --mode detection \
        --det-model yolo11s_320x320.engine

    # Segmentation only
    python3 perception_node.py --mode segmentation \
        --seg-model model.engine --seg-tensorrt --debug

Options:
    --mode {both,detection,segmentation}   Which model(s) to run (default: both)
    --det-model PATH    YOLO .engine file        (required for detection)
    --seg-model PATH    seg .onnx/.engine file    (required for segmentation)
    --seg-tensorrt      Use TensorRT backend for the seg model (else ONNX)
    --debug             Publish /lane_follower/debug_image (default: on)
    --nodebug           Disable /lane_follower/debug_image
    --calibration PATH  Calibration json (default: calibration.json)
    --max-fps FLOAT     Cap the combined loop to this FPS (0 = unlimited)
    --print-debug       Print per-frame timing to console
"""
import argparse
import ctypes
import threading
import time
import json

import cv2
import numpy as np
import rospy
# cv_bridge is not used -- raw numpy conversion avoids Python 2/3 issues
from sensor_msgs.msg import Image
from std_msgs.msg import String

from jetauto_autonomous.perception.auto_calibration import AutoCalibration


# -- Topics --------------------------------------------------------------------

CAMERA_TOPIC        = "/depth_cam/rgb/image_raw"   # Or "/astra_cam/rgb/image_raw"
DEBUG_TOPIC         = "/lane_follower/debug_image"
LANE_MASK_BEV_TOPIC = "/lane_mask_bev"             # mono8, consumed by lane_controller
DETECTION_VIDEO_TOPIC = "/object_detection/video"  # bgr8, consumed by dashboard
DETECTION_DRIVING_INFO = "/object_detection/drive" # info for the orchestrator

# -- Segmentation configuration (must match training) --------------------------

MODEL_H = 128
MODEL_W = 320

CROP_TOP_FRAC = 0.45

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Fused constants: out = pixel * NORM_SCALE + NORM_SHIFT  (single pass)
NORM_SCALE = (1.0 / (255.0 * IMAGENET_STD)).reshape(1, 1, 3).astype(np.float32)
NORM_SHIFT = (-IMAGENET_MEAN / IMAGENET_STD).reshape(1, 1, 3).astype(np.float32)

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

# Top/bottom cut of the BEV (kept for AutoCalibration parametrisation)
TOP_LINE    = MODEL_H - (MODEL_H // 3)
BOTTOM_LINE = MODEL_H - 20


# =============================================================================
# OBJECT DETECTION (YOLO11) -- authored by ANOTHER EXAM GROUP.
# Source: yolo_publisher.py. Adapted only to use our shared cudart TRTEngine and
# camera pipeline. See the "Authorship / attribution" note in the module
# docstring. The code from here through draw_detections() is theirs.
# =============================================================================

# -- Object-detection configuration --------------------------------------------

DET_INPUT_W = 320
DET_INPUT_H = 320
CONF_THRESH = 0.5
IOU_THRESH  = 0.45

MY_CLASS_NAMES = {
    0: "stop_s", 1: "10_limit", 2: "20_limits", 3: "bidirectional_Dx",
    4: "green_TL", 5: "yellow_TL", 6: "red_TL", 7: "person",
    8: "car_front", 9: "car_rear", 10: "car_side_left", 11: "car_side_right",
    12: "tournaround", 13: "right_turn", 14: "left_turn", 15: "bidirectional_Sx",
    16: "parking"
}


# -- Shared TensorRT engine (pycuda-free, cudart/ctypes) -----------------------

class TRTEngine:
    """
    Generic TensorRT inference via ctypes + the CUDA Runtime API (libcudart).

    No pycuda -- the runtime API shares the same implicit primary CUDA context
    that TensorRT uses, so multiple engines (detection + segmentation) created in
    this process all share one context. Mixing pycuda (driver API, a separate
    context) with this would raise "invalid resource handle".

    Assumes a single float32 input binding and a single output binding.
    `infer()` returns the output reshaped to its binding shape in its NATIVE
    dtype (int32 for the seg model, float32 for YOLO) -- no forced cast.

    Requires only: tensorrt + numpy + libcudart.so (always present on Jetson).
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

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(
                f.read())
        self.context = self.engine.create_execution_context()

        # Per-engine CUDA stream via runtime API (cudaStreamCreate).
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

        rospy.loginfo("[perception] TRTEngine loaded: %s", engine_path)
        rospy.loginfo("[perception]   input  shape: %s",
                      tuple(self.engine.get_binding_shape(
                          list(self.engine)[self.in_idx[0]])))
        rospy.loginfo("[perception]   output shape: %s", self.out_shapes[0])

    def _rt(self, result):
        """Check cudaError_t -- 0 = cudaSuccess."""
        if result != 0:
            raise RuntimeError("CUDA runtime error code: %s" % result)

    def infer(self, inp: np.ndarray) -> np.ndarray:
        """
        inp: contiguous float32 array matching the input binding size.
        Returns: output array reshaped to the output binding shape, native dtype.
        """
        ii = self.in_idx[0]
        oi = self.out_idx[0]

        h_in = self.host_bufs[ii][1]
        n_in = inp.size
        np.copyto(h_in[:n_in], inp.ravel())  # inp is already float32

        # cudaMemcpyHostToDevice = 1
        self._rt(self.rt.cudaMemcpyAsync(
            self.dev_bufs[ii], self.host_bufs[ii][0],
            n_in * 4, ctypes.c_int(1), self.stream))

        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.value)

        n_out = int(np.prod(self.out_shapes[0]))
        item_size = self.host_bufs[oi][2]
        # cudaMemcpyDeviceToHost = 2
        self._rt(self.rt.cudaMemcpyAsync(
            self.host_bufs[oi][0], self.dev_bufs[oi],
            n_out * item_size, ctypes.c_int(2), self.stream))
        self._rt(self.rt.cudaStreamSynchronize(self.stream))

        # Return a copy in native dtype reshaped to the binding shape.
        return self.host_bufs[oi][1][:n_out].reshape(self.out_shapes[0]).copy()


# -- Segmentation ONNX backend -------------------------------------------------

class ONNXBackend:
    """Inference via onnxruntime -- works on any machine without TensorRT."""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if self._has_cuda() else ["CPUExecutionProvider"])
        self.sess       = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        rospy.loginfo("[perception] ONNX backend loaded: %s", model_path)
        rospy.loginfo("[perception] Providers: %s", self.sess.get_providers())

    def infer(self, img_chw: np.ndarray) -> np.ndarray:
        """img_chw: float32 (3, H, W) -> int class mask (H, W)."""
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


# -- Segmentation preprocessing ------------------------------------------------

def preprocess_inplace(img_rgb: np.ndarray, crop_top_frac: float,
                       out_buf: np.ndarray, gpu_src=None):
    """
    Crop top, resize to model input, normalize with ImageNet stats.
    Writes the result into out_buf (1, 3, H, W) float32. If gpu_src
    (pre-allocated cv2.cuda_GpuMat) is provided, the resize runs on GPU.
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


# -- Object-detection pre/post processing --------------------------------------

def preprocess_detection(img_rgb: np.ndarray) -> np.ndarray:
    """
    Build the YOLO input blob from an RGB frame.

    The original unidrive_brain fed a BGR frame with swapRB=True (net effect:
    RGB). We already have RGB, so swapRB=False yields the identical input.
    Returns float32 (1, 3, DET_INPUT_H, DET_INPUT_W).
    """
    return cv2.dnn.blobFromImage(
        img_rgb, 1.0 / 255.0, (DET_INPUT_W, DET_INPUT_H),
        swapRB=False, crop=False)


def postprocess_detections(frame_shape, out_array):
    """
    Decode YOLO output into a list of {class_name, score, box} dicts.

    out_array: engine output reshaped to its binding shape, e.g.
               (1, 4 + n_classes, n_anchors). Boxes are scaled from the
               320x320 detector input back to the full frame_shape.
    """
    out = out_array[0]
    boxes_raw, classes_raw = out[:4, :], out[4:, :]
    scores = np.amax(classes_raw, axis=0)
    mask = scores > CONF_THRESH
    filtered_scores = scores[mask]

    if len(filtered_scores) == 0:
        return []

    filtered_boxes = boxes_raw[:, mask]
    filtered_classes_raw = classes_raw[:, mask]
    filtered_class_ids = np.argmax(filtered_classes_raw, axis=0)

    y_scale, x_scale = frame_shape[0] / DET_INPUT_H, frame_shape[1] / DET_INPUT_W

    cx, cy = filtered_boxes[0, :], filtered_boxes[1, :]
    w, h = filtered_boxes[2, :], filtered_boxes[3, :]

    left   = ((cx - w / 2) * x_scale).astype(np.int32)
    top    = ((cy - h / 2) * y_scale).astype(np.int32)
    width  = (w * x_scale).astype(np.int32)
    height = (h * y_scale).astype(np.int32)

    boxes_list     = np.column_stack((left, top, width, height)).tolist()
    scores_list    = filtered_scores.tolist()
    class_ids_list = filtered_class_ids.tolist()

    indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, CONF_THRESH, IOU_THRESH)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                "class_name": MY_CLASS_NAMES[class_ids_list[i]],
                "score": scores_list[i],
                "box": boxes_list[i]
            })
    return detections


def draw_detections(frame_bgr: np.ndarray, detections) -> np.ndarray:
    """Draw bounding boxes + labels on a BGR frame (in place) and return it."""
    for det in detections:
        x, y, w, h = det["box"]
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_bgr, det["class_name"], (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame_bgr


# -- Debug visualization (segmentation only, no movement info) -----------------

def make_debug_image(img_bgr: np.ndarray, mask: np.ndarray,
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


# -- Merged perception node ----------------------------------------------------

class PerceptionNode:

    def __init__(self, args):
        rospy.init_node("perception_node", anonymous=False)

        self.mode = args.mode
        self.run_detection    = self.mode in ("both", "detection")
        self.run_segmentation = self.mode in ("both", "segmentation")
        self.debug       = args.debug
        self.print_debug = args.print_debug
        self.crop_top_frac = CROP_TOP_FRAC

        self.max_fps = args.max_fps
        self._min_period = (1.0 / args.max_fps) if args.max_fps > 0 else 0.0

        # -- Detection model ---------------------------------------------------
        self.det_engine = None
        if self.run_detection:
            self.det_engine = TRTEngine(args.det_model)

        # -- Segmentation model + calibration ----------------------------------
        self.seg_model  = None
        self.auto_calib = None
        if self.run_segmentation:
            if args.seg_tensorrt:
                self.seg_model = TRTEngine(args.seg_model)
            else:
                self.seg_model = ONNXBackend(args.seg_model)

            self.calib_lane_label = CLASS_LANE_MARKING
            self.log_calibration_once = True
            points, angle = self.check_calibration(args.calibration)
            if points is None:
                self.auto_calib = AutoCalibration(TOP_LINE, BOTTOM_LINE)
                self.pending_calibration = self._prompt_calibration()
                if not self.pending_calibration:
                    rospy.signal_shutdown("Calibration declined")
                    raise SystemExit(0)
            else:
                rospy.loginfo(
                    "[perception] Loaded calibration from: %s", args.calibration)
                self.auto_calib = AutoCalibration(
                    TOP_LINE, BOTTOM_LINE, last_src_pts=points, calib_angle=angle)
                self.pending_calibration = False

            # Seg input buffer
            self._inp_buf = np.empty((1, 3, MODEL_H, MODEL_W), dtype=np.float32)

            # Pre-allocated GPU source buffer for the seg resize
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self._gpu_crop_src = cv2.cuda_GpuMat()
                rospy.loginfo("[perception] CUDA resize enabled")
            else:
                self._gpu_crop_src = None
                rospy.logwarn("[perception] CUDA not available -- resize on CPU")

        # -- Publishers --------------------------------------------------------
        if self.run_detection:
            # Annotated video for the dashboard (Image) ...
            self.detection_video_pub = rospy.Publisher(
                DETECTION_VIDEO_TOPIC, Image, queue_size=1)
            # ... and the structured detections for the orchestrator (JSON String).
            self.detection_info_pub = rospy.Publisher(
                DETECTION_DRIVING_INFO, String, queue_size=5)
        if self.run_segmentation:
            self.lane_mask_bev_pub = rospy.Publisher(
                LANE_MASK_BEV_TOPIC, Image, queue_size=1)
        if self.debug and self.run_segmentation:
            self.debug_pub = rospy.Publisher(DEBUG_TOPIC, Image, queue_size=1)

        # -- Shared frame buffer (camera cb writes, inference thread reads) ----
        self.frames_total = 0
        self._frame_lock  = threading.Lock()
        self._frame_event = threading.Event()
        self._latest_frame = None   # (img_rgb, header)
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="perception_inference")
        self._inference_thread.start()

        self.sub = rospy.Subscriber(
            CAMERA_TOPIC, Image, self.image_cb,
            queue_size=1, buff_size=2**24)

        rospy.loginfo("[perception] Node started (mode=%s)", self.mode)
        rospy.loginfo("[perception] detection=%s segmentation=%s debug=%s",
                      self.run_detection, self.run_segmentation, self.debug)
        rospy.on_shutdown(self._on_shutdown)

    # -- Calibration helpers ---------------------------------------------------

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
                rospy.loginfo("[perception] Calibration loaded from: %s", path)
                return points, angle
            except Exception as e:
                rospy.logwarn(
                    "[perception] Failed to load calibration %s: %s", path, e)
        return None, None

    # -- Camera callback (lightweight) -----------------------------------------

    def image_cb(self, msg: Image):
        """Convert the ROS image and store it; inference runs in a thread."""
        try:
            n_ch = {"rgb8": 3, "bgr8": 3, "mono8": 1,
                    "rgba8": 4, "bgra8": 4}.get(msg.encoding, 3)
            img_raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, n_ch)
            rospy.loginfo_once("[perception] Camera encoding: %s shape=%s",
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
            rospy.logwarn_throttle(
                5, "[perception] image conversion failed: %s", e)
            return

        with self._frame_lock:
            self._latest_frame = (img_rgb, msg.header)
        self._frame_event.set()

    # -- Inference loop (background thread) -------------------------------------

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

            # 1. Object detection on the FULL frame
            if self.run_detection:
                self._run_detection(img_rgb, header)

            # 2. Lane segmentation on the bottom CROP of the same frame
            if self.run_segmentation:
                self._run_segmentation(img_rgb, header)

            self.frames_total += 1

            elapsed_ms = (time.time() - t0) * 1000
            if self.print_debug:
                rospy.loginfo_throttle(
                    1, "[perception] fps=%.1f frames=%d",
                    1000.0 / max(elapsed_ms, 1), self.frames_total)
            else:
                rospy.loginfo_once("[perception] Started inference")

            if self._min_period > 0:
                remaining = self._min_period - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)

    def _run_detection(self, img_rgb, header):
        # Nobody watching the video -> skip inference + draw + encode entirely.
        #if self.detection_pub.get_num_connections() == 0:
        #    return
        try:
            blob = preprocess_detection(img_rgb)
            out  = self.det_engine.infer(blob)
            detections = postprocess_detections(img_rgb.shape, out)
        except Exception as e:
            rospy.logwarn_throttle(5, "[perception] detection failed: %s", e)
            return

        # Structured detections for the orchestrator (always published).
        self.detection_driving_topic(detections)

        # Annotated video for the dashboard -- skip the draw/encode if unwatched.
        if self.detection_video_pub.get_num_connections() > 0:
            frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            draw_detections(frame_bgr, detections)
            self.detection_video_pub.publish(
                self._make_bgr8_msg(frame_bgr, header))

    def _run_segmentation(self, img_rgb, header):
        preprocess_inplace(img_rgb, self.crop_top_frac, self._inp_buf,
                           self._gpu_crop_src)
        try:
            mask = self.seg_model.infer(self._inp_buf[0])
        except Exception as e:
            rospy.logerr("[perception] segmentation inference failed: %s", e)
            return

        # mask may be (H, W) (ONNX) or (1, H, W) (TRT) -- normalise to (H, W).
        if mask.ndim == 3:
            mask = mask[0]

        # Calibration on the first valid frame
        if self.pending_calibration:
            self.auto_calib.calibrate(mask, self.calib_lane_label)
            self.pending_calibration = False

        bev_mask    = self.auto_calib.make_bev(mask)
        bev_mask_u8 = np.clip(bev_mask, 0, 255).astype(np.uint8, copy=False)

        if self.log_calibration_once:
            self.auto_calib.save_debug(mask, prefix="lane_calibration")
            rospy.loginfo("[perception] Saved calibration debug images")
            self.log_calibration_once = False

        try:
            self.lane_mask_bev_pub.publish(
                self._make_mono8_msg(bev_mask_u8, header))
        except Exception as e:
            rospy.logwarn_throttle(5, "[perception] bev publish err: %s", e)

        if self.debug and self.debug_pub.get_num_connections() > 0:
            crop_px  = int(img_rgb.shape[0] * self.crop_top_frac)
            img_crop = cv2.resize(img_rgb[crop_px:], (MODEL_W, MODEL_H))
            img_disp = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
            dbg = make_debug_image(img_disp, mask, bev_mask_u8)
            try:
                self.debug_pub.publish(self._make_bgr8_msg(dbg, header))
            except Exception:
                pass

    def detection_driving_topic(self, detections):

        payload = {
            "detections": detections
        }
        json_str = json.dumps(payload)

        # Publish the JSON detections for the orchestrator.
        self.detection_info_pub.publish(json_str)


    # -- Message helpers -------------------------------------------------------

    @staticmethod
    def _make_mono8_msg(mask, header):
        msg          = Image()
        msg.header   = header
        msg.height   = mask.shape[0]
        msg.width    = mask.shape[1]
        msg.encoding = "mono8"
        msg.step     = mask.shape[1]
        msg.data     = mask.astype(np.uint8).tobytes()
        return msg

    @staticmethod
    def _make_bgr8_msg(img_bgr, header):
        msg          = Image()
        msg.header   = header
        msg.height   = img_bgr.shape[0]
        msg.width    = img_bgr.shape[1]
        msg.encoding = "bgr8"
        msg.step     = img_bgr.shape[1] * 3
        msg.data     = np.ascontiguousarray(img_bgr).tobytes()
        return msg

    def _on_shutdown(self):
        rospy.loginfo("[perception] Shutting down")

    def run(self):
        rospy.spin()


# -- Entry point ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merged perception node (object detection + lane segmentation)")
    parser.add_argument("--mode", choices=("both", "detection", "segmentation"),
                        default="both",
                        help="Which model(s) to run (default: both)")
    parser.add_argument("--det-model", dest="det_model",
                        help="YOLO .engine file (required for detection)")
    parser.add_argument("--seg-model", dest="seg_model",
                        help="Segmentation .onnx/.engine file (required for segmentation)")
    parser.add_argument("--seg-tensorrt", action="store_true", dest="seg_tensorrt",
                        help="Use TensorRT backend for the seg model (else ONNX)")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Publish %s (default: on)" % DEBUG_TOPIC)
    parser.add_argument("--nodebug", action="store_false", dest="debug",
                        help="Disable %s" % DEBUG_TOPIC)
    parser.add_argument("--calibration", default="calibration.json",
                        help="Path to calibration json file")
    parser.add_argument("--max-fps", type=float, default=0.0, dest="max_fps",
                        help="Cap the combined loop to this FPS (0 = unlimited)")
    parser.add_argument("--print-debug", action="store_true", dest="print_debug",
                        help="Print per-frame timing to console")

    # ROS passes extra args -- filter them out
    args = parser.parse_args(rospy.myargv()[1:])

    if args.mode in ("both", "detection") and not args.det_model:
        parser.error("--det-model is required for mode '%s'" % args.mode)
    if args.mode in ("both", "segmentation") and not args.seg_model:
        parser.error("--seg-model is required for mode '%s'" % args.mode)

    node = PerceptionNode(args)
    node.run()


if __name__ == "__main__":
    main()