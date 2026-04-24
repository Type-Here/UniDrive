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
    BEV warpPerspective (homography from bev_config.json)
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
    python3 lane_follower.py --model model.onnx --bev bev_config.json

    # When TensorRT engine is ready:
    python3 lane_follower.py --model model.engine --bev bev_config.json --tensorrt

Options:
    --model PATH       Path to .onnx or .engine model file
    --bev   PATH       Path to bev_config.json from calibrate_bev.py
    --speed FLOAT      Forward speed in m/s (default: 0.15)
    --kp    FLOAT      PID proportional gain (default: 1.2)
    --ki    FLOAT      PID integral gain     (default: 0.0)
    --kd    FLOAT      PID derivative gain   (default: 0.3)
    --tensorrt         Use TensorRT engine instead of ONNX runtime
    --debug            Publish debug image on /lane_follower/debug_image
    --dry-run          Run inference but do not publish cmd_vel
"""

import argparse
import json
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

# Optional debug image publisher
try:
    from sensor_msgs.msg import Image as RosImage
    HAS_ROS_IMAGE = True
except ImportError:
    HAS_ROS_IMAGE = False


# -- Configuration defaults ----------------------------------------------------

CAMERA_TOPIC  = "/depth_cam/rgb/image_raw"
CMDVEL_TOPIC  = "/jetauto_controller/cmd_vel"
DEBUG_TOPIC   = "/lane_follower/debug_image"

# Model input size -- must match training config
MODEL_H = 256
MODEL_W = 640

# ImageNet normalization (same as training pipeline)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Segmentation class ids (from config.yaml)
CLASS_ROAD         = 1
CLASS_LANE_MARKING = 2
CLASS_LANE_DASHED  = 3
CLASS_ZEBRA        = 4

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
    Inference via TensorRT -- optimal for Jetson Nano.

    Requires:
        pip install tensorrt pycuda
    The engine file must have been compiled on the same Jetson device.
    """

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.autoinit          # noqa: F401 -- initialises CUDA context
        import pycuda.driver as cuda

        self.cuda = cuda
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate host/device buffers
        self.inputs  = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size  = abs(int(np.prod(shape)))
            dtype = np.float32

            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem,
                                    "shape": shape})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem,
                                     "shape": shape})

        self.stream = cuda.Stream()
        rospy.loginfo("[lane_follower] TensorRT backend loaded: %s", engine_path)

    def infer(self, img_chw: np.ndarray) -> np.ndarray:
        inp = img_chw[np.newaxis].astype(np.float32).ravel()
        np.copyto(self.inputs[0]["host"], inp)
        self.cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        self.cuda.memcpy_dtoh_async(
            self.outputs[0]["host"],
            self.outputs[0]["device"],
            self.stream)
        self.stream.synchronize()

        out   = self.outputs[0]["host"]
        shape = self.outputs[0]["shape"]
        return out.reshape(shape)[0].astype(np.int64)   # (H, W)


# -- Image preprocessing -------------------------------------------------------

def preprocess(img_bgr: np.ndarray, crop_top_frac: float) -> np.ndarray:
    """
    Crop top, resize to model input, normalize with ImageNet stats.
    Returns float32 array (3, MODEL_H, MODEL_W) ready for inference.
    """
    h = img_bgr.shape[0]
    crop_px   = int(h * crop_top_frac)
    cropped   = img_bgr[crop_px:, :]
    resized   = cv2.resize(cropped, (MODEL_W, MODEL_H),
                           interpolation=cv2.INTER_LINEAR)
    rgb       = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalised = (rgb - IMAGENET_MEAN) / IMAGENET_STD   # (H, W, 3)
    return normalised.transpose(2, 0, 1)                 # (3, H, W)


# -- BEV + lateral error -------------------------------------------------------

class BEVProcessor:
    """
    Applies the homography from bev_config.json to the segmentation mask
    and computes the lateral error for the PID controller.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            cfg = json.load(f)

        self.H              = np.array(cfg["homography"], dtype=np.float64)
        self.bev_w          = cfg["bev_width"]
        self.bev_h          = cfg["bev_height"]
        self.crop_top_frac  = cfg["crop_top_frac"]
        self.px_per_m       = cfg["pixels_per_metre"]

        self.bev_cx = self.bev_w / 2.0   # BEV image centre x
        rospy.loginfo("[lane_follower] BEV config loaded: %s", config_path)
        rospy.loginfo("[lane_follower] BEV size: %dx%d  scale: %.1f px/m",
                      self.bev_w, self.bev_h, self.px_per_m)

    def mask_to_bev(self, mask: np.ndarray) -> np.ndarray:
        """
        Warp the segmentation mask (MODEL_H x MODEL_W) to BEV.
        Uses INTER_NEAREST to preserve integer class labels.
        """
        return cv2.warpPerspective(
            mask.astype(np.uint8), self.H,
            (self.bev_w, self.bev_h),
            flags=cv2.INTER_NEAREST)

    def lateral_error(self, bev_mask: np.ndarray) -> tuple:
        """
        Compute the lateral error from the BEV mask.

        Strategy:
            1. Extract pixels belonging to lane marking classes
               in the lower half of the BEV (the near region -- more reliable)
            2. Compute the x-centroid of those pixels
            3. Error = centroid_x - bev_cx  (positive = robot is left of center)
            4. Normalize to [-1, 1] by dividing by half BEV width

        Returns:
            error_norm  -- float in [-1, 1], positive means steer right
            n_pixels    -- number of lane pixels found (used for confidence check)
        """
        # Focus on lower half of BEV (near road, more reliable)
        roi = bev_mask[self.bev_h // 2:, :]

        lane_mask = np.zeros_like(roi, dtype=bool)
        for cls in LANE_CLASSES:
            lane_mask |= (roi == cls)

        n_pixels = int(lane_mask.sum())
        if n_pixels < MIN_LANE_PIXELS:
            return 0.0, n_pixels

        # x-coordinates of all lane pixels
        xs          = np.where(lane_mask)[1].astype(np.float32)
        centroid_x  = float(xs.mean())
        error_px    = centroid_x - self.bev_cx
        error_norm  = error_px / (self.bev_w / 2.0)   # normalize to [-1, 1]

        return float(np.clip(error_norm, -1.0, 1.0)), n_pixels


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

        # Load BEV processor
        self.bev = BEVProcessor(args.bev)

        # PID controller
        self.pid = PIDController(
            kp=args.kp, ki=args.ki, kd=args.kd,
            output_limit=1.5)

        self.speed   = args.speed
        self.dry_run = args.dry_run
        self.debug   = args.debug
        self.bridge  = CvBridge()

        # State
        self.last_error    = 0.0
        self.frames_total  = 0
        self.frames_no_lane = 0

        # Publishers
        if not self.dry_run:
            self.cmd_pub = rospy.Publisher(
                CMDVEL_TOPIC, Twist, queue_size=1)
        if self.debug:
            self.debug_pub = rospy.Publisher(
                DEBUG_TOPIC, Image, queue_size=1)

        # Subscriber -- process every frame (queue_size=1 drops old frames)
        self.sub = rospy.Subscriber(
            CAMERA_TOPIC, Image, self.image_cb,
            queue_size=1, buff_size=2**24)

        rospy.loginfo("[lane_follower] Node started")
        rospy.loginfo("[lane_follower] Speed: %.2f m/s  Kp=%.2f Ki=%.2f Kd=%.2f",
                      self.speed, args.kp, args.ki, args.kd)
        rospy.loginfo("[lane_follower] Dry-run: %s  Debug: %s",
                      self.dry_run, self.debug)
        if self.dry_run:
            rospy.logwarn("[lane_follower] DRY-RUN mode -- no cmd_vel published")

        rospy.on_shutdown(self._on_shutdown)

    # -- Camera callback -------------------------------------------------------

    def image_cb(self, msg: Image):
        t0 = time.time()

        # Convert ROS image to BGR numpy array
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(5, "[lane_follower] imgmsg_to_cv2 failed: %s", e)
            return

        # Preprocess
        img_chw = preprocess(img_bgr, self.bev.crop_top_frac)

        # Inference
        try:
            mask = self.model.infer(img_chw)   # (MODEL_H, MODEL_W) int
        except Exception as e:
            rospy.logerr("[lane_follower] Inference failed: %s", e)
            self._publish_stop()
            return

        # BEV transform
        bev_mask = self.bev.mask_to_bev(mask)

        # Lateral error
        error_norm, n_pixels = self.bev.lateral_error(bev_mask)

        self.frames_total += 1

        if n_pixels < MIN_LANE_PIXELS:
            self.frames_no_lane += 1
            rospy.logwarn_throttle(
                2, "[lane_follower] Few lane pixels (%d) -- holding last error", n_pixels)
            # Hold last known error but reduce speed
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

        # Timing
        elapsed_ms = (time.time() - t0) * 1000

        rospy.logdebug("[lane_follower] err=%.3f  ang=%.3f  px=%d  t=%.1fms",
                       error_norm, angular_z, n_pixels, elapsed_ms)
        rospy.loginfo_throttle(
            1, "[lane_follower] err=%+.3f ang=%+.3f px=%d fps=%.1f no_lane=%d/%d",
            error_norm, angular_z, n_pixels,
            1000.0 / max(elapsed_ms, 1),
            self.frames_no_lane, self.frames_total)

        # Debug image
        if self.debug and self.debug_pub.get_num_connections() > 0:
            # Resize mask to match img_bgr for display
            h_orig, w_orig = img_bgr.shape[:2]
            crop_px  = int(h_orig * self.bev.crop_top_frac)
            img_crop = img_bgr[crop_px:, :]
            img_disp = cv2.resize(img_crop, (MODEL_W, MODEL_H))
            dbg = make_debug_image(img_disp, mask, bev_mask,
                                   error_norm, n_pixels, angular_z)
            try:
                dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
                dbg_msg.header = msg.header
                self.debug_pub.publish(dbg_msg)
            except Exception:
                pass

    # -- Helpers ---------------------------------------------------------------

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
    parser.add_argument("--bev",      required=True,
                        help="Path to bev_config.json")
    parser.add_argument("--speed",    type=float, default=0.15,
                        help="Forward speed m/s (default: 0.15)")
    parser.add_argument("--kp",       type=float, default=1.2)
    parser.add_argument("--ki",       type=float, default=0.0)
    parser.add_argument("--kd",       type=float, default=0.3)
    parser.add_argument("--tensorrt", action="store_true",
                        help="Use TensorRT backend instead of ONNX")
    parser.add_argument("--debug",    action="store_true",
                        help="Publish debug image on /lane_follower/debug_image")
    parser.add_argument("--dry-run",  action="store_true", dest="dry_run",
                        help="Run inference without publishing cmd_vel")

    # ROS passes extra args -- filter them out
    import rospy
    args = parser.parse_args(rospy.myargv()[1:])

    node = LaneFollowerNode(args)
    node.run()


if __name__ == "__main__":
    main()