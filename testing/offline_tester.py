#!/usr/bin/env python3
"""
offline_tester.py - Offline test of the driving pipeline on a recorded video.

Reads an MP4 video from the robot, runs the same pipeline as the robot
without ROS, and produces an annotated video to validate the driving scripts.

Pipeline:
    video frame  ->  preprocess (crop+resize+normalize)
                 ->  ONNX inference (CoreML EP on Mac)
                 ->  BEV warp (AutoCalibration)
                 ->  LaneControllerCore (HoughLinesP + steering)
                 ->  visualization: [original | mask | BEV+overlay]

Usage:
    # With files in video/ and model/ (standard folders):
    python3 offline_tester.py

    # With explicit paths:
    python3 offline_tester.py --video video/driving.mp4 --model model/model.onnx \
        --calibration calibration.json --output output/annotated.mp4

    # Live stream from the robot + send BEV mask back for closed-loop driving:
    python3 offline_tester.py \
        --video "http://ROBOT_IP:8080/stream?topic=/depth_cam/rgb/image_raw" \
        --model model/segformer_b0.mlpackage \
        --calibration calibration.json \
        --robot-ip ROBOT_IP

Standard folders (inside testing/):
    video/              -> default --video  (looks for a single .mp4)
    model/              -> default --model  (looks for a single .onnx/.mlpackage)
    calibration.json    -> default --calibration (single JSON file, not a subdirectory)
    output/             -> default --output and calib_debug_*.jpg

Options:
    --video       PATH   Input video (.mp4 or any cv2-supported format).
                         Also accepts a URL - use this to stream directly from the robot:
                         http://<robot_ip>:8080/stream?topic=/depth_cam/rgb/image_raw
                         (web_video_server must be running on the robot, started by start_all.sh)
    --model       PATH   ONNX model (.onnx) or CoreML (.mlpackage)
    --calibration PATH   BEV calibration JSON (from auto_calibration.py).
                         If absent, calibrates automatically on the first valid frame.
    --output      PATH   Annotated output video (default: output/<video_name>_output.mp4)
    --params      PATH   lane_params.yaml (default: ../jetauto_autonomous/config/lane_params.yaml)
    --crop-top    FLOAT  Top fraction to crop (default: 0.45)
    --no-display         Do not open an OpenCV window (useful headless / on a server)
    --max-fps     FLOAT  Limit processing to N fps (0 = no limit)
    --start-frame INT    Start from the given frame index
"""

import argparse
import base64
import glob
import math
import os
import sys
import time

import cv2
import numpy as np

_IS_MACOS = sys.platform == "darwin"

if _IS_MACOS:
    try:
        import coremltools as ct
        _COREML_AVAILABLE = True
    except ImportError:
        _COREML_AVAILABLE = False
        print("[offline_tester] WARN: coremltools not installed - .mlpackage backend unavailable "
              "(pip install coremltools)")
else:
    ct = None
    _COREML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants (must match those in lane_follower.py)
# ---------------------------------------------------------------------------

MODEL_H = 256
MODEL_W = 640
CROP_TOP_FRAC = 0.45

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_LANE_MARKING = 2
CLASS_LANE_DASHED  = 3

# RGB colors per class (same palette as lane_follower.py)
CLASS_COLORS_RGB = np.array([
    [0,   0,   0],    # 0 background
    [180, 130,  70],  # 1 road
    [0,   255, 255],  # 2 lane_marking
    [255, 255,   0],  # 3 lane_dashed
    [0,   0,   255],  # 4 zebra
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# ONNX backend (Mac Apple Silicon: prefers CoreML EP -> Neural Engine)
# ---------------------------------------------------------------------------

class ONNXBackend:
    # Fallback chain: CPU first (faster for models with many non-CoreML nodes),
    # then CoreML GPU if explicitly requested with --coreml.
    def __init__(self, model_path: str, use_coreml: bool = False):
        import onnxruntime as ort
        self._ort        = ort
        self._model_path = model_path

        available = ort.get_available_providers()
        self._provider_chain = [("CPU", ["CPUExecutionProvider"])]
        has_coreml = _IS_MACOS and "CoreMLExecutionProvider" in available
        if has_coreml:
            self._provider_chain.append(
                ("CoreML GPU", [("CoreMLExecutionProvider", {"MLComputeUnits": "CPUAndGPU"}),
                                "CPUExecutionProvider"])
            )

        self._chain_idx = 0
        if use_coreml:
            if not _IS_MACOS:
                print("[offline_tester] WARN: --coreml ignored (macOS only)")
            elif not has_coreml:
                print("[offline_tester] WARN: CoreMLExecutionProvider not available, falling back to CPU")
            else:
                self._chain_idx = 1   # start from CoreML GPU

        self._create_session()

    def _create_session(self):
        label, providers = self._provider_chain[self._chain_idx]
        self.sess       = self._ort.InferenceSession(self._model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        print(f"[offline_tester] Provider: {label}  ->  {self.sess.get_providers()}")

    def _next_fallback(self):
        self._chain_idx += 1
        if self._chain_idx >= len(self._provider_chain):
            raise RuntimeError("All ONNX providers failed")
        label = self._provider_chain[self._chain_idx][0]
        print(f"[offline_tester] Fallback -> {label}")
        self._create_session()

    def infer(self, img_chw: np.ndarray) -> np.ndarray:
        """img_chw: float32 (3, H, W)  ->  int array (H, W) with class IDs"""
        inp = img_chw[np.newaxis]
        try:
            return self.sess.run(None, {self.input_name: inp})[0][0]
        except Exception as e:
            if self._chain_idx < len(self._PROVIDER_CHAIN) - 1:
                self._next_fallback()
                return self.sess.run(None, {self.input_name: inp})[0][0]
            raise e

    @property
    def provider_label(self) -> str:
        return self._provider_chain[self._chain_idx][0]


# ---------------------------------------------------------------------------
# Native CoreML backend (.mlpackage) - Neural Engine on Apple Silicon
# ---------------------------------------------------------------------------

class CoreMLBackend:
    def __init__(self, model_path: str):
        if not _IS_MACOS:
            raise RuntimeError("CoreML backend is only available on macOS")
        if not _COREML_AVAILABLE:
            raise RuntimeError("coremltools not installed: pip install coremltools")
        self._model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
        print(f"[offline_tester] [ANE] CoreML model loaded: {model_path}")

    def infer(self, img_chw: np.ndarray) -> np.ndarray:
        """img_chw: float32 (3, H, W)  ->  int array (H, W) with class IDs"""
        inp = img_chw[np.newaxis]   # (1, 3, H, W)
        result = self._model.predict({"pixel_values": inp})
        return list(result.values())[0][0]  # (H, W)

    @property
    def provider_label(self) -> str:
        return "ANE"


# ---------------------------------------------------------------------------
# Preprocessing (identical to preprocess() in lane_follower.py)
# ---------------------------------------------------------------------------

def preprocess(img_rgb: np.ndarray, crop_top_frac: float) -> np.ndarray:
    """Returns float32 (3, MODEL_H, MODEL_W) normalized with ImageNet stats."""
    h       = img_rgb.shape[0]
    crop_px = int(h * crop_top_frac)
    cropped = img_rgb[crop_px:, :]
    resized = cv2.resize(cropped, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
    normalised = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return normalised.transpose(2, 0, 1)   # HWC -> CHW


# ---------------------------------------------------------------------------
# BEV calibration
# ---------------------------------------------------------------------------

def load_auto_calibration(calib_path: str, top_line: int, bottom_line: int):
    """
    Load AutoCalibration from the JSON produced by auto_calibration.py.
    If the file does not exist, returns an uncalibrated instance
    (automatic calibration on the first valid frame).
    """
    import json
    from auto_calibration import AutoCalibration

    if os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            data = json.load(f)
        src_pts = np.float32(data["src_points"])
        angle   = float(data["calibration_angle"])
        print(f"[offline_tester] Calibration loaded: {calib_path}")
        return AutoCalibration(top_line, bottom_line,
                               last_src_pts=src_pts, calib_angle=angle), False
    else:
        print(f"[offline_tester] {calib_path} not found - automatic calibration on first frame")
        return AutoCalibration(top_line, bottom_line), True


# ---------------------------------------------------------------------------
# Lane controller - logic in jetauto_autonomous/scripts/lane_core.py
# LaneControllerCore is imported in main() after adding the path.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

def make_debug_frame(orig_crop_bgr: np.ndarray,
                     mask: np.ndarray,
                     bev_mask: np.ndarray,
                     steering: float,
                     angular_z: float,
                     state: str,
                     info: dict,
                     frame_idx: int,
                     fps: float) -> np.ndarray:
    """
    Produce a side-by-side frame [original | mask+overlay | BEV+Hough].
    All panels are MODEL_W × MODEL_H.
    """
    h, w = MODEL_H, MODEL_W

    # Panel 1: original frame (top cropped, resized to model size)
    p1 = cv2.resize(orig_crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    # Panel 2: colored mask overlaid on the original
    colored    = CLASS_COLORS_RGB[mask.clip(0, 4)]
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    p2 = cv2.addWeighted(p1, 0.55, colored_bgr, 0.45, 0)

    # Panel 3: colored BEV with HoughLinesP overlay + steering arrow
    bev_colored = CLASS_COLORS_RGB[bev_mask.clip(0, 4)]
    p3 = cv2.cvtColor(bev_colored, cv2.COLOR_RGB2BGR)

    left_line   = info["left_line"]
    right_line  = info["right_line"]
    valid_l     = info["valid_l"]
    valid_r     = info["valid_r"]
    lane_center = info["lane_center"]
    center_y    = info["center_y"]
    roi_top_px  = info["roi_top_px"]

    # Hough lines
    if valid_l and left_line:
        x1, y1, x2, y2 = left_line
        cv2.line(p3, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(p3, "L", (max(x1-18, 0), y1+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if valid_r and right_line:
        x1, y1, x2, y2 = right_line
        cv2.line(p3, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(p3, "R", (x1+5, y1+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Red vertical line: image center (splits left and right)
    cx = w // 2
    cv2.line(p3, (cx, 0), (cx, h), (0, 0, 255), 1)

    # Orange line: Hough ROI upper limit
    if roi_top_px > 0:
        cv2.line(p3, (0, roi_top_px), (w-1, roi_top_px), (0, 165, 255), 1)

    # Magenta line + circle: measurement point and lane center
    if center_y is not None:
        cv2.line(p3, (0, center_y), (w-1, center_y), (255, 0, 255), 1)
    if lane_center is not None and center_y is not None:
        cv2.circle(p3, (int(lane_center), center_y), 6, (255, 0, 255), -1)

    # Steering arrow (green <15°, yellow <30°, red >=30°)
    arr_cx, arr_by = w // 2, h - 8
    angle_rad = math.radians(steering)
    arr_tx = int(arr_cx + 40 * math.sin(angle_rad))
    arr_ty = int(arr_by - 40 * math.cos(angle_rad))
    arr_color = (0, 255, 0) if abs(steering) < 15 else ((0, 255, 255) if abs(steering) < 30 else (0, 0, 255))
    cv2.arrowedLine(p3, (arr_cx, arr_by), (arr_tx, arr_ty), arr_color, 2, tipLength=0.3)

    # Panel 3 overlay text
    cv2.putText(p3, f"{steering:+.1f}deg  {state}",
                (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, arr_color, 1, cv2.LINE_AA)
    cv2.putText(p3, f"wz={angular_z:+.3f} rad/s",
                (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(p3, f"f={frame_idx}  fps={fps:.1f}",
                (5, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    # Column labels at the bottom of each panel
    for panel, label in [(p1, "ORIGINALE"), (p2, "MASCHERA"), (p3, "BEV+HOUGH")]:
        cv2.putText(panel, label, (5, panel.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Normalize panel 3 height (may differ from MODEL_H when bev_scale>1)
    if p3.shape[0] != h or p3.shape[1] != w:
        p3 = cv2.resize(p3, (w, h), interpolation=cv2.INTER_AREA)

    return np.hstack([p1, p2, p3])


# ---------------------------------------------------------------------------
# Interactive calibration frame selection
# ---------------------------------------------------------------------------

def _interactive_calib_select(cap, model, args):
    """
    Shows frames one at a time.
      n / -> / SPACE  ->  advance one frame
      y / ENTER      ->  calibrate on this frame
      q / ESC        ->  exit the program

    Returns (frame_idx, frame_bgr, frame_rgb, mask) for the chosen frame,
    or None if the user pressed q.
    """
    frame_idx   = args.start_frame
    WIN         = "offline_tester  [q=esci]"
    FONT        = cv2.FONT_HERSHEY_SIMPLEX

    print("[offline_tester] Calibration frame selection: "
          "n=advance  y=calibrate here  q=exit")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("[offline_tester] End of video reached during calibration selection")
            return None

        frame_idx += 1
        frame_rgb = frame_bgr[:, :, ::-1]
        img_chw   = preprocess(frame_rgb, args.crop_top)
        mask      = model.infer(img_chw).astype(np.int64)

        # Display: original | colored mask
        crop_px  = int(frame_rgb.shape[0] * args.crop_top)
        orig_dis = cv2.resize(frame_bgr[crop_px:, :], (MODEL_W, MODEL_H))
        colored  = CLASS_COLORS_RGB[mask.clip(0, 4).astype(np.uint8)]
        mask_dis = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        panel    = np.hstack([orig_dis, mask_dis,
                              np.zeros((MODEL_H, MODEL_W, 3), dtype=np.uint8)])

        n_lane = int(((mask == CLASS_LANE_MARKING) | (mask == CLASS_LANE_DASHED)).sum())
        color_hint = (0, 255, 0) if n_lane >= 50 else (0, 100, 255)
        cv2.putText(panel, f"frame {frame_idx}   lane_px={n_lane}   "
                           f"[n=advance  y=calibrate  q=exit]",
                    (8, 18), FONT, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "OK: enough pixels" if n_lane >= 50 else "WARN: too few lane pixels",
                    (8, 34), FONT, 0.42, color_hint, 1, cv2.LINE_AA)

        cv2.imshow(WIN, panel)
        k = cv2.waitKey(0)

        if k in (ord('n'), ord(' '), 83, 0xFF & ord('n')):   # n / space / ->
            continue
        if k in (ord('y'), 13):    # y / ENTER -> calibrate
            return frame_idx, frame_bgr, frame_rgb, mask
        if k in (ord('q'), 27):    # q / ESC -> exit
            return None


# ---------------------------------------------------------------------------
# BEV calibration visualization (like the robot when you press 'y')
# ---------------------------------------------------------------------------

def _show_calib_debug(auto_calib, mask: np.ndarray, frame_bgr: np.ndarray,
                      frame_rgb: np.ndarray, args, calib_prefix: str,
                      cap, writer) -> bool:
    """
    Saves calibration debug JPGs and shows the confirmation window.
    Returns False if the user pressed q/ESC (exit signal), True otherwise.
    """
    mask_u8 = np.clip(mask, 0, 255).astype(np.uint8)

    # Save _points.jpg and _warp.jpg in the same folder as the output
    auto_calib.save_debug(mask_u8, prefix=calib_prefix)
    print(f"[offline_tester] Calibration debug -> {calib_prefix}_points.jpg  "
          f"{calib_prefix}_warp.jpg")

    if args.no_display:
        return True

    pts_data = auto_calib._compute_warp_points(mask_u8)
    if pts_data is None:
        return True

    _, src_pts, _ = pts_data
    crop_px = int(frame_rgb.shape[0] * args.crop_top)

    # Left panel: original frame with the 4 points in green
    orig_vis = cv2.resize(frame_bgr[crop_px:, :], (MODEL_W, MODEL_H))
    for x, y in src_pts:
        cv2.circle(orig_vis, (int(x), int(y)), 8, (0, 255, 0), 2)
        cv2.circle(orig_vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Center panel: colored mask with the 4 points in white
    vis_bgr = cv2.cvtColor(CLASS_COLORS_RGB[mask_u8.clip(0, 4)], cv2.COLOR_RGB2BGR)
    vis_bgr = cv2.resize(vis_bgr, (MODEL_W, MODEL_H))
    for x, y in src_pts:
        cv2.circle(vis_bgr, (int(x), int(y)), 8, (255, 255, 255), -1)

    # Right panel: resulting BEV
    bev_now = auto_calib.make_bev(mask)
    bev_bgr = cv2.resize(
        cv2.cvtColor(CLASS_COLORS_RGB[np.clip(bev_now, 0, 4).astype(np.uint8)],
                     cv2.COLOR_RGB2BGR),
        (MODEL_W, MODEL_H))

    calib_frame = np.hstack([orig_vis, vis_bgr, bev_bgr])
    cv2.putText(calib_frame, "BEV CALIBRATION  [ENTER/SPACE=continue  q=exit]",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(calib_frame,
                f"Saved: {calib_prefix}_points.jpg   |   {calib_prefix}_warp.jpg",
                (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow("offline_tester  [q=esci]", calib_frame)
    while True:
        k = cv2.waitKey(0)
        if k in (13, ord(' ')):   # ENTER or SPACE -> continue
            return True
        if k in (ord('q'), 27):   # q or ESC -> exit
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("[offline_tester] Exiting after calibration.")
            return False


# ---------------------------------------------------------------------------
# Publisher rosbridge -> /lane_mask_bev  (sensor_msgs/Image mono8)
# ---------------------------------------------------------------------------

class RosBridgePublisher:
    """
    Publishes the BEV mask on /lane_mask_bev via rosbridge_websocket (roslibpy).
    Does not require ROS on the PC - uses only WebSocket JSON.
    lane_controller_node.py on the robot receives the mask and computes cmd_vel.
    """

    def __init__(self, host: str, port: int = 9090, topic: str = "/lane_mask_bev"):
        try:
            import roslibpy
        except ImportError:
            raise ImportError("roslibpy not found: pip install roslibpy")
        self._ros = roslibpy.Ros(host=host, port=port)
        self._ros.run()
        self._pub = roslibpy.Topic(self._ros, topic, "sensor_msgs/Image")
        self._topic = topic
        print(f"[offline_tester] rosbridge connected: {host}:{port}  ->  {topic}")

    def publish(self, mask_u8: np.ndarray):
        h, w = mask_u8.shape[:2]
        now   = time.time()
        secs  = int(now)
        nsecs = int((now - secs) * 1e9)
        self._pub.publish({
            "header": {"stamp": {"secs": secs, "nsecs": nsecs}, "frame_id": "camera"},
            "height": h,
            "width":  w,
            "encoding": "mono8",
            "is_bigendian": 0,
            "step": w,
            "data": base64.b64encode(mask_u8.tobytes()).decode("ascii"),
        })

    def close(self):
        self._ros.terminate()


# ---------------------------------------------------------------------------
# YAML parameter loading
# ---------------------------------------------------------------------------

def load_params(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        print(f"[offline_tester] WARN: params not found ({yaml_path}), using defaults")
        return {}
    try:
        import yaml
    except ImportError:
        print("[offline_tester] WARN: pyyaml not installed, using defaults (pip install pyyaml)")
        return {}
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    p = data.get("lane_controller", data)
    print(f"[offline_tester] Params: {yaml_path}")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    default_params = os.path.normpath(
        os.path.join(script_dir, "..", "jetauto_autonomous", "config", "lane_params.yaml"))

    parser = argparse.ArgumentParser(
        description="Offline driving pipeline test on a recorded video (no ROS)")
    parser.add_argument("--video",       default=None,
                        help="Input video (e.g. driving.mp4); if omitted, looks in Video/")
    parser.add_argument("--model",       default=None,
                        help="ONNX model (.onnx) or CoreML (.mlpackage); if omitted, looks in model/")
    parser.add_argument("--calibration", default=os.path.normpath(
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "calibration.json")),
                        help="BEV calibration JSON (default: calibration.json)")
    parser.add_argument("--output",      default=None,
                        help="Annotated output video (e.g. annotated.mp4)")
    parser.add_argument("--params",      default=default_params,
                        help="lane_params.yaml (default: looks in the repo)")
    parser.add_argument("--crop-top",    type=float, default=CROP_TOP_FRAC, dest="crop_top",
                        help=f"Top fraction to crop (default: {CROP_TOP_FRAC})")
    parser.add_argument("--no-display",  action="store_true", dest="no_display",
                        help="Do not open an OpenCV window")
    parser.add_argument("--max-fps",     type=float, default=0.0, dest="max_fps",
                        help="Limit processing FPS (0=no limit)")
    parser.add_argument("--start-frame", type=int, default=0, dest="start_frame",
                        help="Start from the given frame index")
    parser.add_argument("--calib-frame", type=int, default=1, dest="calib_frame",
                        help="Frame on which to calibrate the BEV (default: 1 = first frame). "
                             "Use a higher number to skip unrepresentative initial frames.")
    parser.add_argument("--coreml", action="store_true",
                        help="Use CoreML GPU (Metal) instead of CPU (macOS only) - may be slower if "
                             "the model has many nodes unsupported by CoreML")
    parser.add_argument("--seg-only", action="store_true", dest="seg_only",
                        help="Show segmentation + BEV only, without HoughLinesP/steering. "
                             "Useful for evaluating model quality in isolation.")
    parser.add_argument("--robot-ip", default=None, dest="robot_ip",
                        help="Robot IP: enables /lane_mask_bev publishing via rosbridge "
                             "(e.g. 192.168.x.x). "
                             "lane_controller_node.py on the robot will drive on the received mask.")
    parser.add_argument("--robot-port", type=int, default=9090, dest="robot_port",
                        help="rosbridge_websocket port on the robot (default: 9090)")
    args = parser.parse_args()

    # Smart default for --video: look in Video/ if not specified
    if args.video is None:
        video_dir = os.path.normpath(os.path.join(script_dir, "video"))
        candidates = glob.glob(os.path.join(video_dir, "*.mp4"))
        if len(candidates) == 1:
            args.video = candidates[0]
            print(f"[offline_tester] Video auto-detected: {args.video}")
        elif len(candidates) == 0:
            parser.error(f"No .mp4 found in {video_dir}. Use --video.")
        else:
            names = ", ".join(os.path.basename(c) for c in candidates)
            parser.error(f"Multiple videos in {video_dir} ({names}): specify --video.")

    # Smart default for --model: look in model/ if not specified
    if args.model is None:
        model_dir = os.path.normpath(os.path.join(script_dir, "model"))
        candidates = (glob.glob(os.path.join(model_dir, "*.onnx")) +
                      glob.glob(os.path.join(model_dir, "*.mlpackage")))
        if len(candidates) == 1:
            args.model = candidates[0]
            print(f"[offline_tester] Model auto-detected: {args.model}")
        elif len(candidates) == 0:
            parser.error(f"No model found in {model_dir}. Use --model.")
        else:
            names = ", ".join(os.path.basename(c) for c in candidates)
            parser.error(f"Multiple models in {model_dir} ({names}): specify --model.")

    # Default output: ../Output/<video_name>_output.mp4
    if args.output is None:
        out_dir = os.path.normpath(os.path.join(script_dir, "output"))
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(out_dir, f"{stem}_output.mp4")
        print(f"[offline_tester] Default output: {args.output}")

    # Ensure auto_calibration.py is importable (lives in drive_segm/)
    sys.path.insert(0, os.path.normpath(
        os.path.join(script_dir, "..", "on_jetauto_scripts", "drive_segm")))

    # Lane controller core (lane_core.py in jetauto_autonomous/scripts/)
    sys.path.insert(0, os.path.normpath(
        os.path.join(script_dir, "..", "jetauto_autonomous", "scripts")))
    from lane_core import LaneControllerCore  # noqa: E402

    # Load model (automatic selection by file extension)
    if os.path.splitext(args.model)[1].lower() == ".mlpackage":
        model = CoreMLBackend(args.model)
    else:
        model = ONNXBackend(args.model, use_coreml=args.coreml)

    # Load lane controller parameters
    params     = load_params(args.params)
    controller = LaneControllerCore(params)

    # rosbridge publisher (optional)
    ros_pub = None
    if args.robot_ip:
        ros_pub = RosBridgePublisher(args.robot_ip, args.robot_port)
        print(f"[offline_tester] Publishing /lane_mask_bev to {args.robot_ip}:{args.robot_port}")

    # BEV calibration
    top_line    = MODEL_H - (MODEL_H // 2)   # 192
    bottom_line = MODEL_H - 10               # 246
    auto_calib, pending_calib = load_auto_calibration(
        args.calibration, top_line, bottom_line)

    # Folder for calibration debug images (next to the output file)
    calib_prefix = os.path.join(
        os.path.dirname(os.path.abspath(args.output)),
        "calib_debug")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[offline_tester] ERROR: cannot open {args.video}")
        sys.exit(1)

    vid_fps      = cap.get(cv2.CAP_PROP_FPS) or 15.0   # 0 on live stream -> default 15
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 0 on live stream
    is_live      = total_frames <= 0
    total_label  = "live" if is_live else str(total_frames)
    print(f"[offline_tester] {'Live stream' if is_live else 'Video'}: "
          f"{vid_w}x{vid_h} @ {vid_fps:.1f}fps  "
          f"{'(Ctrl+C or q to stop)' if is_live else total_label + ' frames'}")

    if args.start_frame > 0 and not is_live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        print(f"[offline_tester] Starting from frame {args.start_frame}")

    # Writer output
    writer = None
    if args.output:
        out_w  = MODEL_W * 3
        out_h  = MODEL_H
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, vid_fps, (out_w, out_h))
        print(f"[offline_tester] Output: {args.output}  ({out_w}x{out_h} @ {vid_fps:.0f}fps)")

    frame_idx = args.start_frame   # always initialised before conditional blocks

    # Interactive calibration (only if not loaded from file)
    if pending_calib and not args.no_display:
        result = _interactive_calib_select(cap, model, args)
        if result is None:
            cap.release()
            cv2.destroyAllWindows()
            return
        calib_frame_idx, calib_bgr, calib_rgb, calib_mask = result
        auto_calib.calibrate(calib_mask, CLASS_LANE_MARKING)
        pending_calib = False
        print(f"[offline_tester] BEV calibration on frame {calib_frame_idx}")
        if not _show_calib_debug(auto_calib, calib_mask, calib_bgr, calib_rgb, args,
                                 calib_prefix, cap, writer):
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            return
        frame_idx = calib_frame_idx   # loop resumes from the next frame
    elif pending_calib and args.no_display:
        # Headless: calibrate on the first valid frame (previous behaviour)
        print("[offline_tester] Headless mode: automatic calibration on first valid frame")

    # Show loaded-from-file calibration debug once (at the first loop iteration)
    show_loaded_calib = not pending_calib and not args.no_display
    calib_shown = False

    min_frame_time = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    t_start        = time.time()

    print("[offline_tester] Processing... (press q in the window to quit)")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        t0 = time.time()
        frame_idx += 1

        # BGR -> RGB (same behaviour as lane_follower.py with encoding rgb8)
        frame_rgb = frame_bgr[:, :, ::-1]

        # Preprocessing
        img_chw = preprocess(frame_rgb, args.crop_top)

        # Inference ONNX
        mask = model.infer(img_chw).astype(np.int64)

        # Headless: automatic calibration on the first valid frame
        if pending_calib:
            n_lane = int(((mask == CLASS_LANE_MARKING) | (mask == CLASS_LANE_DASHED)).sum())
            if n_lane >= 50:
                auto_calib.calibrate(mask, CLASS_LANE_MARKING)
                pending_calib = False
                print(f"[offline_tester] Automatic BEV calibration (frame {frame_idx})")

        # First iteration with file-loaded calibration: show debug once
        if show_loaded_calib and not calib_shown and not pending_calib:
            calib_shown = True
            if not _show_calib_debug(auto_calib, mask, frame_bgr, frame_rgb, args,
                                     calib_prefix, cap, writer):
                return

        # BEV warp
        bev_mask   = auto_calib.make_bev(mask)
        bev_u8     = np.clip(bev_mask, 0, 255).astype(np.uint8, copy=False)

        # Publish /lane_mask_bev to the robot (if --robot-ip is set)
        if ros_pub is not None:
            try:
                ros_pub.publish(bev_u8)
            except Exception as e:
                print(f"[offline_tester] WARN rosbridge: {e}")

        # Measured FPS
        elapsed = time.time() - t_start
        fps     = (frame_idx - args.start_frame) / max(elapsed, 1e-6)

        crop_px  = int(frame_rgb.shape[0] * args.crop_top)
        orig_crop = frame_bgr[crop_px:, :]
        mask_u8   = np.clip(mask, 0, 255).astype(np.uint8)

        if args.seg_only:
            # Segmentation + BEV only, no Hough/steering
            p1 = cv2.resize(orig_crop, (MODEL_W, MODEL_H))
            colored_bgr = cv2.cvtColor(CLASS_COLORS_RGB[mask_u8.clip(0, 4)],
                                       cv2.COLOR_RGB2BGR)
            p2 = cv2.addWeighted(p1, 0.55, colored_bgr, 0.45, 0)
            bev_bgr = cv2.cvtColor(CLASS_COLORS_RGB[bev_u8.clip(0, 4)],
                                   cv2.COLOR_RGB2BGR)
            dbg = np.hstack([p1, p2, bev_bgr])
            n_lane = int(((mask == CLASS_LANE_MARKING) | (mask == CLASS_LANE_DASHED)).sum())
            cv2.putText(dbg, f"frame={frame_idx}  fps={fps:.1f} [{model.provider_label}]  "
                             f"lane_px={n_lane}",
                        (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            for panel, label in [(p1, "ORIGINALE"), (p2, "MASCHERA"), (bev_bgr, "BEV")]:
                cv2.putText(panel, label, (5, MODEL_H - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            dbg = np.hstack([p1, p2, bev_bgr])
        else:
            # Full pipeline with HoughLinesP + steering
            # use_bev=False -> pass raw mask; BEV panel shows the perspective view
            input_mask = bev_u8 if controller.use_bev else mask_u8
            steering, angular_z, state, debug_info = controller.step(input_mask)
            viz_bev = bev_u8 if controller.use_bev else mask_u8
            # bev_scale>1: scale viz_bev to align debug_info coordinates to the panel
            if controller.bev_scale != 1.0:
                viz_bev = cv2.resize(viz_bev, None,
                                     fx=controller.bev_scale, fy=controller.bev_scale,
                                     interpolation=cv2.INTER_NEAREST)
            dbg = make_debug_frame(orig_crop, mask_u8, viz_bev,
                                   steering, angular_z, state, debug_info,
                                   frame_idx, fps)

        if writer:
            writer.write(dbg)

        if not args.no_display:
            cv2.imshow("offline_tester  [q=esci]", dbg)
            wait_ms = max(1, int((min_frame_time - (time.time() - t0)) * 1000)) if min_frame_time > 0 else 1
            if cv2.waitKey(wait_ms) in (ord('q'), 27):
                break
        elif min_frame_time > 0:
            remaining = min_frame_time - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)

        if frame_idx % 60 == 0 or frame_idx == args.start_frame + 1:
            if args.seg_only:
                print(f"  frame={frame_idx}/{total_label}  fps={fps:.1f} [{model.provider_label}]")
            else:
                print(f"  frame={frame_idx}/{total_label}  fps={fps:.1f} [{model.provider_label}]  "
                      f"steer={steering:+.1f}°  wz={angular_z:+.3f}  state={state}")

    cap.release()
    if writer:
        writer.release()
    if ros_pub is not None:
        ros_pub.close()
    cv2.destroyAllWindows()

    elapsed_total = time.time() - t_start
    n = frame_idx - args.start_frame
    print(f"[offline_tester] Done: {n} frames in {elapsed_total:.1f}s  "
          f"({n/max(elapsed_total,1e-6):.1f} avg fps)")
    if args.output:
        print(f"[offline_tester] Video saved: {args.output}")


if __name__ == "__main__":
    main()
