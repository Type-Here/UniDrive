#!/usr/bin/env python3
"""
offline_tester.py — Test offline della pipeline di guida su video registrato.

Legge un video MP4 della macchinina, esegue la stessa pipeline del robot
senza ROS e produce un video annotato per validare gli script di guida.

Pipeline:
    video frame  →  preprocess (crop+resize+normalize)
                 →  ONNX inference (CoreML EP su Mac M3)
                 →  BEV warp (AutoCalibration)
                 →  LaneControllerCore (HoughLinesP + steering)
                 →  visualizzazione: [originale | maschera | BEV+overlay]

Usage:
    # Con file in ../Video/ e ../model/ (cartelle standard):
    python3 offline_tester.py

    # Con path espliciti:
    python3 offline_tester.py --video ../Video/driving.mp4 --model ../model/model.onnx \\
        --calibration ../Calibration/calibration.json --output ../Output/annotated.mp4

Cartelle standard (dentro Testing/):
    Video/         → default --video  (cerca unico .mp4)
    model/         → default --model  (cerca unico .onnx/.mlpackage)
    Calibration/   → default --calibration
    Output/        → default --output e calib_debug_*.jpg

Options:
    --video       PATH   Video di input (.mp4 o altro formato cv2)
    --model       PATH   Modello ONNX (.onnx) o CoreML (.mlpackage)
    --calibration PATH   File JSON di calibrazione BEV (da auto_calibration.py).
                         Se assente, calibra automaticamente sul primo frame valido.
    --output      PATH   Video di output annotato (default: ../Output/<nome_video>_output.mp4)
    --params      PATH   lane_params.yaml (default: ../jetauto_autonomous/config/lane_params.yaml)
    --crop-top    FLOAT  Frazione top da croppare (default: 0.45)
    --no-display         Non aprire finestra OpenCV (utile headless / server)
    --max-fps     FLOAT  Limita l'elaborazione a N fps (0 = nessun limite)
    --start-frame INT    Inizia dall'indice di frame specificato
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
        print("[offline_tester] WARN: coremltools non installato — backend .mlpackage non disponibile "
              "(pip install coremltools)")
else:
    ct = None
    _COREML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Costanti (devono corrispondere a quelle di lane_follower.py)
# ---------------------------------------------------------------------------

MODEL_H = 256
MODEL_W = 640
CROP_TOP_FRAC = 0.45

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_LANE_MARKING = 2
CLASS_LANE_DASHED  = 3

# Colori RGB per classe (stessa palette di lane_follower.py)
CLASS_COLORS_RGB = np.array([
    [0,   0,   0],    # 0 background
    [180, 130,  70],  # 1 road
    [0,   255, 255],  # 2 lane_marking
    [255, 255,   0],  # 3 lane_dashed
    [0,   0,   255],  # 4 zebra
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Backend ONNX (Mac M3: preferisce CoreML EP → Neural Engine)
# ---------------------------------------------------------------------------

class ONNXBackend:
    # Sequenza di fallback: CPU first (più veloce per modelli con molti nodi non-CoreML),
    # poi CoreML GPU se esplicitamente richiesto con --coreml.
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
                print("[offline_tester] WARN: --coreml ignorato (solo macOS)")
            elif not has_coreml:
                print("[offline_tester] WARN: CoreMLExecutionProvider non disponibile, uso CPU")
            else:
                self._chain_idx = 1   # parte da CoreML GPU

        self._create_session()

    def _create_session(self):
        label, providers = self._provider_chain[self._chain_idx]
        self.sess       = self._ort.InferenceSession(self._model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        print(f"[offline_tester] Provider: {label}  →  {self.sess.get_providers()}")

    def _next_fallback(self):
        self._chain_idx += 1
        if self._chain_idx >= len(self._provider_chain):
            raise RuntimeError("Tutti i provider ONNX hanno fallito")
        label = self._provider_chain[self._chain_idx][0]
        print(f"[offline_tester] Fallback → {label}")
        self._create_session()

    def infer(self, img_chw: np.ndarray) -> np.ndarray:
        """img_chw: float32 (3, H, W)  →  int array (H, W) con class IDs"""
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
# Backend CoreML nativo (.mlpackage) — Neural Engine su Apple Silicon
# ---------------------------------------------------------------------------

class CoreMLBackend:
    def __init__(self, model_path: str):
        if not _IS_MACOS:
            raise RuntimeError("Backend CoreML disponibile solo su macOS")
        if not _COREML_AVAILABLE:
            raise RuntimeError("coremltools non installato: pip install coremltools")
        self._model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
        print(f"[offline_tester] [ANE] CoreML model caricato: {model_path}")

    def infer(self, img_chw: np.ndarray) -> np.ndarray:
        """img_chw: float32 (3, H, W)  →  int array (H, W) con class IDs"""
        inp = img_chw[np.newaxis]   # (1, 3, H, W)
        result = self._model.predict({"pixel_values": inp})
        return list(result.values())[0][0]  # (H, W)

    @property
    def provider_label(self) -> str:
        return "ANE"


# ---------------------------------------------------------------------------
# Preprocessing (identico a preprocess() in lane_follower.py)
# ---------------------------------------------------------------------------

def preprocess(img_rgb: np.ndarray, crop_top_frac: float) -> np.ndarray:
    """Ritorna float32 (3, MODEL_H, MODEL_W) normalizzato ImageNet."""
    h       = img_rgb.shape[0]
    crop_px = int(h * crop_top_frac)
    cropped = img_rgb[crop_px:, :]
    resized = cv2.resize(cropped, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
    normalised = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return normalised.transpose(2, 0, 1)   # HWC → CHW


# ---------------------------------------------------------------------------
# Calibrazione BEV
# ---------------------------------------------------------------------------

def load_auto_calibration(calib_path: str, top_line: int, bottom_line: int):
    """
    Carica AutoCalibration dal JSON prodotto da auto_calibration.py.
    Se il file non esiste, restituisce un'istanza non calibrata
    (calibrazione automatica sul primo frame).
    """
    import json
    from auto_calibration import AutoCalibration

    if os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            data = json.load(f)
        src_pts = np.float32(data["src_points"])
        angle   = float(data["calibration_angle"])
        print(f"[offline_tester] Calibrazione caricata: {calib_path}")
        return AutoCalibration(top_line, bottom_line,
                               last_src_pts=src_pts, calib_angle=angle), False
    else:
        print(f"[offline_tester] {calib_path} non trovato — calibrazione automatica al primo frame")
        return AutoCalibration(top_line, bottom_line), True


# ---------------------------------------------------------------------------
# Lane controller — logica in jetauto_autonomous/scripts/lane_core.py
# LaneControllerCore viene importato in main() dopo aver aggiunto il path.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Visualizzazione debug
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
    Produce un frame affiancato [originale | maschera+overlay | BEV+Hough].
    Tutti i pannelli sono MODEL_W × MODEL_H.
    """
    h, w = MODEL_H, MODEL_W

    # Pannello 1: frame originale (crop del top, resizato al modello)
    p1 = cv2.resize(orig_crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    # Pannello 2: maschera colorata sovrapposta all'originale
    colored    = CLASS_COLORS_RGB[mask.clip(0, 4)]
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    p2 = cv2.addWeighted(p1, 0.55, colored_bgr, 0.45, 0)

    # Pannello 3: BEV colorata con overlay HoughLinesP + freccia
    bev_colored = CLASS_COLORS_RGB[bev_mask.clip(0, 4)]
    p3 = cv2.cvtColor(bev_colored, cv2.COLOR_RGB2BGR)

    left_line   = info["left_line"]
    right_line  = info["right_line"]
    valid_l     = info["valid_l"]
    valid_r     = info["valid_r"]
    lane_center = info["lane_center"]
    center_y    = info["center_y"]
    roi_top_px  = info["roi_top_px"]

    # Linee Hough
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

    # Linea verticale rossa: centro immagine (divide SX e DX)
    cx = w // 2
    cv2.line(p3, (cx, 0), (cx, h), (0, 0, 255), 1)

    # Linea arancione: limite superiore ROI Hough
    if roi_top_px > 0:
        cv2.line(p3, (0, roi_top_px), (w-1, roi_top_px), (0, 165, 255), 1)

    # Linea magenta + cerchio: punto di misura e centro corsia
    if center_y is not None:
        cv2.line(p3, (0, center_y), (w-1, center_y), (255, 0, 255), 1)
    if lane_center is not None and center_y is not None:
        cv2.circle(p3, (int(lane_center), center_y), 6, (255, 0, 255), -1)

    # Freccia di sterzata (verde <15°, giallo <30°, rosso >=30°)
    arr_cx, arr_by = w // 2, h - 8
    angle_rad = math.radians(steering)
    arr_tx = int(arr_cx + 40 * math.sin(angle_rad))
    arr_ty = int(arr_by - 40 * math.cos(angle_rad))
    arr_color = (0, 255, 0) if abs(steering) < 15 else ((0, 255, 255) if abs(steering) < 30 else (0, 0, 255))
    cv2.arrowedLine(p3, (arr_cx, arr_by), (arr_tx, arr_ty), arr_color, 2, tipLength=0.3)

    # Testo overlay pannello 3
    cv2.putText(p3, f"{steering:+.1f}deg  {state}",
                (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, arr_color, 1, cv2.LINE_AA)
    cv2.putText(p3, f"wz={angular_z:+.3f} rad/s",
                (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(p3, f"f={frame_idx}  fps={fps:.1f}",
                (5, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    # Etichette colonna in basso su ogni pannello
    for panel, label in [(p1, "ORIGINALE"), (p2, "MASCHERA"), (p3, "BEV+HOUGH")]:
        cv2.putText(panel, label, (5, panel.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Normalizza altezza pannello 3 (può differire da MODEL_H con bev_scale>1)
    if p3.shape[0] != h or p3.shape[1] != w:
        p3 = cv2.resize(p3, (w, h), interpolation=cv2.INTER_AREA)

    return np.hstack([p1, p2, p3])


# ---------------------------------------------------------------------------
# Selezione interattiva del frame di calibrazione
# ---------------------------------------------------------------------------

def _interactive_calib_select(cap, model, args):
    """
    Mostra i frame uno alla volta.
      n / → / SPAZIO  →  avanza di un frame
      y / INVIO       →  calibra su questo frame
      q / ESC         →  esci dal programma

    Ritorna (frame_idx, frame_bgr, mask) del frame scelto,
    oppure None se l'utente ha premuto q.
    """
    frame_idx   = args.start_frame
    WIN         = "offline_tester  [q=esci]"
    FONT        = cv2.FONT_HERSHEY_SIMPLEX

    print("[offline_tester] Selezione frame di calibrazione: "
          "n=avanza  y=calibra qui  q=esci")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("[offline_tester] Fine video raggiunta durante la selezione calibrazione")
            return None

        frame_idx += 1
        frame_rgb = frame_bgr[:, :, ::-1]
        img_chw   = preprocess(frame_rgb, args.crop_top)
        mask      = model.infer(img_chw).astype(np.int64)

        # Mostra: originale | maschera colorata
        crop_px  = int(frame_rgb.shape[0] * args.crop_top)
        orig_dis = cv2.resize(frame_bgr[crop_px:, :], (MODEL_W, MODEL_H))
        colored  = CLASS_COLORS_RGB[mask.clip(0, 4).astype(np.uint8)]
        mask_dis = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        panel    = np.hstack([orig_dis, mask_dis,
                              np.zeros((MODEL_H, MODEL_W, 3), dtype=np.uint8)])

        n_lane = int(((mask == CLASS_LANE_MARKING) | (mask == CLASS_LANE_DASHED)).sum())
        color_hint = (0, 255, 0) if n_lane >= 50 else (0, 100, 255)
        cv2.putText(panel, f"frame {frame_idx}   lane_px={n_lane}   "
                           f"[n=avanza  y=calibra  q=esci]",
                    (8, 18), FONT, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "OK: abbastanza pixel" if n_lane >= 50 else "WARN: pochi pixel di corsia",
                    (8, 34), FONT, 0.42, color_hint, 1, cv2.LINE_AA)

        cv2.imshow(WIN, panel)
        k = cv2.waitKey(0)

        if k in (ord('n'), ord(' '), 83, 0xFF & ord('n')):   # n / spazio / →
            continue
        if k in (ord('y'), 13):    # y / INVIO → calibra
            return frame_idx, frame_bgr, frame_rgb, mask
        if k in (ord('q'), 27):    # q / ESC → esci
            return None


# ---------------------------------------------------------------------------
# Visualizzazione calibrazione BEV (come il robot quando premi 'y')
# ---------------------------------------------------------------------------

def _show_calib_debug(auto_calib, mask: np.ndarray, frame_bgr: np.ndarray,
                      frame_rgb: np.ndarray, args, calib_prefix: str,
                      cap, writer) -> bool:
    """
    Salva i debug JPG della calibrazione e mostra la finestra di conferma.
    Ritorna False se l'utente ha premuto q/ESC (segnale di uscita), True altrimenti.
    """
    mask_u8 = np.clip(mask, 0, 255).astype(np.uint8)

    # Salva _points.jpg e _warp.jpg nella stessa cartella dell'output
    auto_calib.save_debug(mask_u8, prefix=calib_prefix)
    print(f"[offline_tester] Debug calibrazione → {calib_prefix}_points.jpg  "
          f"{calib_prefix}_warp.jpg")

    if args.no_display:
        return True

    pts_data = auto_calib._compute_warp_points(mask_u8)
    if pts_data is None:
        return True

    _, src_pts, _ = pts_data
    crop_px = int(frame_rgb.shape[0] * args.crop_top)

    # Pannello sinistra: frame originale con i 4 punti in verde
    orig_vis = cv2.resize(frame_bgr[crop_px:, :], (MODEL_W, MODEL_H))
    for x, y in src_pts:
        cv2.circle(orig_vis, (int(x), int(y)), 8, (0, 255, 0), 2)
        cv2.circle(orig_vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Pannello centro: maschera colorata con i 4 punti in bianco
    vis_bgr = cv2.cvtColor(CLASS_COLORS_RGB[mask_u8.clip(0, 4)], cv2.COLOR_RGB2BGR)
    vis_bgr = cv2.resize(vis_bgr, (MODEL_W, MODEL_H))
    for x, y in src_pts:
        cv2.circle(vis_bgr, (int(x), int(y)), 8, (255, 255, 255), -1)

    # Pannello destra: BEV risultante
    bev_now = auto_calib.make_bev(mask)
    bev_bgr = cv2.resize(
        cv2.cvtColor(CLASS_COLORS_RGB[np.clip(bev_now, 0, 4).astype(np.uint8)],
                     cv2.COLOR_RGB2BGR),
        (MODEL_W, MODEL_H))

    calib_frame = np.hstack([orig_vis, vis_bgr, bev_bgr])
    cv2.putText(calib_frame, "CALIBRAZIONE BEV  [INVIO/SPAZIO=continua  q=esci]",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(calib_frame,
                f"Salvato: {calib_prefix}_points.jpg   |   {calib_prefix}_warp.jpg",
                (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow("offline_tester  [q=esci]", calib_frame)
    while True:
        k = cv2.waitKey(0)
        if k in (13, ord(' ')):   # INVIO o SPAZIO → continua
            return True
        if k in (ord('q'), 27):   # q o ESC → esci
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("[offline_tester] Uscita dopo calibrazione.")
            return False


# ---------------------------------------------------------------------------
# Publisher rosbridge → /lane_mask_bev  (sensor_msgs/Image mono8)
# ---------------------------------------------------------------------------

class RosBridgePublisher:
    """
    Pubblica la BEV mask su /lane_mask_bev via rosbridge_websocket (roslibpy).
    Non richiede ROS installato sul Mac — usa solo WebSocket JSON.
    Il lane_controller_node.py sul robot riceve la maschera e calcola cmd_vel.
    """

    def __init__(self, host: str, port: int = 9090, topic: str = "/lane_mask_bev"):
        try:
            import roslibpy
        except ImportError:
            raise ImportError("roslibpy non trovato: pip install roslibpy")
        self._ros = roslibpy.Ros(host=host, port=port)
        self._ros.run()
        self._pub = roslibpy.Topic(self._ros, topic, "sensor_msgs/Image")
        self._topic = topic
        print(f"[offline_tester] rosbridge connesso: {host}:{port}  →  {topic}")

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
# Caricamento parametri YAML
# ---------------------------------------------------------------------------

def load_params(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        print(f"[offline_tester] WARN: params non trovato ({yaml_path}), uso default")
        return {}
    try:
        import yaml
    except ImportError:
        print("[offline_tester] WARN: pyyaml non installato, uso default (pip install pyyaml)")
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
        description="Test offline pipeline guida su video registrato (senza ROS)")
    parser.add_argument("--video",       default=None,
                        help="Video di input (es. driving.mp4); se omesso cerca in Video/")
    parser.add_argument("--model",       default=None,
                        help="Modello ONNX (.onnx) o CoreML (.mlpackage); se omesso cerca in model/")
    parser.add_argument("--calibration", default=os.path.normpath(
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "Calibration", "calibration.json")),
                        help="JSON calibrazione BEV (default: Calibration/calibration.json)")
    parser.add_argument("--output",      default=None,
                        help="Video annotato di output (es. annotated.mp4)")
    parser.add_argument("--params",      default=default_params,
                        help="lane_params.yaml (default: cerca nel repo)")
    parser.add_argument("--crop-top",    type=float, default=CROP_TOP_FRAC, dest="crop_top",
                        help=f"Frazione top da croppare (default: {CROP_TOP_FRAC})")
    parser.add_argument("--no-display",  action="store_true", dest="no_display",
                        help="Non aprire finestra OpenCV")
    parser.add_argument("--max-fps",     type=float, default=0.0, dest="max_fps",
                        help="Limita FPS elaborazione (0=nessun limite)")
    parser.add_argument("--start-frame", type=int, default=0, dest="start_frame",
                        help="Inizia dall'indice di frame specificato")
    parser.add_argument("--calib-frame", type=int, default=1, dest="calib_frame",
                        help="Frame su cui calibrare la BEV (default: 1 = primo frame). "
                             "Usa un numero più alto per saltare frame iniziali poco rappresentativi.")
    parser.add_argument("--coreml", action="store_true",
                        help="Usa CoreML GPU (Metal) invece di CPU (solo macOS) — più lento se "
                             "il modello ha molti nodi non supportati da CoreML")
    parser.add_argument("--seg-only", action="store_true", dest="seg_only",
                        help="Mostra solo segmentazione + BEV senza HoughLinesP/steering. "
                             "Utile per valutare la qualità del modello in isolamento.")
    parser.add_argument("--robot-ip", default=None, dest="robot_ip",
                        help="IP del robot: abilita la pubblicazione di /lane_mask_bev "
                             "via rosbridge (es. 192.168.4.89). "
                             "Il lane_controller_node.py sul robot guiderà sulla maschera ricevuta.")
    parser.add_argument("--robot-port", type=int, default=9090, dest="robot_port",
                        help="Porta rosbridge_websocket sul robot (default: 9090)")
    args = parser.parse_args()

    # Smart default per --video: cerca in ../Video/ se non specificato
    if args.video is None:
        video_dir = os.path.normpath(os.path.join(script_dir, "Video"))
        candidates = glob.glob(os.path.join(video_dir, "*.mp4"))
        if len(candidates) == 1:
            args.video = candidates[0]
            print(f"[offline_tester] Video auto-rilevato: {args.video}")
        elif len(candidates) == 0:
            parser.error(f"Nessun .mp4 trovato in {video_dir}. Usa --video.")
        else:
            names = ", ".join(os.path.basename(c) for c in candidates)
            parser.error(f"Più video in {video_dir} ({names}): specifica --video.")

    # Smart default per --model: cerca in ../model/ se non specificato
    if args.model is None:
        model_dir = os.path.normpath(os.path.join(script_dir, "model"))
        candidates = (glob.glob(os.path.join(model_dir, "*.onnx")) +
                      glob.glob(os.path.join(model_dir, "*.mlpackage")))
        if len(candidates) == 1:
            args.model = candidates[0]
            print(f"[offline_tester] Modello auto-rilevato: {args.model}")
        elif len(candidates) == 0:
            parser.error(f"Nessun modello trovato in {model_dir}. Usa --model.")
        else:
            names = ", ".join(os.path.basename(c) for c in candidates)
            parser.error(f"Più modelli in {model_dir} ({names}): specifica --model.")

    # Default output: ../Output/<nome_video>_output.mp4
    if args.output is None:
        out_dir = os.path.normpath(os.path.join(script_dir, "Output"))
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(out_dir, f"{stem}_output.mp4")
        print(f"[offline_tester] Output di default: {args.output}")

    # Assicura che auto_calibration.py sia importabile (rimane in drive_segm/)
    sys.path.insert(0, os.path.normpath(
        os.path.join(script_dir, "..", "on_jetauto_scripts", "drive_segm")))

    # Lane controller core (lane_core.py in jetauto_autonomous/scripts/)
    sys.path.insert(0, os.path.normpath(
        os.path.join(script_dir, "..", "jetauto_autonomous", "scripts")))
    from lane_core import LaneControllerCore  # noqa: E402

    # Carica modello (selezione automatica per estensione)
    if os.path.splitext(args.model)[1].lower() == ".mlpackage":
        model = CoreMLBackend(args.model)
    else:
        model = ONNXBackend(args.model, use_coreml=args.coreml)

    # Carica parametri lane controller
    params     = load_params(args.params)
    controller = LaneControllerCore(params)

    # Publisher rosbridge (opzionale)
    ros_pub = None
    if args.robot_ip:
        ros_pub = RosBridgePublisher(args.robot_ip, args.robot_port)
        print(f"[offline_tester] Pubblicazione /lane_mask_bev verso {args.robot_ip}:{args.robot_port}")

    # Calibrazione BEV
    top_line    = MODEL_H - (MODEL_H // 2)   # 192
    bottom_line = MODEL_H - 10               # 246
    auto_calib, pending_calib = load_auto_calibration(
        args.calibration, top_line, bottom_line)

    # Cartella per i debug della calibrazione (accanto all'output se specificato)
    calib_prefix = os.path.join(
        os.path.dirname(os.path.abspath(args.output)),
        "calib_debug")

    # Apri video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[offline_tester] ERRORE: impossibile aprire {args.video}")
        sys.exit(1)

    vid_fps      = cap.get(cv2.CAP_PROP_FPS) or 15.0   # 0 su stream live → default 15
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 0 su stream live
    is_live      = total_frames <= 0
    total_label  = "live" if is_live else str(total_frames)
    print(f"[offline_tester] {'Stream live' if is_live else 'Video'}: "
          f"{vid_w}x{vid_h} @ {vid_fps:.1f}fps  "
          f"{'(Ctrl+C o q per fermare)' if is_live else total_label + ' frame'}")

    if args.start_frame > 0 and not is_live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        print(f"[offline_tester] Inizio da frame {args.start_frame}")

    # Writer output
    writer = None
    if args.output:
        out_w  = MODEL_W * 3
        out_h  = MODEL_H
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, vid_fps, (out_w, out_h))
        print(f"[offline_tester] Output: {args.output}  ({out_w}x{out_h} @ {vid_fps:.0f}fps)")

    frame_idx = args.start_frame   # sempre inizializzato prima dei blocchi condizionali

    # Calibrazione interattiva (solo se non caricata da file)
    if pending_calib and not args.no_display:
        result = _interactive_calib_select(cap, model, args)
        if result is None:
            cap.release()
            cv2.destroyAllWindows()
            return
        calib_frame_idx, calib_bgr, calib_rgb, calib_mask = result
        auto_calib.calibrate(calib_mask, CLASS_LANE_MARKING)
        pending_calib = False
        print(f"[offline_tester] Calibrazione BEV su frame {calib_frame_idx}")
        if not _show_calib_debug(auto_calib, calib_mask, calib_bgr, calib_rgb, args,
                                 calib_prefix, cap, writer):
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            return
        frame_idx = calib_frame_idx   # il loop riparte dal frame successivo
    elif pending_calib and args.no_display:
        # Headless: calibra sul primo frame valido (comportamento precedente)
        print("[offline_tester] Modalità headless: calibrazione automatica al primo frame valido")

    # Mostra debug calibrazione caricata da file (al primo frame del loop)
    show_loaded_calib = not pending_calib and not args.no_display
    calib_shown = False

    min_frame_time = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    t_start        = time.time()

    print("[offline_tester] Elaborazione... (premi q nella finestra per uscire)")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        t0 = time.time()
        frame_idx += 1

        # BGR → RGB (stesso comportamento di lane_follower.py con encoding rgb8)
        frame_rgb = frame_bgr[:, :, ::-1]

        # Preprocessing
        img_chw = preprocess(frame_rgb, args.crop_top)

        # Inference ONNX
        mask = model.infer(img_chw).astype(np.int64)

        # Headless: calibrazione automatica al primo frame valido
        if pending_calib:
            n_lane = int(((mask == CLASS_LANE_MARKING) | (mask == CLASS_LANE_DASHED)).sum())
            if n_lane >= 50:
                auto_calib.calibrate(mask, CLASS_LANE_MARKING)
                pending_calib = False
                print(f"[offline_tester] Calibrazione BEV automatica (frame {frame_idx})")

        # Prima iterazione con calibrazione da file: mostra debug una volta sola
        if show_loaded_calib and not calib_shown and not pending_calib:
            calib_shown = True
            if not _show_calib_debug(auto_calib, mask, frame_bgr, frame_rgb, args,
                                     calib_prefix, cap, writer):
                return

        # BEV warp
        bev_mask   = auto_calib.make_bev(mask)
        bev_u8     = np.clip(bev_mask, 0, 255).astype(np.uint8, copy=False)

        # Pubblica /lane_mask_bev al robot (se --robot-ip specificato)
        if ros_pub is not None:
            try:
                ros_pub.publish(bev_u8)
            except Exception as e:
                print(f"[offline_tester] WARN rosbridge: {e}")

        # FPS misurati
        elapsed = time.time() - t_start
        fps     = (frame_idx - args.start_frame) / max(elapsed, 1e-6)

        crop_px  = int(frame_rgb.shape[0] * args.crop_top)
        orig_crop = frame_bgr[crop_px:, :]
        mask_u8   = np.clip(mask, 0, 255).astype(np.uint8)

        if args.seg_only:
            # Solo segmentazione + BEV, niente Hough/steering
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
            # Pipeline completa con HoughLinesP + steering
            # use_bev=False → passa la maschera raw; il pannello BEV mostra la prospettiva
            input_mask = bev_u8 if controller.use_bev else mask_u8
            steering, angular_z, state, debug_info = controller.step(input_mask)
            viz_bev = bev_u8 if controller.use_bev else mask_u8
            # bev_scale>1: scala viz_bev per allineare le coordinate debug_info al pannello
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
    print(f"[offline_tester] Completato: {n} frame in {elapsed_total:.1f}s  "
          f"({n/max(elapsed_total,1e-6):.1f} fps medio)")
    if args.output:
        print(f"[offline_tester] Video salvato: {args.output}")


if __name__ == "__main__":
    main()
