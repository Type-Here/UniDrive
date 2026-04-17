#!/usr/bin/env python3
"""
capture_frames.py
=================
Script to capture RGB frame (optionally depth) from camera
Orbbec Astra Pro Plus mounted on auto — Jetson Nano/Orin.

Capture Modality:
  - AUTOMATIC : saves a frame every INTERVAL_MS ms
  - MANUAL    : click SPACE button to save current frame
  - BOTH   : both modalities coexist in same session

Output Structure:
  dataset/
  └-- session_YYYYMMDD_HHMMSS/
      ├-- rgb/
      │   ├-- frame_000001.jpg
      │   └-- ...
      └-- depth/          ← solo se SAVE_DEPTH = True
          ├-- frame_000001.png
          └-- ...

Dependencies:
  pip install opencv-python numpy

Orbbec Astra Pro Plus supports two backends:
  1. OpenNI2 + pyopenni2  (recommended for depth)
  2. OpenCV VideoCapture  (only RGB via UVC, easier)

Set USE_OPENNI = True if driver OpenNI2/Orbbec SDK is installed
"""

import cv2
import numpy as np
import os
import time
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------
#  CONFIGURATION
# ---------------------------------------------

# Backend: True = OpenNI2 (depth available), False = only OpenCV (RGB only)
USE_OPENNI = False

# OpenCV Device Index (0 or 1 usually, check with `v4l2-ctl --list-devices`)
OPENCV_DEVICE_INDEX = 0

# Target Resolution RGB
RGB_WIDTH  = 640
RGB_HEIGHT = 480

# Automatic Mode
AUTO_CAPTURE      = True
INTERVAL_MS       = 500           # interval (ms)

# Manual Mode (SPACE to SAVE)
MANUAL_CAPTURE    = True

# Save depth channel also (only if USE_OPENNI = True)
SAVE_DEPTH        = False

# Output Folder
OUTPUT_BASE_DIR   = Path("dataset")

# JPEG quality for RGB (0-100)
JPEG_QUALITY      = 95

# Show Preview Windows
SHOW_PREVIEW      = False

# ---------------------------------------------
#  SESSION SETUP
# ---------------------------------------------

def create_session_dirs(base: Path):
    session_name = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir  = base / session_name
    rgb_dir      = session_dir / "rgb"
    depth_dir    = session_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_DEPTH:
        depth_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Sessione avviata: {session_dir}")
    return session_dir, rgb_dir, depth_dir


# ---------------------------------------------
#  BACKEND OPENNI2
# ---------------------------------------------

def init_openni():
    """Init OpenNI2 and open RGB flow and (optionally) depth."""
    try:
        from openni import openni2
        openni2.initialize()
        dev = openni2.Device.open_any()

        rgb_stream = dev.create_color_stream()
        rgb_stream.set_video_mode(openni2.c_api.OniVideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_RGB888,
            resolutionX=RGB_WIDTH,
            resolutionY=RGB_HEIGHT,
            fps=30
        ))
        rgb_stream.start()

        depth_stream = None
        if SAVE_DEPTH:
            depth_stream = dev.create_depth_stream()
            depth_stream.start()

        print("[INFO] OpenNI2 inizializzato correttamente.")
        return openni2, dev, rgb_stream, depth_stream

    except Exception as e:
        print(f"[ERRORE] OpenNI2 non disponibile: {e}")
        sys.exit(1)


def read_openni_frame(rgb_stream, depth_stream):
    """Read RGB frame from OpenNI2."""
    frame = rgb_stream.read_frame()
    buf   = frame.get_buffer_as_uint8()
    rgb   = np.frombuffer(buf, dtype=np.uint8).reshape(RGB_HEIGHT, RGB_WIDTH, 3)
    rgb   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    depth_colored = None
    if depth_stream and SAVE_DEPTH:
        d_frame = depth_stream.read_frame()
        d_buf   = d_frame.get_buffer_as_uint16()
        depth   = np.frombuffer(d_buf, dtype=np.uint16).reshape(
                      d_frame.height, d_frame.width)
        # Normalizza a 8-bit per visualizzazione/salvataggio
        depth_colored = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.05),
            cv2.COLORMAP_JET
        )

    return rgb, depth_colored


# ---------------------------------------------
#  BACKEND OPENCV
# ---------------------------------------------

def init_opencv():
    cap = cv2.VideoCapture(OPENCV_DEVICE_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback senza specificare backend
        cap = cv2.VideoCapture(OPENCV_DEVICE_INDEX)
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire il device {OPENCV_DEVICE_INDEX}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RGB_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RGB_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"[INFO] OpenCV VideoCapture aperto su /dev/video{OPENCV_DEVICE_INDEX}.")
    return cap


def read_opencv_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame non ricevuto, skip.")
        return None, None
    return frame, None


# ---------------------------------------------
#  SAVE FRAME
# ---------------------------------------------

def save_frame(rgb, depth_colored, rgb_dir, depth_dir, counter):
    filename = f"frame_{counter:06d}"

    # RGB -> JPEG
    rgb_path = rgb_dir / f"{filename}.jpg"
    cv2.imwrite(str(rgb_path),
                rgb,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    # Depth -> PNG (lossless)
    if depth_colored is not None and SAVE_DEPTH:
        depth_path = depth_dir / f"{filename}.png"
        cv2.imwrite(str(depth_path), depth_colored)

    return rgb_path


# ---------------------------------------------
#  OVERLAY HUD on preview
# ---------------------------------------------

def draw_hud(frame, counter, auto_active, last_saved_path):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent background on top
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    mode_str = ""
    if AUTO_CAPTURE:  mode_str += "AUTO "
    if MANUAL_CAPTURE: mode_str += "MANUALE"

    cv2.putText(frame,
                f"Frame salvati: {counter}  |  Modalita: {mode_str}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)
    cv2.putText(frame,
                "SPAZIO=salva  A=avvia/pausa auto  Q=esci",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if last_saved_path:
        cv2.putText(frame,
                    f"Salvato: {Path(last_saved_path).name}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 220, 255), 1)
    return frame


# ---------------------------------------------
#  MAIN LOOP
# ---------------------------------------------

def main():
    print("=" * 55)
    print("  Frame Capture — Orbbec Astra Pro Plus / Jetson")
    print("=" * 55)

    session_dir, rgb_dir, depth_dir = create_session_dirs(OUTPUT_BASE_DIR)

    # Init backend
    if USE_OPENNI:
        openni2_mod, dev, rgb_stream, depth_stream = init_openni()
        read_frame = lambda: read_openni_frame(rgb_stream, depth_stream)
        cleanup    = lambda: (rgb_stream.stop(), openni2_mod.unload())
    else:
        cap        = init_opencv()
        read_frame = lambda: read_opencv_frame(cap)
        cleanup    = lambda: cap.release()

    counter        = 0
    auto_active    = AUTO_CAPTURE
    last_auto_time = time.time()
    last_saved     = None

    print("\nControlli:")
    print("  SPAZIO -> salva frame manuale")
    print("  A      -> avvia / pausa cattura automatica")
    print("  Q      -> esci e chiudi\n")

    try:
        while True:
            rgb, depth_colored = read_frame()
            if rgb is None:
                time.sleep(0.05)
                continue

            now = time.time()

            # AUTO CAPTURE
            if AUTO_CAPTURE and auto_active:
                if (now - last_auto_time) * 1000 >= INTERVAL_MS:
                    counter    += 1
                    last_saved  = save_frame(rgb, depth_colored,
                                             rgb_dir, depth_dir, counter)
                    print(f"[AUTO] Salvato frame {counter:06d}")
                    last_auto_time = now

            # Preview
            if SHOW_PREVIEW:
                preview = rgb.copy()
                # Indicatore rosso lampeggiante durante auto
                if AUTO_CAPTURE and auto_active:
                    if int(now * 2) % 2 == 0:
                        cv2.circle(preview, (preview.shape[1] - 20, 20),
                                   8, (0, 0, 220), -1)
                preview = draw_hud(preview, counter, auto_active, last_saved)
                cv2.imshow("Frame Capture — Orbbec Astra", preview)

                key = cv2.waitKey(1) & 0xFF

                #  MANUAL CAPTURE
                if MANUAL_CAPTURE and key == ord(' '):
                    counter   += 1
                    last_saved = save_frame(rgb, depth_colored,
                                            rgb_dir, depth_dir, counter)
                    print(f"[MANUALE] Salvato frame {counter:06d}")

                # -- Toggle AUTO -----------------------------
                elif key == ord('a'):
                    auto_active = not auto_active
                    stato = "AVVIATA" if auto_active else "IN PAUSA"
                    print(f"[INFO] Cattura automatica {stato}")

                # -- EXIT ------------------------------------
                elif key == ord('q'):
                    break
            else:
                # No preview: only auto + Ctrl-C to exit
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Interrotto con Ctrl-C.")

    finally:
        cleanup()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        print(f"\n[FINE] Totale frame salvati: {counter}")
        print(f"[FINE] Dataset in: {session_dir.resolve()}")


if __name__ == "__main__":
    main()