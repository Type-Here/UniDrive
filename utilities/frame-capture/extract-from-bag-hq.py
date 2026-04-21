#!/usr/bin/env python3
"""
High-fidelity ROS1 bag extractor (video + odometry).

Key differences vs basic extractor:
- Lossless video mode by default: FFV1 in MKV (very large files, high fidelity).
- FPS auto-estimation from bag timestamps (better temporal fidelity than fixed 30 FPS).
- Proper handling of msg.step / row stride.
- Optional H.264 export mode for smaller files.

Requirements:
    pip install rosbags numpy
    ffmpeg installed in PATH
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

IMAGE_TOPIC = "/depth_cam/rgb/image_raw"
ODOM_TOPIC = "/odom"


def _find_connection(reader, topic):
    return next((c for c in reader.connections if c.topic == topic), None)


def _reshape_with_step(data_u8, height, width, channels, step):
    """Reshape honoring row stride (step), then trim padded bytes."""
    expected_row = width * channels
    if step < expected_row:
        raise ValueError(f"Invalid step={step}, expected at least {expected_row}")
    rows = data_u8.reshape(height, step)
    rows = rows[:, :expected_row]
    return rows.reshape(height, width, channels)


def ros_image_to_rgb8(msg):
    """Convert supported ROS image encodings to RGB8 numpy array."""
    enc = msg.encoding.lower()
    h, w = int(msg.height), int(msg.width)
    step = int(msg.step)
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        return _reshape_with_step(data, h, w, 3, step)
    if enc == "bgr8":
        bgr = _reshape_with_step(data, h, w, 3, step)
        return bgr[:, :, ::-1]
    if enc == "mono8":
        # mono8 rows can include padding in step
        if step < w:
            raise ValueError(f"Invalid mono8 step={step}, expected >= {w}")
        rows = data.reshape(h, step)[:, :w]
        return np.repeat(rows[:, :, None], 3, axis=2)
    if enc == "rgba8":
        rgba = _reshape_with_step(data, h, w, 4, step)
        return rgba[:, :, :3]
    if enc == "bgra8":
        bgra = _reshape_with_step(data, h, w, 4, step)
        return bgra[:, :, :3][:, :, ::-1]

    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def estimate_fps(reader, typestore, conn):
    """Estimate source FPS from bag message timestamps."""
    first_ts = None
    last_ts = None
    count = 0

    for _conn, ts, _raw in reader.messages(connections=[conn]):
        if first_ts is None:
            first_ts = ts
        last_ts = ts
        count += 1

    if count < 2 or first_ts is None or last_ts is None or last_ts <= first_ts:
        return 30.0

    duration = (last_ts - first_ts) * 1e-9
    fps = (count - 1) / duration
    # Keep sane bounds
    fps = max(1.0, min(240.0, fps))
    return float(f"{fps:.6f}")


def start_ffmpeg_writer(out_path, width, height, fps, codec):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    if codec == "ffv1":
        # Lossless intra codec, very high fidelity, large files.
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", f"{fps}",
            "-i", "-",
            "-an",
            "-c:v", "ffv1",
            "-level", "3",
            "-g", "1",
            "-slices", "24",
            "-slicecrc", "1",
            str(out_path),
        ]
    elif codec == "h264":
        # Near-lossless option if size matters.
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", f"{fps}",
            "-i", "-",
            "-an",
            "-c:v", "libx264rgb",
            "-crf", "0",
            "-preset", "veryslow",
            "-pix_fmt", "rgb24",
            str(out_path),
        ]
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    return proc


def extract_video(reader, typestore, out_path, codec="ffv1", fps_mode="auto", fixed_fps=30.0):
    conn = _find_connection(reader, IMAGE_TOPIC)
    if conn is None:
        print(f"  WARNING: {IMAGE_TOPIC} not in bag - skipping video")
        return

    fps = fixed_fps if fps_mode == "fixed" else estimate_fps(reader, typestore, conn)
    print(f"  Extracting video -> {out_path}")
    print(f"  Codec: {codec}  FPS: {fps}")

    proc = None
    count = 0

    for _conn, ts, rawdata in reader.messages(connections=[conn]):
        msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
        frame_rgb = ros_image_to_rgb8(msg)

        if proc is None:
            h, w = frame_rgb.shape[:2]
            proc = start_ffmpeg_writer(out_path, w, h, fps, codec)
            print(f"  Resolution: {w}x{h}  encoding: {msg.encoding}")

        try:
            proc.stdin.write(frame_rgb.tobytes())
        except BrokenPipeError:
            raise RuntimeError("ffmpeg writer pipe broke; check codec/container support")

        count += 1
        if count % 100 == 0:
            print(f"  {count} frames...", end="\r")

    if proc is not None:
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with code {rc}")

    print(f"  Done - {count} frames -> {out_path}          ")


def extract_odom(reader, typestore, out_path):
    conn = _find_connection(reader, ODOM_TOPIC)
    if conn is None:
        print(f"  WARNING: {ODOM_TOPIC} not in bag - skipping odom")
        return

    print(f"  Extracting odometry -> {out_path}")
    count = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp",
            "x", "y", "z",
            "qx", "qy", "qz", "qw",
            "vx", "vy", "vz",
            "wx", "wy", "wz",
        ])
        for _conn, ts, rawdata in reader.messages(connections=[conn]):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear
            ang = msg.twist.twist.angular
            w.writerow([
                round(ts * 1e-9, 6),
                round(p.x, 6), round(p.y, 6), round(p.z, 6),
                round(q.x, 6), round(q.y, 6), round(q.z, 6), round(q.w, 6),
                round(v.x, 6), round(v.y, 6), round(v.z, 6),
                round(ang.x, 6), round(ang.y, 6), round(ang.z, 6),
            ])
            count += 1

    print(f"  Done - {count} poses -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", help="Input .bag file")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--odom", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--codec", choices=["ffv1", "h264"], default="ffv1",
                        help="Video codec (default: ffv1 lossless)")
    parser.add_argument("--fps-mode", choices=["auto", "fixed"], default="auto",
                        help="auto estimates FPS from bag timestamps")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="used when --fps-mode fixed")
    args = parser.parse_args()

    if not args.video and not args.odom and not args.both:
        args.both = True

    do_video = args.video or args.both
    do_odom = args.odom or args.both

    bag_path = Path(args.bag)
    base = bag_path.with_suffix("")

    if args.codec == "ffv1":
        video_out = base.with_suffix(".mkv")
    else:
        video_out = base.with_suffix(".mp4")

    print(f"\n  Input   : {bag_path}")
    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(bag_path) as reader:
        print("  Topics  :")
        for c in reader.connections:
            print(f"    {c.topic}  [{c.msgtype}]")
        print()

        if do_video:
            extract_video(
                reader,
                typestore,
                video_out,
                codec=args.codec,
                fps_mode=args.fps_mode,
                fixed_fps=args.fps,
            )
        if do_odom:
            extract_odom(reader, typestore, Path(str(base) + "_odom.csv"))

    print("\n  All done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
