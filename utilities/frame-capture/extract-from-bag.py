#!/usr/bin/env python3
"""
bag_extract.py - Extract video and/or odometry from a ROS1 bag
===============================================================
Works on any PC with:
    pip install rosbags opencv-python

Usage:
    python3 bag_extract.py <file.bag> [--video] [--odom] [--both]

    --video   Extract camera to MP4  (default: both)
    --odom    Extract odometry to CSV
    --both    Extract both (default if no flag given)

Examples:
    python3 bag_extract.py session.bag
    python3 bag_extract.py session.bag --video
    python3 bag_extract.py session.bag --odom
"""

import sys, os, argparse
import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

IMAGE_TOPIC = "/depth_cam/rgb/image_raw"
ODOM_TOPIC  = "/odom"
FPS         = 30


def extract_video(reader, typestore, out_path):
    try:
        import cv2
    except ImportError:
        print("ERROR: pip install opencv-python")
        sys.exit(1)

    writer  = None
    count   = 0
    conn    = next((c for c in reader.connections
                    if c.topic == IMAGE_TOPIC), None)
    if conn is None:
        print(f"  WARNING: {IMAGE_TOPIC} not in bag - skipping video")
        return

    print(f"  Extracting video -> {out_path}")
    for conn, ts, rawdata in reader.messages(connections=[conn]):
        msg = typestore.deserialize_ros1(rawdata, conn.msgtype)

        # Build numpy array from raw image data
        data = np.frombuffer(msg.data, dtype=np.uint8)

        enc = msg.encoding.lower()
        if enc in ("rgb8",):
            frame = data.reshape(msg.height, msg.width, 3)
            frame = frame[:, :, ::-1]          # RGB -> BGR for OpenCV
        elif enc in ("bgr8",):
            frame = data.reshape(msg.height, msg.width, 3)
        elif enc in ("mono8",):
            frame = data.reshape(msg.height, msg.width)
            frame = np.stack([frame]*3, axis=-1)
        else:
            # Try generic reshape
            frame = data.reshape(msg.height, msg.width, -1)
            if frame.shape[2] == 3:
                frame = frame[:, :, ::-1]

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (w, h))
            print(f"  Resolution: {w}x{h}  encoding: {msg.encoding}")

        writer.write(frame)
        count += 1
        if count % 100 == 0:
            print(f"  {count} frames...", end="\r")

    if writer:
        writer.release()
    print(f"  Done - {count} frames -> {out_path}          ")


def extract_odom(reader, typestore, out_path):
    import csv

    conn = next((c for c in reader.connections
                 if c.topic == ODOM_TOPIC), None)
    if conn is None:
        print(f"  WARNING: {ODOM_TOPIC} not in bag - skipping odom")
        return

    print(f"  Extracting odometry -> {out_path}")
    count = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp",
                    "x","y","z",
                    "qx","qy","qz","qw",
                    "vx","vy","vz",
                    "wx","wy","wz"])
        for conn, ts, rawdata in reader.messages(connections=[conn]):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear
            ω = msg.twist.twist.angular
            w.writerow([
                round(ts * 1e-9, 6),
                round(p.x,6), round(p.y,6), round(p.z,6),
                round(q.x,6), round(q.y,6), round(q.z,6), round(q.w,6),
                round(v.x,6), round(v.y,6), round(v.z,6),
                round(ω.x,6), round(ω.y,6), round(ω.z,6),
            ])
            count += 1

    print(f"  Done - {count} poses -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag",     help="Input .bag file")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--odom",  action="store_true")
    parser.add_argument("--both",  action="store_true")
    args = parser.parse_args()

    if not args.video and not args.odom and not args.both:
        args.both = True

    do_video = args.video or args.both
    do_odom  = args.odom  or args.both

    bag_path = Path(args.bag)
    base     = bag_path.with_suffix("")

    print(f"\n  Input   : {bag_path}")
    typestore = get_typestore(Stores.ROS1_NOETIC)

    with Reader(bag_path) as reader:
        print(f"  Topics  :")
        for c in reader.connections:
            print(f"    {c.topic}  [{c.msgtype}]")
        print()

        if do_video:
            extract_video(reader, typestore, base.with_suffix(".mp4"))
        if do_odom:
            extract_odom(reader, typestore, Path(str(base) + "_odom.csv"))

    print("\n  All done!")


if __name__ == "__main__":
    main()