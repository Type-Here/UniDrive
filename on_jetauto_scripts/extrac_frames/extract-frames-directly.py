#!/usr/bin/env python3
"""
extract_frames.py — Extract frames from a rosbag to JPEG files
==============================================================
Reads /astra_cam/rgb/image_raw from a rosbag and saves one frame
every FRAME_INTERVAL seconds.

Usage:
    python3 extract_frames-directly.py <file.bag> [output_dir] [interval_sec]

Example:
    python3 extract_frames-directly.py session_01.bag ./frames 0.5
    # saves one frame every 0.5 seconds
"""

import sys
import os
import cv2

# -- Parameters -----------------------------------------------------------------
IMAGE_TOPIC    = "/depth_cam/rgb/image_raw"
FRAME_INTERVAL = 0.5      # seconds between saved frames (default)
JPEG_QUALITY   = 92       # 0-100
# ------------------------------------------------------------------------------


def extract(bag_path, out_dir, interval):
    try:
        import rosbag
        from cv_bridge import CvBridge
    except ImportError:
        print("ERROR: rosbag and cv_bridge must be available.")
        print("  Run this script in the ROS1 environment on the robot,")
        print("  or install: pip install rosbag cv_bridge")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    bridge     = CvBridge()
    count      = 0
    last_t     = None

    print(f"\n  Bag    : {bag_path}")
    print(f"  Topic  : {IMAGE_TOPIC}")
    print(f"  Output : {out_dir}")
    print(f"  Interval: {interval} s\n")

    with rosbag.Bag(bag_path, "r") as bag:
        total = bag.get_message_count(IMAGE_TOPIC)
        print(f"  Messages on topic: {total}")

        for topic, msg, t in bag.read_messages(topics=[IMAGE_TOPIC]):
            t_sec = t.to_sec()

            # Skip if not enough time has passed
            if last_t is not None and (t_sec - last_t) < interval:
                continue

            last_t = t_sec

            # Convert ROS image to OpenCV
            try:
                if msg.encoding in ("rgb8", "RGB8"):
                    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
                elif msg.encoding in ("bgr8", "BGR8"):
                    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
                else:
                    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                print(f"  Warning: could not convert frame at t={t_sec:.2f}: {e}")
                continue

            filename = os.path.join(out_dir, f"frame_{count:05d}.jpg")
            cv2.imwrite(filename, frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            count += 1

            if count % 20 == 0:
                print(f"  Saved {count} frames  (t={t_sec:.1f}s)", end="\r")

    print(f"\n  Done — {count} frames saved to '{out_dir}'")
    print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_frames.py <file.bag> [output_dir] [interval_sec]")
        sys.exit(1)

    bag_path = sys.argv[1]
    out_dir  = sys.argv[2] if len(sys.argv) > 2 \
        else os.path.splitext(bag_path)[0] + "_frames"
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else FRAME_INTERVAL

    extract(bag_path, out_dir, interval)


if __name__ == "__main__":
    main()