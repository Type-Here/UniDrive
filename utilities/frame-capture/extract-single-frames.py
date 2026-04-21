#!/usr/bin/env python3
"""
extract_frames.py - Extract N evenly-spaced frames from a video
================================================================
Usage:
    python3 extract_frames.py <video> [n_frames] [output_dir]

Defaults:
    n_frames   = 90
    output_dir = <video_name>_frames/
"""

import sys, os
import cv2

def extract(video_path, n_frames=90, out_dir=None, use_jpg=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}"); sys.exit(1)

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur    = total / fps if fps > 0 else 0

    if out_dir is None:
        out_dir = os.path.splitext(video_path)[0] + "_frames"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Video    : {video_path}")
    print(f"  Duration : {dur:.1f}s  ({total} frames @ {fps:.1f} fps)")
    print(f"  Size     : {w}x{h}")
    print(f"  Extract  : {n_frames} frames -> {out_dir}\n")

    # Pick n_frames evenly spaced frame indices
    indices = [int(round(i * (total - 1) / (n_frames - 1)))
               for i in range(n_frames)]

    saved = 0
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: could not read frame {idx}")
            continue
        t_sec = idx / fps
        if use_jpg:
            fname = os.path.join(out_dir, f"frame_{i:03d}_t{t_sec:.2f}s.jpg")
            cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            fname = os.path.join(out_dir, f"frame_{i:03d}_t{t_sec:.2f}s.png")
            cv2.imwrite(fname, frame)
        saved += 1
        print(f"  [{i+1:3d}/{n_frames}] t={t_sec:6.2f}s  -> {os.path.basename(fname)}")

    cap.release()
    print(f"\n  Done - {saved} frames saved to '{out_dir}'")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 extract_frames.py <video> [n_frames] [output_dir] [jpg]")
        sys.exit(1)
    video   = sys.argv[1]
    n       = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    out     = sys.argv[3]      if len(sys.argv) > 3 else None
    use_jpg = sys.argv[4].lower() == "jpg" if len(sys.argv) > 4 else False
    extract(video, n, out, use_jpg)