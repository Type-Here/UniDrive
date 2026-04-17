#!/bin/bash
# record_camera.sh — Record RGB camera stream to rosbag
# Usage: ./record_camera.sh [output_name]
# Example: ./record_camera.sh session_01

NAME=${1:-recording_$(date +%Y%m%d_%H%M%S)}
OUTFILE="$HOME/${NAME}.bag"

echo "============================================"
echo "  Camera recorder — JetAuto RGB"
echo "  Topic  : /astra_cam/rgb/image_raw"
echo "  Output : $OUTFILE"
echo "  Press Ctrl+C to stop"
echo "============================================"

rosbag record \
    /astra_cam/rgb/image_raw \
    /odom \
    -O "$OUTFILE"

echo ""
echo "Saved: $OUTFILE"
echo "$(rosbag info $OUTFILE | grep -E 'size|duration|messages')"