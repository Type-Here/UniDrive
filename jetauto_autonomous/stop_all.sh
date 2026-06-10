#!/usr/bin/env bash
# =============================================================================
# stop_all.sh
# -----------------------------------------------------------------------------
# Stop all processes started by start_all.sh, reading PIDs from
# /tmp/jetauto_autonomous.pids
# =============================================================================

PID_FILE="/tmp/jetauto_autonomous.pids"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found at $PID_FILE."
  echo "Nothing to stop, or use pkill manually:"
  echo "  pkill -f new_orchestrator.py"
  echo "  pkill -f lane_controller_node.py"
  echo "  pkill -f waypoint_manager_node.py"
  echo "  pkill -f serve_dashboard.py"
  echo "  pkill -f rosbridge_websocket"
  echo "  pkill -f web_video_server"
  exit 0
fi

echo "Stopping processes listed in $PID_FILE..."
while read -r pid name; do
  if [[ -z "$pid" ]]; then continue; fi
  if kill -0 "$pid" 2>/dev/null; then
    echo "  kill $pid ($name)"
    kill "$pid" 2>/dev/null || true
  else
    echo "  $pid ($name) already dead"
  fi
done < "$PID_FILE"

# Wait briefly and SIGKILL any survivors
sleep 1
while read -r pid name; do
  if [[ -z "$pid" ]]; then continue; fi
  if kill -0 "$pid" 2>/dev/null; then
    echo "  SIGKILL $pid ($name) - did not terminate"
    kill -9 "$pid" 2>/dev/null || true
  fi
done < "$PID_FILE"

# Cleanup: also publish a zero Twist to stop the robot safely
echo "Sending zero Twist to stop the robot..."
rostopic pub -1 /jetauto_controller/cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}' \
  > /dev/null 2>&1 || true

rm -f "$PID_FILE"
echo "Done."
