#!/usr/bin/env bash
# =============================================================================
# stop_all.sh
# -----------------------------------------------------------------------------
# Stop all processes started by start_all.sh, reading PIDs from
# /tmp/jetauto_autonomous.pids, then sweep any orphans by name (a stale or
# overwritten PID file would otherwise leave nodes running — the recurring
# "zombie n_orchestrator.py" problem).
# =============================================================================

PID_FILE="/tmp/jetauto_autonomous.pids"

# Same patterns as the start_all.sh pre-check. Python-prefixed so an editor
# with the file open is not killed; "orchestrator.py" matches
# n_orchestrator.py (and any legacy *orchestrator.py).
ZOMBIE_PATTERNS=(
  "python.*orchestrator\.py"
  "python.*lane_controller_node\.py"
  "python.*waypoint_manager_node\.py"
  "python.*serve_dashboard\.py"
  "rosbridge_websocket"
  "web_video_server"
)

if [[ -f "$PID_FILE" ]]; then
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
else
  echo "No PID file found at $PID_FILE — sweeping by process name only."
fi

# Orphan sweep: catch anything the PID file missed (stale file, double
# start_all.sh, nodes started by hand).
SWEPT=0
for pat in "${ZOMBIE_PATTERNS[@]}"; do
  if pgrep -f "$pat" > /dev/null 2>&1; then
    SWEPT=1
    echo "  orphan(s) matching '$pat':"
    pgrep -af "$pat" | sed 's/^/    /'
    pkill -f "$pat" 2>/dev/null || true
  fi
done
if [[ "$SWEPT" == "1" ]]; then
  sleep 1
  for pat in "${ZOMBIE_PATTERNS[@]}"; do
    pkill -9 -f "$pat" 2>/dev/null || true
  done
fi

# Cleanup: also publish a zero Twist to stop the robot safely
echo "Sending zero Twist to stop the robot..."
rostopic pub -1 /jetauto_controller/cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}' \
  > /dev/null 2>&1 || true

rm -f "$PID_FILE"
echo "Done."