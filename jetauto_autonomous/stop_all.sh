#!/usr/bin/env bash
# =============================================================================
# stop_all.sh
# -----------------------------------------------------------------------------
# Ferma tutti i processi avviati da start_all.sh leggendoli da
# /tmp/jetauto_autonomous.pids
# =============================================================================

PID_FILE="/tmp/jetauto_autonomous.pids"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Nessun PID file trovato in $PID_FILE."
  echo "Forse non c'è nulla da fermare, oppure usa pkill manualmente:"
  echo "  pkill -f lane_controller_node.py"
  echo "  pkill -f waypoint_manager_node.py"
  echo "  pkill -f serve_dashboard.py"
  echo "  pkill -f rosbridge_websocket"
  echo "  pkill -f web_video_server"
  exit 0
fi

echo "Fermo i processi elencati in $PID_FILE..."
while read -r pid name; do
  if [[ -z "$pid" ]]; then continue; fi
  if kill -0 "$pid" 2>/dev/null; then
    echo "  kill $pid ($name)"
    kill "$pid" 2>/dev/null || true
  else
    echo "  $pid ($name) gia' morto"
  fi
done < "$PID_FILE"

# Aspetta un attimo e SIGKILL ai sopravvissuti
sleep 1
while read -r pid name; do
  if [[ -z "$pid" ]]; then continue; fi
  if kill -0 "$pid" 2>/dev/null; then
    echo "  SIGKILL $pid ($name) - non terminava"
    kill -9 "$pid" 2>/dev/null || true
  fi
done < "$PID_FILE"

# Pulizia: invia anche un Twist zero per fermare il robot in sicurezza
echo "Invio Twist zero per fermare il robot..."
rostopic pub -1 /jetauto_controller/cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}}' \
  > /dev/null 2>&1 || true

rm -f "$PID_FILE"
echo "Fatto."
