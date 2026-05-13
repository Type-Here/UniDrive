#!/usr/bin/env bash
# =============================================================================
# start_test.sh
# -----------------------------------------------------------------------------
# Variante di start_all.sh che lancia ANCHE il fake_mask_publisher.py al
# posto del modello vero. Utile per testare la pipeline di controllo +
# dashboard senza accendere SegNet/SegFormer (risparmio RAM enorme sul
# Jetson Nano 4GB).
#
# Si aspetta una cartella di PNG (maschere mono8 con valori 0..4) generabili
# con scripts/labelme_to_mask.py a partire da JSON LabelMe.
#
# USO:
#   ./start_test.sh [--masks /path/to/dir] [--random] [--hold 0.5]
#                   [--rate 10] [--bev|--camera]
#
# Default:
#   --masks ./test_masks
#   --rate  10
#   modalita' sequenziale (toglie --random)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PARAMS_FILE="$SCRIPT_DIR/config/lane_params.yaml"
MAP_FILE="$SCRIPT_DIR/maps/map_clean-edited_smooth.yaml"
WEB_DIR="$SCRIPT_DIR/web"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
PID_FILE="/tmp/jetauto_autonomous.pids"
LOG_DIR="/tmp/jetauto_autonomous_logs"

PY="python2"

# ---- Default ----
MASKS_DIR="$SCRIPT_DIR/test_masks"
RATE="10"
RANDOM_FLAG=""
HOLD="0"
INPUT_MODE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --masks)  MASKS_DIR="$2"; shift 2 ;;
    --rate)   RATE="$2"; shift 2 ;;
    --random) RANDOM_FLAG="--random"; shift ;;
    --hold)   HOLD="$2"; shift 2 ;;
    --camera) INPUT_MODE_OVERRIDE="camera"; shift ;;
    --bev)    INPUT_MODE_OVERRIDE="bev_topic"; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//' | head -25
      exit 0 ;;
    *) echo "Argomento sconosciuto: $1"; exit 1 ;;
  esac
done

echo "================================================="
echo "  JetAuto Autonomous - start_test.sh (FAKE MASK)"
echo "================================================="
echo "  MASKS_DIR  : $MASKS_DIR"
echo "  RATE       : $RATE Hz"
echo "  RANDOM     : ${RANDOM_FLAG:-no}"
echo "  HOLD       : $HOLD s"
echo "================================================="

if [[ ! -d "$MASKS_DIR" ]]; then
  echo "ERRORE: cartella maschere non trovata: $MASKS_DIR"
  echo "Generala con:"
  echo "  $PY $SCRIPTS_DIR/labelme_to_mask.py /path/al/dataset -o $MASKS_DIR --resize 512x256"
  exit 1
fi

N_MASKS=$(ls "$MASKS_DIR"/*.png 2>/dev/null | grep -v '_vis.png' | wc -l)
echo "  Maschere trovate: $N_MASKS"
if [[ $N_MASKS -eq 0 ]]; then
  echo "ERRORE: nessuna PNG in $MASKS_DIR"
  exit 1
fi

# ---- Pre-check ----
if ! pgrep -f rosmaster > /dev/null; then
  echo "ERRORE: roscore non attivo"
  exit 1
fi
if ! $PY -c "import rospy, cv_bridge, yaml, networkx" 2>/dev/null; then
  echo "ERRORE: dipendenze Python mancanti"
  exit 1
fi

mkdir -p "$LOG_DIR"
> "$PID_FILE"

# ---- Carica parametri ----
echo "[1/6] rosparam load"
rosparam load "$PARAMS_FILE"
if [[ -n "$INPUT_MODE_OVERRIDE" ]]; then
  rosparam set "lane_controller/input_mode" "$INPUT_MODE_OVERRIDE"
fi

start_proc () {
  local name="$1"; shift
  local logfile="$LOG_DIR/${name}.log"
  echo "       avvio $name (log: $logfile)"
  "$@" > "$logfile" 2>&1 &
  local pid=$!
  echo "$pid $name" >> "$PID_FILE"
  sleep 0.5
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERRORE: $name morto subito. Vedi $logfile"
    tail -20 "$logfile"
    exit 1
  fi
}

# ---- Servizi web ----
echo "[2/6] rosbridge_websocket"
start_proc rosbridge \
  rosrun rosbridge_server rosbridge_websocket _port:=9090

echo "[3/6] web_video_server"
start_proc web_video_server \
  rosrun web_video_server web_video_server _port:=8080

echo "[4/6] serve_dashboard"
start_proc dashboard_http \
  $PY "$SCRIPTS_DIR/serve_dashboard.py" \
       --port 8000 --web-dir "$WEB_DIR" --map "$MAP_FILE"

# ---- Nodi nostri ----
echo "[5/6] lane_controller + waypoint_manager"
start_proc lane_controller \
  $PY "$SCRIPTS_DIR/lane_controller_node.py"

rosparam set "waypoint_manager/map_file" "$MAP_FILE"
start_proc waypoint_manager \
  $PY "$SCRIPTS_DIR/waypoint_manager_node.py"

# ---- Fake mask publisher (al posto di SegNet) ----
echo "[6/6] fake_mask_publisher"
FAKE_ARGS=(--dir "$MASKS_DIR" --rate "$RATE" --publish-rgb)
if [[ -n "$RANDOM_FLAG" ]]; then
  FAKE_ARGS+=(--random)
  if [[ "$HOLD" != "0" ]]; then
    FAKE_ARGS+=(--hold "$HOLD")
  fi
fi
start_proc fake_mask_publisher \
  $PY "$SCRIPTS_DIR/fake_mask_publisher.py" "${FAKE_ARGS[@]}"

echo
echo "================================================="
echo "  Test mode pronto!"
echo "================================================="
echo "  Dashboard:  http://$(hostname -I | awk '{print $1}'):8000"
echo "  Logs:       $LOG_DIR/"
echo
echo "  Per controlli runtime del fake publisher:"
echo "    rostopic pub /fake_mask_publisher/cmd std_msgs/String 'data: pause'"
echo "    rostopic pub /fake_mask_publisher/cmd std_msgs/String 'data: resume'"
echo "    rostopic pub /fake_mask_publisher/cmd std_msgs/String 'data: next'"
echo
echo "  Per fermare: ./stop_all.sh"
echo "================================================="
echo
echo "ATTENZIONE: in modalita' test il /jetauto_controller/cmd_vel viene"
echo "            comunque pubblicato. Tieni il robot SOLLEVATO o disattiva"
echo "            i motori se non vuoi che si muova."
