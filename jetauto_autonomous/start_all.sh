#!/usr/bin/env bash
# =============================================================================
# start_all.sh
# -----------------------------------------------------------------------------
# Avvia il sistema autonomo di guida JetAuto. Sostituisce dashboard.launch
# (non usiamo catkin_ws su questo Jetson).
#
# COSA FA:
#   1. Verifica che roscore sia attivo
#   2. Carica i parametri da config/lane_params.yaml in rosparam
#   3. Avvia in background:
#        - rosbridge_websocket (porta 9090)
#        - web_video_server    (porta 8080)
#        - serve_dashboard.py  (porta 8000)
#        - lane_controller_node.py
#        - waypoint_manager_node.py
#   4. Salva i PID in /tmp/jetauto_autonomous.pids
#
# COSA NON FA:
#   - NON avvia il lane_follower.py del SegFormer (lo lancia il tuo amico
#     dal suo ambiente conda Python 3.13).
#   - NON avvia roscore (parte da solo all'accensione del Jetson).
#
# UTILIZZO:
#   ./start_all.sh                # avvia tutto
#   ./start_all.sh --camera       # input_mode camera (default)
#   ./start_all.sh --bev          # forza input_mode bev_topic
#
# Per fermare tutto: ./stop_all.sh
# =============================================================================

set -e

# Path del progetto (la directory dove vive questo script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PARAMS_FILE="$SCRIPT_DIR/config/lane_params.yaml"
MAP_FILE="$SCRIPT_DIR/maps/map_clean-edited_smooth.yaml"
WEB_DIR="$SCRIPT_DIR/web"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
PID_FILE="/tmp/jetauto_autonomous.pids"
LOG_DIR="/tmp/jetauto_autonomous_logs"

# Interprete Python: usiamo python2 perché ROS Melodic vive lì
# (Python 3.13 conda è il setup del SegFormer ed è separato)
PY="python2"

# ---- Parsing argomenti ----
INPUT_MODE_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --camera) INPUT_MODE_OVERRIDE="camera"; shift ;;
    --bev)    INPUT_MODE_OVERRIDE="bev_topic"; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//' | head -40
      exit 0 ;;
    *) echo "Argomento sconosciuto: $1"; exit 1 ;;
  esac
done

# ---- Banner ----
echo "================================================="
echo "  JetAuto Autonomous - start_all.sh"
echo "================================================="
echo "  SCRIPT_DIR : $SCRIPT_DIR"
echo "  PARAMS     : $PARAMS_FILE"
echo "  MAP        : $MAP_FILE"
echo "  Python     : $PY ($($PY --version 2>&1))"
echo "================================================="

# ---- Pre-check: file e direttori ----
for f in "$PARAMS_FILE" "$MAP_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "ERRORE: file mancante: $f"; exit 1
  fi
done
for d in "$WEB_DIR" "$SCRIPTS_DIR"; do
  if [[ ! -d "$d" ]]; then
    echo "ERRORE: directory mancante: $d"; exit 1
  fi
done

# ---- Pre-check: roscore attivo ----
if ! pgrep -f rosmaster > /dev/null; then
  echo "ERRORE: roscore/rosmaster NON è attivo."
  echo "        Avvialo prima (di solito è auto-startato da systemd)."
  echo "        Per controllare: pgrep -af rosmaster"
  exit 1
fi
echo "[ok] roscore attivo"

# ---- Pre-check: Python ha cv_bridge, rospy, yaml, networkx ----
if ! $PY -c "import rospy, cv_bridge, yaml, networkx" 2>/dev/null; then
  echo "ERRORE: dipendenze Python mancanti su $PY."
  echo "        Verifica con: $PY -c \"import rospy, cv_bridge, yaml, networkx\""
  echo "        Per installarle:"
  echo "          sudo apt install python-yaml python-networkx"
  exit 1
fi
echo "[ok] dipendenze Python OK"

# ---- Cartella log ----
mkdir -p "$LOG_DIR"
> "$PID_FILE"

# ---- 1. Carica parametri YAML in rosparam ----
echo "[1/5] rosparam load $PARAMS_FILE"
rosparam load "$PARAMS_FILE"
# Override input_mode se richiesto da CLI
if [[ -n "$INPUT_MODE_OVERRIDE" ]]; then
  rosparam set "lane_controller/input_mode" "$INPUT_MODE_OVERRIDE"
  echo "       input_mode override -> $INPUT_MODE_OVERRIDE"
fi

# ---- Funzione helper per lanciare un processo in bg + log + PID ----
start_proc () {
  local name="$1"; shift
  local logfile="$LOG_DIR/${name}.log"
  echo "       avvio $name (log: $logfile)"
  "$@" > "$logfile" 2>&1 &
  local pid=$!
  echo "$pid $name" >> "$PID_FILE"
  sleep 0.5
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERRORE: $name è morto subito dopo l'avvio. Vedi $logfile"
    tail -20 "$logfile"
    exit 1
  fi
}

# ---- 2. rosbridge_websocket (porta 9090) ----
echo "[2/5] rosbridge_websocket"
start_proc rosbridge \
  rosrun rosbridge_server rosbridge_websocket _port:=9090

# ---- 3. web_video_server (porta 8080) ----
echo "[3/5] web_video_server"
start_proc web_video_server \
  rosrun web_video_server web_video_server _port:=8080

# ---- 4. dashboard HTTP (porta 8000) ----
echo "[4/5] serve_dashboard"
start_proc dashboard_http \
  $PY "$SCRIPTS_DIR/serve_dashboard.py" \
       --port 8000 \
       --web-dir "$WEB_DIR" \
       --map "$MAP_FILE"

# ---- 5. nodi nostri ----
echo "[5/5] lane_controller + waypoint_manager"
start_proc lane_controller \
  $PY "$SCRIPTS_DIR/lane_controller_node.py"

# Override del path mappa nel waypoint_manager (per evitare $(find ...))
rosparam set "waypoint_manager/map_file" "$MAP_FILE"

start_proc waypoint_manager \
  $PY "$SCRIPTS_DIR/waypoint_manager_node.py"

# ---- Riepilogo ----
echo
echo "================================================="
echo "  Tutto avviato!"
echo "================================================="
echo "  Dashboard web:  http://$(hostname -I | awk '{print $1}'):8000"
echo "  Video stream:   http://$(hostname -I | awk '{print $1}'):8080"
echo "  Rosbridge WS:   ws://$(hostname -I | awk '{print $1}'):9090"
echo
echo "  Logs:           $LOG_DIR/"
echo "  PID file:       $PID_FILE"
echo
echo "  Per fermare tutto: ./stop_all.sh"
echo "  Per guardare un log: tail -f $LOG_DIR/lane_controller.log"
echo "================================================="
echo
echo "RICORDA: il lane_follower.py del modello va lanciato"
echo "         separatamente nel suo ambiente conda Python x.x."
echo "         Senza di lui, /lane_mask non viene pubblicato e il"
echo "         lane_controller resta in stato STOP."
