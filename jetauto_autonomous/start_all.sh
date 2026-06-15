#!/usr/bin/env bash
# =============================================================================
# start_all.sh
# -----------------------------------------------------------------------------
# Start the JetAuto autonomous driving system. Replaces dashboard.launch
# (catkin_ws is not used on this Jetson).
#
# WHAT IT DOES:
#   1. Check that roscore is running
#   2. Load parameters from config/lane_params.yaml into rosparam
#   3. Start in background:
#        - rosbridge_websocket (port 9090)
#        - web_video_server    (port 8080)
#        - serve_dashboard.py  (port 8000)
#        - lane_controller_node.py
#        - waypoint_manager_node.py
#        - n_orchestrator.py    (sole cmd_vel publisher)
#   4. Save PIDs to /tmp/jetauto_autonomous.pids
#
# WHAT IT DOES NOT DO:
#   - Does NOT start lane_follower.py for the segmentation models 
#     (launched from their conda Python 3.6.9 environment).
#   - Does NOT start roscore (it auto-starts at Jetson boot).
#
# USAGE:
#   ./start_all.sh                # start everything
#
# To stop everything: ./stop_all.sh
# =============================================================================

set -e

# Project path (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PARAMS_FILE="$SCRIPT_DIR/config/lane_params.yaml"
MAP_FILE="$SCRIPT_DIR/maps/map_clean-edited_smooth.yaml"
WEB_DIR="$SCRIPT_DIR/web"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
PID_FILE="/tmp/jetauto_autonomous.pids"
LOG_DIR="/tmp/jetauto_autonomous_logs"
ORCHESTRATOR_NAME="n_orchestrator.py"

# Python interpreter: python2 because ROS Melodic lives there
# (Python 3.13 conda is the SegFormer setup and is separate)
PY="python2"

# ---- Argument parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//' | head -40
      exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
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

# ---- Pre-check: files and directories ----
for f in "$PARAMS_FILE" "$MAP_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing file: $f"; exit 1
  fi
done
for d in "$WEB_DIR" "$SCRIPTS_DIR"; do
  if [[ ! -d "$d" ]]; then
    echo "ERROR: missing directory: $d"; exit 1
  fi
done

# ---- Pre-check: roscore running ----
if ! pgrep -f rosmaster > /dev/null; then
  echo "ERROR: roscore/rosmaster is NOT running."
  echo "       Start it first (usually manually from script on dashboard)."
  echo "       To check: pgrep -af rosmaster"
  exit 1
fi
echo "[ok] roscore running"

# ---- Pre-check: leftover node processes from a previous run ----
# stop_all.sh only kills PIDs recorded in $PID_FILE; if start_all.sh is run
# twice (PID file overwritten) or a node was started by hand, orphans survive
# and two orchestrators publishing cmd_vel fight each other. Sweep them here.
# Patterns are python-prefixed so an editor with the file open is not killed;
# "orchestrator.py" matches n_orchestrator.py (and any legacy *orchestrator.py).
ZOMBIE_PATTERNS=(
  "python.*orchestrator\.py"
  "python.*lane_controller_node\.py"
  "python.*waypoint_manager_node\.py"
  "python.*serve_dashboard\.py"
  "rosbridge_websocket"
  "web_video_server"
)
FOUND_ZOMBIES=0
for pat in "${ZOMBIE_PATTERNS[@]}"; do
  if pgrep -f "$pat" > /dev/null 2>&1; then
    FOUND_ZOMBIES=1
    echo "[warn] leftover process(es) matching '$pat':"
    pgrep -af "$pat" | sed 's/^/         /'
    pkill -f "$pat" 2>/dev/null || true
  fi
done
if [[ "$FOUND_ZOMBIES" == "1" ]]; then
  sleep 1
  for pat in "${ZOMBIE_PATTERNS[@]}"; do
    if pgrep -f "$pat" > /dev/null 2>&1; then
      echo "[warn] SIGKILL survivors of '$pat'"
      pkill -9 -f "$pat" 2>/dev/null || true
    fi
  done
  echo "[ok] leftover processes cleaned up"
else
  echo "[ok] no leftover processes"
fi

# ---- Pre-check: Python has cv_bridge, rospy, yaml, networkx ----
if ! $PY -c "import rospy, cv_bridge, yaml, networkx" 2>/dev/null; then
  echo "ERROR: missing Python dependencies on $PY."
  echo "       Check with: $PY -c \"import rospy, cv_bridge, yaml, networkx\""
  echo "       To install them:"
  echo "         sudo apt install python-yaml python-networkx"
  exit 1
fi
echo "[ok] Python dependencies OK"

# ---- Log directory ----
mkdir -p "$LOG_DIR"
> "$PID_FILE"

# ---- 1. Load YAML parameters into rosparam ----
echo "[1/7] rosparam load $PARAMS_FILE"
rosparam load "$PARAMS_FILE"

# ---- Helper function to launch a background process + log + PID ----
start_proc () {
  local name="$1"; shift
  local logfile="$LOG_DIR/${name}.log"
  echo "       starting $name (log: $logfile)"
  "$@" > "$logfile" 2>&1 &
  local pid=$!
  echo "$pid $name" >> "$PID_FILE"
  sleep 0.5
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: $name died immediately after launch. See $logfile"
    tail -20 "$logfile"
    exit 1
  fi
}

# ---- 2. rosbridge_websocket (port 9090) ----
echo "[2/7] rosbridge_websocket"
start_proc rosbridge \
  rosrun rosbridge_server rosbridge_websocket _port:=9090

# ---- 3. web_video_server (port 8080) ----
echo "[3/7] web_video_server"
start_proc web_video_server \
  rosrun web_video_server web_video_server _port:=8080

# ---- 4. dashboard HTTP (port 8000) ----
echo "[4/7] serve_dashboard"
start_proc dashboard_http \
  $PY "$SCRIPTS_DIR/serve_dashboard.py" \
       --port 8000 \
       --web-dir "$WEB_DIR" \
       --map "$MAP_FILE"

# ---- 5. our nodes ----
echo "[5/7] lane_controller"
start_proc lane_controller \
  $PY "$SCRIPTS_DIR/lane_controller_node.py"

# Override map path (to avoid using $(find ...))
rosparam set "waypoint_manager/map_file" "$MAP_FILE"

echo "[6/7] waypoint_manager (path-tracker)"
start_proc waypoint_manager \
  $PY "$SCRIPTS_DIR/waypoint_manager_node.py"

echo "[7/7] n_orchestrator (sole cmd_vel publisher)"
start_proc orchestrator \
  $PY "$SCRIPTS_DIR/$ORCHESTRATOR_NAME"

# ---- Summary ----
echo
echo "================================================="
echo "  All services started!"
echo "================================================="
echo "  Dashboard web:  http://$(hostname -I | awk '{print $1}'):8000"
echo "  Video stream:   http://$(hostname -I | awk '{print $1}'):8080"
echo "  Rosbridge WS:   ws://$(hostname -I | awk '{print $1}'):9090"
echo
echo "  Logs:           $LOG_DIR/"
echo "  PID file:       $PID_FILE"
echo
echo "  To stop everything: ./stop_all.sh"
echo "  To watch a log: tail -f $LOG_DIR/lane_controller.log
                  tail -f $LOG_DIR/orchestrator.log"
echo "================================================="
echo
echo "REMINDER: lane_follower.py must be started separately"
echo "          in its conda Python 3.6.9 environment."
echo "          Without it, /lane_mask_bev is not published"
echo "          and lane_controller stays in STOP state."

# Define colors using printf to ensure maximum compatibility on Zsh/Bash
G=$(printf '\033[32m') # Green (Body)
W=$(printf '\033[36m') # Cyan (Windows)
Y=$(printf '\033[33m') # Golden Yellow (Wheels and details)
R=$(printf '\033[0m')  # Color reset
C_WARN=$(printf '\033[93m') # Light yellow for warning text

# Print the artwork with colors applied at specific points
cat <<EOF
${G}                                                       ___________                        ${R}
${G}                                             __..--""""           """"--..__              ${R}
${G}                                         _.-"""""""""-----...      ______ \`.            ${R}
${G}                                      .-"                      ${W}l ,-""    \\ "-.\`.          ${R}
${G}                                   .-"                         ${W}; ;        ;   \\ ""--.._   ${R}
${G}                                 .'                           ${W}: :         |    ;      .l  ${R}
${G}                           _.._.'                             ${W}; ;  ___    |    ;    .' :  ${R}
${G}                          (  .'                              ${W}: :  :   ".  :..-'   .'    ; ${R}
${G}                           )'                                ${W}| ;  ; __.'-"(     .'  .--.: ${R}
${G}                   ___...-'""""----....____          ______.-' ${W}:-/.'       \\_.-'  .' .-\\l${R}
${G}           __..--""                        """"""""          ${W}/\\"          ;    / ${Y}.gs./\\;${R}
${G}       _.-"                                                   ${W}/  ;          |   . ${Y}d\$P"Tb  ${R}
${G}    .-""-,                       ____                        ${W}/   |          :   ;:${Y}\$\$   \$; ${R}
${G}  .'     ;                    ,""    ""--..__               ${W}/    :          |   ${Y}\$\$\$;   :\$ ${R}
${G} /"-._  /                     ;       ____..-'    .-"""-.  ${W}/     :          ;  _${Y}\$\$\$;   :\$ ${R}
${G}:     ""--.._          ___....+---""""          .'  _._  \\/${W}      |         _:-" ${Y}\$\$\$;   :\$ ${R}
${G};                                              ${W}/  .${Y}d\$\$\$b.${W}/       ;      .-".'   :${Y}\$\$\$   \$P ${R}
${G}:            .----...____                      ${W}:  ${Y}dP' \`T\$P        ${W}|   .-" .' ${Y}_.gd\$\$\$\$b_d\$' ${R}
${G};    __...---|    bug    |----....____         ${W}| :${Y}\$     \$b        ${W}: .'   (.-"  ${Y}\`T\$\$\$\$\$\$P'  ${R}
${G};  .';       '----...____;       /    "-.      ${W}; ${Y}\$;     :\$;${G}_____..-"  .-"                  ${R}
${G}: /  :                          /        \\__..-'${W}:${Y}\$       \$\$ ${G};-.    .-"                     ${R}
${G} Y    ;                        /          ;     ${Y}\$;       :\$;${G}|  \`.-"                        ${R}
${G} :    :                       /           |     ${Y}\$\$       \$\$;${G}:.-"                           ${R}
${Y} '\$\$\$ggggp...${G}____            /            :     ${Y}:\$;     :\$\$                                ${R}
${Y}  \$\$\$\$\$\$\$\$\$\$\$\$   ${G}""""----...:________....gggg${Y}\$\$\$\$\$\$     \$\$;                                ${R}
${Y}  'T\$\$\$\$\$\$\$\$P'                           T\$\$\$\$\$\$\$\$\$b._.d\$P                                 ${R}
${Y}    \`T\$\$\$\$P'                              T\$\$\$\$\$\$\$\$\$\$\$\$\$P                                  ${R}
${Y}                                          \`T\$\$\$\$\$\$\$\$\$P'${R}
EOF
