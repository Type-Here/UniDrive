#!/bin/bash
# ros_restart.sh — Clean ROS1 restart for JetAuto (Jetson Nano)
# =============================================================
#
# Usage:
#   ./ros_restart.sh [--local|--remote] [--slam] [--record]
#
# Network flags (default: --remote):
#   --local    Bind ROS to localhost (127.0.0.1)
#              Stable across WiFi changes. Remote RViz won't work.
#   --remote   Bind ROS to lab WiFi IP (auto-detected)
#              Required for RViz on a remote PC.
#              Override IP with: ROS_IP=x.x.x.x ./ros_restart.sh
#
# Action flags (optional, combinable):
#   --slam     Launch gmapping after bringup
#   --record   Start rosbag recording (camera + odom) after bringup
#
# Examples:
#   ./ros_restart.sh --local
#   ./ros_restart.sh --local --record
#   ./ros_restart.sh --remote --slam
#   ROS_IP=192.168.1.42 ./ros_restart.sh --remote

# ── Parse flags ───────────────────────────────────────────────────────────────
USE_LOCAL=false
DO_SLAM=false
DO_RECORD=false

for arg in "$@"; do
    case "$arg" in
        --local)   USE_LOCAL=true  ;;
        --remote)  USE_LOCAL=false ;;
        --slam)    DO_SLAM=true    ;;
        --record)  DO_RECORD=true  ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: ./ros_restart.sh [--local|--remote] [--slam] [--record]"
            exit 1
            ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[ros_restart]${NC} $*"; }
warn()  { echo -e "${YELLOW}[ros_restart]${NC} $*"; }
error() { echo -e "${RED}[ros_restart]${NC} $*"; exit 1; }

# ZSH + .zshrc is required to have roscore and all ROS tools in PATH.
# Every ROS command is run as: zsh -c 'source ~/.zshrc; <command>'
ZSH_ROS="zsh -c 'source $HOME/jetauto_ws/.zshrc"

# ── Step 1: resolve ROS IP ────────────────────────────────────────────────────
if [ "$USE_LOCAL" = true ]; then
    ROS_IP=127.0.0.1
    info "Network mode : LOCAL (loopback) — remote RViz will not work"
else
    if [ -z "$ROS_IP" ]; then
        ROS_IP=$(ip route get 8.8.8.8 2>/dev/null \
            | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' \
            | head -1)
    fi
    if [ -z "$ROS_IP" ]; then
        ROS_IP=$(hostname -I | tr ' ' '\n' \
            | grep -v '^127\.' | grep -v '^169\.' | head -1)
    fi
    if [ -z "$ROS_IP" ]; then
        warn "Could not detect IP — falling back to localhost"
        ROS_IP=127.0.0.1
    fi
    if ! ping -c 1 -W 2 "$ROS_IP" &>/dev/null; then
        warn "Cannot ping $ROS_IP — falling back to localhost"
        ROS_IP=127.0.0.1
    fi
    info "Network mode : REMOTE ($ROS_IP)"
fi

export ROS_IP=$ROS_IP
export ROS_HOSTNAME=$ROS_IP
export ROS_MASTER_URI=http://${ROS_IP}:11311

# ── Step 2: kill stale ROS processes ─────────────────────────────────────────
info "Stopping all ROS processes..."

sudo systemctl stop start_app_node 2>/dev/null && \
    info "  start_app_node stopped" || \
    warn "  start_app_node was not running"

for proc in roscore rosmaster roslaunch rosout; do
    pkill -SIGTERM -f "$proc" 2>/dev/null || true
done
sleep 2
for proc in roscore rosmaster roslaunch rosout; do
    pkill -SIGKILL -f "$proc" 2>/dev/null || true
done
sleep 1

# ── Step 3: wait for port 11311 to be free ────────────────────────────────────
info "Waiting for port 11311 to be free..."
for i in $(seq 1 15); do
    if ! ss -tulpn 2>/dev/null | grep -q ':11311'; then
        info "  Port 11311 is free"; break
    fi
    if [ $i -eq 15 ]; then
        warn "Still busy — force killing port 11311"
        sudo fuser -k 11311/tcp 2>/dev/null || true
        sleep 2
    fi
    sleep 1
done

# ── Step 4: start roscore via zsh ────────────────────────────────────────────
info "Starting roscore..."
info "  ROS_MASTER_URI=$ROS_MASTER_URI"

zsh -c "
    source $HOME/jetauto_ws/.zshrc
    export ROS_IP=$ROS_IP
    export ROS_HOSTNAME=$ROS_IP
    export ROS_MASTER_URI=http://${ROS_IP}:11311
    roscore &
    sleep 5
" &
ROSCORE_PID=$!

# Wait until master responds
info "Waiting for roscore to be ready..."
for i in $(seq 1 20); do
    if zsh -c "source $HOME/jetauto_ws/.zshrc; \
               export ROS_MASTER_URI=http://${ROS_IP}:11311; \
               rostopic list" &>/dev/null 2>&1; then
        info "  roscore is up"
        break
    fi
    [ $i -eq 20 ] && error "roscore did not start — check network config"
    sleep 1
done

# ── Step 5: bringup ───────────────────────────────────────────────────────────
info "Launching jetauto bringup..."

BRINGUP_LAUNCH=$(find ~/jetauto_ws -name "bringup.launch" 2>/dev/null | head -1)
[ -z "$BRINGUP_LAUNCH" ] && error "Could not find bringup.launch"
info "  Using: $BRINGUP_LAUNCH"

zsh -c "
    source $HOME/jetauto_ws/.zshrc
    export ROS_IP=$ROS_IP
    export ROS_HOSTNAME=$ROS_IP
    export ROS_MASTER_URI=http://${ROS_IP}:11311
    roslaunch $BRINGUP_LAUNCH &
    wait
" &
BRINGUP_PID=$!

info "Waiting for /odom and camera topics..."
for i in $(seq 1 30); do
    ODOM=$(zsh -c "source $HOME/jetauto_ws/.zshrc; \
                   export ROS_MASTER_URI=http://${ROS_IP}:11311; \
                   rostopic list 2>/dev/null" | grep -c "^/odom$" || true)
    CAM=$(zsh  -c "source $HOME/jetauto_ws/.zshrc; \
                   export ROS_MASTER_URI=http://${ROS_IP}:11311; \
                   rostopic list 2>/dev/null" | grep -c "astra_cam" || true)
    if [ "$ODOM" -gt 0 ] && [ "$CAM" -gt 0 ]; then
        info "  Topics are up"; break
    fi
    [ $i -eq 30 ] && warn "Timeout — some topics may not be ready yet"
    sleep 1
done

# ── Step 6: optional actions ─────────────────────────────────────────────────
if [ "$DO_SLAM" = true ]; then
    info "Launching gmapping SLAM..."
    SLAM_LAUNCH=$(find ~/jetauto_ws -name "*.launch" 2>/dev/null \
        | xargs grep -l "gmapping" 2>/dev/null | head -1)
    if [ -n "$SLAM_LAUNCH" ]; then
        zsh -c "
            source $HOME/jetauto_ws/.zshrc
            export ROS_IP=$ROS_IP
            export ROS_HOSTNAME=$ROS_IP
            export ROS_MASTER_URI=http://${ROS_IP}:11311
            roslaunch $SLAM_LAUNCH &
            wait
        " &
    else
        zsh -c "
            source $HOME/jetauto_ws/.zshrc
            export ROS_IP=$ROS_IP
            export ROS_MASTER_URI=http://${ROS_IP}:11311
            rosrun gmapping slam_gmapping scan:=/scan _odom_frame:=odom &
            wait
        " &
    fi
fi

if [ "$DO_RECORD" = true ]; then
    info "Starting rosbag recording..."
    BAGFILE="$HOME/recording_$(date +%Y%m%d_%H%M%S).bag"
    zsh -c "
        source $HOME/jetauto_ws/.zshrc
        export ROS_IP=$ROS_IP
        export ROS_MASTER_URI=http://${ROS_IP}:11311
        rosbag record /astra_cam/rgb/image_raw /odom -O $BAGFILE
    " &
    info "  Recording to: $BAGFILE"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
info "================================================"
info "  ROS up — $ROS_MASTER_URI"
info "  Active topics:"
zsh -c "source $HOME/jetauto_ws/.zshrc; \
        export ROS_MASTER_URI=http://${ROS_IP}:11311; \
        rostopic list 2>/dev/null" \
    | grep -E "odom|cmd_vel|joy|astra|scan|imu" \
    | sed 's/^/    /'
info "================================================"
info "  Ctrl+C to stop everything"

trap "info 'Stopping...'; kill $BRINGUP_PID $ROSCORE_PID 2>/dev/null; exit 0" \
    SIGINT SIGTERM
wait $BRINGUP_PID