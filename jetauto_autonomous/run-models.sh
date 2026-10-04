#!/usr/bin/env bash
#
# run-models.sh -- launch the merged perception node (object detection + lane
# segmentation) on the JetAuto, in the background.
#
# Behaviour:
#   * Starts perception_node.py detached in the background (survives terminal
#     close) and writes its output to a log file under /tmp (like start_all.sh).
#   * The BEV calibration prompt is auto-answered "yes" by feeding it on stdin,
#     so no interaction is needed (when calibration.json already exists the
#     prompt is skipped and the answer is simply ignored).
#   * The node's startup output is streamed live to this terminal until it
#     reports it is ready ("Started inference") -- this can take ~30s while the
#     TensorRT engines load -- then the stream detaches and the node keeps
#     running in the background, logging to the file.
#
# Usage:
#   ./run-models.sh                 # both models (detection + segmentation)
#   ./run-models.sh detection       # object detection only
#   ./run-models.sh segmentation    # lane segmentation only
#   ./run-models.sh both --nodebug  # both models, no /lane_follower/debug_image
#
# The debug image (/lane_follower/debug_image) is published by default; pass
# --nodebug to turn it off.
#
# Any extra arguments are forwarded verbatim to perception_node.py, e.g.:
#   ./run-models.sh both --max-fps 15 --print-debug
#
# Prerequisites:
#   * roscore (and the camera driver) are already running.
#   * Run inside the conda Python 3 environment that has tensorrt + rospy.
#     Alternatively export PERCEPTION_CONDA_ENV=<env name> and this script
#     activates it itself (that is how perception_supervisor_node.py, and so
#     the dashboard's "start perception" button, launches it -- that process
#     inherits the system Python 2 environment and cannot activate conda
#     beforehand). PERCEPTION_CONDA_SH overrides the conda hook location.
#   * The TensorRT engines live in perception/models/ with these exact names:
#         perception/models/segmentation.engine
#         perception/models/object_detection.engine
#
# To stop it gracefully: ./stop-models.sh
#
set -uo pipefail

# Resolve paths relative to this script so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERCEPTION_DIR="${SCRIPT_DIR}/perception"
MODELS_DIR="${PERCEPTION_DIR}/models"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# perception_node.py imports auto_calibration as
# jetauto_autonomous.perception.auto_calibration when the repo root is
# importable, and falls back to a plain local import otherwise. Put the repo
# root on PYTHONPATH so the package form works too.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Activate the conda env ourselves when asked (see the header note). A caller
# that has already activated it leaves PERCEPTION_CONDA_ENV unset and nothing
# happens here.
if [ -n "${PERCEPTION_CONDA_ENV:-}" ]; then
    CONDA_SH="${PERCEPTION_CONDA_SH:-}"
    if [ -z "${CONDA_SH}" ]; then
        for candidate in \
            "${CONDA_PREFIX:-}/etc/profile.d/conda.sh" \
            "${HOME}/miniconda3/etc/profile.d/conda.sh" \
            "${HOME}/anaconda3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh"
        do
            if [ -n "${candidate}" ] && [ -f "${candidate}" ]; then
                CONDA_SH="${candidate}"
                break
            fi
        done
    fi
    if [ -z "${CONDA_SH}" ] || [ ! -f "${CONDA_SH}" ]; then
        echo "Cannot activate conda env '${PERCEPTION_CONDA_ENV}': conda.sh not found." >&2
        echo "Set PERCEPTION_CONDA_SH (or perception/conda_sh in config/lane_params.yaml)" >&2
        echo "to the full path of <conda>/etc/profile.d/conda.sh." >&2
        exit 1
    fi
    # conda's shell hook dereferences unset variables; -u would abort on it.
    set +u
    # shellcheck disable=SC1090
    . "${CONDA_SH}"
    if ! conda activate "${PERCEPTION_CONDA_ENV}"; then
        set -u
        echo "conda activate '${PERCEPTION_CONDA_ENV}' failed." >&2
        echo "Check perception/conda_env in config/lane_params.yaml." >&2
        exit 1
    fi
    set -u
    echo "[conda] activated '${PERCEPTION_CONDA_ENV}' ($(command -v python3))"
fi

DET_MODEL="${MODELS_DIR}/object_detection.engine"
SEG_MODEL="${MODELS_DIR}/segmentation.engine"

LOG_DIR="/tmp/jetauto_perception_logs"
LOG_FILE="${LOG_DIR}/perception.log"
PID_FILE="/tmp/jetauto_perception.pid"

# Marker(s) that mean "the node has processed its first frame and is ready".
READY_RE="Started inference|fps="
# How long to wait for readiness before detaching anyway (engines can be slow).
READY_TIMEOUT_S=120

# First positional argument is the mode (default: both); the rest pass through.
MODE="${1:-both}"
if [ "$#" -gt 0 ]; then
    shift
fi

# Refuse to start a second instance (two nodes fight over the same topics).
if pgrep -f "perception_node.py" > /dev/null 2>&1; then
    echo "A perception_node.py process is already running:" >&2
    pgrep -af "perception_node.py" | sed 's/^/    /' >&2
    echo "Stop it first with ./stop-models.sh" >&2
    exit 1
fi

# Validate that the engines required by the chosen mode are present.
case "${MODE}" in
    both)
        [ -f "${DET_MODEL}" ] || { echo "Missing engine: ${DET_MODEL}" >&2; exit 1; }
        [ -f "${SEG_MODEL}" ] || { echo "Missing engine: ${SEG_MODEL}" >&2; exit 1; }
        ;;
    detection)
        [ -f "${DET_MODEL}" ] || { echo "Missing engine: ${DET_MODEL}" >&2; exit 1; }
        ;;
    segmentation)
        [ -f "${SEG_MODEL}" ] || { echo "Missing engine: ${SEG_MODEL}" >&2; exit 1; }
        ;;
    *)
        echo "Unknown mode '${MODE}' (expected: both | detection | segmentation)" >&2
        exit 1
        ;;
esac

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

# Run from the perception dir so calibration.json and module imports resolve.
cd "${PERCEPTION_DIR}"

echo "----------------------------------------------------------------"
echo "Starting perception_node.py (mode=${MODE}) in the background ..."
echo "  log: ${LOG_FILE}"
echo "  PID file: ${PID_FILE}"
echo "----------------------------------------------------------------"
echo "  (Press Ctrl-C to detach from the log view, the node keeps running.)"
echo "----------------------------------------------------------------"
echo " If you wanto to stop the node, run: ./stop-models.sh"
echo " If the node fails to start, check the log above or in ${LOG_FILE}."
echo " If you want to run in foreground, run: ./perception_node.py --help"
echo "----------------------------------------------------------------"

# Launch detached (nohup -> survives terminal close, exec's so $! is the python
# PID). stdin is the here-string "y" which auto-answers the calibration prompt;
# stdout+stderr go to the log file.
nohup python3 perception_node.py \
    --mode "${MODE}" \
    --det-model "${DET_MODEL}" \
    --seg-model "${SEG_MODEL}" \
    --seg-tensorrt \
    "$@" > "${LOG_FILE}" 2>&1 <<< "y" &
PID=$!
echo "${PID}" > "${PID_FILE}"

# Stream the log live to this terminal until the node is ready (or dies / times
# out), so we can watch the slow engine-loading phase.
echo "----------------------------------------------------------------"
tail -n +1 -f "${LOG_FILE}" &
TAIL_PID=$!

ready=0
start_ts=${SECONDS}
while kill -0 "${PID}" 2>/dev/null; do
    if grep -Eq "${READY_RE}" "${LOG_FILE}" 2>/dev/null; then
        ready=1
        break
    fi
    if (( SECONDS - start_ts >= READY_TIMEOUT_S )); then
        break
    fi
    sleep 0.5
done

# Stop the live view.
kill "${TAIL_PID}" 2>/dev/null
wait "${TAIL_PID}" 2>/dev/null
echo "----------------------------------------------------------------"

if (( ready == 1 )); then
    echo "[ok] Perception node is ready and running in the background (PID ${PID})."
    echo "     Watch the log:  tail -f ${LOG_FILE}"
    echo "     Stop it:        ./stop-models.sh"
    exit 0
elif kill -0 "${PID}" 2>/dev/null; then
    echo "[warn] Not ready after ${READY_TIMEOUT_S}s, but still running (PID ${PID})."
    echo "       It may just be slow -- keep watching: tail -f ${LOG_FILE}"
    exit 0
else
    echo "[error] Perception node exited during startup. Last log lines:" >&2
    tail -n 30 "${LOG_FILE}" >&2
    rm -f "${PID_FILE}"
    exit 1
fi
