#!/usr/bin/env bash
#
# stop-models.sh -- gracefully stop the perception node started by run-models.sh.
#
# run-models.sh launches perception_node.py in the foreground of its own
# terminal. This script finds that process (in any terminal) and shuts it down
# the same way Ctrl-C would: it sends SIGINT, which rospy turns into a clean
# shutdown (rospy.spin() returns and the node's on_shutdown handler runs).
#
# Usage:
#   ./stop-models.sh          # SIGINT, wait up to 10s, then escalate if needed
#
# Exit status:
#   0  the node was stopped (or was not running)
#   1  the node could not be stopped
#
set -uo pipefail

PATTERN="perception_node.py"
PID_FILE="/tmp/jetauto_perception.pid"
GRACE_SECONDS=10

# Collect PIDs of the running perception node (exclude this script itself).
mapfile -t PIDS < <(pgrep -f "${PATTERN}" || true)

if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "No running '${PATTERN}' process found -- nothing to stop."
    rm -f "${PID_FILE}"
    exit 0
fi

echo "Stopping ${PATTERN} (PIDs: ${PIDS[*]}) ..."

# 1. Ask politely: SIGINT == Ctrl-C -> rospy clean shutdown.
kill -INT "${PIDS[@]}" 2>/dev/null || true

# 2. Wait for the process(es) to exit gracefully.
for ((i = 0; i < GRACE_SECONDS; i++)); do
    still_running=false
    for pid in "${PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            still_running=true
            break
        fi
    done
    if ! ${still_running}; then
        echo "Perception node stopped gracefully."
        rm -f "${PID_FILE}"
        exit 0
    fi
    sleep 1
done

# 3. Escalate: SIGTERM, then SIGKILL as a last resort.
echo "Still running after ${GRACE_SECONDS}s -- sending SIGTERM ..."
kill -TERM "${PIDS[@]}" 2>/dev/null || true
sleep 2

for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
        echo "PID ${pid} ignored SIGTERM -- sending SIGKILL ..."
        kill -KILL "${pid}" 2>/dev/null || true
    fi
done
sleep 1

# Final check.
for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
        echo "Failed to stop PID ${pid}." >&2
        exit 1
    fi
done

echo "Perception node stopped."
rm -f "${PID_FILE}"
exit 0
