#!/usr/bin/env python2
"""
perception_supervisor_node.py -- start/stop the perception node from the dashboard.

The perception node lives in a different Python (conda 3.6) than the rest of the
stack (system 2.7), so it cannot be imported or launched in-process: it has
always been started by hand in a second terminal via ./run-models.sh. This node
wraps those two shell scripts behind a ROS topic pair so the dashboard can start
it, stop it, and show whether it is alive.

    /perception/cmd     std_msgs/String   "start" | "stop" | "restart"
    /perception/status  std_msgs/String   JSON, latched, republished ~1 Hz

Status payload:
    {"state": "STOPPED|STARTING|RUNNING|STOPPING|ERROR",
     "pid": 1234 | null, "mode": "both", "calibrated": true,
     "log": "/tmp/...", "message": "..."}

`calibrated` is simply whether the BEV calibration file exists -- it is what
makes the dashboard offer a calibration on first boot.

Conda: run-models.sh activates the environment itself when PERCEPTION_CONDA_ENV
is set, so both a manual launch from an already-active conda shell and this
supervised launch (which inherits the system Python 2 environment from
start_all.sh) end up running the same interpreter. Configure it under
`perception:` in config/lane_params.yaml.

Work happens on a single worker thread: run-models.sh blocks until the TensorRT
engines are loaded (~30 s), which must never happen inside a ROS callback.
"""
from __future__ import print_function

import json
import os
import subprocess
import threading

import rospy
from std_msgs.msg import String


PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PerceptionSupervisor(object):

    STOPPED  = "STOPPED"
    STARTING = "STARTING"
    RUNNING  = "RUNNING"
    STOPPING = "STOPPING"
    ERROR    = "ERROR"

    def __init__(self):
        rospy.init_node("perception_supervisor", anonymous=False)

        ns = "perception/"

        def rp(k, d):
            return rospy.get_param(ns + k, d)

        self.cmd_topic    = rp("cmd_topic", "/perception/cmd")
        self.status_topic = rp("status_topic", "/perception/status")
        self.run_script   = rp("run_script", os.path.join(PKG_DIR, "run-models.sh"))
        self.stop_script  = rp("stop_script", os.path.join(PKG_DIR, "stop-models.sh"))
        self.mode         = rp("mode", "both")
        self.extra_args   = rp("extra_args", "")
        self.conda_env    = rp("conda_env", "")
        self.conda_sh     = rp("conda_sh", "")
        self.calibration_file = rp(
            "calibration_file", os.path.join(PKG_DIR, "perception", "calibration.json"))
        self.proc_pattern = rp("process_pattern", "perception_node.py")
        self.status_rate  = float(rp("status_rate_hz", 1.0))
        self.log_dir      = rp("log_dir", "/tmp/jetauto_autonomous_logs")
        self.log_file     = os.path.join(self.log_dir, "perception_supervisor_run.log")

        self._lock    = threading.Lock()
        self._busy    = False          # a start/stop is in flight
        self._state   = self.STOPPED
        self._message = ""
        self._last_published = None

        for path, what in ((self.run_script, "run_script"),
                           (self.stop_script, "stop_script")):
            if not os.path.isfile(path):
                rospy.logerr("[perception_sup] %s not found: %s", what, path)
        if not self.conda_env:
            rospy.logwarn(
                "[perception_sup] perception/conda_env is not set -- run-models.sh "
                "will use whatever python3 is on PATH. If starting perception from "
                "the dashboard fails, set it in config/lane_params.yaml.")

        try:
            os.makedirs(self.log_dir)
        except OSError:
            pass

        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=1, latch=True)
        rospy.Subscriber(self.cmd_topic, String, self._cmd_cb, queue_size=4)

        # Adopt a node that is already running (e.g. started by hand).
        self._state = self.RUNNING if self._pid() else self.STOPPED
        self._publish_status(force=True)

        rospy.loginfo("[perception_sup] ready. mode=%s run=%s",
                      self.mode, self.run_script)

    # -- Process helpers -------------------------------------------------------

    def _pid(self):
        """PID of the running perception node, or None. Same lookup the shell
        scripts use, so the three agree on what 'running' means."""
        try:
            out = subprocess.check_output(["pgrep", "-f", self.proc_pattern])
        except (subprocess.CalledProcessError, OSError):
            return None
        pids = [line for line in out.split() if line.strip()]
        if not pids:
            return None
        try:
            return int(pids[0])
        except ValueError:
            return None

    def _child_env(self):
        env = os.environ.copy()
        if self.conda_env:
            env["PERCEPTION_CONDA_ENV"] = str(self.conda_env)
        if self.conda_sh:
            env["PERCEPTION_CONDA_SH"] = str(self.conda_sh)
        return env

    def _run(self, argv):
        """Run a script to completion, appending its output to the log."""
        with open(self.log_file, "a") as log:
            log.write("\n=== %s : %s ===\n" % (rospy.get_time(), " ".join(argv)))
            log.flush()
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=open(os.devnull, "r"),
                                    env=self._child_env(), cwd=PKG_DIR)
            return proc.wait()

    # -- Commands --------------------------------------------------------------

    def _cmd_cb(self, msg):
        cmd = (msg.data or "").strip().lower()
        if cmd not in ("start", "stop", "restart"):
            rospy.logwarn("[perception_sup] unknown command: %r", cmd)
            return
        with self._lock:
            if self._busy:
                rospy.logwarn("[perception_sup] busy, ignoring '%s'", cmd)
                self._message = "busy: another start/stop is in progress"
                return
            self._busy = True
        worker = threading.Thread(target=self._worker, args=(cmd,))
        worker.daemon = True   # never hold up node shutdown
        worker.start()

    def _worker(self, cmd):
        try:
            if cmd == "start":
                self._do_start()
            elif cmd == "stop":
                self._do_stop()
            else:
                self._do_stop()
                if not rospy.is_shutdown():
                    self._do_start()
        except Exception as exc:                      # never kill the thread silently
            rospy.logerr("[perception_sup] '%s' failed: %s", cmd, exc)
            self._set(self.ERROR, "%s failed: %s" % (cmd, exc))
        finally:
            with self._lock:
                self._busy = False

    def _do_start(self):
        if self._pid():
            self._set(self.RUNNING, "already running")
            return
        self._set(self.STARTING, "loading engines, this takes ~30s")
        argv = [self.run_script, self.mode] + self.extra_args.split()
        rc = self._run(argv)
        pid = self._pid()
        if pid:
            self._set(self.RUNNING, "")
            rospy.loginfo("[perception_sup] perception started (pid %d)", pid)
        else:
            hint = ""
            if not self.conda_env:
                hint = (" -- perception/conda_env is unset; the conda Python 3 "
                        "environment is probably not being activated")
            self._set(self.ERROR,
                      "start failed (exit %d), see %s%s" % (rc, self.log_file, hint))
            rospy.logerr("[perception_sup] start failed (exit %d); see %s",
                         rc, self.log_file)

    def _do_stop(self):
        if not self._pid():
            self._set(self.STOPPED, "not running")
            return
        self._set(self.STOPPING, "")
        rc = self._run([self.stop_script])
        if self._pid():
            self._set(self.ERROR, "stop failed (exit %d), see %s"
                      % (rc, self.log_file))
        else:
            self._set(self.STOPPED, "")
            rospy.loginfo("[perception_sup] perception stopped")

    # -- Status ----------------------------------------------------------------

    def _set(self, state, message):
        self._state = state
        self._message = message
        self._publish_status(force=True)

    def _status_dict(self):
        pid = self._pid()
        state = self._state
        # Trust the process table over our own bookkeeping: perception can be
        # started or killed from a terminal behind our back.
        if state not in (self.STARTING, self.STOPPING):
            if pid and state != self.RUNNING:
                state = self.RUNNING
            elif not pid and state == self.RUNNING:
                state = self.STOPPED
            self._state = state
        return {
            "state":      state,
            "pid":        pid,
            "mode":       self.mode,
            "calibrated": os.path.exists(self.calibration_file),
            "log":        self.log_file,
            "message":    self._message,
        }

    def _publish_status(self, force=False):
        payload = json.dumps(self._status_dict(), sort_keys=True)
        if not force and payload == self._last_published:
            return
        self._last_published = payload
        self.status_pub.publish(String(data=payload))

    def run(self):
        rate = rospy.Rate(max(0.2, self.status_rate))
        while not rospy.is_shutdown():
            self._publish_status()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break


if __name__ == "__main__":
    try:
        PerceptionSupervisor().run()
    except rospy.ROSInterruptException:
        pass
