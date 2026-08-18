"""
Interactive BEV calibration session, driven from the web dashboard.

The perception node owns an ``AutoCalibration`` that turns every segmentation
mask into the BEV published on ``/lane_mask_bev``. Historically that warp was
computed blind, once, on the first frame the node ever saw, and could only be
redone by deleting calibration.json and restarting (~30 s of engine reload).

This module makes it a deliberate, reversible operation:

    start / redo  ->  capture the next mask, compute a CANDIDATE calibration
                      and publish two preview JPEGs (picked corners + the warp
                      they produce). The live warp is untouched.
    apply         ->  commit the candidate and persist it. Takes effect on the
                      next frame; no engine reload.
    abort         ->  drop the candidate. Nothing to undo -- the live warp was
                      never modified, which is the whole point of staging.

ROS is optional: without it the FSM still runs (used by the off-robot tests),
it simply publishes nothing.

Threading: commands arrive on a rospy callback thread but are only *parked*
there. They are consumed by ``poll()`` on the inference thread, so the
calibrator is never mutated from two threads at once.
"""
import json
import threading

import cv2
import numpy as np

try:
    import rospy
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
    _HAVE_ROS = True
except ImportError:                                   # off-robot tests
    _HAVE_ROS = False


CMD_TOPIC     = "/bev_calibration/cmd"
STATUS_TOPIC  = "/bev_calibration/status"
POINTS_TOPIC  = "/bev_calibration/preview_points"
WARP_TOPIC    = "/bev_calibration/preview_warp"

VALID_COMMANDS = ("start", "redo", "apply", "abort")


class BevCalibrationSession(object):
    # FSM states (mirrored verbatim in the dashboard)
    IDLE        = "IDLE"         # nothing staged
    CAPTURING   = "CAPTURING"    # armed, waiting for the next segmented frame
    PREVIEW     = "PREVIEW"      # candidate computed, previews published
    APPLIED     = "APPLIED"      # candidate committed + saved (transient)
    FAILED      = "FAILED"       # last attempt found no lane corners
    UNAVAILABLE = "UNAVAILABLE"  # node running without segmentation

    def __init__(self, calib, lane_label, calibration_file,
                 available=True, jpeg_quality=80):
        self.calib            = calib
        self.lane_label       = int(lane_label)
        self.calibration_file = calibration_file
        self.jpeg_quality     = int(jpeg_quality)

        self.state  = self.IDLE if available else self.UNAVAILABLE
        self.reason = ""

        self._staged   = None      # CalibrationResult awaiting apply/abort
        self._pending  = None      # command parked by the ROS callback thread
        self._lock     = threading.Lock()

        self._status_pub = None
        self._points_pub = None
        self._warp_pub   = None
        self._sub        = None

    # -- ROS wiring ------------------------------------------------------------

    def start_ros(self):
        """Advertise the status/preview topics and subscribe to commands.

        Everything is latched: a dashboard that connects (or opens the panel)
        after the fact immediately receives the current state and the last
        previews, instead of staring at an empty panel until the next event.
        """
        if not _HAVE_ROS:
            return
        self._status_pub = rospy.Publisher(
            STATUS_TOPIC, String, queue_size=1, latch=True)
        self._points_pub = rospy.Publisher(
            POINTS_TOPIC, CompressedImage, queue_size=1, latch=True)
        self._warp_pub = rospy.Publisher(
            WARP_TOPIC, CompressedImage, queue_size=1, latch=True)
        self._sub = rospy.Subscriber(
            CMD_TOPIC, String, self._cmd_cb, queue_size=4)
        self.publish_status()

    def _cmd_cb(self, msg):
        cmd = (msg.data or "").strip().lower()
        if cmd not in VALID_COMMANDS:
            self._log_warn("[bev_calib] ignoring unknown command: %r" % cmd)
            return
        if self.state == self.UNAVAILABLE:
            self._log_warn("[bev_calib] '%s' ignored: node is running without "
                           "segmentation" % cmd)
            return
        with self._lock:
            self._pending = cmd
        self._log_info("[bev_calib] command queued: %s" % cmd)

    # -- Driven from the inference thread --------------------------------------

    def poll(self, mask):
        """
        Called once per segmented frame. Consumes a queued command and, when
        armed, captures this mask as the calibration candidate.
        """
        if self.state == self.UNAVAILABLE:
            return
        with self._lock:
            cmd, self._pending = self._pending, None

        if cmd in ("start", "redo"):
            self._staged = None
            self.state, self.reason = self.CAPTURING, ""
            self.publish_status()
        elif cmd == "apply":
            self._apply()
            return
        elif cmd == "abort":
            self._abort()
            return

        if self.state == self.CAPTURING:
            self._capture(mask)

    def arm(self):
        """Arm a capture without going through ROS (first-run auto-calibration)."""
        if self.state == self.UNAVAILABLE:
            return
        self._staged = None
        self.state, self.reason = self.CAPTURING, ""

    # -- Session steps ---------------------------------------------------------

    def _capture(self, mask):
        result = self.calib.compute(mask, self.lane_label)
        if not result.ok:
            self._staged = None
            self.state, self.reason = self.FAILED, result.reason
            self._log_warn("[bev_calib] calibration failed: %s" % result.reason)
            self.publish_status()
            return

        self._staged = result
        self.state, self.reason = self.PREVIEW, ""
        try:
            points_img, warp_img = self.calib.render_debug(mask, result.src_points)
            self._publish_preview(self._points_pub, points_img)
            self._publish_preview(self._warp_pub, warp_img)
        except Exception as exc:                      # previews are best-effort
            self._log_warn("[bev_calib] preview rendering failed: %s" % exc)
        self._log_info("[bev_calib] candidate ready: %s"
                       % np.asarray(result.src_points).tolist())
        self.publish_status()

    def _apply(self):
        if self._staged is None:
            self.state = self.FAILED
            self.reason = "nothing to apply: run a calibration first"
            self.publish_status()
            return
        self.calib.apply_points(self._staged.src_points, self._staged.angle)
        try:
            path = self.calib.save(self.calibration_file)
            self.reason = ""
            self._log_info("[bev_calib] calibration applied and saved to %s" % path)
        except Exception as exc:
            # The warp is live either way -- be explicit that it did not persist.
            self.reason = "calibration applied but NOT saved: %s" % exc
            self._log_error("[bev_calib] %s" % self.reason)
        self._staged = None
        self.state = self.APPLIED
        self.publish_status()
        self.state = self.IDLE
        self.publish_status()

    def _abort(self):
        self._staged = None
        self.state, self.reason = self.IDLE, ""
        self._log_info("[bev_calib] calibration aborted, previous settings kept")
        self.publish_status()

    # -- Publishing ------------------------------------------------------------

    def status_dict(self):
        src = self.calib.src_points
        staged = self._staged.src_points if self._staged is not None else None
        return {
            "state":            self.state,
            "reason":           self.reason,
            "calibrated":       bool(self.calib.is_calibrated),
            "src_points":       np.asarray(src).tolist() if src is not None else None,
            "staged_points":    np.asarray(staged).tolist() if staged is not None else None,
            "angle_deg":        float(np.rad2deg(self.calib.calibration_angle)),
            "calibration_file": self.calibration_file,
        }

    def publish_status(self):
        if self._status_pub is None:
            return
        try:
            self._status_pub.publish(String(data=json.dumps(self.status_dict())))
        except Exception as exc:
            self._log_warn("[bev_calib] status publish failed: %s" % exc)

    def _publish_preview(self, pub, img):
        if pub is None or img is None:
            return
        ok, buf = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = buf.tobytes()
        pub.publish(msg)

    # -- Logging (rospy when available, print otherwise) -----------------------

    @staticmethod
    def _log_info(text):
        if _HAVE_ROS:
            rospy.loginfo(text)
        else:
            print(text)

    @staticmethod
    def _log_warn(text):
        if _HAVE_ROS:
            rospy.logwarn(text)
        else:
            print(text)

    @staticmethod
    def _log_error(text):
        if _HAVE_ROS:
            rospy.logerr(text)
        else:
            print(text)
