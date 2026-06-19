#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Object-detection driving overrides: traffic lights and STOP signs.

This module is the orchestrator-side consumer of the perception node. The
perception node (``jetauto_autonomous/perception/perception_node.py``, running
on the Jetson) runs YOLO and publishes its detections as a JSON string on
``/object_detection/drive``. This handler turns those detections into a simple
stop/go decision that ``n_orchestrator.py`` applies once per control tick.

Usage from the orchestrator (called via function, no extra ROS node):

    from object_detection.traffic_sign_handler import (
        TrafficSignHandler, TRAFFIC_STOP, STOP_SIGN)

    self._obj_det = TrafficSignHandler(enable=..., topic=..., ...)
    ...
    decision = self._obj_det.evaluate()   # at the top of _step()
    if decision.stop:
        self._publish_orc_state(decision.state)
        self._cmd_pub.publish(Twist())
        return

Behaviour (faithful to feature/merge-object-detection's new_orchestrator):
  * A frame with the red label latches RED -> stop until a green is seen.
  * A frame with the green label latches GREEN -> go.
  * A frame with the stop label arms a timed hold (``stop_hold_s`` seconds).
  * A frame with none of these keeps the previously latched state.
Default state is GREEN so a silent/absent detection topic never blocks driving.
"""
from __future__ import print_function

import json
from collections import namedtuple

import rospy
from std_msgs.msg import String

# State strings published on /orchestrator/state while an override is active.
# The orchestrator does NOT store these in self._state, so navigation resumes
# from wherever it left off once the override clears.
TRAFFIC_STOP = "TRAFFIC_STOP"   # held at a red light
STOP_SIGN    = "STOP_SIGN"      # held at a stop sign

# Per-tick decision returned to the orchestrator.
Decision = namedtuple("Decision", ["stop", "state"])
GO = Decision(False, None)


class TrafficSignHandler(object):
    """Subscribes to the perception node's JSON detections and decides stop/go.

    Owns its own ``rospy.Subscriber``; the orchestrator only calls
    :meth:`evaluate` each control tick.
    """

    def __init__(self, enable=True, topic="/object_detection/drive",
                 red_label="red_TL", green_label="green_TL",
                 stop_label="stop_s", stop_hold_s=3.0):
        self.enabled       = bool(enable)
        self._red          = red_label
        self._green        = green_label
        self._stop_label   = stop_label
        self._stop_hold_s  = float(stop_hold_s)

        # Latched light: GREEN (go) until a red is detected; a red stays in
        # effect until a green is seen ("stop until you see the green light").
        self._traffic_light = "GREEN"
        # Stop sign: a timed hold, auto-cleared by evaluate() once it expires.
        self._stop_active = False
        self._stop_until  = rospy.Time(0)

        if self.enabled:
            rospy.Subscriber(topic, String, self._traffic_cb, queue_size=1)
            rospy.loginfo(
                "[obj_det] override ON: topic=%s red='%s' green='%s' "
                "stop='%s' hold=%.1fs",
                topic, red_label, green_label, stop_label, self._stop_hold_s)
        else:
            rospy.loginfo("[obj_det] override OFF (driving-only mode)")

    # ----------------------------------------------------------- subscriber

    def _traffic_cb(self, msg):
        """Parse the perception node's JSON detections and latch the state.

        Payload: ``{"detections": [{"class_name": ..., "score": ..., ...}]}``.
        A frame with a red label -> RED; a green label -> GREEN; the stop label
        arms a timed hold; a frame with none of these keeps the latched state.
        """
        try:
            detections = json.loads(msg.data).get("detections", [])
            labels = [d.get("class_name") for d in detections]
        except (ValueError, AttributeError, TypeError):
            rospy.logwarn_throttle(5.0, "[obj_det] bad detection payload")
            return

        if self._red in labels:
            if self._traffic_light != "RED":
                rospy.logwarn("[obj_det] RED light detected -> STOP")
            self._traffic_light = "RED"
        elif self._green in labels:
            if self._traffic_light != "GREEN":
                rospy.loginfo("[obj_det] GREEN light detected -> GO")
            self._traffic_light = "GREEN"
        elif self._stop_label in labels:
            if not self._stop_active:
                rospy.logwarn("[obj_det] STOP sign detected -> STOP")
            self._stop_active = True
            self._stop_until  = rospy.Time.now() + rospy.Duration(self._stop_hold_s)
        # else: neither seen this frame -> keep the latched state.

    # -------------------------------------------------------------- decision

    def evaluate(self):
        """Return ``Decision(stop, state)`` for this tick.

        Red light: stop indefinitely (until green). Stop sign: stop until the
        hold expires, then auto-clear and go. Called once per control tick.
        """
        if not self.enabled:
            return GO

        if self._traffic_light == "RED":
            return Decision(True, TRAFFIC_STOP)

        if self._stop_active:
            if rospy.Time.now() >= self._stop_until:
                self._stop_active = False
                rospy.loginfo("[obj_det] STOP sign hold expired -> GO")
            else:
                return Decision(True, STOP_SIGN)

        return GO
