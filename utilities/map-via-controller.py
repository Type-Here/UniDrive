#!/usr/bin/env python
"""
map_builder.py - ROS1 Headless Lane Map Builder
=================================================
Drive the robot manually with a joystick controller.
The script records the odometry path and lets you place
graph nodes (intersections, curves, etc.) by pressing a
button on the controller - no RViz clicking required.

Dependencies:
    rospy, geometry_msgs, nav_msgs, visualization_msgs,
    sensor_msgs, tf, pyyaml

Usage:
    rosrun <your_package> map_builder.py
    or
    python map_builder.py

Subscribed topics:
    /odom         (nav_msgs/Odometry)   - robot path tracking
    /joy          (sensor_msgs/Joy)     - joystick input

Published topics:
    /map_markers  (visualization_msgs/MarkerArray) - RViz visualization
    /cmd_vel      (geometry_msgs/Twist)            - velocity commands

Terminal commands:
    save              - save graph to YAML + PNG image
    list              - print all nodes
    edge A B          - connect node A to node B
    label N name      - rename node N
    type N typename   - set node type (intersection|curve|straight|start|end)
    undo              - remove last node
    quit / exit / q   - shutdown the node cleanly

Controller mapping (default: generic PS3/PS4/Xbox layout):
    Left stick        - drive (linear.x)
    Right stick       - steer (angular.z)
    Button 0  (X/A)   - place node at current position
    Button 1  (O/B)   - undo last node
    Button 2  (sq/X)  - cycle node type for last placed node
    Button 3  (tri/Y) - save map to file
    Button 4  (L1)    - decrease linear speed scale
    Button 5  (R1)    - increase linear speed scale

Output files:
    track_map.yaml        - graph in YAML format
    track_map_image.png   - PNG image of the graph (requires matplotlib)
"""

import rospy
import yaml
import math
import threading

from geometry_msgs.msg import Twist, Point, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

# -- Configuration

OUTPUT_FILE   = "track_map.yaml"
FRAME_ID      = "odom"

# Topics - edit to match your robot setup
ODOM_TOPIC    = "/odom"
JOY_TOPIC     = "/ros_robot_controller/joy"     
MARKER_TOPIC  = "/map_markers"
CMDVEL_TOPIC  = "/cmd_vel"

# Path recording
PATH_MIN_DIST   = 0.05   # metres - minimum distance between path samples
NODE_MERGE_DIST = 0.15   # metres - snap radius when placing near existing node

# Controller axes indices (sensor_msgs/Joy.axes[])
AXIS_LINEAR  = 1    # left stick vertical
AXIS_ANGULAR = 3    # right stick horizontal (set to 0 for single-stick robots)

# Controller button indices (sensor_msgs/Joy.buttons[])
BTN_PLACE_NODE = 0   # X / A   - place node at current position
BTN_UNDO       = 1   # O / B   - remove last node
BTN_CYCLE_TYPE = 2   # sq / X  - cycle type of last node
BTN_SAVE       = 3   # tri/ Y  - save map
BTN_SPEED_DOWN = 4   # L1      - decrease speed
BTN_SPEED_UP   = 5   # R1      - increase speed

# Driving
LINEAR_SCALE_DEFAULT = 0.3   # m/s
ANGULAR_SCALE        = 1.0   # rad/s
SPEED_STEP           = 0.05  # m/s per button press

# Node type cycle order
NODE_TYPES = ["intersection", "curve", "straight", "start", "end"]


class MapBuilder:

    def __init__(self):
        rospy.init_node("map_builder", anonymous=False)

        # Internal state
        self.nodes        = []    # list of dicts: {id, x, y, type, label}
        self.edges        = []    # list of dicts: {from, to, length}
        self.path_pts     = []    # continuous odometry trail
        self.last_pt      = None  # last sampled path point
        self.node_id_cnt  = 0
        self.linear_scale = LINEAR_SCALE_DEFAULT
        self.running      = True

        # Button debounce: tracks previous state to detect rising edge only
        self._prev_buttons = {}

        # -- Publishers
        self.marker_pub = rospy.Publisher(MARKER_TOPIC, MarkerArray,
                                          queue_size=1, latch=True)
        self.cmdvel_pub = rospy.Publisher(CMDVEL_TOPIC, Twist, queue_size=1)

        # -- Subscribers
        rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom_cb)
        rospy.Subscriber(JOY_TOPIC,  Joy,      self.joy_cb)

        # -- Periodic marker publisher (0.5 s)
        rospy.Timer(rospy.Duration(0.5), self.publish_markers)

        # -- Terminal input thread (non-blocking for ROS spin)
        t = threading.Thread(target=self.terminal_input, daemon=True)
        t.start()

        self._print_banner()

    # -- Startup banner

    def _print_banner(self):
        rospy.loginfo("=" * 58)
        rospy.loginfo("  MAP BUILDER - headless ROS1 lane graph recorder")
        rospy.loginfo("  Odometry topic : %s", ODOM_TOPIC)
        rospy.loginfo("  Joy topic      : %s", JOY_TOPIC)
        rospy.loginfo("-" * 58)
        rospy.loginfo("  CONTROLLER MAPPING")
        rospy.loginfo("    Left stick          drive (linear.x)")
        rospy.loginfo("    Right stick         steer (angular.z)")
        rospy.loginfo("    X / A  (btn %d)      place node at current pos", BTN_PLACE_NODE)
        rospy.loginfo("    O / B  (btn %d)      undo last node",            BTN_UNDO)
        rospy.loginfo("    sq/ X  (btn %d)      cycle last node type",      BTN_CYCLE_TYPE)
        rospy.loginfo("    tri/Y  (btn %d)      save map to file",          BTN_SAVE)
        rospy.loginfo("    L1/R1  (btn %d/%d)   speed down / up",
                      BTN_SPEED_DOWN, BTN_SPEED_UP)
        rospy.loginfo("-" * 58)
        rospy.loginfo("  TERMINAL COMMANDS")
        rospy.loginfo("    save | list | edge A B | label N name")
        rospy.loginfo("    type N typename | undo | quit")
        rospy.loginfo("=" * 58)

    # -- Odometry callback

    def odom_cb(self, msg):
        """Record path point when robot moves more than PATH_MIN_DIST."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.last_pt is None or \
                self._dist(x, y, *self.last_pt) >= PATH_MIN_DIST:
            self.path_pts.append((x, y))
            self.last_pt = (x, y)

    # -- Joystick callback

    def joy_cb(self, msg):
        # Forward axes to cmd_vel
        twist = Twist()
        if len(msg.axes) > max(AXIS_LINEAR, AXIS_ANGULAR):
            twist.linear.x  =  msg.axes[AXIS_LINEAR]  * self.linear_scale
            twist.angular.z =  msg.axes[AXIS_ANGULAR] * ANGULAR_SCALE
        self.cmdvel_pub.publish(twist)

        # Detect rising edge on each button (press, not hold)
        for idx, pressed in enumerate(msg.buttons):
            prev = self._prev_buttons.get(idx, 0)
            if pressed and not prev:
                self._on_button(idx)
            self._prev_buttons[idx] = pressed

    def _on_button(self, idx):
        """Dispatch action on first frame of button press."""
        if   idx == BTN_PLACE_NODE: self._place_node_at_current()
        elif idx == BTN_UNDO:       self._undo()
        elif idx == BTN_CYCLE_TYPE: self._cycle_last_type()
        elif idx == BTN_SAVE:       self.save()
        elif idx == BTN_SPEED_DOWN:
            self.linear_scale = max(0.05, self.linear_scale - SPEED_STEP)
            rospy.loginfo("  Speed : %.2f m/s", self.linear_scale)
        elif idx == BTN_SPEED_UP:
            self.linear_scale = min(1.0, self.linear_scale + SPEED_STEP)
            rospy.loginfo("  Speed : %.2f m/s", self.linear_scale)

    # -- Node operations

    def _place_node_at_current(self):
        """Place a new node at the robot's current odometry position."""
        if self.last_pt is None:
            rospy.logwarn("  No odometry received yet - cannot place node.")
            return
        x, y = self.last_pt

        # Snap to nearby node instead of creating a duplicate
        close = self._find_close_node(x, y)
        if close is not None:
            rospy.loginfo("  Snapped to existing node id=%d  (%.2f, %.2f)",
                          close["id"], close["x"], close["y"])
            return

        node = {
            "id":    self.node_id_cnt,
            "x":     round(x, 3),
            "y":     round(y, 3),
            "type":  "intersection",
            "label": f"node_{self.node_id_cnt}"
        }
        self.nodes.append(node)
        rospy.loginfo("  + Node %d placed at (%.2f, %.2f)",
                      self.node_id_cnt, x, y)
        self.node_id_cnt += 1

    def _undo(self):
        """Remove the last placed node and any edges connected to it."""
        if not self.nodes:
            rospy.loginfo("  Nothing to undo.")
            return
        removed = self.nodes.pop()
        self.edges = [e for e in self.edges
                      if e["from"] != removed["id"] and e["to"] != removed["id"]]
        rospy.loginfo("  Removed node %d", removed["id"])

    def _cycle_last_type(self):
        """Cycle the type of the most recently placed node."""
        if not self.nodes:
            rospy.loginfo("  No nodes yet.")
            return
        n   = self.nodes[-1]
        idx = NODE_TYPES.index(n["type"]) if n["type"] in NODE_TYPES else 0
        n["type"] = NODE_TYPES[(idx + 1) % len(NODE_TYPES)]
        rospy.loginfo("  Node %d type -> %s", n["id"], n["type"])

    def _add_edge(self, id_a, id_b):
        na = next((n for n in self.nodes if n["id"] == id_a), None)
        nb = next((n for n in self.nodes if n["id"] == id_b), None)
        if na is None or nb is None:
            print(f"  Node {id_a} or {id_b} not found.")
            return
        # Prevent duplicate edges (undirected)
        if any({e["from"], e["to"]} == {id_a, id_b} for e in self.edges):
            print(f"  Edge {id_a}<->{id_b} already exists.")
            return
        length = round(self._dist(na["x"], na["y"], nb["x"], nb["y"]), 3)
        self.edges.append({"from": id_a, "to": id_b, "length": length})
        print(f"  Edge added: {id_a} <-> {id_b}  (length: {length:.2f} m)")

    def _set_label(self, nid, label):
        for n in self.nodes:
            if n["id"] == nid:
                n["label"] = label
                print(f"  Node {nid} labelled '{label}'")
                return
        print(f"  Node {nid} not found.")

    def _set_type(self, nid, typ):
        if typ not in NODE_TYPES:
            print(f"  Invalid type. Choose from: {NODE_TYPES}")
            return
        for n in self.nodes:
            if n["id"] == nid:
                n["type"] = typ
                print(f"  Node {nid} type -> '{typ}'")
                return
        print(f"  Node {nid} not found.")

    # -- Terminal input (runs in separate thread)

    def terminal_input(self):
        while self.running and not rospy.is_shutdown():
            try:
                line = input(">>> ").strip()
            except EOFError:
                break
            if not line:
                continue

            parts = line.split()
            cmd   = parts[0].lower()

            if cmd in ("quit", "exit", "q"):
                rospy.loginfo("  Shutting down...")
                self.running = False
                rospy.signal_shutdown("User quit")
                break

            elif cmd == "save":
                self.save()

            elif cmd == "list":
                if not self.nodes:
                    print("  No nodes yet.")
                for n in self.nodes:
                    print(f"  [{n['id']}] ({n['x']:.2f}, {n['y']:.2f})"
                          f"  type={n['type']}  label={n['label']}")

            elif cmd == "edge" and len(parts) == 3:
                try:
                    self._add_edge(int(parts[1]), int(parts[2]))
                except ValueError:
                    print("  Usage: edge <id_A> <id_B>")

            elif cmd == "label" and len(parts) >= 3:
                try:
                    self._set_label(int(parts[1]), " ".join(parts[2:]))
                except ValueError:
                    print("  Usage: label <id> <name>")

            elif cmd == "type" and len(parts) == 3:
                try:
                    self._set_type(int(parts[1]), parts[2])
                except ValueError:
                    print("  Usage: type <id> <typename>")

            elif cmd == "undo":
                self._undo()

            elif cmd == "help":
                print("  Commands: save | list | edge A B | label N name"
                      " | type N typename | undo | quit")
            else:
                print(f"  Unknown command '{line}' - type 'help'")

    # -- Save to YAML + PNG

    def save(self):
        data = {
            "frame_id":     FRAME_ID,
            "nodes":        self.nodes,
            "edges":        self.edges,
            "path_samples": [{"x": p[0], "y": p[1]} for p in self.path_pts]
        }
        with open(OUTPUT_FILE, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        rospy.loginfo("  Saved '%s'  (%d nodes, %d edges)",
                      OUTPUT_FILE, len(self.nodes), len(self.edges))
        try:
            self._save_image()
        except ImportError:
            rospy.logwarn("  matplotlib not available - skipping PNG export.")

    def _save_image(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect("equal")
        ax.set_title("Track Map")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, linestyle="--", alpha=0.4)

        # Odometry trail (light grey)
        if self.path_pts:
            xs, ys = zip(*self.path_pts)
            ax.plot(xs, ys, color="lightgray", linewidth=1, label="path")

        # Edges
        for e in self.edges:
            na = next(n for n in self.nodes if n["id"] == e["from"])
            nb = next(n for n in self.nodes if n["id"] == e["to"])
            ax.plot([na["x"], nb["x"]], [na["y"], nb["y"]], "b-", linewidth=2)

        # Nodes
        colors = {
            "intersection": "red",   "curve":  "orange",
            "straight":    "green",  "start":  "blue",
            "end":         "purple"
        }
        for n in self.nodes:
            c = colors.get(n["type"], "gray")
            ax.scatter(n["x"], n["y"], color=c, s=120, zorder=5)
            ax.annotate(f"  {n['id']}: {n['label']}", (n["x"], n["y"]),
                        fontsize=8, color=c)

        img_path = OUTPUT_FILE.replace(".yaml", "_image.png")
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close()
        rospy.loginfo("  Image saved: '%s'", img_path)

    # -- RViz marker publisher

    def publish_markers(self, event=None):
        ma  = MarkerArray()
        now = rospy.Time.now()
        mid = 0

        # Clear previous markers before redrawing
        del_m = Marker()
        del_m.action = Marker.DELETEALL
        del_m.header.frame_id = FRAME_ID
        del_m.header.stamp = now
        ma.markers.append(del_m)

        # Odometry trail - grey LINE_STRIP
        if len(self.path_pts) >= 2:
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = now
            m.ns = "path"; m.id = mid; mid += 1
            m.type = Marker.LINE_STRIP; m.action = Marker.ADD
            m.scale.x = 0.02
            m.color   = ColorRGBA(0.6, 0.6, 0.6, 0.8)
            m.points  = [Point(p[0], p[1], 0.0) for p in self.path_pts]
            ma.markers.append(m)

        # Graph edges - blue LINE_LIST
        for e in self.edges:
            na = next((n for n in self.nodes if n["id"] == e["from"]), None)
            nb = next((n for n in self.nodes if n["id"] == e["to"]),   None)
            if na is None or nb is None:
                continue
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = now
            m.ns = "edges"; m.id = mid; mid += 1
            m.type = Marker.LINE_LIST; m.action = Marker.ADD
            m.scale.x = 0.04
            m.color   = ColorRGBA(0.2, 0.4, 1.0, 1.0)
            m.points  = [Point(na["x"], na["y"], 0.02),
                         Point(nb["x"], nb["y"], 0.02)]
            ma.markers.append(m)

        # Graph nodes - coloured SPHERE + TEXT label
        type_color = {
            "intersection": ColorRGBA(1.0, 0.2, 0.2, 1.0),
            "curve":        ColorRGBA(1.0, 0.6, 0.0, 1.0),
            "straight":     ColorRGBA(0.2, 0.8, 0.2, 1.0),
            "start":        ColorRGBA(0.0, 0.4, 1.0, 1.0),
            "end":          ColorRGBA(0.6, 0.0, 0.8, 1.0),
        }
        for n in self.nodes:
            color = type_color.get(n["type"], ColorRGBA(0.5, 0.5, 0.5, 1.0))

            # Sphere marker
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = now
            m.ns = "nodes"; m.id = mid; mid += 1
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = n["x"]
            m.pose.position.y = n["y"]
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale = Vector3(0.12, 0.12, 0.12)
            m.color = color
            ma.markers.append(m)

            # Text label
            t = Marker()
            t.header.frame_id = FRAME_ID
            t.header.stamp = now
            t.ns = "labels"; t.id = mid; mid += 1
            t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
            t.pose.position.x = n["x"]
            t.pose.position.y = n["y"]
            t.pose.position.z = 0.20
            t.pose.orientation.w = 1.0
            t.scale.z = 0.12
            t.color   = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            t.text    = f"{n['id']}: {n['label']}"
            ma.markers.append(t)

        self.marker_pub.publish(ma)

    # -- Helpers

    def _dist(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _find_close_node(self, x, y):
        """Return existing node within NODE_MERGE_DIST, or None."""
        for n in self.nodes:
            if self._dist(x, y, n["x"], n["y"]) < NODE_MERGE_DIST:
                return n
        return None

    def run(self):
        rospy.spin()


# -- Entry point

if __name__ == "__main__":
    try:
        MapBuilder().run()
    except rospy.ROSInterruptException:
        pass