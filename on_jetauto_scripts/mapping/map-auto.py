#!/usr/bin/env python
"""
map_builder.py - ROS1 Auto-Node Path Recorder
==============================================
Drive the robot manually with joystick_controller.launch.
Every NODE_SPACING metres, a node is automatically placed and
connected to the previous one - no clicking or button presses needed.

Subscribed topics:
    /odom  (nav_msgs/Odometry) - robot position

Published topics:
    /map_markers  (visualization_msgs/MarkerArray) - RViz visualization

Terminal commands:
    save   - save graph to YAML + PNG
    list   - print all nodes
    undo   - remove last node and edge
    quit   - exit cleanly

RViz setup:
    Fixed Frame  -> odom
    Add -> MarkerArray -> /map_markers
"""

import rospy
import yaml
import math
import threading

from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA

# -- Configuration --------------------------------------------------------------

OUTPUT_FILE  = "/home/jetauto/track_map.yaml"
FRAME_ID     = "odom"

ODOM_TOPIC   = "/odom"
MARKER_TOPIC = "/map_markers"

# Distance between auto-placed nodes (metres)
# 0.05 = one node every 5 cm - good density for a small lab track
NODE_SPACING = 0.05

# ------------------------------------------------------------------------------


class MapBuilder:

    def __init__(self):
        rospy.init_node("map_builder", anonymous=False)

        self.nodes       = []   # list of {id, x, y}
        self.edges       = []   # list of {from, to, length}
        self.node_id_cnt = 0
        self.last_pt     = None
        self.running     = True

        self.marker_pub = rospy.Publisher(
            MARKER_TOPIC, MarkerArray, queue_size=1, latch=True)

        rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom_cb)
        rospy.Timer(rospy.Duration(0.5), self.publish_markers)
        threading.Thread(target=self.terminal_input, daemon=True).start()

        self._print_banner()

    # -- Banner ----------------------------------------------------------------

    def _print_banner(self):
        rospy.loginfo("=" * 55)
        rospy.loginfo("  MAP BUILDER - auto node every %.0f cm", NODE_SPACING * 100)
        rospy.loginfo("  Odom topic   : %s", ODOM_TOPIC)
        rospy.loginfo("  Markers      : %s  (add in RViz)", MARKER_TOPIC)
        rospy.loginfo("  Fixed Frame  : %s", FRAME_ID)
        rospy.loginfo("  Output file  : %s", OUTPUT_FILE)
        rospy.loginfo("-" * 55)
        rospy.loginfo("  Just drive - nodes appear automatically!")
        rospy.loginfo("  Terminal: save | list | undo | quit")
        rospy.loginfo("=" * 55)

    # -- Odometry callback - core logic ----------------------------------------

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Skip until robot has moved NODE_SPACING from last node
        if self.last_pt is not None and \
                self._dist(x, y, *self.last_pt) < NODE_SPACING:
            return

        # Place new node
        node = {"id": self.node_id_cnt, "x": round(x, 4), "y": round(y, 4)}
        self.nodes.append(node)

        # Connect to previous node automatically
        if self.node_id_cnt > 0:
            prev = self.nodes[-2]
            length = round(self._dist(
                node["x"], node["y"], prev["x"], prev["y"]), 4)
            self.edges.append({
                "from":   prev["id"],
                "to":     node["id"],
                "length": length
            })

        rospy.logdebug("  Node %d at (%.3f, %.3f)", self.node_id_cnt, x, y)
        self.node_id_cnt += 1
        self.last_pt = (x, y)

    # -- Terminal input --------------------------------------------------------

    def terminal_input(self):
        while self.running and not rospy.is_shutdown():
            try:
                line = input(">>> ").strip().lower()
            except EOFError:
                break
            if not line:
                continue

            if line in ("quit", "exit", "q"):
                rospy.loginfo("  Shutting down...")
                self.running = False
                rospy.signal_shutdown("User quit")
                break

            elif line == "save":
                self.save()

            elif line == "list":
                print(f"  {len(self.nodes)} nodes, {len(self.edges)} edges")
                for n in self.nodes:
                    print(f"  [{n['id']}] ({n['x']:.3f}, {n['y']:.3f})")

            elif line == "undo":
                self._undo()

            elif line == "help":
                print("  Commands: save | list | undo | quit")

            else:
                print(f"  Unknown command '{line}' - type 'help'")

    # -- Undo last node --------------------------------------------------------

    def _undo(self):
        if not self.nodes:
            print("  Nothing to undo.")
            return
        removed = self.nodes.pop()
        if self.edges:
            self.edges.pop()
        self.node_id_cnt -= 1
        self.last_pt = (self.nodes[-1]["x"], self.nodes[-1]["y"]) \
            if self.nodes else None
        print(f"  Removed node {removed['id']}  ({removed['x']}, {removed['y']})")

    # -- Save YAML + PNG -------------------------------------------------------

    def save(self):
        data = {
            "frame_id": FRAME_ID,
            "node_spacing_m": NODE_SPACING,
            "nodes": self.nodes,
            "edges": self.edges,
        }
        with open(OUTPUT_FILE, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        rospy.loginfo("  Saved '%s'  (%d nodes, %d edges)",
                      OUTPUT_FILE, len(self.nodes), len(self.edges))
        try:
            self._save_image()
        except ImportError:
            rospy.logwarn("  matplotlib not found - skipping PNG.")

    def _save_image(self):
        import matplotlib.pyplot as plt

        xs = [n["x"] for n in self.nodes]
        ys = [n["y"] for n in self.nodes]

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect("equal")
        ax.set_title(f"Track Map - node every {NODE_SPACING*100:.0f} cm"
                     f"  ({len(self.nodes)} nodes)")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.grid(True, linestyle="--", alpha=0.4)

        # Draw edges
        for e in self.edges:
            na = self.nodes[e["from"]]
            nb = self.nodes[e["to"]]
            ax.plot([na["x"], nb["x"]], [na["y"], nb["y"]],
                    color="royalblue", linewidth=1.5, zorder=2)

        # Draw nodes as small dots
        ax.scatter(xs, ys, color="tomato", s=20, zorder=3)

        # Mark start and end larger
        if self.nodes:
            ax.scatter(xs[0],  ys[0],  color="green",  s=120,
                       zorder=4, label="start")
            ax.scatter(xs[-1], ys[-1], color="purple", s=120,
                       zorder=4, label="end")
            ax.legend()

        img_path = OUTPUT_FILE.replace(".yaml", "_image.png")
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close()
        rospy.loginfo("  Image saved: '%s'", img_path)

    # -- RViz markers ----------------------------------------------------------

    def publish_markers(self, event=None):
        ma  = MarkerArray()
        now = rospy.Time.now()
        mid = 0

        # Clear previous markers
        d = Marker()
        d.action = Marker.DELETEALL
        d.header.frame_id = FRAME_ID
        d.header.stamp = now
        ma.markers.append(d)

        # Path as continuous LINE_STRIP (fast, single marker for all nodes)
        if len(self.nodes) >= 2:
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = now
            m.ns = "path"; m.id = mid; mid += 1
            m.type = Marker.LINE_STRIP; m.action = Marker.ADD
            m.scale.x = 0.03
            m.color   = ColorRGBA(0.3, 0.6, 1.0, 0.9)   # light blue
            m.points  = [Point(n["x"], n["y"], 0.0) for n in self.nodes]
            ma.markers.append(m)

        # Start node - green sphere
        if self.nodes:
            self._sphere(ma, now, mid, self.nodes[0],
                         ColorRGBA(0.1, 0.9, 0.1, 1.0), 0.12)
            mid += 1

        # End / current node - purple sphere
        if len(self.nodes) > 1:
            self._sphere(ma, now, mid, self.nodes[-1],
                         ColorRGBA(0.7, 0.0, 0.9, 1.0), 0.12)
            mid += 1

        # Node count text (top-left of path)
        if self.nodes:
            t = Marker()
            t.header.frame_id = FRAME_ID
            t.header.stamp = now
            t.ns = "info"; t.id = mid; mid += 1
            t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
            t.pose.position.x = self.nodes[0]["x"]
            t.pose.position.y = self.nodes[0]["y"]
            t.pose.position.z = 0.3
            t.pose.orientation.w = 1.0
            t.scale.z = 0.15
            t.color   = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            t.text    = f"nodes: {len(self.nodes)}"
            ma.markers.append(t)

        self.marker_pub.publish(ma)

    def _sphere(self, ma, now, mid, node, color, size):
        m = Marker()
        m.header.frame_id = FRAME_ID
        m.header.stamp = now
        m.ns = "endpoints"; m.id = mid
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x = node["x"]
        m.pose.position.y = node["y"]
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale = Vector3(size, size, size)
        m.color = color
        ma.markers.append(m)

    # -- Helpers ---------------------------------------------------------------

    def _dist(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        MapBuilder().run()
    except rospy.ROSInterruptException:
        pass