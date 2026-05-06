#!/usr/bin/env python
"""
map_builder.py - ROS1 Minimal RViz Graph Builder
=================================================
Drive the robot manually with joystick_controller.launch (already running).
Use RViz "Publish Point" tool to place nodes by clicking on the map.
Connect nodes via terminal commands to build the track graph.

Subscribed topics:
    /clicked_point  (geometry_msgs/PointStamped) - RViz "Publish Point" clicks
    /odom           (nav_msgs/Odometry)          - path trail (optional, visual only)

Published topics:
    /map_markers    (visualization_msgs/MarkerArray) - graph visualization in RViz

Terminal commands:
    list              - print all nodes
    edge A B          - connect node A to node B
    label N name      - rename node N
    type N typename   - set type: intersection | curve | straight | start | end
    undo              - remove last node
    save              - save graph to YAML + PNG
    quit              - exit cleanly

How to use in RViz:
    1. Add display: MarkerArray -> topic /map_markers
    2. Select "Publish Point" tool in the toolbar
    3. Click on the map where you want to place a node
    4. Use terminal commands to connect nodes with edges
"""

import rospy
import yaml
import math
import threading

from geometry_msgs.msg import PointStamped, Point, Vector3
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

# -- Configuration --------------------------------------------------------------

OUTPUT_FILE     = "/home/jetauto/track_map.yaml"
FRAME_ID        = "map"           # must match RViz Fixed Frame
ODOM_TOPIC      = "/odom"
CLICKED_TOPIC   = "/clicked_point"
MARKER_TOPIC    = "/map_markers"
PATH_MIN_DIST   = 0.05            # metres between path samples
NODE_MERGE_DIST = 0.15            # metres - snap radius to existing node
NODE_TYPES      = ["intersection", "curve", "straight", "start", "end"]

# ------------------------------------------------------------------------------


class MapBuilder:

    def __init__(self):
        rospy.init_node("map_builder", anonymous=False)

        self.nodes       = []
        self.edges       = []
        self.path_pts    = []
        self.last_pt     = None
        self.node_id_cnt = 0
        self.running     = True

        # Publishers
        self.marker_pub = rospy.Publisher(
            MARKER_TOPIC, MarkerArray, queue_size=1, latch=True)

        # Subscribers
        rospy.Subscriber(CLICKED_TOPIC, PointStamped, self.clicked_cb)
        rospy.Subscriber(ODOM_TOPIC,    Odometry,     self.odom_cb)

        # Marker refresh timer
        rospy.Timer(rospy.Duration(0.5), self.publish_markers)

        # Terminal input in background thread
        threading.Thread(target=self.terminal_input, daemon=True).start()

        self._print_banner()

    # -- Banner ----------------------------------------------------------------

    def _print_banner(self):
        rospy.loginfo("=" * 55)
        rospy.loginfo("  MAP BUILDER - RViz click-to-place graph editor")
        rospy.loginfo("  Fixed Frame  : %s  (set this in RViz!)", FRAME_ID)
        rospy.loginfo("  MarkerArray  : %s", MARKER_TOPIC)
        rospy.loginfo("-" * 55)
        rospy.loginfo("  IN RViz:")
        rospy.loginfo("    1. Add -> MarkerArray -> /map_markers")
        rospy.loginfo("    2. Click 'Publish Point' in toolbar")
        rospy.loginfo("    3. Click on the map to place a node")
        rospy.loginfo("-" * 55)
        rospy.loginfo("  TERMINAL COMMANDS:")
        rospy.loginfo("    list | edge A B | label N name | type N typename")
        rospy.loginfo("    undo | save | quit")
        rospy.loginfo("=" * 55)

    # -- Odometry - records path trail for visual reference --------------------

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.last_pt is None or \
                self._dist(x, y, *self.last_pt) >= PATH_MIN_DIST:
            self.path_pts.append((x, y))
            self.last_pt = (x, y)

    # -- RViz "Publish Point" click --------------------------------------------

    def clicked_cb(self, msg):
        x = round(msg.point.x, 3)
        y = round(msg.point.y, 3)

        # Snap to nearby node instead of creating duplicate
        close = self._find_close_node(x, y)
        if close is not None:
            rospy.loginfo("  Snapped to existing node id=%d  (%.2f, %.2f)",
                          close["id"], close["x"], close["y"])
            return

        node = {
            "id":    self.node_id_cnt,
            "x":     x,
            "y":     y,
            "type":  "intersection",
            "label": f"node_{self.node_id_cnt}"
        }
        self.nodes.append(node)
        rospy.loginfo("  + Node %d placed at (%.2f, %.2f)  [type: %s]",
                      self.node_id_cnt, x, y, node["type"])
        rospy.loginfo("    -> use 'type %d <typename>' or 'label %d <name>' to customise",
                      self.node_id_cnt, self.node_id_cnt)
        self.node_id_cnt += 1

    # -- Terminal commands -----------------------------------------------------

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
                    print("  No nodes yet - click 'Publish Point' in RViz.")
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
                print("  Commands: list | edge A B | label N name |"
                      " type N typename | undo | save | quit")
            else:
                print(f"  Unknown command '{line}' - type 'help'")

    # -- Graph operations ------------------------------------------------------

    def _add_edge(self, id_a, id_b):
        na = next((n for n in self.nodes if n["id"] == id_a), None)
        nb = next((n for n in self.nodes if n["id"] == id_b), None)
        if na is None or nb is None:
            print(f"  Node {id_a} or {id_b} not found - use 'list' to see nodes.")
            return
        if any({e["from"], e["to"]} == {id_a, id_b} for e in self.edges):
            print(f"  Edge {id_a}<->{id_b} already exists.")
            return
        length = round(self._dist(na["x"], na["y"], nb["x"], nb["y"]), 3)
        self.edges.append({"from": id_a, "to": id_b, "length": length})
        print(f"  Edge {id_a} <-> {id_b}  (length: {length:.2f} m)")

    def _set_label(self, nid, label):
        for n in self.nodes:
            if n["id"] == nid:
                n["label"] = label
                print(f"  Node {nid} -> '{label}'")
                return
        print(f"  Node {nid} not found.")

    def _set_type(self, nid, typ):
        if typ not in NODE_TYPES:
            print(f"  Invalid type. Choose: {NODE_TYPES}")
            return
        for n in self.nodes:
            if n["id"] == nid:
                n["type"] = typ
                print(f"  Node {nid} type -> '{typ}'")
                return
        print(f"  Node {nid} not found.")

    def _undo(self):
        if not self.nodes:
            print("  Nothing to undo.")
            return
        removed = self.nodes.pop()
        self.edges = [e for e in self.edges
                      if e["from"] != removed["id"] and e["to"] != removed["id"]]
        print(f"  Removed node {removed['id']}  ({removed['x']}, {removed['y']})")

    # -- Save ------------------------------------------------------------------

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
            rospy.logwarn("  matplotlib not found - skipping PNG.")

    def _save_image(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect("equal")
        ax.set_title("Track Map")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.grid(True, linestyle="--", alpha=0.4)

        # Odometry trail
        if self.path_pts:
            xs, ys = zip(*self.path_pts)
            ax.plot(xs, ys, color="lightgray", linewidth=1, label="path")

        # Edges
        for e in self.edges:
            na = next(n for n in self.nodes if n["id"] == e["from"])
            nb = next(n for n in self.nodes if n["id"] == e["to"])
            ax.plot([na["x"], nb["x"]], [na["y"], nb["y"]], "b-", linewidth=2)
            # Length label on edge midpoint
            mx, my = (na["x"]+nb["x"])/2, (na["y"]+nb["y"])/2
            ax.annotate(f"{e['length']}m", (mx, my), fontsize=7,
                        color="blue", ha="center")

        # Nodes
        colors = {"intersection": "red",  "curve":    "orange",
                  "straight":    "green", "start":    "blue", "end": "purple"}
        for n in self.nodes:
            c = colors.get(n["type"], "gray")
            ax.scatter(n["x"], n["y"], color=c, s=150, zorder=5)
            ax.annotate(f"  {n['id']}: {n['label']}", (n["x"], n["y"]),
                        fontsize=8, color=c)

        img_path = OUTPUT_FILE.replace(".yaml", "_image.png")
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close()
        rospy.loginfo("  Image saved: '%s'", img_path)

    # -- RViz markers ----------------------------------------------------------

    def publish_markers(self, event=None):
        ma  = MarkerArray()
        now = rospy.Time.now()
        mid = 0

        # Clear old markers
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
            m.color   = ColorRGBA(0.5, 0.5, 0.5, 0.6)
            m.points  = [Point(p[0], p[1], 0.0) for p in self.path_pts]
            ma.markers.append(m)

        # Edges - blue LINE_LIST
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
            m.scale.x = 0.05
            m.color   = ColorRGBA(0.2, 0.4, 1.0, 1.0)
            m.points  = [Point(na["x"], na["y"], 0.02),
                         Point(nb["x"], nb["y"], 0.02)]
            ma.markers.append(m)

        # Nodes - coloured sphere + white label
        type_color = {
            "intersection": ColorRGBA(1.0, 0.2, 0.2, 1.0),   # red
            "curve":        ColorRGBA(1.0, 0.6, 0.0, 1.0),   # orange
            "straight":     ColorRGBA(0.2, 0.8, 0.2, 1.0),   # green
            "start":        ColorRGBA(0.0, 0.4, 1.0, 1.0),   # blue
            "end":          ColorRGBA(0.7, 0.0, 0.9, 1.0),   # purple
        }
        for n in self.nodes:
            color = type_color.get(n["type"], ColorRGBA(0.5, 0.5, 0.5, 1.0))

            # Sphere
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = now
            m.ns = "nodes"; m.id = mid; mid += 1
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = n["x"]
            m.pose.position.y = n["y"]
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale = Vector3(0.15, 0.15, 0.15)
            m.color = color
            ma.markers.append(m)

            # Text label above sphere
            t = Marker()
            t.header.frame_id = FRAME_ID
            t.header.stamp = now
            t.ns = "labels"; t.id = mid; mid += 1
            t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
            t.pose.position.x = n["x"]
            t.pose.position.y = n["y"]
            t.pose.position.z = 0.25
            t.pose.orientation.w = 1.0
            t.scale.z = 0.15
            t.color   = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            t.text    = f"{n['id']}: {n['label']}"
            ma.markers.append(t)

        self.marker_pub.publish(ma)

    # -- Helpers ---------------------------------------------------------------

    def _dist(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _find_close_node(self, x, y):
        for n in self.nodes:
            if self._dist(x, y, n["x"], n["y"]) < NODE_MERGE_DIST:
                return n
        return None

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        MapBuilder().run()
    except rospy.ROSInterruptException:
        pass