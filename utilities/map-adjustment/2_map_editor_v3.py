#!/usr/bin/env python3
"""
map_editor.py — Interactive Track Graph Editor (optimised)
===========================================================
Uses incremental artist updates instead of full redraws to stay fast
even with hundreds of nodes and edges.

Usage:
    python3 map_editor.py <input.yaml> [output.yaml]

MODES
    M   MOVE mode   — left-drag to move nodes
    A   ADD  mode   — left-click on empty space to add a node

MOUSE
    Left click            select node
    Left click (2nd node) add edge between selected and clicked node
    Left drag             move selected node (MOVE mode)
    Right click           delete node + its edges

KEYBOARD
    M / A        switch mode
    R            remove all edges of selected node
    E            remove edge between selected ↔ prev selected
    Delete       delete selected node + its edges
    Ctrl+Z       undo (100 steps)
    S            save YAML
    Q / Escape   quit
"""

import sys, yaml, math, copy
from collections import defaultdict
import numpy as np
import matplotlib

from config import ROOT_DIR

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.patches as mpatches

# ── Config ─────────────────────────────────────────────────────────────────────
PICK_RADIUS = 0.10   # metres — node selection radius

C_BG   = "#1e1e2e"
C_GRID = "#2e2e4e"
C_TEXT = "white"

# node colours by role
C_ROAD = "#4a90d9"
C_JUNC = "#f5a623"
C_END  = "#e05252"
C_SEL  = "#f5a623"
C_PREV = "#a0d9f5"
C_ADD  = "#27ae60"

# edge colours
C_EDGE     = "#4a90d9"
C_EDGE_SEL = "#f5a623"
# ──────────────────────────────────────────────────────────────────────────────

def dist(ax, ay, bx, by):
    return math.sqrt((ax-bx)**2 + (ay-by)**2)

def edge_key(a, b):
    return (min(a, b), max(a, b))


def renumber(nodes, edges):
    """
    Reassign sequential ids 0..N-1 to nodes and remap edge from/to.
    Must be called after every structural change (add/delete node).
    Drops dangling edges whose endpoints no longer exist.
    """
    id_map = {n["id"]: i for i, n in enumerate(nodes)}
    for i, n in enumerate(nodes):
        n["id"] = i
    new_edges = []
    for e in edges:
        a = id_map.get(e["from"])
        b = id_map.get(e["to"])
        if a is None or b is None:
            continue  # dangling edge — drop silently
        new_edges.append({"from": a, "to": b, "length": e["length"]})
    return nodes, new_edges

def load(path):
    with open(path) as f:
        return yaml.safe_load(f)

def save_yaml(nodes, edges, frame_id, path):
    # Final renumber pass — guarantees consistency regardless of edit history
    nodes, edges = renumber(nodes, edges)
    clean_nodes = [{"id": int(n["id"]), "x": float(n["x"]), "y": float(n["y"])}
                   for n in nodes]
    clean_edges = [{"from": int(e["from"]), "to": int(e["to"]),
                    "length": float(e["length"])} for e in edges]
    with open(path, "w") as f:
        yaml.dump({"frame_id": frame_id, "nodes": clean_nodes, "edges": clean_edges},
                  f, default_flow_style=False)
    print(f"  Saved: {path}  ({len(nodes)} nodes, {len(edges)} edges)")


class MapEditor:

    def __init__(self, input_path, output_path):
        self.input_path  = input_path
        self.output_path = output_path

        data = load(input_path)
        self.nodes    = data["nodes"]
        self.edges    = data.get("edges", [])
        self.frame_id = data.get("frame_id", "odom")
        for i, n in enumerate(self.nodes):
            n["id"] = i

        self.dirty    = False
        self.history  = []
        self.selected = None
        self.prev_sel = None
        self.mode     = "move"
        self.dragging = False
        self.drag_id  = None

        self._build_fig()
        self._full_redraw(fit=True)
        self._connect()
        self._print_banner()

    # ── Figure setup ──────────────────────────────────────────────────────────

    def _build_fig(self):
        self.fig, self.ax = plt.subplots(figsize=(13, 9))
        self.fig.patch.set_facecolor(C_BG)
        self.ax.set_facecolor(C_BG)
        self.ax.set_aspect("equal")
        self.ax.tick_params(colors=C_TEXT)
        for sp in self.ax.spines.values():
            sp.set_edgecolor("#555577")
        self.ax.set_xlabel("x [m]", color=C_TEXT)
        self.ax.set_ylabel("y [m]", color=C_TEXT)
        self.ax.grid(True, color=C_GRID, ls="--", alpha=0.5)

        # Persistent artist handles — updated in place, never recreated
        # Edges: two LineCollections (normal + highlighted)
        self._lc_normal = mc.LineCollection(
            [], colors=C_EDGE, linewidths=1.5, alpha=0.85, zorder=2)
        self._lc_sel    = mc.LineCollection(
            [], colors=C_EDGE_SEL, linewidths=2.5, alpha=1.0, zorder=3)
        self.ax.add_collection(self._lc_normal)
        self.ax.add_collection(self._lc_sel)

        # Nodes: three scatter groups (normal, selected, prev)
        self._sc_normal = self.ax.scatter([], [], s=40, zorder=4,
                                          linewidths=0)
        self._sc_sel    = self.ax.scatter([], [], c=C_SEL, s=200, zorder=6,
                                          linewidths=0)
        self._sc_prev   = self.ax.scatter([], [], c=C_PREV, s=120, zorder=5,
                                          linewidths=0)

        # Status text
        self._status = self.fig.text(
            0.01, 0.01, "", color=C_TEXT, fontsize=7.5,
            verticalalignment="bottom", family="monospace")

        # Label annotations — rebuilt only on structural changes
        self._labels = []

        # Legend (static)
        legend = [
            mpatches.Patch(color=C_JUNC, label="junction (deg>2)"),
            mpatches.Patch(color=C_END,  label="endpoint (deg=1)"),
            mpatches.Patch(color=C_ROAD, label="road node"),
            mpatches.Patch(color=C_SEL,  label="selected"),
            mpatches.Patch(color=C_PREV, label="prev selected"),
        ]
        self.ax.legend(handles=legend, loc="upper right",
                       facecolor="#2a2a3e", labelcolor=C_TEXT, fontsize=7)

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _degree(self):
        deg = defaultdict(int)
        for e in self.edges:
            deg[e["from"]] += 1
            deg[e["to"]]   += 1
        return deg

    def _fit_view(self):
        """
        Set axis limits from node data.
        Must be called manually because LineCollection is invisible to
        ax.relim() / autoscale_view() — known matplotlib limitation.
        """
        if not self.nodes:
            return
        xs = [n["x"] for n in self.nodes]
        ys = [n["y"] for n in self.nodes]
        pad = 0.3
        self.ax.set_xlim(min(xs) - pad, max(xs) + pad)
        self.ax.set_ylim(min(ys) - pad, max(ys) + pad)

    def _full_redraw(self, fit=False):
        """Rebuild all artists. Called after structural changes (add/delete).
        Pass fit=True to also reset the view to fit all nodes."""
        self._update_edges()
        self._update_nodes()
        self._update_labels()
        self._update_title()
        self._update_status()
        if fit:
            self._fit_view()
        self.fig.canvas.draw_idle()

    def _fast_update(self):
        """Update only positions — called during drag for speed."""
        self._update_edges()
        self._update_nodes()
        self._update_status()
        self.fig.canvas.draw_idle()

    def _update_edges(self):
        if not self.nodes:
            self._lc_normal.set_segments([])
            self._lc_sel.set_segments([])
            return

        node_map = {n["id"]: n for n in self.nodes}
        sel_keys = set()
        if self.selected is not None:
            for e in self.edges:
                if e["from"] == self.selected or e["to"] == self.selected:
                    sel_keys.add(edge_key(e["from"], e["to"]))

        segs_normal, segs_sel = [], []
        for e in self.edges:
            na = node_map.get(e["from"])
            nb = node_map.get(e["to"])
            if na is None or nb is None:
                continue
            seg = [[na["x"], na["y"]], [nb["x"], nb["y"]]]
            if edge_key(e["from"], e["to"]) in sel_keys:
                segs_sel.append(seg)
            else:
                segs_normal.append(seg)

        self._lc_normal.set_segments(segs_normal)
        self._lc_sel.set_segments(segs_sel)

    def _update_nodes(self):
        if not self.nodes:
            self._sc_normal.set_offsets(np.empty((0, 2)))
            self._sc_sel.set_offsets(np.empty((0, 2)))
            self._sc_prev.set_offsets(np.empty((0, 2)))
            return

        deg = self._degree()

        normal_xy, normal_c = [], []
        sel_xy,  prev_xy    = [], []

        for n in self.nodes:
            nid = n["id"]
            xy  = [n["x"], n["y"]]
            if nid == self.selected:
                sel_xy.append(xy)
            elif nid == self.prev_sel:
                prev_xy.append(xy)
            else:
                normal_xy.append(xy)
                d = deg[nid]
                if d == 1:    normal_c.append(C_END)
                elif d > 2:   normal_c.append(C_JUNC)
                else:         normal_c.append(C_ROAD)

        def set_sc(sc, xys):
            if xys:
                sc.set_offsets(np.array(xys))
            else:
                sc.set_offsets(np.empty((0, 2)))

        if normal_xy:
            self._sc_normal.set_offsets(np.array(normal_xy))
            self._sc_normal.set_facecolors(normal_c)
            self._sc_normal.set_sizes([40] * len(normal_xy))
        else:
            self._sc_normal.set_offsets(np.empty((0, 2)))
            self._sc_normal.set_facecolors([])
            self._sc_normal.set_sizes([])

        set_sc(self._sc_sel,  sel_xy)
        set_sc(self._sc_prev, prev_xy)

    def _update_labels(self):
        """Rebuild text annotations — only called on structural changes."""
        for ann in self._labels:
            ann.remove()
        self._labels = []
        step = max(1, len(self.nodes) // 40)
        for i, n in enumerate(self.nodes):
            show = (i % step == 0 or
                    n["id"] == self.selected or
                    n["id"] == self.prev_sel)
            if show:
                ann = self.ax.annotate(
                    str(n["id"]), (n["x"], n["y"]),
                    fontsize=6, color=C_TEXT, alpha=0.65,
                    xytext=(4, 4), textcoords="offset points", zorder=7)
                self._labels.append(ann)

    def _update_title(self):
        mode_label = ("MODE: MOVE (M)" if self.mode == "move"
                      else "MODE: ADD NODE (A)")
        mode_color = C_SEL if self.mode == "move" else C_ADD
        self.ax.set_title(mode_label, color=mode_color, fontsize=10, pad=4)

    def _update_status(self):
        sel_s  = f"sel={self.selected}"  if self.selected is not None else "sel=─"
        prev_s = f" prev={self.prev_sel}" if self.prev_sel is not None else ""
        dirty  = " *" if self.dirty else ""
        mode   = "MOVE" if self.mode == "move" else "ADD"
        self._status.set_text(
            f"[{mode}]  {sel_s}{prev_s}  "
            f"nodes={len(self.nodes)}  edges={len(self.edges)}{dirty}  │  "
            f"M=move  A=add  R=rm edges  E=rm edge  Del=rm node  "
            f"click×2=add edge  S=save  Ctrl+Z=undo  Q=quit")

    # ── Event connection ──────────────────────────────────────────────────────

    def _connect(self):
        c = self.fig.canvas
        c.mpl_connect("button_press_event",   self._on_press)
        c.mpl_connect("button_release_event", self._on_release)
        c.mpl_connect("motion_notify_event",  self._on_motion)
        c.mpl_connect("key_press_event",      self._on_key)
        c.mpl_connect("close_event",          self._on_close)

    # ── Nearest node ──────────────────────────────────────────────────────────

    def _nearest(self, xd, yd):
        best_d, best_id = float("inf"), None
        for n in self.nodes:
            d = dist(n["x"], n["y"], xd, yd)
            if d < PICK_RADIUS and d < best_d:
                best_d, best_id = d, n["id"]
        return best_id

    # ── Undo stack ────────────────────────────────────────────────────────────

    def _push(self):
        self.history.append((copy.deepcopy(self.nodes),
                             copy.deepcopy(self.edges)))
        if len(self.history) > 100:
            self.history.pop(0)

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        xd, yd = event.xdata, event.ydata
        hit    = self._nearest(xd, yd)

        # Right click — delete node
        if event.button == 3:
            if hit is not None:
                self._push()
                self._delete_node(hit)
                self.selected = None
                self.prev_sel = None
                self.dirty    = True
                self._full_redraw()
            return

        # Left click
        if event.button != 1:
            return

        if hit is not None:
            if self.selected is None:
                # First selection
                self._push()
                self.selected = hit
                self.dragging = True
                self.drag_id  = hit
                self._fast_update()

            elif self.selected == hit:
                # Click same node → deselect
                self.prev_sel = self.selected
                self.selected = None
                self.dragging = False
                self._fast_update()

            else:
                # Second node → add edge
                self._add_edge(self.selected, hit)
                self.prev_sel = self.selected
                self.selected = hit
                self.dragging = True
                self.drag_id  = hit
                self._full_redraw()

        else:
            # Empty space
            if self.mode == "add":
                self._push()
                self._add_node(xd, yd)
                self.dirty = True
                self._full_redraw(fit=True)
            else:
                self.prev_sel = self.selected
                self.selected = None
                self.dragging = False
                self._fast_update()

    def _on_motion(self, event):
        if not self.dragging or self.drag_id is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        for n in self.nodes:
            if n["id"] == self.drag_id:
                n["x"] = round(float(event.xdata), 4)
                n["y"] = round(float(event.ydata), 4)
                break
        self.dirty = True
        self._fast_update()   # ← no label rebuild during drag

    def _on_release(self, event):
        if self.dragging:
            # Rebuild edge lengths after drag ends
            node_map = {n["id"]: n for n in self.nodes}
            for e in self.edges:
                na = node_map.get(e["from"])
                nb = node_map.get(e["to"])
                if na and nb:
                    e["length"] = round(
                        dist(na["x"], na["y"], nb["x"], nb["y"]), 4)
            self._full_redraw()
        self.dragging = False
        self.drag_id  = None

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def _on_key(self, event):
        k = event.key

        if k == "ctrl+z":
            if not self.history:
                print("  Nothing to undo.")
                return
            self.nodes, self.edges = self.history.pop()
            self.selected = None
            self.dirty    = True
            self._full_redraw()
            print(f"  Undo — {len(self.nodes)} nodes, {len(self.edges)} edges")

        elif k in ("s", "S"):
            save_yaml(self.nodes, self.edges, self.frame_id, self.output_path)
            self.dirty = False
            self._update_status()
            self.fig.canvas.draw_idle()

        elif k in ("m", "M"):
            self.mode = "move"
            self._update_title()
            self._update_status()
            self.fig.canvas.draw_idle()

        elif k in ("a", "A"):
            self.mode = "add"
            self._update_title()
            self._update_status()
            self.fig.canvas.draw_idle()

        elif k in ("r", "R"):
            if self.selected is not None:
                self._push()
                before = len(self.edges)
                self.edges = [e for e in self.edges
                              if e["from"] != self.selected
                              and e["to"]   != self.selected]
                print(f"  Removed {before - len(self.edges)} edges "
                      f"from node {self.selected}")
                self.dirty = True
                self._full_redraw()

        elif k in ("e", "E"):
            if self.selected is not None and self.prev_sel is not None:
                self._push()
                ek = edge_key(self.selected, self.prev_sel)
                before = len(self.edges)
                self.edges = [e for e in self.edges
                              if edge_key(e["from"], e["to"]) != ek]
                print(f"  Removed {before - len(self.edges)} edge(s) "
                      f"between {self.selected} ↔ {self.prev_sel}")
                self.dirty = True
                self._full_redraw()

        elif k in ("delete", "backspace"):
            if self.selected is not None:
                self._push()
                self._delete_node(self.selected)
                self.selected = None
                self.prev_sel = None
                self.dirty    = True
                self._full_redraw()

        elif k in ("q", "Q", "escape"):
            self._quit()

    # ── Graph ops ─────────────────────────────────────────────────────────────

    def _add_node(self, x, y):
        nid = max((n["id"] for n in self.nodes), default=-1) + 1
        self.nodes.append({"id": nid, "x": round(float(x), 4), "y": round(float(y), 4)})
        self.nodes, self.edges = renumber(self.nodes, self.edges)
        # After renumber the new node gets the last sequential id
        self.selected = self.nodes[-1]["id"]
        print(f"  + Node {self.selected} at ({x:.3f}, {y:.3f})")

    def _delete_node(self, nid):
        self.nodes = [n for n in self.nodes if n["id"] != nid]
        self.edges = [e for e in self.edges
                      if e["from"] != nid and e["to"] != nid]
        self.nodes, self.edges = renumber(self.nodes, self.edges)
        # Selection ids may have shifted — clear to avoid pointing to wrong node
        self.selected = None
        self.prev_sel = None
        print(f"  - Deleted node {nid}")

    def _add_edge(self, a, b):
        if a == b:
            return
        ek = edge_key(a, b)
        if any(edge_key(e["from"], e["to"]) == ek for e in self.edges):
            print(f"  Edge {a}↔{b} already exists")
            return
        self._push()
        nm = {n["id"]: n for n in self.nodes}
        na, nb = nm[a], nm[b]
        length = round(dist(na["x"], na["y"], nb["x"], nb["y"]), 4)
        self.edges.append({"from": a, "to": b, "length": length})
        self.dirty = True
        print(f"  + Edge {a} ↔ {b}  ({length:.3f} m)")

    # ── Save / quit ───────────────────────────────────────────────────────────

    def _quit(self):
        if self.dirty:
            print("  Unsaved changes. Save? [y/n]")
            try:
                ans = input("  > ").strip().lower()
            except EOFError:
                ans = "n"
            if ans == "y":
                save_yaml(self.nodes, self.edges,
                          self.frame_id, self.output_path)
        plt.close(self.fig)

    def _on_close(self, event):
        if self.dirty:
            print("  Auto-saving on close...")
            save_yaml(self.nodes, self.edges, self.frame_id, self.output_path)

    def _print_banner(self):
        print(f"\n  MAP EDITOR  —  {len(self.nodes)} nodes, "
              f"{len(self.edges)} edges")
        print("  M=move  A=add node  R=rm edges  E=rm edge  "
              "Del=rm node  click×2=add edge  S=save  Ctrl+Z  Q=quit\n")

    def run(self):
        plt.show()


def main():
    print("Usage: python3 map_editor.py [input.yaml] [output.yaml]")

    default_input_path = ROOT_DIR + "/artifacts/map/map_clean-edited.yaml"
    default_output_path = ROOT_DIR + "/artifacts/map/map_clean-edited.yaml"
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input_path
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output_path

    editor = MapEditor(input_path, output_path)
    editor.run()


if __name__ == "__main__":
    main()