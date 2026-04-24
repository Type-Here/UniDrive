#!/usr/bin/env python3
"""
smooth_map.py -- Smooth a track map YAML graph using per-segment B-spline.

The map is a topological graph with junctions (nodes with degree > 2).
A global B-spline cannot be applied to the whole graph because the node
ordering by id does not match the path order, and junctions create branches.

This script:
    1. Detects junctions and endpoints from the graph topology
    2. Extracts each path segment between junction/endpoint nodes
    3. Applies a B-spline independently on each segment
    4. Reconnects the smoothed segments at their shared junction nodes
    5. Saves the result as YAML and ROS-compatible JSON

Usage:
    python3 smooth_map.py <input.yaml> [options]

Options:
    --out-yaml PATH    Output YAML (default: <input>_smooth.yaml)
    --out-json PATH    Output JSON (default: <input>_smooth.json)
    --smoothing S      B-spline smoothing per segment (default: 0 = exact)
                       Increase slightly (e.g. 1.0, 2.0) if segments are jagged
    --degree K         B-spline degree: 3=cubic (default)
    --preview          Save before/after PNG
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import splev, splprep


# -- Load / save ---------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: Path):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  YAML saved : {path}")


def save_json(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  JSON saved : {path}")


# -- Graph topology ------------------------------------------------------------

def build_adjacency(edges: list) -> dict:
    adj = defaultdict(list)
    for e in edges:
        adj[e['from']].append(e['to'])
        adj[e['to']].append(e['from'])
    return adj


def classify_nodes(adj: dict, all_node_ids: set) -> tuple:
    """
    Returns:
        branch_nodes -- set of junction (degree>2) and endpoint (degree=1) ids
        degree       -- dict {node_id: degree}
    """
    degree = {nid: len(neighbors) for nid, neighbors in adj.items()}
    # Nodes with no edges at all
    for nid in all_node_ids:
        if nid not in degree:
            degree[nid] = 0
    branch_nodes = {nid for nid, d in degree.items() if d != 2}
    return branch_nodes, degree


def extract_segments(adj: dict, branch_nodes: set) -> list:
    """
    Walk the graph and extract all path segments.
    A segment is a list of node ids forming a chain between two branch nodes
    (junctions or endpoints). Road nodes (degree=2) appear in only one segment.

    Returns:
        list of lists of node ids, e.g. [[0, 5, 6, 7, 3], [3, 8, 9, 1], ...]
    """
    visited_edges = set()
    segments = []

    def edge_key(a, b):
        return (min(a, b), max(a, b))

    for start in sorted(branch_nodes):
        for neighbor in adj[start]:
            ek = edge_key(start, neighbor)
            if ek in visited_edges:
                continue
            seg = [start, neighbor]
            visited_edges.add(ek)
            prev, curr = start, neighbor

            # Walk until we hit another branch node
            while curr not in branch_nodes:
                nexts = [n for n in adj[curr] if n != prev]
                if not nexts:
                    break
                nxt = nexts[0]
                ek2 = edge_key(curr, nxt)
                if ek2 in visited_edges:
                    break
                visited_edges.add(ek2)
                seg.append(nxt)
                prev, curr = curr, nxt

            if len(seg) >= 2:
                segments.append(seg)

    return segments


# -- B-spline smoothing per segment -------------------------------------------

def smooth_segment(seg_ids: list, node_map: dict,
                   smoothing: float, degree: int) -> list:
    """
    Apply B-spline smoothing to a single path segment.
    Junction/endpoint nodes at both ends are kept fixed (not smoothed).
    Returns a list of new {id, x, y} dicts for the interior nodes.

    If the segment is too short for the chosen degree, returns it unchanged.
    """
    xs = np.array([node_map[nid]['x'] for nid in seg_ids])
    ys = np.array([node_map[nid]['y'] for nid in seg_ids])
    n  = len(seg_ids)

    # Need at least degree+1 points for splprep
    if n <= degree:
        return [{"id": nid, "x": float(node_map[nid]['x']),
                 "y": float(node_map[nid]['y'])}
                for nid in seg_ids]

    try:
        tck, u = splprep([xs, ys], s=smoothing, k=min(degree, n - 1))
        t_new  = np.linspace(0, 1, n)
        xs_s, ys_s = splev(t_new, tck)
    except Exception as exc:
        print(f"    WARNING: spline failed for segment of {n} nodes: {exc}")
        xs_s, ys_s = xs, ys

    # Force endpoints to stay exactly at junction positions
    xs_s[0],  ys_s[0]  = xs[0],  ys[0]
    xs_s[-1], ys_s[-1] = xs[-1], ys[-1]

    return [{"id": nid, "x": round(float(x), 4), "y": round(float(y), 4)}
            for nid, x, y in zip(seg_ids, xs_s, ys_s)]


# -- Rebuild graph from smoothed nodes ----------------------------------------

def rebuild_graph(smoothed_nodes: dict, original_edges: list,
                  frame_id: str) -> dict:
    """
    Use the original edge topology but update node positions to smoothed values.
    Edge lengths are recomputed from the new positions.
    """
    nodes = [{"id": nid, "x": n["x"], "y": n["y"]}
             for nid, n in sorted(smoothed_nodes.items())]

    edges = []
    for e in original_edges:
        na = smoothed_nodes[e['from']]
        nb = smoothed_nodes[e['to']]
        length = round(math.sqrt(
            (nb['x'] - na['x']) ** 2 +
            (nb['y'] - na['y']) ** 2), 4)
        edges.append({"from": e['from'], "to": e['to'], "length": length})

    return {"frame_id": frame_id, "nodes": nodes, "edges": edges}


# -- ROS JSON format -----------------------------------------------------------

def to_ros_json(nodes: list, edges: list, frame_id: str,
                branch_nodes: set, degree: dict) -> dict:
    """
    ROS-compatible JSON with waypoints, edges and junction metadata.
    Junctions carry their degree so the navigator can handle branching.
    """
    waypoints = []
    for n in nodes:
        wp = {"id": n["id"], "x": n["x"], "y": n["y"]}
        d  = degree.get(n["id"], 0)
        if d > 2:
            wp["type"] = "junction"
        elif d == 1:
            wp["type"] = "endpoint"
        else:
            wp["type"] = "road"
        waypoints.append(wp)

    return {
        "frame_id":  frame_id,
        "waypoints": waypoints,
        "edges":     edges,
        "junctions": sorted(branch_nodes),
    }


# -- Preview -------------------------------------------------------------------

def save_preview(orig_nodes: dict, orig_edges: list,
                 smooth_nodes: dict, smooth_edges: list,
                 branch_nodes: set, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping preview")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#1e1e2e")

    def draw_graph(ax, nodes, edges, title, branch_nodes):
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555577")
        ax.set_xlabel("x [m]", color="white")
        ax.set_ylabel("y [m]", color="white")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, color="#2e2e4e", ls="--", alpha=0.4)

        # Draw edges
        for e in edges:
            na, nb = nodes[e['from']], nodes[e['to']]
            ax.plot([na['x'], nb['x']], [na['y'], nb['y']],
                    color="#4a90d9", lw=1.5, alpha=0.8, zorder=2)

        # Draw nodes
        for nid, n in nodes.items():
            color = "#f5a623" if nid in branch_nodes else "#4a90d9"
            size  = 60        if nid in branch_nodes else 15
            ax.scatter(n['x'], n['y'], color=color, s=size, zorder=4)

    draw_graph(axes[0], orig_nodes, orig_edges,
               f"Original  ({len(orig_nodes)} nodes, {len(orig_edges)} edges)",
               branch_nodes)
    draw_graph(axes[1], smooth_nodes, smooth_edges,
               f"Smoothed  ({len(smooth_nodes)} nodes, {len(smooth_edges)} edges)",
               branch_nodes)

    plt.suptitle("smooth_map.py -- per-segment B-spline", color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Preview    : {out_path}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Smooth track map YAML graph with per-segment B-spline")
    parser.add_argument("input",        help="Input YAML map file")
    parser.add_argument("--out-yaml",   default=None)
    parser.add_argument("--out-json",   default=None)
    parser.add_argument("--smoothing",  type=float, default=0.0,
                        help="B-spline smoothing per segment (default: 0 = exact)")
    parser.add_argument("--degree",     type=int,   default=3,
                        choices=[1, 2, 3, 4, 5])
    parser.add_argument("--preview",    action="store_true")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_yaml = Path(args.out_yaml) if args.out_yaml \
        else in_path.with_name(in_path.stem + "_smooth.yaml")
    out_json = Path(args.out_json) if args.out_json \
        else in_path.with_name(in_path.stem + "_smooth.json")

    data     = load_yaml(in_path)
    nodes    = data["nodes"]
    edges    = data["edges"]
    frame_id = data.get("frame_id", "odom")

    node_map     = {n["id"]: n for n in nodes}
    all_node_ids = set(node_map.keys())

    print(f"\n  Input      : {in_path}")
    print(f"  Nodes      : {len(nodes)}")
    print(f"  Edges      : {len(edges)}")
    print(f"  Smoothing  : {args.smoothing}")
    print(f"  Degree     : {args.degree}")

    # Topology analysis
    adj = build_adjacency(edges)
    branch_nodes, degree = classify_nodes(adj, all_node_ids)
    junctions  = {nid for nid, d in degree.items() if d > 2}
    endpoints  = {nid for nid, d in degree.items() if d == 1}

    print(f"\n  Junctions  (degree>2): {sorted(junctions)}")
    print(f"  Endpoints  (degree=1): {sorted(endpoints)}")

    # Extract path segments
    segments = extract_segments(adj, branch_nodes)
    print(f"  Segments found: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"    Seg {i:2d}: {seg[0]:3d} -> ... -> {seg[-1]:3d}"
              f"  ({len(seg)} nodes)")

    # Smooth each segment independently
    smoothed_nodes = dict(node_map)  # start from original positions

    for seg in segments:
        smoothed = smooth_segment(seg, node_map, args.smoothing, args.degree)
        for nd in smoothed:
            smoothed_nodes[nd["id"]] = nd

    # Rebuild graph with updated positions
    graph = rebuild_graph(smoothed_nodes, edges, frame_id)

    # Save YAML
    save_yaml(graph, out_yaml)

    # Save JSON
    ros_json = to_ros_json(
        graph["nodes"], graph["edges"], frame_id,
        branch_nodes, degree)
    save_json(ros_json, out_json)

    # Preview
    if args.preview:
        preview_path = in_path.with_name(in_path.stem + "_smooth_preview.png")
        orig_node_map   = {n["id"]: n for n in nodes}
        smooth_node_map = {n["id"]: n for n in graph["nodes"]}
        save_preview(orig_node_map, edges,
                     smooth_node_map, graph["edges"],
                     branch_nodes, preview_path)

    print(f"\n  Done.")
    print(f"  Junctions in JSON: {sorted(junctions)}")
    print(f"  These nodes are decision points for the path planner.\n")


if __name__ == "__main__":
    main()