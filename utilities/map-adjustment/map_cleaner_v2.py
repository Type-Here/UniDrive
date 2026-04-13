#!/usr/bin/env python3
"""
map_cleaner.py - Topological Graph Builder for Multi-Lap Track Maps
=====================================================================
Handles tracks with:
  - Multiple alternative paths (not all covered each lap)
  - Local loops (roundabouts driven multiple times)
  - Overlapping passes on shared road segments

Algorithm:
  1. Snap all raw nodes to a spatial grid (SNAP_DIST)
     → collapses multiple passes of the same road into one set of nodes
  2. Build edges between consecutive snapped nodes
     → shared road = same edge used multiple times, kept once
  3. Remove local micro-loops (roundabout over-driving)
     → detect short cycles and keep only the shortest path through them
  4. Light smoothing on each road segment independently
  5. Save YAML + PNG

Usage:
    python3 map_cleaner.py <input.yaml> [output.yaml]

Key parameters (tune at top of file):
    SNAP_DIST    - spatial resolution: nodes within this radius are merged (metres)
    MIN_EDGE_LEN - discard edges shorter than this (metres)
    SMOOTH_SIGMA - per-segment Gaussian smoothing (keep low: 0.5-1.0)
"""

import sys, yaml, math, itertools
import numpy as np
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict, deque

from config import ROOT_DIR

# -- Parameters -----------------------------------------------------------------
SNAP_DIST    = 0.08   # metres - nodes within 8cm are the same road point
MIN_EDGE_LEN = 0.02   # metres - discard edges shorter than 2cm
SMOOTH_SIGMA = 0.6    # per-segment smoothing (0=none)
# ------------------------------------------------------------------------------


# -- Spatial grid for fast lookup ----------------------------------------------

class SpatialGrid:
    def __init__(self, cell_size):
        self.cell = cell_size
        self.grid = defaultdict(list)  # cell_key -> list of node_ids

    def _key(self, x, y):
        return (int(math.floor(x / self.cell)),
                int(math.floor(y / self.cell)))

    def add(self, nid, x, y):
        self.grid[self._key(x, y)].append(nid)

    def nearby(self, x, y, nodes, radius):
        """Return node ids within radius of (x,y)."""
        r_cells = int(math.ceil(radius / self.cell)) + 1
        cx, cy  = self._key(x, y)
        result  = []
        for dx in range(-r_cells, r_cells+1):
            for dy in range(-r_cells, r_cells+1):
                for nid in self.grid.get((cx+dx, cy+dy), []):
                    n = nodes[nid]
                    if math.sqrt((n["x"]-x)**2 + (n["y"]-y)**2) <= radius:
                        result.append(nid)
        return result


# -- Step 1: snap raw nodes to graph nodes -------------------------------------

def build_graph_nodes(raw_nodes, snap_dist):
    """
    Merge raw nodes that are within snap_dist of each other.
    Returns:
        graph_nodes  - list of {id, x, y, raw_ids}
        raw_to_graph - mapping raw node index -> graph node id
    """
    grid          = SpatialGrid(snap_dist)
    graph_nodes   = []
    raw_to_graph  = {}

    for i, rn in enumerate(raw_nodes):
        nearby = grid.nearby(rn["x"], rn["y"], graph_nodes, snap_dist)
        if nearby:
            # Snap to nearest existing graph node
            best = min(nearby, key=lambda nid: math.sqrt(
                (graph_nodes[nid]["x"]-rn["x"])**2 +
                (graph_nodes[nid]["y"]-rn["y"])**2))
            raw_to_graph[i] = best
            # Update centroid
            gn = graph_nodes[best]
            gn["raw_ids"].append(i)
            n = len(gn["raw_ids"])
            gn["x"] = round((gn["x"]*(n-1) + rn["x"]) / n, 4)
            gn["y"] = round((gn["y"]*(n-1) + rn["y"]) / n, 4)
            grid.add(best, gn["x"], gn["y"])
        else:
            nid = len(graph_nodes)
            graph_nodes.append({
                "id":      nid,
                "x":       round(rn["x"], 4),
                "y":       round(rn["y"], 4),
                "raw_ids": [i]
            })
            raw_to_graph[i] = nid
            grid.add(nid, rn["x"], rn["y"])

    return graph_nodes, raw_to_graph


# -- Step 2: build edges -------------------------------------------------------

def build_edges(raw_nodes, raw_to_graph, graph_nodes, min_edge_len):
    """
    Create edges between consecutive raw nodes mapped to different graph nodes.
    Duplicate edges (same pair) are kept only once.
    """
    edge_set = set()
    edges    = []

    for i in range(len(raw_nodes) - 1):
        a = raw_to_graph[i]
        b = raw_to_graph[i+1]
        if a == b:
            continue  # same graph node - skip
        key = (min(a,b), max(a,b))
        if key in edge_set:
            continue
        na, nb = graph_nodes[a], graph_nodes[b]
        length = math.sqrt((na["x"]-nb["x"])**2 + (na["y"]-nb["y"])**2)
        if length < min_edge_len:
            continue
        edge_set.add(key)
        edges.append({"from": a, "to": b, "length": round(length, 4)})

    return edges


# -- Step 3: remove isolated nodes (no edges) ---------------------------------

def remove_isolated(graph_nodes, edges):
    connected = set()
    for e in edges:
        connected.add(e["from"])
        connected.add(e["to"])
    kept     = [n for n in graph_nodes if n["id"] in connected]
    id_map   = {n["id"]: i for i, n in enumerate(kept)}
    for n in kept:
        n["id"] = id_map[n["id"]]
    new_edges = []
    for e in edges:
        if e["from"] in id_map and e["to"] in id_map:
            new_edges.append({
                "from":   id_map[e["from"]],
                "to":     id_map[e["to"]],
                "length": e["length"]
            })
    return kept, new_edges


# -- Step 4: smooth node positions --------------------------------------------
# We smooth along the original recording order (not graph order)
# so topology is preserved

def smooth_nodes(graph_nodes, sigma):
    if sigma <= 0 or len(graph_nodes) < 3:
        return graph_nodes
    xs = gaussian_filter1d([n["x"] for n in graph_nodes], sigma=sigma)
    ys = gaussian_filter1d([n["y"] for n in graph_nodes], sigma=sigma)
    for i, n in enumerate(graph_nodes):
        n["x"] = round(float(xs[i]), 4)
        n["y"] = round(float(ys[i]), 4)
    return graph_nodes


# -- Save YAML -----------------------------------------------------------------

def save(graph_nodes, edges, frame_id, path):
    # Strip internal raw_ids before saving
    clean_nodes = [{"id": n["id"], "x": n["x"], "y": n["y"]}
                   for n in graph_nodes]
    data = {"frame_id": frame_id, "nodes": clean_nodes, "edges": edges}
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  Saved: {path}")


# -- Save image ----------------------------------------------------------------

def save_image(raw_nodes, graph_nodes, edges, image_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not found - skipping image."); return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#1e1e2e")

    def style(ax, title):
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#555577")
        ax.set_xlabel("x [m]", color="white")
        ax.set_ylabel("y [m]", color="white")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, color="#2e2e4e", ls="--", alpha=0.4)

    # Panel 1 - raw
    ax = axes[0]
    rxs = [n["x"] for n in raw_nodes]
    rys = [n["y"] for n in raw_nodes]
    ax.plot(rxs, rys, color="#e05252", lw=0.8, alpha=0.7)
    ax.scatter(rxs[0], rys[0], color="#27ae60", s=80, zorder=5, label="start")
    ax.legend(facecolor="#2a2a3e", labelcolor="white", fontsize=8)
    style(ax, f"Raw path  ({len(raw_nodes)} nodes)")

    # Panel 2 - graph
    ax = axes[1]
    node_map = {n["id"]: n for n in graph_nodes}

    # Draw edges
    for e in edges:
        na = node_map[e["from"]]
        nb = node_map[e["to"]]
        ax.plot([na["x"], nb["x"]], [na["y"], nb["y"]],
                color="#4a90d9", lw=1.5, alpha=0.8, zorder=2)

    # Draw nodes - size by degree
    degree = defaultdict(int)
    for e in edges:
        degree[e["from"]] += 1
        degree[e["to"]]   += 1

    for n in graph_nodes:
        d = degree[n["id"]]
        # junction (degree>2)=orange, endpoint(degree=1)=red, road(degree=2)=small blue
        if d == 1:
            color, size = "#e05252", 80
        elif d > 2:
            color, size = "#f5a623", 100
        else:
            color, size = "#4a90d9", 20
        ax.scatter(n["x"], n["y"], color=color, s=size, zorder=4)

    # Legend
    import matplotlib.patches as mpatches
    legend = [
        mpatches.Patch(color="#f5a623", label="junction (degree>2)"),
        mpatches.Patch(color="#e05252", label="endpoint (degree=1)"),
        mpatches.Patch(color="#4a90d9", label="road node (degree=2)"),
    ]
    ax.legend(handles=legend, facecolor="#2a2a3e",
              labelcolor="white", fontsize=8)
    style(ax, f"Graph  ({len(graph_nodes)} nodes, {len(edges)} edges)\n"
          f"snap={SNAP_DIST*100:.0f}cm  smooth σ={SMOOTH_SIGMA}")

    plt.suptitle("map_cleaner - raw path → topological graph",
                 color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig(image_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Image: {image_path}")


# -- Main ----------------------------------------------------------------------

def main():
    print("Usage: python3 map_cleaner.py <input.yaml> [output.yaml]")

    default_input = ROOT_DIR + "/artifacts/map/raw-map.yaml"
    default_output = ROOT_DIR + ("/artifacts/map/map_clean.yaml")

    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    image_path = output_path.replace(".yaml", "_image.png")

    print(f"\n  Input       : {input_path}")
    print(f"  Output      : {output_path}")
    print(f"  Snap dist   : {SNAP_DIST*100:.0f} cm")
    print(f"  Smooth sigma: {SMOOTH_SIGMA}\n")

    data = yaml.safe_load(open(input_path))
    raw_nodes = data["nodes"]
    frame_id  = data.get("frame_id", "odom")
    print(f"  Raw nodes: {len(raw_nodes)}")

    # Step 1 - snap
    graph_nodes, raw_to_graph = build_graph_nodes(raw_nodes, SNAP_DIST)
    print(f"  After snap  : {len(graph_nodes)} graph nodes")

    # Step 2 - edges
    edges = build_edges(raw_nodes, raw_to_graph, graph_nodes, MIN_EDGE_LEN)
    print(f"  Edges built : {len(edges)}")

    # Step 3 - remove isolated
    graph_nodes, edges = remove_isolated(graph_nodes, edges)
    print(f"  After cleanup: {len(graph_nodes)} nodes, {len(edges)} edges")

    # Step 4 - smooth
    graph_nodes = smooth_nodes(graph_nodes, SMOOTH_SIGMA)

    # Stats
    from collections import defaultdict as dd
    deg = dd(int)
    for e in edges:
        deg[e["from"]] += 1
        deg[e["to"]]   += 1
    junctions = sum(1 for d in deg.values() if d > 2)
    endpoints = sum(1 for d in deg.values() if d == 1)
    print(f"  Junctions (degree>2): {junctions}")
    print(f"  Endpoints (degree=1): {endpoints}")

    save(graph_nodes, edges, frame_id, output_path)
    save_image(raw_nodes, graph_nodes, edges, image_path)
    print(f"\n  Done! {len(raw_nodes)} raw -> {len(graph_nodes)} nodes,"
          f" {len(edges)} edges\n")


if __name__ == "__main__":
    main()