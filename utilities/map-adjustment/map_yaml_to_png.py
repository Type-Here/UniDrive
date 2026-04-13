#!/usr/bin/env python3
"""
map_plot.py — Save track map YAML as PNG
Usage:
    python3 map_plot.py <input.yaml> [output.png]
"""

import sys, yaml, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 map_plot.py <input.yaml> [output.png]")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 \
        else input_path.replace(".yaml", "_image.png")

    with open(input_path) as f:
        data = yaml.safe_load(f)

    nodes = data["nodes"]
    edges = data.get("edges", [])
    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#1e1e2e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")
    ax.set_xlabel("x [m]", color="white")
    ax.set_ylabel("y [m]", color="white")
    ax.set_aspect("equal")
    ax.grid(True, color="#2e2e4e", linestyle="--", alpha=0.5)

    # Edges
    for e in edges:
        na = nodes[e["from"]]
        nb = nodes[e["to"]]
        ax.plot([na["x"], nb["x"]], [na["y"], nb["y"]],
                color="#4a90d9", lw=1.5, alpha=0.7, zorder=2)

    # Nodes
    ax.scatter(xs, ys, color="#e05252", s=25, zorder=4)

    # Start / end
    ax.scatter(xs[0],  ys[0],  color="#27ae60", s=120,
               zorder=5, label="start")
    ax.scatter(xs[-1], ys[-1], color="#8e44ad", s=120,
               zorder=5, label="end")

    # Labels every ~30 nodes
    step = max(1, len(nodes) // 30)
    for i in range(0, len(nodes), step):
        ax.annotate(str(i), (xs[i], ys[i]),
                    fontsize=6, color="white", alpha=0.7,
                    xytext=(4, 4), textcoords="offset points", zorder=6)

    ax.legend(facecolor="#2a2a3e", labelcolor="white", fontsize=9)
    ax.set_title(
        f"{input_path}  —  {len(nodes)} nodes  {len(edges)} edges\n"
        f"x: [{min(xs):.2f}, {max(xs):.2f}]  "
        f"y: [{min(ys):.2f}, {max(ys):.2f}]",
        color="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {output_path}  ({len(nodes)} nodes)")

if __name__ == "__main__":
    main()