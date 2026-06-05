#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
map_loader.py
-------------
YAML map loading -> NetworkX graph, path computation (Dijkstra),
automatic node classification as junction/endpoint/regular.

Compatible with Python 2.7 / 3.6.9 (ROS Melodic/Noetic).
"""

from __future__ import print_function
import math
import yaml
import networkx as nx


# Node types
NODE_REGULAR  = "regular"   # degree == 2 (intermediate waypoint)
NODE_JUNCTION = "junction"  # degree  > 2 (intersection)
NODE_ENDPOINT = "endpoint"  # degree == 1 (free endpoint)
NODE_ISOLATED = "isolated"  # degree == 0


class MapLoader(object):
    """
    Load a YAML map with the structure:
        frame_id: odom
        nodes: [ {id, x, y}, ... ]
        edges: [ {from, to, length}, ... ]   # DIRECTIONAL edges

    Edges are one-way: from→to only. Dijkstra respects direction.
    Pass bidirectional=True only for testing/debugging with undirected maps.
    """

    def __init__(self, yaml_path, bidirectional=False):
        self.yaml_path = yaml_path
        self.bidirectional = bidirectional
        self.frame_id = "odom"
        self.graph = nx.DiGraph()
        self.node_types = {}        # id -> NODE_*
        self._xy = {}               # id -> (x, y)  — NX 1.x compat (nodes is a method)
        self._roundabout_nodes = set()
        self._load()

    # load ------------------------------------------------------------------
    def _load(self):
        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f)

        self.frame_id = data.get("frame_id", "odom")
        self._roundabout_nodes = set(
            int(n) for n in data.get("roundabout_nodes", []))

        # Nodes
        for n in data.get("nodes", []):
            nid = int(n["id"])
            self.graph.add_node(nid, x=float(n["x"]), y=float(n["y"]))
            self._xy[nid] = (float(n["x"]), float(n["y"]))

        # Edges
        for e in data.get("edges", []):
            a = int(e["from"])
            b = int(e["to"])
            # Use explicit check to avoid eager evaluation of _euclid default
            length = float(e["length"]) if "length" in e else self._euclid(a, b)
            self.graph.add_edge(a, b, length=length, weight=length)
            if self.bidirectional and not self.graph.has_edge(b, a):
                self.graph.add_edge(b, a, length=length, weight=length)

        self._classify_nodes()

    def _euclid(self, a, b):
        ax, ay = self._xy[a]
        bx, by = self._xy[b]
        return math.hypot(ax - bx, ay - by)

    def _classify_nodes(self):
        # For classification we use the underlying undirected graph
        und = self.graph.to_undirected()
        for nid in self.graph.nodes():
            d = und.degree(nid)
            if d == 0:
                self.node_types[nid] = NODE_ISOLATED
            elif d == 1:
                self.node_types[nid] = NODE_ENDPOINT
            elif d == 2:
                self.node_types[nid] = NODE_REGULAR
            else:
                self.node_types[nid] = NODE_JUNCTION

    # API ------------------------------------------------------------------
    def get_path(self, start_id, end_id):
        """
        Dijkstra on weight 'length'. Returns a list of node_ids [start, ..., end].
        Raises networkx.NetworkXNoPath if no path exists.
        """
        start_id = int(start_id)
        end_id   = int(end_id)
        if start_id not in self.graph or end_id not in self.graph:
            raise ValueError("Start or end node not found in map")
        return nx.dijkstra_path(self.graph, start_id, end_id, weight="length")

    def path_length(self, path):
        total = 0.0
        for i in range(len(path) - 1):
            total += self.graph[path[i]][path[i+1]]["length"]
        return total

    def node_xy(self, nid):
        return self._xy[int(nid)]

    def node_type(self, nid):
        return self.node_types.get(int(nid), NODE_ISOLATED)

    def is_junction(self, nid):
        return self.node_type(nid) == NODE_JUNCTION

    def is_roundabout_node(self, nid):
        return int(nid) in self._roundabout_nodes

    def neighbors(self, nid):
        return list(self.graph.successors(int(nid)))

    def all_nodes(self):
        return list(self.graph.nodes())

    def all_edges(self):
        """Returns (from, to, length) tuples for all directed edges."""
        return [(a, b, d["length"]) for a, b, d in self.graph.edges(data=True)]

    def closest_node(self, x, y):
        """Find the node nearest to a Cartesian coordinate (m)."""
        best, best_d = None, float("inf")
        for nid, (nx_, ny_) in self._xy.items():
            d = math.hypot(nx_ - x, ny_ - y)
            if d < best_d:
                best_d, best = d, nid
        return best, best_d

    # info -----------------------------------------------------------------
    def stats(self):
        n_junc = sum(1 for t in self.node_types.values() if t == NODE_JUNCTION)
        n_end  = sum(1 for t in self.node_types.values() if t == NODE_ENDPOINT)
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "junctions": n_junc,
            "endpoints": n_end,
            "roundabout_nodes": len(self._roundabout_nodes),
            "frame_id": self.frame_id,
        }


# main -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "map_clean-edited_smooth.yaml"
    m = MapLoader(path)
    print("Stats:", m.stats())
    juncs = [n for n in m.all_nodes() if m.is_junction(n)]
    print("Junctions:", juncs)
    if len(sys.argv) >= 4:
        s, e = int(sys.argv[2]), int(sys.argv[3])
        p = m.get_path(s, e)
        print("Path %d -> %d (%.2f m): %s" % (s, e, m.path_length(p), p))
