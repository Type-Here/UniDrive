#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
map_loader.py
-------------
Caricamento mappa YAML -> grafo NetworkX, calcolo percorsi (Dijkstra),
classificazione automatica di nodi come junction/endpoint/regular.

Compatibile Python 2.7 / 3.x (ROS Melodic/Noetic).
"""

from __future__ import print_function
import math
import yaml
import networkx as nx


# Tipi di nodo
NODE_REGULAR  = "regular"   # degree == 2 (punto di passaggio)
NODE_JUNCTION = "junction"  # degree  > 2 (incrocio)
NODE_ENDPOINT = "endpoint"  # degree == 1 (estremità libera)
NODE_ISOLATED = "isolated"  # degree == 0


class MapLoader(object):
    """
    Carica una mappa YAML con la struttura:
        frame_id: odom
        nodes: [ {id, x, y}, ... ]
        edges: [ {from, to, length}, ... ]   # archi DIREZIONALI

    NB: nello YAML fornito molti archi sono presenti SOLO in una direzione,
    ma in pratica il robot può percorrerli in entrambe. Il flag `bidirectional`
    duplica gli archi mancanti per consentire la pianificazione completa.
    """

    def __init__(self, yaml_path, bidirectional=True):
        self.yaml_path = yaml_path
        self.bidirectional = bidirectional
        self.frame_id = "odom"
        self.graph = nx.DiGraph()
        self.node_types = {}        # id -> NODE_*
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self):
        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f)

        self.frame_id = data.get("frame_id", "odom")

        # Nodi
        for n in data.get("nodes", []):
            nid = int(n["id"])
            self.graph.add_node(nid, x=float(n["x"]), y=float(n["y"]))

        # Archi
        for e in data.get("edges", []):
            a = int(e["from"])
            b = int(e["to"])
            length = float(e.get("length", self._euclid(a, b)))
            self.graph.add_edge(a, b, length=length, weight=length)
            if self.bidirectional and not self.graph.has_edge(b, a):
                self.graph.add_edge(b, a, length=length, weight=length)

        self._classify_nodes()

    def _euclid(self, a, b):
        ax, ay = self.graph.nodes[a]["x"], self.graph.nodes[a]["y"]
        bx, by = self.graph.nodes[b]["x"], self.graph.nodes[b]["y"]
        return math.hypot(ax - bx, ay - by)

    def _classify_nodes(self):
        # Per la classificazione usiamo il grafo non orientato sottostante
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

    # ------------------------------------------------------------------ API
    def get_path(self, start_id, end_id):
        """
        Dijkstra sul peso 'length'. Ritorna lista di node_id [start, ..., end].
        Lancia networkx.NetworkXNoPath se non c'è soluzione.
        """
        start_id = int(start_id)
        end_id   = int(end_id)
        if start_id not in self.graph or end_id not in self.graph:
            raise ValueError("Nodo start o end non presente in mappa")
        return nx.dijkstra_path(self.graph, start_id, end_id, weight="length")

    def path_length(self, path):
        total = 0.0
        for i in range(len(path) - 1):
            total += self.graph[path[i]][path[i+1]]["length"]
        return total

    def node_xy(self, nid):
        n = self.graph.nodes[int(nid)]
        return (n["x"], n["y"])

    def node_type(self, nid):
        return self.node_types.get(int(nid), NODE_ISOLATED)

    def is_junction(self, nid):
        return self.node_type(nid) == NODE_JUNCTION

    def neighbors(self, nid):
        return list(self.graph.successors(int(nid)))

    def all_nodes(self):
        return list(self.graph.nodes())

    def all_edges(self):
        # Per visualizzazione: ritorna tuple (a,b,length) di edge unici
        seen = set()
        out = []
        for a, b, d in self.graph.edges(data=True):
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            out.append((a, b, d["length"]))
        return out

    def closest_node(self, x, y):
        """Trova il nodo più vicino a una coordinata cartesiana (m)."""
        best, best_d = None, float("inf")
        for nid, attr in self.graph.nodes(data=True):
            d = math.hypot(attr["x"] - x, attr["y"] - y)
            if d < best_d:
                best_d, best = d, nid
        return best, best_d

    # ----------------------------------------------------------------- info
    def stats(self):
        n_junc = sum(1 for t in self.node_types.values() if t == NODE_JUNCTION)
        n_end  = sum(1 for t in self.node_types.values() if t == NODE_ENDPOINT)
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "junctions": n_junc,
            "endpoints": n_end,
            "frame_id": self.frame_id,
        }


# ----------------------------------------------------------------------- main
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
