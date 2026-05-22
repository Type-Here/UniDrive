#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
serve_dashboard.py
------------------
Mini server HTTP per la dashboard web. Standalone, NON è un nodo ROS.

Espone:
  /                -> dashboard.html
  /dashboard.html  -> dashboard.html
  /map.yaml        -> file mappa configurato

Argomenti CLI:
  --port    porta HTTP (default 8000)
  --web-dir directory dei file web (default ./web rispetto allo script)
  --map     path al file YAML della mappa

Esempio:
  python2 serve_dashboard.py \
        --port 8000 \
        --web-dir /home/jetauto/jetauto_autonomous/web \
        --map /home/jetauto/jetauto_autonomous/maps/map_clean-edited_smooth.yaml

Compatibile Python 2.7 / Python 3.x.
"""

from __future__ import print_function
import argparse
import json
import os
import sys
import socket

import yaml

try:
    # Python 3
    from http.server import SimpleHTTPRequestHandler, HTTPServer
except ImportError:
    # Python 2
    from SimpleHTTPServer import SimpleHTTPRequestHandler
    from BaseHTTPServer import HTTPServer


def get_local_ip():
    """Rileva l'IP locale usato per raggiungere la rete (stesso che usa ROS)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


class DashboardHandler(SimpleHTTPRequestHandler):
    web_dir = "."
    map_file = ""
    video_server_ip = "localhost"
    _cached_html = None  # dashboard.html pre-elaborato in memoria all'avvio

    @classmethod
    def preload(cls):
        """Legge dashboard.html una volta e sostituisce il placeholder IP."""
        fpath = os.path.join(cls.web_dir, "dashboard.html")
        with open(fpath, "rb") as f:
            content = f.read().decode("utf-8")
        content = content.replace("__VIDEO_SERVER_IP__", cls.video_server_ip)
        cls._cached_html = content.encode("utf-8")
        sys.stderr.write("[dashboard_http] dashboard.html caricato in cache (%d bytes)\n" % len(cls._cached_html))

    def guess_type(self, path):
        if path.endswith('.yaml') or path.endswith('.yml'):
            return 'text/plain; charset=utf-8'
        return SimpleHTTPRequestHandler.guess_type(self, path)

    def log_message(self, fmt, *args):
        pass  # sopprimi log per-richiesta: riduce I/O su Jetson

    def end_headers(self):
        # Cache lunga per asset vendor statici (non cambiano mai)
        p = self.path.split("?", 1)[0]
        if p.endswith('.js') or p.endswith('.css'):
            self.send_header("Cache-Control", "public, max-age=86400, immutable")
        SimpleHTTPRequestHandler.end_headers(self)

    def translate_path(self, path):
        # /map.yaml -> file mappa configurato
        if path.split("?", 1)[0] in ("/map.yaml", "/maps/map.yaml"):
            return self.map_file
        # / -> dashboard.html
        if path in ("", "/"):
            path = "/dashboard.html"
        clean = path.lstrip("/").split("?", 1)[0]
        return os.path.join(self.web_dir, clean)

    def do_POST(self):
        """Handle POST /save_remap: persist transformed map for JS and Python nodes.

        Expects a JSON body with keys:
            nodes     {id_str: {x, y}}   transformed map nodes
            edges     [{from, to, length}]
            frame     str                coordinate frame name
            transform {theta, scale, tx, ty}

        Saves:
            <web_dir>/remap.json               -- loaded by dashboard.html on reload
            <map_dir>/<stem>_remapped.yaml      -- used by Python nodes on next start
        """
        if self.path.split("?", 1)[0] != "/save_remap":
            self.send_error(404)
            return

        length = int(self.headers.get("content-length") or 0)
        body   = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception as exc:
            self.send_error(400, str(exc))
            return

        # 1. remap.json in web_dir (served as static file to the browser)
        remap_json = os.path.join(self.web_dir, "remap.json")
        with open(remap_json, "w") as fout:
            json.dump(data, fout)

        # 2. _remapped.yaml alongside the original map file
        map_dir   = os.path.dirname(self.map_file)
        map_stem  = os.path.splitext(os.path.basename(self.map_file))[0]
        # Avoid double-suffixing when map_file is already a _remapped.yaml
        if map_stem.endswith("_remapped"):
            map_stem = map_stem[:-len("_remapped")]
        remapped_yaml = os.path.join(map_dir, map_stem + "_remapped.yaml")

        try:
            with open(self.map_file, "r") as fin:
                original = yaml.safe_load(fin)
            frame_id = (original or {}).get("frame_id", "odom")
        except Exception:
            frame_id = "odom"

        yaml_nodes = sorted(
            [{"id": int(k), "x": float(v["x"]), "y": float(v["y"])}
             for k, v in data["nodes"].items()],
            key=lambda n: n["id"]
        )
        yaml_edges = [
            {"from": int(e["from"]), "to": int(e["to"]), "length": float(e.get("length", 0))}
            for e in data["edges"]
        ]
        with open(remapped_yaml, "w") as fout:
            yaml.dump(
                {"frame_id": frame_id, "nodes": yaml_nodes, "edges": yaml_edges},
                fout,
                default_flow_style=False,
                allow_unicode=True
            )

        reply = json.dumps({"ok": True, "remap_yaml": remapped_yaml}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)
        sys.stderr.write("[dashboard_http] remap saved: %s\n" % remapped_yaml)

    def do_GET(self):
        clean_path = self.path.split("?", 1)[0]
        if clean_path in ("", "/", "/dashboard.html"):
            # Serve dalla cache in memoria: nessuna lettura disco
            content = self._cached_html
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
        else:
            SimpleHTTPRequestHandler.do_GET(self)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",     type=int, default=8000)
    ap.add_argument("--web-dir",  default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "web"))
    ap.add_argument("--map",      default="", required=True,
                    help="Path al file YAML della mappa")
    args = ap.parse_args()

    DashboardHandler.web_dir         = os.path.abspath(args.web_dir)
    DashboardHandler.video_server_ip = get_local_ip()

    # Use remapped map if one exists alongside the original
    abs_map  = os.path.abspath(args.map)
    map_dir  = os.path.dirname(abs_map)
    map_stem = os.path.splitext(os.path.basename(abs_map))[0]
    remapped = os.path.join(map_dir, map_stem + "_remapped.yaml")
    if os.path.isfile(remapped):
        DashboardHandler.map_file = remapped
        print("[dashboard_http] Usando mappa remappata: %s" % remapped)
    else:
        DashboardHandler.map_file = abs_map

    if not os.path.isdir(DashboardHandler.web_dir):
        print("ERRORE: web-dir non esiste: %s" % DashboardHandler.web_dir)
        sys.exit(1)
    if not os.path.isfile(DashboardHandler.map_file):
        print("ERRORE: map non esiste: %s" % DashboardHandler.map_file)
        sys.exit(1)

    DashboardHandler.preload()  # cache HTML in RAM una volta sola
    srv = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print("[dashboard_http] http://0.0.0.0:%d" % args.port)
    print("[dashboard_http] web_dir          = %s" % DashboardHandler.web_dir)
    print("[dashboard_http] map              = %s" % DashboardHandler.map_file)
    print("[dashboard_http] video_server_ip  = %s" % DashboardHandler.video_server_ip)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard_http] stop")
        srv.server_close()


if __name__ == "__main__":
    main()
