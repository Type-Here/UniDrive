#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
serve_dashboard.py
------------------
Minimal HTTP server for the web dashboard. Standalone, NOT a ROS node.

Exposes:
  /                -> dashboard.html
  /dashboard.html  -> dashboard.html
  /map.yaml        -> configured map file

CLI arguments:
  --port    HTTP port (default 8000)
  --web-dir web files directory (default ./web relative to the script)
  --map     path to the map YAML file

Example:
  python2 serve_dashboard.py \
        --port 8000 \
        --web-dir /home/jetauto/jetauto_autonomous/web \
        --map /home/jetauto/jetauto_autonomous/maps/map_clean-edited_smooth.yaml

Compatible with Python 2.7 / Python 3.6.9
"""

from __future__ import print_function
import argparse
import json
import os
import sys
import socket

try:
    # Python 3
    from http.server import SimpleHTTPRequestHandler, HTTPServer
except ImportError:
    # Python 2
    from SimpleHTTPServer import SimpleHTTPRequestHandler
    from BaseHTTPServer import HTTPServer


def get_local_ip():
    """Detect the local IP used to reach the network (same as ROS)."""
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
    _cached_html = None  # dashboard.html pre-loaded into memory at startup

    @classmethod
    def preload(cls):
        """Read dashboard.html once and substitute the IP placeholder."""
        fpath = os.path.join(cls.web_dir, "dashboard.html")
        with open(fpath, "rb") as f:
            content = f.read().decode("utf-8")
        content = content.replace("__VIDEO_SERVER_IP__", cls.video_server_ip)
        cls._cached_html = content.encode("utf-8")
        sys.stderr.write("[dashboard_http] dashboard.html loaded into cache (%d bytes)\n" % len(cls._cached_html))

    def guess_type(self, path):
        if path.endswith('.yaml') or path.endswith('.yml'):
            return 'text/plain; charset=utf-8'
        return SimpleHTTPRequestHandler.guess_type(self, path)

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs: reduces I/O on Jetson

    def end_headers(self):
        # Long cache for static vendor assets (they never change)
        p = self.path.split("?", 1)[0]
        if p.endswith('.js') or p.endswith('.css'):
            self.send_header("Cache-Control", "public, max-age=86400, immutable")
        SimpleHTTPRequestHandler.end_headers(self)

    def translate_path(self, path):
        # /map.yaml -> configured map file
        if path.split("?", 1)[0] in ("/map.yaml", "/maps/map.yaml"):
            return self.map_file
        # / -> dashboard.html
        if path in ("", "/"):
            path = "/dashboard.html"
        clean = path.lstrip("/").split("?", 1)[0]
        return os.path.join(self.web_dir, clean)

    def do_POST(self):
        """Handle POST /save_remap_params: persist the 2D similarity transform.

        Expects a JSON body with keys:
            theta   float   rotation angle (rad)
            scale   float   scale factor
            tx      float   translation x
            ty      float   translation y

        Saves <web_dir>/remap_params.json (loaded by dashboard on next page load
        and by Python nodes at startup via relative path ../web/remap_params.json).
        """
        if self.path.split("?", 1)[0] != "/save_remap_params":
            self.send_error(404)
            return

        length = int(self.headers.get("content-length") or 0)
        body   = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception as exc:
            self.send_error(400, str(exc))
            return

        remap_params = os.path.join(self.web_dir, "remap_params.json")
        with open(remap_params, "w") as fout:
            json.dump(data, fout)

        reply = json.dumps({"ok": True, "file": remap_params}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)
        sys.stderr.write("[dashboard_http] remap_params saved: %s\n" % remap_params)

    def do_GET(self):
        clean_path = self.path.split("?", 1)[0]
        if clean_path in ("", "/", "/dashboard.html"):
            # Serve from in-memory cache: no disk read
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
                    help="Path to the map YAML file")
    args = ap.parse_args()

    DashboardHandler.web_dir         = os.path.abspath(args.web_dir)
    DashboardHandler.video_server_ip = get_local_ip()

    DashboardHandler.map_file = os.path.abspath(args.map)

    if not os.path.isdir(DashboardHandler.web_dir):
        print("ERROR: web-dir does not exist: %s" % DashboardHandler.web_dir)
        sys.exit(1)
    if not os.path.isfile(DashboardHandler.map_file):
        print("ERROR: map does not exist: %s" % DashboardHandler.map_file)
        sys.exit(1)

    DashboardHandler.preload()  # load HTML into RAM once
    srv = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print("[dashboard_http] http://0.0.0.0:%d" % args.port)
    print("[dashboard_http] web_dir          = %s" % DashboardHandler.web_dir)
    print("[dashboard_http] map              = %s" % DashboardHandler.map_file)
    print("[dashboard_http] video_server_ip  = %s" % DashboardHandler.video_server_ip)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard_http] stopped")
        srv.server_close()


if __name__ == "__main__":
    main()
