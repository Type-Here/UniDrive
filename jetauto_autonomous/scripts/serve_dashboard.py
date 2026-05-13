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
import os
import sys

try:
    # Python 3
    from http.server import SimpleHTTPRequestHandler, HTTPServer
except ImportError:
    # Python 2
    from SimpleHTTPServer import SimpleHTTPRequestHandler
    from BaseHTTPServer import HTTPServer


class DashboardHandler(SimpleHTTPRequestHandler):
    web_dir = "."
    map_file = ""

    def log_message(self, fmt, *args):
        sys.stderr.write("[dashboard_http] " + (fmt % args) + "\n")

    def translate_path(self, path):
        # /map.yaml -> file mappa configurato
        if path.split("?", 1)[0] in ("/map.yaml", "/maps/map.yaml"):
            return self.map_file
        # / -> dashboard.html
        if path in ("", "/"):
            path = "/dashboard.html"
        clean = path.lstrip("/").split("?", 1)[0]
        return os.path.join(self.web_dir, clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",     type=int, default=8000)
    ap.add_argument("--web-dir",  default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "web"))
    ap.add_argument("--map",      default="", required=True,
                    help="Path al file YAML della mappa")
    args = ap.parse_args()

    DashboardHandler.web_dir = os.path.abspath(args.web_dir)
    DashboardHandler.map_file = os.path.abspath(args.map)

    if not os.path.isdir(DashboardHandler.web_dir):
        print("ERRORE: web-dir non esiste: %s" % DashboardHandler.web_dir)
        sys.exit(1)
    if not os.path.isfile(DashboardHandler.map_file):
        print("ERRORE: map non esiste: %s" % DashboardHandler.map_file)
        sys.exit(1)

    srv = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print("[dashboard_http] http://0.0.0.0:%d" % args.port)
    print("[dashboard_http] web_dir = %s" % DashboardHandler.web_dir)
    print("[dashboard_http] map     = %s" % DashboardHandler.map_file)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard_http] stop")
        srv.server_close()


if __name__ == "__main__":
    main()
