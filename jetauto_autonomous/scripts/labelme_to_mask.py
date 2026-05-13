#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
labelme_to_mask.py
------------------
Converte annotazioni LabelMe (formato JSON con "shapes": [{label, points,
shape_type=polygon}]) in maschere mono8 utilizzabili da
fake_mask_publisher.py.

Schema classi (deve corrispondere a `lane_params.yaml`):
  0 = background  (default per pixel non annotati)
  1 = road
  2 = lane_marking
  3 = lane_dashed
  4 = zebra

USO:
  # Converte un singolo file
  python labelme_to_mask.py input.json -o output.png

  # Converte una cartella intera
  python labelme_to_mask.py /path/to/dataset -o /path/to/masks/

  # Ridimensiona output (utile per ridurre carico su Jetson)
  python labelme_to_mask.py /path/to/dataset -o /path/to/masks/ --resize 512x256

  # Genera anche versione colorata di debug (suffisso _vis.png)
  python labelme_to_mask.py /path/to/dataset -o /path/to/masks/ --debug

Compatibile Python 2.7 / Python 3.x.
"""

from __future__ import print_function
import argparse
import json
import os
import sys

import numpy as np
import cv2


CLASS_MAP = {
    'background':   0,
    'road':         1,
    'lane_marking': 2,
    'lane_dashed':  3,
    'zebra':        4,
}

# Ordine di disegno: classi in cima si sovrappongono a quelle precedenti.
# Importante: lane_marking e lane_dashed sopra a road; zebra in cima a tutto.
DRAW_ORDER = ['road', 'lane_marking', 'lane_dashed', 'zebra']

DEBUG_COLORS = {
    0: (30, 30, 30),
    1: (60, 60, 60),
    2: (0, 0, 255),     # rosso
    3: (0, 255, 255),   # giallo
    4: (255, 0, 255),   # magenta
}


def json_to_mask(json_path):
    """Carica un JSON LabelMe e restituisce (mask, h, w, unknown_labels)."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    h = data.get('imageHeight')
    w = data.get('imageWidth')
    if not h or not w:
        raise ValueError("JSON senza imageHeight/imageWidth: %s" % json_path)

    mask = np.zeros((h, w), dtype=np.uint8)
    unknown = set()

    for label_name in DRAW_ORDER:
        cid = CLASS_MAP[label_name]
        for s in data.get('shapes', []):
            if s.get('label') != label_name:
                continue
            if s.get('shape_type') != 'polygon':
                # Per ora gestiamo solo poligoni. Estensibile a rettangoli/punti.
                continue
            pts = np.array(s['points'], dtype=np.int32)
            cv2.fillPoly(mask, [pts], cid)

    # Conta label non riconosciuti per warning
    for s in data.get('shapes', []):
        lbl = s.get('label')
        if lbl not in CLASS_MAP:
            unknown.add(lbl)

    return mask, h, w, unknown


def resize_mask(mask, target):
    """target = "WxH" stringa. Resize NEAREST per preservare class IDs."""
    if not target:
        return mask
    try:
        w, h = [int(x) for x in target.lower().split('x')]
    except Exception:
        raise ValueError("--resize deve essere nel formato WxH (es. 512x256)")
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def save_outputs(mask, out_path, debug=False):
    cv2.imwrite(out_path, mask)
    if debug:
        h, w = mask.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for cid, col in DEBUG_COLORS.items():
            vis[mask == cid] = col
        vis_path = os.path.splitext(out_path)[0] + '_vis.png'
        cv2.imwrite(vis_path, vis)


def find_jsons(input_path):
    """Restituisce la lista di JSON da processare e la modalità (file/dir)."""
    if os.path.isfile(input_path) and input_path.endswith('.json'):
        return [input_path], 'file'
    if os.path.isdir(input_path):
        out = []
        for name in sorted(os.listdir(input_path)):
            if name.endswith('.json'):
                out.append(os.path.join(input_path, name))
        return out, 'dir'
    raise ValueError("input deve essere un file .json o una cartella: %s"
                     % input_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='file JSON o cartella con JSON')
    ap.add_argument('-o', '--output', required=True,
                    help='file PNG o cartella di output')
    ap.add_argument('--resize', default='',
                    help='ridimensiona output (es. 512x256)')
    ap.add_argument('--debug', action='store_true',
                    help='salva anche versione colorata _vis.png')
    args = ap.parse_args()

    jsons, mode = find_jsons(args.input)
    if not jsons:
        print("Nessun JSON trovato in:", args.input)
        sys.exit(1)

    if mode == 'file':
        out_path = args.output
        out_dir = os.path.dirname(out_path) or '.'
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
    else:
        if not os.path.isdir(args.output):
            os.makedirs(args.output)

    all_unknown = set()
    n_ok = 0
    for jp in jsons:
        try:
            mask, h, w, unknown = json_to_mask(jp)
            mask = resize_mask(mask, args.resize)
            all_unknown.update(unknown)

            if mode == 'file':
                op = args.output
            else:
                base = os.path.splitext(os.path.basename(jp))[0]
                op = os.path.join(args.output, base + '.png')

            save_outputs(mask, op, debug=args.debug)
            n_ok += 1
            if mode == 'dir' and n_ok % 25 == 0:
                print("  ... %d/%d" % (n_ok, len(jsons)))
        except Exception as e:
            print("ERRORE in %s: %s" % (jp, e))

    print("Convertiti %d/%d file." % (n_ok, len(jsons)))
    if all_unknown:
        print("ATTENZIONE: label non riconosciuti (ignorati): %s"
              % sorted(all_unknown))


if __name__ == '__main__':
    main()
