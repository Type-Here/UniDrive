#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fake_mask_publisher.py
----------------------
Pubblica `/lane_mask` (sensor_msgs/Image, mono8) leggendo PNG di maschera
da una cartella. Sostituisce SegFormer/SegNet per testare a valle la
pipeline di controllo + dashboard senza accendere il modello.

I PNG devono contenere già i class IDs (0..4) come valori grayscale.
Generabili da JSON LabelMe con `labelme_to_mask.py`.

USO:
  # Sequenza ordinata, loop infinito a 10 Hz
  python fake_mask_publisher.py --dir /path/to/masks --rate 10

  # Ordine casuale (cambia maschera ogni N secondi)
  python fake_mask_publisher.py --dir /path/to/masks --random --hold 0.5

  # Pubblica anche un finto frame RGB sincronizzato (utile per il debug
  # overlay del lane_controller, che si aspetta /depth_cam/rgb/image_raw)
  python fake_mask_publisher.py --dir /path/to/masks --publish-rgb

CONTROLLI A RUNTIME (via topic):
  rostopic pub /fake_mask_publisher/cmd std_msgs/String "data: 'pause'"
  rostopic pub /fake_mask_publisher/cmd std_msgs/String "data: 'resume'"
  rostopic pub /fake_mask_publisher/cmd std_msgs/String "data: 'next'"

Compatibile Python 2.7 / Python 3.x.
"""

from __future__ import print_function
import argparse
import os
import random
import sys
import threading

import numpy as np
import cv2
import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


def list_pngs(folder):
    pngs = []
    for name in sorted(os.listdir(folder)):
        # Esclude i file di debug "_vis.png" generati da labelme_to_mask.py
        if name.endswith('.png') and not name.endswith('_vis.png'):
            pngs.append(os.path.join(folder, name))
    return pngs


class FakeMaskPublisher(object):
    def __init__(self, args):
        rospy.init_node('fake_mask_publisher', anonymous=False)
        self.args = args
        self.bridge = CvBridge()

        # Carica lista file
        self.files = list_pngs(args.dir)
        if not self.files:
            rospy.logfatal("[fake_mask] nessun PNG in %s", args.dir)
            sys.exit(1)

        # Pre-carica tutte le maschere in RAM (sono piccole, ~200 file ~ 1 MB)
        rospy.loginfo("[fake_mask] caricamento di %d maschere...", len(self.files))
        self.masks = []
        for fp in self.files:
            m = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if m is None:
                rospy.logwarn("[fake_mask] skip %s (illeggibile)", fp)
                continue
            self.masks.append((os.path.basename(fp), m))
        rospy.loginfo("[fake_mask] caricate %d maschere, shape esempio: %s",
                      len(self.masks), str(self.masks[0][1].shape))

        # Stato
        self.idx = 0
        self.paused = False
        self.lock = threading.Lock()

        # Pub/Sub
        self.mask_pub = rospy.Publisher(args.mask_topic, Image, queue_size=1)
        if args.publish_rgb:
            self.rgb_pub = rospy.Publisher(args.rgb_topic, Image, queue_size=1)
        else:
            self.rgb_pub = None

        rospy.Subscriber('/fake_mask_publisher/cmd', String, self._cmd_cb)

        rospy.loginfo("[fake_mask] avviato. topic=%s rate=%.1f Hz random=%s",
                      args.mask_topic, args.rate, args.random)

    def _cmd_cb(self, msg):
        cmd = msg.data.strip().lower()
        with self.lock:
            if cmd == 'pause':
                self.paused = True
                rospy.loginfo("[fake_mask] PAUSED")
            elif cmd == 'resume':
                self.paused = False
                rospy.loginfo("[fake_mask] RESUMED")
            elif cmd == 'next':
                self.idx = (self.idx + 1) % len(self.masks)
                rospy.loginfo("[fake_mask] next -> %s", self.masks[self.idx][0])
            else:
                rospy.logwarn("[fake_mask] cmd sconosciuto: %s", cmd)

    def _make_rgb_from_mask(self, mask):
        """Crea un RGB sintetico colorato dalla maschera (per debug overlay)."""
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # palette gradevole
        rgb[mask == 0] = (40, 40, 40)
        rgb[mask == 1] = (80, 80, 80)
        rgb[mask == 2] = (200, 200, 220)   # continue chiare
        rgb[mask == 3] = (180, 180, 200)
        rgb[mask == 4] = (220, 200, 220)
        return rgb

    def run(self):
        rate = rospy.Rate(self.args.rate)
        last_change = rospy.Time.now()
        hold_dur = rospy.Duration(self.args.hold) if self.args.hold > 0 else None

        while not rospy.is_shutdown():
            with self.lock:
                paused = self.paused
                idx = self.idx

            if not paused:
                # Decidi se cambiare maschera
                if self.args.random:
                    if hold_dur is None or (rospy.Time.now() - last_change) >= hold_dur:
                        with self.lock:
                            self.idx = random.randrange(len(self.masks))
                            idx = self.idx
                        last_change = rospy.Time.now()
                else:
                    # sequenza: avanza ad ogni tick
                    with self.lock:
                        self.idx = (self.idx + 1) % len(self.masks)
                        idx = self.idx

            name, mask = self.masks[idx]
            stamp = rospy.Time.now()

            # Pubblica maschera
            try:
                msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                msg.header.stamp = stamp
                msg.header.frame_id = self.args.frame_id
                self.mask_pub.publish(msg)
            except Exception as e:
                rospy.logwarn_throttle(5.0, "[fake_mask] pub mask: %s" % e)

            # Pubblica RGB sintetico (opzionale)
            if self.rgb_pub is not None:
                try:
                    rgb = self._make_rgb_from_mask(mask)
                    rmsg = self.bridge.cv2_to_imgmsg(rgb, encoding='bgr8')
                    rmsg.header.stamp = stamp
                    rmsg.header.frame_id = self.args.frame_id
                    self.rgb_pub.publish(rmsg)
                except Exception as e:
                    rospy.logwarn_throttle(5.0, "[fake_mask] pub rgb: %s" % e)

            rate.sleep()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True,
                    help='cartella con PNG di maschere (mono8, valori 0..4)')
    ap.add_argument('--rate', type=float, default=10.0,
                    help='frequenza di pubblicazione (Hz)')
    ap.add_argument('--random', action='store_true',
                    help='maschere in ordine casuale (invece di sequenziale)')
    ap.add_argument('--hold', type=float, default=0.0,
                    help='in modalità --random, secondi minimi su ogni maschera '
                         '(0 = nuova maschera ad ogni tick)')
    ap.add_argument('--publish-rgb', action='store_true',
                    help='pubblica anche /depth_cam/rgb/image_raw sintetico')
    ap.add_argument('--mask-topic', default='/lane_mask')
    ap.add_argument('--rgb-topic',  default='/depth_cam/rgb/image_raw')
    ap.add_argument('--frame-id',   default='camera_link')
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print("ERRORE: cartella non trovata:", args.dir)
        sys.exit(1)

    try:
        FakeMaskPublisher(args).run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
