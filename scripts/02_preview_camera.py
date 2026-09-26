#!/usr/bin/env python3
"""Etapa 1/5: previsualizar camaras para colocarlas fisicamente y fijar exposicion.

Uso:
    python scripts/02_preview_camera.py                 # todas, en mosaico
    python scripts/02_preview_camera.py --cams cam2     # una sola
    python scripts/02_preview_camera.py --charuco       # superpone deteccion

Teclas:  q = salir   |   s = guardar snapshot de todas   |   g = rejilla on/off
Con --charuco muestra cuantas esquinas ve cada camara: es la forma practica de
colocar las camaras de modo que todas cubran el volumen de captura.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

import cv2
import numpy as np

from tesis_picking.common import paths
from tesis_picking.common.config import load_cameras, load_charuco
from tesis_picking.common.logging_utils import setup
from tesis_picking.demostraciones_humanas.calibration.charuco import CharucoBoardWrapper
from tesis_picking.demostraciones_humanas.capture.multi_capture import SyncedRig


def mosaic(frames: dict, width: int = 640) -> np.ndarray:
    """Compone los frames en una rejilla, escalados a `width` de ancho."""
    tiles = []
    for cid, f in frames.items():
        if f is None:
            f = np.zeros((360, 640, 3), np.uint8)
            cv2.putText(f, f"{cid}: SIN SENAL", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        h, w = f.shape[:2]
        f = cv2.resize(f, (width, int(h * width / w)))
        tiles.append(f)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=0) for t in tiles]
    return np.hstack(tiles)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--charuco", action="store_true",
                    help="detectar y dibujar el tablero ChArUco")
    ap.add_argument("--width", type=int, default=640, help="ancho de cada tile")
    args = ap.parse_args()
    setup()

    cfg = load_cameras().subset(args.cams)
    board = CharucoBoardWrapper(load_charuco()) if args.charuco else None
    grid = False

    with SyncedRig(cfg.cameras) as rig:
        rig.warmup(10)
        print("q = salir | s = snapshot | g = rejilla")
        while True:
            fs = rig.capture()
            shown = {}
            for cid, frame in fs.frames.items():
                if frame is None:
                    shown[cid] = None
                    continue
                vis = frame.copy()
                label = cid
                if board is not None:
                    det = board.detect(frame)
                    vis = board.draw(vis, det)
                    label = f"{cid}  esquinas: {det.n_corners}/{board.n_corners}"
                if grid:
                    h, w = vis.shape[:2]
                    for x in range(0, w, 80):
                        cv2.line(vis, (x, 0), (x, h), (0, 180, 0), 1)
                    for y in range(0, h, 80):
                        cv2.line(vis, (0, y), (w, y), (0, 180, 0), 1)
                cv2.putText(vis, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2, cv2.LINE_AA)
                shown[cid] = vis

            view = mosaic(shown, args.width)
            cv2.putText(view, f"dt_max = {fs.spread_s * 1e3:6.1f} ms",
                        (12, view.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("preview", view)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("g"):
                grid = not grid
            if k == ord("s"):
                stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
                out = paths.RAW / "snapshots" / stamp
                out.mkdir(parents=True, exist_ok=True)
                for cid, frame in fs.frames.items():
                    if frame is not None:
                        cv2.imwrite(str(out / f"{cid}.png"), frame)
                print(f"snapshot -> {out}")

    cv2.destroyAllWindows()
    print("\nmetricas de sincronizacion de la sesion de preview:")
    for k, v in rig.metrics.summary().items():
        print(f"  {k:<28} {v:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
