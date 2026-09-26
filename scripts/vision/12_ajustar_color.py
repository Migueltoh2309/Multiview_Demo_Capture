#!/usr/bin/env python3
"""Etapa V2: ajustar el rango HSV de la mandarina bajo la iluminacion real.

Los valores de config/objetos.yaml NO son transferibles entre montajes: el
matiz aparente de la fruta depende de la temperatura de color de la luz, y la
saturacion y el valor dependen de la exposicion. Hay que reajustarlos cada vez
que cambie la iluminacion de la estacion.

Uso:
    python scripts/vision/12_ajustar_color.py                # camara en vivo
    python scripts/vision/12_ajustar_color.py --sintetico    # sin camara
    python scripts/vision/12_ajustar_color.py --imagen foto.png

Teclas:
    click izquierdo sobre la fruta = muestrear el HSV bajo el cursor
    m = aplicar el rango deducido de las muestras tomadas
    r = reiniciar el rango al del YAML y borrar las muestras
    g = guardar el rango actual en config/objetos.yaml
    q = salir

Metodo recomendado: pinchar sobre varias mandarinas y sobre varias zonas de
cada una (la parte iluminada y la sombreada), luego apretar los sliders de S y
V hasta que desaparezca el fondo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from tesis_picking.common import paths
from tesis_picking.common.logging_utils import setup
from tesis_picking.vision.deteccion import DetectorColorHSV, ObjetoConfig

VENTANA = "ajuste HSV"
_muestras: list[np.ndarray] = []
_hsv_actual: np.ndarray | None = None


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and _hsv_actual is not None:
        h, w = _hsv_actual.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            # Promedia un parche 5x5 para no depender de un pixel ruidoso
            x0, x1 = max(0, x - 2), min(w, x + 3)
            y0, y1 = max(0, y - 2), min(h, y + 3)
            parche = _hsv_actual[y0:y1, x0:x1].reshape(-1, 3)
            _muestras.append(np.median(parche, axis=0))
            print(f"  muestra {len(_muestras)}: HSV = {np.round(_muestras[-1]).astype(int)}")


def rango_desde_muestras(margen=(8, 60, 60)) -> dict[str, list[int]] | None:
    if not _muestras:
        return None
    M = np.vstack(_muestras)
    lo = np.maximum(M.min(axis=0) - np.array(margen), [0, 0, 0])
    hi = np.minimum(M.max(axis=0) + np.array(margen), [179, 255, 255])
    return {"h": [int(lo[0]), int(hi[0])],
            "s": [int(lo[1]), int(hi[1])],
            "v": [int(lo[2]), int(hi[2])]}


def fuente_frames(args):
    """Genera imagenes BGR de la camara, de un archivo o del simulador."""
    if args.imagen:
        img = cv2.imread(str(args.imagen))
        if img is None:
            raise SystemExit(f"no se pudo leer {args.imagen}")
        while True:
            yield img.copy()
    elif args.sintetico:
        from tesis_picking.vision.simulacion import (EscenaSintetica, EsferaSintetica,
                                                     intrinsecos_d435i)
        intr = intrinsecos_d435i(848, 480)
        esc = EscenaSintetica(intr=intr, esferas=[
            EsferaSintetica(np.array([-0.08, 0.02, 0.65]), 0.0325),
            EsferaSintetica(np.array([0.06, -0.03, 0.72]), 0.030)])
        k = 0
        while True:
            esc.seed = k; esc.__post_init__(); k += 1
            yield esc.render().color
    else:
        from tesis_picking.vision.realsense import RealSenseCamera, RealSenseConfig
        with RealSenseCamera(RealSenseConfig.load(), allow_usb2=args.allow_usb2) as cam:
            for fr in cam.stream():
                yield fr.color


def main() -> int:
    global _hsv_actual
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objeto", default="mandarina")
    ap.add_argument("--imagen", type=Path, default=None)
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--allow-usb2", action="store_true")
    args = ap.parse_args()
    setup()

    cfg = ObjetoConfig.load(args.objeto)
    inicial = {"h": list(cfg.h), "s": list(cfg.s), "v": list(cfg.v)}

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(VENTANA, on_mouse)
    cv2.createTrackbar("H min", VENTANA, cfg.h[0], 179, lambda v: None)
    cv2.createTrackbar("H max", VENTANA, cfg.h[1], 179, lambda v: None)
    cv2.createTrackbar("S min", VENTANA, cfg.s[0], 255, lambda v: None)
    cv2.createTrackbar("S max", VENTANA, cfg.s[1], 255, lambda v: None)
    cv2.createTrackbar("V min", VENTANA, cfg.v[0], 255, lambda v: None)
    cv2.createTrackbar("V max", VENTANA, cfg.v[1], 255, lambda v: None)

    print(__doc__.split("Teclas:")[1] if "Teclas:" in __doc__ else "")

    for color in fuente_frames(args):
        _hsv_actual = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        cfg.h = (cv2.getTrackbarPos("H min", VENTANA), cv2.getTrackbarPos("H max", VENTANA))
        cfg.s = (cv2.getTrackbarPos("S min", VENTANA), cv2.getTrackbarPos("S max", VENTANA))
        cfg.v = (cv2.getTrackbarPos("V min", VENTANA), cv2.getTrackbarPos("V max", VENTANA))

        det = DetectorColorHSV(cfg)
        mask = det.mascara_color(color)
        dets = det.detectar(color)

        vis = det.dibujar(color, dets)
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        segmentado = cv2.bitwise_and(color, color, mask=mask)
        fila = np.hstack([vis, mask3, segmentado])
        fila = cv2.resize(fila, None, fx=0.62, fy=0.62)

        txt = (f"H[{cfg.h[0]},{cfg.h[1]}] S[{cfg.s[0]},{cfg.s[1]}] "
               f"V[{cfg.v[0]},{cfg.v[1]}]   detecciones: {len(dets)}   "
               f"muestras: {len(_muestras)}")
        cv2.putText(fila, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(VENTANA, fila)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            _muestras.clear()
            for nombre, val in [("H min", inicial["h"][0]), ("H max", inicial["h"][1]),
                                ("S min", inicial["s"][0]), ("S max", inicial["s"][1]),
                                ("V min", inicial["v"][0]), ("V max", inicial["v"][1])]:
                cv2.setTrackbarPos(nombre, VENTANA, val)
            print("  rango reiniciado")
        if k == ord("m"):
            r = rango_desde_muestras()
            if r:
                for nombre, val in [("H min", r["h"][0]), ("H max", r["h"][1]),
                                    ("S min", r["s"][0]), ("S max", r["s"][1]),
                                    ("V min", r["v"][0]), ("V max", r["v"][1])]:
                    cv2.setTrackbarPos(nombre, VENTANA, val)
                print(f"  rango desde {len(_muestras)} muestras: {r}")
        if k == ord("g"):
            p = paths.CONFIG / "objetos.yaml"
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            raw[args.objeto]["hsv"] = {"h": list(cfg.h), "s": list(cfg.s),
                                       "v": list(cfg.v)}
            p.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
                         encoding="utf-8")
            print(f"  guardado en {p}: H{list(cfg.h)} S{list(cfg.s)} V{list(cfg.v)}")

    cv2.destroyAllWindows()
    print(f"\nrango final: H{list(cfg.h)} S{list(cfg.s)} V{list(cfg.v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
