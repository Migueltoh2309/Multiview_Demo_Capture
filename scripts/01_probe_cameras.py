#!/usr/bin/env python3
"""Etapa 1 del roadmap: probar las camaras individualmente.

Uso:
    python scripts/01_probe_cameras.py --scan          # descubre devices
    python scripts/01_probe_cameras.py                 # prueba las de cameras.yaml
    python scripts/01_probe_cameras.py --cams cam1     # solo una

Registra, para cada camara: device, resolucion, FOURCC, FPS pedido y FPS MEDIDO.
Criterio de validacion (roadmap seccion 8): las tres camaras deben funcionar
individualmente antes de continuar.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2

from tesis_picking.common import paths
from tesis_picking.common.config import CameraSpec, load_cameras
from tesis_picking.common.logging_utils import setup
from tesis_picking.demostraciones_humanas.capture.camera_source import probe_device


def scan_devices() -> None:
    """Lista los /dev/video* que realmente entregan imagen, con su nombre USB."""
    print("\n=== Devices de video presentes ===")
    nodes = sorted(glob.glob("/dev/video*"),
                   key=lambda p: int("".join(c for c in p if c.isdigit()) or 0))
    if not nodes:
        print("  (ninguno) -> conectar las camaras USB y reintentar")
        return

    try:
        listing = subprocess.run(["v4l2-ctl", "--list-devices"],
                                 capture_output=True, text=True, timeout=10).stdout
        print(listing)
    except (FileNotFoundError, subprocess.SubprocessError):
        print("  (v4l2-ctl no disponible: instalar con `sudo apt install v4l-utils`)\n")

    print(f"{'device':<16}{'abre':<8}{'entrega frame':<16}{'resolucion'}")
    print("-" * 60)
    for node in nodes:
        cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
        opened = cap.isOpened()
        ok, frame = (cap.read() if opened else (False, None))
        res = f"{frame.shape[1]}x{frame.shape[0]}" if ok and frame is not None else "-"
        print(f"{node:<16}{str(opened):<8}{str(bool(ok)):<16}{res}")
        cap.release()
    print("\nNota: cada camara UVC suele exponer DOS nodos (p.ej. video0 y video1);")
    print("solo el primero entrega imagen. Usar ese en config/cameras.yaml.")
    print("\nPara ids estables ante reinicios, preferir /dev/v4l/by-id/... :")
    for p in sorted(glob.glob("/dev/v4l/by-id/*")):
        print(f"  {p}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true",
                    help="solo descubrir devices, sin abrir las de cameras.yaml")
    ap.add_argument("--cams", nargs="*", default=None, help="ids a probar")
    ap.add_argument("--frames", type=int, default=90,
                    help="frames para medir el FPS real (default: 90 = 3 s a 30 FPS)")
    ap.add_argument("--save", action="store_true",
                    help="guardar el reporte en results/metricas/")
    args = ap.parse_args()
    setup()

    if args.scan:
        scan_devices()
        return 0

    cfg = load_cameras().subset(args.cams)
    print(f"\nProbando {len(cfg.cameras)} camara(s), {args.frames} frames cada una...\n")

    reports = []
    for spec in cfg.cameras:
        print(f"--- {spec.id} ({spec.device}, rol: {spec.role or 'sin definir'})")
        rep = probe_device(spec, n_frames=args.frames)
        reports.append(rep)
        row = rep.as_row()
        for k, v in row.items():
            if k != "camera_id":
                print(f"    {k:<16} {v}")
        print()

    print("=" * 72)
    ok_all = True
    for rep in reports:
        status = "OK" if (rep.opened and rep.resolution_ok and rep.fps_ok) else "REVISAR"
        ok_all &= status == "OK"
        print(f"  {rep.cam_id:<8} {status:<10} "
              f"{rep.actual.get('width')}x{rep.actual.get('height')} "
              f"@ {rep.measured_fps:.1f} FPS medidos "
              f"[{rep.actual.get('fourcc','')}] {rep.error}")
    print("=" * 72)

    if not ok_all:
        print("\nDiagnostico habitual:")
        print("  * FPS medido muy por debajo del pedido -> el FOURCC no quedo en")
        print("    MJPG. Comprobar con: v4l2-ctl -d <dev> --list-formats-ext")
        print("  * Resolucion distinta -> la camara no soporta esa combinacion.")
        print("  * Varias camaras en el mismo hub USB2 se quedan sin ancho de banda:")
        print("    repartirlas entre controladoras distintas.")

    if args.save:
        paths.ensure(paths.METRICAS)
        out = paths.METRICAS / f"probe_cameras_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.write_text(json.dumps([r.as_row() for r in reports], indent=2))
        print(f"\nreporte -> {out}")

    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
