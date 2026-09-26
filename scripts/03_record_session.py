#!/usr/bin/env python3
"""Etapas 2 y 3: captura simultanea sincronizada y grabacion de una sesion.

Uso:
    python scripts/03_record_session.py --name test_sync --seconds 20
    python scripts/03_record_session.py --name demo_001 --out dataset --no-preview

Genera <out>/<name>/ con camN.mp4, synchronization.csv y metadata.yaml.
Al terminar imprime las metricas de sincronizacion de la seccion 40 del roadmap.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from tesis_picking.common import paths
from tesis_picking.common.config import load_cameras
from tesis_picking.common.logging_utils import setup
from tesis_picking.demostraciones_humanas.capture.multi_capture import SyncedRig
from tesis_picking.demostraciones_humanas.capture.recorder import SessionRecorder

OUT_DIRS = {"raw": paths.RAW, "dataset": paths.DATASET, "calib": paths.CALIB_SHOTS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None, help="nombre de la sesion (default: timestamp)")
    ap.add_argument("--out", choices=list(OUT_DIRS), default="raw")
    ap.add_argument("--seconds", type=float, default=None,
                    help="duracion; sin este flag graba hasta pulsar q / Ctrl-C")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--subject", default="", help="id del sujeto (para el dataset)")
    ap.add_argument("--task", default="", help="descripcion de la tarea demostrada")
    ap.add_argument("--no-video", action="store_true",
                    help="solo timestamps, sin escribir mp4 (para medir sync a solas)")
    ap.add_argument("--no-preview", action="store_true",
                    help="sin ventana: reduce carga de CPU y mejora el ritmo")
    args = ap.parse_args()
    setup()

    name = args.name or f"{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = OUT_DIRS[args.out] / name
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"ERROR: {out_dir} ya existe y no esta vacia. Elegir otro --name.")
        return 1

    cfg = load_cameras().subset(args.cams)
    metadata = {"demostracion": {"subject": args.subject, "task": args.task,
                                 "cli": " ".join(sys.argv)}}

    with SyncedRig(cfg.cameras) as rig:
        print("calentando camaras...")
        rig.warmup(25)
        rec = SessionRecorder(rig, out_dir, save_video=not args.no_video,
                              metadata=metadata)

        print(f"\nGRABANDO -> {out_dir}")
        print("  q en la ventana, o Ctrl-C, para detener\n")
        t0 = time.perf_counter()
        n = 0
        try:
            while True:
                fs = rig.capture()
                rec.add(fs)
                n += 1

                if not args.no_preview:
                    ref = next((f for f in fs.frames.values() if f is not None), None)
                    if ref is not None:
                        vis = cv2.resize(ref, (640, int(ref.shape[0] * 640 / ref.shape[1])))
                        el = time.perf_counter() - t0
                        cv2.putText(vis, f"REC {el:5.1f}s  {n} frames  "
                                         f"dt {fs.spread_s * 1e3:5.1f} ms",
                                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow("rec", vis)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            break
                if n % 60 == 0:
                    el = time.perf_counter() - t0
                    print(f"  {el:6.1f}s  {n:5d} framesets  "
                          f"{n / el:5.1f} fps efectivos", end="\r")
                if args.seconds and (time.perf_counter() - t0) >= args.seconds:
                    break
        except KeyboardInterrupt:
            print("\ndetenido por el usuario")

        cv2.destroyAllWindows()
        meta = rec.close()

    print(f"\n\n=== Sesion '{name}' ===")
    print(f"framesets: {meta['session']['n_framesets']} "
          f"(incompletos: {meta['session']['n_framesets_incompletos']})")
    print("\n-- sincronizacion (checkpoint 2 del roadmap) --")
    for k, v in meta["synchronization"].items():
        print(f"  {k:<28} {v}")
    print("\n-- por camara --")
    for cid, c in meta["cameras"].items():
        t = c["timing"]
        print(f"  {cid:<6} {t.get('fps', float('nan')):5.1f} fps  "
              f"dt_med {t.get('dt_median_ms', float('nan')):5.1f} ms  "
              f"perdidos~{int(t.get('dropped_est', 0))}")

    dtmax = meta["synchronization"].get("dt_max_ms", float("nan"))
    print()
    if np.isfinite(dtmax):
        if dtmax > 1000.0 / min(s.fps for s in cfg.cameras):
            print(f"AVISO: dt_max = {dtmax:.1f} ms supera un periodo de frame.")
            print("  A 0.5 m/s de velocidad de muneca eso son "
                  f"{0.5 * dtmax:.0f} mm de error de reconstruccion.")
            print("  Bajar la carga de CPU (--no-preview, --no-video) o la resolucion.")
        else:
            print(f"dt_max = {dtmax:.1f} ms, dentro de un periodo de frame. OK.")
    print(f"\nsalida -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
