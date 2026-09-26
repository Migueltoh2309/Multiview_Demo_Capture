#!/usr/bin/env python3
"""Etapas 5, 6 y 13: calibracion intrinseca por camara y su validacion.

Uso:
    python scripts/06_calibrate_intrinsics.py --set calib_01
    python scripts/06_calibrate_intrinsics.py --set calib_01 --cams cam2 --rational

Escribe calib/<cam>.yaml (K, D, RMS) y results/metricas/intrinsics_<set>.json,
y guarda comparativas original/corregida en results/figuras/.

Criterio de avance (checkpoint 3): RMS < ~0.5 px y cobertura del encuadre alta.
Un RMS bajo con cobertura baja es una calibracion ENGANOSA: parece buena y
tiene D mal estimada.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tesis_picking.common import paths
from tesis_picking.common.camera_model import CameraModel
from tesis_picking.common.config import load_cameras, load_charuco
from tesis_picking.common.logging_utils import setup
from tesis_picking.demostraciones_humanas.calibration.charuco import CharucoBoardWrapper
from tesis_picking.demostraciones_humanas.calibration.intrinsics import (
    calibrate_intrinsics, undistortion_preview)


def load_shots(set_dir: Path, cam_id: str) -> list[tuple[str, Path]]:
    shots = sorted(d for d in set_dir.iterdir() if d.is_dir() and d.name.startswith("shot_"))
    return [(d.name, d / f"{cam_id}.png") for d in shots if (d / f"{cam_id}.png").exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--min-corners", type=int, default=8)
    ap.add_argument("--max-view-rms", type=float, default=1.0,
                    help="descarta vistas por encima de este RMS y recalibra")
    ap.add_argument("--rational", action="store_true",
                    help="modelo racional (lentes gran angular)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    setup()

    set_dir = paths.CALIB_SHOTS / args.set
    if not set_dir.exists():
        print(f"ERROR: no existe {set_dir}. Ejecutar antes 05_capture_charuco.py")
        return 1

    board = CharucoBoardWrapper(load_charuco())
    cfg = load_cameras().subset(args.cams)
    paths.ensure(paths.CALIB, paths.METRICAS, paths.FIGURAS)

    report: dict[str, dict] = {}
    ok_all = True

    for spec in cfg.cameras:
        shots = load_shots(set_dir, spec.id)
        print(f"\n=== {spec.id}: {len(shots)} imagenes en el set ===")
        if len(shots) < 6:
            print("  ERROR: menos de 6 imagenes. Capturar mas tomas.")
            ok_all = False
            continue

        dets, names, size = [], [], None
        for name, path in shots:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if size is None:
                size = (img.shape[1], img.shape[0])
            elif (img.shape[1], img.shape[0]) != size:
                print(f"  ERROR: {name} tiene {img.shape[1]}x{img.shape[0]}, "
                      f"distinto de {size[0]}x{size[1]}. "
                      "Todas las tomas deben usar la MISMA resolucion.")
                ok_all = False
                break
            det = board.detect(img)
            if det.is_usable(args.min_corners):
                dets.append(det)
                names.append(name)
        if size is None or not dets:
            print("  ERROR: ninguna deteccion utilizable.")
            ok_all = False
            continue

        print(f"  vistas utilizables: {len(dets)}/{len(shots)}")
        try:
            res = calibrate_intrinsics(
                dets, board, size, view_names=names,
                min_corners=args.min_corners,
                max_view_rms_px=args.max_view_rms,
                rational_model=args.rational)
        except ValueError as exc:
            print(f"  ERROR: {exc}")
            ok_all = False
            continue

        rep = res.report()
        report[spec.id] = rep
        for k, v in rep.items():
            print(f"  {k:<22} {v}")

        # --- criterios de aceptacion (checkpoint 3) ---
        problems = []
        if res.intr.rms_reprojection_px > 0.7:
            problems.append(f"RMS {res.intr.rms_reprojection_px:.3f} px > 0.7 "
                            "(tablero curvado, desenfoque o square_length_m erroneo)")
        if res.coverage["grid_fill"] < 0.60:
            problems.append(f"cobertura del encuadre {res.coverage['grid_fill']:.0%} < 60% "
                            "(mover el tablero a mas zonas de la imagen)")
        if res.coverage["edge_point_ratio"] < 0.20:
            problems.append(f"solo {res.coverage['edge_point_ratio']:.0%} de puntos "
                            "cerca de los bordes: D quedara mal estimada")
        if res.intr.n_views < 15:
            problems.append(f"solo {res.intr.n_views} vistas (objetivo >= 20)")
        if problems:
            ok_all = False
            print("  REVISAR:")
            for p in problems:
                print(f"    - {p}")
        else:
            print("  ACEPTADA")

        # Preserva los extrinsecos si ya existian: recalibrar K los invalida,
        # asi que se avisa explicitamente en lugar de dejarlos en silencio.
        out = paths.CALIB / f"{spec.id}.yaml"
        had_extr = False
        if out.exists():
            prev = CameraModel.load(out)
            had_extr = prev.T_world_cam is not None
        CameraModel(spec.id, res.intr, None).save(out)
        print(f"  -> {out}")
        if had_extr:
            print("  AVISO: se borro la calibracion EXTRINSECA previa de esta camara.")
            print("         Al cambiar K hay que rehacer 08_calibrate_extrinsics.py")

        if not args.no_figures:
            img = cv2.imread(str(shots[len(shots) // 2][1]))
            if img is not None:
                orig, und = undistortion_preview(img, res.intr)
                fig = np.hstack([orig, und])
                cv2.putText(fig, "ORIGINAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2)
                cv2.putText(fig, "CORREGIDA", (orig.shape[1] + 20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                fp = paths.FIGURAS / f"undistort_{spec.id}_{args.set}.png"
                cv2.imwrite(str(fp), fig)
                print(f"  figura -> {fp}")

    out = paths.METRICAS / f"intrinsics_{args.set}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreporte -> {out}")
    print("\nCHECKPOINT 3 " + ("SUPERADO" if ok_all else "NO SUPERADO: revisar arriba"))
    print("En las figuras, comprobar que las lineas rectas del mundo salen")
    print("rectas en la imagen corregida, especialmente en esquinas y bordes.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
