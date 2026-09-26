#!/usr/bin/env python3
"""Etapas 6, 7 y 14: calibracion extrinseca multivista y definicion de {W}.

Uso:
    python scripts/07_calibrate_extrinsics.py --set calib_01
    python scripts/07_calibrate_extrinsics.py --set calib_01 --anchor shot_000

Requiere calib/<cam>.yaml con los intrinsecos ya validados (script 06).
Escribe los extrinsecos en esos mismos archivos y el reporte en
results/metricas/extrinsics_<set>.json.

Checkpoint 4: las tres camaras quedan expresadas en el mismo {W}.
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
from tesis_picking.common.config import load_cameras, load_charuco, load_world_frame
from tesis_picking.common.logging_utils import setup
from tesis_picking.demostraciones_humanas.calibration.charuco import CharucoBoardWrapper
from tesis_picking.demostraciones_humanas.calibration.extrinsics import calibrate_extrinsics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--anchor", default=None,
                    help="toma que define {W} (default: la marcada en index.json)")
    ap.add_argument("--min-corners", type=int, default=8)
    ap.add_argument("--no-refine", action="store_true",
                    help="omitir el bundle adjustment (solo para depurar)")
    args = ap.parse_args()
    setup()

    set_dir = paths.CALIB_SHOTS / args.set
    index_path = set_dir / "index.json"
    if not index_path.exists():
        print(f"ERROR: falta {index_path}. Ejecutar antes 05_capture_charuco.py")
        return 1
    index = json.loads(index_path.read_text())

    anchor = args.anchor or index.get("anchor")
    if not anchor:
        print("ERROR: el set no tiene toma ancla. Indicarla con --anchor shot_NNN,")
        print("o recapturar con --anchor colocando el tablero en el origen de {W}.")
        return 1

    cfg = load_cameras().subset(args.cams)
    board = CharucoBoardWrapper(load_charuco())
    wf = load_world_frame()

    # --- intrinsecos ---
    intrinsics = {}
    for spec in cfg.cameras:
        p = paths.CALIB / f"{spec.id}.yaml"
        if not p.exists():
            print(f"ERROR: falta {p}. Ejecutar antes 06_calibrate_intrinsics.py")
            return 1
        intrinsics[spec.id] = CameraModel.load(p).intr

    # --- detecciones ---
    print(f"\ndetectando el tablero en el set '{args.set}'...")
    detections: dict[str, dict] = {}
    for shot_dir in sorted(d for d in set_dir.iterdir()
                           if d.is_dir() and d.name.startswith("shot_")):
        per_cam = {}
        for cid in cfg.ids:
            img_path = shot_dir / f"{cid}.png"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            det = board.detect(img)
            if det.n_corners >= args.min_corners:
                per_cam[cid] = det
        if per_cam:
            detections[shot_dir.name] = per_cam

    n_multi = sum(1 for v in detections.values() if len(v) >= 2)
    print(f"  tomas con deteccion: {len(detections)}  "
          f"(con >= 2 camaras a la vez: {n_multi})")
    if n_multi < 4 and len(cfg.ids) > 1:
        print("  AVISO: muy pocas tomas compartidas. La union entre camaras sera fragil.")

    if anchor not in detections:
        print(f"ERROR: la toma ancla '{anchor}' no tiene detecciones validas.")
        return 1

    # --- calibracion ---
    print(f"\nancla: {anchor}   T_world_board de config/world_frame.yaml")
    try:
        res = calibrate_extrinsics(
            detections, board, intrinsics,
            anchor_key=anchor, T_world_anchor=wf.T_world_board,
            min_corners=args.min_corners, refine=not args.no_refine)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    rep = res.report()
    print("\n=== Resultado extrinseco ===")
    print(json.dumps(rep, indent=2, ensure_ascii=False))

    # --- guardar ---
    for cam in res.rig:
        cam.save(paths.CALIB / f"{cam.cam_id}.yaml")
        print(f"  -> {paths.CALIB / f'{cam.cam_id}.yaml'}")

    paths.ensure(paths.METRICAS)
    out = paths.METRICAS / f"extrinsics_{args.set}.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\nreporte -> {out}")

    missing = [c for c in cfg.ids if c not in res.rig.ids]
    ok = res.rms_global_px < 1.0 and not missing
    print()
    if missing:
        print(f"NO SUPERADO: camaras sin referir a {{W}}: {missing}")
    if res.rms_global_px >= 1.0:
        print(f"NO SUPERADO: RMS global {res.rms_global_px:.2f} px >= 1.0")
        print("  Causas tipicas: alguna camara se movio durante la captura;")
        print("  intrinsecos flojos; tablero no plano.")
    print("CHECKPOINT 4 " + ("SUPERADO" if ok else "NO SUPERADO"))
    print("\nSiguiente: python scripts/08_validate_geometry.py --set", args.set)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
