#!/usr/bin/env python3
"""Etapas 7, 16 y 17: validacion metrologica del rig calibrado.

Reconstruye las esquinas del tablero ChArUco --- cuya geometria se conoce por
construccion --- y compara contra la verdad de terreno. Es la evidencia
experimental de la frase "caracterizar su exactitud metrologica" del objetivo
especifico 1 de la tesis.

Uso:
    python scripts/08_validate_geometry.py --set calib_01
    python scripts/08_validate_geometry.py --set calib_01 --live   # en vivo

Metricas producidas:
    error_3d_*        error absoluto punto a punto (solo en la toma ancla,
                      donde se conoce T_world_board)
    error_distancia_* error de distancias entre esquinas: INDEPENDIENTE de la
                      pose del tablero, por eso se evalua en todas las tomas
    planaridad_rms    dispersion respecto al plano ajustado: ruido puro
    camaras_por_punto reparto 3 / 2 / <2 camaras (metrica de robustez, sec. 40)

Checkpoints 5 y 6 del roadmap.
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

from tesis_picking.common import paths
from tesis_picking.common.camera_model import CameraRig
from tesis_picking.common.config import load_cameras, load_charuco, load_world_frame
from tesis_picking.common.logging_utils import setup
from tesis_picking.demostraciones_humanas.calibration.charuco import CharucoBoardWrapper
from tesis_picking.demostraciones_humanas.calibration.validation import (
    check_board_reconstruction, known_distance_check)
from tesis_picking.demostraciones_humanas.capture.multi_capture import SyncedRig


def aggregate(values: list[float]) -> dict[str, float]:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {}
    return {"mean": round(float(np.mean(a)), 3),
            "rms": round(float(np.sqrt(np.mean(a ** 2))), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
            "max": round(float(np.max(a)), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=None, help="set de tomas a validar")
    ap.add_argument("--live", action="store_true",
                    help="validar en vivo con las camaras (util al recolocarlas)")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--max-reproj-px", type=float, default=5.0)
    args = ap.parse_args()
    setup()

    cfg = load_cameras().subset(args.cams)
    board = CharucoBoardWrapper(load_charuco())
    wf = load_world_frame()
    try:
        rig = CameraRig.load(paths.CALIB, cfg.ids)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    if not rig.calibrated:
        print("ERROR: falta la calibracion extrinseca. Ejecutar 07_calibrate_extrinsics.py")
        return 1

    print("\n=== Rig ===")
    for cam in rig:
        print(f"  {cam.cam_id:<8} pos {np.round(cam.center_world, 3)} m   "
              f"RMS intrinseco {cam.intr.rms_reprojection_px:.3f} px")
    print("  baselines:", {f"{a}-{b}": round(v, 3)
                           for (a, b), v in rig.baselines_m().items()})

    if args.live:
        return run_live(cfg, board, rig, args)

    if not args.set:
        print("ERROR: indicar --set o usar --live")
        return 1
    set_dir = paths.CALIB_SHOTS / args.set
    index = json.loads((set_dir / "index.json").read_text())
    anchor = index.get("anchor")

    results, all_dist, all_pt, all_plan = [], [], [], []
    hist_total: dict[int, int] = {}

    for shot_dir in sorted(d for d in set_dir.iterdir()
                           if d.is_dir() and d.name.startswith("shot_")):
        dets = {}
        for cid in rig.ids:
            p = shot_dir / f"{cid}.png"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    dets[cid] = board.detect(img)
        if sum(1 for d in dets.values() if d.n_corners >= 6) < 2:
            continue

        # El error 3D absoluto solo tiene sentido donde se conoce la pose real
        # del tablero, es decir, en la toma ancla.
        T_wb = wf.T_world_board if shot_dir.name == anchor else None
        chk = check_board_reconstruction(
            dets, board, rig, frame_key=shot_dir.name,
            T_world_board=T_wb, max_reproj_px=args.max_reproj_px)
        if chk.n_points < 4:
            continue
        results.append(chk.summary())
        all_dist.extend(chk.distance_errors_mm.tolist())
        all_pt.extend(chk.point_errors_mm.tolist())
        all_plan.append(chk.planarity_rms_mm)
        for k, v in chk.n_cameras_hist.items():
            hist_total[k] = hist_total.get(k, 0) + v

    if not results:
        print("ERROR: ninguna toma con >= 2 camaras viendo el tablero.")
        return 1

    total = sum(hist_total.values()) or 1
    summary = {
        "set": args.set,
        "n_tomas_evaluadas": len(results),
        "error_distancia_mm": aggregate(all_dist),
        "error_3d_absoluto_mm_toma_ancla": aggregate(all_pt),
        "planaridad_rms_mm": aggregate(all_plan),
        "robustez_camaras_por_punto_pct": {
            f"{k}_camaras": round(100 * v / total, 1)
            for k, v in sorted(hist_total.items(), reverse=True)},
        "por_toma": results,
    }

    print("\n=== Validacion metrologica ===")
    for key in ("error_distancia_mm", "error_3d_absoluto_mm_toma_ancla",
                "planaridad_rms_mm", "robustez_camaras_por_punto_pct"):
        print(f"  {key}: {summary[key]}")

    # Chequeo explicito de distancia conocida (etapa 17), sobre la toma ancla.
    if anchor:
        adets = {}
        for cid in rig.ids:
            p = set_dir / anchor / f"{cid}.png"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    adets[cid] = board.detect(img)
        nx = board.cfg.squares_x - 1
        kd = known_distance_check(adets, board, rig, 0, nx * (board.cfg.squares_y - 2))
        summary["distancia_conocida"] = kd
        print(f"  distancia_conocida: {kd}")

    paths.ensure(paths.METRICAS)
    out = paths.METRICAS / f"validacion_geometrica_{args.set}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreporte -> {out}")

    d_rms = summary["error_distancia_mm"].get("rms", float("inf"))
    ok5 = d_rms < 5.0
    ok6 = summary["error_distancia_mm"].get("mean", float("inf")) < 3.0
    print(f"\nCHECKPOINT 5 (puntos conocidos) {'SUPERADO' if ok5 else 'NO SUPERADO'}")
    print(f"CHECKPOINT 6 (distancia conocida) {'SUPERADO' if ok6 else 'NO SUPERADO'}")
    if not (ok5 and ok6):
        print("\nSi el error medio es SISTEMATICO (mismo signo, proporcional a la")
        print("distancia), es un error de ESCALA: volver a medir el tablero con")
        print("calibrador y corregir square_length_m en config/charuco.yaml.")
    return 0 if (ok5 and ok6) else 1


def run_live(cfg, board, rig, args) -> int:
    """Validacion en vivo: util mientras se recolocan las camaras."""
    print("\nmodo en vivo: mover el tablero por el volumen de captura. q = salir")
    with SyncedRig(cfg.cameras) as srig:
        srig.warmup(15)
        while True:
            fs = srig.capture()
            dets = {c: board.detect(f) for c, f in fs.frames.items() if f is not None}
            chk = check_board_reconstruction(dets, board, rig, frame_key="live",
                                             max_reproj_px=args.max_reproj_px)
            tiles = []
            for cid, frame in fs.frames.items():
                if frame is None:
                    continue
                vis = board.draw(frame, dets[cid])
                cv2.putText(vis, cid, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2)
                tiles.append(cv2.resize(vis, (520, int(vis.shape[0] * 520 / vis.shape[1]))))
            if not tiles:
                continue
            view = np.hstack(tiles)
            s = chk.summary()
            txt = (f"pts {chk.n_points}  dist_rms "
                   f"{s.get('error_distancia_rms_mm', float('nan')):.2f} mm  "
                   f"planaridad {chk.planarity_rms_mm:.2f} mm  "
                   f"reproj {chk.reproj_rms_px:.2f} px")
            cv2.putText(view, txt, (12, view.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("validacion en vivo", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
