#!/usr/bin/env python3
"""Etapas 8-13: MediaPipe Pose por camara + triangulacion en vivo del skeleton 3D.

Uso:
    python scripts/09_pose_triangulate.py                        # solo vivo, no graba
    python scripts/09_pose_triangulate.py --record-seconds 15     # graba 15 s
    python scripts/09_pose_triangulate.py --cams cam1 cam2 cam3 --max-reproj-px 20

Ventana con mosaico de las camaras (landmarks 2D superpuestos) + un panel con
el skeleton 3D reconstruido en {W}, junto a la posicion de las camaras para
dar contexto espacial. `q` para salir (o para cortar la grabacion antes de
tiempo, sin perder lo ya capturado).

Con `--record-seconds`, ademas de mostrar la ventana en vivo, guarda en
`data/raw/pose_sessions/<--out>/`:
    mosaico.mp4       las 3 camaras + panel 3D, tal como se ve en vivo
    skeleton_3d.mp4   solo el panel 3D, mas grande y rotando lentamente
    trayectoria.csv   x,y,z + validez + n_camaras + error de reproyeccion
                      por landmark y por frame (formato roadmap, seccion 39)
    metadata.yaml     fps real, duracion, camaras usadas, parametros

El FPS de los video se calcula DESPUES de grabar, a partir del tiempo real
transcurrido (correr MediaPipe en 3 camaras a la vez no llega a tiempo real
en este equipo -- ver docs/bitacora/2026-09-16) -- igual que en
temporal/mediapipe_demo.py, para que la duracion de reproduccion sea correcta.

Requiere `calib/<cam>.yaml` con intrinseca Y extrinseca (checkpoints 3 y 4).
Si el checkpoint 4 no paso limpio (ver docs/bitacora/2026-09-22 y 24), los
puntos igual se triangulan pero con `--max-reproj-px` mas permisivo que el
default de `common.triangulation` -- el error 3D resultante no es de
produccion, es para ver la geometria funcionando end-to-end.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")   # sin esto, pyplot pelea por el mismo event loop que cv2.waitKey
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tesis_picking.common import paths
from tesis_picking.common.camera_model import CameraRig
from tesis_picking.common.config import load_cameras
from tesis_picking.common.logging_utils import setup
from tesis_picking.demostraciones_humanas.capture.multi_capture import SyncedRig
from tesis_picking.demostraciones_humanas.pose.detector import (
    LANDMARK_INDEX, SKELETON_EDGES, TORSO_QUAD, PoseDetector,
)
from tesis_picking.demostraciones_humanas.pose.skeleton import triangulate_skeleton

TILE_WIDTH = 480
PLOT_SIZE_PX = 480
VIDEO_PLOT_SIZE_PX = 720
# grados/s en vez de grados/frame: el fps real varia mucho (1.7-2.5 fps segun
# la sesion), y con un incremento fijo por frame una toma corta apenas gira
# unos grados -- no alcanza a mostrar el torso desde un angulo bueno. Con una
# tasa de tiempo real, una toma de 15 s siempre completa una vuelta.
ROTATION_DEG_PER_S = 24.0


def draw_landmarks_2d(frame, lms: dict, min_visibility: float) -> np.ndarray:
    vis = frame.copy()
    for a, b in SKELETON_EDGES:
        la, lb = lms.get(a), lms.get(b)
        if la and lb and la.visibility >= min_visibility and lb.visibility >= min_visibility:
            cv2.line(vis, (int(la.u), int(la.v)), (int(lb.u), int(lb.v)),
                      (255, 255, 255), 2, cv2.LINE_AA)
    for name, lm in lms.items():
        color = (0, 255, 0) if lm.visibility >= min_visibility else (0, 140, 255)
        cv2.circle(vis, (int(lm.u), int(lm.v)), 5, color, -1, cv2.LINE_AA)
    return vis


def mosaic(tiles: list[np.ndarray], width: int = TILE_WIDTH) -> np.ndarray:
    resized = []
    for t in tiles:
        h, w = t.shape[:2]
        resized.append(cv2.resize(t, (width, int(h * width / w))))
    max_h = max(t.shape[0] for t in resized)
    resized = [cv2.copyMakeBorder(t, 0, max_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
               for t in resized]
    return cv2.hconcat(resized)


def render_skeleton_3d(fig, ax, points: dict[str, np.ndarray], rig: CameraRig,
                        center: np.ndarray, radius: float, *,
                        elev: float = 15.0, azim: float = -60.0) -> np.ndarray:
    ax.clear()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"skeleton 3D en {{W}}  ({len(points)}/{len(LANDMARK_INDEX)} puntos)")

    cams = np.array([c.center_world for c in rig])
    ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], c="tab:blue", marker="^", s=60)
    for c in rig:
        ax.text(*c.center_world, c.cam_id, color="tab:blue", fontsize=8)

    if points:
        pts = np.array(list(points.values()))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="tab:red", s=40)
        for a, b in SKELETON_EDGES:
            if a in points and b in points:
                seg = np.stack([points[a], points[b]])
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c="tab:red", linewidth=2)
        if all(name in points for name in TORSO_QUAD):
            verts = [[points[name] for name in TORSO_QUAD]]
            torso = Poly3DCollection(verts, facecolor="tab:red", alpha=0.35,
                                      edgecolor="darkred", linewidth=1.5)
            ax.add_collection3d(torso)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def write_outputs(out_dir: Path, mosaic_frames: list, plot_frames: list,
                   traj_rows: list[dict], real_elapsed_s: float,
                   cam_ids: list[str], args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(mosaic_frames)
    fps_real = max(n / real_elapsed_s, 1.0) if n else 1.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if mosaic_frames:
        h, w = mosaic_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(out_dir / "mosaico.mp4"), fourcc, fps_real, (w, h))
        for f in mosaic_frames:
            writer.write(f)
        writer.release()

    if plot_frames:
        h, w = plot_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(out_dir / "skeleton_3d.mp4"), fourcc, fps_real, (w, h))
        for f in plot_frames:
            writer.write(f)
        writer.release()

    cols = ["timestamp"]
    for name in LANDMARK_INDEX:
        cols += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_valid",
                 f"{name}_n_cameras", f"{name}_reproj_rms_px"]
    with open(out_dir / "trayectoria.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in traj_rows:
            w.writerow(row)

    meta = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camaras": cam_ids,
        "n_frames": n,
        "duracion_real_s": round(real_elapsed_s, 2),
        "fps_real": round(fps_real, 2),
        "min_visibility": args.min_visibility,
        "min_cameras": args.min_cameras,
        "max_reproj_px": args.max_reproj_px,
        "nota": "checkpoint 4 (extrinseca) sin cerrar limpio a esta fecha -- "
                "ver docs/bitacora/2026-09-24_recalibracion_a3.md. Error 3D "
                "esperado ~1-2 mm relativo segun checkpoints 5-6, no de "
                "produccion.",
    }
    (out_dir / "metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"\nGrabacion guardada -> {out_dir}")
    print(f"  {n} frames @ {fps_real:.1f} fps reales ({real_elapsed_s:.1f} s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--min-visibility", type=float, default=0.5)
    ap.add_argument("--min-cameras", type=int, default=2)
    ap.add_argument("--max-reproj-px", type=float, default=15.0,
                    help="mas permisivo que el default de triangulation.py "
                         "mientras el checkpoint 4 no pase limpio")
    ap.add_argument("--model-complexity", type=int, default=2, choices=[0, 1, 2],
                    help="complejidad del modelo MediaPipe Pose (0=liviano, "
                         "2=pesado/mas preciso). Con model_complexity=1 (default "
                         "viejo) se vieron huecos de cobertura en codo/muneca "
                         "derechos -- ver docs/bitacora/2026-09-24. Mas pesado "
                         "baja el fps, aceptable porque esto es captura offline, "
                         "no control en vivo del robot.")
    ap.add_argument("--plot-radius-m", type=float, default=1.2)
    ap.add_argument("--record-seconds", type=float, default=None,
                    help="si se indica, graba esa duracion a data/raw/pose_sessions/<--out>/")
    ap.add_argument("--preview-seconds", type=float, default=0.0,
                    help="cuenta regresiva en vivo (sin grabar) antes de empezar a grabar, "
                         "para ubicarte")
    ap.add_argument("--out", default=None, help="nombre de la sesion (default: timestamp)")
    args = ap.parse_args()
    setup()

    cfg = load_cameras().subset(args.cams)
    rig = CameraRig.load(paths.CALIB, cfg.ids)
    if not rig.calibrated:
        missing = [c.cam_id for c in rig if c.T_world_cam is None]
        print(f"ERROR: sin extrinseca para {missing}. Ejecutar 07_calibrate_extrinsics.py")
        return 1

    recording_enabled = args.record_seconds is not None
    out_dir = paths.RAW / "pose_sessions" / (args.out or time.strftime("%Y%m%d_%H%M%S"))

    center = np.mean([c.center_world for c in rig], axis=0)
    detectors = {cid: PoseDetector(model_complexity=args.model_complexity) for cid in cfg.ids}

    fig = plt.figure(figsize=(PLOT_SIZE_PX / 100, PLOT_SIZE_PX / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig_rec = plt.figure(figsize=(VIDEO_PLOT_SIZE_PX / 100, VIDEO_PLOT_SIZE_PX / 100), dpi=100)
    ax_rec = fig_rec.add_subplot(111, projection="3d")

    cv2.namedWindow("pose 3D (etapa 8-13)", cv2.WINDOW_NORMAL)
    if recording_enabled:
        if args.preview_seconds > 0:
            print(f"q = salir\n{args.preview_seconds:.0f} s para ubicarte, "
                  f"luego graba {args.record_seconds:.0f} s -> {out_dir}\n")
        else:
            print(f"q = salir (corta la grabacion, guarda lo ya capturado)\n"
                  f"grabando hasta {args.record_seconds:.0f} s -> {out_dir}\n")
    else:
        print("q = salir\n")

    mosaic_frames: list[np.ndarray] = []
    plot_frames: list[np.ndarray] = []
    traj_rows: list[dict] = []
    t_start = time.perf_counter()
    t_record_start: float | None = None

    try:
        with SyncedRig(cfg.cameras) as srig:
            srig.warmup(15)
            while True:
                fs = srig.capture()
                detections = {}
                tiles = []
                for cid in cfg.ids:
                    frame = fs.frames.get(cid)
                    if frame is None:
                        continue
                    lms = detectors[cid].detect(frame)
                    detections[cid] = lms
                    vis = draw_landmarks_2d(frame, lms, args.min_visibility)
                    cv2.putText(vis, cid, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 255), 2, cv2.LINE_AA)
                    tiles.append(vis)

                skeleton = triangulate_skeleton(
                    detections, rig, min_visibility=args.min_visibility,
                    min_cameras=args.min_cameras, max_reproj_px=args.max_reproj_px)
                points = {n: t.X for n, t in skeleton.items() if t.valid}

                azim_live = (-60.0 + ROTATION_DEG_PER_S * (time.perf_counter() - t_start)) % 360
                plot_img = render_skeleton_3d(fig, ax, points, rig, center,
                                              args.plot_radius_m, azim=azim_live)
                plot_img = cv2.resize(plot_img, (TILE_WIDTH, TILE_WIDTH))
                view = mosaic(tiles + [plot_img])

                if recording_enabled:
                    since_start = time.perf_counter() - t_start
                    in_preview = since_start < args.preview_seconds

                    if in_preview:
                        remaining = args.preview_seconds - since_start
                        cv2.putText(view, f"empieza en {remaining:4.1f}s",
                                    (10, view.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 200, 255), 2, cv2.LINE_AA)
                    else:
                        if t_record_start is None:
                            t_record_start = time.perf_counter()
                        elapsed = time.perf_counter() - t_record_start
                        cv2.putText(view, f"REC {elapsed:5.1f}/{args.record_seconds:.0f}s",
                                    (10, view.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        mosaic_frames.append(view.copy())
                        azim = (-60.0 + ROTATION_DEG_PER_S * elapsed) % 360
                        plot_frames.append(render_skeleton_3d(
                            fig_rec, ax_rec, points, rig, center, args.plot_radius_m, azim=azim))
                        row = {"timestamp": round(elapsed, 4)}
                        for name in LANDMARK_INDEX:
                            t = skeleton[name]
                            x, y, z = (t.X.tolist() if t.valid else [float("nan")] * 3)
                            row[f"{name}_x"] = x
                            row[f"{name}_y"] = y
                            row[f"{name}_z"] = z
                            row[f"{name}_valid"] = t.valid
                            row[f"{name}_n_cameras"] = t.n_cameras
                            row[f"{name}_reproj_rms_px"] = (
                                round(t.rms_reproj_px, 3) if t.valid else "")
                        traj_rows.append(row)
                        if elapsed >= args.record_seconds:
                            break

                cv2.imshow("pose 3D (etapa 8-13)", view)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        for d in detectors.values():
            d.close()
        plt.close(fig)
        plt.close(fig_rec)
        cv2.destroyAllWindows()

    if recording_enabled and mosaic_frames and t_record_start is not None:
        write_outputs(out_dir, mosaic_frames, plot_frames, traj_rows,
                      time.perf_counter() - t_record_start, cfg.ids, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
