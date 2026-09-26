#!/usr/bin/env python3
"""Etapa V4: pose de la RealSense en el frame de trabajo {W}.

Sin esto, la posicion 3D de la fruta existe solo en el frame de la camara y no
sirve para mandar el efector del H1-2 a ningun sitio.

Metodo: se coloca el tablero ChArUco en la posicion que MATERIALIZA el origen
de {W} (sobre la mesa o la faja, marcado con cinta) y se resuelve la pose por
PnP usando los intrinsecos de fabrica de la camara. Como hay una sola camara y
un solo tablero, no hace falta ajuste de haces: basta promediar sobre varios
frames para reducir el ruido de deteccion.

    T_world_cam = T_world_board @ inv(T_cam_board)

`T_world_board` sale de config/world_frame.yaml: es la identidad si el tablero
se apoya alineado con los ejes de {W} (X = avance de la faja, Y transversal,
Z vertical hacia arriba).

Uso:
    python scripts/vision/14_extrinsecos_camara.py --frames 60
    python scripts/vision/14_extrinsecos_camara.py --frames 60 --verificar

REQUISITOS FISICOS (no negociables):
  * La camara NO se puede mover despues de esto. Montarla en tripode o soporte
    rigido y no tocarla. Si se mueve, hay que repetir la calibracion.
  * El tablero apoyado PLANO y rigido, en la posicion marcada del origen.
  * El tablero debe verse ENTERO y nitido; si el ChArUco no llega a ~15
    esquinas, la pose sale ruidosa.
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

from tesis_picking.common import paths
from tesis_picking.common.camera_model import CameraModel
from tesis_picking.common.config import load_charuco, load_world_frame
from tesis_picking.common.logging_utils import setup
from tesis_picking.common.transforms import (R_to_rpy_deg, angle_between_R_deg,
                                             average_transforms, invert_T,
                                             rvec_tvec_to_T, split_T,
                                             transform_points)
from tesis_picking.demostraciones_humanas.calibration.charuco import CharucoBoardWrapper
from tesis_picking.vision.realsense import (RealSenseCamera, RealSenseConfig,
                                            RealSenseNotAvailable)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=60,
                    help="frames a promediar con el tablero quieto")
    ap.add_argument("--min-corners", type=int, default=12)
    ap.add_argument("--allow-usb2", action="store_true")
    ap.add_argument("--verificar", action="store_true",
                    help="tras calibrar, comprueba que las esquinas del tablero "
                         "caen donde deben en {W}")
    ap.add_argument("--sin-ventana", action="store_true")
    args = ap.parse_args()
    setup()

    board = CharucoBoardWrapper(load_charuco())
    wf = load_world_frame()

    try:
        cam_ctx = RealSenseCamera(RealSenseConfig.load(), allow_usb2=args.allow_usb2)
    except RealSenseNotAvailable as exc:
        print(f"ERROR: {exc}")
        return 1

    poses: list[np.ndarray] = []
    n_esq: list[int] = []

    with cam_ctx as cam:
        cam_id = f"rs_{cam.serial}"
        K, D = cam.intr.K, cam.intr.D
        print(f"\ncamara {cam_id}  {cam.intr.image_size[0]}x{cam.intr.image_size[1]}")
        print(f"tablero {board.cfg.squares_x}x{board.cfg.squares_y}, "
              f"casilla {board.cfg.square_length_m*1000:.1f} mm, "
              f"{board.n_corners} esquinas")
        print(f"\nrecogiendo {args.frames} frames con el tablero QUIETO en el origen...")

        for fr in cam.stream():
            det = board.detect(fr.color)
            ok = det.n_corners >= args.min_corners
            if ok:
                pose = board.estimate_pose(det, K, D, min_corners=args.min_corners)
                if pose is not None:
                    poses.append(rvec_tvec_to_T(*pose))
                    n_esq.append(det.n_corners)

            if not args.sin_ventana:
                vis = board.draw(fr.color, det)
                col = (0, 255, 0) if ok else (0, 0, 255)
                cv2.putText(vis, f"esquinas {det.n_corners}/{board.n_corners}   "
                                 f"poses {len(poses)}/{args.frames}",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
                cv2.imshow("extrinsecos", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            if len(poses) >= args.frames:
                break
        cv2.destroyAllWindows()

        if len(poses) < max(5, args.frames // 4):
            print(f"\nERROR: solo {len(poses)} poses validas. El tablero no se ve "
                  "bien. Comprobar enfoque, iluminacion y que se vea entero.")
            return 1

        # Dispersion entre poses individuales: es la incertidumbre de la
        # calibracion, y hay que reportarla, no solo la media.
        T_cam_board = average_transforms(poses)
        t_all = np.array([p[:3, 3] for p in poses])
        sigma_t_mm = t_all.std(axis=0) * 1000
        ang = [angle_between_R_deg(T_cam_board[:3, :3], p[:3, :3]) for p in poses]

        T_world_cam = wf.T_world_board @ invert_T(T_cam_board)
        R, t = split_T(T_world_cam)

        print(f"\n=== Resultado ({len(poses)} poses, {np.mean(n_esq):.0f} esquinas de media) ===")
        print(f"  camara en {{W}}:  posicion [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m")
        print(f"                  rpy      {np.round(R_to_rpy_deg(R), 2)} deg")
        print(f"  distancia camara-origen: {np.linalg.norm(t):.4f} m")
        print(f"\n  dispersion entre frames:")
        print(f"    traslacion  [{sigma_t_mm[0]:.2f} {sigma_t_mm[1]:.2f} "
              f"{sigma_t_mm[2]:.2f}] mm")
        print(f"    rotacion    media {np.mean(ang):.3f} deg, max {np.max(ang):.3f} deg")

        avisos = []
        if np.linalg.norm(sigma_t_mm) > 5.0:
            avisos.append("dispersion de traslacion > 5 mm: el tablero o la camara "
                          "se movieron, o el tablero se ve demasiado pequeno/lejos")
        if np.mean(ang) > 0.5:
            avisos.append("dispersion de rotacion > 0.5 deg: pocas esquinas o "
                          "tablero casi frontal (mala condicion para PnP)")
        for a in avisos:
            print(f"  AVISO: {a}")

        # --- guardar sobre el mismo YAML que ya tiene los intrinsecos ---
        p = paths.CALIB / f"{cam_id}.yaml"
        modelo = CameraModel(cam_id=cam_id, intr=cam.intr, T_world_cam=T_world_cam)
        modelo.save(p)
        print(f"\n  -> {p}")

        reporte = {
            "cam_id": cam_id,
            "n_poses": len(poses),
            "esquinas_media": float(np.mean(n_esq)),
            "T_world_cam": T_world_cam.tolist(),
            "posicion_m": [round(float(v), 5) for v in t],
            "rpy_deg": [round(float(v), 3) for v in R_to_rpy_deg(R)],
            "sigma_traslacion_mm": [round(float(v), 3) for v in sigma_t_mm],
            "sigma_rotacion_deg": round(float(np.mean(ang)), 4),
            "avisos": avisos,
        }

        # --- verificacion opcional ---
        if args.verificar:
            print("\n=== Verificacion: esquinas del tablero reproyectadas a {W} ===")
            fr = cam.read()
            det = board.detect(fr.color) if fr else None
            if det and det.n_corners >= args.min_corners:
                obj = board.object_points()[det.ids_flat()]
                # Verdad de terreno: donde DEBEN estar las esquinas en {W}
                real_w = transform_points(wf.T_world_board, obj)
                # Estimacion: deproyectar con la profundidad medida y pasar a {W}
                est_w, usados = [], []
                for (u, v), Xr in zip(det.corners_flat(), real_w):
                    z = fr.depth_at(u, v)
                    if z <= 0:
                        continue
                    Xc = fr.deproject(u, v, z)
                    est_w.append(transform_points(T_world_cam, Xc))
                    usados.append(Xr)
                if len(est_w) >= 5:
                    E = np.vstack(est_w); Rr = np.vstack(usados)
                    err = np.linalg.norm(E - Rr, axis=1) * 1000
                    print(f"  {len(E)} esquinas con profundidad")
                    print(f"  error 3D: media {err.mean():.1f} mm  "
                          f"p95 {np.percentile(err, 95):.1f} mm  max {err.max():.1f} mm")
                    print(f"  error en Z (altura sobre el plano del tablero): "
                          f"{np.abs(E[:, 2] - Rr[:, 2]).mean()*1000:.1f} mm")
                    reporte["verificacion"] = {
                        "n_esquinas": int(len(E)),
                        "error_medio_mm": round(float(err.mean()), 2),
                        "error_p95_mm": round(float(np.percentile(err, 95)), 2),
                        "error_max_mm": round(float(err.max()), 2),
                    }
                    print("\n  Este error mezcla DOS fuentes: el de la extrinseca y")
                    print("  el del sensor de profundidad. Para separarlos, ejecutar")
                    print("  15_exactitud_profundidad.py, que usa la geometria del")
                    print("  tablero como referencia independiente de la profundidad.")
                else:
                    print("  sin profundidad suficiente sobre el tablero")

        paths.ensure(paths.METRICAS)
        q = paths.METRICAS / f"extrinsecos_{cam_id}.json"
        q.write_text(json.dumps(reporte, indent=2, ensure_ascii=False))
        print(f"  reporte -> {q}")

    print("\nA partir de ahora, NO MOVER LA CAMARA.")
    print(f"Uso:  python scripts/vision/13_detectar_y_localizar.py --world --cam-id {cam_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
