#!/usr/bin/env python3
"""Etapa V5: caracterizacion metrologica del sensor de profundidad.

El propio marco teorico de la tesis lo exige: "los errores de localizacion
RGB-D reportados abarcan un rango amplio --- de pocos milimetros a mas de
veinte --- segun el sensor, la oclusion y las condiciones de iluminacion, lo
que obliga a caracterizar el error en el montaje especifico antes de
dimensionar la tolerancia de agarre".

Metodo: verdad de terreno sin instrumentos adicionales
------------------------------------------------------
El tablero ChArUco tiene geometria conocida. Su pose se resuelve por PnP a
partir de la imagen de COLOR, usando solo las esquinas y los intrinsecos de
fabrica --- sin tocar el mapa de profundidad. Eso da un plano de referencia
metricamente exacto e INDEPENDIENTE del sensor estereo.

Comparando la profundidad medida contra ese plano se separan tres cosas que
normalmente se confunden en una sola cifra:

    sesgo      : diferencia sistematica entre la profundidad medida y la real.
                 Es lo que hay que compensar; suele crecer con la distancia.
    ruido      : dispersion alrededor del plano ajustado. Es lo que limita la
                 repetibilidad y lo que se puede promediar.
    planaridad : curvatura residual del plano medido. Delata errores de la
                 calibracion estereo de fabrica.

Uso:
    # colocar el tablero a una distancia, medir, repetir a varias distancias
    python scripts/vision/15_exactitud_profundidad.py --distancia 0.6
    python scripts/vision/15_exactitud_profundidad.py --distancia 0.9
    python scripts/vision/15_exactitud_profundidad.py --resumen

El resultado de este script es una TABLA DE ERROR frente a la distancia. De ahi
salen dos numeros que la tesis necesita: hasta que distancia el error cabe en la
tolerancia de agarre, y que sesgo hay que compensar en el rango de trabajo.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import cv2
import numpy as np

from tesis_picking.common import paths
from tesis_picking.common.config import load_charuco
from tesis_picking.common.logging_utils import setup
from tesis_picking.common.transforms import rvec_tvec_to_T
from tesis_picking.demostraciones_humanas.calibration.charuco import CharucoBoardWrapper
from tesis_picking.vision.realsense import (RealSenseCamera, RealSenseConfig,
                                            RealSenseNotAvailable)


def plano_del_tablero(T_cam_board: np.ndarray) -> tuple[np.ndarray, float]:
    """Plano del tablero en el frame de la camara: (normal unitaria, d).

    Puntos del plano cumplen  n . X + d = 0.  El tablero es Z=0 en su propio
    frame, asi que su normal es la tercera columna de la rotacion.
    """
    n = T_cam_board[:3, 2]
    p0 = T_cam_board[:3, 3]
    n = n / np.linalg.norm(n)
    return n, float(-n @ p0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distancia", type=float, default=None,
                    help="distancia nominal a la que se coloco el tablero, en m "
                         "(solo etiqueta; la real la mide el PnP)")
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--min-corners", type=int, default=12)
    ap.add_argument("--allow-usb2", action="store_true")
    ap.add_argument("--sin-ventana", action="store_true")
    ap.add_argument("--resumen", action="store_true",
                    help="solo mostrar la tabla acumulada de medidas previas")
    args = ap.parse_args()
    setup()

    paths.ensure(paths.METRICAS)
    acumulado_path = paths.METRICAS / "exactitud_profundidad.json"
    historico = (json.loads(acumulado_path.read_text())
                 if acumulado_path.exists() else [])

    if args.resumen:
        return mostrar_resumen(historico)

    board = CharucoBoardWrapper(load_charuco())
    try:
        ctx = RealSenseCamera(RealSenseConfig.load(), allow_usb2=args.allow_usb2)
    except RealSenseNotAvailable as exc:
        print(f"ERROR: {exc}")
        return 1

    muestras: list[dict] = []

    with ctx as cam:
        K, D = cam.intr.K, cam.intr.D
        print(f"\ncamara {cam.serial}, filtros: {cam.info()['filtros']}")
        print(f"recogiendo {args.frames} frames con el tablero QUIETO...\n")

        for fr in cam.stream():
            det = board.detect(fr.color)
            if det.n_corners < args.min_corners:
                if not args.sin_ventana:
                    _mostrar(board, fr, det, len(muestras), args.frames, False)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                continue

            pose = board.estimate_pose(det, K, D, min_corners=args.min_corners)
            if pose is None:
                continue
            T_cam_board = rvec_tvec_to_T(*pose)
            n_plano, d_plano = plano_del_tablero(T_cam_board)

            # Region de interes: el interior del casco convexo de las esquinas
            # ChArUco, erosionado para no coger el borde de la hoja.
            esquinas = det.corners_flat().astype(np.int32)
            mask = np.zeros(fr.depth_m.shape, np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(esquinas), 255)
            mask = cv2.erode(mask, np.ones((9, 9), np.uint8))

            vs, us = np.nonzero(mask)
            z = fr.depth_m[vs, us]
            ok = z > 0
            vs, us, z = vs[ok], us[ok], z[ok]
            if z.size < 200:
                continue

            # Puntos medidos por el sensor
            X = np.column_stack([(us - cam.intr.cx) * z / cam.intr.fx,
                                 (vs - cam.intr.cy) * z / cam.intr.fy, z])
            # Distancia con signo al plano de referencia del ChArUco
            residuo = (X @ n_plano + d_plano)
            # Profundidad REAL en cada pixel: interseccion del rayo con el plano
            dirs = np.column_stack([(us - cam.intr.cx) / cam.intr.fx,
                                    (vs - cam.intr.cy) / cam.intr.fy,
                                    np.ones_like(z)])
            denom = dirs @ n_plano
            t_real = np.where(np.abs(denom) > 1e-9, -d_plano / denom, np.nan)
            z_real = t_real                     # la componente Z del rayo es 1

            valido = np.isfinite(z_real) & (z_real > 0)
            dz = (z[valido] - z_real[valido])

            cobertura = float(z.size / max(np.count_nonzero(mask), 1))
            muestras.append({
                "distancia_pnp_m": float(np.linalg.norm(T_cam_board[:3, 3])),
                "z_media_real_m": float(np.mean(z_real[valido])),
                "sesgo_mm": float(np.mean(dz) * 1000),
                "ruido_rms_mm": float(np.std(dz) * 1000),
                "planaridad_rms_mm": float(np.sqrt(np.mean(residuo ** 2)) * 1000),
                "cobertura": cobertura,
                "n_puntos": int(z.size),
                "n_esquinas": int(det.n_corners),
            })

            if not args.sin_ventana:
                _mostrar(board, fr, det, len(muestras), args.frames, True,
                         muestras[-1])
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            if len(muestras) >= args.frames:
                break
        cv2.destroyAllWindows()

    if not muestras:
        print("ERROR: ninguna medida valida. El tablero debe verse entero y con "
              "profundidad encima (no reflejando la luz).")
        return 1

    def agg(k):
        v = np.array([m[k] for m in muestras], float)
        return {"media": round(float(v.mean()), 3), "std": round(float(v.std()), 3)}

    z_real = float(np.mean([m["z_media_real_m"] for m in muestras]))
    entrada = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "distancia_nominal_m": args.distancia,
        "distancia_real_m": round(z_real, 4),
        "n_frames": len(muestras),
        "sesgo_mm": agg("sesgo_mm"),
        "ruido_rms_mm": agg("ruido_rms_mm"),
        "planaridad_rms_mm": agg("planaridad_rms_mm"),
        "cobertura": agg("cobertura"),
        "n_puntos_medio": int(np.mean([m["n_puntos"] for m in muestras])),
    }

    print(f"\n=== Distancia real (PnP): {z_real:.4f} m, {len(muestras)} frames ===")
    print(f"  sesgo       {entrada['sesgo_mm']['media']:+7.2f} mm  "
          f"(std {entrada['sesgo_mm']['std']:.2f})")
    print(f"    -> la profundidad medida esta "
          f"{'MAS LEJOS' if entrada['sesgo_mm']['media'] > 0 else 'MAS CERCA'} "
          "de lo real; es sistematico y se puede compensar")
    print(f"  ruido RMS   {entrada['ruido_rms_mm']['media']:7.2f} mm  "
          "-> se reduce promediando frames si el objeto esta quieto")
    print(f"  planaridad  {entrada['planaridad_rms_mm']['media']:7.2f} mm")
    print(f"  cobertura   {entrada['cobertura']['media']*100:6.1f}% de la region")
    print(f"  error relativo: {abs(entrada['sesgo_mm']['media'])/1000/z_real*100:.2f}% "
          "de la distancia")

    historico.append(entrada)
    acumulado_path.write_text(json.dumps(historico, indent=2, ensure_ascii=False))
    print(f"\n  -> {acumulado_path}")
    mostrar_resumen(historico)
    return 0


def _mostrar(board, fr, det, n, total, ok, m=None):
    vis = board.draw(fr.color, det)
    col = (0, 255, 0) if ok else (0, 0, 255)
    cv2.putText(vis, f"esquinas {det.n_corners}   muestras {n}/{total}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    if m:
        cv2.putText(vis, f"d={m['distancia_pnp_m']:.3f}m  sesgo={m['sesgo_mm']:+.1f}mm  "
                         f"ruido={m['ruido_rms_mm']:.1f}mm",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
    cv2.imshow("exactitud de profundidad", vis)


def mostrar_resumen(historico: list[dict]) -> int:
    if not historico:
        print("todavia no hay medidas. Ejecutar sin --resumen a varias distancias.")
        return 0
    print(f"\n{'d real (m)':>11} {'sesgo (mm)':>12} {'ruido (mm)':>12} "
          f"{'planar (mm)':>12} {'cobertura':>10} {'rel (%)':>9}")
    print("-" * 72)
    for e in sorted(historico, key=lambda x: x["distancia_real_m"]):
        d = e["distancia_real_m"]
        rel = abs(e["sesgo_mm"]["media"]) / 1000 / d * 100 if d > 0 else float("nan")
        print(f"{d:11.3f} {e['sesgo_mm']['media']:+12.2f} "
              f"{e['ruido_rms_mm']['media']:12.2f} "
              f"{e['planaridad_rms_mm']['media']:12.2f} "
              f"{e['cobertura']['media']*100:9.1f}% {rel:8.2f}%")
    if len(historico) >= 3:
        d = np.array([e["distancia_real_m"] for e in historico])
        s = np.array([e["sesgo_mm"]["media"] for e in historico])
        r = np.array([e["ruido_rms_mm"]["media"] for e in historico])
        # El ruido estereo crece con z^2; el ajuste da el coeficiente del montaje
        c = float(np.sum(r * d**2) / np.sum(d**4)) if np.any(d) else float("nan")
        print(f"\n  ajuste del ruido: sigma(z) ~ {c:.3f} * z^2  mm  (z en m)")
        print(f"  sesgo: pendiente {np.polyfit(d, s, 1)[0]:+.2f} mm/m")
        print("\n  Estos dos coeficientes son el resultado que hay que llevar a la")
        print("  tesis y contra el que se dimensiona la tolerancia de agarre.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
