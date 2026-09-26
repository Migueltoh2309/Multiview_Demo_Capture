#!/usr/bin/env python3
"""Etapas V2-V3: detectar la mandarina y dar su posicion 3D.

Es el script central del subsistema de vision. Produce, para cada fruta:

    posicion del CENTRO en el frame de la camara {C}
    posicion del CENTRO en el frame de trabajo {W}   (si hay extrinsecos)
    radio estimado, calidad de la medida y motivo de rechazo si lo hay

Uso:
    # sin camara, sobre escena sintetica: sirve para validar la cadena hoy
    python scripts/vision/13_detectar_y_localizar.py --sintetico

    # con camara, posicion en {C} solamente
    python scripts/vision/13_detectar_y_localizar.py

    # con camara y extrinsecos ya calibrados: posicion en {W}
    python scripts/vision/13_detectar_y_localizar.py --world --cam-id rs_327122073685

    # medir una fruta QUIETA promediando N frames (etapa actual del proyecto)
    python scripts/vision/13_detectar_y_localizar.py --estatico 100 --guardar

Teclas: q = salir | s = guardar frame y medida | espacio = pausa
        r = reiniciar el enganche al objeto y la serie acumulada
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import cv2
import numpy as np

from tesis_picking.common import paths
from tesis_picking.common.config import load_world_frame
from tesis_picking.common.logging_utils import setup
from tesis_picking.vision.deteccion import DetectorColorHSV, ObjetoConfig
from tesis_picking.vision.localizacion import Localizador3D, TransformadorFrames


def fuente(args):
    """Genera RGBDFrame de la camara o del simulador."""
    if args.sintetico:
        from tesis_picking.vision.simulacion import (EscenaSintetica, EsferaSintetica,
                                                     intrinsecos_d435i)
        intr = intrinsecos_d435i(848, 480)
        esferas = [EsferaSintetica(np.array([-0.07, 0.02, 0.65]), 0.0325),
                   EsferaSintetica(np.array([0.07, -0.03, 0.78]), 0.030)]
        esc = EscenaSintetica(intr=intr, esferas=esferas, plano_z_m=1.4)
        print("  MODO SINTETICO. Posiciones reales de las esferas (frame camara):")
        for i, e in enumerate(esferas):
            print(f"    esfera {i}: {np.round(e.centro_cam, 4)} m  r={e.radio_m*1000:.1f} mm")
        k = 0
        while True:
            esc.seed = k; esc.__post_init__(); k += 1
            yield esc.render()
    else:
        from tesis_picking.vision.realsense import RealSenseCamera, RealSenseConfig
        with RealSenseCamera(RealSenseConfig.load(), allow_usb2=args.allow_usb2) as cam:
            print(f"  camara {cam.serial}, USB {cam.usb_type}, "
                  f"{cam.intr.image_size[0]}x{cam.intr.image_size[1]}")
            yield from cam.stream()


def seleccionar(objetos, enganche, max_salto_m):
    """Elige el objeto valido a medir, manteniendo la identidad entre frames."""
    validos = [o for o in objetos if o.valido]
    if not validos:
        return None
    if enganche is None:
        # Primer frame: la deteccion de mayor score (ya vienen ordenadas).
        return validos[0]
    d = [float(np.linalg.norm(o.centro_cam - enganche)) for o in validos]
    i = int(np.argmin(d))
    if d[i] > max_salto_m:
        # Salto imposible para un objeto quieto: se perdio el enganche.
        return None
    return validos[i]


def colorear_profundidad(depth_m, vmin=0.2, vmax=1.6):
    d = np.clip((depth_m - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    d = (d * 255).astype(np.uint8)
    vis = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
    vis[depth_m <= 0] = 0                 # huecos en negro, no en color falso
    return vis


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objeto", default="mandarina")
    ap.add_argument("--metodo", choices=["esfera", "centroide"], default="esfera")
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--allow-usb2", action="store_true")
    ap.add_argument("--world", action="store_true",
                    help="expresar la posicion en {W} (requiere extrinsecos)")
    ap.add_argument("--cam-id", default=None,
                    help="id de calibracion, p.ej. rs_327122073685")
    ap.add_argument("--estatico", type=int, default=0, metavar="N",
                    help="promediar N frames con el objeto QUIETO y reportar "
                         "media y desviacion: es la medida de repetibilidad")
    ap.add_argument("--guardar", action="store_true")
    ap.add_argument("--sin-ventana", action="store_true")
    ap.add_argument("--max-salto-m", type=float, default=0.05,
                    help="salto maximo admitido entre frames al seguir el objeto")
    args = ap.parse_args()
    setup()

    objeto = ObjetoConfig.load(args.objeto)
    detector = DetectorColorHSV(objeto)
    localizador = Localizador3D(objeto, metodo=args.metodo)

    tf = None
    if args.world:
        if not args.cam_id:
            print("ERROR: --world necesita --cam-id (ver calib/)")
            return 1
        try:
            tf = TransformadorFrames.desde_calib(args.cam_id, load_world_frame())
            print(f"  extrinsecos: {json.dumps(tf.describir(), ensure_ascii=False)}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1

    acumulado: list[np.ndarray] = []
    ultimo = None
    enganche: np.ndarray | None = None   # posicion del objeto seguido
    n = 0
    pausa = False
    dets: list = []
    objetos: list = []

    for fr in fuente(args):
        if not pausa:
            dets = detector.detectar(fr.color)
            objetos = localizador.localizar_todas(dets, fr.depth_m, fr.intr)
            n += 1

            # Seleccion del objeto a medir. Con varias frutas en escena, quedarse
            # con "la de mayor score" hace que la eleccion CAMBIE entre frames y
            # la serie mezcle dos frutas distintas: la sigma resultante seria la
            # separacion entre ellas, no el ruido del sensor. Se engancha a la
            # fruta mas proxima a la medida anterior --- un seguidor minimo, que
            # es tambien la semilla de la etapa V8 (ByteTrack).
            elegido = seleccionar(objetos, enganche, args.max_salto_m)
            if elegido is not None:
                ultimo = elegido
                enganche = elegido.centro_cam
                p = elegido.centro_cam
                if tf is not None:
                    p = tf.cam_a_world(p)
                acumulado.append(p)

        if not args.sin_ventana:
            vis = detector.dibujar(fr.color, dets)
            if ultimo is not None and ultimo.valido:
                cv2.putText(vis, f"distancia: {ultimo.distancia_m:.3f} m",
                            (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(vis, f"distancia: {ultimo.distancia_m:.3f} m",
                            (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 0), 2, cv2.LINE_AA)
            for i, ob in enumerate(objetos):
                x, y, w, h = ob.deteccion.bbox
                if ob.valido:
                    p = ob.centro_cam if tf is None else tf.cam_a_world(ob.centro_cam)
                    marco = "W" if tf is not None else "C"
                    txt = f"{marco}: {p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f} m"
                    col = (0, 255, 0)
                    cv2.putText(vis, f"r={ob.radio_m*1000:.0f}mm d={ob.distancia_m:.3f}m",
                                (x, y + h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 200, 0), 1, cv2.LINE_AA)
                else:
                    txt = ob.motivo[:42]
                    col = (0, 0, 255)
                cv2.putText(vis, txt, (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, col, 1, cv2.LINE_AA)

            dvis = colorear_profundidad(fr.depth_m)
            fila = np.hstack([vis, dvis])
            fila = cv2.resize(fila, None, fx=0.72, fy=0.72)
            cv2.putText(fila, f"frame {n}  detecciones {len(dets)}  "
                              f"profundidad valida {100*fr.valid_depth_ratio:.0f}%"
                              + ("  [PAUSA]" if pausa else ""),
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("deteccion y localizacion", fila)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                pausa = not pausa
            if k == ord("r"):
                enganche = None
                acumulado.clear()
                print("  enganche reiniciado")
            if k == ord("s") and ultimo is not None:
                guardar_muestra(fr, ultimo, tf, args)

        if args.estatico and len(acumulado) >= args.estatico:
            break

    cv2.destroyAllWindows()

    if ultimo is not None:
        print("\n=== ultima medida ===")
        print(json.dumps(ultimo.resumen(), indent=2, ensure_ascii=False))

    if acumulado:
        A = np.vstack(acumulado)
        media = A.mean(axis=0)
        sigma = A.std(axis=0)
        marco = "W" if tf is not None else "C"
        print(f"\n=== repetibilidad sobre {len(A)} medidas (frame {{{marco}}}) ===")
        print(f"  media  : [{media[0]:+.4f} {media[1]:+.4f} {media[2]:+.4f}] m")
        print(f"  sigma  : [{sigma[0]*1000:6.2f} {sigma[1]*1000:6.2f} "
              f"{sigma[2]*1000:6.2f}] mm")
        print(f"  sigma_3D: {np.linalg.norm(sigma)*1000:.2f} mm")
        print("\n  La sigma es la REPETIBILIDAD, no la exactitud: mide cuanto")
        print("  fluctua la medida, no si esta centrada en el valor real.")
        print("  Para exactitud hace falta verdad de terreno: usar")
        print("  scripts/vision/15_exactitud_profundidad.py")

        if args.guardar:
            paths.ensure(paths.METRICAS)
            out = paths.METRICAS / f"localizacion_{datetime.now():%Y%m%d_%H%M%S}.json"
            out.write_text(json.dumps({
                "frame": marco, "n": len(A), "metodo": args.metodo,
                "media_m": media.tolist(), "sigma_mm": (sigma * 1000).tolist(),
                "sigma_3d_mm": float(np.linalg.norm(sigma) * 1000),
                "ultima": ultimo.resumen() if ultimo else None,
                "objeto": objeto.to_dict(),
            }, indent=2, ensure_ascii=False))
            print(f"\n  -> {out}")
    else:
        print("\nNinguna localizacion valida. Revisar, en este orden:")
        print("  1. que el detector encuentre la fruta -> 12_ajustar_color.py")
        print("  2. que haya profundidad sobre la fruta (panel derecho no negro)")
        print("  3. el rango de distancias del localizador")
    return 0


def guardar_muestra(fr, obj, tf, args) -> None:
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    d = paths.DATA / "vision" / "muestras" / stamp
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "color.png"), fr.color)
    np.save(d / "depth_m.npy", fr.depth_m)
    r = obj.resumen()
    if tf is not None:
        r["centro_world_m"] = [round(float(v), 4)
                               for v in tf.cam_a_world(obj.centro_cam)]
    (d / "medida.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"  muestra -> {d}")


if __name__ == "__main__":
    sys.exit(main())
