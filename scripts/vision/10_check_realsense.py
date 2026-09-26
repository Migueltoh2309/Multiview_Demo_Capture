#!/usr/bin/env python3
"""Etapa V0: diagnostico de la conexion de la RealSense.

Ejecutar esto SIEMPRE antes que nada. Comprueba, en orden:

  1. que el SDK esta instalado;
  2. que la camara enumera;
  3. que el enlace es USB 3.x  <-- la causa numero uno de fallos;
  4. que el firmware es el recomendado;
  5. que la camara SOSTIENE el streaming, no solo que arranca.

El punto 5 es el que importa: en USB 2 la camara enumera, expone sus perfiles
y arranca el pipeline sin error --- y luego entrega unos pocos frames y se
detiene. Un test que solo compruebe `pipeline.start()` da un falso OK.

Uso:
    python scripts/vision/10_check_realsense.py
    python scripts/vision/10_check_realsense.py --allow-usb2   # forzar prueba
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from tesis_picking.common import paths
from tesis_picking.common.logging_utils import setup
from tesis_picking.vision.realsense import (RS_AVAILABLE, RealSenseCamera,
                                            RealSenseConfig, RealSenseNotAvailable,
                                            describe_devices)

AYUDA_USB = """
COMO ARREGLAR UN ENLACE USB 2:

  1. EL CABLE es la causa mas frecuente. La D435i necesita el cable USB-C ->
     USB-A de USB 3 que viene en la caja. Un cable de carga de movil, o
     cualquier USB-C generico, enumera la camara pero no sostiene el streaming.
     Los cables USB 3 llevan mas conductores; por fuera no se distinguen.
  2. EL PUERTO. Usar un puerto azul o marcado SS (SuperSpeed). En este equipo:
       lsusb -t     -> los buses a 5000M/10000M son USB 3
  3. NADA DE HUBS USB 2 ni alargadores pasivos largos.
  4. Comprobar despues:  rs-enumerate-devices | grep "Usb Type"
     Debe decir 3.1 o 3.2, nunca 2.1.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-usb2", action="store_true",
                    help="intentar streaming aunque el enlace sea USB 2")
    ap.add_argument("--frames", type=int, default=60,
                    help="frames a sostener para dar el enlace por bueno")
    ap.add_argument("--save", action="store_true",
                    help="guardar el reporte y los intrinsecos en calib/")
    args = ap.parse_args()
    setup()

    print("\n=== 1. SDK ===")
    if not RS_AVAILABLE:
        print("  FALLO: pyrealsense2 no esta instalado.")
        print("  Solucion:  source env.sh && pip install pyrealsense2")
        return 1
    import pyrealsense2 as rs
    print(f"  OK  pyrealsense2 {rs.__version__}")

    print("\n=== 2. Dispositivos ===")
    devices = describe_devices()
    if not devices:
        print("  FALLO: ninguna RealSense detectada.")
        print("  Comprobar:  lsusb | grep 8086")
        print(AYUDA_USB)
        return 1
    for d in devices:
        print(f"  {d['name']}  serial {d['serial']}")
        print(f"    firmware   {d['firmware']}  (recomendado {d['firmware_recomendado']})")
        print(f"    enlace USB {d['usb_type']}")

    dev = devices[0]

    print("\n=== 3. Enlace USB ===")
    if dev["usb3"]:
        print(f"  OK  USB {dev['usb_type']}")
    else:
        print(f"  FALLO: USB {dev['usb_type']} --- se necesita 3.x")
        print(AYUDA_USB)
        if not args.allow_usb2:
            print("  (usar --allow-usb2 para intentar el streaming igualmente)")
            return 1

    print("\n=== 4. Firmware ===")
    if dev["firmware"] == dev["firmware_recomendado"]:
        print("  OK  coincide con el recomendado")
    else:
        print(f"  AVISO  instalado {dev['firmware']} != recomendado "
              f"{dev['firmware_recomendado']}")
        print("  Normalmente no impide trabajar. Actualizar con rs-fw-update "
              "solo si aparecen fallos de streaming CON un cable USB 3 bueno.")

    print(f"\n=== 5. Streaming sostenido ({args.frames} frames) ===")
    cfg = RealSenseConfig.load()
    try:
        with RealSenseCamera(cfg, allow_usb2=args.allow_usb2) as cam:
            info = cam.info()
            print(f"  perfil     : {cam.intr.image_size[0]}x{cam.intr.image_size[1]}")
            print(f"  depth scale: {cam.depth_scale:.6f} m/unidad")
            print(f"  filtros    : {', '.join(info['filtros']) or '(ninguno)'}")
            print(f"  intrinsecos: fx={cam.intr.fx:.2f} fy={cam.intr.fy:.2f} "
                  f"cx={cam.intr.cx:.2f} cy={cam.intr.cy:.2f}")

            t0 = time.perf_counter()
            n = 0
            cobertura = []
            for fr in cam.stream(n=args.frames, timeout_ms=4000):
                n += 1
                if n > 10:
                    cobertura.append(fr.valid_depth_ratio)
            dt = time.perf_counter() - t0

            if n < args.frames:
                print(f"  FALLO: solo {n}/{args.frames} frames. El enlace no "
                      "sostiene el streaming.")
                print(AYUDA_USB)
                return 1

            fps = (n - 1) / dt if dt > 0 else float("nan")
            cob = 100 * (sum(cobertura) / len(cobertura)) if cobertura else float("nan")
            print(f"  OK  {n} frames a {fps:.1f} FPS")
            print(f"  profundidad valida: {cob:.1f}% de los pixeles")
            if cob < 60:
                print("  AVISO: cobertura baja. Suele ser superficie sin textura,")
                print("  luz solar directa (satura el IR), o emisor apagado.")

            if args.save:
                p = cam.save_calibration()
                paths.ensure(paths.METRICAS)
                rep = {"dispositivos": devices, "camara": info,
                       "fps_medidos": round(fps, 2),
                       "cobertura_profundidad_pct": round(cob, 1)}
                q = paths.METRICAS / "realsense_check.json"
                q.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
                print(f"\n  intrinsecos -> {p}")
                print(f"  reporte     -> {q}")

    except RealSenseNotAvailable as exc:
        print(f"  FALLO: {exc}")
        return 1

    print("\nTODO OK. Siguiente: scripts/vision/12_ajustar_color.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
