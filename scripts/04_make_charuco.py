#!/usr/bin/env python3
"""Etapa 4: generar el tablero ChArUco para imprimir.

Uso:
    python scripts/04_make_charuco.py
    python scripts/04_make_charuco.py --dpi 600 --out docs/charuco_A3.png

IMPRESION --- lo que arruina una calibracion, en orden de frecuencia:
  1. Imprimir con "ajustar a pagina" activado: cambia la escala y el
     square_length_m del YAML deja de ser cierto. Imprimir SIEMPRE al 100%.
  2. Dejar la hoja suelta o pegada a algo que se curva. Montar sobre carton
     pluma o MDF; el tablero debe quedar PLANO.
  3. No volver a medir. Tras imprimir, medir con calibrador 5 casillas seguidas,
     dividir entre 5, y actualizar square_length_m en config/charuco.yaml. Ese
     numero fija la ESCALA de todas las reconstrucciones 3D del sistema.
"""
from __future__ import annotations

import argparse
import sys

import yaml

from tesis_picking.common import paths
from tesis_picking.common.config import load_charuco
from tesis_picking.common.logging_utils import setup
from tesis_picking.demostraciones_humanas.calibration.charuco import CharucoBoardWrapper


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--margin-mm", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    setup()

    cfg = load_charuco()
    board = CharucoBoardWrapper(cfg)
    out = paths.DOCS / f"charuco_{cfg.squares_x}x{cfg.squares_y}_{int(cfg.square_length_m*1000)}mm.png"
    if args.out:
        out = paths.ROOT / args.out

    path, info = board.generate_image(out, dpi=args.dpi, margin_mm=args.margin_mm)

    print("\n=== Tablero ChArUco generado ===")
    for k, v in info.items():
        print(f"  {k:<22} {v}")
    w_mm, h_mm = info["board_size_mm"]
    def fits(w, h, pw, ph):
        """Cabe en la hoja, probando tambien en apaisado."""
        return (w <= pw and h <= ph) or (w <= ph and h <= pw)

    print(f"\n  Tamano fisico a imprimir: {w_mm:.1f} x {h_mm:.1f} mm")
    print("  (area util A4 ~ 190x277 mm | A3 ~ 287x400 mm)")
    if fits(w_mm, h_mm, 190, 277):
        print("  Cabe en A4. Aun asi, un tablero A3 mejora la calibracion:")
        print("  mas area de tablero = esquinas mas separadas = pose mejor condicionada.")
    elif fits(w_mm, h_mm, 287, 400):
        print("  Necesita A3.")
    else:
        print("  AVISO: no cabe en A3. Reducir squares_x/y o square_length_m,")
        print("  o imprimir por secciones (no recomendado: introduce errores de union).")

    (path.with_suffix(".yaml")).write_text(
        yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    print(f"\n  imagen  -> {path}")
    print(f"  ficha   -> {path.with_suffix('.yaml')}")
    print("\n  Imprimir al 100% (sin escalar), montar sobre superficie rigida,")
    print("  MEDIR una casilla con calibrador y actualizar config/charuco.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
