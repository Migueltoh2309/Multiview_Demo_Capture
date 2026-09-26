#!/usr/bin/env python3
"""Etapas 5-6: capturar tomas del tablero ChArUco para calibrar.

Una sola sesion sirve para AMBAS calibraciones:
  * intrinseca: cada toma en que una camara ve el tablero es una vista valida
    para esa camara (script 06);
  * extrinseca: cada toma en que DOS O MAS camaras ven el tablero a la vez ata
    esas camaras al mismo {W} (script 08).

Uso:
    # 1) toma que ANCLA el origen: tablero en su posicion de referencia
    python scripts/05_capture_charuco.py --set calib_01 --anchor

    # 2) el resto: mover el tablero por todo el volumen y las tres vistas
    python scripts/05_capture_charuco.py --set calib_01 --auto --target 40

Teclas: espacio = capturar | a = marcar la toma como ancla | q = terminar

COMO MOVER EL TABLERO (esto determina la calidad de K y D):
  * llenar el encuadre: acercar el tablero hasta ocupar >1/3 de la imagen;
  * llegar a las ESQUINAS y BORDES del encuadre, no solo al centro --- la
    distorsion solo es observable lejos del centro optico;
  * inclinarlo +-30-45 grados en los dos ejes, no solo frontal --- las vistas
    frontales dejan fx/fy mal condicionadas;
  * mantenerlo quieto al disparar: el motion blur destruye la precision subpixel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tesis_picking.common import paths
from tesis_picking.common.config import load_cameras, load_charuco
from tesis_picking.common.logging_utils import setup
from tesis_picking.demostraciones_humanas.calibration.charuco import CharucoBoardWrapper
from tesis_picking.demostraciones_humanas.capture.multi_capture import SyncedRig


def blur_score(gray: np.ndarray) -> float:
    """Varianza del laplaciano: mas alto = mas nitido. Descarta motion blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pose_signature(det, board) -> np.ndarray | None:
    """Firma compacta de la vista: centroide + extension + area, normalizados.

    Se usa para exigir DIVERSIDAD entre tomas: dos vistas casi identicas no
    aportan informacion nueva a la calibracion, solo tiempo de computo.
    """
    pts = det.corners_flat()
    if len(pts) < 6:
        return None
    w, h = det.image_size
    c = pts.mean(axis=0) / np.array([w, h])
    ext = (pts.max(axis=0) - pts.min(axis=0)) / np.array([w, h])
    return np.concatenate([c, ext])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="nombre del set (p.ej. calib_01)")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--anchor", action="store_true",
                    help="la primera toma define el origen {W}")
    ap.add_argument("--auto", action="store_true",
                    help="capturar automaticamente al detectar una vista nueva y nitida")
    ap.add_argument("--target", type=int, default=40, help="tomas objetivo en modo auto")
    ap.add_argument("--min-corners", type=int, default=12)
    ap.add_argument("--min-blur", type=float, default=60.0)
    ap.add_argument("--min-novelty", type=float, default=0.10,
                    help="distancia minima a la firma de las tomas ya guardadas")
    args = ap.parse_args()
    setup()

    cfg = load_cameras().subset(args.cams)
    board = CharucoBoardWrapper(load_charuco())
    out_dir = paths.CALIB_SHOTS / args.set
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"shots": {}}
    shot_n = len(index["shots"])
    sigs: dict[str, list[np.ndarray]] = {c: [] for c in cfg.ids}
    pending_anchor = args.anchor

    print(f"\nset: {out_dir}   tomas ya existentes: {shot_n}")
    print("espacio = capturar | a = marcar ancla | q = terminar\n")

    with SyncedRig(cfg.cameras) as rig:
        rig.warmup(20)
        while True:
            fs = rig.capture()
            dets, tiles, quality = {}, [], {}
            for cid, frame in fs.frames.items():
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                det = board.detect(gray)
                dets[cid] = det
                b = blur_score(gray)
                sig = pose_signature(det, board)
                novel = (float("inf") if not sigs[cid] or sig is None
                         else float(min(np.linalg.norm(sig - s) for s in sigs[cid])))
                good = (det.n_corners >= args.min_corners and b >= args.min_blur
                        and novel >= args.min_novelty)
                quality[cid] = {"corners": det.n_corners, "blur": b,
                                "novelty": novel, "good": good}

                vis = board.draw(frame, det)
                color = (0, 255, 0) if good else (0, 165, 255)
                cv2.putText(vis, f"{cid}  {det.n_corners:3d} esq  nitidez {b:6.0f}",
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                tiles.append(cv2.resize(vis, (560, int(vis.shape[0] * 560 / vis.shape[1]))))

            if tiles:
                view = np.hstack(tiles)
                n_good = sum(q["good"] for q in quality.values())
                banner = (f"tomas: {shot_n}/{args.target}   camaras validas ahora: "
                          f"{n_good}/{len(cfg.ids)}")
                if pending_anchor:
                    banner += "   [ESPERANDO TOMA ANCLA]"
                cv2.putText(view, banner, (12, view.shape[0] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("charuco", view)

            k = cv2.waitKey(1) & 0xFF
            take = k == ord(" ")
            if args.auto and not take:
                # En auto, exige que al menos 2 camaras vean una vista nueva y
                # nitida: es la condicion que ata camaras entre si.
                take = sum(q["good"] for q in quality.values()) >= min(2, len(cfg.ids))
            if k == ord("a"):
                pending_anchor = True
                print("la proxima toma se marcara como ancla")
            if k == ord("q"):
                break

            if take and dets:
                key = f"shot_{shot_n:03d}"
                sd = out_dir / key
                sd.mkdir(exist_ok=True)
                saved = {}
                for cid, frame in fs.frames.items():
                    if frame is None:
                        continue
                    cv2.imwrite(str(sd / f"{cid}.png"), frame)
                    saved[cid] = {k2: (round(v, 2) if isinstance(v, float) else v)
                                  for k2, v in quality[cid].items()}
                    if quality[cid]["good"]:
                        s = pose_signature(dets[cid], board)
                        if s is not None:
                            sigs[cid].append(s)
                index["shots"][key] = {
                    "anchor": bool(pending_anchor),
                    "dt_max_ms": round(fs.spread_s * 1e3, 3),
                    "cameras": saved,
                }
                if pending_anchor:
                    index["anchor"] = key
                    print(f"  {key}  ANCLA (define el origen de {{W}})")
                    pending_anchor = False
                else:
                    print(f"  {key}  " + "  ".join(
                        f"{c}:{q['corners']}esq" for c, q in quality.items()))
                shot_n += 1
                index_path.write_text(json.dumps(index, indent=2))
                if args.auto and shot_n >= args.target:
                    print("\nobjetivo de tomas alcanzado")
                    break

    cv2.destroyAllWindows()
    index_path.write_text(json.dumps(index, indent=2))

    # Resumen de cobertura: cuantas vistas utiles tiene cada camara y cuantas
    # tomas atan cada par de camaras. Se basa en TODAS las camaras que
    # aparezcan en el set, no solo en --cams de esta sesion: el flujo normal
    # es ir agregando camaras una por una al mismo --set (una sesion por
    # camara), asi que el indice ya guardado en disco puede traer ids que
    # esta sesion ni siquiera abrio.
    all_cam_ids = {c for sh in index["shots"].values() for c in sh["cameras"]}
    per_cam = {c: 0 for c in sorted(all_cam_ids | set(cfg.ids))}
    pairs: dict[str, int] = {}
    for sh in index["shots"].values():
        seen = [c for c, q in sh["cameras"].items() if q["corners"] >= args.min_corners]
        for c in seen:
            per_cam[c] += 1
        for i, a in enumerate(sorted(seen)):
            for b in sorted(seen)[i + 1:]:
                pairs[f"{a}-{b}"] = pairs.get(f"{a}-{b}", 0) + 1

    print(f"\n=== set '{args.set}': {shot_n} tomas ===")
    print("vistas utiles por camara (para la calibracion intrinseca):")
    for c, n in per_cam.items():
        flag = "OK" if n >= 20 else "POCAS (objetivo >= 20)"
        print(f"  {c:<8} {n:3d}   {flag}")
    print("tomas compartidas por par (para la extrinseca):")
    if not pairs:
        print("  NINGUNA -> las camaras no quedaran en el mismo {W}.")
    for p, n in sorted(pairs.items()):
        flag = "OK" if n >= 8 else "POCAS (objetivo >= 8)"
        print(f"  {p:<14} {n:3d}   {flag}")
    if "anchor" not in index:
        print("\nAVISO: no hay toma ancla. Relanzar con --anchor y colocar el")
        print("tablero en la posicion que define el origen de {W}.")
    print(f"\nsalida -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
