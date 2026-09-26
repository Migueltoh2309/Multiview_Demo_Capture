"""Calibracion intrinseca por camara (etapas 5 y 6 del roadmap).

Obtiene K y D para cada camara y reporta el error de reproyeccion POR VISTA,
no solo el RMS global: una unica vista mala (tablero borroso, casi frontal,
en el borde) puede inflar el RMS y, sobre todo, sesgar D. Por eso el flujo
incluye un descarte iterativo de las vistas peores.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...common.camera_model import Intrinsics
from .charuco import BoardDetection, CharucoBoardWrapper

log = logging.getLogger(__name__)


@dataclass
class IntrinsicsResult:
    intr: Intrinsics
    per_view_rms_px: np.ndarray            # (n_views,)
    view_names: list[str]
    rejected: list[str] = field(default_factory=list)
    coverage: dict[str, float] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        return {
            "rms_reprojection_px": round(float(self.intr.rms_reprojection_px), 4),
            "n_views_usadas": int(self.intr.n_views),
            "n_views_descartadas": len(self.rejected),
            "per_view_rms_px": {
                "mean": round(float(np.mean(self.per_view_rms_px)), 4),
                "max": round(float(np.max(self.per_view_rms_px)), 4),
                "p95": round(float(np.percentile(self.per_view_rms_px, 95)), 4),
            },
            "fx": round(self.intr.fx, 2), "fy": round(self.intr.fy, 2),
            "cx": round(self.intr.cx, 2), "cy": round(self.intr.cy, 2),
            "distortion": [round(float(v), 6) for v in np.ravel(self.intr.D)],
            "coverage": {k: round(v, 3) for k, v in self.coverage.items()},
            "vistas_descartadas": self.rejected,
        }


def image_coverage(detections: list[BoardDetection],
                   image_size: tuple[int, int], grid: int = 6) -> dict[str, float]:
    """Fraccion del encuadre cubierta por esquinas del tablero.

    Es la metrica que determina si D esta bien estimada: la distorsion es un
    efecto radial que solo se observa LEJOS del centro. Una calibracion con
    todas las vistas en el centro da un RMS excelente y una D inservible.
    """
    w, h = image_size
    occupied = np.zeros((grid, grid), dtype=bool)
    edge_hits = 0
    total = 0
    for det in detections:
        pts = det.corners_flat()
        if pts.size == 0:
            continue
        total += len(pts)
        gx = np.clip((pts[:, 0] / w * grid).astype(int), 0, grid - 1)
        gy = np.clip((pts[:, 1] / h * grid).astype(int), 0, grid - 1)
        occupied[gy, gx] = True
        # "borde" = el 20% exterior del encuadre en cualquiera de los dos ejes
        edge_hits += int(np.sum((pts[:, 0] < 0.2 * w) | (pts[:, 0] > 0.8 * w) |
                                (pts[:, 1] < 0.2 * h) | (pts[:, 1] > 0.8 * h)))
    return {
        "grid_fill": float(occupied.mean()),
        "edge_point_ratio": float(edge_hits / total) if total else 0.0,
        "n_points": float(total),
    }


def calibrate_intrinsics(
    detections: list[BoardDetection],
    board: CharucoBoardWrapper,
    image_size: tuple[int, int],
    *,
    view_names: list[str] | None = None,
    min_corners: int = 8,
    max_view_rms_px: float | None = 1.0,
    max_reject_frac: float = 0.25,
    fix_k3: bool = True,
    rational_model: bool = False,
) -> IntrinsicsResult:
    """Calibra K y D a partir de detecciones ChArUco.

    Args:
        detections: una por imagen de calibracion.
        board: tablero usado.
        image_size: (width, height) --- debe ser la MISMA en todas las imagenes.
        max_view_rms_px: vistas con RMS por encima se descartan y se recalibra.
            None desactiva el descarte.
        max_reject_frac: cota superior a la fraccion de vistas descartadas, para
            no acabar calibrando con un puñado de vistas "faciles".
        fix_k3: k3 suele quedar mal condicionado en lentes de webcam y absorbe
            ruido; fijarlo a 0 da una D mas estable.
        rational_model: activar solo para lentes gran angular / ojo de pez leve.
    """
    names = view_names or [f"view_{i:03d}" for i in range(len(detections))]
    keep = [i for i, d in enumerate(detections) if d.is_usable(min_corners)]
    if len(keep) < 6:
        raise ValueError(
            f"solo {len(keep)} vistas utiles (>= {min_corners} esquinas); "
            "se necesitan al menos 6, y idealmente 20-40 con orientaciones variadas")

    flags = 0
    if fix_k3:
        flags |= cv2.CALIB_FIX_K3
    if rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
        flags &= ~cv2.CALIB_FIX_K3

    rejected: list[str] = []
    n_start = len(keep)

    while True:
        objs, imgs = [], []
        for i in keep:
            o, p = board.match_points(detections[i])
            objs.append(o.reshape(-1, 1, 3).astype(np.float32))
            imgs.append(p.reshape(-1, 1, 2).astype(np.float32))

        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objs, imgs, image_size, None, None, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8))

        per_view = np.array([
            _view_rms(objs[j], imgs[j], rvecs[j], tvecs[j], K, D)
            for j in range(len(keep))])

        if max_view_rms_px is None:
            break
        worst = int(np.argmax(per_view))
        room = int(np.floor(max_reject_frac * n_start)) - len(rejected)
        if per_view[worst] <= max_view_rms_px or room <= 0 or len(keep) <= 8:
            break
        rejected.append(names[keep[worst]])
        log.info("descartando vista %s (RMS %.3f px)", names[keep[worst]], per_view[worst])
        keep.pop(worst)

    intr = Intrinsics(
        K=np.asarray(K, dtype=float), D=np.asarray(D, dtype=float).ravel(),
        image_size=tuple(image_size), rms_reprojection_px=float(rms),
        n_views=len(keep),
        model="pinhole_rational" if rational_model else "pinhole_radtan",
    )
    return IntrinsicsResult(
        intr=intr,
        per_view_rms_px=per_view,
        view_names=[names[i] for i in keep],
        rejected=rejected,
        coverage=image_coverage([detections[i] for i in keep], image_size),
    )


def _view_rms(obj, img, rvec, tvec, K, D) -> float:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = proj.reshape(-1, 2) - img.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


def undistortion_preview(image: np.ndarray, intr: Intrinsics,
                         alpha: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (original con rejilla, corregida con rejilla) --- etapa 13.

    La rejilla hace visible el efecto: en la imagen corregida las lineas rectas
    del mundo deben verse rectas, sobre todo cerca de esquinas y bordes.
    """
    h, w = image.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(intr.K, intr.D, (w, h), alpha, (w, h))
    und = cv2.undistort(image, intr.K, intr.D, None, newK)
    return _draw_grid(image), _draw_grid(und)


def _draw_grid(img: np.ndarray, step: int = 80) -> np.ndarray:
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    for x in range(0, w, step):
        cv2.line(vis, (x, 0), (x, h), (0, 200, 0), 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(vis, (0, y), (w, y), (0, 200, 0), 1, cv2.LINE_AA)
    return vis
