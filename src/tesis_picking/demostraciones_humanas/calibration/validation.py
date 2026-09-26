"""Validacion metrologica del rig (etapas 7 y 17 del roadmap).

Estas son las metricas que sostienen la frase "caracterizar su exactitud
metrologica" del objetivo especifico 1. Se apoyan en un patron de geometria
CONOCIDA (el propio ChArUco) para tener verdad de terreno sin equipo adicional:

  * error 3D punto a punto:   e_3D = || X_est - X_real ||
  * error de distancia conocida: e_d = | d_est - d_real |
  * error de planaridad: las esquinas del tablero son coplanares por
    construccion; la dispersion respecto al plano ajustado mide el ruido de
    reconstruccion sin depender de la pose del tablero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from ...common.camera_model import CameraRig
from ...common.transforms import invert_T
from ...common.triangulation import triangulate_point
from .charuco import BoardDetection, CharucoBoardWrapper


@dataclass
class ReconstructionCheck:
    """Resultado de reconstruir un tablero de geometria conocida."""
    frame_key: str
    n_points: int
    point_errors_mm: np.ndarray = field(default_factory=lambda: np.empty(0))
    distance_errors_mm: np.ndarray = field(default_factory=lambda: np.empty(0))
    planarity_rms_mm: float = float("nan")
    reproj_rms_px: float = float("nan")
    n_cameras_hist: dict[int, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        def stats(a: np.ndarray, name: str) -> dict[str, float]:
            a = a[np.isfinite(a)]
            if a.size == 0:
                return {}
            return {f"{name}_mean_mm": round(float(np.mean(a)), 3),
                    f"{name}_rms_mm": round(float(np.sqrt(np.mean(a ** 2))), 3),
                    f"{name}_p95_mm": round(float(np.percentile(a, 95)), 3),
                    f"{name}_max_mm": round(float(np.max(a)), 3)}
        out: dict[str, Any] = {
            "frame": self.frame_key,
            "n_puntos_reconstruidos": self.n_points,
            "planaridad_rms_mm": round(self.planarity_rms_mm, 3),
            "reproyeccion_rms_px": round(self.reproj_rms_px, 3),
            "camaras_por_punto": self.n_cameras_hist,
        }
        out.update(stats(self.point_errors_mm, "error_3d"))
        out.update(stats(self.distance_errors_mm, "error_distancia"))
        return out


def check_board_reconstruction(
    detections: dict[str, BoardDetection],
    board: CharucoBoardWrapper,
    rig: CameraRig,
    *,
    frame_key: str = "check",
    T_world_board: np.ndarray | None = None,
    min_cameras: int = 2,
    max_reproj_px: float = 5.0,
    n_distance_pairs: int = 400,
    rng_seed: int = 0,
) -> ReconstructionCheck:
    """Triangula las esquinas ChArUco y las compara con su geometria real.

    Args:
        detections: {cam_id: BoardDetection} del MISMO instante.
        T_world_board: si se conoce la pose del tablero en {W}, se calcula
            tambien el error 3D absoluto punto a punto. Si es None, solo se
            evaluan magnitudes invariantes a la pose (distancias y planaridad),
            que son las utiles cuando el tablero se coloca en cualquier sitio.
    """
    obj_all = board.object_points()
    per_id: dict[int, dict[str, np.ndarray]] = {}
    for cam_id, det in detections.items():
        if det is None:
            continue
        for cid, uv in det.corner_by_id().items():
            per_id.setdefault(int(cid), {})[cam_id] = uv

    ids: list[int] = []
    X: list[np.ndarray] = []
    reproj: list[float] = []
    hist: dict[int, int] = {}
    for cid, obs in sorted(per_id.items()):
        tri = triangulate_point(obs, rig, min_cameras=min_cameras,
                                max_reproj_px=max_reproj_px)
        hist[tri.n_cameras] = hist.get(tri.n_cameras, 0) + 1
        if not tri.valid:
            continue
        ids.append(cid)
        X.append(tri.X)
        reproj.append(tri.rms_reproj_px)

    res = ReconstructionCheck(frame_key=frame_key, n_points=len(ids),
                              n_cameras_hist=hist)
    if len(ids) < 4:
        return res

    Xw = np.vstack(X)
    obj = obj_all[np.asarray(ids, dtype=int)]
    res.reproj_rms_px = float(np.sqrt(np.mean(np.square(reproj))))
    res.planarity_rms_mm = _planarity_rms_mm(Xw)

    # Error de distancia: invariante a la pose del tablero (etapa 17).
    rng = np.random.default_rng(rng_seed)
    pairs = list(combinations(range(len(ids)), 2))
    if len(pairs) > n_distance_pairs:
        pairs = [pairs[k] for k in rng.choice(len(pairs), n_distance_pairs, replace=False)]
    d_real = np.array([np.linalg.norm(obj[i] - obj[j]) for i, j in pairs])
    d_est = np.array([np.linalg.norm(Xw[i] - Xw[j]) for i, j in pairs])
    res.distance_errors_mm = np.abs(d_est - d_real) * 1000.0

    if T_world_board is not None:
        X_real = obj @ T_world_board[:3, :3].T + T_world_board[:3, 3]
        res.point_errors_mm = np.linalg.norm(Xw - X_real, axis=1) * 1000.0

    return res


def _planarity_rms_mm(X: np.ndarray) -> float:
    """RMS de la distancia al plano de mejor ajuste (por PCA)."""
    X = np.asarray(X, dtype=float)
    c = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - c)
    n = Vt[-1]
    return float(np.sqrt(np.mean((np.abs((X - c) @ n) * 1000.0) ** 2)))


def scale_error_ratio(check: ReconstructionCheck) -> float:
    """Error relativo de escala, en partes por mil.

    Un sesgo sistematico aqui apunta casi siempre a `square_length_m` mal medido:
    si el tablero real mide 29.8 mm por casilla y el YAML dice 30.0, TODAS las
    reconstrucciones salen escaladas un 0.7 %.
    """
    e = check.distance_errors_mm
    e = e[np.isfinite(e)]
    return float(np.mean(e)) if e.size else float("nan")


def known_distance_check(
    detections: dict[str, BoardDetection],
    board: CharucoBoardWrapper,
    rig: CameraRig,
    id_a: int, id_b: int,
    *, min_cameras: int = 2,
) -> dict[str, Any]:
    """Etapa 17: reconstruye dos esquinas concretas y compara su distancia."""
    obj = board.object_points()
    obs: dict[int, dict[str, np.ndarray]] = {id_a: {}, id_b: {}}
    for cam_id, det in detections.items():
        if det is None:
            continue
        cmap = det.corner_by_id()
        for k in (id_a, id_b):
            if k in cmap:
                obs[k][cam_id] = cmap[k]

    ta = triangulate_point(obs[id_a], rig, min_cameras=min_cameras)
    tb = triangulate_point(obs[id_b], rig, min_cameras=min_cameras)
    d_real = float(np.linalg.norm(obj[id_a] - obj[id_b]))
    if not (ta.valid and tb.valid):
        return {"ok": False, "d_real_mm": d_real * 1000}
    d_est = float(np.linalg.norm(ta.X - tb.X))
    return {
        "ok": True,
        "d_real_mm": round(d_real * 1000, 3),
        "d_est_mm": round(d_est * 1000, 3),
        "error_mm": round(abs(d_est - d_real) * 1000, 3),
        "error_relativo_pct": round(abs(d_est - d_real) / d_real * 100, 4),
        "camaras": {"a": ta.cam_ids, "b": tb.cam_ids},
    }
