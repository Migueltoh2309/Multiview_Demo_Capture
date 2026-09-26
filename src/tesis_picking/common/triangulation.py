"""Triangulacion multivista y error de reproyeccion.

Etapas 11, 12 y 15 del roadmap. Todo trabaja en coordenadas NORMALIZADAS de
camara (pixeles ya corregidos de distorsion y multiplicados por K^-1), de modo
que la matriz de proyeccion es simplemente [R|t] y la distorsion no entra en la
minimizacion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .camera_model import CameraModel, CameraRig


@dataclass
class Triangulation:
    """Resultado de reconstruir un punto 3D."""
    X: np.ndarray                 # (3,) en {W}; NaN si no reconstruido
    cam_ids: list[str]            # camaras efectivamente usadas
    reproj_errors_px: dict[str, float]   # error por camara usada
    valid: bool

    @property
    def n_cameras(self) -> int:
        return len(self.cam_ids)

    @property
    def rms_reproj_px(self) -> float:
        if not self.reproj_errors_px:
            return float("nan")
        e = np.asarray(list(self.reproj_errors_px.values()), float)
        return float(np.sqrt(np.mean(e ** 2)))

    @property
    def max_reproj_px(self) -> float:
        if not self.reproj_errors_px:
            return float("nan")
        return float(np.max(list(self.reproj_errors_px.values())))


def triangulate_dlt(xn: np.ndarray, Ps: np.ndarray) -> np.ndarray:
    """DLT lineal de N vistas.

    Args:
        xn: (N, 2) puntos en coordenadas normalizadas de camara.
        Ps: (N, 3, 4) matrices [R|t] correspondientes.

    Returns:
        (3,) punto en el frame en que estan expresadas las Ps (aqui, {W}).

    Cada vista aporta dos ecuaciones  x*(P3.X) - (P1.X) = 0  y
    y*(P3.X) - (P2.X) = 0; se resuelve el sistema homogeneo por SVD.
    """
    xn = np.asarray(xn, dtype=float).reshape(-1, 2)
    Ps = np.asarray(Ps, dtype=float).reshape(-1, 3, 4)
    n = xn.shape[0]
    if n < 2:
        return np.full(3, np.nan)

    A = np.empty((2 * n, 4), dtype=float)
    A[0::2] = xn[:, 0:1] * Ps[:, 2, :] - Ps[:, 0, :]
    A[1::2] = xn[:, 1:2] * Ps[:, 2, :] - Ps[:, 1, :]

    # Normalizar cada fila evita que una vista con residuo de mayor magnitud
    # domine la solucion de minimos cuadrados.
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    A = A / norms

    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:      # punto en el infinito: reconstruccion invalida
        return np.full(3, np.nan)
    return Xh[:3] / Xh[3]


def refine_triangulation(X0: np.ndarray, xn: np.ndarray, Ps: np.ndarray,
                         max_nfev: int = 30) -> np.ndarray:
    """Refina X minimizando el error de reproyeccion geometrico (no algebraico).

    El DLT minimiza un residuo algebraico que no corresponde a la distancia en
    pixeles; este refinamiento no lineal corrige ese sesgo. Con >= 3 vistas la
    diferencia es apreciable.
    """
    X0 = np.asarray(X0, dtype=float).reshape(3)
    if not np.all(np.isfinite(X0)):
        return X0
    xn = np.asarray(xn, dtype=float).reshape(-1, 2)
    Ps = np.asarray(Ps, dtype=float).reshape(-1, 3, 4)

    def residuals(X: np.ndarray) -> np.ndarray:
        Xh = np.append(X, 1.0)
        p = Ps @ Xh                       # (N, 3)
        z = p[:, 2]
        # Evita division por ~0 para puntos en el plano principal
        z = np.where(np.abs(z) < 1e-9, np.sign(z) * 1e-9 + 1e-9, z)
        proj = p[:, :2] / z[:, None]
        return (proj - xn).ravel()

    try:
        sol = least_squares(residuals, X0, method="lm", max_nfev=max_nfev)
    except Exception:
        return X0
    return sol.x if sol.success or np.all(np.isfinite(sol.x)) else X0


def triangulate_point(
    observations: dict[str, np.ndarray],
    rig: CameraRig,
    *,
    min_cameras: int = 2,
    refine: bool = True,
    max_reproj_px: float | None = 8.0,
) -> Triangulation:
    """Reconstruye un punto 3D en {W} a partir de observaciones por camara.

    Args:
        observations: {cam_id: (u, v)} en PIXELES DISTORSIONADOS. Las entradas
            con NaN o con cam_id ausente del rig se ignoran (etapa 10: seleccion
            de camaras validas).
        rig: camaras calibradas (intrinsecos + extrinsecos en el mismo {W}).
        min_cameras: minimo de vistas validas (roadmap: N_valid >= 2).
        refine: aplicar refinamiento no lineal tras el DLT.
        max_reproj_px: si el error maximo de reproyeccion lo supera, la
            reconstruccion se marca invalida (etapa 16). None desactiva el test.

    Returns:
        `Triangulation`. Si no es reconstruible, X = [NaN, NaN, NaN] --- nunca
        ceros (etapa 16 del roadmap).
    """
    cam_ids: list[str] = []
    uvs: list[np.ndarray] = []
    for cid, uv in observations.items():
        if cid not in rig.ids or uv is None:
            continue
        uv = np.asarray(uv, dtype=float).reshape(2)
        if not np.all(np.isfinite(uv)):
            continue
        cam_ids.append(cid)
        uvs.append(uv)

    if len(cam_ids) < max(2, min_cameras):
        return Triangulation(np.full(3, np.nan), cam_ids, {}, False)

    cams: list[CameraModel] = [rig[c] for c in cam_ids]
    xn = np.vstack([c.normalize_points(uv) for c, uv in zip(cams, uvs)])
    Ps = np.stack([c.P_normalized for c in cams])

    X = triangulate_dlt(xn, Ps)
    if refine and np.all(np.isfinite(X)):
        X = refine_triangulation(X, xn, Ps)

    if not np.all(np.isfinite(X)):
        return Triangulation(np.full(3, np.nan), cam_ids, {}, False)

    errs = reprojection_errors(X, dict(zip(cam_ids, uvs)), rig)

    # Cheirality: el punto debe estar DELANTE de todas las camaras usadas.
    in_front = all(float((c.R_cam_world @ X + c.t_cam_world)[2]) > 1e-6 for c in cams)
    ok = in_front and all(np.isfinite(v) for v in errs.values())
    if ok and max_reproj_px is not None:
        ok = max(errs.values()) <= max_reproj_px

    return Triangulation(X if ok else np.full(3, np.nan), cam_ids, errs, bool(ok))


def reprojection_errors(X_world: np.ndarray, observations: dict[str, np.ndarray],
                        rig: CameraRig) -> dict[str, float]:
    """Error de reproyeccion en pixeles por camara (etapa 15).

    Reproyecta CON distorsion, para comparar contra el pixel medido crudo.
    """
    X = np.asarray(X_world, dtype=float).reshape(3)
    out: dict[str, float] = {}
    for cid, uv in observations.items():
        if cid not in rig.ids or uv is None:
            continue
        uv = np.asarray(uv, dtype=float).reshape(2)
        if not np.all(np.isfinite(uv)):
            continue
        proj = rig[cid].project(X.reshape(1, 3)).reshape(2)
        out[cid] = float(np.linalg.norm(proj - uv)) if np.all(np.isfinite(proj)) else float("nan")
    return out
