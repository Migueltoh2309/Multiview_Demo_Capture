"""Transformaciones rigidas SE(3) y utilidades de rotacion.

Convencion de nombres usada en TODO el proyecto:

    T_a_b  = pose del frame {b} expresada en el frame {a}
           = matriz que transforma un punto de {b} a {a}:  p_a = T_a_b @ p_b

Asi la composicion se lee encadenando indices:
    T_world_cam @ p_cam = p_world
    T_world_cam = T_world_board @ inv(T_cam_board)
"""
from __future__ import annotations

import numpy as np
import cv2


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Construye una 4x4 homogenea a partir de R (3x3) y t (3,)."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def split_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (R, t) de una 4x4."""
    T = np.asarray(T, dtype=float)
    return T[:3, :3].copy(), T[:3, 3].copy()


def invert_T(T: np.ndarray) -> np.ndarray:
    """Inversa de una transformada rigida, sin invertir la matriz completa."""
    R, t = split_T(T)
    return make_T(R.T, -R.T @ t)


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convierte la salida de solvePnP (Rodrigues) a una 4x4."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return make_T(R, np.asarray(tvec, dtype=float).reshape(3))


def T_to_rvec_tvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inversa de `rvec_tvec_to_T`."""
    R, t = split_T(T)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3), t


def rpy_deg_to_R(rpy_deg: np.ndarray) -> np.ndarray:
    """Angulos (roll, pitch, yaw) en grados -> R, con convencion R = Rz@Ry@Rx."""
    r, p, y = np.deg2rad(np.asarray(rpy_deg, dtype=float).reshape(3))
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_rpy_deg(R: np.ndarray) -> np.ndarray:
    """Inversa de `rpy_deg_to_R` (rama principal, pitch en [-90, 90])."""
    R = np.asarray(R, dtype=float)
    sp = -R[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    p = np.arcsin(sp)
    if abs(sp) < 0.99999:
        r = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        r = np.arctan2(-R[1, 2], R[1, 1])
        y = 0.0
    return np.rad2deg(np.array([r, p, y]))


def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Aplica T (4x4) a puntos P de forma (..., 3). Propaga NaN."""
    P = np.asarray(P, dtype=float)
    shape = P.shape
    flat = P.reshape(-1, 3)
    R, t = split_T(T)
    out = flat @ R.T + t
    return out.reshape(shape)


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Proyecta R a SO(3) por SVD. Corrige la deriva numerica al promediar."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    R_ = U @ Vt
    if np.linalg.det(R_) < 0:          # evita reflexiones
        U[:, -1] *= -1
        R_ = U @ Vt
    return R_


def average_rotations(Rs: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Media de rotaciones por proyeccion del promedio aritmetico a SO(3).

    Suficiente cuando las rotaciones estan proximas entre si, que es el caso al
    promediar la pose de una camara estatica sobre varios frames.
    """
    Rs = np.asarray(Rs, dtype=float).reshape(-1, 3, 3)
    if len(Rs) == 0:
        raise ValueError("no hay rotaciones que promediar")
    return orthonormalize(Rs.mean(axis=0))


def average_transforms(Ts: list[np.ndarray]) -> np.ndarray:
    """Media de transformadas rigidas: media aritmetica de t, SO(3) de R."""
    Ts = [np.asarray(T, dtype=float) for T in Ts]
    if not Ts:
        raise ValueError("no hay transformadas que promediar")
    R = average_rotations([T[:3, :3] for T in Ts])
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    return make_T(R, t)


def angle_between_R_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angulo del giro relativo entre dos rotaciones, en grados."""
    dR = np.asarray(Ra, dtype=float).T @ np.asarray(Rb, dtype=float)
    c = (np.trace(dR) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0))))
