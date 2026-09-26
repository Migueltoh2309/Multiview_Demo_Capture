"""Calibracion extrinseca multivista y definicion de {W} (etapas 6, 7 y 14).

Problema
--------
Se busca `T_world_cam` para cada camara, todas en el MISMO frame global {W}.
No basta con hacer PnP del tablero en una sola toma: (a) alguna camara puede no
ver el tablero-origen desde su angulo, y (b) una unica toma da una estimacion
ruidosa.

Solucion en tres pasos
----------------------
1. Se captura el tablero en MUCHAS poses distintas, simultaneamente con todas
   las camaras. Cada (camara, pose) que detecta el tablero aporta una medida
   `T_cam_board`.
2. Inicializacion por recorrido del grafo bipartito camaras <-> poses del
   tablero: partiendo de la pose que ANCLA el origen, se propagan camaras y
   poses alternadamente. Asi una camara que nunca ve el tablero-origen queda
   igualmente referida a {W} a traves de poses intermedias compartidas.
3. Ajuste de haces (bundle adjustment): se refinan conjuntamente TODAS las
   poses de camara y TODAS las poses del tablero minimizando el error de
   reproyeccion de cada esquina ChArUco. La pose ancla se mantiene fija --- es
   la que fija el gauge (origen y escala del problema).

La escala fisica proviene enteramente de `square_length_m` del tablero: de ahi
que medirlo bien sea el requisito metrologico numero uno del sistema.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ...common.camera_model import CameraModel, CameraRig, Intrinsics
from ...common.transforms import (invert_T, make_T, rvec_tvec_to_T, split_T,
                                  T_to_rvec_tvec)
from .charuco import BoardDetection, CharucoBoardWrapper

log = logging.getLogger(__name__)


@dataclass
class BoardObservation:
    """Deteccion del tablero por una camara en una pose concreta del tablero."""
    frame_key: str            # identifica la pose del tablero (p.ej. 'shot_007')
    cam_id: str
    corner_ids: np.ndarray    # (M,) ids de esquina ChArUco
    image_points: np.ndarray  # (M, 2) pixeles medidos (distorsionados)
    T_cam_board: np.ndarray   # 4x4 inicial por PnP
    n_corners: int


@dataclass
class ExtrinsicsResult:
    rig: CameraRig
    board_poses: dict[str, np.ndarray]          # T_world_board por pose
    rms_per_camera_px: dict[str, float]
    rms_global_px: float
    n_observations: int
    anchor_key: str
    baselines_m: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rms_global_px": round(self.rms_global_px, 4),
            "rms_per_camera_px": {k: round(v, 4) for k, v in self.rms_per_camera_px.items()},
            "n_observaciones": self.n_observations,
            "n_poses_tablero": len(self.board_poses),
            "pose_ancla": self.anchor_key,
            "baselines_m": {k: round(v, 4) for k, v in self.baselines_m.items()},
            "camaras": {},
        }
        for cam in self.rig:
            R, t = split_T(cam.T_world_cam)
            from ...common.transforms import R_to_rpy_deg
            d["camaras"][cam.cam_id] = {
                "posicion_world_m": [round(float(v), 4) for v in t],
                "rpy_world_deg": [round(float(v), 2) for v in R_to_rpy_deg(R)],
                "distancia_al_origen_m": round(float(np.linalg.norm(t)), 4),
            }
        if self.warnings:
            d["advertencias"] = self.warnings
        return d


# --------------------------------------------------------------------------- #
def build_observations(
    detections: dict[str, dict[str, BoardDetection]],
    board: CharucoBoardWrapper,
    intrinsics: dict[str, Intrinsics],
    *,
    min_corners: int = 6,
) -> list[BoardObservation]:
    """Convierte detecciones {frame_key: {cam_id: BoardDetection}} en observaciones."""
    obs: list[BoardObservation] = []
    for fkey, per_cam in detections.items():
        for cam_id, det in per_cam.items():
            if det is None or det.n_corners < min_corners:
                continue
            intr = intrinsics[cam_id]
            pose = board.estimate_pose(det, intr.K, intr.D, min_corners=min_corners)
            if pose is None:
                continue
            rvec, tvec = pose
            obs.append(BoardObservation(
                frame_key=fkey, cam_id=cam_id,
                corner_ids=det.ids_flat(),
                image_points=det.corners_flat(),
                T_cam_board=rvec_tvec_to_T(rvec, tvec),
                n_corners=det.n_corners,
            ))
    return obs


def _initialize_poses(
    observations: list[BoardObservation],
    cam_ids: list[str],
    anchor_key: str,
    T_world_anchor: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Propaga poses por el grafo bipartito camaras <-> poses del tablero.

    Returns:
        (T_world_cam por camara, T_world_board por pose, advertencias)
    """
    by_frame: dict[str, list[BoardObservation]] = {}
    by_cam: dict[str, list[BoardObservation]] = {}
    for o in observations:
        by_frame.setdefault(o.frame_key, []).append(o)
        by_cam.setdefault(o.cam_id, []).append(o)

    if anchor_key not in by_frame:
        raise ValueError(
            f"la pose ancla '{anchor_key}' no fue detectada por ninguna camara. "
            "Es la toma que define {W}: el tablero debe estar en su posicion de "
            "origen y verse nitido desde al menos una camara.")

    T_world_board: dict[str, np.ndarray] = {anchor_key: np.asarray(T_world_anchor, float)}
    T_world_cam: dict[str, np.ndarray] = {}

    queue: deque[tuple[str, str]] = deque([("frame", anchor_key)])
    while queue:
        kind, key = queue.popleft()
        if kind == "frame":
            # Con la pose del tablero conocida, se resuelven las camaras que la ven.
            for o in by_frame.get(key, []):
                if o.cam_id in T_world_cam:
                    continue
                # T_world_cam = T_world_board @ inv(T_cam_board)
                T_world_cam[o.cam_id] = T_world_board[key] @ invert_T(o.T_cam_board)
                queue.append(("cam", o.cam_id))
        else:
            # Con la camara conocida, se resuelven las poses del tablero que ve.
            for o in by_cam.get(key, []):
                if o.frame_key in T_world_board:
                    continue
                T_world_board[o.frame_key] = T_world_cam[key] @ o.T_cam_board
                queue.append(("frame", o.frame_key))

    warns: list[str] = []
    missing = [c for c in cam_ids if c not in T_world_cam]
    if missing:
        warns.append(
            f"camaras sin conexion al grafo de {{W}}: {missing}. "
            "Falta alguna toma donde esa camara y otra ya referida vean el mismo "
            "tablero a la vez.")
    return T_world_cam, T_world_board, warns


# --------------------------------------------------------------------------- #
def _pack(T_world_cam: dict[str, np.ndarray], cam_order: list[str],
          T_world_board: dict[str, np.ndarray], frame_order: list[str]) -> np.ndarray:
    """Empaqueta las incognitas en un vector. Las camaras se parametrizan como
    T_cam_world (que es lo que entra en projectPoints)."""
    parts = []
    for c in cam_order:
        r, t = T_to_rvec_tvec(invert_T(T_world_cam[c]))
        parts.append(np.concatenate([r, t]))
    for f in frame_order:
        r, t = T_to_rvec_tvec(T_world_board[f])
        parts.append(np.concatenate([r, t]))
    return np.concatenate(parts) if parts else np.empty(0)


def _unpack(x: np.ndarray, cam_order: list[str], frame_order: list[str]
            ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    nc = len(cam_order)
    cams = {c: rvec_tvec_to_T(x[6 * i:6 * i + 3], x[6 * i + 3:6 * i + 6])
            for i, c in enumerate(cam_order)}          # T_cam_world
    boards = {f: rvec_tvec_to_T(x[6 * (nc + j):6 * (nc + j) + 3],
                                x[6 * (nc + j) + 3:6 * (nc + j) + 6])
              for j, f in enumerate(frame_order)}       # T_world_board
    return cams, boards


def bundle_adjust(
    observations: list[BoardObservation],
    board: CharucoBoardWrapper,
    intrinsics: dict[str, Intrinsics],
    T_world_cam: dict[str, np.ndarray],
    T_world_board: dict[str, np.ndarray],
    anchor_key: str,
    *,
    max_nfev: int = 200,
    verbose: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Refina poses de camara y de tablero minimizando el error de reproyeccion.

    La pose ancla queda FIJA (no entra en el vector de incognitas): es el gauge
    que impide que la solucion derive globalmente.
    """
    cam_order = sorted(T_world_cam)
    frame_order = [f for f in sorted(T_world_board) if f != anchor_key]
    obs = [o for o in observations
           if o.cam_id in T_world_cam and o.frame_key in T_world_board]
    if not obs:
        raise ValueError("no hay observaciones utilizables para el bundle adjustment")

    all_obj = board.object_points()
    obj_pts = [all_obj[o.corner_ids] for o in obs]
    img_pts = [o.image_points for o in obs]
    cam_idx = [cam_order.index(o.cam_id) for o in obs]
    frame_idx = [frame_order.index(o.frame_key) if o.frame_key != anchor_key else -1
                 for o in obs]
    T_anchor = T_world_board[anchor_key]
    Ks = {c: intrinsics[c].K for c in cam_order}
    Ds = {c: intrinsics[c].D for c in cam_order}

    x0 = _pack(T_world_cam, cam_order, T_world_board, frame_order)

    def residuals(x: np.ndarray) -> np.ndarray:
        cams, boards = _unpack(x, cam_order, frame_order)
        out = []
        for k, o in enumerate(obs):
            Tb = T_anchor if frame_idx[k] < 0 else boards[o.frame_key]
            # esquinas del tablero -> {W} -> frame de camara
            Xw = obj_pts[k] @ Tb[:3, :3].T + Tb[:3, 3]
            Tc = cams[o.cam_id]
            Xc = Xw @ Tc[:3, :3].T + Tc[:3, 3]
            proj, _ = cv2.projectPoints(
                Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                Ks[o.cam_id], Ds[o.cam_id])
            out.append((proj.reshape(-1, 2) - img_pts[k]).ravel())
        return np.concatenate(out)

    # Patron de dispersion: cada observacion solo toca su camara y su pose.
    n_res = sum(2 * len(p) for p in obj_pts)
    J = lil_matrix((n_res, x0.size), dtype=int)
    row = 0
    nc = len(cam_order)
    for k, o in enumerate(obs):
        m = 2 * len(obj_pts[k])
        ci = cam_idx[k]
        J[row:row + m, 6 * ci:6 * ci + 6] = 1
        if frame_idx[k] >= 0:
            fi = nc + frame_idx[k]
            J[row:row + m, 6 * fi:6 * fi + 6] = 1
        row += m

    sol = least_squares(residuals, x0, jac_sparsity=J, method="trf",
                        loss="huber", f_scale=2.0,   # robusto a esquinas atipicas
                        max_nfev=max_nfev, verbose=verbose, xtol=1e-12, ftol=1e-12)

    cams, boards = _unpack(sol.x, cam_order, frame_order)
    T_world_cam_out = {c: invert_T(T) for c, T in cams.items()}
    T_world_board_out = {**boards, anchor_key: T_anchor}
    return T_world_cam_out, T_world_board_out, sol.fun


# --------------------------------------------------------------------------- #
def calibrate_extrinsics(
    detections: dict[str, dict[str, BoardDetection]],
    board: CharucoBoardWrapper,
    intrinsics: dict[str, Intrinsics],
    *,
    anchor_key: str,
    T_world_anchor: np.ndarray,
    min_corners: int = 6,
    refine: bool = True,
) -> ExtrinsicsResult:
    """Punto de entrada: detecciones -> rig calibrado en {W}.

    Args:
        detections: {frame_key: {cam_id: BoardDetection}}. Cada frame_key es una
            pose distinta del tablero, capturada simultaneamente por las camaras.
        anchor_key: frame_key de la toma en que el tablero define el origen.
        T_world_anchor: pose del tablero en {W} en esa toma (identidad si el
            tablero se coloco alineado con {W}); viene de world_frame.yaml.
    """
    cam_ids = sorted(intrinsics)
    obs = build_observations(detections, board, intrinsics, min_corners=min_corners)
    if not obs:
        raise ValueError(
            "ninguna deteccion valida del tablero. Revisar iluminacion, enfoque "
            "y que el tablero impreso coincida con config/charuco.yaml")

    T_world_cam, T_world_board, warns = _initialize_poses(
        obs, cam_ids, anchor_key, T_world_anchor)

    if refine and len(T_world_board) > 1:
        T_world_cam, T_world_board, _ = bundle_adjust(
            obs, board, intrinsics, T_world_cam, T_world_board, anchor_key)

    rig = CameraRig([
        CameraModel(c, intrinsics[c], T_world_cam[c])
        for c in cam_ids if c in T_world_cam
    ])

    rms_per_cam, rms_global = evaluate_extrinsics(
        obs, board, rig, T_world_board)

    for cid, e in rms_per_cam.items():
        if e > 1.5:
            warns.append(
                f"{cid}: RMS de reproyeccion extrinseca {e:.2f} px (> 1.5 px). "
                "Suele indicar que la camara se movio entre tomas o que sus "
                "intrinsecos no son fiables.")

    return ExtrinsicsResult(
        rig=rig, board_poses=T_world_board,
        rms_per_camera_px=rms_per_cam, rms_global_px=rms_global,
        n_observations=len(obs), anchor_key=anchor_key,
        baselines_m={f"{a}-{b}": v for (a, b), v in rig.baselines_m().items()},
        warnings=warns,
    )


def evaluate_extrinsics(
    observations: Iterable[BoardObservation],
    board: CharucoBoardWrapper,
    rig: CameraRig,
    T_world_board: dict[str, np.ndarray],
) -> tuple[dict[str, float], float]:
    """RMS del error de reproyeccion, por camara y global."""
    all_obj = board.object_points()
    sq: dict[str, list[float]] = {c: [] for c in rig.ids}
    for o in observations:
        if o.cam_id not in rig.ids or o.frame_key not in T_world_board:
            continue
        Tb = T_world_board[o.frame_key]
        Xw = all_obj[o.corner_ids] @ Tb[:3, :3].T + Tb[:3, 3]
        proj = rig[o.cam_id].project(Xw)
        err = np.linalg.norm(proj - o.image_points, axis=1)
        sq[o.cam_id].extend(err[np.isfinite(err)].tolist())

    per_cam = {c: float(np.sqrt(np.mean(np.square(v)))) if v else float("nan")
               for c, v in sq.items()}
    allv = [e for v in sq.values() for e in v]
    return per_cam, float(np.sqrt(np.mean(np.square(allv)))) if allv else float("nan")
