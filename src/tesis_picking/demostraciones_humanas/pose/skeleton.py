"""Etapas 9-13 del roadmap: correspondencias, seleccion de camaras validas,
triangulacion multivista y reconstruccion del skeleton 3D.

Con una sola persona en escena, la "correspondencia" (etapa 9) es trivial:
el landmark `left_wrist` de cada camara es el mismo punto fisico por
construccion de MediaPipe, no hace falta resolverla por geometria.
"""
from __future__ import annotations

import numpy as np

from ...common.camera_model import CameraRig
from ...common.triangulation import Triangulation, triangulate_point
from .detector import LANDMARK_INDEX, LandmarkObs


def triangulate_skeleton(
    detections: dict[str, dict[str, LandmarkObs]],
    rig: CameraRig,
    *,
    min_visibility: float = 0.5,
    min_cameras: int = 2,
    max_reproj_px: float = 15.0,
) -> dict[str, Triangulation]:
    """Reconstruye en 3D cada landmark del alcance inicial.

    Args:
        detections: {cam_id: {landmark_name: LandmarkObs}}, salida de
            `PoseDetector.detect()` por camara para el mismo frameset.
        rig: camaras calibradas (intrinseco + extrinseco en el mismo {W}).
        min_visibility: etapa 10 (seleccion de camaras validas) -- una
            camara con `visibility` por debajo de esto no aporta esa vista.
        min_cameras: minimo de vistas validas para intentar triangular
            (roadmap: N_valid >= 2).
        max_reproj_px: umbral de rechazo de `triangulate_point`. El default
            del modulo de triangulacion (8 px) asume una extrinseca ya
            validada (checkpoint 4); mientras el rig este en calibracion de
            practica (ver bitacora 2026-09-22/23) conviene un umbral mas
            permisivo para no descartar todo.

    Returns:
        {landmark_name: Triangulation}. Los 8 landmarks siempre estan en el
        dict; los no reconstruibles llevan X = [NaN, NaN, NaN] (nunca ceros,
        etapa 16 del roadmap).
    """
    out: dict[str, Triangulation] = {}
    for name in LANDMARK_INDEX:
        obs: dict[str, np.ndarray] = {}
        for cam_id, lms in detections.items():
            lm = lms.get(name)
            if lm is not None and lm.visibility >= min_visibility:
                obs[cam_id] = np.array([lm.u, lm.v])
        out[name] = triangulate_point(
            obs, rig, min_cameras=min_cameras, max_reproj_px=max_reproj_px)
    return out


def valid_points(skeleton: dict[str, Triangulation]) -> dict[str, np.ndarray]:
    """Subconjunto de `skeleton` con reconstruccion valida (para plot/CSV)."""
    return {name: t.X for name, t in skeleton.items() if t.valid}
