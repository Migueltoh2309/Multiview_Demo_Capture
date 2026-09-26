"""Etapa 8 del roadmap: deteccion de pose 2D por camara con MediaPipe Pose.

Landmarks del alcance inicial (roadmap, seccion 19) -- hombros, codos y
munecas de ambos brazos, mas caderas para poder construir despues un frame
corporal:

    11 LEFT_SHOULDER   13 LEFT_ELBOW    15 LEFT_WRIST
    12 RIGHT_SHOULDER  14 RIGHT_ELBOW   16 RIGHT_WRIST
    23 LEFT_HIP        24 RIGHT_HIP
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

LANDMARK_INDEX: dict[str, int] = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
}

# Pares que se dibujan/triangulan como esqueleto (roadmap, seccion 41).
SKELETON_EDGES: list[tuple[str, str]] = [
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]

# Orden de los 4 puntos del torso como perimetro (no cruzado), para dibujarlo
# como cara solida en el plot 3D en vez de solo el wireframe de SKELETON_EDGES.
TORSO_QUAD: tuple[str, str, str, str] = (
    "left_shoulder", "right_shoulder", "right_hip", "left_hip",
)


@dataclass
class LandmarkObs:
    """Una observacion 2D de un landmark, en pixeles de la imagen original."""
    u: float
    v: float
    visibility: float   # 0-1, confianza de MediaPipe de que el punto es visible


class PoseDetector:
    """Envoltorio de MediaPipe Pose para UNA camara.

    Cada camara necesita su propia instancia: `mp.solutions.pose.Pose`
    mantiene tracking temporal interno (mas rapido y estable frame a frame
    que static_image_mode=True), y ese estado no se puede compartir entre
    streams de video distintos.
    """

    def __init__(self, *, model_complexity: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> dict[str, LandmarkObs]:
        """Corre MediaPipe sobre un frame BGR y devuelve solo los landmarks
        del alcance inicial, en pixeles de `frame_bgr`. Dict vacio si no
        detecto a nadie."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return {}
        lms = result.pose_landmarks.landmark
        return {
            name: LandmarkObs(u=lms[idx].x * w, v=lms[idx].y * h,
                               visibility=lms[idx].visibility)
            for name, idx in LANDMARK_INDEX.items()
        }

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
