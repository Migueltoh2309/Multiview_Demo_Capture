"""Cambio de coordenadas: camara {C} -> mundo de trabajo {W} -> robot {B}.

La cadena completa que necesita el H1-2 para llevar la palma a la fruta:

    p_cam  --(T_world_cam)-->  p_world  --(T_base_world)-->  p_base
                                                                |
                                                       cinematica inversa

`T_world_cam` sale de la calibracion extrinseca con ChArUco (etapa V4 del
roadmap de vision) y se guarda en calib/rs_<serial>.yaml.

`T_base_world` es la relacion entre el mundo de trabajo y la base del robot.
Si el ChArUco que define {W} se coloca en una posicion MEDIDA respecto a la
base del H1-2, esta transformada se conoce por construccion. La alternativa
rigurosa es la calibracion mano-ojo, que necesita el robot y se aborda en la
etapa V7 --- por eso aqui se deja explicitamente opcional y separada.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ...common import paths
from ...common.camera_model import CameraModel
from ...common.config import WorldFrameConfig
from ...common.transforms import invert_T, make_T, rpy_deg_to_R, transform_points


def cargar_extrinsecos(cam_id: str, calib_dir: Path | None = None) -> np.ndarray:
    """Lee T_world_cam de calib/<cam_id>.yaml. Lanza si no esta calibrada."""
    calib_dir = calib_dir or paths.CALIB
    p = Path(calib_dir) / f"{cam_id}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"no existe {p}. Ejecutar scripts/vision/14_extrinsecos_camara.py")
    cam = CameraModel.load(p)
    if cam.T_world_cam is None:
        raise ValueError(
            f"{p} tiene intrinsecos pero NO extrinsecos. Sin la pose de la "
            "camara en {W}, la posicion 3D solo existe en el frame de la "
            "camara y no sirve para mandar el efector.")
    return cam.T_world_cam


@dataclass
class TransformadorFrames:
    """Aplica la cadena {C} -> {W} -> {B} y valida el volumen de trabajo."""
    T_world_cam: np.ndarray
    world: WorldFrameConfig | None = None
    T_base_world: np.ndarray | None = None

    @classmethod
    def desde_calib(cls, cam_id: str, world: WorldFrameConfig | None = None,
                    calib_dir: Path | None = None,
                    base_yaml: Path | None = None) -> "TransformadorFrames":
        T_base_world = None
        p = base_yaml or (paths.CONFIG / "robot_frame.yaml")
        if Path(p).exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            bw = raw.get("base_to_world")
            if bw:
                # El YAML declara la pose de {B} en {W}; se invierte para
                # obtener la transformada que lleva puntos de {W} a {B}.
                T_world_base = make_T(rpy_deg_to_R(bw.get("rpy_deg", [0, 0, 0])),
                                      bw.get("translation_m", [0, 0, 0]))
                T_base_world = invert_T(T_world_base)
        return cls(T_world_cam=cargar_extrinsecos(cam_id, calib_dir),
                   world=world, T_base_world=T_base_world)

    # ------------------------------------------------------------------ #
    @property
    def T_cam_world(self) -> np.ndarray:
        return invert_T(self.T_world_cam)

    def cam_a_world(self, P_cam: np.ndarray) -> np.ndarray:
        """Puntos (..., 3) de {C} a {W}. Propaga NaN."""
        return transform_points(self.T_world_cam, P_cam)

    def world_a_cam(self, P_world: np.ndarray) -> np.ndarray:
        return transform_points(self.T_cam_world, P_world)

    def world_a_base(self, P_world: np.ndarray) -> np.ndarray:
        """Puntos de {W} a la base del robot {B}. Requiere T_base_world."""
        if self.T_base_world is None:
            raise ValueError(
                "no hay T_base_world. Crear config/robot_frame.yaml con la pose "
                "de la base del H1-2 respecto a {W}, o hacer la calibracion "
                "mano-ojo (etapa V7 del roadmap de vision).")
        return transform_points(self.T_base_world, P_world)

    def cam_a_base(self, P_cam: np.ndarray) -> np.ndarray:
        return self.world_a_base(self.cam_a_world(P_cam))

    # ------------------------------------------------------------------ #
    def en_volumen(self, P_world: np.ndarray) -> np.ndarray | bool:
        """True si el punto cae dentro de W_capture (config/world_frame.yaml).

        Es un filtro de seguridad barato: descarta detecciones que, aun siendo
        geometricamente consistentes, caen fuera de la zona de trabajo --- una
        mandarina en una caja del suelo, o un reflejo naranja en la pared.
        """
        if self.world is None:
            return True
        return self.world.inside_volume(P_world)

    def describir(self) -> dict[str, Any]:
        from ...common.transforms import R_to_rpy_deg, split_T
        R, t = split_T(self.T_world_cam)
        return {
            "camara_en_world": {
                "posicion_m": [round(float(v), 4) for v in t],
                "rpy_deg": [round(float(v), 2) for v in R_to_rpy_deg(R)],
            },
            "tiene_T_base_world": self.T_base_world is not None,
            "volumen_definido": self.world is not None,
        }
