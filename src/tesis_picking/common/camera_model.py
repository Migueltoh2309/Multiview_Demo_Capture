"""Modelo de camara: intrinsecos, extrinsecos y matriz de proyeccion.

Etapas 5, 6 y 15 del roadmap. Un `CameraModel` reune todo lo necesario para
proyectar 3D->2D y para triangular, y sabe serializarse a YAML.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .transforms import invert_T, make_T, split_T


@dataclass
class Intrinsics:
    """K, D y la resolucion a la que fueron calibrados.

    K y D SOLO son validos a la resolucion `image_size`. Cambiar la resolucion
    de captura despues de calibrar invalida la calibracion.
    """
    K: np.ndarray                  # 3x3
    D: np.ndarray                  # (k,) coeficientes de distorsion
    image_size: tuple[int, int]    # (width, height)
    rms_reprojection_px: float = float("nan")
    n_views: int = 0
    model: str = "pinhole_radtan"  # radtan = radial-tangencial de OpenCV

    @property
    def fx(self) -> float: return float(self.K[0, 0])
    @property
    def fy(self) -> float: return float(self.K[1, 1])
    @property
    def cx(self) -> float: return float(self.K[0, 2])
    @property
    def cy(self) -> float: return float(self.K[1, 2])

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "image_width": int(self.image_size[0]),
            "image_height": int(self.image_size[1]),
            "camera_matrix": np.asarray(self.K, float).reshape(3, 3).tolist(),
            "distortion_coefficients": np.asarray(self.D, float).ravel().tolist(),
            "rms_reprojection_px": float(self.rms_reprojection_px),
            "n_views": int(self.n_views),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Intrinsics":
        return cls(
            K=np.asarray(d["camera_matrix"], float).reshape(3, 3),
            D=np.asarray(d["distortion_coefficients"], float).ravel(),
            image_size=(int(d["image_width"]), int(d["image_height"])),
            rms_reprojection_px=float(d.get("rms_reprojection_px", float("nan"))),
            n_views=int(d.get("n_views", 0)),
            model=d.get("model", "pinhole_radtan"),
        )


@dataclass
class CameraModel:
    """Camara calibrada: intrinsecos + pose en {W}."""
    cam_id: str
    intr: Intrinsics
    T_world_cam: np.ndarray | None = None   # 4x4; None si aun no hay extrinsecos

    # ---------------- extrinsecos ---------------- #
    @property
    def T_cam_world(self) -> np.ndarray:
        if self.T_world_cam is None:
            raise ValueError(f"{self.cam_id}: sin calibracion extrinseca")
        return invert_T(self.T_world_cam)

    @property
    def center_world(self) -> np.ndarray:
        """Centro optico C de la camara en {W}."""
        return split_T(self.T_world_cam)[1]

    @property
    def R_cam_world(self) -> np.ndarray:
        return split_T(self.T_cam_world)[0]

    @property
    def t_cam_world(self) -> np.ndarray:
        return split_T(self.T_cam_world)[1]

    @property
    def P(self) -> np.ndarray:
        """Matriz de proyeccion P = K [R|t] con [R|t] = T_cam_world (3x4).

        Proyecta puntos de {W} a PIXELES DISTORSIONADOS SOLO si la distorsion
        es despreciable. Para triangular, usar `normalized()` (ver abajo), que
        deja P = [R|t] y exige puntos ya undistorted+normalizados.
        """
        R, t = split_T(self.T_cam_world)
        return self.intr.K @ np.hstack([R, t.reshape(3, 1)])

    @property
    def P_normalized(self) -> np.ndarray:
        """P = [R|t] (3x4), para usar con coordenadas normalizadas de camara."""
        R, t = split_T(self.T_cam_world)
        return np.hstack([R, t.reshape(3, 1)])

    # ---------------- proyeccion / undistort ---------------- #
    def project(self, X_world: np.ndarray) -> np.ndarray:
        """Proyecta puntos (..., 3) de {W} a pixeles (..., 2) CON distorsion.

        Los puntos detras de la camara devuelven NaN: proyectarlos daria un
        pixel aparentemente valido y contaminaria el error de reproyeccion.
        """
        X = np.asarray(X_world, dtype=float)
        shape = X.shape[:-1]
        flat = X.reshape(-1, 3)
        out = np.full((flat.shape[0], 2), np.nan)
        finite = np.all(np.isfinite(flat), axis=1)
        if not np.any(finite):
            return out.reshape(*shape, 2)

        R, t = split_T(self.T_cam_world)
        Xc = flat[finite] @ R.T + t
        ok = Xc[:, 2] > 1e-6
        idx = np.where(finite)[0][ok]
        if idx.size:
            rvec, _ = cv2.Rodrigues(np.eye(3))
            px, _ = cv2.projectPoints(
                Xc[ok].reshape(-1, 1, 3), rvec, np.zeros(3),
                self.intr.K, self.intr.D)
            out[idx] = px.reshape(-1, 2)
        return out.reshape(*shape, 2)

    def undistort_points(self, uv: np.ndarray) -> np.ndarray:
        """Pixeles distorsionados (..., 2) -> pixeles ideales (sin distorsion)."""
        return self._undistort(uv, P=self.intr.K)

    def normalize_points(self, uv: np.ndarray) -> np.ndarray:
        """Pixeles distorsionados (..., 2) -> coords normalizadas de camara."""
        return self._undistort(uv, P=None)

    def _undistort(self, uv: np.ndarray, P: np.ndarray | None) -> np.ndarray:
        uv = np.asarray(uv, dtype=float)
        shape = uv.shape[:-1]
        flat = uv.reshape(-1, 2)
        out = np.full_like(flat, np.nan)
        finite = np.all(np.isfinite(flat), axis=1)
        if np.any(finite):
            src = flat[finite].reshape(-1, 1, 2).astype(np.float64)
            dst = cv2.undistortPoints(src, self.intr.K, self.intr.D, P=P)
            out[finite] = dst.reshape(-1, 2)
        return out.reshape(*shape, 2)

    # ---------------- serializacion ---------------- #
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"camera_id": self.cam_id, **self.intr.to_dict()}
        if self.T_world_cam is not None:
            R, t = split_T(self.T_world_cam)
            d["T_world_cam"] = np.asarray(self.T_world_cam, float).tolist()
            d["translation_world_m"] = t.tolist()
            d["rotation_world"] = R.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraModel":
        T = np.asarray(d["T_world_cam"], float).reshape(4, 4) if "T_world_cam" in d else None
        return cls(cam_id=d["camera_id"], intr=Intrinsics.from_dict(d), T_world_cam=T)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=None)

    @classmethod
    def load(cls, path: Path) -> "CameraModel":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


class CameraRig:
    """Conjunto de camaras calibradas que comparten el mismo {W}."""

    def __init__(self, cameras: list[CameraModel]):
        if not cameras:
            raise ValueError("un rig necesita al menos una camara")
        self.cameras = list(cameras)
        self._by_id = {c.cam_id: c for c in self.cameras}
        if len(self._by_id) != len(self.cameras):
            raise ValueError("ids de camara duplicados en el rig")

    def __len__(self) -> int: return len(self.cameras)
    def __iter__(self): return iter(self.cameras)
    def __getitem__(self, cam_id: str) -> CameraModel: return self._by_id[cam_id]

    @property
    def ids(self) -> list[str]:
        return [c.cam_id for c in self.cameras]

    @property
    def calibrated(self) -> bool:
        return all(c.T_world_cam is not None for c in self.cameras)

    def subset(self, ids: list[str]) -> "CameraRig":
        return CameraRig([self._by_id[i] for i in ids])

    def baselines_m(self) -> dict[tuple[str, str], float]:
        """Distancia entre centros opticos de cada par. Util para diagnostico:
        una baseline demasiado corta degrada la precision en profundidad."""
        out: dict[tuple[str, str], float] = {}
        for i, a in enumerate(self.cameras):
            for b in self.cameras[i + 1:]:
                out[(a.cam_id, b.cam_id)] = float(
                    np.linalg.norm(a.center_world - b.center_world))
        return out

    @classmethod
    def load(cls, calib_dir: Path, cam_ids: list[str]) -> "CameraRig":
        calib_dir = Path(calib_dir)
        cams = []
        for cid in cam_ids:
            p = calib_dir / f"{cid}.yaml"
            if not p.exists():
                raise FileNotFoundError(
                    f"falta la calibracion de '{cid}' en {p}. "
                    "Ejecutar 06_calibrate_intrinsics.py y 08_calibrate_extrinsics.py")
            cams.append(CameraModel.load(p))
        return cls(cams)

    def save(self, calib_dir: Path) -> None:
        for c in self.cameras:
            c.save(Path(calib_dir) / f"{c.cam_id}.yaml")
