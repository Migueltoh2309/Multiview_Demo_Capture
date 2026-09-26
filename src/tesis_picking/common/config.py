"""Carga y validacion de los YAML de config/."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import paths
from .transforms import rpy_deg_to_R, make_T


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------- #
# Camaras
# --------------------------------------------------------------------------- #
@dataclass
class CameraSpec:
    """Configuracion de adquisicion de una camara."""
    id: str
    device: int | str
    width: int
    height: int
    fps: float
    fourcc: str
    buffersize: int
    controls: dict[str, Any]
    backend: str = "V4L2"
    role: str = ""
    serial: str | None = None
    notes: str = ""

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass
class CamerasConfig:
    cameras: list[CameraSpec]

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.cameras]

    def by_id(self, cam_id: str) -> CameraSpec:
        for c in self.cameras:
            if c.id == cam_id:
                return c
        raise KeyError(f"camara '{cam_id}' no esta en cameras.yaml (hay: {self.ids})")

    def subset(self, ids: list[str] | None) -> "CamerasConfig":
        if not ids:
            return self
        return CamerasConfig([self.by_id(i) for i in ids])


def load_cameras(path: Path | None = None) -> CamerasConfig:
    path = path or paths.CONFIG / "cameras.yaml"
    raw = load_yaml(path)
    backend = raw.get("backend", "V4L2")
    d = raw.get("defaults", {}) or {}
    d_controls = d.get("controls", {}) or {}

    specs: list[CameraSpec] = []
    seen: set[str] = set()
    for entry in raw.get("cameras", []) or []:
        cam_id = entry["id"]
        if cam_id in seen:
            raise ValueError(f"id de camara duplicado en {path}: '{cam_id}'")
        seen.add(cam_id)
        controls = {**d_controls, **(entry.get("controls") or {})}
        specs.append(CameraSpec(
            id=cam_id,
            device=entry["device"],
            width=int(entry.get("width", d.get("width", 1280))),
            height=int(entry.get("height", d.get("height", 720))),
            fps=float(entry.get("fps", d.get("fps", 30))),
            fourcc=str(entry.get("fourcc", d.get("fourcc", "MJPG"))),
            buffersize=int(entry.get("buffersize", d.get("buffersize", 1))),
            controls=controls,
            backend=entry.get("backend", backend),
            role=entry.get("role", ""),
            serial=entry.get("serial"),
            notes=entry.get("notes", ""),
        ))
    if not specs:
        raise ValueError(f"{path} no define ninguna camara")
    return CamerasConfig(specs)


# --------------------------------------------------------------------------- #
# ChArUco
# --------------------------------------------------------------------------- #
@dataclass
class CharucoConfig:
    dictionary: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    legacy_pattern: bool = False
    print_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.marker_length_m < self.square_length_m:
            raise ValueError(
                "marker_length_m debe ser > 0 y estrictamente menor que "
                f"square_length_m (recibido {self.marker_length_m} vs "
                f"{self.square_length_m})")

    @property
    def n_corners(self) -> int:
        """Numero de esquinas interiores del damero (los puntos ChArUco)."""
        return (self.squares_x - 1) * (self.squares_y - 1)


def load_charuco(path: Path | None = None) -> CharucoConfig:
    path = path or paths.CONFIG / "charuco.yaml"
    raw = load_yaml(path)
    return CharucoConfig(
        dictionary=raw["dictionary"],
        squares_x=int(raw["squares_x"]),
        squares_y=int(raw["squares_y"]),
        square_length_m=float(raw["square_length_m"]),
        marker_length_m=float(raw["marker_length_m"]),
        legacy_pattern=bool(raw.get("legacy_pattern", False)),
        print_info=raw.get("print", {}) or {},
    )


# --------------------------------------------------------------------------- #
# World frame
# --------------------------------------------------------------------------- #
@dataclass
class WorldFrameConfig:
    """Define {W} y el volumen de captura W_capture."""
    T_world_board: np.ndarray          # 4x4, pose del tablero-origen en {W}
    volume_min: np.ndarray             # (3,)
    volume_max: np.ndarray             # (3,)

    def inside_volume(self, X: np.ndarray) -> np.ndarray:
        """Mascara booleana: puntos (..., 3) dentro de W_capture."""
        X = np.asarray(X, dtype=float)
        ok = np.all((X >= self.volume_min) & (X <= self.volume_max), axis=-1)
        return ok & np.all(np.isfinite(X), axis=-1)


def load_world_frame(path: Path | None = None) -> WorldFrameConfig:
    path = path or paths.CONFIG / "world_frame.yaml"
    raw = load_yaml(path)
    b2w = raw.get("board_to_world", {}) or {}
    t = np.asarray(b2w.get("translation_m", [0, 0, 0]), dtype=float)
    rpy = np.asarray(b2w.get("rpy_deg", [0, 0, 0]), dtype=float)
    vol = raw.get("capture_volume", {}) or {}
    xr = vol.get("x", [-1, 1]); yr = vol.get("y", [-1, 1]); zr = vol.get("z", [-1, 2])
    return WorldFrameConfig(
        T_world_board=make_T(rpy_deg_to_R(rpy), t),
        volume_min=np.array([xr[0], yr[0], zr[0]], dtype=float),
        volume_max=np.array([xr[1], yr[1], zr[1]], dtype=float),
    )
