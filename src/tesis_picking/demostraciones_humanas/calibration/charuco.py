"""Tablero ChArUco: generacion y deteccion (etapa 4 del roadmap).

Se usa ChArUco y no un damero simple porque los marcadores ArUco identifican
cada esquina de forma univoca: el tablero puede aparecer PARCIALMENTE visible u
ocluido y aun asi aportar correspondencias validas. Con tres camaras que miran
el mismo volumen desde angulos muy distintos, eso deja de ser un lujo.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...common.config import CharucoConfig


def get_dictionary(name: str) -> cv2.aruco.Dictionary:
    """Resuelve el nombre del diccionario (p.ej. 'DICT_5X5_1000')."""
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"diccionario ArUco desconocido: '{name}'")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


@dataclass
class BoardDetection:
    """Esquinas ChArUco detectadas en una imagen."""
    charuco_corners: np.ndarray | None   # (M, 1, 2) float32, subpixel
    charuco_ids: np.ndarray | None       # (M, 1) int32
    marker_corners: list | None
    marker_ids: np.ndarray | None
    image_size: tuple[int, int]          # (width, height)

    @property
    def n_corners(self) -> int:
        return 0 if self.charuco_ids is None else int(len(self.charuco_ids))

    def is_usable(self, min_corners: int = 8) -> bool:
        """Una vista con pocas esquinas aporta mas ruido que informacion."""
        return self.n_corners >= min_corners

    def ids_flat(self) -> np.ndarray:
        return (np.empty(0, dtype=int) if self.charuco_ids is None
                else self.charuco_ids.reshape(-1).astype(int))

    def corners_flat(self) -> np.ndarray:
        return (np.empty((0, 2), dtype=float) if self.charuco_corners is None
                else self.charuco_corners.reshape(-1, 2).astype(float))

    def corner_by_id(self) -> dict[int, np.ndarray]:
        """{id_esquina: (u, v)} --- la clave para cruzar vistas entre camaras."""
        return dict(zip(self.ids_flat().tolist(), self.corners_flat()))


class CharucoBoardWrapper:
    """Envuelve `cv2.aruco.CharucoBoard` con generacion, deteccion y utilidades."""

    def __init__(self, cfg: CharucoConfig):
        self.cfg = cfg
        self.dictionary = get_dictionary(cfg.dictionary)
        self.board = cv2.aruco.CharucoBoard(
            (cfg.squares_x, cfg.squares_y),
            float(cfg.square_length_m), float(cfg.marker_length_m),
            self.dictionary,
        )
        self.board.setLegacyPattern(bool(cfg.legacy_pattern))

        det_params = cv2.aruco.DetectorParameters()
        # Refinamiento por contorno: mejora notablemente la precision subpixel
        # de las esquinas, que es lo que acaba limitando la exactitud metrica.
        det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        charuco_params = cv2.aruco.CharucoParameters()
        self.detector = cv2.aruco.CharucoDetector(
            self.board, charuco_params, det_params)

    # ------------------------------------------------------------------ #
    @property
    def n_corners(self) -> int:
        return self.cfg.n_corners

    def object_points(self) -> np.ndarray:
        """Coordenadas 3D (N, 3) de todas las esquinas en el frame del tablero.

        Z = 0 para todas: el tablero es plano por construccion.
        """
        return np.asarray(self.board.getChessboardCorners(), dtype=np.float64)

    def object_point(self, corner_id: int) -> np.ndarray:
        return self.object_points()[int(corner_id)]

    # ------------------------------------------------------------------ #
    def generate_image(self, out_path: Path, *, dpi: int = 300,
                       margin_mm: float = 10.0) -> tuple[Path, dict]:
        """Renderiza el tablero a PNG con escala fisica exacta para imprimir."""
        px_per_m = dpi / 0.0254
        w_m = self.cfg.squares_x * self.cfg.square_length_m
        h_m = self.cfg.squares_y * self.cfg.square_length_m
        margin_px = int(round(margin_mm / 1000.0 * px_per_m))
        w_px = int(round(w_m * px_per_m))
        h_px = int(round(h_m * px_per_m))

        img = self.board.generateImage((w_px, h_px), marginSize=0, borderBits=1)
        img = cv2.copyMakeBorder(img, margin_px, margin_px, margin_px, margin_px,
                                 cv2.BORDER_CONSTANT, value=255)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)

        info = {
            "file": str(out_path),
            "dictionary": self.cfg.dictionary,
            "squares_x": self.cfg.squares_x,
            "squares_y": self.cfg.squares_y,
            "square_length_mm": self.cfg.square_length_m * 1000,
            "marker_length_mm": self.cfg.marker_length_m * 1000,
            "board_size_mm": [round(w_m * 1000, 2), round(h_m * 1000, 2)],
            "image_px": [img.shape[1], img.shape[0]],
            "dpi": dpi,
            "n_charuco_corners": self.n_corners,
        }
        return out_path, info

    # ------------------------------------------------------------------ #
    def detect(self, image: np.ndarray) -> BoardDetection:
        """Detecta el tablero. Acepta imagen en color o en escala de grises."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        cc, ci, mc, mi = self.detector.detectBoard(gray)
        return BoardDetection(cc, ci, mc, mi, (w, h))

    def draw(self, image: np.ndarray, det: BoardDetection) -> np.ndarray:
        """Dibuja las detecciones sobre una copia de la imagen (para preview)."""
        vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if det.marker_ids is not None and len(det.marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, det.marker_corners, det.marker_ids)
        if det.charuco_ids is not None and len(det.charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(
                vis, det.charuco_corners, det.charuco_ids, (0, 255, 255))
        return vis

    # ------------------------------------------------------------------ #
    def match_points(self, det: BoardDetection) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (objectPoints (M,3), imagePoints (M,2)) para calibrar/PnP."""
        if det.charuco_ids is None or det.n_corners == 0:
            return np.empty((0, 3)), np.empty((0, 2))
        obj, img = self.board.matchImagePoints(det.charuco_corners, det.charuco_ids)
        if obj is None or img is None:
            return np.empty((0, 3)), np.empty((0, 2))
        return (np.asarray(obj, dtype=np.float64).reshape(-1, 3),
                np.asarray(img, dtype=np.float64).reshape(-1, 2))

    def estimate_pose(self, det: BoardDetection, K: np.ndarray, D: np.ndarray,
                      *, min_corners: int = 6) -> tuple[np.ndarray, np.ndarray] | None:
        """Pose del tablero vista por la camara: devuelve (rvec, tvec) o None.

        Usa IPPE_SQUARE-libre (`SOLVEPNP_ITERATIVE` sobre puntos coplanares) con
        refinamiento; devuelve None si no hay esquinas suficientes.
        """
        obj, img = self.match_points(det)
        if len(obj) < min_corners:
            return None
        ok, rvec, tvec = cv2.solvePnP(
            obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, D,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(
            obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, D, rvec, tvec)
        return rvec.reshape(3), tvec.reshape(3)
