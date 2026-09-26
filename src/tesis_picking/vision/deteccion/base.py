"""Contrato comun de los detectores 2D.

Cualquier detector (color HSV hoy, YOLO manana) devuelve `Deteccion2D`. El
localizador 3D depende SOLO de esta interfaz, de modo que cambiar de detector
no toca la cadena de localizacion ni la de coordenadas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Deteccion2D:
    """Una fruta detectada en la imagen de color.

    Atributos:
        centro_px: (u, v) del centroide, en pixeles.
        bbox: (x, y, w, h) del rectangulo envolvente.
        mascara: (H, W) booleana. Es lo que de verdad usa el localizador 3D:
            muestrear la profundidad dentro de la MASCARA y no del bbox evita
            incluir el fondo que rodea a una fruta redonda (hasta un 21% del
            area del bbox de un circulo inscrito).
        score: confianza en [0, 1].
        clase: etiqueta ('mandarina').
        radio_px: radio aparente en pixeles, si el detector lo estima.
    """
    centro_px: np.ndarray
    bbox: tuple[int, int, int, int]
    mascara: np.ndarray
    score: float
    clase: str = "mandarina"
    radio_px: float = float("nan")
    metricas: dict[str, float] = field(default_factory=dict)

    @property
    def area_px(self) -> int:
        return int(np.count_nonzero(self.mascara))

    @property
    def u(self) -> float:
        return float(self.centro_px[0])

    @property
    def v(self) -> float:
        return float(self.centro_px[1])

    def resumen(self) -> dict[str, Any]:
        return {
            "clase": self.clase,
            "centro_px": [round(self.u, 1), round(self.v, 1)],
            "bbox": list(self.bbox),
            "area_px": self.area_px,
            "radio_px": round(float(self.radio_px), 1),
            "score": round(float(self.score), 3),
            **{k: round(float(v), 3) for k, v in self.metricas.items()},
        }


class Detector(Protocol):
    """Interfaz que debe cumplir todo detector 2D."""

    def detectar(self, color_bgr: np.ndarray) -> list[Deteccion2D]:
        """Devuelve las detecciones ordenadas de mayor a menor score."""
        ...
