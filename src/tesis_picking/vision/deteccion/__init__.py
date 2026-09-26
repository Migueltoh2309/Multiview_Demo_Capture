"""Deteccion 2D de fruta en la imagen de color."""
from .base import Deteccion2D, Detector
from .color_hsv import DetectorColorHSV, ObjetoConfig

__all__ = ["Deteccion2D", "Detector", "DetectorColorHSV", "ObjetoConfig"]
