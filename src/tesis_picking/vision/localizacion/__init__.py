"""Localizacion 3D del objeto detectado y cambio al frame de trabajo {W}."""
from .profundidad import Localizador3D, Objeto3D, muestrear_profundidad
from .frames import TransformadorFrames, cargar_extrinsecos

__all__ = ["Localizador3D", "Objeto3D", "muestrear_profundidad",
           "TransformadorFrames", "cargar_extrinsecos"]
