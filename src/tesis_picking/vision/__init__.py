"""Subsistema 2: percepcion RGB-D para el picking (objetivo especifico 2).

Cadena implementada:

    RealSense D435i
          |
    RGBDFrame (color + depth alineado + intrinsecos)
          |
    Detector 2D  -----------------> Deteccion2D (bbox, mascara, centroide)
          |
    Localizador3D (profundidad robusta + correccion de radio)
          |
    Objeto3D en frame de camara {C}
          |
    T_world_cam  (extrinseca por ChArUco)
          |
    Objeto3D en frame de trabajo {W}  ---> meta para el efector del H1-2

Comparte con `demostraciones_humanas` toda la capa geometrica de
`tesis_picking.common`: el mismo `CameraModel`/`CameraRig`, las mismas
convenciones SE(3) y el mismo {W}.
"""
from .realsense import RealSenseCamera, RGBDFrame, RealSenseConfig, describe_devices

__all__ = ["RealSenseCamera", "RGBDFrame", "RealSenseConfig", "describe_devices"]
