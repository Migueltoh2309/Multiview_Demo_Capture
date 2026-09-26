"""Etapas 1-3: apertura de camaras, captura simultanea y sincronizacion."""
from .camera_source import CameraSource, open_capture, probe_device
from .multi_capture import SyncedRig, FrameSet

__all__ = ["CameraSource", "open_capture", "probe_device", "SyncedRig", "FrameSet"]
