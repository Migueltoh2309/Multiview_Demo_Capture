"""Etapas 4-7: tablero ChArUco, calibracion intrinseca, extrinseca y validacion."""
from .charuco import CharucoBoardWrapper, BoardDetection
from .intrinsics import calibrate_intrinsics, IntrinsicsResult
from .extrinsics import calibrate_extrinsics, ExtrinsicsResult

__all__ = [
    "CharucoBoardWrapper", "BoardDetection",
    "calibrate_intrinsics", "IntrinsicsResult",
    "calibrate_extrinsics", "ExtrinsicsResult",
]
