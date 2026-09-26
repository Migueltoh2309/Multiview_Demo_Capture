"""Rutas canonicas del proyecto.

Todo el codigo resuelve rutas a traves de aqui, nunca con rutas relativas al
cwd: asi los scripts funcionan igual desde cualquier directorio.
"""
from __future__ import annotations

from pathlib import Path

# .../Maestria_tesis/src/tesis_picking/common/paths.py -> .../Maestria_tesis
ROOT = Path(__file__).resolve().parents[3]

CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
CALIB_SHOTS = DATA / "calib_shots"
DATASET = DATA / "dataset"
CALIB = ROOT / "calib"
RESULTS = ROOT / "results"
FIGURAS = RESULTS / "figuras"
METRICAS = RESULTS / "metricas"
DOCS = ROOT / "docs"


def ensure(*dirs: Path) -> None:
    """Crea los directorios indicados si no existen."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
