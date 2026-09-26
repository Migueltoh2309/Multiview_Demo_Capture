"""Logging uniforme para scripts y modulos."""
from __future__ import annotations

import logging
import sys


def setup(level: int = logging.INFO) -> None:
    """Configura el logger raiz. Idempotente."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                                     datefmt="%H:%M:%S"))
    root.addHandler(h)
    root.setLevel(level)
