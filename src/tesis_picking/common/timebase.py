"""Base de tiempo y metricas de sincronizacion (etapa 3 del roadmap).

Regla del proyecto: TODOS los timestamps de adquisicion usan
`time.perf_counter()` (reloj monotono de alta resolucion). `time.time()` puede
saltar hacia atras si NTP ajusta el reloj durante una demostracion.
El instante de arranque en tiempo de pared se guarda una sola vez en el
metadata, para poder relacionar la sesion con eventos externos.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

now = time.perf_counter


@dataclass
class SyncMetrics:
    """Acumula los desfases entre camaras de un conjunto de frames.

    Para cada frame k se registra el vector de timestamps de las N camaras y se
    calcula el spread  dt_max(k) = max_i t_i(k) - min_i t_i(k), que es la
    generalizacion a N camaras de la metrica |dt_ij| del roadmap.
    """
    cam_ids: list[str]
    spreads_s: list[float] = field(default_factory=list)
    pairwise_s: dict[tuple[str, str], list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pairwise_s:
            self.pairwise_s = {
                (a, b): []
                for i, a in enumerate(self.cam_ids) for b in self.cam_ids[i + 1:]
            }

    def add(self, timestamps: dict[str, float]) -> float:
        """Registra un frame y devuelve su spread en segundos (NaN si falta alguna)."""
        ts = [timestamps.get(c, np.nan) for c in self.cam_ids]
        if not all(np.isfinite(t) for t in ts):
            self.spreads_s.append(np.nan)
            return float("nan")
        spread = float(max(ts) - min(ts))
        self.spreads_s.append(spread)
        for (a, b), lst in self.pairwise_s.items():
            lst.append(float(timestamps[a] - timestamps[b]))
        return spread

    def summary(self) -> dict[str, float]:
        """Metricas de la seccion 40 del roadmap, en MILISEGUNDOS."""
        s = np.asarray(self.spreads_s, dtype=float)
        valid = s[np.isfinite(s)]
        out: dict[str, float] = {
            "n_frames": float(len(s)),
            "n_frames_incompletos": float(len(s) - len(valid)),
        }
        if valid.size:
            out.update({
                "dt_mean_ms": float(np.mean(valid) * 1e3),
                "dt_std_ms": float(np.std(valid) * 1e3),
                "dt_p95_ms": float(np.percentile(valid, 95) * 1e3),
                "dt_max_ms": float(np.max(valid) * 1e3),
            })
        for (a, b), lst in self.pairwise_s.items():
            v = np.asarray(lst, dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                out[f"dt_{a}_{b}_mean_ms"] = float(np.mean(v) * 1e3)
                out[f"dt_{a}_{b}_std_ms"] = float(np.std(v) * 1e3)
        return out


def measured_fps(timestamps: list[float] | np.ndarray) -> float:
    """FPS efectivo a partir de una serie de timestamps monotonos."""
    t = np.asarray(timestamps, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return float("nan")
    span = t[-1] - t[0]
    return float((t.size - 1) / span) if span > 0 else float("nan")


def frame_interval_stats(timestamps: list[float] | np.ndarray) -> dict[str, float]:
    """Estadisticas del intervalo entre frames: detecta frames perdidos."""
    t = np.asarray(timestamps, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return {"n": float(t.size)}
    dt = np.diff(t) * 1e3
    med = float(np.median(dt))
    return {
        "n": float(t.size),
        "fps": measured_fps(t),
        "dt_median_ms": med,
        "dt_std_ms": float(np.std(dt)),
        "dt_max_ms": float(np.max(dt)),
        # Un intervalo > 1.5x la mediana implica al menos un frame perdido.
        "dropped_est": float(np.sum(dt > 1.5 * med)) if med > 0 else float("nan"),
    }
