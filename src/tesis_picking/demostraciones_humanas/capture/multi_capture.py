"""Captura simultanea de N camaras (etapas 2 y 3 del roadmap).

Estrategia de sincronizacion
----------------------------
Las webcams UVC no admiten trigger por hardware: cada una corre libre a su
propio reloj y con su propia fase. La sincronizacion es, por tanto, POR
SOFTWARE y de mejor esfuerzo:

  * Un hilo por camara. Todos esperan en una barrera.
  * Al liberarse la barrera, los N hilos llaman a `grab()` casi a la vez
    (dispersion de microsegundos).
  * `grab()` bloquea hasta que SU camara tiene el siguiente frame listo; el
    timestamp se toma justo al retornar, que es la mejor estimacion disponible
    del instante de llegada del frame.
  * La decodificacion (`retrieve()`, cara con MJPG) ocurre DESPUES, en paralelo,
    y no contamina el timestamp.

El desfase residual entre camaras esta acotado por el periodo de frame
(~33 ms a 30 FPS) y se MIDE en cada sesion con `SyncMetrics`. Ese numero es un
resultado experimental de la tesis (seccion 40 del roadmap), no un supuesto:
determina la incertidumbre temporal de las trayectorias reconstruidas y, por
tanto, el error espacial al reconstruir una muneca en movimiento.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...common.config import CameraSpec
from ...common.timebase import SyncMetrics
from .camera_source import CameraSource

log = logging.getLogger(__name__)


@dataclass
class FrameSet:
    """Un conjunto de frames de todas las camaras, nominalmente simultaneos."""
    frame_id: int
    frames: dict[str, np.ndarray | None]
    timestamps: dict[str, float]        # t_grab por camara, en perf_counter()
    t_trigger: float                    # instante en que se libero la barrera

    @property
    def cam_ids(self) -> list[str]:
        return list(self.frames.keys())

    @property
    def complete(self) -> bool:
        """True si TODAS las camaras entregaron frame."""
        return all(f is not None for f in self.frames.values())

    @property
    def valid_ids(self) -> list[str]:
        return [c for c, f in self.frames.items() if f is not None]

    @property
    def spread_s(self) -> float:
        """dt_max del roadmap: max_i t_i - min_i t_i sobre las camaras validas."""
        ts = [self.timestamps[c] for c in self.valid_ids
              if np.isfinite(self.timestamps.get(c, np.nan))]
        return float(max(ts) - min(ts)) if len(ts) >= 2 else float("nan")

    @property
    def t_mean(self) -> float:
        """Timestamp representativo del conjunto: media de las camaras validas."""
        ts = [self.timestamps[c] for c in self.valid_ids
              if np.isfinite(self.timestamps.get(c, np.nan))]
        return float(np.mean(ts)) if ts else float("nan")

    def sync_row(self) -> dict[str, Any]:
        """Fila para synchronization.csv (seccion 9 del roadmap)."""
        row: dict[str, Any] = {"frame_id": self.frame_id,
                               "t_trigger": round(self.t_trigger, 6)}
        for cid in self.frames:
            row[f"timestamp_{cid}"] = round(self.timestamps.get(cid, float("nan")), 6)
            row[f"ok_{cid}"] = int(self.frames[cid] is not None)
        row["dt_max_ms"] = round(self.spread_s * 1e3, 3)
        return row


class SyncedRig:
    """Rig de N camaras capturadas en paralelo con barrera de disparo.

    Uso:
        with SyncedRig(specs) as rig:
            rig.warmup()
            for _ in range(300):
                fs = rig.capture()
                ...
            print(rig.metrics.summary())
    """

    def __init__(self, specs: list[CameraSpec], *, open_timeout_s: float = 10.0):
        if not specs:
            raise ValueError("SyncedRig necesita al menos una camara")
        self.specs = list(specs)
        self.sources: dict[str, CameraSource] = {}
        self.open_timeout_s = open_timeout_s

        self._n = len(specs)
        self._start = threading.Barrier(self._n + 1)
        self._done = threading.Barrier(self._n + 1)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._results: dict[str, tuple[np.ndarray | None, float]] = {}
        self._results_lock = threading.Lock()
        self._frame_id = -1
        self._errors: dict[str, str] = {}

        self.metrics = SyncMetrics([s.id for s in self.specs])

    # ------------------------------------------------------------------ #
    def open(self) -> "SyncedRig":
        """Abre todas las camaras y lanza los hilos de captura.

        Las camaras se abren SECUENCIALMENTE a proposito: abrir varias webcams
        USB a la vez suele fallar por negociacion de ancho de banda del hub.
        """
        for spec in self.specs:
            log.info("abriendo %s (%s)...", spec.id, spec.device)
            src = CameraSource(spec).open()
            actual = src.actual_config()
            if (actual["width"], actual["height"]) != (spec.width, spec.height):
                log.warning("[%s] resolucion negociada %dx%d != pedida %dx%d",
                            spec.id, actual["width"], actual["height"],
                            spec.width, spec.height)
            self.sources[spec.id] = src

        self._stop.clear()
        for spec in self.specs:
            th = threading.Thread(target=self._worker, args=(spec.id,),
                                  name=f"cap-{spec.id}", daemon=True)
            th.start()
            self._threads.append(th)
        return self

    def close(self) -> None:
        """Detiene los hilos y libera las camaras."""
        self._stop.set()
        # Rompe las barreras para desbloquear cualquier hilo que este esperando.
        self._start.abort()
        self._done.abort()
        for th in self._threads:
            th.join(timeout=2.0)
        self._threads.clear()
        for src in self.sources.values():
            src.close()
        self.sources.clear()

    def __enter__(self) -> "SyncedRig":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def _worker(self, cam_id: str) -> None:
        src = self.sources[cam_id]
        while not self._stop.is_set():
            try:
                self._start.wait()
            except threading.BrokenBarrierError:
                return
            if self._stop.is_set():
                try:
                    self._done.wait()
                except threading.BrokenBarrierError:
                    pass
                return

            frame, t = None, float("nan")
            try:
                if src.grab():
                    t = src.t_grab
                    frame = src.retrieve()
            except Exception as exc:            # una camara caida no debe
                self._errors[cam_id] = str(exc)  # tumbar toda la sesion
                log.error("[%s] fallo de captura: %s", cam_id, exc)

            with self._results_lock:
                self._results[cam_id] = (frame, t)
            try:
                self._done.wait()
            except threading.BrokenBarrierError:
                return

    def capture(self) -> FrameSet:
        """Dispara un conjunto de frames y espera a que todas las camaras lo entreguen."""
        import time
        with self._results_lock:
            self._results.clear()
        t_trigger = time.perf_counter()
        self._start.wait()
        self._done.wait()
        with self._results_lock:
            res = dict(self._results)

        self._frame_id += 1
        fs = FrameSet(
            frame_id=self._frame_id,
            frames={cid: res.get(cid, (None, float("nan")))[0] for cid in self.ids},
            timestamps={cid: res.get(cid, (None, float("nan")))[1] for cid in self.ids},
            t_trigger=t_trigger,
        )
        self.metrics.add({c: fs.timestamps[c] for c in fs.valid_ids})
        return fs

    def warmup(self, n: int = 20) -> None:
        """Descarta los primeros conjuntos: el arranque nunca es representativo."""
        for _ in range(n):
            self.capture()
        self._frame_id = -1
        self.metrics = SyncMetrics(self.ids)

    # ------------------------------------------------------------------ #
    @property
    def ids(self) -> list[str]:
        return [s.id for s in self.specs]

    @property
    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    def actual_configs(self) -> dict[str, dict[str, Any]]:
        return {cid: src.actual_config() for cid, src in self.sources.items()}
