"""Grabacion de una sesion/demostracion a disco (etapas 2, 3 y 26 del roadmap).

Estructura generada, siguiendo la seccion 38 del roadmap:

    <out_dir>/
      cam1.mp4
      cam2.mp4
      cam3.mp4
      synchronization.csv     # frame_id, timestamp_camN, ok_camN, dt_max_ms
      metadata.yaml           # config real de cada camara + metricas de sync

La codificacion corre en un hilo POR CAMARA, alimentado por una cola: si el
encoder se retrasa, el hilo de captura no se bloquea y no se pierde el ritmo de
adquisicion. Si la cola se llena, se descarta el frame y se contabiliza --- es
preferible perder un frame y saberlo, que falsear la base de tiempo.
"""
from __future__ import annotations

import csv
import logging
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from ...common.timebase import frame_interval_stats
from .multi_capture import FrameSet, SyncedRig

log = logging.getLogger(__name__)


class _VideoWriterThread(threading.Thread):
    """Escribe frames de una camara en un hilo aparte."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float,
                 fourcc: str = "mp4v", maxsize: int = 120):
        super().__init__(daemon=True, name=f"writer-{path.stem}")
        self.path = Path(path)
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.written = 0
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*fourcc), float(fps), size)
        if not self._writer.isOpened():
            raise RuntimeError(f"no se pudo abrir el VideoWriter en {self.path}")

    def put(self, frame: np.ndarray) -> None:
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            self._writer.write(item)
            self.written += 1
        self._writer.release()

    def stop(self) -> None:
        self.q.put(None)
        self.join(timeout=30.0)


class SessionRecorder:
    """Graba una sesion sincronizada del rig."""

    def __init__(self, rig: SyncedRig, out_dir: Path, *,
                 save_video: bool = True, fourcc: str = "mp4v",
                 metadata: dict[str, Any] | None = None):
        self.rig = rig
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.save_video = save_video
        self.fourcc = fourcc
        self.extra_metadata = metadata or {}

        self._writers: dict[str, _VideoWriterThread] = {}
        self._sync_rows: list[dict[str, Any]] = []
        self._ts: dict[str, list[float]] = {c: [] for c in rig.ids}
        self._t0_wall = datetime.now().astimezone()
        self._started = False

    # ------------------------------------------------------------------ #
    def _ensure_writers(self, fs: FrameSet) -> None:
        if self._writers or not self.save_video:
            return
        for cid, frame in fs.frames.items():
            if frame is None:
                continue
            h, w = frame.shape[:2]
            spec = next(s for s in self.rig.specs if s.id == cid)
            wt = _VideoWriterThread(self.out_dir / f"{cid}.mp4", (w, h),
                                    spec.fps, self.fourcc)
            wt.start()
            self._writers[cid] = wt

    def add(self, fs: FrameSet) -> None:
        """Registra un FrameSet en la sesion."""
        if not self._started:
            self._t0_wall = datetime.now().astimezone()
            self._started = True
        self._ensure_writers(fs)
        self._sync_rows.append(fs.sync_row())
        for cid, frame in fs.frames.items():
            if np.isfinite(fs.timestamps.get(cid, np.nan)):
                self._ts[cid].append(fs.timestamps[cid])
            if frame is not None and cid in self._writers:
                self._writers[cid].put(frame)

    # ------------------------------------------------------------------ #
    def close(self) -> dict[str, Any]:
        """Cierra los videos y escribe synchronization.csv y metadata.yaml."""
        for wt in self._writers.values():
            wt.stop()

        sync_path = self.out_dir / "synchronization.csv"
        if self._sync_rows:
            with open(sync_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(self._sync_rows[0].keys()))
                w.writeheader()
                w.writerows(self._sync_rows)

        meta = self.build_metadata()
        with open(self.out_dir / "metadata.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
        log.info("sesion guardada en %s (%d frames)", self.out_dir, len(self._sync_rows))
        return meta

    def build_metadata(self) -> dict[str, Any]:
        per_cam: dict[str, Any] = {}
        for cid in self.rig.ids:
            spec = next(s for s in self.rig.specs if s.id == cid)
            entry: dict[str, Any] = {
                "device": str(spec.device),
                "role": spec.role,
                "requested": {"width": spec.width, "height": spec.height,
                              "fps": spec.fps, "fourcc": spec.fourcc},
                "actual": self.rig.actual_configs().get(cid, {}),
                "timing": {k: round(v, 3) for k, v in
                           frame_interval_stats(self._ts[cid]).items()},
            }
            src = self.rig.sources.get(cid)
            if src is not None:
                entry["stale_frames_dropped"] = src.stale_dropped
            if cid in self._writers:
                entry["video"] = {"frames_written": self._writers[cid].written,
                                  "frames_dropped_by_encoder": self._writers[cid].dropped}
            per_cam[cid] = entry

        return {
            "session": {
                "name": self.out_dir.name,
                "started_wall_clock": self._t0_wall.isoformat(),
                "n_framesets": len(self._sync_rows),
                "n_framesets_incompletos": sum(
                    1 for r in self._sync_rows
                    if any(r.get(f"ok_{c}") == 0 for c in self.rig.ids)),
            },
            "cameras": per_cam,
            "synchronization": {k: round(v, 4) for k, v in
                                self.rig.metrics.summary().items()},
            "capture_errors": self.rig.errors,
            **self.extra_metadata,
        }
