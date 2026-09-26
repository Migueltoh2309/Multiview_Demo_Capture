"""Apertura y configuracion de una camara USB (etapa 1 del roadmap).

Puntos criticos aprendidos con webcams UVC en Linux, codificados aqui:

1. El orden importa: FOURCC ANTES que la resolucion. Si se fija la resolucion
   con el formato por defecto (YUYV), el driver puede negociar 1280x720@10 y
   luego ignorar el cambio a MJPG.
2. `CAP_PROP_BUFFERSIZE` NO debe ponerse a 1. Medido en este equipo sobre el
   driver uvcvideo: con buffersize=1 el framerate cae a la MITAD (15 FPS en
   lugar de 30 a 1280x720 MJPG), porque sin doble buffer el driver no puede
   encolar el frame siguiente mientras el espacio de usuario retiene el actual.
   El minimo seguro es 2.
3. Fijar el buffer pequeno NO impide que lleguen frames rancios: se comprobo que
   tras una pausa de 500 ms, `grab()` retorna en 0.03 ms --- es decir, entrega un
   frame ya encolado, no uno nuevo. Si el bucle de captura se retrasa, los
   timestamps dejarian de corresponder al instante real de la escena. Por eso
   `CameraSource.grab()` descarta explicitamente los frames encolados: ver
   `flush_stale`.
4. El autofoco DEBE desactivarse: si el foco cambia despues de calibrar, K deja
   de ser valida.
5. `set()` devuelve True aunque el driver ignore el valor. Por eso siempre se
   RELEE la propiedad y se compara con lo pedido.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ...common.config import CameraSpec

log = logging.getLogger(__name__)

BACKENDS = {"V4L2": cv2.CAP_V4L2, "ANY": cv2.CAP_ANY, "GSTREAMER": cv2.CAP_GSTREAMER}

# Mapea las claves de `controls` en cameras.yaml a propiedades de OpenCV.
_CONTROL_PROPS: dict[str, int] = {
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "auto_wb": cv2.CAP_PROP_AUTO_WB,
    "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
    "focus": cv2.CAP_PROP_FOCUS,
    "gain": cv2.CAP_PROP_GAIN,
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
}

# Estos controles deben aplicarse en este orden: desactivar el modo automatico
# antes de escribir el valor manual, o el driver lo revierte.
_CONTROL_ORDER = ["autofocus", "focus", "auto_exposure", "exposure",
                  "auto_wb", "wb_temperature", "gain", "brightness",
                  "contrast", "saturation"]


def fourcc_to_str(v: float) -> str:
    i = int(v)
    return "".join(chr((i >> (8 * k)) & 0xFF) for k in range(4)) if i else ""


@dataclass
class DeviceReport:
    """Lo que la camara realmente entrego, frente a lo que se pidio (etapa 1)."""
    cam_id: str
    device: int | str
    opened: bool
    requested: dict[str, Any] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)
    measured_fps: float = float("nan")
    n_frames_read: int = 0
    error: str = ""

    @property
    def resolution_ok(self) -> bool:
        return (self.actual.get("width") == self.requested.get("width")
                and self.actual.get("height") == self.requested.get("height"))

    @property
    def fps_ok(self) -> bool:
        """El FPS MEDIDO es el que cuenta; el declarado por el driver miente."""
        req = self.requested.get("fps", 0) or 0
        return bool(np.isfinite(self.measured_fps) and self.measured_fps >= 0.9 * req)

    def as_row(self) -> dict[str, Any]:
        return {
            "camera_id": self.cam_id,
            "device": str(self.device),
            "opened": self.opened,
            "req_resolution": f"{self.requested.get('width')}x{self.requested.get('height')}",
            "act_resolution": f"{self.actual.get('width')}x{self.actual.get('height')}",
            "req_fourcc": self.requested.get("fourcc", ""),
            "act_fourcc": self.actual.get("fourcc", ""),
            "req_fps": self.requested.get("fps"),
            "driver_fps": self.actual.get("fps"),
            "measured_fps": round(self.measured_fps, 2),
            "resolution_ok": self.resolution_ok,
            "fps_ok": self.fps_ok,
            "error": self.error,
        }


def open_capture(spec: CameraSpec) -> cv2.VideoCapture:
    """Abre y configura un `VideoCapture` segun `spec`. Lanza si no abre."""
    backend = BACKENDS.get(str(spec.backend).upper(), cv2.CAP_ANY)
    dev = spec.device if isinstance(spec.device, int) else str(spec.device)
    cap = cv2.VideoCapture(dev, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"[{spec.id}] no se pudo abrir el device {spec.device!r}. "
            "Comprobar que existe (ls /dev/video*), que el usuario esta en el "
            "grupo 'video' y que ninguna otra aplicacion la esta usando.")

    # 1) FOURCC primero (ver docstring del modulo).
    if spec.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*spec.fourcc))
    # 2) resolucion y FPS.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
    cap.set(cv2.CAP_PROP_FPS, spec.fps)
    # 3) buffer: 2 es el minimo seguro (ver punto 2 del docstring del modulo).
    bs = max(2, int(spec.buffersize))
    if bs != spec.buffersize:
        log.warning("[%s] buffersize=%s elevado a 2: con 1 el driver puede "
                    "entregar la mitad de FPS", spec.id, spec.buffersize)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, bs)
    except cv2.error:
        pass
    # 4) controles de imagen, en orden.
    apply_controls(cap, spec)
    return cap


def apply_controls(cap: cv2.VideoCapture, spec: CameraSpec) -> None:
    """Aplica los controles de `spec.controls`, saltando los que sean None."""
    for name in _CONTROL_ORDER:
        if name not in spec.controls:
            continue
        value = spec.controls[name]
        if value is None:
            continue
        prop = _CONTROL_PROPS.get(name)
        if prop is None:
            log.warning("[%s] control desconocido '%s', ignorado", spec.id, name)
            continue
        if not cap.set(prop, float(value)):
            log.warning("[%s] el driver rechazo %s=%s", spec.id, name, value)


def read_actual(cap: cv2.VideoCapture) -> dict[str, Any]:
    """Relee de la camara la configuracion que quedo realmente activa."""
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 2),
        "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
        "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "autofocus": cap.get(cv2.CAP_PROP_AUTOFOCUS),
        "auto_wb": cap.get(cv2.CAP_PROP_AUTO_WB),
    }


def probe_device(spec: CameraSpec, n_frames: int = 90,
                 warmup: int = 15) -> DeviceReport:
    """Etapa 1: abre la camara, mide el FPS real y reporta lo negociado.

    El warmup es necesario: las primeras lecturas tras abrir el device son
    irregulares (negociacion USB + auto-ajustes) y sesgarian el FPS medido.
    """
    rep = DeviceReport(
        cam_id=spec.id, device=spec.device, opened=False,
        requested={"width": spec.width, "height": spec.height,
                   "fps": spec.fps, "fourcc": spec.fourcc},
    )
    cap = None
    try:
        cap = open_capture(spec)
        rep.opened = True
        rep.actual = read_actual(cap)

        for _ in range(warmup):
            cap.read()

        ts: list[float] = []
        for _ in range(n_frames):
            ok, frame = cap.read()
            t = time.perf_counter()
            if not ok or frame is None:
                break
            ts.append(t)
        rep.n_frames_read = len(ts)
        if len(ts) >= 2:
            span = ts[-1] - ts[0]
            rep.measured_fps = (len(ts) - 1) / span if span > 0 else float("nan")
        if rep.n_frames_read < n_frames:
            rep.error = f"solo se leyeron {rep.n_frames_read}/{n_frames} frames"
    except Exception as exc:
        rep.error = str(exc)
    finally:
        if cap is not None:
            cap.release()
    return rep


class CameraSource:
    """Envoltura de un `VideoCapture` con timestamps y grab/retrieve separados.

    La separacion grab/retrieve es lo que hace posible la sincronizacion por
    software: `grab()` es barato y dispara la captura en el driver, mientras que
    `retrieve()` decodifica (caro, sobre todo con MJPG). Disparando el `grab()`
    de las N camaras casi a la vez y decodificando despues, el desfase entre
    vistas se reduce al tiempo entre llamadas a grab (~decimas de ms) en lugar
    del tiempo de decodificacion (~varios ms por camara).
    """

    # Un `grab()` que retorna en menos de esto no espero a un frame nuevo:
    # entrego uno que ya estaba en la cola del driver, es decir, rancio.
    FRESH_THRESHOLD_S = 0.002
    MAX_FLUSH = 4

    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self.cap: cv2.VideoCapture | None = None
        self.t_grab: float = float("nan")
        self.frame_index: int = -1
        self.stale_dropped: int = 0     # total de frames rancios descartados

    # --- ciclo de vida --- #
    def open(self) -> "CameraSource":
        self.cap = open_capture(self.spec)
        return self

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "CameraSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def actual_config(self) -> dict[str, Any]:
        return read_actual(self.cap) if self.cap is not None else {}

    def warmup(self, n: int = 15) -> None:
        """Descarta los primeros frames (negociacion USB, auto-ajuste)."""
        for _ in range(n):
            if self.cap is not None:
                self.cap.read()

    # --- captura --- #
    def grab(self, flush_stale: bool = True) -> bool:
        """Dispara la captura y marca el timestamp. No decodifica.

        Con `flush_stale`, descarta los frames que el driver ya tenia encolados
        y se queda con el primero que hubo que ESPERAR. Sin esto, un bucle de
        captura que se retrasa entrega frames antiguos con timestamp actual: el
        error resultante es invisible en los datos y contamina toda la
        reconstruccion de una trayectoria en movimiento.
        """
        if self.cap is None:
            return False
        ok = False
        for i in range(self.MAX_FLUSH + 1):
            t0 = time.perf_counter()
            ok = bool(self.cap.grab())
            t1 = time.perf_counter()
            if not ok:
                break
            if not flush_stale or (t1 - t0) >= self.FRESH_THRESHOLD_S:
                break
            self.stale_dropped += 1
        self.t_grab = t1
        return ok

    def retrieve(self) -> np.ndarray | None:
        """Decodifica el frame del ultimo `grab()`."""
        if self.cap is None:
            return None
        ok, frame = self.cap.retrieve()
        if ok and frame is not None:
            self.frame_index += 1
            return frame
        return None

    def read(self) -> tuple[np.ndarray | None, float]:
        """Conveniencia: grab + retrieve. Devuelve (frame, t_grab)."""
        if not self.grab():
            return None, float("nan")
        return self.retrieve(), self.t_grab
