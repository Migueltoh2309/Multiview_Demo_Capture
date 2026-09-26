"""Adquisicion RGB-D con Intel RealSense D435i.

Por que esta camara no necesita calibracion intrinseca con ChArUco
-----------------------------------------------------------------
La D400 almacena en su propia memoria los intrinsecos de cada stream y los
extrinsecos entre streams, calibrados en fabrica. `librealsense` los entrega ya
adaptados al perfil que se solicita. Ademas, el stream de color sale
RECTIFICADO por el ASIC: sus coeficientes de distorsion son exactamente cero.

Lo que si hay que calibrar en este proyecto es la POSE de la camara respecto al
mundo de trabajo {W} --- eso es geometria del montaje, no del sensor, y no hay
forma de que venga de fabrica. Ver `scripts/vision/14_extrinsecos_camara.py`.

Alineacion depth->color
-----------------------
Los sensores de profundidad y de color son fisicamente distintos y estan
separados ~15 mm (medido en esta unidad: t = [14.85, 0.08, 0.19] mm). Sin
alinear, el pixel (u,v) donde el detector encuentra la mandarina NO corresponde
al mismo punto en el mapa de profundidad, y el error resultante crece al
acercarse el objeto. `align_to: color` resuelve esto en el SDK.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

from ..common import paths
from ..common.camera_model import CameraModel, Intrinsics

log = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs
    RS_AVAILABLE = True
except ImportError:                     # permite importar el modulo sin SDK
    rs = None                           # (tests, maquinas sin la camara)
    RS_AVAILABLE = False


class RealSenseNotAvailable(RuntimeError):
    """El SDK no esta instalado o no hay camara conectada."""


# --------------------------------------------------------------------------- #
@dataclass
class StreamSpec:
    width: int
    height: int
    fps: int
    format: str = "bgr8"

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass
class RealSenseConfig:
    """Configuracion cargada de config/realsense.yaml."""
    serial: str | None
    require_usb3: bool
    color: StreamSpec
    depth: StreamSpec
    align_to: str
    fallback_color: StreamSpec
    fallback_depth: StreamSpec
    sensor: dict[str, Any] = field(default_factory=dict)
    postprocess: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "RealSenseConfig":
        path = path or paths.CONFIG / "realsense.yaml"
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        dev = raw.get("device", {}) or {}
        st = raw.get("streams", {}) or {}
        fb = raw.get("fallback_usb2", {}) or {}
        return cls(
            serial=dev.get("serial"),
            require_usb3=bool(dev.get("require_usb3", True)),
            color=StreamSpec(**st["color"]),
            depth=StreamSpec(**st["depth"]),
            align_to=st.get("align_to", "color"),
            fallback_color=StreamSpec(**fb.get("color", st["color"])),
            fallback_depth=StreamSpec(**fb.get("depth", st["depth"])),
            sensor=raw.get("sensor", {}) or {},
            postprocess=raw.get("postprocess", {}) or {},
        )


@dataclass
class RGBDFrame:
    """Un par color+profundidad alineado, con sus intrinsecos."""
    color: np.ndarray                # (H, W, 3) BGR uint8
    depth_m: np.ndarray              # (H, W) float32, METROS; 0.0 = sin dato
    intr: Intrinsics                 # del stream de color (tras alinear)
    timestamp: float                 # perf_counter() en la recepcion
    frame_number: int = -1
    device_timestamp_ms: float = float("nan")

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth_m.shape[:2]

    @property
    def valid_depth_ratio(self) -> float:
        """Fraccion de pixeles con profundidad valida. Diagnostico rapido."""
        return float(np.mean(self.depth_m > 0))

    def depth_at(self, u: float, v: float) -> float:
        """Profundidad en un pixel (vecino mas proximo). 0.0 si no hay dato."""
        h, w = self.shape
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            return 0.0
        return float(self.depth_m[vi, ui])

    def deproject(self, u: float, v: float, z: float) -> np.ndarray:
        """Pixel + profundidad -> punto 3D en el frame de la camara {C}.

        Modelo pinhole simple: el stream de color de la D400 sale rectificado,
        con coeficientes de distorsion nulos, asi que no hace falta deshacer
        distorsion aqui. `Intrinsics.D` se comprueba en `RealSenseCamera`.
        """
        if z <= 0:
            return np.full(3, np.nan)
        K = self.intr
        return np.array([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z])

    def project(self, X_cam: np.ndarray) -> np.ndarray:
        """Inversa de `deproject`: punto 3D en {C} -> pixel."""
        X = np.asarray(X_cam, dtype=float).reshape(3)
        if X[2] <= 0:
            return np.full(2, np.nan)
        K = self.intr
        return np.array([K.fx * X[0] / X[2] + K.cx, K.fy * X[1] / X[2] + K.cy])


# --------------------------------------------------------------------------- #
def describe_devices() -> list[dict[str, Any]]:
    """Informa de las RealSense conectadas, sin abrir streams.

    Es la comprobacion que hay que hacer ANTES de nada: si el enlace no es
    USB 3.x, la camara enumera y expone sus perfiles pero NO sostiene el
    streaming --- entrega unos pocos frames y se cuelga (ver bitacora).
    """
    if not RS_AVAILABLE:
        raise RealSenseNotAvailable(
            "pyrealsense2 no esta instalado. En el venv del proyecto: "
            "pip install pyrealsense2")
    out = []
    for dev in rs.context().query_devices():
        def info(k, default=""):
            try:
                return dev.get_info(k)
            except Exception:
                return default
        usb = info(rs.camera_info.usb_type_descriptor, "?")
        out.append({
            "name": info(rs.camera_info.name),
            "serial": info(rs.camera_info.serial_number),
            "firmware": info(rs.camera_info.firmware_version),
            "firmware_recomendado": info(rs.camera_info.recommended_firmware_version),
            "usb_type": usb,
            "usb3": str(usb).startswith("3"),
            "physical_port": info(rs.camera_info.physical_port),
        })
    return out


class RealSenseCamera:
    """Envoltura de un pipeline de RealSense con alineacion y post-procesado.

    Uso:
        with RealSenseCamera(RealSenseConfig.load()) as cam:
            for frame in cam.stream(n=100):
                ...
    """

    def __init__(self, cfg: RealSenseConfig, *, allow_usb2: bool = False):
        if not RS_AVAILABLE:
            raise RealSenseNotAvailable(
                "pyrealsense2 no esta instalado (pip install pyrealsense2)")
        self.cfg = cfg
        self.allow_usb2 = allow_usb2
        self.pipeline: Any = None
        self.profile: Any = None
        self.depth_scale: float = 0.001
        self.intr: Intrinsics | None = None
        self.usb_type: str = "?"
        self.serial: str = ""
        self._align: Any = None
        self._filters: list[tuple[str, Any]] = []
        self._frame_count = 0

    # ------------------------------------------------------------------ #
    def open(self) -> "RealSenseCamera":
        devices = describe_devices()
        if not devices:
            raise RealSenseNotAvailable(
                "no se detecta ninguna RealSense. Comprobar el cable y ejecutar "
                "scripts/vision/10_check_realsense.py")

        dev = next((d for d in devices if d["serial"] == self.cfg.serial),
                   devices[0]) if self.cfg.serial else devices[0]
        self.usb_type = dev["usb_type"]
        self.serial = dev["serial"]
        usb3 = dev["usb3"]

        if not usb3:
            msg = (f"la camara {self.serial} esta enlazada como USB {self.usb_type}, "
                   "no USB 3.x. En USB 2 no sostiene el streaming: entrega unos "
                   "pocos frames y se detiene. Causa casi siempre el CABLE: la "
                   "D435i necesita el cable USB-C -> USB-A de USB 3 que trae de "
                   "fabrica, en un puerto azul/SS.")
            if self.cfg.require_usb3 and not self.allow_usb2:
                raise RealSenseNotAvailable(msg)
            log.warning("%s  --- continuando en modo degradado", msg)

        color = self.cfg.color if usb3 else self.cfg.fallback_color
        depth = self.cfg.depth if usb3 else self.cfg.fallback_depth

        self.pipeline = rs.pipeline()
        conf = rs.config()
        if self.cfg.serial:
            conf.enable_device(self.cfg.serial)
        conf.enable_stream(rs.stream.depth, depth.width, depth.height,
                           rs.format.z16, depth.fps)
        conf.enable_stream(rs.stream.color, color.width, color.height,
                           getattr(rs.format, color.format), color.fps)

        log.info("arrancando pipeline: color %dx%d@%d, depth %dx%d@%d (USB %s)",
                 color.width, color.height, color.fps,
                 depth.width, depth.height, depth.fps, self.usb_type)
        self.profile = self.pipeline.start(conf)

        dev_h = self.profile.get_device()
        ds = dev_h.first_depth_sensor()
        self.depth_scale = float(ds.get_depth_scale())
        self._apply_sensor_options(dev_h, ds)

        self._align = rs.align(rs.stream.color if self.cfg.align_to == "color"
                               else rs.stream.depth)
        self._build_filters()
        self.intr = self._read_intrinsics()
        return self

    def close(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

    def __enter__(self) -> "RealSenseCamera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def _apply_sensor_options(self, dev_h: Any, depth_sensor: Any) -> None:
        s = self.cfg.sensor

        preset = s.get("visual_preset")
        if preset and depth_sensor.supports(rs.option.visual_preset):
            rng = depth_sensor.get_option_range(rs.option.visual_preset)
            wanted = str(preset).lower().replace("_", " ")
            for i in range(int(rng.min), int(rng.max) + 1):
                name = depth_sensor.get_option_value_description(
                    rs.option.visual_preset, i)
                if str(name).lower().replace("_", " ") == wanted:
                    depth_sensor.set_option(rs.option.visual_preset, float(i))
                    log.info("visual_preset -> %s", name)
                    break
            else:
                log.warning("visual_preset '%s' no existe en este firmware", preset)

        self._try_set(depth_sensor, rs.option.laser_power, s.get("laser_power"))
        self._try_set(depth_sensor, rs.option.emitter_enabled,
                      None if s.get("emitter_enabled") is None
                      else float(bool(s["emitter_enabled"])))

        color_sensor = next((x for x in dev_h.query_sensors()
                             if x.get_info(rs.camera_info.name) == "RGB Camera"), None)
        if color_sensor is None:
            return
        # El orden importa: desactivar el automatico antes de fijar el manual.
        self._try_set(color_sensor, rs.option.enable_auto_exposure,
                      None if s.get("color_auto_exposure") is None
                      else float(bool(s["color_auto_exposure"])))
        if not s.get("color_auto_exposure", True):
            self._try_set(color_sensor, rs.option.exposure, s.get("color_exposure"))
        self._try_set(color_sensor, rs.option.enable_auto_white_balance,
                      None if s.get("color_auto_white_balance") is None
                      else float(bool(s["color_auto_white_balance"])))
        if not s.get("color_auto_white_balance", True):
            self._try_set(color_sensor, rs.option.white_balance,
                          s.get("color_white_balance"))

    @staticmethod
    def _try_set(sensor: Any, option: Any, value: Any) -> None:
        if value is None:
            return
        try:
            if sensor.supports(option):
                sensor.set_option(option, float(value))
        except Exception as exc:
            log.warning("no se pudo fijar %s=%s: %s", option, value, exc)

    def _build_filters(self) -> None:
        """Construye la cadena de post-procesado segun el YAML."""
        pp = self.cfg.postprocess
        chain: list[tuple[str, Any]] = []

        def on(name: str) -> dict:
            c = pp.get(name, {}) or {}
            return c if c.get("enabled") else {}

        if (c := on("decimation")):
            f = rs.decimation_filter()
            f.set_option(rs.option.filter_magnitude, float(c.get("magnitude", 2)))
            chain.append(("decimation", f))
        if (c := on("threshold")):
            f = rs.threshold_filter()
            f.set_option(rs.option.min_distance, float(c.get("min_m", 0.1)))
            f.set_option(rs.option.max_distance, float(c.get("max_m", 4.0)))
            chain.append(("threshold", f))
        # disparity <-> depth alrededor de spatial/temporal: esos filtros estan
        # formulados en disparidad, y aplicarlos en profundidad deforma la escala.
        if on("spatial") or on("temporal"):
            chain.append(("to_disparity", rs.disparity_transform(True)))
        if (c := on("spatial")):
            f = rs.spatial_filter()
            f.set_option(rs.option.filter_magnitude, float(c.get("magnitude", 2)))
            f.set_option(rs.option.filter_smooth_alpha, float(c.get("smooth_alpha", 0.5)))
            f.set_option(rs.option.filter_smooth_delta, float(c.get("smooth_delta", 20)))
            f.set_option(rs.option.holes_fill, float(c.get("hole_fill", 0)))
            chain.append(("spatial", f))
        if (c := on("temporal")):
            f = rs.temporal_filter()
            f.set_option(rs.option.filter_smooth_alpha, float(c.get("smooth_alpha", 0.4)))
            f.set_option(rs.option.filter_smooth_delta, float(c.get("smooth_delta", 20)))
            f.set_option(rs.option.holes_fill, float(c.get("persistence", 3)))
            chain.append(("temporal", f))
        if on("spatial") or on("temporal"):
            chain.append(("to_depth", rs.disparity_transform(False)))
        if (c := on("hole_filling")):
            f = rs.hole_filling_filter()
            f.set_option(rs.option.holes_fill, float(c.get("mode", 1)))
            chain.append(("hole_filling", f))

        self._filters = chain
        log.info("post-procesado: %s", " -> ".join(n for n, _ in chain) or "(ninguno)")

    def _read_intrinsics(self) -> Intrinsics:
        """Intrinsecos del stream al que se alinea (color por defecto)."""
        stream = (rs.stream.color if self.cfg.align_to == "color" else rs.stream.depth)
        vsp = self.profile.get_stream(stream).as_video_stream_profile()
        i = vsp.get_intrinsics()
        K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], dtype=float)
        D = np.asarray(i.coeffs, dtype=float)
        if np.any(np.abs(D) > 1e-6):
            log.warning("el stream tiene distorsion no nula (%s): "
                        "deproject() asume modelo pinhole puro", i.model)
        return Intrinsics(K=K, D=D, image_size=(i.width, i.height),
                          rms_reprojection_px=0.0, n_views=0,
                          model=f"realsense_{i.model}")

    # ------------------------------------------------------------------ #
    def read(self, timeout_ms: int = 5000) -> RGBDFrame | None:
        """Lee un frame alineado y post-procesado. None si no llega a tiempo."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as exc:
            log.error("no llego frame en %d ms: %s", timeout_ms, exc)
            return None
        t = time.perf_counter()

        frames = self._align.process(frames)
        dfr = frames.get_depth_frame()
        cfr = frames.get_color_frame()
        if not dfr or not cfr:
            return None
        for _, f in self._filters:
            dfr = f.process(dfr)

        depth = np.asanyarray(dfr.get_data()).astype(np.float32) * self.depth_scale
        color = np.asanyarray(cfr.get_data())
        self._frame_count += 1
        return RGBDFrame(
            color=color, depth_m=depth, intr=self.intr, timestamp=t,
            frame_number=self._frame_count,
            device_timestamp_ms=float(cfr.get_timestamp()),
        )

    def stream(self, n: int | None = None, timeout_ms: int = 5000
               ) -> Iterator[RGBDFrame]:
        """Itera frames. `n=None` hasta que se interrumpa."""
        i = 0
        while n is None or i < n:
            fr = self.read(timeout_ms)
            if fr is None:
                return
            yield fr
            i += 1

    # ------------------------------------------------------------------ #
    def save_calibration(self, out_dir: Path | None = None) -> Path:
        """Guarda los intrinsecos de fabrica en calib/, para trazabilidad.

        Se guardan SIN extrinsecos: la pose en {W} la escribe despues
        `14_extrinsecos_camara.py` sobre este mismo archivo.
        """
        out_dir = out_dir or paths.CALIB
        cam = CameraModel(cam_id=f"rs_{self.serial}", intr=self.intr, T_world_cam=None)
        p = Path(out_dir) / f"{cam.cam_id}.yaml"
        cam.save(p)
        return p

    def info(self) -> dict[str, Any]:
        return {
            "serial": self.serial,
            "usb_type": self.usb_type,
            "depth_scale_m": self.depth_scale,
            "align_to": self.cfg.align_to,
            "intrinsics": self.intr.to_dict() if self.intr else None,
            "filtros": [n for n, _ in self._filters],
        }
