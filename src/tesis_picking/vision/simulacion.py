"""Escenas RGB-D sinteticas para validar la cadena sin la camara.

Renderiza esferas de color naranja sobre un plano, con la MISMA geometria
pinhole que usa `RGBDFrame.deproject`, y devuelve color + profundidad con la
posicion real de cada fruta. Sirve para tres cosas:

  * desarrollar y probar detector, localizador y cambio de frames sin hardware;
  * medir el error del localizador con verdad de terreno EXACTA, cosa
    imposible con la camara real (donde el "valor verdadero" ya lleva el error
    de la regla con que se midio);
  * dejar tests de regresion que corren en CI.

No pretende ser un simulador fotorrealista: modela lo que afecta al algoritmo
--- proyeccion, oclusion por profundidad, ruido de profundidad dependiente de
la distancia, y huecos --- y nada mas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..common.camera_model import Intrinsics
from .realsense import RGBDFrame


@dataclass
class EsferaSintetica:
    centro_cam: np.ndarray      # (3,) posicion real del CENTRO, en {C}, metros
    radio_m: float
    color_bgr: tuple[int, int, int] = (30, 130, 240)   # naranja mandarina


@dataclass
class EscenaSintetica:
    """Generador de frames RGB-D con verdad de terreno."""
    intr: Intrinsics
    esferas: list[EsferaSintetica] = field(default_factory=list)
    plano_z_m: float = 1.5          # fondo plano perpendicular al eje optico
    color_fondo: tuple[int, int, int] = (110, 110, 105)
    # Ruido de profundidad de la D435: crece aproximadamente con el cuadrado de
    # la distancia (sigma ~ z^2 * baseline^-1 * disparidad_err). El coeficiente
    # da ~2.5 mm a 1 m, coherente con el datasheet (<2% a 2 m).
    ruido_coef: float = 0.0025
    prob_hueco: float = 0.02        # pixeles sin dato, como en el sensor real
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ #
    def render(self, con_ruido: bool = True) -> RGBDFrame:
        w, h = self.intr.image_size
        us, vs = np.meshgrid(np.arange(w, dtype=np.float64),
                             np.arange(h, dtype=np.float64))
        # Rayos unitarios por pixel, en coordenadas normalizadas de camara.
        xn = (us - self.intr.cx) / self.intr.fx
        yn = (vs - self.intr.cy) / self.intr.fy

        depth = np.full((h, w), self.plano_z_m, dtype=np.float32)
        color = np.zeros((h, w, 3), dtype=np.uint8)
        color[:] = self.color_fondo

        for esf in self.esferas:
            z_sup = self._interseccion_esfera(xn, yn, esf)
            visible = np.isfinite(z_sup) & (z_sup < depth)
            depth[visible] = z_sup[visible].astype(np.float32)
            # Sombreado lambertiano simple: da textura al blob, lo que hace que
            # el detector trabaje sobre algo mas realista que un disco plano.
            sombra = self._sombreado(xn, yn, z_sup, esf, visible)
            for c in range(3):
                color[..., c][visible] = np.clip(
                    esf.color_bgr[c] * sombra[visible], 0, 255).astype(np.uint8)

        if con_ruido:
            sigma = self.ruido_coef * depth ** 2
            depth = depth + self._rng.normal(0.0, 1.0, depth.shape).astype(np.float32) * sigma
            huecos = self._rng.random(depth.shape) < self.prob_hueco
            depth[huecos] = 0.0
            ruido_color = self._rng.normal(0, 4, color.shape)
            color = np.clip(color.astype(np.float32) + ruido_color, 0, 255).astype(np.uint8)

        return RGBDFrame(color=color, depth_m=depth.astype(np.float32),
                         intr=self.intr, timestamp=0.0, frame_number=0)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _interseccion_esfera(xn: np.ndarray, yn: np.ndarray,
                             esf: EsferaSintetica) -> np.ndarray:
        """Z de la primera interseccion del rayo con la esfera. NaN si no corta.

        El rayo es  P(t) = t * d  con d = (xn, yn, 1) sin normalizar, de modo
        que t coincide directamente con la coordenada Z buscada.
        """
        c = np.asarray(esf.centro_cam, dtype=float)
        # |t*d - c|^2 = r^2  ->  a t^2 + b t + cc = 0
        a = xn ** 2 + yn ** 2 + 1.0
        b = -2.0 * (xn * c[0] + yn * c[1] + c[2])
        cc = float(c @ c - esf.radio_m ** 2)
        disc = b ** 2 - 4.0 * a * cc
        z = np.full_like(xn, np.nan)
        hit = disc >= 0
        if np.any(hit):
            t = (-b[hit] - np.sqrt(disc[hit])) / (2.0 * a[hit])
            t = np.where(t > 0, t, np.nan)
            z[hit] = t
        return z

    @staticmethod
    def _sombreado(xn: np.ndarray, yn: np.ndarray, z_sup: np.ndarray,
                   esf: EsferaSintetica, visible: np.ndarray) -> np.ndarray:
        s = np.zeros_like(xn)
        if not np.any(visible):
            return s
        P = np.stack([xn * z_sup, yn * z_sup, z_sup], axis=-1)
        N = P - np.asarray(esf.centro_cam, float)
        n = np.linalg.norm(N, axis=-1, keepdims=True)
        N = np.divide(N, n, out=np.zeros_like(N), where=n > 0)
        # Direccion desde la superficie HACIA la luz. La luz se situa detras y
        # arriba de la camara, que es el caso habitual en una estacion de
        # inspeccion: por eso su componente Z es negativa (hacia la camara).
        luz = np.array([-0.35, -0.45, -1.0]); luz /= np.linalg.norm(luz)
        s = np.clip(0.45 + 0.55 * (N @ luz), 0.30, 1.15)
        return np.nan_to_num(s)


def intrinsecos_d435i(width: int = 848, height: int = 480) -> Intrinsics:
    """Intrinsecos REPRESENTATIVOS de la D435i para simulacion.

    Referencia medida en la unidad 327122073685 a 640x480:
        fx=606.18  fy=605.94  cx=329.10  cy=247.94  distorsion nula

    El sensor tiene pixeles cuadrados, asi que fx == fy salvo ruido de
    calibracion. Al cambiar de resolucion NO se escalan fx y fy por razones
    distintas: entre 640x480 (4:3) y 848x480 (16:9) lo que cambia es el campo
    de vision HORIZONTAL, no la escala vertical. Se toma por tanto la distancia
    focal comun a partir de la altura, que es la que ambos perfiles comparten.

    Para trabajo real NO se usa esta funcion: los intrinsecos se leen del
    dispositivo con `RealSenseCamera`, que entrega los de fabrica del perfil
    exacto que se haya solicitado.
    """
    f = 606.06 * height / 480.0
    K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])
    return Intrinsics(K=K, D=np.zeros(5), image_size=(width, height),
                      rms_reprojection_px=0.0, n_views=0,
                      model="realsense_inverse_brown_conrady")
