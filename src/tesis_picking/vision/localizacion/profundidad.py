"""De una deteccion 2D a la posicion 3D del CENTRO de la fruta.

El problema no es la deproyeccion --- eso es una division --- sino tres cosas
que, si se ignoran, producen un error de centimetros:

1. **La profundidad de un solo pixel no sirve.** El mapa de profundidad estereo
   tiene huecos (valor 0) y pixeles atipicos. Justo el centroide de la fruta
   puede ser un hueco. Se muestrea toda la mascara y se usa un estimador
   robusto (mediana + rechazo por MAD).

2. **El borde de la mascara mezcla fruta y fondo.** Un pixel del contorno puede
   caer sobre la mesa, 30 cm mas atras. La mascara se EROSIONA antes de
   muestrear, para quedarse con el interior seguro.

3. **La profundidad mide la SUPERFICIE FRONTAL, no el centro.** Para una
   mandarina de 65 mm, el centro esta ~32 mm mas lejos que lo que ve la camara.
   Si el efector apunta a la superficie, agarra aire. Se corrige empujando el
   punto a lo largo del rayo optico, o directamente ajustando una esfera.

Metodos disponibles
-------------------
`centroide`: deproyecta el centroide con la profundidad robusta y suma el radio
    a lo largo del rayo. Rapido y suficiente cuando la fruta se ve entera.
`esfera`: ajusta una esfera de radio conocido a la nube de puntos de la
    mascara. Mas exacto con oclusion parcial, porque usa toda la superficie
    visible en lugar de un solo punto. Es el metodo por defecto.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares

from ...common.camera_model import Intrinsics
from ..deteccion.base import Deteccion2D
from ..deteccion.color_hsv import ObjetoConfig

log = logging.getLogger(__name__)

Metodo = Literal["centroide", "esfera"]


@dataclass
class Objeto3D:
    """Fruta localizada en 3D, en el frame de la camara {C}."""
    centro_cam: np.ndarray              # (3,) metros; NaN si no localizable
    superficie_cam: np.ndarray          # (3,) punto medido de la superficie
    radio_m: float
    radio_fuente: str                   # 'medido' | 'prior' | 'acotado'
    n_puntos: int                       # pixeles con profundidad usados
    profundidad_m: float                # distancia robusta a la superficie
    dispersion_mm: float                # MAD de la profundidad: ruido del sensor
    residuo_ajuste_mm: float = float("nan")
    valido: bool = False
    motivo: str = ""
    deteccion: Deteccion2D | None = None
    metodo: str = ""

    @property
    def distancia_m(self) -> float:
        """Norma del vector al centro. NaN si no es valido."""
        return (float(np.linalg.norm(self.centro_cam))
                if np.all(np.isfinite(self.centro_cam)) else float("nan"))

    def resumen(self) -> dict[str, Any]:
        return {
            "valido": self.valido,
            "motivo": self.motivo,
            "metodo": self.metodo,
            "centro_cam_m": ([round(float(v), 4) for v in self.centro_cam]
                             if np.all(np.isfinite(self.centro_cam)) else None),
            "distancia_m": round(self.distancia_m, 4) if self.valido else None,
            "radio_m": round(float(self.radio_m), 4),
            "radio_fuente": self.radio_fuente,
            "profundidad_superficie_m": round(float(self.profundidad_m), 4),
            "n_puntos_profundidad": self.n_puntos,
            "dispersion_mm": round(float(self.dispersion_mm), 2),
            "residuo_ajuste_mm": (round(float(self.residuo_ajuste_mm), 2)
                                  if np.isfinite(self.residuo_ajuste_mm) else None),
        }


# --------------------------------------------------------------------------- #
def muestrear_profundidad(
    depth_m: np.ndarray,
    mascara: np.ndarray,
    *,
    erosion_px: int = 3,
    k_mad: float = 3.0,
    min_puntos: int = 20,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Extrae las profundidades fiables dentro de una mascara.

    Returns:
        (vs, us, z_robusto, dispersion_mm) --- `vs`/`us` son los indices de los
        pixeles conservados tras erosionar y rechazar atipicos.

    El rechazo usa MAD (desviacion absoluta mediana) y no desviacion tipica:
    la media y sigma se desplazan con unos pocos pixeles de fondo colados en la
    mascara, mientras que la mediana no.
    """
    import cv2

    m = mascara.astype(np.uint8)
    if erosion_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * erosion_px + 1, 2 * erosion_px + 1))
        erosionada = cv2.erode(m, k)
        # Si erosionar deja la mascara vacia (fruta lejana y pequena),
        # se vuelve a la original: mejor un poco de borde que ningun dato.
        m = erosionada if np.count_nonzero(erosionada) >= min_puntos else m

    vs, us = np.nonzero(m)
    if vs.size == 0:
        return np.empty(0, int), np.empty(0, int), float("nan"), float("nan")

    z = depth_m[vs, us]
    ok = np.isfinite(z) & (z > 0)
    vs, us, z = vs[ok], us[ok], z[ok]
    if z.size == 0:
        return vs, us, float("nan"), float("nan")

    mediana = float(np.median(z))
    mad = float(np.median(np.abs(z - mediana)))
    # 1.4826 convierte MAD en un estimador consistente de sigma para ruido normal
    sigma = 1.4826 * mad
    if sigma > 0:
        keep = np.abs(z - mediana) <= k_mad * sigma
        vs, us, z = vs[keep], us[keep], z[keep]

    if z.size == 0:
        return vs, us, float("nan"), float("nan")
    return vs, us, float(np.median(z)), float(sigma * 1000.0)


def _deproyectar(us: np.ndarray, vs: np.ndarray, z: np.ndarray,
                 intr: Intrinsics) -> np.ndarray:
    """(N,) pixeles + profundidad -> (N, 3) puntos en {C}."""
    x = (us - intr.cx) * z / intr.fx
    y = (vs - intr.cy) * z / intr.fy
    return np.column_stack([x, y, z])


def ajustar_esfera_radio_fijo(P: np.ndarray, radio: float,
                              c0: np.ndarray) -> tuple[np.ndarray, float]:
    """Ajusta el CENTRO de una esfera de radio conocido a los puntos P.

    Se fija el radio a proposito. La camara solo ve el casquete frontal de la
    fruta, y ajustar centro y radio a la vez sobre un casquete es un problema
    mal condicionado: pequenos errores de profundidad producen radios absurdos
    y arrastran el centro con ellos. Con el radio fijado por el prior fisico
    (config/objetos.yaml), queda un ajuste de 3 incognitas bien planteado.

    Returns:
        (centro, residuo_rms_metros)
    """
    def residuos(c: np.ndarray) -> np.ndarray:
        return np.linalg.norm(P - c, axis=1) - radio

    sol = least_squares(residuos, c0, method="lm", max_nfev=100)
    r = residuos(sol.x)
    return sol.x, float(np.sqrt(np.mean(r ** 2)))


# --------------------------------------------------------------------------- #
class Localizador3D:
    """Convierte detecciones 2D en posiciones 3D del centro de la fruta."""

    def __init__(
        self,
        objeto: ObjetoConfig,
        *,
        metodo: Metodo = "esfera",
        erosion_px: int = 3,
        min_puntos: int = 20,
        cobertura_min: float = 0.25,
        dispersion_max_mm: float = 25.0,
        rango_m: tuple[float, float] = (0.20, 2.00),
        residuo_max_mm: float = 15.0,
    ):
        """
        Args:
            cobertura_min: fraccion minima de pixeles de la mascara con
                profundidad valida. Por debajo, la medida no es fiable: suele
                indicar reflejo especular en la piel de la fruta o que el
                emisor IR esta apagado.
            dispersion_max_mm: sigma robusto maximo dentro de la fruta. Si la
                profundidad dentro de una mandarina se dispersa mas que su
                propio diametro, la mascara esta cogiendo fondo.
        """
        self.objeto = objeto
        self.metodo = metodo
        self.erosion_px = erosion_px
        self.min_puntos = min_puntos
        self.cobertura_min = cobertura_min
        self.dispersion_max_mm = dispersion_max_mm
        self.rango_m = rango_m
        self.residuo_max_mm = residuo_max_mm

    # ------------------------------------------------------------------ #
    def estimar_radio(self, det: Deteccion2D, z: float,
                      intr: Intrinsics) -> tuple[float, str]:
        """Radio fisico de la fruta a partir de su tamano aparente.

        r = radio_px * z / f.  Se contrasta con el rango de diametros
        declarado en config/objetos.yaml: si la medida cae fuera, se recorta al
        rango y se marca como 'acotado'. Una medida aparente absurda casi
        siempre viene de una mascara que se comio dos frutas o parte del fondo.
        """
        if not (np.isfinite(det.radio_px) and np.isfinite(z) and z > 0):
            return self.objeto.radio_nominal_m, "prior"
        f = 0.5 * (intr.fx + intr.fy)
        r = float(det.radio_px * z / f)
        r_min, r_max = self.objeto.diametro_min_m / 2, self.objeto.diametro_max_m / 2
        if r < r_min:
            return r_min, "acotado"
        if r > r_max:
            return r_max, "acotado"
        return r, "medido"

    # ------------------------------------------------------------------ #
    def localizar(self, det: Deteccion2D, depth_m: np.ndarray,
                  intr: Intrinsics) -> Objeto3D:
        """Localiza una deteccion. Devuelve `Objeto3D` con `valido` a False y
        el motivo cuando no es posible --- nunca una posicion inventada."""
        nan3 = np.full(3, np.nan)
        vs, us, z_rob, disp_mm = muestrear_profundidad(
            depth_m, det.mascara, erosion_px=self.erosion_px,
            min_puntos=self.min_puntos)

        base = dict(superficie_cam=nan3.copy(), radio_m=self.objeto.radio_nominal_m,
                    radio_fuente="prior", n_puntos=int(vs.size),
                    profundidad_m=z_rob, dispersion_mm=disp_mm,
                    deteccion=det, metodo=self.metodo)

        if vs.size < self.min_puntos:
            return Objeto3D(centro_cam=nan3, valido=False, **base,
                            motivo=f"solo {vs.size} pixeles con profundidad "
                                   f"(minimo {self.min_puntos})")

        cobertura = vs.size / max(det.area_px, 1)
        if cobertura < self.cobertura_min:
            return Objeto3D(centro_cam=nan3, valido=False, **base,
                            motivo=f"cobertura de profundidad {cobertura:.0%} "
                                   f"< {self.cobertura_min:.0%}")
        if not np.isfinite(z_rob):
            return Objeto3D(centro_cam=nan3, valido=False, **base,
                            motivo="profundidad no finita")
        if not (self.rango_m[0] <= z_rob <= self.rango_m[1]):
            return Objeto3D(centro_cam=nan3, valido=False, **base,
                            motivo=f"profundidad {z_rob:.3f} m fuera del rango "
                                   f"{self.rango_m}")
        if disp_mm > self.dispersion_max_mm:
            return Objeto3D(centro_cam=nan3, valido=False, **base,
                            motivo=f"dispersion {disp_mm:.1f} mm > "
                                   f"{self.dispersion_max_mm} mm: la mascara "
                                   "probablemente incluye fondo")

        radio, fuente = self.estimar_radio(det, z_rob, intr)
        base.update(radio_m=radio, radio_fuente=fuente)

        # Punto de superficie: el centroide de la deteccion a la profundidad
        # robusta. Es la referencia comun a los dos metodos.
        superficie = np.array([(det.u - intr.cx) * z_rob / intr.fx,
                               (det.v - intr.cy) * z_rob / intr.fy,
                               z_rob])
        base["superficie_cam"] = superficie

        # Direccion del rayo optico hacia la fruta, normalizada.
        rayo = superficie / np.linalg.norm(superficie)

        if self.metodo == "centroide":
            centro = superficie + radio * rayo
            return Objeto3D(centro_cam=centro, valido=True, motivo="ok", **base)

        # --- metodo esfera ---
        P = _deproyectar(us.astype(float), vs.astype(float),
                         depth_m[vs, us].astype(float), intr)
        c0 = superficie + radio * rayo
        centro, residuo = ajustar_esfera_radio_fijo(P, radio, c0)
        residuo_mm = residuo * 1000.0
        base["residuo_ajuste_mm"] = residuo_mm

        if not np.all(np.isfinite(centro)):
            return Objeto3D(centro_cam=nan3, valido=False, **base,
                            motivo="el ajuste de esfera no convergio")
        if residuo_mm > self.residuo_max_mm:
            # La superficie no es esferica: fruta mal segmentada, dos frutas
            # juntas, o algo que no es una mandarina.
            return Objeto3D(centro_cam=nan3, valido=False, **base,
                            motivo=f"residuo de esfera {residuo_mm:.1f} mm > "
                                   f"{self.residuo_max_mm} mm")
        # El centro debe quedar DETRAS de la superficie observada.
        if centro[2] < superficie[2] - 1e-3:
            return Objeto3D(centro_cam=nan3, valido=False, **base,
                            motivo="el centro ajustado queda delante de la superficie")

        return Objeto3D(centro_cam=centro, valido=True, motivo="ok", **base)

    def localizar_todas(self, dets: list[Deteccion2D], depth_m: np.ndarray,
                        intr: Intrinsics) -> list[Objeto3D]:
        return [self.localizar(d, depth_m, intr) for d in dets]
