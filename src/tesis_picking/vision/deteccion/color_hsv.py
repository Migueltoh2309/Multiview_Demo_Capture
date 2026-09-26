"""Deteccion de mandarinas por segmentacion de color en HSV.

Por que empezar por aqui y no por YOLO
--------------------------------------
YOLO es la opcion correcta para la tesis --- la literatura del propio marco
teorico reporta mAP@50 > 0.92 en citricos --- pero necesita un dataset
etiquetado que todavia no existe. Este detector clasico:

  * funciona HOY, sin datos de entrenamiento;
  * permite validar toda la cadena aguas abajo (localizacion 3D, extrinsecos,
    metrologia) de forma independiente al detector;
  * sirve despues como LINEA BASE contra la que comparar YOLO, y como
    herramienta de pre-etiquetado semiautomatico para construir el dataset.

Limitaciones asumidas, y por eso el diseno lo aisla tras la interfaz `Detector`:
es sensible a la iluminacion, no separa frutas en contacto salvo por watershed,
y confunde cualquier objeto naranja del fondo.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from ...common import paths
from .base import Deteccion2D


@dataclass
class ObjetoConfig:
    """Descripcion fisica y de apariencia de un objeto (config/objetos.yaml)."""
    nombre: str
    diametro_min_m: float
    diametro_nominal_m: float
    diametro_max_m: float
    h: tuple[int, int]
    s: tuple[int, int]
    v: tuple[int, int]
    area_min_px: int
    area_max_px: int
    circularidad_min: float
    relacion_aspecto: tuple[float, float]
    solidez_min: float

    @property
    def radio_nominal_m(self) -> float:
        return self.diametro_nominal_m / 2.0

    @classmethod
    def load(cls, nombre: str = "mandarina", path: Path | None = None) -> "ObjetoConfig":
        path = path or paths.CONFIG / "objetos.yaml"
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if nombre not in raw:
            raise KeyError(f"'{nombre}' no esta en {path} (hay: {list(raw)})")
        o = raw[nombre]
        d, hsv, forma = o["diametro_m"], o["hsv"], o["forma"]
        return cls(
            nombre=nombre,
            diametro_min_m=float(d["min"]),
            diametro_nominal_m=float(d["nominal"]),
            diametro_max_m=float(d["max"]),
            h=tuple(hsv["h"]), s=tuple(hsv["s"]), v=tuple(hsv["v"]),
            area_min_px=int(forma["area_min_px"]),
            area_max_px=int(forma["area_max_px"]),
            circularidad_min=float(forma["circularidad_min"]),
            relacion_aspecto=tuple(forma["relacion_aspecto"]),
            solidez_min=float(forma["solidez_min"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nombre": self.nombre,
            "diametro_m": {"min": self.diametro_min_m,
                           "nominal": self.diametro_nominal_m,
                           "max": self.diametro_max_m},
            "hsv": {"h": list(self.h), "s": list(self.s), "v": list(self.v)},
            "forma": {"area_min_px": self.area_min_px,
                      "area_max_px": self.area_max_px,
                      "circularidad_min": self.circularidad_min,
                      "relacion_aspecto": list(self.relacion_aspecto),
                      "solidez_min": self.solidez_min},
        }


class DetectorColorHSV:
    """Segmenta por color y filtra por forma."""

    def __init__(self, cfg: ObjetoConfig, *, kernel_px: int = 5,
                 separar_contacto: bool = True):
        self.cfg = cfg
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
        self.separar_contacto = separar_contacto

    # ------------------------------------------------------------------ #
    def mascara_color(self, color_bgr: np.ndarray) -> np.ndarray:
        """Mascara binaria del color objetivo, ya limpiada morfologicamente."""
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        lo = np.array([self.cfg.h[0], self.cfg.s[0], self.cfg.v[0]], np.uint8)
        hi = np.array([self.cfg.h[1], self.cfg.s[1], self.cfg.v[1]], np.uint8)

        if self.cfg.h[0] <= self.cfg.h[1]:
            m = cv2.inRange(hsv, lo, hi)
        else:
            # El matiz es circular: un rango que cruza el 0 (rojos) necesita
            # dos intervalos. OpenCV usa H en [0, 179].
            m1 = cv2.inRange(hsv, np.array([self.cfg.h[0], lo[1], lo[2]], np.uint8),
                             np.array([179, hi[1], hi[2]], np.uint8))
            m2 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8),
                             np.array([self.cfg.h[1], hi[1], hi[2]], np.uint8))
            m = cv2.bitwise_or(m1, m2)

        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return m

    # ------------------------------------------------------------------ #
    def detectar(self, color_bgr: np.ndarray) -> list[Deteccion2D]:
        mask = self.mascara_color(color_bgr)
        etiquetas = (self._watershed(mask) if self.separar_contacto
                     else self._componentes(mask))

        dets: list[Deteccion2D] = []
        for lbl in np.unique(etiquetas):
            if lbl <= 0:
                continue
            blob = (etiquetas == lbl).astype(np.uint8)
            det = self._evaluar_blob(blob)
            if det is not None:
                dets.append(det)
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets

    def _componentes(self, mask: np.ndarray) -> np.ndarray:
        n, lab = cv2.connectedComponents(mask)
        return lab

    def _watershed(self, mask: np.ndarray) -> np.ndarray:
        """Separa frutas en contacto por watershed sobre la transformada de distancia.

        Dos mandarinas que se tocan forman un solo blob. El maximo local de la
        distancia al borde marca el centro de cada una, y el watershed reparte
        el blob entre esos centros.
        """
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            return np.zeros_like(mask, dtype=np.int32)
        _, picos = cv2.threshold(dist, 0.55 * dist.max(), 255, cv2.THRESH_BINARY)
        picos = picos.astype(np.uint8)
        n, marcadores = cv2.connectedComponents(picos)
        if n <= 1:
            return self._componentes(mask)
        marcadores = marcadores + 1
        desconocido = cv2.subtract(mask, picos)
        marcadores[desconocido == 255] = 0
        img3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        marcadores = cv2.watershed(img3, marcadores.astype(np.int32))
        salida = np.where(marcadores > 1, marcadores - 1, 0).astype(np.int32)
        salida[mask == 0] = 0
        return salida

    # ------------------------------------------------------------------ #
    def _evaluar_blob(self, blob: np.ndarray) -> Deteccion2D | None:
        """Aplica los filtros de forma. Devuelve None si el blob no es fruta."""
        area = float(np.count_nonzero(blob))
        if not (self.cfg.area_min_px <= area <= self.cfg.area_max_px):
            return None

        contornos, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None
        c = max(contornos, key=cv2.contourArea)
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            return None

        circularidad = 4.0 * np.pi * cv2.contourArea(c) / (perim ** 2)
        if circularidad < self.cfg.circularidad_min:
            return None

        x, y, w, h = cv2.boundingRect(c)
        aspecto = w / h if h else 0.0
        if not (self.cfg.relacion_aspecto[0] <= aspecto <= self.cfg.relacion_aspecto[1]):
            return None

        casco = cv2.convexHull(c)
        area_casco = cv2.contourArea(casco)
        solidez = cv2.contourArea(c) / area_casco if area_casco > 0 else 0.0
        if solidez < self.cfg.solidez_min:
            return None

        M = cv2.moments(blob, binaryImage=True)
        if M["m00"] <= 0:
            return None
        cu, cv_ = M["m10"] / M["m00"], M["m01"] / M["m00"]
        (_, _), radio_min = cv2.minEnclosingCircle(c)
        # El radio equivalente por area es mas estable que el circulo minimo
        # cuando la fruta esta parcialmente ocluida por el borde de la imagen.
        radio_area = float(np.sqrt(area / np.pi))

        score = float(np.clip(
            0.5 * circularidad + 0.3 * solidez
            + 0.2 * min(1.0, radio_area / max(radio_min, 1e-6)), 0.0, 1.0))

        return Deteccion2D(
            centro_px=np.array([cu, cv_]),
            bbox=(int(x), int(y), int(w), int(h)),
            mascara=blob.astype(bool),
            score=score,
            clase=self.cfg.nombre,
            radio_px=radio_area,
            metricas={"circularidad": circularidad, "solidez": solidez,
                      "aspecto": aspecto, "radio_min_px": float(radio_min)},
        )

    # ------------------------------------------------------------------ #
    def dibujar(self, color_bgr: np.ndarray,
                dets: list[Deteccion2D]) -> np.ndarray:
        vis = color_bgr.copy()
        for i, d in enumerate(dets):
            x, y, w, h = d.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(vis, (int(d.u), int(d.v)), 4, (0, 0, 255), -1)
            cv2.putText(vis, f"{i}:{d.score:.2f}", (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return vis
