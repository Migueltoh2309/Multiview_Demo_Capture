"""Cadena completa de vision sobre escenas sinteticas con verdad de terreno.

Valida deteccion 2D -> localizacion 3D -> cambio a {W}, y mide el error del
localizador contra la posicion real de la esfera.
"""
import numpy as np
import pytest

from tesis_picking.common.config import WorldFrameConfig
from tesis_picking.common.transforms import make_T, rpy_deg_to_R, invert_T
from tesis_picking.vision.deteccion import DetectorColorHSV, ObjetoConfig
from tesis_picking.vision.localizacion import Localizador3D, TransformadorFrames
from tesis_picking.vision.simulacion import (EscenaSintetica, EsferaSintetica,
                                             intrinsecos_d435i)


@pytest.fixture
def objeto():
    return ObjetoConfig.load("mandarina")


@pytest.fixture
def detector(objeto):
    return DetectorColorHSV(objeto)


def escena(centros, radio=0.0325, seed=0, **kw):
    intr = intrinsecos_d435i(848, 480)
    esferas = [EsferaSintetica(np.asarray(c, float), radio) for c in centros]
    return EscenaSintetica(intr=intr, esferas=esferas, seed=seed, **kw)


# --------------------------------------------------------------------------- #
def test_deteccion_encuentra_la_fruta(detector):
    fr = escena([[0.0, 0.0, 0.60]]).render()
    dets = detector.detectar(fr.color)
    assert len(dets) == 1
    d = dets[0]
    # La esfera esta centrada en el eje optico: su centroide debe caer en (cx, cy)
    assert abs(d.u - fr.intr.cx) < 4 and abs(d.v - fr.intr.cy) < 4
    assert d.metricas["circularidad"] > 0.85
    assert d.score > 0.7


def test_deteccion_multiple_y_separada(detector):
    fr = escena([[-0.12, 0.0, 0.70], [0.0, 0.05, 0.65], [0.13, -0.04, 0.75]]).render()
    dets = detector.detectar(fr.color)
    assert len(dets) == 3, [d.resumen() for d in dets]


def test_localizacion_exactitud_metodo_esfera(objeto, detector):
    """El error del centro debe quedar en pocos milimetros a distancias de trabajo."""
    loc = Localizador3D(objeto, metodo="esfera")
    errores = {}
    for z in (0.40, 0.60, 0.80, 1.00, 1.20):
        real = np.array([0.03, -0.02, z])
        fr = escena([real], seed=int(z * 100)).render()
        dets = detector.detectar(fr.color)
        assert dets, f"sin deteccion a z={z}"
        obj = loc.localizar(dets[0], fr.depth_m, fr.intr)
        assert obj.valido, obj.resumen()
        errores[z] = float(np.linalg.norm(obj.centro_cam - real)) * 1000
    # Tolerancia acorde al ruido simulado (sigma ~ 2.5 mm a 1 m, ~3.6 mm a 1.2 m)
    for z, e in errores.items():
        assert e < 12.0, f"z={z} m -> error {e:.1f} mm ({errores})"


def test_correccion_de_radio_es_necesaria(objeto, detector):
    """Sin corregir superficie->centro el error es del orden del radio."""
    loc = Localizador3D(objeto, metodo="esfera")
    real = np.array([0.0, 0.0, 0.70])
    fr = escena([real], seed=7).render()
    obj = loc.localizar(detector.detectar(fr.color)[0], fr.depth_m, fr.intr)
    assert obj.valido
    err_centro = np.linalg.norm(obj.centro_cam - real)
    err_superficie = np.linalg.norm(obj.superficie_cam - real)
    # La superficie esta ~1 radio mas cerca que el centro real
    assert err_superficie > 0.025, err_superficie
    assert err_centro < 0.012
    assert err_centro < err_superficie / 2


def test_metodo_centroide_tambien_funciona(objeto, detector):
    loc = Localizador3D(objeto, metodo="centroide")
    real = np.array([-0.05, 0.03, 0.55])
    fr = escena([real], seed=3).render()
    obj = loc.localizar(detector.detectar(fr.color)[0], fr.depth_m, fr.intr)
    assert obj.valido and obj.metodo == "centroide"
    assert np.linalg.norm(obj.centro_cam - real) < 0.015


def test_radio_estimado_se_aproxima_al_real(objeto, detector):
    loc = Localizador3D(objeto, metodo="esfera")
    radio_real = 0.030
    fr = escena([[0.0, 0.0, 0.65]], radio=radio_real, seed=11).render()
    obj = loc.localizar(detector.detectar(fr.color)[0], fr.depth_m, fr.intr)
    assert obj.valido and obj.radio_fuente == "medido"
    assert abs(obj.radio_m - radio_real) < 0.006, obj.resumen()


def test_rechaza_sin_profundidad(objeto, detector):
    """Sin datos de profundidad devuelve NaN e indica el motivo, no una posicion."""
    loc = Localizador3D(objeto, metodo="esfera")
    fr = escena([[0.0, 0.0, 0.60]]).render()
    dets = detector.detectar(fr.color)
    obj = loc.localizar(dets[0], np.zeros_like(fr.depth_m), fr.intr)
    assert not obj.valido
    assert np.all(np.isnan(obj.centro_cam))
    assert "profundidad" in obj.motivo


def test_rechaza_fuera_de_rango(objeto, detector):
    loc = Localizador3D(objeto, metodo="esfera", rango_m=(0.2, 0.5))
    fr = escena([[0.0, 0.0, 0.90]], seed=5).render()
    obj = loc.localizar(detector.detectar(fr.color)[0], fr.depth_m, fr.intr)
    assert not obj.valido and "rango" in obj.motivo


def test_cadena_completa_hasta_world(objeto, detector, tmp_path):
    """De pixel a {W}: la fruta sobre la mesa debe salir a la altura correcta."""
    # Camara a 1.2 m de altura mirando hacia abajo 45 grados
    T_world_cam = make_T(rpy_deg_to_R([-135.0, 0.0, 0.0]), [0.0, -0.5, 1.2])
    world = WorldFrameConfig(T_world_board=np.eye(4),
                             volume_min=np.array([-1.0, -1.0, -0.2]),
                             volume_max=np.array([1.0, 1.0, 1.5]))
    tf = TransformadorFrames(T_world_cam=T_world_cam, world=world)

    real_world = np.array([0.10, 0.05, 0.35])
    real_cam = tf.world_a_cam(real_world)
    assert real_cam[2] > 0, "la fruta debe quedar delante de la camara"

    fr = escena([real_cam], seed=21, plano_z_m=2.0).render()
    dets = detector.detectar(fr.color)
    assert dets
    obj = Localizador3D(objeto, metodo="esfera").localizar(
        dets[0], fr.depth_m, fr.intr)
    assert obj.valido, obj.resumen()

    est_world = tf.cam_a_world(obj.centro_cam)
    assert np.linalg.norm(est_world - real_world) < 0.015, (est_world, real_world)
    assert tf.en_volumen(est_world)
    assert not tf.en_volumen(np.array([5.0, 0.0, 0.3]))


def test_transformador_sin_base_avisa():
    tf = TransformadorFrames(T_world_cam=np.eye(4))
    with pytest.raises(ValueError, match="hand-eye|mano-ojo|T_base_world"):
        tf.world_a_base(np.zeros(3))
