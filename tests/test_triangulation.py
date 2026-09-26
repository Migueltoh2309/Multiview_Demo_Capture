"""Test sintetico: rig de 3 camaras conocidas -> triangular puntos conocidos.

Valida la cadena completa CameraModel.project -> triangulate_point sin
depender de hardware ni de calibracion real.
"""
import numpy as np

from tesis_picking.common.camera_model import CameraModel, CameraRig, Intrinsics
from tesis_picking.common.transforms import invert_T, make_T
from tesis_picking.common.triangulation import triangulate_point


def look_at(eye, target, up=(0, 0, 1)):
    """T_world_cam para una camara en `eye` mirando a `target` (convencion
    OpenCV: x derecha, y abajo, z hacia adelante)."""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float), z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.column_stack([x, y, z]), eye)


def make_rig(with_distortion=True):
    K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    D = np.array([-0.12, 0.05, 0.001, -0.0005, 0.0]) if with_distortion else np.zeros(5)
    poses = {
        "cam1": look_at([-1.2, -0.8, 0.9], [0, 0, 0.3]),
        "cam2": look_at([0.0, -1.0, 1.8], [0, 0, 0.2]),
        "cam3": look_at([1.2, -0.8, 0.9], [0, 0, 0.3]),
    }
    return CameraRig([
        CameraModel(cid, Intrinsics(K.copy(), D.copy(), (1280, 720)), T)
        for cid, T in poses.items()
    ])


def test_project_normalize_roundtrip():
    rig = make_rig()
    X = np.array([0.05, 0.1, 0.35])
    for cam in rig:
        uv = cam.project(X.reshape(1, 3)).reshape(2)
        assert np.all(np.isfinite(uv))
        xn = cam.normalize_points(uv)
        Xc = cam.R_cam_world @ X + cam.t_cam_world
        assert np.allclose(xn, Xc[:2] / Xc[2], atol=1e-4)


def test_triangulation_exact_three_views():
    rig = make_rig()
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(200):
        X = rng.uniform([-0.4, -0.4, 0.0], [0.4, 0.4, 0.7])
        obs = {c.cam_id: c.project(X.reshape(1, 3)).reshape(2) for c in rig}
        tri = triangulate_point(obs, rig)
        assert tri.valid and tri.n_cameras == 3
        errs.append(np.linalg.norm(tri.X - X))
    assert np.max(errs) < 1e-6, f"max err {np.max(errs)}"


def test_triangulation_two_views_and_occlusion():
    rig = make_rig()
    X = np.array([0.1, 0.05, 0.4])
    obs = {c.cam_id: c.project(X.reshape(1, 3)).reshape(2) for c in rig}
    obs["cam2"] = np.array([np.nan, np.nan])          # camara ocluida
    tri = triangulate_point(obs, rig)
    assert tri.valid and tri.n_cameras == 2
    assert np.linalg.norm(tri.X - X) < 1e-6
    # Con una sola camara no debe reconstruir: NaN, nunca ceros.
    obs["cam3"] = np.array([np.nan, np.nan])
    tri1 = triangulate_point(obs, rig)
    assert not tri1.valid and np.all(np.isnan(tri1.X))


def test_noise_and_reprojection_gate():
    rig = make_rig()
    rng = np.random.default_rng(7)
    X = np.array([0.0, 0.0, 0.35])
    obs = {c.cam_id: c.project(X.reshape(1, 3)).reshape(2) + rng.normal(0, 0.5, 2)
           for c in rig}
    tri = triangulate_point(obs, rig)
    assert tri.valid
    assert np.linalg.norm(tri.X - X) < 5e-3          # ruido 0.5 px -> < 5 mm
    assert tri.rms_reproj_px < 1.0
    # Una observacion groseramente equivocada debe ser rechazada por el gate.
    obs["cam3"] = obs["cam3"] + np.array([90.0, -60.0])
    bad = triangulate_point(obs, rig, max_reproj_px=8.0)
    assert not bad.valid and np.all(np.isnan(bad.X))
