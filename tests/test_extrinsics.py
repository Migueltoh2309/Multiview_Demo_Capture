"""Test sintetico de la calibracion extrinseca + validacion metrologica.

Se fabrica un rig conocido y se generan detecciones ChArUco proyectando las
esquinas reales del tablero (con ruido de 0.3 px). El test comprueba que el
bundle adjustment recupera las poses verdaderas y que las metricas de
validacion dan errores del orden esperado.
"""
import numpy as np

from tesis_picking.common.camera_model import Intrinsics
from tesis_picking.common.config import CharucoConfig
from tesis_picking.common.transforms import (angle_between_R_deg, make_T,
                                             rpy_deg_to_R, split_T)
from tesis_picking.demostraciones_humanas.calibration.charuco import (
    BoardDetection, CharucoBoardWrapper)
from tesis_picking.demostraciones_humanas.calibration.extrinsics import calibrate_extrinsics
from tesis_picking.demostraciones_humanas.calibration.validation import (
    check_board_reconstruction, known_distance_check)

from test_triangulation import look_at, make_rig


def synth_detection(cam, board, T_world_board, rng, noise_px=0.3, visible=None):
    """Proyecta las esquinas del tablero en una camara -> BoardDetection."""
    obj = board.object_points()
    ids = np.arange(len(obj)) if visible is None else np.asarray(visible)
    Xw = obj[ids] @ T_world_board[:3, :3].T + T_world_board[:3, 3]
    uv = cam.project(Xw)
    keep = np.all(np.isfinite(uv), axis=1)
    # descarta lo que cae fuera del encuadre
    w, h = cam.intr.image_size
    keep &= (uv[:, 0] > 0) & (uv[:, 0] < w) & (uv[:, 1] > 0) & (uv[:, 1] < h)
    if keep.sum() < 6:
        return None
    uv = uv[keep] + rng.normal(0, noise_px, (int(keep.sum()), 2))
    return BoardDetection(
        charuco_corners=uv.reshape(-1, 1, 2).astype(np.float32),
        charuco_ids=ids[keep].reshape(-1, 1).astype(np.int32),
        marker_corners=None, marker_ids=None, image_size=(w, h))


def make_board():
    return CharucoBoardWrapper(CharucoConfig(
        dictionary="DICT_5X5_1000", squares_x=8, squares_y=6,
        square_length_m=0.030, marker_length_m=0.022))


def build_scene(seed=3, n_poses=14, noise_px=0.3):
    rig_true = make_rig()
    board = make_board()
    rng = np.random.default_rng(seed)
    poses = {"anchor": np.eye(4)}
    for k in range(n_poses):
        poses[f"shot_{k:02d}"] = make_T(
            rpy_deg_to_R(rng.uniform([-35, -35, -40], [35, 35, 40])),
            rng.uniform([-0.25, -0.25, -0.05], [0.25, 0.25, 0.45]))
    dets = {
        key: {c.cam_id: synth_detection(c, board, T, rng, noise_px) for c in rig_true}
        for key, T in poses.items()
    }
    return rig_true, board, poses, dets


def test_extrinsics_exact_without_noise():
    """Sin ruido de deteccion, el estimador debe recuperar las poses EXACTAS.

    Este es el test de correccion del bundle adjustment: cualquier error de
    convencion (T_world_cam vs T_cam_world, orden de composicion) lo rompe.
    """
    rig_true, board, poses, dets = build_scene(seed=1, n_poses=10, noise_px=0.0)
    res = calibrate_extrinsics(
        dets, board, {c.cam_id: c.intr for c in rig_true},
        anchor_key="anchor", T_world_anchor=np.eye(4))
    assert res.rms_global_px < 1e-3, res.rms_global_px
    for cam in res.rig:
        R_t, t_t = split_T(rig_true[cam.cam_id].T_world_cam)
        R_e, t_e = split_T(cam.T_world_cam)
        assert np.linalg.norm(t_t - t_e) < 1e-4
        assert angle_between_R_deg(R_t, R_e) < 1e-3


def test_extrinsics_recovers_true_poses():
    """Con ruido realista de 0.3 px, las poses quedan dentro de tolerancia.

    Nota: la POSICION de cada camara es la magnitud peor condicionada del
    problema (patron plano, pequeno y lejano) y no mejora anadiendo mas poses
    del tablero. Lo que si es preciso, y es lo que importa para la tesis, es la
    reconstruccion 3D --- ver `test_metrology_checks`: los errores de pose entre
    camaras estan correlados y se cancelan en gran medida al triangular.
    """
    rig_true, board, poses, dets = build_scene()
    res = calibrate_extrinsics(
        dets, board,
        {c.cam_id: c.intr for c in rig_true},
        anchor_key="anchor", T_world_anchor=np.eye(4))

    assert res.rms_global_px < 0.6, res.rms_global_px
    for cam in res.rig:
        R_t, t_t = split_T(rig_true[cam.cam_id].T_world_cam)
        R_e, t_e = split_T(cam.T_world_cam)
        pos_err_mm = np.linalg.norm(t_t - t_e) * 1000
        rot_err_deg = angle_between_R_deg(R_t, R_e)
        assert pos_err_mm < 12.0, f"{cam.cam_id}: {pos_err_mm:.2f} mm"
        assert rot_err_deg < 0.35, f"{cam.cam_id}: {rot_err_deg:.3f} deg"


def test_extrinsics_with_camera_that_never_sees_anchor():
    """cam3 no ve la pose ancla: debe quedar referida a {W} por encadenamiento."""
    rig_true, board, poses, dets = build_scene(seed=11)
    dets["anchor"]["cam3"] = None
    res = calibrate_extrinsics(
        dets, board, {c.cam_id: c.intr for c in rig_true},
        anchor_key="anchor", T_world_anchor=np.eye(4))
    assert "cam3" in res.rig.ids
    R_t, t_t = split_T(rig_true["cam3"].T_world_cam)
    R_e, t_e = split_T(res.rig["cam3"].T_world_cam)
    assert np.linalg.norm(t_t - t_e) * 1000 < 12.0
    assert angle_between_R_deg(R_t, R_e) < 0.35


def test_metrology_checks():
    rig_true, board, poses, dets = build_scene(seed=5)
    res = calibrate_extrinsics(
        dets, board, {c.cam_id: c.intr for c in rig_true},
        anchor_key="anchor", T_world_anchor=np.eye(4))

    chk = check_board_reconstruction(
        dets["anchor"], board, res.rig,
        frame_key="anchor", T_world_board=np.eye(4))
    s = chk.summary()
    assert chk.n_points > 20
    assert s["error_3d_rms_mm"] < 2.0, s
    assert s["error_distancia_rms_mm"] < 1.5, s
    assert chk.planarity_rms_mm < 1.5

    kd = known_distance_check(dets["anchor"], board, res.rig, 0, 34)
    assert kd["ok"] and kd["error_mm"] < 2.0, kd
