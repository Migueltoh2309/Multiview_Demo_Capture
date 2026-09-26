"""Tests de la capa de captura que no necesitan hardware."""
import numpy as np

from tesis_picking.common.config import load_cameras, load_charuco, load_world_frame
from tesis_picking.common.timebase import (SyncMetrics, frame_interval_stats,
                                           measured_fps)
from tesis_picking.demostraciones_humanas.capture.multi_capture import FrameSet


def test_sync_metrics():
    m = SyncMetrics(["cam1", "cam2", "cam3"])
    # cam2 va 5 ms por detras, cam3 12 ms
    for k in range(100):
        t = k / 30.0
        m.add({"cam1": t, "cam2": t + 0.005, "cam3": t + 0.012})
    s = m.summary()
    assert np.isclose(s["dt_max_ms"], 12.0, atol=0.1)
    assert np.isclose(s["dt_mean_ms"], 12.0, atol=0.1)
    assert np.isclose(s["dt_cam1_cam2_mean_ms"], -5.0, atol=0.1)
    assert s["n_frames"] == 100 and s["n_frames_incompletos"] == 0


def test_sync_metrics_incomplete_frames():
    m = SyncMetrics(["cam1", "cam2"])
    m.add({"cam1": 0.0, "cam2": 0.01})
    m.add({"cam1": 0.033})                  # cam2 no entrego frame
    s = m.summary()
    assert s["n_frames"] == 2 and s["n_frames_incompletos"] == 1


def test_frameset_spread_and_row():
    f = np.zeros((4, 4, 3), np.uint8)
    fs = FrameSet(frame_id=7,
                  frames={"cam1": f, "cam2": None, "cam3": f},
                  timestamps={"cam1": 1.000, "cam2": float("nan"), "cam3": 1.008},
                  t_trigger=0.999)
    assert not fs.complete
    assert fs.valid_ids == ["cam1", "cam3"]
    assert np.isclose(fs.spread_s, 0.008)
    row = fs.sync_row()
    assert row["frame_id"] == 7 and row["ok_cam2"] == 0 and row["ok_cam1"] == 1
    assert np.isclose(row["dt_max_ms"], 8.0)


def test_frame_interval_stats_detects_drop():
    # 30 FPS con un frame perdido en medio
    t = list(np.arange(0, 1.0, 1 / 30.0))
    del t[10]
    st = frame_interval_stats(t)
    assert st["dropped_est"] == 1
    assert 25 < st["fps"] < 31
    assert np.isnan(measured_fps([1.0]))


def test_configs_load_and_validate():
    """Los YAML del repo deben cargar y ser coherentes."""
    cams = load_cameras()
    assert len(cams.cameras) >= 1
    assert len(set(cams.ids)) == len(cams.ids)
    for c in cams.cameras:
        assert c.buffersize >= 2, "buffersize 1 reduce el FPS a la mitad"

    ch = load_charuco()
    assert ch.marker_length_m < ch.square_length_m
    assert ch.n_corners == (ch.squares_x - 1) * (ch.squares_y - 1)

    wf = load_world_frame()
    assert wf.T_world_board.shape == (4, 4)
    assert wf.inside_volume(np.array([0.0, 0.0, 0.5]))
    assert not wf.inside_volume(np.array([9.0, 0.0, 0.5]))
    assert not wf.inside_volume(np.array([0.0, np.nan, 0.5]))
