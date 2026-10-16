"""Missing optional rendering assets must not prevent Go2 playback."""

import numpy as np
import mujoco
import pytest

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2_config import Go2Cfg
from gym.utils.task_registry import task_registry
from gym.envs.go2.go2_config import Go2RunnerCfg


def test_missing_visual_donor_preserves_go2_physics(tmp_path):
    cfg = Go2Cfg()
    task_registry.convert_frequencies_to_params(cfg, Go2RunnerCfg())
    backend = MuJocoCPUBackend()
    baseline = backend._load_model(cfg)
    with pytest.warns(RuntimeWarning, match="using URDF geometry"):
        fallback = backend._load_model(
            cfg, visual_model_path=str(tmp_path / "missing-go2.xml")
        )
    assert fallback.nq == baseline.nq == 19
    assert fallback.nv == baseline.nv == 18
    assert fallback.ngeom == baseline.ngeom
    # Without detailed meshes, the robot's collision shapes are its picture.
    robot = fallback.geom_bodyid != 0
    visible_groups = mujoco.MjvOption().geomgroup
    assert robot.any()
    assert np.all(visible_groups[fallback.geom_group[robot]] != 0)
    assert np.all(fallback.geom_rgba[robot, 3] > 0)

    for name in (
        "body_mass",
        "body_inertia",
        "jnt_range",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_contype",
        "geom_conaffinity",
    ):
        np.testing.assert_array_equal(getattr(fallback, name), getattr(baseline, name))


def test_existing_invalid_visual_donor_still_reports_error(tmp_path):
    donor = tmp_path / "invalid.xml"
    donor.write_text("<broken", encoding="utf8")
    from xml.etree.ElementTree import ParseError

    cfg = Go2Cfg()
    task_registry.convert_frequencies_to_params(cfg, Go2RunnerCfg())
    with pytest.raises(ParseError):
        MuJocoCPUBackend()._load_model(cfg, visual_model_path=str(donor))
