"""Tests for the per-frame validator used by the data logger."""


import numpy as np
from il_pipeline.dataset.frame_validator import FrameValidator


def _make_frame(
    n_joints: int = 7,
    action_dim: int = 7,
    timestamp: float = 0.1,
    quat_unit: bool = True,
) -> dict:
    joint_pos = np.zeros(n_joints, dtype=np.float32)
    joint_vel = np.zeros(n_joints, dtype=np.float32)
    ee_xyz = np.zeros(3, dtype=np.float32)
    if quat_unit:
        ee_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        ee_quat = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32) * 1.1  # unnormalised
    state = np.concatenate([joint_pos, joint_vel, ee_xyz, ee_quat])
    return {
        "observation.state": state,
        "action": np.zeros(action_dim, dtype=np.float32),
        "timestamp": timestamp,
        "frame_index": 0,
    }


def test_accepts_well_formed_frame():
    v = FrameValidator(expected_n_joints=7, expected_action_dim=7, expected_fps=30.0)
    ok, reason = v.validate(_make_frame(timestamp=0.0))
    assert ok, reason


def test_rejects_wrong_state_dim():
    v = FrameValidator(expected_n_joints=7, expected_action_dim=7, expected_fps=30.0)
    frame = _make_frame()
    frame["observation.state"] = frame["observation.state"][:-1]  # drop one element
    ok, reason = v.validate(frame)
    assert not ok
    assert "observation.state" in reason


def test_rejects_wrong_action_dim():
    v = FrameValidator(expected_n_joints=7, expected_action_dim=7, expected_fps=30.0)
    frame = _make_frame()
    frame["action"] = frame["action"][:-1]
    ok, reason = v.validate(frame)
    assert not ok
    assert "action" in reason


def test_rejects_unnormalised_quaternion():
    v = FrameValidator(expected_n_joints=7, expected_action_dim=7, expected_fps=30.0)
    frame = _make_frame(quat_unit=False)
    ok, reason = v.validate(frame)
    assert not ok
    assert "quaternion" in reason


def test_rejects_non_monotonic_timestamp():
    v = FrameValidator(expected_n_joints=7, expected_action_dim=7, expected_fps=30.0)
    assert v.validate(_make_frame(timestamp=0.0))[0]
    ok, reason = v.validate(_make_frame(timestamp=0.0))   # same timestamp again
    assert not ok
    assert "monotonic" in reason


def test_accepts_after_reset():
    v = FrameValidator(expected_n_joints=7, expected_action_dim=7, expected_fps=30.0)
    assert v.validate(_make_frame(timestamp=0.1))[0]
    v.reset()
    ok, _ = v.validate(_make_frame(timestamp=0.0))   # fresh episode, low timestamp ok
    assert ok


def test_zero_quaternion_treated_as_unset():
    """An all-zero quaternion (norm 0) is interpreted as 'not populated' and not rejected."""
    v = FrameValidator(expected_n_joints=7, expected_action_dim=7, expected_fps=30.0)
    frame = _make_frame()
    # Zero out the quaternion entirely
    frame["observation.state"][-4:] = 0.0
    ok, _ = v.validate(frame)
    assert ok
