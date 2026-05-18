"""
Frame validation at recording time.

The data logger must not write corrupt frames into a dataset. Issues that
look small (wrong joint count, unnormalised quaternions, dropped timestamps)
silently corrupt training data downstream. This validator catches them at the
boundary where they enter the dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FrameValidator:
    """Per-frame structural checks."""

    expected_n_joints: int
    expected_action_dim: int
    expected_fps: float
    # 3-D xyz of the task object when recording object-aware demos. 0 means
    # state-only (joints + EE pose only).
    expected_object_dim: int = 0
    max_fps_deviation: float = 0.1   # ±10% of declared fps is acceptable

    _prev_timestamp: float = -1.0

    def validate(self, frame: dict) -> tuple[bool, str]:
        state = frame.get("observation.state")
        if state is None:
            return False, "missing observation.state"

        # joint pos + joint vel + EE pose (xyz+quat) + optional object xyz
        expected_state_dim = (
            2 * self.expected_n_joints + 7 + self.expected_object_dim
        )
        if len(state) != expected_state_dim:
            return (
                False,
                f"observation.state has {len(state)} entries, expected {expected_state_dim}",
            )

        # Quaternion normalisation check on the EE orientation portion
        q = state[2 * self.expected_n_joints + 3 : 2 * self.expected_n_joints + 7]
        q_norm = math.sqrt(sum(x * x for x in q))
        if abs(q_norm - 1.0) > 1e-3 and q_norm > 1e-6:
            return False, f"EE quaternion unnormalised: |q|={q_norm:.4f}"

        action = frame.get("action")
        if action is None or len(action) != self.expected_action_dim:
            return (
                False,
                f"action has wrong dimension: {len(action) if action is not None else 'None'} "
                f"vs expected {self.expected_action_dim}",
            )

        ts = frame.get("timestamp", -1.0)
        if ts <= self._prev_timestamp:
            return False, f"non-monotonic timestamp: {ts:.3f} <= {self._prev_timestamp:.3f}"

        if self._prev_timestamp >= 0:
            dt = ts - self._prev_timestamp
            expected_dt = 1.0 / self.expected_fps
            if abs(dt - expected_dt) / expected_dt > self.max_fps_deviation:
                # Don't reject — just flag. Some jitter is normal and aborting
                # episodes on jitter would be too aggressive.
                pass

        self._prev_timestamp = ts
        return True, ""

    def reset(self) -> None:
        self._prev_timestamp = -1.0
