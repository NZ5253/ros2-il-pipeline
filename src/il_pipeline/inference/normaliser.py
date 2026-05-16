"""
Normaliser for observation and action tensors at inference time.

Uses the per-feature mean/std saved at dataset finalisation. Keeping this in a
small dedicated class makes it trivial to swap normalisation strategies
(min-max, robust statistics, etc.) without touching the inference node.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class Normaliser:
    def __init__(self, stats: dict, device: torch.device) -> None:
        self.device = device
        self._obs_mean = self._to_tensor(stats["observation.state"]["mean"])
        self._obs_std = self._to_tensor(stats["observation.state"]["std"])
        self._act_mean = self._to_tensor(stats["action"]["mean"])
        self._act_std = self._to_tensor(stats["action"]["std"])

    def _to_tensor(self, v) -> torch.Tensor:
        return torch.as_tensor(np.asarray(v, dtype=np.float32), device=self.device)

    def normalise_obs(self, obs: dict | np.ndarray) -> torch.Tensor:
        if isinstance(obs, dict):
            state = obs["observation.state"]
        else:
            state = obs
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return (x - self._obs_mean) / self._obs_std

    def denormalise_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self._act_std + self._act_mean

    @classmethod
    def identity(cls, device: torch.device) -> "Normaliser":
        # Pass-through normaliser for cases where stats are unavailable.
        # Used at startup before a dataset is trained on.
        inst = cls.__new__(cls)
        inst.device = device
        inst._obs_mean = torch.zeros(1, device=device)
        inst._obs_std = torch.ones(1, device=device)
        inst._act_mean = torch.zeros(1, device=device)
        inst._act_std = torch.ones(1, device=device)
        return inst
