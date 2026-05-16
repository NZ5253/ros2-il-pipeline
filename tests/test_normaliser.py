"""Tests for the inference-time normaliser."""

import numpy as np
import pytest
import torch

from il_pipeline.inference.normaliser import Normaliser


def _stats(state_dim: int = 21, action_dim: int = 7) -> dict:
    return {
        "observation.state": {
            "mean": [0.0] * state_dim,
            "std": [2.0] * state_dim,
            "min": [-10.0] * state_dim,
            "max": [10.0] * state_dim,
        },
        "action": {
            "mean": [1.0] * action_dim,
            "std": [0.5] * action_dim,
            "min": [-1.0] * action_dim,
            "max": [3.0] * action_dim,
        },
    }


def test_normalise_obs_from_dict():
    n = Normaliser(_stats(), device=torch.device("cpu"))
    obs = {"observation.state": np.full(21, 4.0, dtype=np.float32)}
    out = n.normalise_obs(obs)
    # (4 - 0) / 2 = 2.0
    assert torch.allclose(out, torch.full((21,), 2.0))


def test_normalise_obs_from_ndarray():
    n = Normaliser(_stats(), device=torch.device("cpu"))
    obs = np.full(21, 4.0, dtype=np.float32)
    out = n.normalise_obs(obs)
    assert torch.allclose(out, torch.full((21,), 2.0))


def test_denormalise_action_is_inverse():
    n = Normaliser(_stats(), device=torch.device("cpu"))
    action_norm = torch.zeros(7)
    out = n.denormalise_action(action_norm)
    # 0 * 0.5 + 1 = 1.0 → mean
    assert torch.allclose(out, torch.ones(7))


def test_identity_normaliser_is_pass_through():
    n = Normaliser.identity(device=torch.device("cpu"))
    obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = n.normalise_obs(obs)
    # Identity uses mean=0, std=1, so (x - 0) / 1 = x
    assert torch.allclose(out, torch.tensor([1.0, 2.0, 3.0]))
