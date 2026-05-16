"""Tests for the BC reference policy.

Trains a 1-layer MLP on a tiny synthetic dataset and checks that the loss
goes down. Doesn't claim convergence — only that the forward/backward
pipeline is intact.
"""

import numpy as np
import torch

from il_pipeline.training.train import BCPolicy


def test_bc_policy_forward_shapes():
    p = BCPolicy(state_dim=21, action_dim=7)
    batch = {
        "observation.state": torch.randn(4, 21),
        "action": torch.randn(4, 7),
    }
    out = p(batch)
    assert "loss" in out
    assert "action" in out
    assert out["action"].shape == (4, 7)
    assert out["loss"].ndim == 0


def test_bc_policy_loss_decreases_on_overfit():
    torch.manual_seed(0)
    p = BCPolicy(state_dim=8, action_dim=2, hidden=32)
    optim = torch.optim.AdamW(p.parameters(), lr=1e-3)

    # Tiny fixed dataset
    obs = torch.randn(16, 8)
    act = torch.randn(16, 2)

    losses = []
    for _ in range(200):
        out = p({"observation.state": obs, "action": act})
        optim.zero_grad()
        out["loss"].backward()
        optim.step()
        losses.append(out["loss"].item())

    # Loss should decrease meaningfully when overfitting a tiny dataset
    assert losses[-1] < losses[0] * 0.5


def test_predict_action_chunk_has_chunk_axis():
    p = BCPolicy(state_dim=21, action_dim=7)
    obs = torch.randn(21)
    chunk = p.predict_action_chunk(obs)
    assert chunk.shape == (1, 7)
