"""
Reference BC policy definition.

This module owns the small behaviour-cloning policy used as a baseline in the
pipeline. The canonical training entry point is `scripts/train.py` at the
repo root, which dispatches to either this BC model or LeRobot's ACTPolicy
based on the `--policy` flag.

Kept under `il_pipeline.training` (rather than the looser `model/`) because
it is the training-time policy implementation; inference uses the same class
loaded from a checkpoint via `il_pipeline.inference.policy_loader`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCPolicy(nn.Module):
    """Three-layer MLP mapping `observation.state` to a single action vector.

    Used as a sanity-baseline IL policy: fast to train, easy to reason about,
    and a known-weak reference point so the value of ACT / Diffusion is
    measurable rather than asserted.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, batch: dict) -> dict:
        pred = self.net(batch["observation.state"])
        loss = F.l1_loss(pred, batch["action"])
        return {"loss": loss, "action": pred}

    @torch.inference_mode()
    def predict_action_chunk(self, observation: torch.Tensor) -> torch.Tensor:
        """Return shape [chunk_size, action_dim] = [1, action_dim] for BC.

        Matches the contract the inference node expects from any policy.
        """
        a = self.net(observation)
        return a.unsqueeze(0)
