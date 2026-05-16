"""
Policy factory.

Returns a torch.nn.Module given a policy type string. ACT and Diffusion
Policy are imported from the lerobot library on the lab PC; BC is the
self-contained reference baseline.

This indirection keeps the rest of the training pipeline policy-agnostic.
"""

from __future__ import annotations

import torch.nn as nn


def build_policy(
    policy_type: str,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 50,
) -> nn.Module:
    if policy_type == "bc":
        from il_pipeline.training.train import BCPolicy
        return BCPolicy(state_dim=state_dim, action_dim=action_dim)

    if policy_type == "act":
        # On the lab PC: from lerobot.common.policies.act.modeling_act import ACTPolicy
        # ACTPolicy(config) with chunk_size, state_dim, action_dim, etc.
        raise NotImplementedError(
            "ACT policy is loaded from the lerobot library on the lab PC. "
            "See docs/01_technical_concept.md section 6.2 for configuration."
        )

    if policy_type == "diffusion":
        # On the lab PC: from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        raise NotImplementedError(
            "Diffusion Policy is loaded from the lerobot library on the lab PC."
        )

    raise ValueError(f"unknown policy_type: {policy_type}")
