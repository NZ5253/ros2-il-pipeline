"""
Policy loader for the inference node.

Loads a trained checkpoint and returns (policy, normaliser). The inference
node treats these as opaque objects with `predict_action_chunk(obs)`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from il_pipeline.inference.normaliser import Normaliser
from il_pipeline.training.policy_factory import build_policy


def load_policy(
    checkpoint_path: Path,
    policy_type: str,
    device: torch.device,
) -> Tuple[nn.Module, Normaliser]:
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = ckpt.get("config", {})
    state_dim = cfg.get("state_dim") or _state_dim_from_ckpt(ckpt)
    action_dim = cfg.get("action_dim") or _action_dim_from_ckpt(ckpt)
    chunk_size = cfg.get("chunk_size", 1)

    policy = build_policy(
        policy_type=policy_type,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
    ).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    # Normaliser uses stats saved alongside the dataset
    stats_path = Path(cfg.get("dataset_path", "")) / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        normaliser = Normaliser(stats=stats, device=device)
    else:
        normaliser = Normaliser.identity(device=device)

    return policy, normaliser


def _state_dim_from_ckpt(ckpt: dict) -> int:
    # Best-effort: infer state_dim from the first Linear layer in the state dict
    for k, v in ckpt["state_dict"].items():
        if "weight" in k and v.ndim == 2:
            return v.shape[1]
    raise ValueError("could not infer state_dim from checkpoint")


def _action_dim_from_ckpt(ckpt: dict) -> int:
    # Best-effort: infer action_dim from the last Linear weight
    last_linear = None
    for k, v in ckpt["state_dict"].items():
        if "weight" in k and v.ndim == 2:
            last_linear = v
    if last_linear is None:
        raise ValueError("could not infer action_dim from checkpoint")
    return last_linear.shape[0]
