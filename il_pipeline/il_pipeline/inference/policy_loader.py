"""
Policy loader for the inference node.

Loads a trained checkpoint and returns an object with a uniform
`predict_action_chunk(obs_tensor)` method, regardless of whether the
underlying model is a BC MLP or a LeRobot ACT policy. Also returns a
Normaliser for input/output normalisation.

The inference node treats these as opaque — no policy-specific code lives
in the ROS layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from il_pipeline.inference.normaliser import Normaliser


def load_policy(
    checkpoint_path: Path,
    policy_type: str,
    device: torch.device,
) -> tuple[nn.Module, Normaliser]:
    """
    Dispatches by policy_type.

    BC checkpoints are simple state_dict + config dicts; we rebuild a
    BCPolicy and load it.

    ACT checkpoints are state_dicts of a LeRobot ACTPolicy. We rebuild
    the policy with the same config and load weights. The returned object
    is wrapped with predict_action_chunk so the inference node sees a
    uniform interface.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    if policy_type == "bc":
        return _load_bc(ckpt, cfg, device)
    if policy_type == "act":
        return _load_act(ckpt, cfg, device)
    if policy_type == "diffusion":
        return _load_diffusion(ckpt, cfg, device)
    raise ValueError(f"unknown policy_type: {policy_type}")


# ── BC ───────────────────────────────────────────────────────────────────


def _load_bc(ckpt: dict, cfg: dict, device: torch.device):
    from il_pipeline.training.train import BCPolicy

    state_dim = cfg.get("state_dim") or _state_dim_from_ckpt(ckpt)
    action_dim = cfg.get("action_dim") or _action_dim_from_ckpt(ckpt)
    hidden = cfg.get("hidden") or _hidden_from_ckpt(ckpt)
    policy = BCPolicy(state_dim=state_dim, action_dim=action_dim, hidden=hidden).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    stats_path = Path(cfg.get("dataset_path", "")) / "meta" / "stats.json"
    if stats_path.exists():
        normaliser = Normaliser(stats=json.loads(stats_path.read_text()), device=device)
    else:
        normaliser = Normaliser.identity(device=device)

    return policy, normaliser


# ── ACT ──────────────────────────────────────────────────────────────────


def _load_act(ckpt: dict, cfg: dict, device: torch.device):
    """
    Rebuild a LeRobot ACTPolicy with the same config used at training time,
    load weights, and wrap with predict_action_chunk that the inference node
    expects.
    """
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    state_dim = cfg["state_dim"]
    action_dim = cfg["action_dim"]
    chunk_size = cfg["chunk_size"]
    n_joints = 7
    proprio_dim = 2 * n_joints
    env_dim = state_dim - proprio_dim

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(proprio_dim,)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(env_dim,)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    config = ACTConfig(
        n_obs_steps=1,
        chunk_size=chunk_size,
        # Must be 1 when temporal_ensemble_coeff is set: the policy is
        # queried every step and the ensembler blends predictions for the
        # current timestep across all overlapping chunks in the window.
        n_action_steps=1,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "STATE": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        dim_model=256,
        n_encoder_layers=2,
        n_decoder_layers=1,
        dim_feedforward=1024,
        kl_weight=10.0,
        dropout=0.1,
        use_vae=True,
        # Canonical ACT deployment from the original paper: blend overlapping
        # chunk predictions with exponential weighting. Without this, the
        # policy runs each chunk open-loop and only re-plans every chunk_size
        # steps — the brittle mode the paper argues against.
        temporal_ensemble_coeff=0.01,
        push_to_hub=False,
    )
    inner = ACTPolicy(config).to(device)
    inner.load_state_dict(ckpt["state_dict"])
    inner.eval()

    return _ActAdapter(inner, proprio_dim).to(device), Normaliser.identity(device=device)


class _ActAdapter(nn.Module):
    """
    Wraps a LeRobot ACTPolicy so the inference node sees the same
    `predict_action_chunk(obs_tensor)` API as the BC policy.

    Splits the 21-D `observation.state` tensor into the STATE+ENV streams
    ACT expects, calls ACTPolicy.select_action (which runs temporal
    ensembling internally), and returns the ensembled action as a
    single-step chunk.
    """

    def __init__(self, inner, proprio_dim: int) -> None:
        super().__init__()
        self.inner = inner
        self.proprio_dim = proprio_dim

    def reset(self) -> None:
        # Clears the temporal ensembler buffer between episodes — otherwise
        # the first few actions of a new rollout are still influenced by
        # the predictions from the end of the previous one.
        self.inner.reset()

    @torch.inference_mode()
    def predict_action_chunk(self, observation: torch.Tensor) -> torch.Tensor:
        # observation comes in as a 1-D 21-vector from the inference node.
        # ACT expects a batched dict with STATE + ENV.
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        state = observation[..., : self.proprio_dim]
        env = observation[..., self.proprio_dim :]
        batch = {
            "observation.state": state,
            "observation.environment_state": env,
        }
        # select_action pops one action from ACT's internal chunk queue and
        # returns shape [batch, action_dim]. Squeeze batch and reshape into
        # the [chunk_size, action_dim] contract the inference node expects.
        action = self.inner.select_action(batch)
        return action.squeeze(0).unsqueeze(0)   # [1, action_dim]


# ── Diffusion Policy ─────────────────────────────────────────────────────


def _load_diffusion(ckpt: dict, cfg: dict, device: torch.device):
    """Rebuild LeRobot Diffusion Policy with the same config used at
    training time, load weights, and wrap with predict_action_chunk."""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    state_dim = cfg["state_dim"]
    action_dim = cfg["action_dim"]
    horizon = cfg.get("horizon") or cfg.get("chunk_size", 16)
    n_joints = 7
    proprio_dim = 2 * n_joints
    env_dim = state_dim - proprio_dim

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(proprio_dim,)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(env_dim,)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    config = DiffusionConfig(
        n_obs_steps=1,
        horizon=horizon,
        n_action_steps=max(1, horizon // 2),
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "STATE": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        down_dims=(64, 128, 256),
        num_inference_steps=10,
        push_to_hub=False,
    )
    inner = DiffusionPolicy(config).to(device)
    inner.load_state_dict(ckpt["state_dict"])
    inner.eval()

    return _DiffusionAdapter(inner, proprio_dim).to(device), Normaliser.identity(device=device)


class _DiffusionAdapter(nn.Module):
    """
    Wraps LeRobot DiffusionPolicy so the inference node sees the same
    `predict_action_chunk(obs)` API as BC and ACT.

    DP's select_action runs the full DDPM denoising chain on each call and
    pops one action from its internal queue (queue refills every
    n_action_steps). Observations need an explicit time dim since
    n_obs_steps=1.
    """

    def __init__(self, inner, proprio_dim: int) -> None:
        super().__init__()
        self.inner = inner
        self.proprio_dim = proprio_dim

    def reset(self) -> None:
        self.inner.reset()

    @torch.inference_mode()
    def predict_action_chunk(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        state = observation[..., : self.proprio_dim]
        env = observation[..., self.proprio_dim :]
        batch = {
            # Diffusion expects [B, n_obs_steps, dim]; we trained with
            # n_obs_steps=1, so add the singleton time dim.
            "observation.state": state.unsqueeze(1),
            "observation.environment_state": env.unsqueeze(1),
        }
        action = self.inner.select_action(batch)
        return action.squeeze(0).unsqueeze(0)   # [1, action_dim]


# ── Helpers for old BC checkpoints without explicit dims ────────────────


def _state_dim_from_ckpt(ckpt: dict) -> int:
    for k, v in ckpt["state_dict"].items():
        if "weight" in k and v.ndim == 2:
            return v.shape[1]
    raise ValueError("could not infer state_dim from checkpoint")


def _action_dim_from_ckpt(ckpt: dict) -> int:
    last = None
    for k, v in ckpt["state_dict"].items():
        if "weight" in k and v.ndim == 2:
            last = v
    if last is None:
        raise ValueError("could not infer action_dim from checkpoint")
    return last.shape[0]


def _hidden_from_ckpt(ckpt: dict) -> int:
    for k, v in ckpt["state_dict"].items():
        if "net.0.weight" in k:
            return v.shape[0]
    return 256
