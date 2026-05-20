"""
Unified training script for BC and ACT policies.

Same script runs on the CPU (this dev box) and on the workstation GPU — the
only difference is `--device cuda:0` and longer `--steps`. All other
arguments are identical, so the workstation work reduces to swapping two
flags.

Usage (CPU smoke run on dev box):
    python3 scripts/train.py --policy bc \
        --dataset /tmp/mybotshop_demos/panda_pickplace_v2 \
        --output runs/panda_bc \
        --epochs 200 --batch-size 64 --device cpu

Usage (workstation, GPU):
    python3 scripts/train.py --policy act \
        --dataset /tmp/mybotshop_demos/panda_pickplace_v2 \
        --output runs/panda_act \
        --epochs 2000 --batch-size 32 --chunk-size 50 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "il_pipeline"))

from il_pipeline.training.lerobot_torch_dataset import LeRobotTorchDataset  # noqa: E402
from il_pipeline.training.train import BCPolicy  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────


def build_bc_policy(state_dim: int, action_dim: int, hidden: int = 256) -> torch.nn.Module:
    return BCPolicy(state_dim=state_dim, action_dim=action_dim, hidden=hidden)


def build_act_policy(state_dim: int, action_dim: int, chunk_size: int,
                     dataset_stats: dict | None = None):
    """Build the LeRobot ACT policy. Only imported on demand to avoid a hard
    dependency on the lerobot package for the BC training path.

    Targets lerobot 0.5.x (post-restructure). PolicyFeature/FeatureType is the
    config surface; normalisation stats come from our LeRobotDataset's stats.json.
    """
    try:
        from lerobot.configs.types import (
            FeatureType,
            NormalizationMode,
            PolicyFeature,
        )
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError as e:
        raise RuntimeError(
            "lerobot is required for --policy act. Install with `pip install lerobot`."
        ) from e

    # Split our 21-D observation.state into the two feature streams ACT
    # expects: STATE (proprioception: joint pos + vel) and ENV (everything
    # external to the robot — here, the end-effector pose, which is a kinematic
    # function of joints but ACT happily treats it as an environment cue).
    # Splitting on the proprio/env boundary keeps the API contract clean and
    # gives us the option to drop in image observations later as VISUAL without
    # restructuring.
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
    normalization_mapping = {
        "STATE": NormalizationMode.MEAN_STD,
        "ENV": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    config = ACTConfig(
        n_obs_steps=1,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping=normalization_mapping,
        # Modest defaults that train in reasonable wall-clock on CPU.
        dim_model=256,
        n_encoder_layers=2,
        n_decoder_layers=1,
        dim_feedforward=1024,
        kl_weight=10.0,
        dropout=0.1,
        use_vae=True,
        # No vision backbone needed (state-only policy).
        push_to_hub=False,
    )
    return ACTPolicy(config, dataset_stats=dataset_stats)


def collate_for_act(batch_list: list[dict]) -> dict:
    """Stack a list of single-frame samples into the dict-of-batched-tensors
    that ACT expects. Each sample's `action` is a chunk of shape [k, A]."""
    out = {}
    for key in batch_list[0]:
        out[key] = torch.stack([b[key] for b in batch_list])
    return out


def build_diffusion_policy(state_dim: int, action_dim: int, horizon: int,
                           dataset_stats: dict | None = None):
    """Build LeRobot Diffusion Policy for the same state-only setup as ACT.

    Uses a smaller UNet (down_dims=(64,128,256), ~4.5M params) rather than
    the default (512,1024,2048) which gives ~250M params and would overfit
    a 40-demo dataset. Sized to match ACT (5.84M) for a fair comparison.
    """
    try:
        from lerobot.configs.types import (
            FeatureType,
            NormalizationMode,
            PolicyFeature,
        )
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    except ImportError as e:
        raise RuntimeError(
            "lerobot is required for --policy diffusion. Install with `pip install lerobot`."
        ) from e

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
        # 10 DDPM steps at inference keeps latency well under 33 ms; default
        # is 100 which is far slower than the control budget allows.
        num_inference_steps=10,
        push_to_hub=False,
    )
    return DiffusionPolicy(config, dataset_stats=dataset_stats)


# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=["bc", "act", "diffusion"], default="bc")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="ACT only — number of future actions to predict per step")
    parser.add_argument("--horizon", type=int, default=16,
                        help="Diffusion only — UNet planning horizon")
    parser.add_argument("--hidden", type=int, default=256, help="BC only — hidden dim")
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu",
                        help="cpu | cuda:0 | cuda:1 etc.")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── dataset ──────────────────────────────────────────────────────────
    if args.policy == "act":
        chunk = args.chunk_size
    elif args.policy == "diffusion":
        chunk = args.horizon
    else:
        chunk = 1
    # ACT and Diffusion both normalize internally from dataset_stats; skip
    # dataset-level normalization for them to avoid double-normalizing.
    ds = LeRobotTorchDataset(args.dataset, chunk_size=chunk,
                             normalize=(args.policy == "bc"))
    print(f"[dataset] {len(ds)} samples  state_dim={ds.state_dim}  action_dim={ds.action_dim}  chunk={chunk}")

    n_val = max(1, int(len(ds) * args.validation_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # ── policy ───────────────────────────────────────────────────────────
    if args.policy == "bc":
        policy = build_bc_policy(state_dim=ds.state_dim, action_dim=ds.action_dim,
                                 hidden=args.hidden).to(device)
    else:
        # ACT and Diffusion both need dataset normalisation stats up front so
        # they build normalisers that get saved into the policy state_dict.
        # Same STATE+ENV split is used by both.
        stats_path = args.dataset / "meta" / "stats.json"
        raw_stats = json.loads(stats_path.read_text())
        n_joints = 7
        proprio_dim = 2 * n_joints

        def split_state_stat(s, idx_slice):
            return {k: torch.tensor(v[idx_slice], dtype=torch.float32) for k, v in s.items()}

        state_stat = raw_stats["observation.state"]
        action_stat = raw_stats["action"]
        dataset_stats = {
            "observation.state": split_state_stat(state_stat, slice(0, proprio_dim)),
            "observation.environment_state": split_state_stat(state_stat, slice(proprio_dim, None)),
            "action": {k: torch.tensor(v, dtype=torch.float32) for k, v in action_stat.items()},
        }
        if args.policy == "act":
            policy = build_act_policy(
                state_dim=ds.state_dim, action_dim=ds.action_dim, chunk_size=chunk,
                dataset_stats=dataset_stats,
            ).to(device)
        else:  # diffusion
            policy = build_diffusion_policy(
                state_dim=ds.state_dim, action_dim=ds.action_dim, horizon=chunk,
                dataset_stats=dataset_stats,
            ).to(device)

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * max(1, len(train_loader))
    )

    print(f"[policy] {args.policy.upper()} on {device}, {sum(p.numel() for p in policy.parameters()):,} params")
    print(f"[train] {n_train} train / {n_val} val, {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")

    log_path = args.output / "train_log.jsonl"
    log_fh = log_path.open("w")

    best_val = float("inf")
    t_start = time.time()

    n_joints = 7
    proprio_dim = 2 * n_joints

    def _adapt_batch_for_policy(batch: dict) -> dict:
        """Split observation.state into STATE (proprio) + ENV (EE pose) and
        add the action_is_pad mask. Diffusion also needs a time dim on
        observations (n_obs_steps=1 → unsqueeze to [B, 1, dim])."""
        if args.policy == "bc":
            return batch
        state = batch["observation.state"]                # [B, 21]
        action = batch["action"]                          # [B, chunk, 7]
        b = {k: v for k, v in batch.items() if k != "observation.state"}
        b["observation.state"] = state[..., :proprio_dim]
        b["observation.environment_state"] = state[..., proprio_dim:]
        if args.policy == "diffusion":
            # Diffusion's policy.forward expects [B, n_obs_steps, dim];
            # n_obs_steps=1 here so just add a singleton time dim.
            b["observation.state"] = b["observation.state"].unsqueeze(1)
            b["observation.environment_state"] = b["observation.environment_state"].unsqueeze(1)
        # All chunk frames are real (we drop the tail in the DataLoader).
        b["action_is_pad"] = torch.zeros(
            action.shape[:2], dtype=torch.bool, device=action.device,
        )
        return b

    def _forward_loss(policy, batch):
        """Run policy(batch) and return the loss tensor. Handles BC dict output
        and ACT (loss, extras) tuple output uniformly."""
        batch = _adapt_batch_for_policy(batch)
        out = policy(batch)
        if isinstance(out, tuple):
            return out[0]
        return out["loss"]

    for epoch in range(args.epochs):
        policy.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = _forward_loss(policy, batch)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu()))

        # ACT's forward() relies on the VAE encoder which is bypassed in
        # eval() mode, so its loss is undefined there. Same trick works for
        # Diffusion (its training loss is the denoising MSE which is the
        # same in train and eval). Keep them in train mode with grad off.
        if args.policy in ("act", "diffusion"):
            policy.train()
        else:
            policy.eval()
        with torch.no_grad():
            val_losses = []
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                val_losses.append(float(_forward_loss(policy, batch).cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        log_fh.write(json.dumps({
            "type": "epoch", "epoch": epoch + 1,
            "train_loss": train_loss, "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": time.time() - t_start,
        }) + "\n")
        log_fh.flush()
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch+1:4d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  elapsed={time.time()-t_start:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict": policy.state_dict(),
                "policy_type": args.policy,
                "val_loss": val_loss,
                "config": {
                    "state_dim": ds.state_dim,
                    "action_dim": ds.action_dim,
                    "dataset_path": str(args.dataset),
                    "hidden": args.hidden,
                    "chunk_size": chunk,
                    "horizon": chunk if args.policy == "diffusion" else None,
                },
            }, args.output / "best.pt")

    log_fh.close()
    elapsed = time.time() - t_start
    print(f"\n[done] {elapsed:.1f}s  best val loss: {best_val:.4f}")
    print(f"[checkpoint] {args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
