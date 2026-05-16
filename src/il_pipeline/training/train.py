"""
Training entry point.

Loads a LeRobotDataset from disk, builds the chosen policy (BC, ACT, or
Diffusion Policy), and runs supervised training with checkpointing. Designed
to be invoked from the FastAPI training endpoint as a long-lived subprocess,
or from the command line for development.

On the lab PC, the LeRobot library itself provides this training loop with
better defaults. This skeleton is a minimal, dependency-light alternative for
the BC baseline and a wrapper for the ACT/Diffusion paths.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Project-local: in the prototype, the lerobot library can replace these.
from il_pipeline.training.lerobot_torch_dataset import LeRobotTorchDataset
from il_pipeline.training.policy_factory import build_policy


@dataclass
class TrainConfig:
    dataset_path: Path
    policy_type: str          # "bc" | "act" | "diffusion"
    output_dir: Path
    epochs: int = 2000
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    chunk_size: int = 50              # ACT only
    kl_weight: float = 10.0           # ACT only
    validation_split: float = 0.1
    checkpoint_every_steps: int = 5000
    device: str = "cuda:0"
    seed: int = 42


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)

    # Dataset + DataLoader
    full_ds = LeRobotTorchDataset(
        cfg.dataset_path,
        chunk_size=cfg.chunk_size if cfg.policy_type == "act" else 1,
    )
    n_val = int(len(full_ds) * cfg.validation_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Policy
    policy = build_policy(
        policy_type=cfg.policy_type,
        state_dim=full_ds.state_dim,
        action_dim=full_ds.action_dim,
        chunk_size=cfg.chunk_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs * len(train_loader),
    )

    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    log_path = cfg.output_dir / "train_log.jsonl"
    log_fh = log_path.open("a")

    for epoch in range(cfg.epochs):
        policy.train()
        epoch_losses = []
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output = policy(batch)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            global_step += 1

            if global_step % cfg.checkpoint_every_steps == 0:
                ckpt_path = cfg.output_dir / f"step_{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "policy_type": cfg.policy_type,
                        "state_dict": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": cfg.__dict__,
                    },
                    ckpt_path,
                )

        # Validation
        policy.eval()
        val_losses = []
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                output = policy(batch)
                val_losses.append(output["loss"].item())

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        log_record = {
            "type": "epoch",
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": time.time(),
        }
        log_fh.write(json.dumps(log_record) + "\n")
        log_fh.flush()
        print(f"epoch={epoch} step={global_step} train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "step": global_step,
                    "policy_type": cfg.policy_type,
                    "state_dict": policy.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg.__dict__,
                },
                cfg.output_dir / "best.pt",
            )

    log_fh.close()


# ── BC reference policy (minimal, self-contained) ────────────────────────


class BCPolicy(nn.Module):
    """Simple MLP from observation to action. Used as a sanity baseline."""

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
        # BC always predicts a single action; reshape to chunk format for
        # consistency with ACT/Diffusion at the inference node.
        a = self.net(observation)
        return a.unsqueeze(0)  # shape [1, action_dim]


# ── CLI entry point ──────────────────────────────────────────────────────


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--policy", choices=["bc", "act", "diffusion"], default="act")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    return TrainConfig(
        dataset_path=args.dataset,
        policy_type=args.policy,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        chunk_size=args.chunk_size,
        device=args.device,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
