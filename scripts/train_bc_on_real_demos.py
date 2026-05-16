"""
Train BC on real demonstrations collected through the full ROS 2 + PyBullet
pipeline. The dataset is at /tmp/mybotshop_demos/panda_reach_v1.

Output:
    runs/panda_reach_v1_bc/
        best.pt
        train_log.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from il_pipeline.training.lerobot_torch_dataset import LeRobotTorchDataset  # noqa: E402
from il_pipeline.training.train import BCPolicy  # noqa: E402


DATASET_ROOT = Path("/tmp/mybotshop_demos/panda_reach_v1")
OUTPUT_DIR = REPO_ROOT / "runs" / "panda_reach_v1_bc"
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
SEED = 42


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds = LeRobotTorchDataset(DATASET_ROOT, chunk_size=1)
    print(f"loaded dataset: {len(ds)} samples  state_dim={ds.state_dim}  action_dim={ds.action_dim}")

    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    policy = BCPolicy(state_dim=ds.state_dim, action_dim=ds.action_dim, hidden=256)
    optim = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=EPOCHS * len(train_loader)
    )

    print(f"training BC: {n_train} train / {n_val} val, {EPOCHS} epochs, batch_size={BATCH_SIZE}")
    best_val = float("inf")
    t_start = time.time()
    log_path = OUTPUT_DIR / "train_log.jsonl"
    log_fh = log_path.open("w")

    for epoch in range(EPOCHS):
        policy.train()
        train_losses = []
        for batch in train_loader:
            out = policy(batch)
            optim.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            train_losses.append(out["loss"].item())

        policy.eval()
        with torch.inference_mode():
            val_losses = [policy(b)["loss"].item() for b in val_loader]

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        log_fh.write(json.dumps({
            "type": "epoch",
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": time.time() - t_start,
        }) + "\n")
        log_fh.flush()
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch+1:3d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict": policy.state_dict(),
                "policy_type": "bc",
                "val_loss": val_loss,
                "config": {
                    "state_dim": ds.state_dim,
                    "action_dim": ds.action_dim,
                    "dataset_path": str(DATASET_ROOT),
                    "hidden": 256,
                },
            }, OUTPUT_DIR / "best.pt")

    log_fh.close()
    elapsed = time.time() - t_start
    print(f"\ndone in {elapsed:.1f}s  best val loss: {best_val:.4f}")
    print(f"checkpoint: {OUTPUT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
