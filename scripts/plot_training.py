"""
Plot training curves from a `train_log.jsonl` file.

The training script writes one JSON line per epoch:
    {"type": "epoch", "epoch": N, "train_loss": ..., "val_loss": ..., "lr": ..., "elapsed_s": ...}

This script reads it and produces a PNG with train/val loss curves. Used to
include training-progress evidence in the deliverable doc and demo video.

Falls back to ASCII output if matplotlib is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_log(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "epoch":
            records.append(r)
    return records


def ascii_plot(records: list[dict]) -> str:
    """Cheap ASCII fallback when matplotlib isn't around."""
    if not records:
        return "(no records)"
    train = [r["train_loss"] for r in records]
    val = [r["val_loss"] for r in records]
    epochs = [r["epoch"] for r in records]
    out = []
    out.append(f"{'epoch':>6} {'train':>10} {'val':>10}")
    out.append("-" * 30)
    for e, t, v in zip(epochs, train, val, strict=False):
        out.append(f"{e:>6} {t:>10.4f} {v:>10.4f}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="path to train_log.jsonl")
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG path (default: same dir as log)")
    args = parser.parse_args()

    records = load_log(args.log)
    if not records:
        print(f"no epoch records in {args.log}")
        return

    print(ascii_plot(records))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available — skipping PNG render")
        return

    epochs = [r["epoch"] for r in records]
    train = [r["train_loss"] for r in records]
    val = [r["val_loss"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, label="train loss", color="#2c5f8a", linewidth=2)
    ax.plot(epochs, val, label="val loss", color="#cc6c2c", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(args.log.parent.name)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = args.out or args.log.with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"\nplot written to {out}")


if __name__ == "__main__":
    main()
