"""Tests for the PyTorch adapter that reads LeRobot parquet shards."""

import json
from pathlib import Path

import numpy as np
import torch

from il_pipeline.dataset.lerobot_writer import LeRobotShardWriter
from il_pipeline.training.lerobot_torch_dataset import LeRobotTorchDataset


def _build_dataset(root: Path, n_episodes: int = 3, frames_per_ep: int = 12) -> Path:
    writer = LeRobotShardWriter(root=root, dataset_name="tiny_ds")
    for ep_idx in range(n_episodes):
        frames = []
        for f in range(frames_per_ep):
            state = np.zeros(21, dtype=np.float32)
            state[0] = float(ep_idx)
            state[1] = float(f) * 0.1
            # Last 4 of state are EE quaternion — leave at unit (0,0,0,1)
            state[-1] = 1.0
            frames.append({
                "observation.state": state,
                "action": np.full(7, float(ep_idx) + 0.01 * f, dtype=np.float32),
                "timestamp": f * 0.033,
                "frame_index": f,
                "next.reward": 0.0,
                "next.done": (f == frames_per_ep - 1),
            })
        writer.write_episode(f"ep-{ep_idx}", frames)
    writer.finalise(["observation.state", "action"])
    return writer.dataset_root


def test_bc_dataset_iterates(tmp_path: Path):
    root = _build_dataset(tmp_path, n_episodes=2, frames_per_ep=10)
    ds = LeRobotTorchDataset(root, chunk_size=1)
    assert len(ds) == 20

    sample = ds[0]
    assert sample["observation.state"].shape == (21,)
    assert sample["action"].shape == (7,)
    assert sample["observation.state"].dtype == torch.float32


def test_act_chunk_dataset_drops_tail(tmp_path: Path):
    root = _build_dataset(tmp_path, n_episodes=2, frames_per_ep=10)
    chunk_size = 4
    ds = LeRobotTorchDataset(root, chunk_size=chunk_size)
    # Each episode contributes (frames_per_ep - chunk_size + 1) starting indices
    assert len(ds) == 2 * (10 - chunk_size + 1)

    sample = ds[0]
    assert sample["action"].shape == (chunk_size, 7)


def test_normalisation_applied_when_stats_exist(tmp_path: Path):
    root = _build_dataset(tmp_path, n_episodes=2, frames_per_ep=10)
    ds = LeRobotTorchDataset(root, chunk_size=1)
    sample = ds[0]
    # With normalisation the mean across the dataset should be near zero
    states = torch.stack([ds[i]["observation.state"] for i in range(len(ds))])
    mean_per_dim = states.mean(dim=0).abs().max().item()
    assert mean_per_dim < 1e-3
