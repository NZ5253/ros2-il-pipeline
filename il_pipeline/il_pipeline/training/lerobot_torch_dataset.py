"""
PyTorch Dataset adapter for LeRobotDataset parquet shards.

On the lab PC, the lerobot library provides this with caching, action
chunking, and image decoding built in. This is a minimal stand-in for
the BC baseline and for development without the full lerobot install.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class LeRobotTorchDataset(Dataset):
    """
    PyTorch Dataset over LeRobotDataset parquet shards.

    Each item is a single transition (BC) or an observation paired with a
    chunk of `chunk_size` future actions (ACT).
    """

    def __init__(
        self,
        dataset_root: Path,
        chunk_size: int = 1,
        stats_path: Optional[Path] = None,
    ) -> None:
        self.root = Path(dataset_root)
        self.chunk_size = chunk_size

        info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = info.get("fps", 30)

        # Load all episodes into memory as tensors. For larger datasets, this
        # would be replaced with lazy parquet reads via the datasets library.
        self._episodes: list[dict] = []
        data_dir = self.root / "data"
        for chunk_dir in sorted(data_dir.iterdir()):
            for parquet in sorted(chunk_dir.glob("episode_*.parquet")):
                table = pq.read_table(parquet)
                ep = {
                    "observation.state": np.array(
                        table.column("observation.state").to_pylist(), dtype=np.float32
                    ),
                    "action": np.array(
                        table.column("action").to_pylist(), dtype=np.float32
                    ),
                }
                self._episodes.append(ep)

        if not self._episodes:
            raise ValueError(f"no episodes found in {self.root}")

        self.state_dim = self._episodes[0]["observation.state"].shape[1]
        self.action_dim = self._episodes[0]["action"].shape[1]

        # Build a flat index of (episode_idx, frame_idx) tuples, excluding
        # frames near the end where a chunk would overrun the episode.
        self._index: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self._episodes):
            n_frames = len(ep["action"])
            for f_idx in range(n_frames - self.chunk_size + 1):
                self._index.append((ep_idx, f_idx))

        # Optional normalisation
        self._stats = None
        stats_path = stats_path or (self.root / "meta" / "stats.json")
        if stats_path.exists():
            self._stats = json.loads(stats_path.read_text())

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, f_idx = self._index[idx]
        ep = self._episodes[ep_idx]

        obs = ep["observation.state"][f_idx]
        actions = ep["action"][f_idx : f_idx + self.chunk_size]
        if self.chunk_size == 1:
            actions = actions[0]  # squeeze for BC

        if self._stats is not None:
            obs = (obs - self._stats["observation.state"]["mean"]) / self._stats["observation.state"]["std"]
            actions = (actions - self._stats["action"]["mean"]) / self._stats["action"]["std"]

        return {
            "observation.state": torch.from_numpy(np.asarray(obs, dtype=np.float32)),
            "action": torch.from_numpy(np.asarray(actions, dtype=np.float32)),
        }
