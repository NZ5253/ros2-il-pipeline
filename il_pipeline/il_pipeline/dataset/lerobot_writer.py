"""
LeRobotDataset shard writer.

Writes episodes as parquet shards following the LeRobotDataset on-disk layout
described in docs/04_dataset_schema.md. Designed for write-only access during
recording — reading is done by the training pipeline via the `datasets`
library directly.

This is a minimal implementation that produces a dataset compatible with the
LeRobot library's reader. On the lab PC, the actual LeRobot library can be
used directly to avoid maintaining a parallel writer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class DatasetInfo:
    codebase_version: str = "v2.1"
    robot_type: str = "generic"
    total_episodes: int = 0
    total_frames: int = 0
    total_tasks: int = 0
    chunks_size: int = 1000
    fps: int = 30
    splits: dict = field(default_factory=lambda: {"train": "0:1"})
    data_path: str = (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    video_path: str = (
        "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
    )
    features: dict = field(default_factory=dict)


class LeRobotShardWriter:
    """Append-only writer of LeRobot-format episodes."""

    def __init__(self, root: Path, dataset_name: str) -> None:
        self.dataset_root = root / dataset_name
        self.dataset_name = dataset_name
        self._meta_dir = self.dataset_root / "meta"
        self._data_dir = self.dataset_root / "data"
        self._dataset_root_init()

        info_path = self._meta_dir / "info.json"
        if info_path.exists():
            self._info = DatasetInfo(**json.loads(info_path.read_text()))
        else:
            self._info = DatasetInfo()
            self._save_info()

    def _dataset_root_init(self) -> None:
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _save_info(self) -> None:
        (self._meta_dir / "info.json").write_text(
            json.dumps(asdict(self._info), indent=2)
        )

    # ── Episode index management ──────────────────────────────────────────

    def next_episode_index(self) -> int:
        return self._info.total_episodes

    def _episode_chunk(self, episode_index: int) -> int:
        return episode_index // self._info.chunks_size

    def _episode_path(self, episode_index: int) -> Path:
        chunk = self._episode_chunk(episode_index)
        chunk_dir = self._data_dir / f"chunk-{chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        return chunk_dir / f"episode_{episode_index:06d}.parquet"

    # ── Write an episode ──────────────────────────────────────────────────

    def write_episode(self, episode_id: str, frames: list[dict]) -> Path:
        if not frames:
            raise ValueError("cannot write an episode with zero frames")

        episode_index = self.next_episode_index()
        for f in frames:
            f["episode_index"] = episode_index

        # Convert frames (list of dicts) to a pyarrow Table.
        # Keys are flattened LeRobot feature names. Tensors become lists.
        columns: dict[str, list] = {}
        for f in frames:
            for key, value in f.items():
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                columns.setdefault(key, []).append(value)

        table = pa.table(columns)
        path = self._episode_path(episode_index)
        pq.write_table(table, path, compression="snappy")

        # Append a record to episodes.jsonl
        episodes_path = self._meta_dir / "episodes.jsonl"
        with episodes_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "episode_index": episode_index,
                        "tasks": ["default"],
                        "length": len(frames),
                    }
                )
                + "\n"
            )

        # Update top-level info
        self._info.total_episodes += 1
        self._info.total_frames += len(frames)
        self._save_info()

        return path

    # ── Finalisation: compute and persist normalisation stats ─────────────

    def finalise(self, stats_features: Iterable[str]) -> None:
        """
        Compute per-feature mean/std/min/max across all written episodes.

        Called once after all demonstrations are collected. The training
        pipeline reads stats.json to normalise inputs.
        """
        stats: dict[str, dict] = {}
        for feature in stats_features:
            values = []
            for chunk_dir in sorted(self._data_dir.iterdir()):
                for parquet in sorted(chunk_dir.glob("episode_*.parquet")):
                    table = pq.read_table(parquet, columns=[feature])
                    arr = np.array(table.column(feature).to_pylist(), dtype=np.float32)
                    values.append(arr)
            if not values:
                continue
            stacked = np.concatenate(values, axis=0)
            stats[feature] = {
                "mean": stacked.mean(axis=0).tolist(),
                "std": (stacked.std(axis=0) + 1e-8).tolist(),
                "min": stacked.min(axis=0).tolist(),
                "max": stacked.max(axis=0).tolist(),
            }
        (self._meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))
