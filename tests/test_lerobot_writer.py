"""Tests for the LeRobot dataset shard writer."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from il_pipeline.dataset.lerobot_writer import LeRobotShardWriter


@pytest.fixture
def writer(tmp_path: Path) -> LeRobotShardWriter:
    return LeRobotShardWriter(root=tmp_path, dataset_name="test_ds")


def _frame(idx: int, n_joints: int = 7, action_dim: int = 7) -> dict:
    state = np.zeros(2 * n_joints + 7, dtype=np.float32)
    state[0] = float(idx)   # vary first joint to make stats non-degenerate
    return {
        "observation.state": state,
        "action": np.full(action_dim, float(idx), dtype=np.float32),
        "timestamp": float(idx) * 0.033,
        "frame_index": idx,
        "next.reward": 0.0,
        "next.done": False,
    }


def test_writes_first_episode(writer: LeRobotShardWriter):
    frames = [_frame(i) for i in range(10)]
    frames[-1]["next.done"] = True

    path = writer.write_episode("ep-001", frames)
    assert path.exists()

    table = pq.read_table(path)
    assert len(table) == 10
    assert "observation.state" in table.column_names
    assert "action" in table.column_names


def test_info_json_updated_after_write(writer: LeRobotShardWriter):
    writer.write_episode("ep-001", [_frame(i) for i in range(5)])
    writer.write_episode("ep-002", [_frame(i) for i in range(8)])

    info = json.loads((writer.dataset_root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 13


def test_episodes_jsonl_grows(writer: LeRobotShardWriter):
    writer.write_episode("ep-001", [_frame(i) for i in range(3)])
    writer.write_episode("ep-002", [_frame(i) for i in range(4)])

    lines = (writer.dataset_root / "meta" / "episodes.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["length"] == 3
    assert records[1]["length"] == 4
    assert records[0]["episode_index"] == 0
    assert records[1]["episode_index"] == 1


def test_rejects_empty_episode(writer: LeRobotShardWriter):
    with pytest.raises(ValueError):
        writer.write_episode("ep-empty", [])


def test_episodes_sharded_by_chunk(writer: LeRobotShardWriter):
    """Two episodes in the same chunk should land in the same chunk directory."""
    writer.write_episode("ep-a", [_frame(i) for i in range(2)])
    writer.write_episode("ep-b", [_frame(i) for i in range(2)])

    chunk_dir = writer.dataset_root / "data" / "chunk-000"
    parquets = sorted(chunk_dir.glob("episode_*.parquet"))
    assert len(parquets) == 2
    assert parquets[0].name == "episode_000000.parquet"
    assert parquets[1].name == "episode_000001.parquet"


def test_finalise_computes_stats(writer: LeRobotShardWriter):
    writer.write_episode("ep-001", [_frame(i) for i in range(10)])
    writer.finalise(["observation.state", "action"])

    stats = json.loads((writer.dataset_root / "meta" / "stats.json").read_text())
    assert "observation.state" in stats
    assert "action" in stats
    # First joint position should have non-trivial std because we varied it
    state_std = stats["observation.state"]["std"]
    assert state_std[0] > 0.0
