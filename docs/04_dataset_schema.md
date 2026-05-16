# Dataset Schema Specification

The pipeline uses **LeRobotDataset** as its on-disk format. This document specifies how robot data, recorded via the data logger node, maps onto that format, and the conventions chosen where LeRobot is flexible.

---

## Why LeRobotDataset

A short justification before the schema:

- **Parquet-backed** — columnar, shardable per episode, efficient for partial reads
- **Stable schema** — the HuggingFace `datasets` library is the reader, with a known compatibility contract
- **Built-in visualisation** — `lerobot.scripts.visualize_dataset` works on any LeRobotDataset out of the box
- **Hub-shareable** — datasets can be pushed to HuggingFace Hub without conversion
- **Community alignment** — ACT, Diffusion Policy, OpenVLA, π₀ all consume this format directly

Inventing a custom format would duplicate work and cut the pipeline off from this ecosystem.

---

## On-Disk Layout

```
datasets/
└── ds-3f1a-9c0b-.../
    ├── meta/
    │   ├── info.json              dataset-level metadata
    │   ├── episodes.jsonl         one line per episode (id, length, task)
    │   ├── stats.json             per-feature mean/std/min/max (for normalisation)
    │   └── tasks.jsonl            task descriptions referenced by episodes
    ├── data/
    │   ├── chunk-000/
    │   │   ├── episode_000000.parquet
    │   │   ├── episode_000001.parquet
    │   │   └── ...
    │   └── chunk-001/
    │       └── ...
    └── videos/                     optional, if cameras are configured
        ├── observation.images.wrist_cam/
        │   ├── chunk-000/
        │   │   ├── episode_000000.mp4
        │   │   └── ...
        └── observation.images.scene_cam/
            └── ...
```

Episodes are sharded into chunks of (default) 1000 episodes per directory to keep filesystem operations cheap.

---

## `info.json` — Dataset-Level Metadata

```json
{
  "codebase_version": "v2.1",
  "robot_type": "franka_panda",
  "total_episodes": 42,
  "total_frames": 18934,
  "total_tasks": 1,
  "total_videos": 84,
  "chunks_size": 1000,
  "fps": 30,
  "splits": { "train": "0:38", "val": "38:42" },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
  "features": { "...": "see below" }
}
```

---

## `features` — Per-Frame Schema

Each frame in a parquet shard has these fields:

| Field | Dtype | Shape | Description |
|---|---|---|---|
| `observation.state` | float32 | `[state_dim]` | Concatenated joint positions, joint velocities, EE pose (xyzquat) |
| `observation.images.wrist_cam` | uint8 video | `[H, W, 3]` | Wrist camera RGB (optional) |
| `observation.images.scene_cam` | uint8 video | `[H, W, 3]` | Scene camera RGB (optional) |
| `action` | float32 | `[action_dim]` | Commanded action at this step |
| `episode_index` | int64 | scalar | Which episode this frame belongs to |
| `frame_index` | int64 | scalar | Frame index within the episode |
| `timestamp` | float32 | scalar | Seconds from episode start |
| `next.reward` | float32 | scalar | Optional, populated for RL training (0.0 by default for IL) |
| `next.done` | bool | scalar | True on the last frame of an episode |
| `task_index` | int64 | scalar | Index into `tasks.jsonl` |

`state_dim` and `action_dim` depend on the robot. For Franka Panda with delta-joint actions: `state_dim = 7 + 7 + 7 = 21` (joint pos + joint vel + EE xyzrpy), `action_dim = 7`.

Images are stored as video files referenced by the video_path template, not inline in parquet, for efficient storage and decoding.

---

## State Vector Conventions

`observation.state` is a concatenation in a documented, fixed order:

```
state[0:N_joints]            joint_positions     (radians)
state[N_joints:2*N_joints]   joint_velocities    (rad/s)
state[2*N_joints:2*N_joints+3]   ee_position_xyz     (metres)
state[2*N_joints+3:2*N_joints+7] ee_orientation_quat (xyzw, normalised)
```

The order is recorded in `info.json` under `features.observation.state.names` so consumers can verify before consuming:

```json
"observation.state": {
  "dtype": "float32",
  "shape": [21],
  "names": [
    "panda_joint1.pos", "panda_joint2.pos", "...", "panda_joint7.pos",
    "panda_joint1.vel", "...", "panda_joint7.vel",
    "ee.x", "ee.y", "ee.z",
    "ee.qx", "ee.qy", "ee.qz", "ee.qw"
  ]
}
```

This makes the dataset self-describing — a downstream policy can verify the state contract instead of assuming.

---

## Action Conventions

Two action modes are supported and recorded in `info.json`:

### `delta_joint` (default)

`action[i]` is the commanded change in joint `i` for the next step, in radians.

```json
"action": {
  "dtype": "float32",
  "shape": [7],
  "names": ["panda_joint1.delta", "...", "panda_joint7.delta"]
}
```

### `delta_ee`

`action` is a 6-DOF end-effector delta plus optional gripper command.

```json
"action": {
  "dtype": "float32",
  "shape": [7],
  "names": ["ee.dx", "ee.dy", "ee.dz", "ee.drx", "ee.dry", "ee.drz", "gripper"]
}
```

The choice is per-dataset, fixed at creation time, and stored in metadata. Mixing modes within a dataset is not supported.

---

## Normalisation Statistics — `stats.json`

After all episodes are collected, the data logger computes per-feature statistics:

```json
{
  "observation.state": {
    "mean": [0.0, 0.0, ...],
    "std": [1.5, 0.8, ...],
    "min": [-2.8, -1.7, ...],
    "max": [2.8, 1.7, ...]
  },
  "action": {
    "mean": [...],
    "std": [...]
  }
}
```

These are used by the training pipeline to normalise inputs and de-normalise predictions. They are computed once at dataset finalisation, not on the fly.

---

## Episode Boundaries

Each episode is a single parquet file. `next.done` is `True` only on the final frame. This convention matters because:

- During training, `DataLoader` collates trajectories — knowing where episodes end prevents the model from learning spurious "wrap-around" transitions
- For RL fine-tuning, `next.done` is the standard signal for episode termination
- For action chunking (ACT), the dataloader needs to avoid sampling chunks that span episode boundaries

---

## Task Descriptions — `tasks.jsonl`

Each line is a JSON object:

```json
{ "task_index": 0, "task": "pick the red cube and place in the tray" }
```

`task_index` in each frame references this file. This structure supports future language-conditioned policies without schema migration.

---

## Validation at Recording Time

The data logger node validates each frame before appending to a shard:

| Check | What it catches |
|---|---|
| Joint state has exactly `N_joints` positions and velocities | Wrong robot connected, message corruption |
| EE pose quaternion is normalised (within 1e-3) | Bad TF lookups |
| Action vector dimension matches `action_spec.dim` | Wrong teleop bridge configuration |
| Image frame matches configured camera resolution | Camera driver misconfiguration |
| Timestamp is monotonically increasing within an episode | Clock skew, replay artefacts |
| Frame rate is within 10% of declared `fps` | Missed frames, system overload |

Failures are logged via ROS 2 `WARN` and either dropped (single-frame issues) or abort the episode (structural issues). A dataset with corrupt frames is worse than a smaller clean dataset.

---

## Conversion and Compatibility

If a future task needs a different format, conversion is one-way and scripted:

- **LeRobotDataset → ALOHA HDF5** — for tools that only support the original ALOHA format
- **LeRobotDataset → MCAP** — for replay in Foxglove or other ROS 2 tools
- **LeRobotDataset → HuggingFace Hub** — `dataset.push_to_hub()` works directly

LeRobot is positioned as the canonical format inside the pipeline; conversion happens only at export.
