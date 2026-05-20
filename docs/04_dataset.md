# Dataset Schema

On-disk format is **LeRobotDataset** (parquet shards, HuggingFace `datasets`-compatible). Picked over a custom format because ACT, Diffusion Policy, and any future VLA consume it directly without conversion; visualisation and HF Hub push come for free.

## Layout

```
panda_pickplace_v2/
├── meta/
│   ├── info.json         dataset-level metadata + feature spec
│   ├── episodes.jsonl    one line per episode (length, task)
│   ├── stats.json        per-feature mean/std/min/max for normalisation
│   └── tasks.jsonl       task strings referenced by episodes
└── data/
    └── chunk-000/
        ├── episode_000000.parquet
        ├── episode_000001.parquet
        └── ...
```

Episodes are sharded into chunks of 1000 per directory.

## Per-frame fields

| Field | Dtype | Shape | Notes |
|---|---|---|---|
| `observation.state` | float32 | `[24]` | See breakdown below |
| `action`            | float32 | `[7]`  | Delta-EE Twist (linear xyz + angular xyz + gripper) |
| `episode_index`     | int64   | scalar | Which episode this frame belongs to |
| `frame_index`       | int64   | scalar | Frame index within the episode |
| `timestamp`         | float32 | scalar | Seconds from episode start |
| `next.reward`       | float32 | scalar | 0.0 in IL; populated only when used for RL fine-tuning |
| `next.done`         | bool    | scalar | True on the last frame of an episode |
| `task_index`        | int64   | scalar | Index into `tasks.jsonl` |
| `observation.images.*` | uint8 video | `[H,W,3]` | Optional — present iff cameras were enabled at collection |

## `observation.state` breakdown

Concatenated in a fixed, documented order:

```
state[0 : 7]    joint positions     (radians)
state[7 : 14]   joint velocities    (rad/s)
state[14 : 17]  ee position xyz     (metres, world frame)
state[17 : 21]  ee orientation quat (xyzw, normalised)
state[21 : 24]  task object xyz     (metres, world frame)
```

The order is recorded in `info.json` under `features.observation.state.names` so consumers can verify before reading.

Including the task object's pose follows the Robomimic and ALOHA conventions — random-spawn manipulation is otherwise under-specified at the supervised learning level. The 3-D xyz is sufficient for a roughly-symmetric task object like a cube; for asymmetric objects, the full pose (xyz + quat) would be the natural extension.

## Stats and normalisation

`stats.json` is computed by `LeRobotShardWriter.finalise()` after collection. Mean/std per feature dimension feed the dataset-level normaliser (training) and the inference-time `Normaliser` class. Both clamp `std` at a 1e-6 floor for action dimensions that are constant across all demonstrations (e.g. the gripper command when grasping is constraint-based).
