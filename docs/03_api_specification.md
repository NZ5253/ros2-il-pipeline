# API Specification — IL Pipeline Web Layer

REST + WebSocket API exposed by the FastAPI service that orchestrates the IL pipeline. Designed to live alongside the MyBotShop platform's existing WebSocket interface, not on top of it. Versioned under `/api/v1/`.

---

## Conventions

- All responses are JSON unless explicitly streaming.
- Errors follow:
  ```json
  { "error": { "code": "STRING_CODE", "message": "human-readable", "details": {} } }
  ```
- Timestamps are ISO 8601 with timezone.
- IDs are UUIDv4 strings.
- Authentication: out of scope for this evaluation; the production deployment would inherit the platform's auth.

---

## Datasets

### `GET /api/v1/datasets`

List all datasets known to the system.

**Response:**
```json
{
  "datasets": [
    {
      "id": "ds-3f1a-...",
      "name": "panda_pick_and_place_v1",
      "created_at": "2026-05-22T10:14:00+02:00",
      "robot_model": "franka_panda",
      "task_description": "pick the red cube and place in tray",
      "episode_count": 42,
      "total_frames": 18934,
      "size_bytes": 245678901
    }
  ]
}
```

### `POST /api/v1/datasets`

Create a new (empty) dataset.

**Body:**
```json
{
  "name": "panda_pick_and_place_v1",
  "robot_model": "franka_panda",
  "task_description": "pick the red cube and place in tray",
  "observation_spec": {
    "joint_names": ["panda_joint1", "..."],
    "cameras": ["wrist_cam", "scene_cam"],
    "include_ee_pose": true
  },
  "action_spec": {
    "type": "delta_joint",
    "dim": 7
  }
}
```

**Response:** `201 Created` with the dataset object.

### `GET /api/v1/datasets/{id}`

Fetch metadata for a specific dataset, including a per-episode summary.

### `DELETE /api/v1/datasets/{id}`

Delete a dataset and its on-disk parquet shards. Idempotent.

---

## Episode Recording

### `POST /api/v1/datasets/{id}/record/start`

Begin recording a new episode. The data logger node starts buffering frames.

**Body:**
```json
{
  "episode_name": "demo_042",
  "task_description": "optional task-specific override"
}
```

**Response:**
```json
{
  "episode_id": "ep-9c0b-...",
  "started_at": "2026-05-22T10:18:33+02:00"
}
```

### `POST /api/v1/datasets/{id}/record/stop`

End the current episode and flush buffered frames to a parquet shard.

**Body:**
```json
{ "outcome": "success" }   // or "discard"
```

**Response:**
```json
{
  "episode_id": "ep-9c0b-...",
  "frame_count": 423,
  "duration_s": 14.1,
  "outcome": "success",
  "saved_to": "datasets/ds-3f1a-.../episodes/ep-9c0b-...parquet"
}
```

---

## Training Jobs

### `POST /api/v1/training/jobs`

Start a training job. The job runs as a long-lived process; status is observable via REST poll or WebSocket stream.

**Body:**
```json
{
  "dataset_id": "ds-3f1a-...",
  "policy_type": "act",
  "config": {
    "epochs": 2000,
    "batch_size": 32,
    "lr": 1e-4,
    "chunk_size": 50,
    "kl_weight": 10.0,
    "validation_split": 0.1
  },
  "device": "cuda:0",
  "checkpoint_every_steps": 5000
}
```

**Response:** `202 Accepted`
```json
{
  "job_id": "job-7e2d-...",
  "status": "queued",
  "websocket_url": "/ws/training/job-7e2d-.../progress"
}
```

### `GET /api/v1/training/jobs`

List training jobs with current status.

### `GET /api/v1/training/jobs/{id}`

```json
{
  "job_id": "job-7e2d-...",
  "status": "running",
  "progress": {
    "current_epoch": 423,
    "total_epochs": 2000,
    "current_step": 13568,
    "train_loss": 0.0421,
    "val_loss": 0.0563,
    "elapsed_s": 1854
  },
  "checkpoints": [
    { "step": 5000, "val_loss": 0.0712, "path": "checkpoints/job-7e2d/step_5000.pt" },
    { "step": 10000, "val_loss": 0.0598, "path": "checkpoints/job-7e2d/step_10000.pt" }
  ]
}
```

### `DELETE /api/v1/training/jobs/{id}`

Cancel a running job. The most recent checkpoint is preserved.

### `WS /ws/training/{job_id}/progress`

WebSocket stream of training updates. One JSON message per epoch boundary:
```json
{
  "type": "epoch",
  "epoch": 423,
  "train_loss": 0.0421,
  "val_loss": 0.0563,
  "timestamp": "2026-05-22T10:48:11+02:00"
}
```

Plus periodic `checkpoint` messages and a terminal `done` or `error` message.

---

## Policies

### `GET /api/v1/policies`

```json
{
  "policies": [
    {
      "id": "pol-1a2b-...",
      "name": "panda_pickplace_act_v1",
      "policy_type": "act",
      "created_at": "2026-05-22T11:30:00+02:00",
      "trained_from_job": "job-7e2d-...",
      "dataset_id": "ds-3f1a-...",
      "val_loss": 0.0421,
      "checkpoint_path": "checkpoints/job-7e2d/best.pt"
    }
  ]
}
```

### `POST /api/v1/policies/{id}/deploy`

Load a policy into the inference node. Idempotent — calling twice with the same policy is a no-op.

**Body:**
```json
{
  "inference_rate_hz": 30,
  "execution_mode": "first_action" 
}
```

`execution_mode` controls how ACT's action chunks are unrolled:
- `"first_action"` — execute only the first action of each chunk, re-predict every step (more reactive)
- `"full_chunk"` — execute the whole chunk before re-predicting (faster, less reactive)
- `"temporal_ensemble"` — ACT's recommended mode, weighted blend of overlapping chunks

**Response:**
```json
{ "deployed": true, "policy_id": "pol-1a2b-...", "warm_up_ms": 184 }
```

### `POST /api/v1/policies/{id}/start`

Begin closed-loop inference. The inference node starts publishing on `/cmd_robot`.

### `POST /api/v1/policies/{id}/stop`

Stop inference. The robot remains at its last commanded state.

### `WS /ws/inference/live`

Real-time stream of inference activity, useful for the webserver UI to visualise the policy in action:
```json
{
  "type": "step",
  "timestamp": "...",
  "observation": { "joint_positions": [...], "ee_pose": [...] },
  "action": [...],
  "inference_latency_ms": 12.4
}
```

---

## Evaluation

### `POST /api/v1/eval/run`

Run a scripted evaluation: deploy a policy, execute N rollouts, report success rate.

**Body:**
```json
{
  "policy_id": "pol-1a2b-...",
  "n_rollouts": 50,
  "max_steps_per_rollout": 500,
  "seed": 42,
  "scene_config": "configs/eval_scenes/pickplace_default.yaml"
}
```

**Response:**
```json
{
  "eval_id": "eval-...",
  "policy_id": "pol-1a2b-...",
  "n_rollouts": 50,
  "successes": 41,
  "success_rate": 0.82,
  "mean_steps_to_success": 187.4,
  "rollouts_detail": "..."
}
```

---

## Health and Status

### `GET /api/v1/health`

```json
{
  "status": "ok",
  "components": {
    "ros_bridge": "connected",
    "data_logger_node": "alive",
    "training_service_node": "alive",
    "inference_node": "idle",
    "gpu": "cuda:0 available"
  },
  "version": "0.1.0"
}
```

---

## OpenAPI Schema

The implementation exposes the full OpenAPI 3.1 schema at `/api/v1/openapi.json` and an interactive Swagger UI at `/api/v1/docs`. FastAPI generates both automatically from the route definitions.
