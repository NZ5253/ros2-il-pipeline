# Web API

REST + WebSocket surface that the MyBotShop webserver UI calls. Versioned under `/api/v1/`. Lives alongside the platform's existing WebSocket interface, not on top of it. Full OpenAPI is auto-generated at `http://localhost:8011/docs`.

## Endpoints

### Datasets

| Method | Path | Action |
|---|---|---|
| `GET`    | `/datasets`              | List datasets |
| `POST`   | `/datasets`              | Create empty dataset |
| `GET`    | `/datasets/{id}`         | Metadata + episode summary |
| `DELETE` | `/datasets/{id}`         | Delete (idempotent) |
| `POST`   | `/datasets/{id}/record/start` | Begin recording — dispatches `StartEpisode.srv` to data_logger |
| `POST`   | `/datasets/{id}/record/stop`  | End recording — dispatches `StopEpisode.srv`, returns frame count + parquet path |

### Training and Policies

| Method | Path | Action |
|---|---|---|
| `POST`   | `/training/jobs`         | Start training job (BC, ACT, or Diffusion) |
| `GET`    | `/training/jobs/{id}`    | Job status + latest metrics |
| `DELETE` | `/training/jobs/{id}`    | Cancel job |
| `GET`    | `/policies`              | List trained policy checkpoints |
| `POST`   | `/policies/{id}/deploy`  | `LoadPolicy.srv` into inference_node |
| `POST`   | `/policies/{id}/start`   | Begin inference at the configured rate |
| `POST`   | `/policies/{id}/stop`    | Stop inference |

### WebSocket channels

| Path | Payload |
|---|---|
| `/ws/training/{job_id}/progress` | per-epoch `{epoch, train_loss, val_loss, lr, elapsed_s}` |
| `/ws/inference/live`             | per-step `{step, action, latency_ms}` + occasional observation |

## Example: record an episode end-to-end

```bash
# 1. Create dataset
curl -X POST http://localhost:8011/api/v1/datasets \
     -H 'Content-Type: application/json' \
     -d '{"name":"panda_pickplace_v2","robot_model":"franka_panda",
          "task_description":"pick red cube to green target"}'
# -> {"id":"ds-3f1a-...","name":"panda_pickplace_v2",...}

# 2. Start recording an episode
curl -X POST http://localhost:8011/api/v1/datasets/ds-3f1a-.../record/start \
     -H 'Content-Type: application/json' \
     -d '{"episode_name":"demo_001","task_description":"pick-and-place"}'

# 3. ...teleoperate the robot through the existing UI...

# 4. Stop and mark outcome
curl -X POST http://localhost:8011/api/v1/datasets/ds-3f1a-.../record/stop \
     -H 'Content-Type: application/json' \
     -d '{"outcome":"success"}'
# -> {"frame_count":523,"saved_to":"data/chunk-000/episode_000000.parquet"}
```

## Example: deploy a trained policy

```bash
curl -X POST http://localhost:8011/api/v1/policies/p-9c0b-.../deploy
curl -X POST http://localhost:8011/api/v1/policies/p-9c0b-.../start
# -> /inference_node now publishes /cmd_robot at the configured rate
```

## Notes

- Errors follow `{"error":{"code":"...","message":"...","details":{...}}}`.
- All bodies and responses are JSON; WebSocket frames are JSON-encoded objects.
- The FastAPI service has a `dry_run` mode (no `rclpy`) that returns stub responses, so the service boots and the OpenAPI page is reachable even on machines without a live ROS 2 graph.
- Authentication is inherited from the host platform; the IL surface adds no separate auth layer.
