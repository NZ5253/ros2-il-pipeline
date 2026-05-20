# API

REST + WebSocket surface the FastAPI service exposes. Versioned under `/api/v1/`. Full OpenAPI auto-generates at `http://localhost:8011/docs` once the service is running.

## Datasets

```
GET    /datasets                     list
POST   /datasets                     create empty
GET    /datasets/{id}                metadata + episode summary
DELETE /datasets/{id}                delete (idempotent)
POST   /datasets/{id}/record/start   dispatches StartEpisode.srv -> data_logger
POST   /datasets/{id}/record/stop    dispatches StopEpisode.srv,  returns frame_count + saved_to
```

## Training and policies

```
POST   /training/jobs                start a job (policy_type: bc | act | diffusion)
GET    /training/jobs/{id}           status + latest metrics
DELETE /training/jobs/{id}           cancel
GET    /policies                     list trained checkpoints
POST   /policies/{id}/deploy         LoadPolicy.srv -> inference_node
POST   /policies/{id}/start          begin inference at the configured rate
POST   /policies/{id}/stop           stop
```

## WebSocket

```
/ws/training/{job_id}/progress       per-epoch {epoch, train_loss, val_loss, lr, elapsed_s}
/ws/inference/live                   per-step  {step, action, latency_ms} + occasional observation
```

## Recording an episode

```bash
# create
curl -X POST http://localhost:8011/api/v1/datasets \
     -H 'Content-Type: application/json' \
     -d '{"name":"panda_pickplace_v2","robot_model":"franka_panda",
          "task_description":"pick red cube to green target"}'
# -> {"id":"ds-3f1a-...",...}

# start episode
curl -X POST http://localhost:8011/api/v1/datasets/ds-3f1a.../record/start \
     -H 'Content-Type: application/json' \
     -d '{"episode_name":"demo_001"}'

# ... teleoperate the robot ...

# stop, mark outcome
curl -X POST http://localhost:8011/api/v1/datasets/ds-3f1a.../record/stop \
     -H 'Content-Type: application/json' \
     -d '{"outcome":"success"}'
# -> {"frame_count":523,"saved_to":"data/chunk-000/episode_000000.parquet"}
```

## Deploying a policy

```bash
curl -X POST http://localhost:8011/api/v1/policies/p-9c0b.../deploy
curl -X POST http://localhost:8011/api/v1/policies/p-9c0b.../start
# inference_node now publishes /cmd_robot at the configured rate
```

## Notes

Errors follow `{"error":{"code":"...","message":"...","details":{...}}}`. JSON in / JSON out. Auth is whatever the host platform uses; the IL surface adds no separate layer.

If `rclpy` isn't on the host (e.g. running locally without a ROS install), the FastAPI service boots in dry-run mode and returns stub responses. Lets the OpenAPI page be reachable without a live ROS graph.
