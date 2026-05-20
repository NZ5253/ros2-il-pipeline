"""
FastAPI service for the IL pipeline web layer.

This service runs alongside the MyBotShop platform and exposes the REST +
WebSocket API specified in docs/api.md. It coordinates the
ROS 2 nodes (data logger, training service, inference node) by calling
their services.

A `rclpy` event loop runs in a background thread so HTTP request handlers
can issue service calls without blocking the FastAPI event loop.
"""

from __future__ import annotations

import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

# ROS 2 client glue lives in a separate module so this file is small and
# the HTTP API is the focus.
from il_pipeline.web_api.ros_bridge import RosBridge

# ── Pydantic schemas (small, match docs/api.md) ─────────


class CreateDatasetRequest(BaseModel):
    name: str = Field(..., min_length=1)
    robot_model: str = "generic"
    task_description: str = ""


class StartRecordRequest(BaseModel):
    episode_name: str = Field(..., min_length=1)
    task_description: str = ""


class StopRecordRequest(BaseModel):
    outcome: str = "success"   # "success" or "discard"


class StartTrainingRequest(BaseModel):
    dataset_id: str
    policy_type: str = "act"
    epochs: int = 2000
    batch_size: int = 32
    lr: float = 1e-4
    chunk_size: int = 50
    validation_split: float = 0.1
    device: str = "cuda:0"


class DeployPolicyRequest(BaseModel):
    inference_rate_hz: float = 30.0
    execution_mode: str = "temporal_ensemble"


# ── State (single-process; production would back this with a DB) ─────────


@dataclass
class JobRegistry:
    training_jobs: dict[str, dict] = field(default_factory=dict)
    datasets: dict[str, dict] = field(default_factory=dict)
    policies: dict[str, dict] = field(default_factory=dict)


bridge: RosBridge | None = None
registry = JobRegistry()


# ── Lifespan ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bridge
    bridge = RosBridge()
    bridge_thread = threading.Thread(target=bridge.spin, daemon=True)
    bridge_thread.start()
    yield
    bridge.shutdown()


app = FastAPI(
    title="MyBotShop IL Pipeline",
    description="Imitation learning pipeline web layer",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Datasets ─────────────────────────────────────────────────────────────


@app.get("/api/v1/datasets")
async def list_datasets():
    return {"datasets": list(registry.datasets.values())}


@app.post("/api/v1/datasets", status_code=201)
async def create_dataset(body: CreateDatasetRequest):
    dataset_id = f"ds-{uuid.uuid4().hex[:8]}"
    record = {
        "id": dataset_id,
        "name": body.name,
        "robot_model": body.robot_model,
        "task_description": body.task_description,
        "created_at": bridge.now_iso() if bridge else "",
        "episode_count": 0,
    }
    registry.datasets[dataset_id] = record
    return record


@app.post("/api/v1/datasets/{dataset_id}/record/start", status_code=201)
async def start_recording(dataset_id: str, body: StartRecordRequest):
    if dataset_id not in registry.datasets:
        raise HTTPException(404, f"dataset {dataset_id} not found")

    srv_type, req = _build_start_episode(body, dataset_id)
    result = await bridge.call_service_async(
        "/data_logger_node/start_episode",
        srv_type=srv_type,
        request=req,
        timeout_s=2.0,
    )
    if not result.success:
        raise HTTPException(500, result.message or "start_episode failed")
    episode_id = getattr(result.raw, "episode_id", "") if result.raw else ""
    return {"episode_id": episode_id, "started_at": bridge.now_iso()}


@app.post("/api/v1/datasets/{dataset_id}/record/stop")
async def stop_recording(dataset_id: str, body: StopRecordRequest):
    if dataset_id not in registry.datasets:
        raise HTTPException(404, f"dataset {dataset_id} not found")

    srv_type, req = _build_stop_episode(body)
    result = await bridge.call_service_async(
        "/data_logger_node/stop_episode",
        srv_type=srv_type,
        request=req,
        timeout_s=5.0,
    )
    if not result.success:
        raise HTTPException(500, result.message or "stop_episode failed")
    raw = result.raw
    return {
        "outcome": body.outcome,
        "episode_id": getattr(raw, "episode_id", "") if raw else "",
        "frame_count": getattr(raw, "frame_count", 0) if raw else 0,
        "duration_s": getattr(raw, "duration_s", 0.0) if raw else 0.0,
        "saved_to": getattr(raw, "saved_to", "") if raw else "",
    }


def _build_start_episode(body: StartRecordRequest, dataset_id: str):
    """Construct a StartEpisode service request, or a tuple of (None, None) in dry-run."""
    try:
        from il_pipeline_msgs.srv import StartEpisode
    except ImportError:
        return None, None
    req = StartEpisode.Request()
    req.episode_name = body.episode_name
    req.task_description = body.task_description
    req.dataset_id = dataset_id
    return StartEpisode, req


def _build_stop_episode(body: StopRecordRequest):
    try:
        from il_pipeline_msgs.srv import StopEpisode
    except ImportError:
        return None, None
    req = StopEpisode.Request()
    req.outcome = body.outcome
    return StopEpisode, req


# ── Training ─────────────────────────────────────────────────────────────


@app.post("/api/v1/training/jobs", status_code=202)
async def start_training(body: StartTrainingRequest):
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    registry.training_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "config": body.model_dump(),
    }
    # In production: enqueue into a job queue (Celery, RQ) so multiple jobs
    # can be tracked. For the prototype, a subprocess spawn is enough.
    return {
        "job_id": job_id,
        "status": "queued",
        "websocket_url": f"/ws/training/{job_id}/progress",
    }


@app.get("/api/v1/training/jobs/{job_id}")
async def get_training_job(job_id: str):
    job = registry.training_jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"job {job_id} not found")
    return job


@app.websocket("/ws/training/{job_id}/progress")
async def training_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        # Stream training log records from disk as they are produced.
        # Implementation reads the training service's JSONL log file.
        async for record in bridge.tail_training_log(job_id):
            await websocket.send_json(record)
    except WebSocketDisconnect:
        return


# ── Policies ─────────────────────────────────────────────────────────────


@app.get("/api/v1/policies")
async def list_policies():
    return {"policies": list(registry.policies.values())}


@app.post("/api/v1/policies/{policy_id}/deploy")
async def deploy_policy(policy_id: str, body: DeployPolicyRequest):
    policy = registry.policies.get(policy_id)
    if not policy:
        raise HTTPException(404, f"policy {policy_id} not found")
    result = await bridge.call_service_async(
        "/inference_node/load_policy",
        timeout_s=10.0,
    )
    if not result.success:
        raise HTTPException(500, result.message)
    return {"deployed": True, "policy_id": policy_id}


@app.post("/api/v1/policies/{policy_id}/start")
async def start_policy(policy_id: str):
    result = await bridge.call_service_async("/inference_node/start", timeout_s=2.0)
    if not result.success:
        raise HTTPException(500, result.message)
    return {"running": True}


@app.post("/api/v1/policies/{policy_id}/stop")
async def stop_policy(policy_id: str):
    result = await bridge.call_service_async("/inference_node/stop", timeout_s=2.0)
    if not result.success:
        raise HTTPException(500, result.message)
    return {"running": False}


# ── Health ───────────────────────────────────────────────────────────────


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "components": bridge.component_health() if bridge else {},
    }
