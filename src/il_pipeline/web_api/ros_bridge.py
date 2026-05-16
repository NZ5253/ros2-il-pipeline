"""
ROS 2 bridge for the FastAPI service.

Runs an rclpy spin loop in a background thread and exposes thin async
helpers that HTTP request handlers can call: service invocation, parameter
reads, and a tail of the training log for WebSocket streaming.

Why a bridge module rather than calling rclpy from inside FastAPI handlers:
rclpy's executor model and asyncio do not naturally coexist. Isolating ROS
into one place keeps the HTTP layer simple and the lifecycles independent.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional


@dataclass
class ServiceResult:
    """Lightweight result shape used by call_service_async."""

    success: bool
    message: str
    raw: object = None   # the full response object; consumers can read typed fields


class RosBridge:
    """
    Background thread running an rclpy executor.

    On a host with rclpy installed, this spins up an rclpy node that issues
    service calls on behalf of HTTP handlers. On a host without rclpy (e.g.,
    during pure-Python development), the bridge operates in `dry_run` mode
    and returns stub responses so the FastAPI app can still start.
    """

    def __init__(self, dry_run: Optional[bool] = None) -> None:
        self._stop_event = threading.Event()
        self._executor = None
        self._node = None
        self._dry_run = dry_run if dry_run is not None else not self._rclpy_available()

        if not self._dry_run:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.node import Node

            rclpy.init()
            self._node = Node("il_pipeline_web_bridge")
            self._executor = MultiThreadedExecutor()
            self._executor.add_node(self._node)
            self._service_clients: dict = {}

    @staticmethod
    def _rclpy_available() -> bool:
        try:
            import rclpy  # noqa: F401
            return True
        except ImportError:
            return False

    def spin(self) -> None:
        """Run the executor until shutdown is requested."""
        if self._dry_run:
            while not self._stop_event.is_set():
                time.sleep(0.1)
            return

        try:
            while not self._stop_event.is_set():
                self._executor.spin_once(timeout_sec=0.1)
        finally:
            import rclpy
            self._node.destroy_node()
            rclpy.shutdown()

    def shutdown(self) -> None:
        self._stop_event.set()

    # ── Service invocation ───────────────────────────────────────────────

    async def call_service_async(
        self,
        service_name: str,
        srv_type=None,
        request=None,
        timeout_s: float = 5.0,
    ) -> ServiceResult:
        """
        Issue a service call asynchronously.

        If `srv_type` and `request` are not provided, defaults to std_srvs/Trigger
        (common case for ad-hoc lifecycle endpoints). The handler in the FastAPI
        app awaits this; the actual rclpy call runs in the executor thread.
        """
        if self._dry_run:
            return ServiceResult(
                success=True,
                message=f"dry-run: {service_name}",
                raw=None,
            )

        if srv_type is None:
            from std_srvs.srv import Trigger
            srv_type = Trigger
            request = Trigger.Request()

        client_key = f"{service_name}::{srv_type.__module__}.{srv_type.__name__}"
        client = self._service_clients.get(client_key)
        if client is None:
            client = self._node.create_client(srv_type, service_name)
            self._service_clients[client_key] = client

        if not client.wait_for_service(timeout_sec=timeout_s):
            return ServiceResult(
                success=False,
                message=f"service {service_name} unavailable after {timeout_s}s",
                raw=None,
            )

        future = client.call_async(request)
        result = await self._await_future(future, timeout_s)
        if result is None:
            return ServiceResult(success=False, message="service call timed out", raw=None)

        success = bool(getattr(result, "success", False))
        message = getattr(result, "message", "") or ""
        return ServiceResult(success=success, message=message, raw=result)

    @staticmethod
    async def _await_future(future, timeout_s: float):
        """Poll an rclpy future from asyncio without blocking the event loop."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if future.done():
                return future.result()
            await asyncio.sleep(0.01)
        return None

    # ── Training log streaming ───────────────────────────────────────────

    async def tail_training_log(self, job_id: str) -> AsyncIterator[dict]:
        """Yield JSONL records from a training log as they are written."""
        log_path = Path(f"/tmp/training/{job_id}/train_log.jsonl")
        for _ in range(100):
            if log_path.exists():
                break
            await asyncio.sleep(0.1)
        if not log_path.exists():
            yield {"type": "error", "message": f"no log for job {job_id}"}
            return

        position = 0
        while True:
            with log_path.open() as fh:
                fh.seek(position)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
                position = fh.tell()
            await asyncio.sleep(0.5)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def component_health(self) -> dict:
        if self._dry_run:
            return {"mode": "dry-run", "ros2": "unavailable"}
        return {
            "mode": "live",
            "ros2": "connected",
            "node": self._node.get_name(),
        }
