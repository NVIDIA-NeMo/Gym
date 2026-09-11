# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP control daemon for SSH-free remote Docker execution.

SSH may bootstrap this process, but every container lifecycle operation after
startup is performed through this authenticated HTTP API. The daemon registers
its dynamically selected port with Gym's head server and renews a TTL lease.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import socket
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


CONTAINER_PORTS = (5000, 9222, 8006, 8080, 5900)
LOG = logging.getLogger("nemo_gym.resources_servers.osworld.worker")


class ContainerCreateRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    deployment_id: str = Field(min_length=1, max_length=128)
    ram_size: str = "4G"
    cpu_cores: str = "4"
    disk_size: str = "32G"


@dataclass
class ContainerRecord:
    session_id: str
    container_name: str
    ports: Dict[int, int]
    created_at: float


class WorkerRuntime:
    def __init__(self) -> None:
        self.worker_id = os.environ.get("OSWORLD_WORKER_ID", socket.gethostname()).strip()
        self.head_url = os.environ.get("NEMO_GYM_HEAD_URL", "").strip().rstrip("/")
        self.token = os.environ.get("NEMO_GYM_REGISTRATION_TOKEN", "").strip()
        self.data_host = os.environ.get("OSWORLD_WORKER_DATA_HOST", "").strip()
        self.image = os.environ.get("OSWORLD_WORKER_IMAGE", "happysixd/osworld-docker:latest").strip()
        self.vm_path = os.path.expanduser(
            os.environ.get("OSWORLD_WORKER_VM_PATH", "~/osworld-assets/Ubuntu.qcow2")
        )
        self.capacity = max(1, int(os.environ.get("OSWORLD_WORKER_CAPACITY", "8")))
        self.publish_host = os.environ.get("OSWORLD_WORKER_PUBLISH_HOST", "0.0.0.0").strip()
        self.deployment_id = os.environ.get("OSWORLD_DEPLOYMENT_ID", "osworld-decoupled").strip()
        self.control_url = ""
        self._containers: Dict[str, ContainerRecord] = {}
        self._reservations = 0
        self._lock = Lock()
        self._heartbeat_task: asyncio.Task[None] | None = None

    @property
    def active(self) -> int:
        with self._lock:
            return len(self._containers) + self._reservations

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def preflight(self) -> None:
        if not self.head_url.startswith(("http://", "https://")):
            raise RuntimeError("NEMO_GYM_HEAD_URL must be an HTTP URL")
        if not self.data_host:
            raise RuntimeError("OSWORLD_WORKER_DATA_HOST is required")
        if not os.path.isfile(self.vm_path):
            raise RuntimeError(f"OSWorld VM image is missing: {self.vm_path}")
        if not os.path.exists("/dev/kvm"):
            raise RuntimeError("/dev/kvm is required")
        self._run(["docker", "version", "--format", "{{.Server.Version}}"])
        self._run(["docker", "image", "inspect", self.image])
        LOG.info(
            "Worker preflight passed worker_id=%s data_host=%s capacity=%d image=%s vm_path=%s",
            self.worker_id,
            self.data_host,
            self.capacity,
            self.image,
            self.vm_path,
        )

    def cleanup_orphans(self) -> None:
        result = self._run(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                "label=osworld.managed=true",
                "--filter",
                f"label=osworld.deployment_id={self.deployment_id}",
            ],
            check=False,
        )
        ids = result.stdout.split()
        if ids:
            LOG.warning("Removing %d orphaned OSWorld container(s): %s", len(ids), ids)
            self._run(["docker", "rm", "-f", *ids], check=False)

    def create_container(self, body: ContainerCreateRequest) -> Dict[str, Any]:
        if body.deployment_id != self.deployment_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"worker belongs to deployment {self.deployment_id!r}, "
                    f"not {body.deployment_id!r}"
                ),
            )
        reservation_held = False
        with self._lock:
            existing = self._containers.get(body.session_id)
            if existing is not None:
                return self._response(existing)
            if len(self._containers) + self._reservations >= self.capacity:
                raise HTTPException(status_code=503, detail="worker capacity exhausted")
            self._reservations += 1
            reservation_held = True

        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", body.session_id).strip("-.")[:48]
        container_name = f"osworld-{slug}-{uuid.uuid4().hex[:8]}"
        try:
            command = [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                container_name,
                "--label",
                "osworld.managed=true",
                "--label",
                f"osworld.deployment_id={body.deployment_id}",
                "--label",
                f"osworld.session_id={body.session_id}",
            ]
            for port in CONTAINER_PORTS:
                command += ["-p", f"{self.publish_host}::{port}"]
            command += [
                "-e",
                f"DISK_SIZE={body.disk_size}",
                "-e",
                f"RAM_SIZE={body.ram_size}",
                "-e",
                f"CPU_CORES={body.cpu_cores}",
                "-e",
                "KVM=Y",
                "-e",
                "USER_PORTS=5000",
                "--device",
                "/dev/kvm",
                "--cap-add",
                "NET_ADMIN",
                "--device",
                "/dev/net/tun",
                "-v",
                f"{self.vm_path}:/System.qcow2:ro",
                self.image,
            ]
            self._run(command, timeout=120)
            ports = self._inspect_ports(container_name)
            record = ContainerRecord(
                session_id=body.session_id,
                container_name=container_name,
                ports=ports,
                created_at=time.time(),
            )
            with self._lock:
                self._reservations = max(0, self._reservations - 1)
                reservation_held = False
                self._containers[body.session_id] = record
            LOG.info(
                "Created OSWorld container session_id=%s container=%s ports=%s active=%d/%d",
                body.session_id,
                container_name,
                ports,
                self.active,
                self.capacity,
            )
            return self._response(record)
        except Exception:
            LOG.exception(
                "Failed to create OSWorld container session_id=%s container=%s",
                body.session_id,
                container_name,
            )
            self._run(["docker", "rm", "-f", container_name], check=False)
            raise
        finally:
            if reservation_held:
                with self._lock:
                    self._reservations = max(0, self._reservations - 1)

    def delete_container(self, session_id: str) -> bool:
        with self._lock:
            record = self._containers.pop(session_id, None)
        if record is None:
            LOG.info("Delete is already satisfied for unknown session_id=%s", session_id)
            return True
        self._run(["docker", "rm", "-f", record.container_name], check=False, timeout=60)
        LOG.info(
            "Deleted OSWorld container session_id=%s container=%s active=%d/%d",
            session_id,
            record.container_name,
            self.active,
            self.capacity,
        )
        return True

    def inspect_container(self, session_id: str) -> Dict[str, Any]:
        """Return worker-local Docker state for HTTP control-plane liveness checks."""

        with self._lock:
            record = self._containers.get(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="session is not owned by this worker")
        result = self._run(
            ["docker", "inspect", "--format", "{{json .State}}", record.container_name],
            check=False,
        )
        state: Dict[str, Any] = {}
        if result.returncode == 0 and result.stdout.strip():
            try:
                parsed = json.loads(result.stdout.strip())
                if isinstance(parsed, dict):
                    state = parsed
            except json.JSONDecodeError:
                state = {"Status": "invalid-inspect-response"}
        running = bool(state.get("Running", False))
        logs = ""
        if not running:
            logs_result = self._run(
                ["docker", "logs", "--tail", "80", record.container_name],
                check=False,
            )
            logs = (logs_result.stdout + logs_result.stderr)[-16000:]
            LOG.warning(
                "Container liveness failed session_id=%s container=%s state=%s",
                session_id,
                record.container_name,
                state,
            )
        return {
            **self._response(record),
            "running": running,
            "state": state,
            "logs": logs,
        }

    def _inspect_ports(self, container_name: str) -> Dict[int, int]:
        result = self._run(
            ["docker", "inspect", "--format", "{{json .NetworkSettings.Ports}}", container_name]
        )
        mappings = json.loads(result.stdout.strip())
        ports: Dict[int, int] = {}
        for port in CONTAINER_PORTS:
            entries = mappings.get(f"{port}/tcp") or []
            if not entries or not entries[0].get("HostPort"):
                raise RuntimeError(f"Docker did not publish container port {port}")
            ports[port] = int(entries[0]["HostPort"])
        return ports

    def _response(self, record: ContainerRecord) -> Dict[str, Any]:
        return {
            "session_id": record.session_id,
            "container_name": record.container_name,
            "data_host": self.data_host,
            "ports": {str(key): value for key, value in record.ports.items()},
            "created_at": record.created_at,
        }

    def _registration_body(self) -> Dict[str, Any]:
        return {
            "service_id": self.worker_id,
            "service_type": "osworld_worker",
            "url": self.control_url,
            "capacity": self.capacity,
            "active": self.active,
            "status": "ready",
            "ttl_seconds": 30,
            "metadata": {
                "deployment_id": self.deployment_id,
                "data_host": self.data_host,
                "assets_dir": os.path.dirname(self.vm_path),
                "vm_filename": os.path.basename(self.vm_path),
                "image": self.image,
                "kvm": True,
                "ram_size": os.environ.get("OSWORLD_WORKER_RAM_SIZE", "4G"),
                "cpu_cores": os.environ.get("OSWORLD_WORKER_CPU_CORES", "4"),
                "disk_size": os.environ.get("OSWORLD_WORKER_DISK_SIZE", "32G"),
            },
        }

    async def start(self) -> None:
        await asyncio.to_thread(self.preflight)
        await asyncio.to_thread(self.cleanup_orphans)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        with self._lock:
            session_ids = list(self._containers)
        await asyncio.gather(
            *(asyncio.to_thread(self.delete_container, session_id) for session_id in session_ids)
        )
        try:
            await asyncio.to_thread(
                requests.delete,
                f"{self.head_url}/services/{self.worker_id}",
                headers=self.headers,
                timeout=5,
            )
        except requests.RequestException:
            pass

    async def _heartbeat_loop(self) -> None:
        registered = False
        while True:
            try:
                if registered:
                    response = await asyncio.to_thread(
                        requests.post,
                        f"{self.head_url}/services/{self.worker_id}/heartbeat",
                        json={
                            "capacity": self.capacity,
                            "active": self.active,
                            "status": "ready",
                        },
                        headers=self.headers,
                        timeout=5,
                    )
                    if response.status_code == 404:
                        registered = False
                    else:
                        response.raise_for_status()
                if not registered:
                    response = await asyncio.to_thread(
                        requests.post,
                        f"{self.head_url}/services/register",
                        json=self._registration_body(),
                        headers=self.headers,
                        timeout=5,
                    )
                    response.raise_for_status()
                    registered = True
                    LOG.info(
                        "Registered worker_id=%s control_url=%s capacity=%d with head=%s",
                        self.worker_id,
                        self.control_url,
                        self.capacity,
                        self.head_url,
                    )
            except requests.RequestException as exc:
                if registered:
                    LOG.warning("Worker heartbeat failed; will re-register: %s", exc)
                else:
                    LOG.warning("Worker registration failed; retrying: %s", exc)
                registered = False
            await asyncio.sleep(10)

    @staticmethod
    def _run(
        command: list[str],
        *,
        check: bool = True,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        if check and result.returncode != 0:
            raise RuntimeError(
                f"command failed ({result.returncode}): {command!r}; "
                f"stdout={result.stdout!r}; stderr={result.stderr!r}"
            )
        return result


runtime = WorkerRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await runtime.start()
    try:
        yield
    finally:
        await runtime.stop()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def authorize(request: Request, call_next):
    if request.url.path == "/healthz" or not runtime.token:
        return await call_next(request)
    authorization = request.headers.get("authorization", "")
    supplied = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    if not supplied or not hmac.compare_digest(supplied, runtime.token):
        raise HTTPException(status_code=401, detail="invalid worker token")
    return await call_next(request)


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "worker_id": runtime.worker_id,
        "capacity": runtime.capacity,
        "active": runtime.active,
        "control_url": runtime.control_url,
        "data_host": runtime.data_host,
    }


@app.post("/v1/containers")
async def create_container(body: ContainerCreateRequest) -> Dict[str, Any]:
    return await asyncio.to_thread(runtime.create_container, body)


@app.get("/v1/containers/{session_id}")
async def inspect_container(session_id: str) -> Dict[str, Any]:
    return await asyncio.to_thread(runtime.inspect_container, session_id)


@app.delete("/v1/containers/{session_id}")
async def delete_container(session_id: str) -> Dict[str, Any]:
    return {"removed": await asyncio.to_thread(runtime.delete_container, session_id)}


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("OSWORLD_WORKER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    bind_host = os.environ.get("OSWORLD_WORKER_BIND_HOST", "0.0.0.0")
    requested_port = int(os.environ.get("OSWORLD_WORKER_PORT", "0"))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((bind_host, requested_port))
    listener.listen(2048)
    port = int(listener.getsockname()[1])
    runtime.control_url = f"http://{runtime.data_host}:{port}"
    LOG.info(
        "Bound worker control API bind_host=%s dynamic_port=%d advertised_url=%s",
        bind_host,
        port,
        runtime.control_url,
    )
    server = uvicorn.Server(uvicorn.Config(app, log_level="info"))
    server.run(sockets=[listener])


if __name__ == "__main__":
    main()
