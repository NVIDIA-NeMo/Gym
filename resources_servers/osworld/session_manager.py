"""OSWorld session affinity and remote Docker worker-pool scheduling."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

from nemo_gym.server_utils import ServerClient
from resources_servers.osworld.config import (
    OSWorldResourcesServerConfig,
    RemoteDockerWorkerConfig,
)
from resources_servers.osworld.models import (
    OSWorldEvaluateResponse,
    OSWorldObservation,
    OSWorldResetRequest,
    OSWorldSeedSessionRequest,
    OSWorldSeedSessionResponse,
    OSWorldSessionStatusResponse,
    OSWorldStepRequest,
    OSWorldStepResponse,
)


LOG = logging.getLogger("nemo_gym.resources_servers.osworld")


class SessionNotFoundError(KeyError):
    pass


class SessionConflictError(RuntimeError):
    pass


class CapacityUnavailableError(RuntimeError):
    pass


@dataclass
class WorkerState:
    config: RemoteDockerWorkerConfig
    active_sessions: set[str] = field(default_factory=set)
    reservations: int = 0
    last_error: Optional[str] = None

    @property
    def load(self) -> int:
        return len(self.active_sessions) + self.reservations

    @property
    def available(self) -> int:
        return max(0, self.config.capacity - self.load)


@dataclass
class SessionState:
    session_id: str
    task_id: str
    worker_name: str
    env: Any
    observation: Dict[str, Any]
    created_at: float
    last_access_at: float
    status: str = "ready"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    operations: OrderedDict[str, OSWorldStepResponse] = field(default_factory=OrderedDict)


EnvFactory = Callable[..., Any]
AdminRunner = Callable[[RemoteDockerWorkerConfig, str, int], subprocess.CompletedProcess]


def _default_env_factory(**kwargs: Any) -> Any:
    from desktop_env.desktop_env import DesktopEnv

    return DesktopEnv(**kwargs)


def _default_admin_runner(
    worker: RemoteDockerWorkerConfig,
    remote_command: str,
    timeout: int,
) -> subprocess.CompletedProcess:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    if worker.ssh_key:
        command += [
            "-i",
            os.path.abspath(os.path.expanduser(worker.ssh_key)),
            "-o",
            "IdentitiesOnly=yes",
        ]
    if worker.ssh_port != 22:
        command += ["-p", str(worker.ssh_port)]
    command += [worker.remote_host, remote_command]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class OSWorldSessionManager:
    """Own live ``DesktopEnv`` objects and bind each Gym session to one worker."""

    def __init__(
        self,
        config: OSWorldResourcesServerConfig,
        *,
        env_factory: EnvFactory = _default_env_factory,
        admin_runner: AdminRunner = _default_admin_runner,
    ) -> None:
        self.config = config
        self._env_factory = env_factory
        self._admin_runner = admin_runner
        self._workers = {
            worker.name: WorkerState(config=worker) for worker in config.workers
        }
        self._sessions: Dict[str, SessionState] = {}
        self._creating: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._reaper_task: Optional[asyncio.Task[None]] = None
        self._started_at = time.time()
        self._state_dir = Path(config.resolved_state_dir())
        self._state_path = self._state_dir / "sessions.json"

    async def start(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        Path(self.config.resolved_cache_dir()).mkdir(parents=True, exist_ok=True)
        if self.config.enable_proxy:
            proxy_path = os.path.abspath(os.path.expanduser(self.config.proxy_config_file or ""))
            if not os.path.isfile(proxy_path):
                raise RuntimeError(f"Configured proxy file does not exist: {proxy_path}")
            os.environ["PROXY_CONFIG_FILE"] = proxy_path
        await self.refresh_registered_workers()
        if self.config.cleanup_orphans_on_start:
            await self.cleanup_orphaned_containers()
        self._persist_state()
        self._reaper_task = asyncio.create_task(
            self._reaper_loop(), name="osworld-session-reaper"
        )

    async def stop(self) -> None:
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None
        async with self._lock:
            session_ids = list(self._sessions)
        await asyncio.gather(
            *(self.close_session(session_id) for session_id in session_ids),
            return_exceptions=True,
        )
        self._persist_state()

    async def cleanup_orphaned_containers(self) -> None:
        """Remove containers left by an earlier instance of this deployment."""

        deployment_filter = shlex.quote(
            f"label=osworld.deployment_id={self.config.deployment_id}"
        )
        command = (
            "ids=$(docker ps -aq --filter label=osworld.managed=true "
            f"--filter {deployment_filter}); "
            "if [ -n \"$ids\" ]; then docker rm -f $ids; fi"
        )

        async def clean(worker_state: WorkerState) -> None:
            if worker_state.config.transport == "http_control":
                return
            try:
                result = await asyncio.to_thread(
                    self._admin_runner,
                    worker_state.config,
                    command,
                    120,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"exit={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"
                    )
                worker_state.last_error = None
                LOG.info("Orphan cleanup completed on worker %s", worker_state.config.name)
            except Exception as exc:  # noqa: BLE001
                worker_state.last_error = f"orphan cleanup failed: {exc}"
                LOG.exception("Orphan cleanup failed on worker %s", worker_state.config.name)

        await asyncio.gather(*(clean(worker) for worker in self._workers.values()))

    async def seed_session(
        self,
        session_id: str,
        body: OSWorldSeedSessionRequest,
    ) -> OSWorldSeedSessionResponse:
        await self.refresh_registered_workers()
        task_id = str(
            body.task_config.get("id") or body.task_config.get("task_id") or "unknown"
        )
        async with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                if existing.task_id != task_id:
                    raise SessionConflictError(
                        f"session {session_id!r} already owns task {existing.task_id!r}"
                    )
                existing.last_access_at = time.time()
                return self._seed_response(existing)
            if session_id in self._creating:
                raise SessionConflictError(f"session {session_id!r} is already being created")
            worker = self._select_worker_locked()
            worker.reservations += 1
            self._creating[session_id] = worker.config.name

        env: Any = None
        try:
            env, observation = await asyncio.to_thread(
                self._create_environment,
                session_id,
                worker.config,
                body,
            )
            now = time.time()
            state = SessionState(
                session_id=session_id,
                task_id=task_id,
                worker_name=worker.config.name,
                env=env,
                observation=observation,
                created_at=now,
                last_access_at=now,
            )
            async with self._lock:
                worker.reservations -= 1
                worker.active_sessions.add(session_id)
                worker.last_error = None
                self._creating.pop(session_id, None)
                self._sessions[session_id] = state
                self._persist_state()
            LOG.info(
                "Seeded Gym session=%s OSWorld task=%s worker=%s",
                session_id,
                task_id,
                worker.config.name,
            )
            return self._seed_response(state)
        except Exception as exc:
            if env is not None:
                try:
                    await asyncio.to_thread(env.close)
                except Exception:  # noqa: BLE001
                    LOG.exception("Cleanup failed after OSWorld seed error")
            async with self._lock:
                worker.reservations = max(0, worker.reservations - 1)
                worker.last_error = f"session creation failed: {exc}"
                self._creating.pop(session_id, None)
                self._persist_state()
            raise

    def _create_environment(
        self,
        session_id: str,
        worker: RemoteDockerWorkerConfig,
        body: OSWorldSeedSessionRequest,
    ) -> tuple[Any, Dict[str, Any]]:
        task_requires_proxy = bool(body.task_config.get("proxy", False))
        if task_requires_proxy and not (
            self.config.enable_proxy and body.environment.enable_proxy
        ):
            raise RuntimeError(
                "task requires proxy, but the Resources Server proxy path is disabled"
            )
        session_cache = os.path.join(self.config.resolved_cache_dir(), session_id)
        os.makedirs(session_cache, exist_ok=True)
        options = body.environment
        env = self._env_factory(
            provider_name="remote_docker",
            provider_options=worker.provider_options(
                session_id=session_id,
                deployment_id=self.config.deployment_id,
                control_token=self.config.registration_token(),
            ),
            action_space=options.action_space,
            screen_size=(options.screen_width, options.screen_height),
            headless=options.headless,
            require_a11y_tree=options.require_a11y_tree,
            require_terminal=options.require_terminal,
            os_type="Ubuntu",
            client_password=options.client_password,
            cache_dir=session_cache,
            enable_proxy=self.config.enable_proxy and options.enable_proxy,
        )
        try:
            observation = env.reset(task_config=body.task_config)
        except Exception:
            env.close()
            raise
        return env, observation

    async def reset_session(
        self,
        session_id: str,
        body: OSWorldResetRequest,
    ) -> OSWorldSeedSessionResponse:
        state = await self._get_session(session_id)
        async with state.lock:
            state.status = "resetting"
            try:
                observation = await asyncio.to_thread(
                    state.env.reset, task_config=body.task_config
                )
                state.task_id = str(
                    body.task_config.get("id")
                    or body.task_config.get("task_id")
                    or "unknown"
                )
                state.observation = observation
                state.operations.clear()
                state.status = "ready"
                state.last_access_at = time.time()
                self._persist_state()
                return self._seed_response(state)
            except Exception:
                state.status = "error"
                self._persist_state()
                raise

    async def observe(self, session_id: str) -> OSWorldObservation:
        state = await self._get_session(session_id)
        async with state.lock:
            observation = await asyncio.to_thread(state.env._get_obs)  # noqa: SLF001
            state.observation = observation
            state.last_access_at = time.time()
            return OSWorldObservation.from_observation(observation)

    async def step(
        self,
        session_id: str,
        body: OSWorldStepRequest,
    ) -> OSWorldStepResponse:
        state = await self._get_session(session_id)
        async with state.lock:
            cached = state.operations.get(body.operation_id)
            if cached is not None:
                state.operations.move_to_end(body.operation_id)
                state.last_access_at = time.time()
                return cached
            state.status = "stepping"
            try:
                observation, reward, done, info = await asyncio.to_thread(
                    state.env.step,
                    body.action,
                    body.pause,
                )
                response = OSWorldStepResponse(
                    operation_id=body.operation_id,
                    observation=OSWorldObservation.from_observation(observation),
                    reward=float(reward or 0.0),
                    done=bool(done),
                    info=info if isinstance(info, dict) else {"value": info},
                )
                state.observation = observation
                state.operations[body.operation_id] = response
                while len(state.operations) > 128:
                    state.operations.popitem(last=False)
                state.status = "ready"
                state.last_access_at = time.time()
                return response
            except Exception:
                state.status = "error"
                self._persist_state()
                raise

    async def evaluate(self, session_id: str) -> OSWorldEvaluateResponse:
        state = await self._get_session(session_id)
        async with state.lock:
            state.status = "evaluating"
            try:
                score = await asyncio.to_thread(state.env.evaluate)
                state.status = "ready"
                state.last_access_at = time.time()
                return OSWorldEvaluateResponse(score=float(score))
            except Exception:
                state.status = "error"
                self._persist_state()
                raise

    async def close_session(self, session_id: str) -> bool:
        async with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is None:
                return True
            state.status = "closing"
        try:
            async with state.lock:
                await asyncio.to_thread(state.env.close)
        finally:
            async with self._lock:
                worker = self._workers[state.worker_name]
                worker.active_sessions.discard(session_id)
                self._persist_state()
        LOG.info("Closed Gym session=%s worker=%s", session_id, state.worker_name)
        return True

    async def session_status(self, session_id: str) -> OSWorldSessionStatusResponse:
        state = await self._get_session(session_id)
        return OSWorldSessionStatusResponse(
            session_id=state.session_id,
            task_id=state.task_id,
            worker=state.worker_name,
            status=state.status,
            created_at=state.created_at,
            last_access_at=state.last_access_at,
        )

    async def health(self) -> Dict[str, Any]:
        await self.refresh_registered_workers()
        async with self._lock:
            return {
                "status": "ok",
                "deployment_id": self.config.deployment_id,
                "uptime_seconds": max(0.0, time.time() - self._started_at),
                "sessions": len(self._sessions),
                "creating": len(self._creating),
                "capacity": sum(worker.config.capacity for worker in self._workers.values()),
                "proxy_configured": bool(
                    self.config.enable_proxy and self.config.proxy_config_file
                ),
                "workers": [
                    {
                        "name": worker.config.name,
                        "remote_host": worker.config.remote_host,
                        "data_host": worker.config.data_host,
                        "transport": worker.config.transport,
                        "capacity": worker.config.capacity,
                        "active": len(worker.active_sessions),
                        "reserved": worker.reservations,
                        "available": worker.available,
                        "last_error": worker.last_error,
                    }
                    for worker in self._workers.values()
                ],
            }

    async def _get_session(self, session_id: str) -> SessionState:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(session_id)
            state.last_access_at = time.time()
            return state

    def _select_worker_locked(self) -> WorkerState:
        candidates = [worker for worker in self._workers.values() if worker.available]
        if not candidates:
            raise CapacityUnavailableError(
                f"all remote Docker workers are full (capacity={self.config.worker_capacity})"
            )
        return min(
            candidates,
            key=lambda worker: (
                worker.load / worker.config.capacity,
                worker.load,
                worker.config.name,
            ),
        )

    def _seed_response(self, state: SessionState) -> OSWorldSeedSessionResponse:
        return OSWorldSeedSessionResponse(
            session_id=state.session_id,
            task_id=state.task_id,
            worker=state.worker_name,
            status=state.status,
            observation=OSWorldObservation.from_observation(state.observation),
        )

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.reaper_interval_seconds)
            await self.refresh_registered_workers()
            cutoff = time.time() - self.config.session_ttl_seconds
            async with self._lock:
                stale = [
                    session_id
                    for session_id, state in self._sessions.items()
                    if state.last_access_at < cutoff
                ]
            if stale:
                LOG.warning("Reaping %d expired OSWorld session(s)", len(stale))
                await asyncio.gather(
                    *(self.close_session(session_id) for session_id in stale),
                    return_exceptions=True,
                )

    async def refresh_registered_workers(self) -> None:
        """Reconcile the schedulable pool with services leased from Gym head."""

        if not self.config.discover_workers:
            return

        def fetch() -> list[dict[str, Any]]:
            head = ServerClient.load_head_server_config()
            headers = {}
            token = self.config.registration_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            response = requests.get(
                f"http://{head.host}:{head.port}/services",
                params={"service_type": "osworld_worker"},
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, list) else []

        try:
            records = await asyncio.to_thread(fetch)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not refresh OSWorld worker registry: %s", exc)
            return

        discovered: Dict[str, RemoteDockerWorkerConfig] = {}
        for record in records:
            if record.get("status") != "ready":
                continue
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            capacity = max(1, int(record.get("capacity") or 1))
            discovered[str(record["service_id"])] = RemoteDockerWorkerConfig(
                name=str(record["service_id"]),
                data_host=str(metadata.get("data_host") or ""),
                control_url=str(record.get("url") or ""),
                capacity=capacity,
                assets_dir=str(metadata.get("assets_dir") or "~/osworld-assets"),
                vm_filename=str(metadata.get("vm_filename") or "Ubuntu.qcow2"),
                image=str(metadata.get("image") or "happysixd/osworld-docker:latest"),
                kvm=bool(metadata.get("kvm", True)),
                ram_size=str(metadata.get("ram_size") or "4G"),
                cpu_cores=str(metadata.get("cpu_cores") or "4"),
                disk_size=str(metadata.get("disk_size") or "32G"),
                transport="http_control",
            )

        async with self._lock:
            for name, worker_config in discovered.items():
                state = self._workers.get(name)
                if state is None:
                    self._workers[name] = WorkerState(config=worker_config)
                else:
                    state.config = worker_config
                    state.last_error = None
            for name in list(self._workers):
                if name in discovered:
                    continue
                state = self._workers[name]
                if state.config.transport != "http_control":
                    continue
                if state.active_sessions or state.reservations:
                    state.last_error = "worker registration expired"
                else:
                    self._workers.pop(name, None)

    def _persist_state(self) -> None:
        """Persist non-secret placement metadata for restart diagnostics."""

        self._state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "deployment_id": self.config.deployment_id,
            "updated_at": time.time(),
            "sessions": [
                {
                    "session_id": state.session_id,
                    "task_id": state.task_id,
                    "worker": state.worker_name,
                    "status": state.status,
                    "created_at": state.created_at,
                    "last_access_at": state.last_access_at,
                    "container_name": getattr(
                        getattr(state.env, "provider", None), "container_name", None
                    ),
                }
                for state in self._sessions.values()
            ],
            "creating": dict(self._creating),
        }
        temporary = self._state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._state_path)
