# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A fixed set of long-lived, SHARED OpenSandbox pods serving the NeMo-Skills sandbox
HTTP protocol, with sessions multiplexed across them by sticky routing.

Sharing is what makes large batches feasible: 16k concurrent sessions ride K pods
(each pod's NS server multiplexes many sessions), instead of 16k pods. Slots fill
and heal with direct async ``Sandbox.create`` calls — a full fan-out warm wave is
~5s per pod on cell-2, so no warm-spare inventory layer is needed.

This module is imported only when ns_tools selects the ``opensandbox_pool`` backend;
the default ``local`` backend never touches it.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import aiohttp
import httpx  # exception types only: the nemo_skills client contract catches httpx errors

from nemo_gym.sandbox.attribution import RUN_KEY, resolve_attribution, resolve_run_id
from nemo_gym.sandbox.providers.opensandbox.provider import (
    DEFAULT_ATTRIBUTION_KEY_PREFIX as _ATTRIBUTION_KEY_PREFIX,
)


LOGGER = logging.getLogger(__name__)


def _parse_connection(provider: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the connection kwargs out of a single-key provider config dict."""
    if not isinstance(provider, dict) or len(provider) != 1:
        raise ValueError("opensandbox_pool.provider must be a single-key provider config dict")
    kwargs = next(iter(provider.values())) or {}
    connection = dict(kwargs.get("connection") or {})
    if not connection.get("domain") or not connection.get("api_key"):
        raise ValueError(
            "opensandbox_pool backend selected but the provider connection has an empty "
            "domain or api_key — set OPENSANDBOX_BASE_URL / OPENSANDBOX_API_KEY"
        )
    return connection


@dataclass
class _Slot:
    index: int
    sandbox: Any = None  # opensandbox.Sandbox
    base_url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    healthy: bool = False
    strikes: int = 0
    creating: bool = False
    sessions: set = field(default_factory=set)


class OpenSandboxPool:
    """K shared NS-sandbox pods with sticky session routing.

    The constructor is pure (validation only); ``start()`` kicks a non-blocking
    warmup and the health/idle-sweep loops. ``route()`` lazily starts everything
    as a safety net if the owner never called ``start()``.
    """

    def __init__(
        self,
        *,
        provider: Dict[str, Any],
        image: str,
        port: int = 6000,
        size: int = 8,
        ttl_s: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
        entrypoint: Optional[list] = None,
        resources: Optional[Dict[str, str]] = None,
        resource_requests: Optional[Dict[str, str]] = None,
        setup_files: Optional[Dict[str, str]] = None,
        setup_commands: Optional[list] = None,
        service_command: Optional[str] = None,
        health_path: str = "/health",
        ready_timeout_s: float = 30.0,
        health_budget_s: float = 300.0,
        warmup_fill_concurrency: int = 0,  # 0 = full fan-out (all slots at once)
        health_interval_s: float = 15.0,
        health_timeout_s: float = 10.0,
        heal_creates_per_s: float = 4.0,
        heal_concurrency: int = 16,
        session_idle_sweep_s: float = 7200.0,
        run_label: Optional[str] = None,
    ) -> None:
        self._connection_kwargs = _parse_connection(provider)
        if not image:
            raise ValueError("opensandbox_pool backend selected but image is empty — set NS_SANDBOX_IMAGE")
        if int(size) < 1:
            raise ValueError(f"opensandbox_pool.size must be >= 1, got {size}")
        self._image = image
        self._port = int(port)
        self._size = int(size)
        self._ttl_s = float(ttl_s) if ttl_s else None
        self._env = dict(env or {})
        self._entrypoint = list(entrypoint) if entrypoint else None
        self._resources = {k: str(v) for k, v in (resources or {}).items()}
        # k8s schedules on REQUESTS; keeping them far below limits packs many more pods
        # (= sessions) per node while bursts still get the limit headroom.
        self._resource_requests = {k: str(v) for k, v in (resource_requests or {}).items()}
        self._setup_files = dict(setup_files or {})
        self._setup_commands = list(setup_commands or [])
        self._service_command = service_command
        self._health_path = health_path
        # First pull of a large image on a fresh node can take minutes; the SDK default
        # (30s) fails creates that would have succeeded.
        self._ready_timeout_s = float(ready_timeout_s)
        self._health_budget_s = float(health_budget_s)
        self._warmup_fill_concurrency = int(warmup_fill_concurrency) or self._size
        self._connection_config: Any = None
        self._health_interval_s = health_interval_s
        self._health_timeout_s = health_timeout_s
        self._heal_min_interval_s = 1.0 / heal_creates_per_s if heal_creates_per_s > 0 else 0.0
        self._heal_concurrency = int(heal_concurrency)
        self._heal_rate_lock = asyncio.Lock()
        self._session_idle_sweep_s = session_idle_sweep_s
        self._run_label = run_label or "ns-tools-pool"
        attribution = resolve_attribution()
        attribution[RUN_KEY] = resolve_run_id(self._run_label)
        self._metadata = {f"{_ATTRIBUTION_KEY_PREFIX}{k}": v for k, v in attribution.items()}
        self._metadata["purpose"] = "ns-tools-sandbox-pool"

        self._slots = [_Slot(index=i) for i in range(self._size)]
        self._session_to_slot: Dict[str, int] = {}
        self._session_last_used: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._started = False
        self._warmup_done = False
        self._closed = False
        self._tasks: list = []
        self._last_heal_create = 0.0
        self._http: Any = None  # aiohttp session; created lazily on the serving loop

    # ------------------------------------------------------------------ lifecycle

    def _http_session(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=512, ttl_dns_cache=300),
                timeout=aiohttp.ClientTimeout(total=self._health_timeout_s),
            )
        return self._http

    async def start(self) -> None:
        """Kick warmup and the maintenance loops; returns immediately."""
        if self._started or self._closed:
            return
        self._started = True
        from opensandbox.config.connection import ConnectionConfig

        self._connection_config = ConnectionConfig(**self._connection_kwargs)
        self._tasks.append(asyncio.create_task(self._warmup(), name="osb-pool-warmup"))
        self._tasks.append(asyncio.create_task(self._heal_loop(), name="osb-pool-heal"))
        self._tasks.append(asyncio.create_task(self._sweep_loop(), name="osb-pool-sweep"))

    async def aclose(self) -> None:
        self._closed = True
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

        async def _kill(slot: _Slot) -> None:
            try:
                await asyncio.wait_for(slot.sandbox.kill(), timeout=30.0)
            except Exception as exc:
                LOGGER.warning("pool slot %d teardown failed (TTL will reap): %s", slot.index, exc)
            slot.sandbox = None
            slot.healthy = False

        await asyncio.gather(*(_kill(slot) for slot in self._slots if slot.sandbox is not None))
        if self._http is not None and not self._http.closed:
            await self._http.close()

    # ------------------------------------------------------------------ slot fill

    @property
    def _needs_prepare(self) -> bool:
        return bool(self._setup_files or self._setup_commands or self._service_command)

    async def _prepare(self, sandbox: Any) -> None:
        """Bootstrap a pod. INVARIANT: the service start is the LAST execd command this pod
        ever sees — execd reaps a completed command's backgrounded children when any later
        command runs (probed empirically: service->touch->10s = dead listener; service->
        nothing = alive)."""
        for target_path, local_path in self._setup_files.items():
            with open(local_path, "rb") as fh:
                await sandbox.files.write_file(target_path, fh.read())
        for command in self._setup_commands:
            execution = await sandbox.commands.run(command)
            if (execution.exit_code or 0) != 0:
                raise RuntimeError(f"setup command failed rc={execution.exit_code}: {command!r}")
        if self._service_command:
            # Plain shell backgrounding (cmd &): setsid and execd background:true both
            # freeze the child during module import (probed); a plain & child reparents
            # to the pod's PID 1 and survives — as long as nothing execs afterwards.
            execution = await sandbox.commands.run(self._service_command)
            if (execution.exit_code or 0) != 0:
                raise RuntimeError(f"service command failed rc={execution.exit_code}")

    def _normalize_endpoint(self, resolved: Any) -> Tuple[str, Dict[str, str]]:
        url = str(getattr(resolved, "endpoint", "") or "")
        if not url:
            raise RuntimeError("SDK returned an empty sandbox endpoint")
        if "://" not in url:
            domain = str(self._connection_kwargs.get("domain") or "")
            scheme = "https" if domain.startswith("https://") else "http"
            url = f"{scheme}://{url}"
        headers = dict(getattr(resolved, "headers", None) or {})
        if not headers and self._connection_kwargs.get("api_key"):
            headers["OPEN-SANDBOX-API-KEY"] = str(self._connection_kwargs["api_key"])
        return url.rstrip("/"), headers

    async def _create_slot(self, slot: _Slot) -> None:
        """Fill one slot. Single-flight per slot: a duplicate landing late would
        overwrite base_url under pinned sessions and leak a pod until TTL."""
        if slot.creating:
            return
        slot.creating = True
        try:
            await self._create_slot_inner(slot)
        finally:
            slot.creating = False

    async def _acquire_sandbox(self) -> Any:
        from opensandbox import Sandbox

        return await Sandbox.create(
            image=self._image,
            entrypoint=self._entrypoint,
            env=self._env or None,
            metadata=dict(self._metadata),
            resource=self._resources or None,
            resource_requests=self._resource_requests or None,
            timeout=timedelta(seconds=self._ttl_s or 14400.0),
            ready_timeout=timedelta(seconds=self._ready_timeout_s),
            connection_config=self._connection_config,
        )

    async def _create_slot_inner(self, slot: _Slot) -> None:
        sandbox = await self._acquire_sandbox()
        try:
            if self._needs_prepare:
                await self._prepare(sandbox)
            resolved = await sandbox.get_endpoint(self._port)
            base_url, headers = self._normalize_endpoint(resolved)
            await self._wait_healthy(base_url, headers, budget_s=self._health_budget_s)
        except Exception:
            try:
                await sandbox.kill()
            except Exception:
                pass
            raise
        slot.sandbox = sandbox
        slot.base_url = base_url
        slot.headers = headers
        slot.strikes = 0
        slot.healthy = True

    async def _wait_healthy(self, base_url: str, headers: Dict[str, str], budget_s: float) -> None:
        """Gate admission on the ACTUAL traffic path: proxied GET /health must return 200."""
        deadline = time.monotonic() + budget_s
        last_error: Optional[str] = None
        while time.monotonic() < deadline:
            try:
                async with self._http_session().get(f"{base_url}{self._health_path}", headers=headers) as response:
                    if response.status == 200:
                        return
                    last_error = f"HTTP {response.status}"
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = repr(exc)
            await asyncio.sleep(1.0)
        raise RuntimeError(f"pod never became healthy through the proxy: {last_error}")

    async def _warmup(self) -> None:
        semaphore = asyncio.Semaphore(self._warmup_fill_concurrency)

        async def one(slot: _Slot) -> None:
            async with semaphore:
                try:
                    await self._create_slot(slot)
                except Exception as exc:
                    LOGGER.warning("pool warmup: slot %d failed (heal loop will retry): %s", slot.index, exc)

        await asyncio.gather(*(one(slot) for slot in self._slots))
        ready = sum(1 for slot in self._slots if slot.healthy)
        self._warmup_done = True
        LOGGER.info("pool ready %d/%d", ready, self._size)

    # ------------------------------------------------------------------ maintenance

    async def _heal_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self._health_interval_s)
            if not self._warmup_done:
                # Warmup owns every slot until it finishes; healing in parallel would
                # race duplicate acquisitions into the same slot.
                continue

            async def check(slot: _Slot) -> bool:
                """Returns True when the slot needs a heal. Health probes run concurrently."""
                if slot.creating:
                    return False
                if slot.sandbox is None or not slot.healthy:
                    return True
                try:
                    async with self._http_session().get(
                        f"{slot.base_url}{self._health_path}", headers=slot.headers
                    ) as response:
                        ok = response.status == 200
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    ok = False
                if ok:
                    slot.strikes = 0
                    return False
                slot.strikes += 1
                if slot.strikes >= 3:
                    LOGGER.warning("pool slot %d failed 3 health checks — evicting and healing in slot", slot.index)
                    slot.healthy = False
                    await self._drop_slot_sessions(slot)
                    if slot.sandbox is not None:
                        try:
                            await slot.sandbox.kill()
                        except Exception:
                            pass
                        slot.sandbox = None
                    return True
                return False

            needs_heal = await asyncio.gather(*(check(slot) for slot in self._slots))
            to_heal = [slot for slot, needed in zip(self._slots, needs_heal) if needed]
            if not to_heal:
                continue
            semaphore = asyncio.Semaphore(self._heal_concurrency)

            async def heal(slot: _Slot) -> None:
                async with semaphore:
                    await self._heal_slot(slot)

            await asyncio.gather(*(heal(slot) for slot in to_heal))

    async def _heal_slot(self, slot: _Slot) -> None:
        """Replace a dead pod in the SAME slot. Heals run concurrently (bounded by
        heal_concurrency); the rate lock spaces create STARTS so a mass heal cannot
        storm the cell's create path."""
        async with self._heal_rate_lock:
            now = time.monotonic()
            start_at = max(now, self._last_heal_create + self._heal_min_interval_s)
            self._last_heal_create = start_at
        wait = start_at - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            await self._create_slot(slot)
            LOGGER.info("pool slot %d healed", slot.index)
        except Exception as exc:
            LOGGER.warning("pool slot %d heal attempt failed (will retry next interval): %s", slot.index, exc)

    async def _drop_slot_sessions(self, slot: _Slot) -> None:
        async with self._lock:
            for session_id in list(slot.sessions):
                self._session_to_slot.pop(session_id, None)
                self._session_last_used.pop(session_id, None)
            slot.sessions.clear()

    async def _sweep_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(min(self._session_idle_sweep_s, 600.0))
            cutoff = time.monotonic() - self._session_idle_sweep_s
            async with self._lock:
                stale = [s for s, t in self._session_last_used.items() if t < cutoff]
                for session_id in stale:
                    index = self._session_to_slot.pop(session_id, None)
                    self._session_last_used.pop(session_id, None)
                    if index is not None:
                        self._slots[index].sessions.discard(session_id)
            if stale:
                LOGGER.info("pool idle sweep dropped %d stale session pins", len(stale))

    # ------------------------------------------------------------------ routing

    async def route(self, session_id: Optional[str]) -> Tuple[str, Dict[str, str]]:
        """Resolve (base_url, headers) for a session; pins new sessions to the least-loaded pod.

        Raises httpx.TimeoutException when no pod is healthy, which the NS client already
        collapses into its timeout contract — total cell loss degrades rewards, never the server.
        """
        if not self._started:
            await self.start()
        async with self._lock:
            if session_id is not None:
                index = self._session_to_slot.get(session_id)
                if index is not None and self._slots[index].healthy:
                    self._session_last_used[session_id] = time.monotonic()
                    return self._slots[index].base_url, self._slots[index].headers
            healthy = [slot for slot in self._slots if slot.healthy]
            if not healthy:
                raise httpx.TimeoutException("no healthy sandbox pods in the pool")
            slot = min(healthy, key=lambda s: len(s.sessions))
            if session_id is not None:
                previous = self._session_to_slot.get(session_id)
                if previous is not None:
                    self._slots[previous].sessions.discard(session_id)
                self._session_to_slot[session_id] = slot.index
                self._session_last_used[session_id] = time.monotonic()
                slot.sessions.add(session_id)
            return slot.base_url, slot.headers

    def release(self, session_id: str) -> None:
        index = self._session_to_slot.pop(session_id, None)
        self._session_last_used.pop(session_id, None)
        if index is not None:
            self._slots[index].sessions.discard(session_id)

    @property
    def ready_count(self) -> int:
        return sum(1 for slot in self._slots if slot.healthy)
