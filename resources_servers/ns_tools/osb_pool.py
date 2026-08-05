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
"""Static pool of long-lived OpenSandbox pods that serve the NeMo-Skills sandbox HTTP protocol.

Each pod runs an unmodified NeMo-Skills sandbox server on a declared port; clients reach it
through the OpenSandbox server proxy endpoint (resolved via ``AsyncSandbox.endpoint``), so the
data path is plain HTTP request/response. The pool pins each NS session uuid to one pod
(dict-assignment stickiness, least-loaded first), heals dead pods in place through a token
bucket, and collapses total-outage into the NS client's existing timeout contract.

This module is imported only when ns_tools selects the ``opensandbox_pool`` backend; the
default ``local`` backend never touches it.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import httpx

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec


LOGGER = logging.getLogger(__name__)


def _provider_domain_and_key(provider: Dict[str, Any]) -> Tuple[str, str]:
    """Pull connection.domain/api_key out of a single-key provider config dict."""
    if not isinstance(provider, dict) or len(provider) != 1:
        raise ValueError("opensandbox_pool.provider must be a single-key provider config dict")
    kwargs = next(iter(provider.values())) or {}
    connection = kwargs.get("connection") or {}
    return str(connection.get("domain") or ""), str(connection.get("api_key") or "")


@dataclass
class _Slot:
    index: int
    sandbox: Optional[AsyncSandbox] = None
    base_url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    healthy: bool = False
    strikes: int = 0
    sessions: set = field(default_factory=set)


class OpenSandboxPool:
    """K long-lived NS-sandbox pods with sticky session routing and in-slot healing.

    The constructor is pure (config validation only); ``start()`` kicks a budgeted,
    non-blocking warmup plus the heal and idle-sweep loops. ``route()`` lazily starts
    the pool as a safety net if the owning server never called ``start()``.
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
        resources: Optional[Dict[str, Any]] = None,
        setup_files: Optional[Dict[str, str]] = None,
        setup_commands: Optional[list] = None,
        service_command: Optional[str] = None,
        health_path: str = "/health",
        warmup_create_concurrency: int = 8,
        health_interval_s: float = 15.0,
        health_timeout_s: float = 10.0,
        heal_creates_per_s: float = 0.5,
        session_idle_sweep_s: float = 7200.0,
        run_label: Optional[str] = None,
    ) -> None:
        domain, api_key = _provider_domain_and_key(provider)
        if not domain or not api_key:
            raise ValueError(
                "opensandbox_pool backend selected but the provider connection has an empty "
                "domain or api_key — set OPENSANDBOX_BASE_URL / OPENSANDBOX_API_KEY"
            )
        if not image:
            raise ValueError("opensandbox_pool backend selected but image is empty — set NS_SANDBOX_IMAGE")
        if int(size) < 1:
            raise ValueError(f"opensandbox_pool.size must be >= 1, got {size}")
        self._provider_config = provider
        self._image = image
        self._port = int(port)
        self._size = int(size)
        self._ttl_s = ttl_s
        self._env = dict(env or {})
        self._entrypoint = list(entrypoint) if entrypoint else None
        self._resources = dict(resources or {})
        self._setup_files = dict(setup_files or {})
        self._setup_commands = list(setup_commands or [])
        self._service_command = service_command
        self._health_path = health_path
        self._warmup_create_concurrency = warmup_create_concurrency
        self._health_interval_s = health_interval_s
        self._health_timeout_s = health_timeout_s
        self._heal_min_interval_s = 1.0 / heal_creates_per_s if heal_creates_per_s > 0 else 0.0
        self._session_idle_sweep_s = session_idle_sweep_s
        self._run_label = run_label

        self._slots = [_Slot(index=i) for i in range(self._size)]
        self._session_to_slot: Dict[str, int] = {}
        self._session_last_used: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self._tasks: list = []
        self._last_heal_create = 0.0
        self._http = httpx.AsyncClient(timeout=self._health_timeout_s)

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Kick warmup + maintenance loops; returns immediately (warmup is budgeted, not blocking)."""
        if self._started or self._closed:
            return
        self._started = True
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
        for slot in self._slots:
            if slot.sandbox is not None:
                try:
                    await asyncio.wait_for(slot.sandbox.stop(), timeout=30.0)
                except Exception as exc:
                    LOGGER.warning("pool slot %d teardown failed (TTL will reap): %s", slot.index, exc)
                slot.sandbox = None
                slot.healthy = False
        await self._http.aclose()

    def _spec(self) -> SandboxSpec:
        metadata = {"purpose": "ns-tools-sandbox-pool"}
        if self._run_label:
            metadata["run"] = self._run_label
        return SandboxSpec(
            image=self._image,
            ttl_s=self._ttl_s,
            env=self._env,
            entrypoint=self._entrypoint,
            resources=self._resources,
            metadata=metadata,
            ports=(self._port,),
        )

    async def _create_slot(self, slot: _Slot) -> None:
        """Create, bootstrap, health-gate, and admit one pod into its slot."""
        sandbox = AsyncSandbox(provider=dict(self._provider_config), spec=self._spec())
        await sandbox.start()
        try:
            for target_path, local_path in self._setup_files.items():
                await sandbox.upload(local_path, target_path)
            for command in self._setup_commands:
                result = await sandbox.exec(command, timeout_s=600)
                if result.return_code != 0:
                    raise RuntimeError(f"setup command failed rc={result.return_code}: {command!r}: {result.stderr[:500]}")
            if self._service_command:
                result = await sandbox.exec(self._service_command, timeout_s=30)
                if result.return_code != 0:
                    raise RuntimeError(f"service command failed rc={result.return_code}: {result.stderr[:500]}")
            resolved = await sandbox.endpoint(self._port)
            base_url = resolved.endpoint.rstrip("/")
            await self._wait_healthy(base_url, dict(resolved.headers), budget_s=120.0)
        except Exception:
            try:
                await sandbox.stop()
            except Exception:
                pass
            raise
        slot.sandbox = sandbox
        slot.base_url = base_url
        slot.headers = dict(resolved.headers)
        slot.strikes = 0
        slot.healthy = True

    async def _wait_healthy(self, base_url: str, headers: Dict[str, str], budget_s: float) -> None:
        """Gate admission on the ACTUAL traffic path: proxied GET /health must return 200."""
        deadline = time.monotonic() + budget_s
        last_error: Optional[str] = None
        while time.monotonic() < deadline:
            try:
                response = await self._http.get(f"{base_url}{self._health_path}", headers=headers)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = repr(exc)
            await asyncio.sleep(2.0)
        raise RuntimeError(f"pod never became healthy through the proxy: {last_error}")

    async def _warmup(self) -> None:
        semaphore = asyncio.Semaphore(self._warmup_create_concurrency)

        async def one(slot: _Slot) -> None:
            async with semaphore:
                try:
                    await self._create_slot(slot)
                except Exception as exc:
                    LOGGER.warning("pool warmup: slot %d failed (heal loop will retry): %s", slot.index, exc)

        await asyncio.gather(*(one(slot) for slot in self._slots))
        ready = sum(1 for slot in self._slots if slot.healthy)
        LOGGER.info("pool ready %d/%d", ready, self._size)

    async def _heal_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self._health_interval_s)
            for slot in self._slots:
                if slot.sandbox is None or not slot.healthy:
                    await self._heal_slot(slot)
                    continue
                try:
                    response = await self._http.get(f"{slot.base_url}{self._health_path}", headers=slot.headers)
                    ok = response.status_code == 200
                except httpx.HTTPError:
                    ok = False
                if ok:
                    slot.strikes = 0
                    continue
                slot.strikes += 1
                if slot.strikes >= 3:
                    LOGGER.warning("pool slot %d failed 3 health checks — evicting and healing in slot", slot.index)
                    slot.healthy = False
                    await self._drop_slot_sessions(slot)
                    if slot.sandbox is not None:
                        try:
                            await slot.sandbox.stop()
                        except Exception:
                            pass
                        slot.sandbox = None
                    await self._heal_slot(slot)

    async def _heal_slot(self, slot: _Slot) -> None:
        """Replace a dead pod in the SAME slot, rate-limited so heals can never storm the cell."""
        now = time.monotonic()
        wait = self._last_heal_create + self._heal_min_interval_s - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_heal_create = time.monotonic()
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
