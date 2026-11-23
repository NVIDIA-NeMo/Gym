# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

import ray

from resources_servers.webarena.browser_worker import WebArenaBrowserWorker
from resources_servers.webarena.schemas import WebArenaResourcesServerConfig


logger = logging.getLogger(__name__)
_BROWSER_CLEANUP_TIMEOUT_SECONDS = 30.0


class BrowserActorPool:
    """Lazily creates Ray browser actors and grants one exclusive lease per episode."""

    def __init__(
        self,
        config: WebArenaResourcesServerConfig,
        worker_factory: Callable[[], Any] | None = None,
    ):
        self.config = config
        self._worker_factory = worker_factory
        self._idle: asyncio.Queue[Any] = asyncio.Queue()
        self._actors: set[Any] = set()
        self._leased: dict[str, Any] = {}
        self._starting = 0
        self._lock = asyncio.Lock()
        self._startup_semaphore = asyncio.Semaphore(config.browser_startup_concurrency)
        self._startup_stagger_lock = asyncio.Lock()
        self._next_actor_ready_at = 0.0
        self._closed = False

    @property
    def size(self) -> int:
        return len(self._actors)

    @property
    def leased_count(self) -> int:
        return len(self._leased)

    async def prewarm(self) -> None:
        target = self.config.initial_browser_workers
        assert target is not None
        if target:
            await asyncio.gather(*(self._create_idle_actor() for _ in range(target)))

    async def _create_idle_actor(self) -> None:
        actor = await self._create_actor()
        async with self._lock:
            if self._closed:
                await self._kill_actor(actor)
                return
            self._actors.add(actor)
            self._idle.put_nowait(actor)

    async def checkout(self, env_id: str) -> Any:
        create_new = False
        async with self._lock:
            if self._closed:
                raise RuntimeError("WebArena browser pool is closed")
            if env_id in self._leased:
                return self._leased[env_id]
            try:
                actor = self._idle.get_nowait()
            except asyncio.QueueEmpty:
                actor = None
                if len(self._actors) + self._starting < self.config.max_browser_workers:
                    self._starting += 1
                    create_new = True

        if create_new:
            try:
                actor = await self._create_actor()
            except BaseException:
                async with self._lock:
                    self._starting -= 1
                raise
            try:
                async with self._lock:
                    self._starting -= 1
                    if self._closed:
                        await self._kill_actor(actor)
                        raise RuntimeError("WebArena browser pool closed during actor startup")
                    self._actors.add(actor)
            except BaseException:
                await self._kill_actor(actor)
                raise
        elif actor is None:
            actor = await asyncio.wait_for(
                self._idle.get(),
                timeout=self.config.browser_pool_wait_timeout_seconds,
            )

        try:
            async with self._lock:
                if self._closed:
                    raise RuntimeError("WebArena browser pool is closed")
                self._leased[env_id] = actor
        except BaseException:
            async with self._lock:
                if not self._closed and actor in self._actors and actor not in self._leased.values():
                    self._idle.put_nowait(actor)
            raise
        return actor

    async def release(self, env_id: str) -> None:
        async with self._lock:
            actor = self._leased.pop(env_id, None)
        if actor is None:
            return
        reusable = False
        cancellation: asyncio.CancelledError | None = None
        try:
            reusable = bool(
                await asyncio.wait_for(
                    actor.close_episode.remote(),
                    timeout=_BROWSER_CLEANUP_TIMEOUT_SECONDS,
                )
            )
        except asyncio.CancelledError as exc:
            cancellation = exc
        except Exception:
            logger.warning("Browser actor cleanup failed; retiring actor", exc_info=True)
        if reusable:
            try:
                health = await asyncio.wait_for(
                    actor.health.remote(),
                    timeout=_BROWSER_CLEANUP_TIMEOUT_SECONDS,
                )
                reusable = bool(health.get("healthy"))
            except asyncio.CancelledError as exc:
                cancellation = exc
                reusable = False
            except Exception:
                reusable = False
        recycled = False
        if reusable:
            async with self._lock:
                if not self._closed and actor in self._actors:
                    self._idle.put_nowait(actor)
                    recycled = True
        if not recycled:
            await self._retire_actor(actor, graceful=False)
        if cancellation is not None:
            raise cancellation

    async def retire(self, env_id: str) -> None:
        async with self._lock:
            actor = self._leased.pop(env_id, None)
        if actor is not None:
            await self._retire_actor(actor, graceful=False)

    async def _create_actor(self) -> Any:
        actor = None
        try:
            async with self._startup_semaphore:
                if self._worker_factory is not None:
                    actor = self._worker_factory()
                else:
                    actor_env = {
                        name: os.environ[name]
                        for name in (
                            "NEMO_GYM_CONFIG_DICT",
                            "NEMO_GYM_CONFIG_PATH",
                        )
                        if name in os.environ
                    }
                    runtime_env: dict[str, Any] = {
                        "py_executable": sys.executable,
                    }
                    if actor_env:
                        runtime_env["env_vars"] = actor_env
                    actor_options: dict[str, Any] = {
                        "num_cpus": self.config.browser_actor_num_cpus,
                        "scheduling_strategy": "SPREAD",
                        "runtime_env": runtime_env,
                    }
                    if self.config.browser_actor_resources:
                        actor_options["resources"] = self.config.browser_actor_resources
                    actor = WebArenaBrowserWorker.options(
                        **actor_options,
                    ).remote(self.config.model_dump(mode="json"))
                health = await asyncio.wait_for(
                    actor.health.remote(),
                    timeout=self.config.browser_startup_timeout_seconds,
                )
            if not health.get("healthy"):
                raise RuntimeError(f"WebArena browser actor failed health check: {health}")
            await self._wait_for_startup_stagger()
        except BaseException:
            if actor is not None:
                await self._kill_actor(actor)
            raise
        return actor

    async def _wait_for_startup_stagger(self) -> None:
        stagger = self.config.browser_startup_stagger_seconds
        if stagger <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self._startup_stagger_lock:
            now = loop.time()
            ready_at = max(now, self._next_actor_ready_at)
            self._next_actor_ready_at = ready_at + stagger
        delay = ready_at - loop.time()
        if delay > 0:
            logger.info("Staggering browser actor readiness by %.3f seconds", delay)
            await asyncio.sleep(delay)

    async def _retire_actor(self, actor: Any, *, graceful: bool = True) -> None:
        async with self._lock:
            if actor not in self._actors:
                return
            self._actors.remove(actor)
        await self._kill_actor(actor, graceful=graceful)

    async def _kill_actor(self, actor: Any, *, graceful: bool = True) -> None:
        try:
            if graceful:
                try:
                    await asyncio.wait_for(
                        actor.shutdown.remote(),
                        timeout=_BROWSER_CLEANUP_TIMEOUT_SECONDS,
                    )
                except Exception:
                    logger.debug("Graceful browser actor shutdown failed", exc_info=True)
        finally:
            # Ray cancellation is best-effort, including for actor methods.
            # Always kill a retired actor so an uncertain operation can never
            # complete and later re-enter the idle pool.
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                logger.debug("ray.kill failed for browser actor", exc_info=True)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            actors = list(self._actors)
            self._actors.clear()
            self._leased.clear()
            while not self._idle.empty():
                self._idle.get_nowait()
        await asyncio.gather(*(self._kill_actor(actor) for actor in actors), return_exceptions=True)
