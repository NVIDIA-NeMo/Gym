# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BrowserGym specialization of the common web session manager."""

from __future__ import annotations

from nemo_gym.web.models import WebRuntimeProfile, WebTask
from nemo_gym.web.operation_runner import ThreadAffineWebOperationRunner, WebOperationRunner
from nemo_gym.web.session import (
    BenchmarkPreconditionError,
    CapacityUnavailableError,
    SessionConflictError,
    SessionNotFoundError,
)
from nemo_gym.web.session_manager import BackendFactory, WebSessionManager
from nemo_gym.web.site_pool import SitePool
from resources_servers.browsergym_web.backend import BrowserGymBackend
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig
from resources_servers.browsergym_web.playwright_runtime import (
    close_thread_local_playwright,
    install_thread_local_playwright,
)


class BrowserGymSessionManager(WebSessionManager):
    """Create BrowserGym backends behind the shared web session contract."""

    def __init__(
        self,
        config: BrowserGymWebResourcesServerConfig,
        *,
        backend_factory: BackendFactory = BrowserGymBackend,
        site_pool: SitePool | None = None,
        operation_runner: WebOperationRunner | None = None,
    ) -> None:
        self._owns_thread_local_playwright = backend_factory is BrowserGymBackend and operation_runner is None
        super().__init__(
            config,
            backend_factory=backend_factory,
            site_pool=site_pool,
            operation_runner=operation_runner,
        )

    def _make_operation_runner(self, session_id: str) -> WebOperationRunner:
        if not self._owns_thread_local_playwright:
            return super()._make_operation_runner(session_id)
        install_thread_local_playwright()
        return ThreadAffineWebOperationRunner(
            thread_name_prefix=f"browsergym-{session_id[:8]}",
            finalizer=close_thread_local_playwright,
        )

    def _validate_task(self, task: WebTask) -> None:
        super()._validate_task(task)
        if task.runtime_profile != WebRuntimeProfile.BROWSERGYM:
            raise ValueError("this resource server only supports the browsergym runtime profile")


__all__ = [
    "BackendFactory",
    "BenchmarkPreconditionError",
    "BrowserGymSessionManager",
    "CapacityUnavailableError",
    "SessionConflictError",
    "SessionNotFoundError",
]
