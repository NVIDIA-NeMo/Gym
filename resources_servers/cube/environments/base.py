# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Abstract hook surface for CUBE resources server ``environment`` values (osworld, …)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from resources_servers.cube.server import CubeResourcesServer


class CubeEnvironmentBase(ABC):
    """Abstract base for a YAML ``environment`` value: load tasks, optional startup warmup, HTTP error strings."""

    @abstractmethod
    def ensure_loaded(self, server: CubeResourcesServer) -> None:
        """Populate ``server._task_configs_list`` / ``server._adapter_state`` when needed."""

    def warm_on_startup(self, server: CubeResourcesServer) -> None:
        """Called after ``ensure_loaded`` when ``eager_benchmark_init`` is true. Default: no-op."""

    @abstractmethod
    def empty_reset_obs_detail(self) -> str:
        """HTTP 500 detail when ``seed_session`` reset yields no observations."""
