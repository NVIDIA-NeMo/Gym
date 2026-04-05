# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``env_domain: osworld`` — :class:`OSWorldEnvironment` (:class:`CubeEnvironmentBase`) + optional QEMU warmup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from resources_servers.cube.domains.base import CubeEnvironmentBase
from resources_servers.cube.domains.osworld.bootstrap import ensure_osworld_tasks


if TYPE_CHECKING:
    from resources_servers.cube.server import CubeResourcesServer


logger = logging.getLogger(__name__)


class OSWorldEnvironment(CubeEnvironmentBase):
    def ensure_loaded(self, server: CubeResourcesServer) -> None:
        ensure_osworld_tasks(server)

    def warm_on_startup(self, server: CubeResourcesServer) -> None:
        if not server.config.eager_osworld_vm_warmup:
            return
        self.ensure_loaded(server)
        n = len(server._task_configs_list)
        if n == 0:
            logger.warning("OSWorld VM warmup skipped: no tasks loaded")
            return
        idx = server.config.eager_osworld_warmup_task_idx
        if idx >= n:
            raise ValueError(
                f"eager_osworld_warmup_task_idx={idx} out of range for {n} loaded task(s) (valid: 0..{n - 1})"
            )
        logger.info(
            "OSWorld VM warmup: disposable reset for task_idx=%s (%d tasks loaded); "
            "QEMU boot + task setup may take several minutes...",
            idx,
            n,
        )
        task_config = server._task_configs_list[idx]
        task = task_config.make()
        try:
            task.reset()
        finally:
            task.close()
        logger.info("OSWorld VM warmup finished (HTTP server will start next).")

    def empty_reset_obs_detail(self) -> str:
        return "OSWorld reset returned no observations"
