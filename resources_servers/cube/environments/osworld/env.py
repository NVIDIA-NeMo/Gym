# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``environment: osworld`` — task load (:func:`ensure_osworld_tasks`), :class:`OSWorldEnvironment`, optional VM warmup before HTTP."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional

from resources_servers.cube.environments.base import CubeEnvironmentBase
from resources_servers.cube.host_tools import require_qemu_img_if_qemu_backend
from resources_servers.cube.server import CubeResourcesServer


logger = logging.getLogger(__name__)


def _import_class(dotted_path: str) -> type:
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _parse_test_set(name: str) -> Any:
    from osworld_cube.benchmark import OSWorldTestSet

    s = name.strip()
    if s.endswith(".json"):
        for member in OSWorldTestSet:
            if member.value == s:
                return member
        raise ValueError(f"Unknown OSWorld test set file: {s}")
    key = s.upper()
    if not key.startswith("TEST_"):
        key = f"TEST_{key}"
    return OSWorldTestSet[key]


def _cfg_extra(server: CubeResourcesServer) -> Dict[str, Any]:
    return server.config.model_extra or {}


def ensure_osworld_tasks(server: CubeResourcesServer) -> None:
    """Populate ``server._task_configs_list`` from ``osworld_cube`` (lazy import)."""
    if server._adapter_state is not None:
        return

    extra = _cfg_extra(server)
    vm_backend_class = extra.get("vm_backend_class", "osworld_cube.vm_backend.OSWorldQEMUVMBackend")
    require_qemu_img_if_qemu_backend(vm_backend_class)

    from cube_computer_tool.computer import ActionSpace
    from osworld_cube.benchmark import OSWorldBenchmark
    from osworld_cube.computer import ComputerConfig

    vm_backend_kwargs = extra.get("vm_backend_kwargs") or {}
    action_space_raw = extra.get("action_space", "computer_13")
    if isinstance(action_space_raw, str):
        action_space = ActionSpace(action_space_raw)
    else:
        action_space = action_space_raw

    computer = ComputerConfig(
        action_space=action_space,
        require_a11y_tree=extra.get("require_a11y_tree", True),
        require_terminal=extra.get("require_terminal", False),
        observe_after_action=extra.get("observe_after_action", True),
    )

    vm_cls = _import_class(vm_backend_class)
    vm_backend = vm_cls(**vm_backend_kwargs) if vm_backend_kwargs else vm_cls()

    test_set_name = extra.get("test_set_name", "TEST_SMALL")
    tasks_file: Optional[str] = extra.get("tasks_file")

    bench_kw: Dict[str, Any] = {
        "default_tool_config": computer,
        "use_som": extra.get("use_som", False),
        "vm_backend": vm_backend,
    }
    if tasks_file:
        bench_kw["tasks_file"] = tasks_file
    else:
        bench_kw["test_set_name"] = _parse_test_set(test_set_name)

    benchmark = OSWorldBenchmark(**bench_kw)
    benchmark.setup()
    server._adapter_state = benchmark
    server._task_configs_list = list(benchmark.get_task_configs())
    logger.info("Cube OSWorld adapter loaded %s tasks", len(server._task_configs_list))


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
