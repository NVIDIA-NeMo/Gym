# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unmodified OSWorld ``DesktopEnv`` wired to Gym Sandbox lifecycle."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Mapping
from typing import Any

import desktop_env.desktop_env as desktop_env_module
from desktop_env.desktop_env import DesktopEnv
from desktop_env.providers.docker.manager import DockerVMManager

from responses_api_agents.osworld_agent.sandbox_provider import GymSandboxDesktopProvider


_FACTORY_LOCK = threading.RLock()


class SandboxDesktopEnv(DesktopEnv):
    """DesktopEnv whose OSWorld provider contract is fulfilled by Gym Sandbox.

    The pinned OSWorld constructor imports its provider factory directly. A
    worker-local, lock-protected injection preserves that constructor and all
    downstream controller/evaluator behavior while substituting only the
    manager/provider pair. The original factory is restored on every path.
    """

    def __init__(
        self,
        *args: Any,
        sandbox_provider: Mapping[str, Any],
        sandbox_spec: Mapping[str, Any],
        sandbox_require_kvm: bool = True,
        sandbox_ready_timeout_s: float = 600.0,
        sandbox_ready_poll_s: float = 2.0,
        **kwargs: Any,
    ) -> None:
        provider = GymSandboxDesktopProvider(
            sandbox_provider,
            sandbox_spec,
            require_kvm=sandbox_require_kvm,
            ready_timeout_s=sandbox_ready_timeout_s,
            ready_poll_s=sandbox_ready_poll_s,
        )
        manager = DockerVMManager()
        requested_provider = str(kwargs.get("provider_name", "docker")).lower().strip()
        if requested_provider != "docker":
            raise ValueError(
                "SandboxDesktopEnv uses OSWorld's 'docker' compatibility semantics; "
                f"provider_name must be 'docker', got {requested_provider!r}"
            )
        kwargs["provider_name"] = "docker"

        original_factory = desktop_env_module.create_vm_manager_and_provider

        def sandbox_factory(
            provider_name: str,
            region: str | None,
            use_proxy: bool = False,
            provider_options: Mapping[str, Any] | None = None,
        ) -> tuple[DockerVMManager, GymSandboxDesktopProvider]:
            del region, use_proxy, provider_options
            if provider_name.lower().strip() != "docker":
                raise ValueError(f"SandboxDesktopEnv received unexpected provider {provider_name!r}")
            return manager, provider

        with _FACTORY_LOCK:
            desktop_env_module.create_vm_manager_and_provider = sandbox_factory
            try:
                super().__init__(*args, **kwargs)
            except BaseException:
                # Do not hide the DesktopEnv construction error if cleanup
                # encounters an independent Docker failure.
                with contextlib.suppress(Exception):
                    provider.stop_emulator(str(kwargs.get("path_to_vm") or ""))
                raise
            finally:
                desktop_env_module.create_vm_manager_and_provider = original_factory
