# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WebArena specialization of the common visual-browser session manager."""

from __future__ import annotations

from nemo_gym.web.models import WebActionProfile, WebBenchmark, WebRuntimeProfile, WebTask
from nemo_gym.web.session_manager import WebSessionManager
from resources_servers.webarena_browser.backend import webarena_backend_factory
from resources_servers.webarena_browser.config import WebArenaBrowserResourcesServerConfig


WEBARENA_VERIFIER_PROFILES = {
    WebBenchmark.WEBARENA: "webarena_classic",
    WebBenchmark.VISUALWEBARENA: "visualwebarena_classic",
}


class WebArenaBrowserSessionManager(WebSessionManager):
    def __init__(self, config: WebArenaBrowserResourcesServerConfig) -> None:
        super().__init__(config, backend_factory=webarena_backend_factory)

    def _validate_task(self, task: WebTask) -> None:
        super()._validate_task(task)
        if task.runtime_profile != WebRuntimeProfile.VISUAL_BROWSER:
            raise ValueError("webarena_browser requires runtime_profile=visual_browser")
        if task.action_profile != WebActionProfile.COMPUTER_USE:
            raise ValueError("webarena_browser requires action_profile=computer_use")
        if task.benchmark not in WEBARENA_VERIFIER_PROFILES:
            raise ValueError("webarena_browser only accepts WebArena-family tasks")
        expected_verifier = WEBARENA_VERIFIER_PROFILES[task.benchmark]
        if task.verifier_profile != expected_verifier:
            raise ValueError(
                f"webarena_browser requires verifier_profile={expected_verifier} for benchmark={task.benchmark.value}"
            )
