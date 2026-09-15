# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session manager for the dedicated WebVoyager visual browser."""

from __future__ import annotations

from nemo_gym.web.models import WebActionProfile, WebBenchmark, WebRuntimeProfile, WebTask
from nemo_gym.web.session_manager import WebSessionManager
from resources_servers.visual_browser.backend import visual_browser_backend_factory
from resources_servers.visual_browser.config import VisualBrowserResourcesServerConfig


class VisualBrowserSessionManager(WebSessionManager):
    def __init__(self, config: VisualBrowserResourcesServerConfig) -> None:
        super().__init__(config, backend_factory=visual_browser_backend_factory)

    def _validate_task(self, task: WebTask) -> None:
        super()._validate_task(task)
        if task.benchmark != WebBenchmark.WEBVOYAGER:
            raise ValueError("visual_browser only accepts benchmark=webvoyager")
        if task.runtime_profile != WebRuntimeProfile.VISUAL_BROWSER:
            raise ValueError("visual_browser requires runtime_profile=visual_browser")
        if task.action_profile != WebActionProfile.COMPUTER_USE:
            raise ValueError("visual_browser requires action_profile=computer_use")
        if task.verifier_profile != "webvoyager_gemini":
            raise ValueError("visual_browser requires verifier_profile=webvoyager_gemini")
