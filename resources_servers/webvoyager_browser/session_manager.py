# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session manager for the dedicated native WebVoyager browser."""

from __future__ import annotations

from nemo_gym.web.models import WebActionProfile, WebBenchmark, WebRuntimeProfile, WebTask
from nemo_gym.web.session_manager import WebSessionManager
from resources_servers.webvoyager_browser.backend import webvoyager_backend_factory
from resources_servers.webvoyager_browser.config import WebVoyagerBrowserResourcesServerConfig


class WebVoyagerBrowserSessionManager(WebSessionManager):
    def __init__(self, config: WebVoyagerBrowserResourcesServerConfig) -> None:
        super().__init__(config, backend_factory=webvoyager_backend_factory)

    def _validate_task(self, task: WebTask) -> None:
        super()._validate_task(task)
        if task.benchmark != WebBenchmark.WEBVOYAGER:
            raise ValueError("webvoyager_browser only accepts benchmark=webvoyager")
        if task.runtime_profile != WebRuntimeProfile.NATIVE_VISUAL:
            raise ValueError("webvoyager_browser requires runtime_profile=native_visual")
        if task.action_profile != WebActionProfile.NATIVE_TOOLCALL:
            raise ValueError("webvoyager_browser requires action_profile=native_toolcall")
        if task.verifier_profile != "native_webvoyager_gemini":
            raise ValueError("webvoyager_browser requires verifier_profile=native_webvoyager_gemini")
