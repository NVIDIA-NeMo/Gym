# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the dedicated native WebVoyager browser runtime."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field

from nemo_gym.web.models import WebBenchmark
from nemo_gym.web.native_browser import NativeBrowserResourcesServerConfig


class WebVoyagerBrowserResourcesServerConfig(NativeBrowserResourcesServerConfig):
    browser_proxy_env: str = "WA_BROWSER_PROXY_SERVER"
    captcha_api_key_env: str = "CAPSOLVER_API_KEY"
    captcha_provider_env: str = "WA_CAPTCHA_PROVIDER"
    captcha_solver_env: str = "WA_CAPTCHA_SOLVER"
    require_captcha_solver: bool = False
    proxy_mode: Literal["webvoyager_domains", "always", "disabled"] = "webvoyager_domains"
    allowed_benchmarks: list[WebBenchmark] = Field(default_factory=lambda: [WebBenchmark.WEBVOYAGER])

    def browser_proxy(self) -> str:
        return os.environ.get(self.browser_proxy_env, "").strip()

    def captcha_api_key(self) -> str:
        return os.environ.get(self.captcha_api_key_env, "").strip()

    def captcha_solver(self) -> str:
        explicit = os.environ.get(self.captcha_solver_env, "").strip()
        if explicit:
            return explicit
        provider = os.environ.get(self.captcha_provider_env, "capsolver").lower()
        if self.captcha_api_key() and provider == "capsolver":
            return "builtin:capsolver"
        return ""
