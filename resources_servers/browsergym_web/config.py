# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the BrowserGym web backend."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from nemo_gym.web.models import WebBenchmark
from nemo_gym.web.resource_config import WebResourcesServerConfig


class BrowserGymWebResourcesServerConfig(WebResourcesServerConfig):
    """BrowserGym-specific settings layered on the common web server contract."""

    artifact_dir: str = "cache/browsergym-web/artifacts"
    headless: bool = True
    tags_to_mark: Literal["all", "standard_html"] = "standard_html"
    pre_observation_delay: float = Field(default=0.5, ge=0.0, le=30.0)
    record_video: bool = False
    allowed_benchmarks: list[WebBenchmark] = Field(default_factory=lambda: [WebBenchmark.WEBVOYAGER])
