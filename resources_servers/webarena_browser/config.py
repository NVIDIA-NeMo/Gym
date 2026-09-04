# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the WebArena visual-browser resource server."""

from __future__ import annotations

from pydantic import Field

from nemo_gym.web.models import WebBenchmark
from nemo_gym.web.visual_browser import VisualBrowserDriverConfig


class WebArenaBrowserResourcesServerConfig(VisualBrowserDriverConfig):
    """Shared visual-browser mechanics plus WebArena-family site policy."""

    artifact_dir: str = "cache/webarena-browser/artifacts"
    allowed_benchmarks: list[WebBenchmark] = Field(
        default_factory=lambda: [WebBenchmark.WEBARENA, WebBenchmark.VISUALWEBARENA]
    )
