# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral configuration for stateful web resource servers."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, model_validator

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.web.models import WebBenchmark


class WebResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration shared by BrowserGym and native visual backends."""

    max_sessions: int = Field(default=1, ge=1)
    artifact_dir: str = "cache/web/artifacts"
    inline_screenshots: bool = True
    max_evidence_screenshots: int = Field(default=3, ge=1, le=20)
    session_ttl_seconds: int = Field(default=3600, ge=60)
    reaper_interval_seconds: int = Field(default=60, ge=5)
    require_auth: bool = False
    auth_token_env: str = "BROWSERGYM_WEB_RESOURCES_TOKEN"
    site_pool_mode: Literal["unmanaged", "local_locks"] = "unmanaged"
    allowed_benchmarks: list[WebBenchmark] = Field(default_factory=lambda: list(WebBenchmark))

    @model_validator(mode="after")
    def validate_stateful_server(self) -> "WebResourcesServerConfig":
        if self.num_workers not in {None, 1}:
            raise ValueError("web sessions are process-local; num_workers must be 1")
        if len(set(self.allowed_benchmarks)) != len(self.allowed_benchmarks):
            raise ValueError("allowed_benchmarks must not contain duplicates")
        return self

    def resolved_artifact_dir(self) -> str:
        return os.path.abspath(os.path.expanduser(self.artifact_dir))

    def auth_token(self) -> str:
        return os.environ.get(self.auth_token_env, "").strip()
