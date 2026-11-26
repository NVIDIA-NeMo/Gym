# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SaaS activation and verification wire contracts."""

from pydantic import ConfigDict, Field

from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse


class SaasBenchRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, str]


class SaasBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    task_id: str
    category_id: str
    artifacts: str
    verify_status: str
    trajectory: list[dict] = Field(default_factory=list)
    native_trajectory: list[dict] = Field(default_factory=list)
    execution_health: dict = Field(default_factory=dict)
    harness_version: str | None = None
    verifier_diagnostics: dict = Field(default_factory=dict)
    execution_error: str | None = None
