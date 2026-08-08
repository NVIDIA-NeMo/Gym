# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP models for the BrowserGym web resource server."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.web.models import WebAction, WebObservation, WebStepResult, WebTask, WebVerifierResult


class WebSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    task: WebTask


class WebResetRequest(BaseModel):
    task: WebTask


class WebSeedSessionResponse(BaseSeedSessionResponse):
    session_id: str
    task_id: str
    status: str
    observation: WebObservation
    info: dict[str, Any] = Field(default_factory=dict)


class WebStepRequest(BaseModel):
    operation_id: str
    action: WebAction


class WebStepResponse(WebStepResult):
    operation_id: str


class WebEvaluateRequest(BaseModel):
    final_answer: Optional[str] = None


class WebEvaluateResponse(BaseModel):
    result: WebVerifierResult


class WebCloseResponse(BaseModel):
    closed: bool


class WebSessionStatusResponse(BaseModel):
    session_id: str
    task_id: str
    benchmark: str
    status: str
    created_at: float
    last_access_at: float
    site_lease_id: str


class WebVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    final_answer: Optional[str] = None


class WebVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    raw_score: float = 0.0
    task_success: bool = False
    mask_sample: bool = False
    failure_kind: Optional[str] = None
