# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral HTTP models for stateful web resource servers.

These models live in the dependency-light web protocol package so agents do
not import a concrete browser or computer-control backend.
Resource servers may re-export them for compatibility, but all backends must
preserve this wire contract.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.web.models import (
    WebAction,
    WebArtifactRef,
    WebObservation,
    WebStepResult,
    WebTask,
    WebVerifierResult,
)


class WebSessionIdentity(BaseModel):
    """Caller-owned identity for retrying seed and closing without a cookie.

    Field aliases follow the proposed core session lifecycle contract. The
    capability never belongs in policy messages, rollout results or logs.
    Legacy cookie-only clients may omit both fields.
    """

    session_identity: str | None = Field(default=None, alias="_ng_session_id", pattern=r"^[a-zA-Z0-9_-]{16,128}$")
    close_token: SecretStr | None = Field(default=None, alias="_ng_session_close_token", min_length=32, max_length=256)

    @model_validator(mode="after")
    def require_identity_pair(self) -> "WebSessionIdentity":
        if (self.session_identity is None) != (self.close_token is None):
            raise ValueError("session identity and close capability must be supplied together")
        return self


class WebSeedSessionRequest(WebSessionIdentity, BaseSeedSessionRequest):
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
    session_id: Optional[str] = None
    recording_artifacts: list[WebArtifactRef] = Field(default_factory=list)
    failure_kind: str | None = None
    failure_reason: str | None = None


class WebSessionStatusResponse(BaseModel):
    session_id: str
    task_id: str
    benchmark: str
    status: str
    created_at: float
    last_access_at: float
    site_lease_id: str
    browser_lease_id: Optional[str] = None
    browser_provider: Optional[str] = None
    browser_transport: Optional[str] = None


class WebVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    final_answer: Optional[str] = None


class WebVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    raw_score: float = 0.0
    task_success: bool = False
    mask_sample: bool = False
    failure_kind: Optional[str] = None
    cleanup_failure_kind: str | None = None
    cleanup_failure_reason: str | None = None
