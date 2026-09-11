# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire models for stateful, multimodal web environments."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Wire value for ``WebStepResult.info["runtime_status"]`` when a browser runtime
# has spent its CAPTCHA solving budget. The rollout is then a site-access
# failure rather than a policy outcome, so agents mask it instead of judging it.
CAPTCHA_BUDGET_EXHAUSTED_STATUS = "captcha_budget_exhausted"
# A Playwright page/context/browser closed while the CAPTCHA integration was
# inspecting it. This is a retryable browser lifecycle failure, not a paid
# solver failure and not a policy outcome.
BROWSER_TARGET_CLOSED_STATUS = "browser_target_closed"


class WebBenchmark(StrEnum):
    WEBARENA = "webarena"
    VISUALWEBARENA = "visualwebarena"
    WEBVOYAGER = "webvoyager"


class WebRuntimeProfile(StrEnum):
    """Browser execution protocol exposed by Gym.

    Model-specific prompting is deliberately not represented here. Nano Omni,
    Qwen, and future policies all drive the same visual browser runtime after
    their policy adapters normalize actions into ``WebAction``.
    """

    VISUAL_BROWSER = "visual_browser"


class WebObservationProfile(StrEnum):
    A11Y = "a11y"
    SCREENSHOT = "screenshot"
    SOM = "som"


class WebActionProfile(StrEnum):
    COMPUTER_USE = "computer_use"


class WebArtifactRef(BaseModel):
    """Stable reference to an artifact retained outside a model request."""

    uri: str
    mime_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WebImage(BaseModel):
    """An image can be transported inline, retained as an artifact, or both."""

    data_url: Optional[str] = None
    artifact: Optional[WebArtifactRef] = None

    @model_validator(mode="after")
    def require_transport(self) -> "WebImage":
        if not self.data_url and self.artifact is None:
            raise ValueError("a web image requires data_url or artifact")
        return self


class WebTab(BaseModel):
    index: int = Field(ge=0)
    url: str = ""
    title: str = ""
    active: bool = False


class WebTask(BaseModel):
    """Normalized task envelope; original benchmark fields remain lossless."""

    model_config = ConfigDict(extra="allow")

    benchmark: WebBenchmark
    task_id: str
    intent: str = ""
    start_urls: list[str] = Field(default_factory=list)
    sites: list[str] = Field(default_factory=list)
    input_images: list[str] = Field(default_factory=list)
    runtime_profile: WebRuntimeProfile = WebRuntimeProfile.VISUAL_BROWSER
    observation_profile: Optional[WebObservationProfile] = WebObservationProfile.SCREENSHOT
    action_profile: WebActionProfile = WebActionProfile.COMPUTER_USE
    verifier_profile: Optional[str] = None
    auth_profile: Optional[str] = None
    seed: int = 0
    task_kwargs: dict[str, Any] = Field(default_factory=dict)
    original_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id", mode="before")
    @classmethod
    def stringify_task_id(cls, value: Any) -> str:
        if value is None:
            raise ValueError("task_id is required")
        return str(value)


class WebObservation(BaseModel):
    """JSON-safe observation sent from the browser runtime to the agent."""

    goal: list[dict[str, Any]] = Field(default_factory=list)
    axtree_text: str = ""
    screenshot: Optional[WebImage] = None
    url: str = ""
    tabs: list[WebTab] = Field(default_factory=list)
    active_tab_index: int = Field(default=0, ge=0)
    element_map: dict[str, dict[str, Any]] = Field(default_factory=dict)
    focused_element_id: str = ""
    last_action: str = ""
    last_action_error: str = ""
    elapsed_time: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebAction(BaseModel):
    """Model-independent computer-use action accepted by visual browsers."""

    name: str
    script: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    terminal: bool = False
    answer: Optional[str] = None
    raw_model_output: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebStepResult(BaseModel):
    observation: WebObservation
    execution_ok: bool
    benchmark_reward: Optional[float] = None
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class WebVerifierResult(BaseModel):
    reward: float = 0.0
    raw_score: float = 0.0
    task_success: bool = False
    valid_sample: bool = True
    failure_kind: Optional[str] = None
    subscores: dict[str, float] = Field(default_factory=dict)
    evidence: list[WebArtifactRef] = Field(default_factory=list)
    verifier_version: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
