# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Strict models for Gym's supported ATIF v1.7 subset.

This is deliberately not a general ATIF implementation. It describes the
fields consumed and produced by Gym's ATIF adapters and rejects unknown
structural fields so schema drift cannot silently alter their behavior. Provider
payloads and extension metadata remain JSON objects at their declared extension
points.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ATIF_SCHEMA_VERSION = "ATIF-v1.7"


class _AtifModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AtifImageSource(_AtifModel):
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    path: str


class AtifContentPart(_AtifModel):
    """Validated ATIF text or image content part."""

    type: Literal["text", "image"]
    text: str | None = None
    source: AtifImageSource | None = None

    @model_validator(mode="after")
    def validate_content_type(self) -> "AtifContentPart":
        if self.type == "text":
            if self.text is None:
                raise ValueError("text content parts require text")
            if self.source is not None:
                raise ValueError("text content parts cannot contain an image source")
        else:
            if self.source is None:
                raise ValueError("image content parts require source")
            if self.text is not None:
                raise ValueError("image content parts cannot contain text")
        return self


AtifContent = str | list[AtifContentPart]


class AtifAgent(_AtifModel):
    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class AtifToolCall(_AtifModel):
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]
    extra: dict[str, Any] | None = None


class AtifObservationResult(_AtifModel):
    source_call_id: str | None = None
    content: AtifContent | None = None
    subagent_trajectory_ref: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class AtifObservation(_AtifModel):
    results: list[AtifObservationResult]


class AtifStepMetrics(_AtifModel):
    prompt_tokens: int | None = Field(default=None, ge=0, strict=True)
    completion_tokens: int | None = Field(default=None, ge=0, strict=True)
    cached_tokens: int | None = Field(default=None, ge=0, strict=True)
    cost_usd: float | None = Field(default=None, ge=0, strict=True)
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, Any] | None = None

    @field_validator("prompt_token_ids", "completion_token_ids", mode="before")
    @classmethod
    def validate_token_id_types(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
        ):
            raise ValueError("token IDs must be non-negative JSON integer arrays")
        return value

    @field_validator("logprobs", mode="before")
    @classmethod
    def validate_logprob_types(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list) or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value
        ):
            raise ValueError("log probabilities must be finite JSON number arrays")
        return value


class AtifFinalMetrics(_AtifModel):
    total_prompt_tokens: int | None = Field(default=None, ge=0, strict=True)
    total_completion_tokens: int | None = Field(default=None, ge=0, strict=True)
    total_cached_tokens: int | None = Field(default=None, ge=0, strict=True)
    total_cost_usd: float | None = Field(default=None, ge=0, strict=True)
    total_steps: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] | None = None


class AtifStep(_AtifModel):
    step_id: int = Field(ge=1, strict=True)
    source: Literal["system", "user", "agent"]
    message: AtifContent
    timestamp: str | None = None
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    reasoning_content: str | None = None
    tool_calls: list[AtifToolCall] | None = None
    observation: AtifObservation | None = None
    metrics: AtifStepMetrics | None = None
    llm_call_count: int | None = Field(default=None, ge=0, strict=True)
    is_copied_context: bool | None = None
    extra: dict[str, Any] | None = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"invalid ISO 8601 timestamp: {exc}") from exc
        return value

    @model_validator(mode="after")
    def validate_agent_only_fields(self) -> "AtifStep":
        if self.source == "agent":
            return self
        for field_name in (
            "model_name",
            "reasoning_effort",
            "reasoning_content",
            "tool_calls",
            "metrics",
        ):
            if getattr(self, field_name) is not None:
                raise ValueError(f"field {field_name!r} is only valid for agent steps")
        return self

    @model_validator(mode="after")
    def validate_zero_llm_call_count(self) -> "AtifStep":
        if self.llm_call_count == 0 and any(
            value is not None for value in (self.reasoning_effort, self.reasoning_content, self.metrics)
        ):
            raise ValueError("llm_call_count=0 cannot include reasoning or LLM metrics")
        return self


class AtifTrajectoryV1_7(_AtifModel):
    """The version-gated ATIF v1.7 subset supported by Gym's adapters."""

    schema_version: str
    session_id: str | None = None
    trajectory_id: str | None = None
    agent: AtifAgent
    steps: list[AtifStep] = Field(min_length=1)
    notes: str | None = None
    final_metrics: AtifFinalMetrics | None = None
    continued_trajectory_ref: str | None = None
    # Projection rejects subagents, so their recursive shape is intentionally
    # not duplicated here.
    subagent_trajectories: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_step_ids_and_tool_references(self) -> "AtifTrajectoryV1_7":
        for index, step in enumerate(self.steps):
            expected_step_id = index + 1
            if step.step_id != expected_step_id:
                raise ValueError(
                    f"steps[{index}].step_id: expected {expected_step_id} (sequential from 1), got {step.step_id}"
                )

            call_ids = {call.tool_call_id for call in step.tool_calls or []}
            for result in step.observation.results if step.observation is not None else []:
                if result.source_call_id is not None and result.source_call_id not in call_ids:
                    raise ValueError(
                        f"step {step.step_id} observation references unknown tool call {result.source_call_id!r}"
                    )
        return self
