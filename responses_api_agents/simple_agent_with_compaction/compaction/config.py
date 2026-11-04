# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration models for semantic context compaction."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageRecencyConfig(BaseModel):
    """Retention controls for image observation groups."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    protect_initial_context: bool = True
    keep_last_groups: int = Field(default=3, ge=0)
    omission_marker: str | None = "[Earlier image omitted]"

    @model_validator(mode="after")
    def validate_disabled_configuration(self) -> "ImageRecencyConfig":
        configured_fields = self.model_fields_set - {"enabled"}
        if not self.enabled and configured_fields:
            raise ValueError(
                f"Image recency settings require images.enabled=true: configured={sorted(configured_fields)}"
            )
        return self


class ReasoningRecencyConfig(BaseModel):
    """Retention controls for reasoning blocks grouped by model-call turn."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    keep_first_block: bool = False
    keep_last_blocks: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def validate_disabled_configuration(self) -> "ReasoningRecencyConfig":
        configured_fields = self.model_fields_set - {"enabled"}
        if not self.enabled and configured_fields:
            raise ValueError(
                f"Reasoning recency settings require reasoning.enabled=true: configured={sorted(configured_fields)}"
            )
        return self


class RecencyHistoryPolicyConfig(BaseModel):
    """Composable per-kind recency configuration."""

    model_config = ConfigDict(extra="forbid")

    images: ImageRecencyConfig = Field(default_factory=ImageRecencyConfig)
    reasoning: ReasoningRecencyConfig = Field(default_factory=ReasoningRecencyConfig)

    @model_validator(mode="after")
    def validate_at_least_one_kind(self) -> "RecencyHistoryPolicyConfig":
        if not self.images.enabled and not self.reasoning.enabled:
            raise ValueError("The recency policy requires images.enabled or reasoning.enabled")
        return self


class HistoryPolicyConfig(BaseModel):
    """Select a built-in or agent-registered history policy."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="identity", min_length=1)
    config: RecencyHistoryPolicyConfig | dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_builtin_policy_config(self) -> "HistoryPolicyConfig":
        if self.type == "recency" and not isinstance(self.config, RecencyHistoryPolicyConfig):
            self.config = RecencyHistoryPolicyConfig.model_validate(self.config)
        return self


class CompactionScheduleConfig(BaseModel):
    """Choose when an already-selected historical base may be rewritten."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["rolling_recency", "turn_chunked_recency"] = "rolling_recency"
    actions_per_chunk: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_actions_per_chunk(self) -> "CompactionScheduleConfig":
        if self.type == "rolling_recency" and self.actions_per_chunk != 1:
            raise ValueError("rolling_recency requires actions_per_chunk=1")
        return self


class ContextGuardConfig(BaseModel):
    """Hard generation-admission limits evaluated only between turns."""

    model_config = ConfigDict(extra="forbid")

    max_total_tokens: int | None = Field(default=None, ge=1)
    reserved_generation_tokens: int = Field(default=0, ge=0)
    max_active_images: int | None = Field(default=None, ge=0)
    max_vision_tokens: int | None = Field(default=None, ge=0)
    projected_vision_tokens_per_image: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_vision_projection(self) -> "ContextGuardConfig":
        if self.max_vision_tokens is not None and self.projected_vision_tokens_per_image is None:
            raise ValueError("max_vision_tokens requires projected_vision_tokens_per_image")
        return self


class ContextHistoryConfig(BaseModel):
    """Capability-gated semantic context-history configuration for a Gym agent."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    policy: HistoryPolicyConfig = Field(default_factory=HistoryPolicyConfig)
    schedule: CompactionScheduleConfig = Field(default_factory=CompactionScheduleConfig)
    guards: ContextGuardConfig = Field(default_factory=ContextGuardConfig)
