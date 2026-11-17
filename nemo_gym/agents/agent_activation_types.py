# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed-time requirements for resuming a harness's saved session."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ContinuationObservation = Literal["trajectory", "timing", "harness_tool_calls", "reasoning", "compaction"]


class AgentContinuationRequirements(BaseModel):
    """Multi-turn continuation and observation support required before session setup."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["resume_harness_session"] = Field(
        default="resume_harness_session", description="Resume the harness's saved session across activations."
    )
    observations: list[ContinuationObservation] = Field(
        default_factory=lambda: ["trajectory", "timing"],
        description="Observation kinds required by the caller, or supported by the adapter in a capability response.",
    )


class AgentContinuationCapabilities(AgentContinuationRequirements):
    """Continuation mode and observations this adapter supports."""
