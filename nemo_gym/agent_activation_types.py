# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed-time requirements for retaining a harness's native conversation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ContinuationObservation = Literal["trajectory", "timing", "native_tools", "reasoning", "compaction"]


class AgentContinuationRequirements(BaseModel):
    """Native continuation and observation support required before session setup."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["native_conversation"] = "native_conversation"
    observations: list[ContinuationObservation] = Field(default_factory=lambda: ["trajectory", "timing"])


class AgentContinuationCapabilities(AgentContinuationRequirements):
    """Continuation mode and observations this adapter supports."""
