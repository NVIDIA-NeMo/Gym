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

"""Request and result models for long-conversation simulation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse


class ConversationBatch(BaseModel):
    """One session of the conversation: its pre-generated user messages and plan context."""

    model_config = ConfigDict(extra="allow")

    messages: list[str]
    current_plan: str = ""
    previous_plans: str = ""
    time_anchor: str | None = None


class LongConversationRunRequest(BaseRunRequest):
    """One conversation per rollout. `responses_create_params` is unused by the simulation and may
    hold an empty input list; it carries the rollout's identity like any Gym task."""

    model_config = ConfigDict(extra="allow")

    topic: str
    theme: str
    batches: list[ConversationBatch] = Field(min_length=1)
    previous_plans_summary: str = ""


class SimulationStats(BaseModel):
    batches_requested: int
    batches_completed: int
    stopped_at_batch: int | None = None
    skipped_questions: list[dict[str, Any]] = Field(default_factory=list)
    interrupted_turns: list[dict[str, Any]] = Field(default_factory=list)
    calls: dict[str, int] = Field(default_factory=dict)
    failed_calls: dict[str, int] = Field(default_factory=dict)


class LongConversationVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    chat: list[list[dict[str, Any]]]
    simulation: SimulationStats
