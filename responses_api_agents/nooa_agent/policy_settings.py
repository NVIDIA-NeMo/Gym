# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The policy model as NOOA must know it: every setting a Gym run has to declare before an agent starts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MediaInput = Literal["image", "audio", "video", "pdf"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "max"]

# Request fields a benchmark's own `responses_create_params` may set; they win over the declared defaults.
REQUEST_SAMPLING_FIELDS = frozenset({"temperature", "top_p", "max_output_tokens", "reasoning"})


class PolicySettings(BaseModel):
    """What the agent server declares about the policy model, and sends on every model call.

    NOOA plans context and recovery from these values and the provider never guesses them, so a run that does
    not declare all of them fails when its config loads instead of running with a model it does not understand.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_window: int = Field(
        gt=0, description="The served context window in tokens; NOOA's context planner budget."
    )
    max_output_tokens: int = Field(
        gt=0, description="Reply cap sent as max_output_tokens on every model call the request does not cap itself."
    )
    temperature: float = Field(
        ge=0, description="Sampling temperature sent on every model call the request does not set."
    )
    top_p: float = Field(gt=0, le=1, description="Nucleus sampling sent on every model call the request does not set.")
    reasoning_effort: ReasoningEffort = Field(
        description=(
            "Reasoning effort sent as reasoning.effort on every model call the request does not set; "
            "`none` sends no reasoning setting, for a model or route that has none."
        )
    )
    media_inputs: tuple[MediaInput, ...] = Field(
        description=(
            "Media the policy model accepts, as NOOA's media_inputs; an empty list is text only and NOOA "
            "replaces other media with a short note instead of sending it."
        )
    )

    def request_defaults(self) -> dict[str, Any]:
        """Responses request fields that carry these settings when the request does not set them itself."""
        defaults: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort != "none":
            defaults["reasoning"] = {"effort": self.reasoning_effort}
        return defaults
