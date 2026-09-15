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
"""No-result outcomes for callers that opt into Gym's structured failure API.

Completed results retain their existing dictionaries, including valid reward-zero and
masked results. A failure carries identifiers and bounded diagnostics, never generation
data. It describes an attempt; retry and replacement decisions belong to the caller.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.failure_kinds import validate_failure_kind


class RolloutFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["failure"] = "failure"
    rollout_id: str = Field(min_length=1)
    run_id: str | None = None
    attempt_index: int = Field(default=0, ge=0, strict=True)
    failure_kind: str = Field(min_length=1)
    stage: Literal["request", "response", "result", "agent", "verifier"]
    failure_reason: str
    exception_type: str | None = None
    http_status: int | None = None
    response_body: str | None = None
    terminal: bool = False

    @property
    def attempt_id(self) -> str:
        return f"{self.rollout_id}-a{self.attempt_index}"

    @field_validator("failure_kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        return validate_failure_kind(value)

    @field_validator("failure_reason", "exception_type", "response_body")
    @classmethod
    def _bound_diagnostics(cls, value: str | None) -> str | None:
        return value[:2000] if value is not None else None


class InvalidRolloutResult(ValueError):
    """An agent returned a body that cannot represent a completed result."""
