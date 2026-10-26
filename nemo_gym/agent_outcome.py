# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-reported causes, independent of environment grading and retry policy."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.failure_kinds import FAILURE_KINDS, MODEL_REQUEST_ERROR


AGENT_OUTCOME_METADATA_KEY = "ng_agent_outcome"


class AgentOutcome(BaseModel):
    """Additional evidence distinguishing causes of a failed agent response.

    Ordinary completion and limits already use response.status. HTTP status here
    belongs to the upstream model request, not the agent endpoint's HTTP 200.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    failure_kind: str
    status_code: int | None = Field(default=None, ge=400, le=599, strict=True)

    @field_validator("failure_kind")
    @classmethod
    def require_registered_kind(cls, value: str) -> str:
        """Accept the shared vocabulary plus an explicitly undetermined cause."""
        if value != "unknown" and value not in FAILURE_KINDS:
            raise ValueError(f"Unregistered failure kind: {value}")
        return value

    @model_validator(mode="after")
    def require_request_status(self) -> "AgentOutcome":
        if (self.failure_kind == MODEL_REQUEST_ERROR) != (self.status_code is not None):
            raise ValueError("Only model_request_error requires an upstream HTTP status")
        return self
