# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-reported causes, independent of environment grading and retry policy."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AGENT_OUTCOME_METADATA_KEY = "ng_agent_outcome"


class AgentOutcome(BaseModel):
    """Additional evidence distinguishing causes of a failed agent response.

    Ordinary completion and limits already use response.status. HTTP status here
    belongs to the upstream model request, not the agent endpoint's HTTP 200.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    failure_kind: Literal["model_request_error", "model_transport_error", "invalid_output", "limit", "unknown"]
    status_code: int | None = Field(default=None, ge=400, le=599, strict=True)

    @model_validator(mode="after")
    def require_request_status(self) -> "AgentOutcome":
        if (self.failure_kind == "model_request_error") != (self.status_code is not None):
            raise ValueError("Only model_request_error requires an upstream HTTP status")
        return self
