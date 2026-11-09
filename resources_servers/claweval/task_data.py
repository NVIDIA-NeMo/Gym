# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harness-independent ClawEval task references; no solutions or grader inputs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.server_runtime import BaseSeedSessionRequest, ResourcesVerifyRequest


class TaskData(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    task_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    prompt: str
    split: Literal["general", "multimodal", "multi_turn"]


class VerificationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation: list[dict]
    model_turns: int = Field(ge=1)


class VerifyRequest(ResourcesVerifyRequest[VerificationInput]):
    """Bind the grader request to the task and episode that own the native state."""


class VerifyResponse(BaseModel):
    """Keep the native score and retained evidence beside Gym's failure fields."""

    model_config = ConfigDict(extra="allow")

    reward: float
    mask_sample: bool = False
    failure_kind: str | None = None
    failure_reason: str | None = None
    reward_components: dict[str, float]
