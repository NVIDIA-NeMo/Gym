# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from resources_servers.visgym.task_data import VisGymTaskFields


class VisGymTaskRow(VisGymTaskFields):
    """One task selected for a session; discard unrelated dataset/dispatcher fields."""

    model_config = ConfigDict(extra="ignore")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        ...,
        description="OpenAI Responses-API request body; input is filled by /seed_session.",
    )


class VisGymResourcesServerConfig(BaseResourcesServerConfig):
    reward_mode: Literal["training", "task_success"] = Field(
        default="training",
        description="Training returns accumulated native reward; evaluation returns audited binary success.",
    )

    """Server-level config for the VisGym resources server."""

    num_workers: Literal[1] | None = Field(
        default=None,
        description="Stateful sessions are process-local, so this server must run with at most one worker.",
    )

    task_jsonl_fpaths: list[str] = Field(
        default_factory=list,
        description="Ordered JSONL files concatenated into the in-memory task table.",
    )
    image_format: Literal["PNG", "JPEG"] = "PNG"
    image_jpeg_quality: int = Field(default=90, ge=1, le=100)
    skip_images: bool = Field(
        default=False,
        description="When True, omit input_image parts from observation messages.",
    )
    enforce_horizon_cap: bool = True
    require_audited_binary_success: bool = Field(
        default=True,
        description=(
            "Reject registry IDs outside VisGym's 17 audited binary-terminal families. "
            "Disable only for reward-only diagnostics; unsupported rows then return "
            "task_success=null."
        ),
    )
    return_transitions: Literal[False] = False
    include_env_feedback: bool = Field(
        default=True,
        description="Include info['env_feedback'] as visible text after each step.",
    )
    render_on_missing_image: bool = Field(
        default=True,
        description="Call env.render() when reset/step observation is not image-like.",
    )
    env_op_threadpool_size: int = Field(
        default=128,
        ge=1,
        description=(
            "Capacity of the anyio worker-thread pool that env.reset/step/render/close "
            "and env construction run through. Starlette's run_in_threadpool defaults to "
            "anyio's process-wide limiter (40 tokens), which becomes the throughput ceiling "
            "for this single-process, stateful server well before generation capacity does. "
            "Raise with node CPU headroom in mind -- each concurrent op is CPU-bound "
            "(physics/rendering), not I/O-bound, so tokens beyond available cores just queue."
        ),
    )
    cap_render_lib_threads: bool = Field(
        default=True,
        description=(
            "Set OpenCV's internal thread count to 1 at startup. With env_op_threadpool_size "
            "raised well above OpenCV's own default parallelism, each of many concurrent "
            "render calls spawning its own OpenCV thread pool causes CPU oversubscription "
            "that can net-negative throughput instead of improving it."
        ),
    )


class VisGymEnvStateEasyInputMessage(NeMoGymEasyInputMessage):
    """User-role message with server-side env metadata for inspection."""

    env_info: dict[str, Any] | None = Field(default=None)


class VisGymSeedSessionRequest(BaseSeedSessionRequest):
    env_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "Optional client-generated session UUID. Supplying it lets the caller "
            "clean up an ambiguously completed /seed_session request without "
            "replaying the mutation."
        ),
    )
    task_idx: int | None = Field(default=None, ge=0)
    task_row: VisGymTaskRow | None = Field(default=None)

    @model_validator(mode="after")
    def validate_task_selector(self) -> Self:
        if self.task_idx is None and self.task_row is None:
            raise ValueError("Either task_row or task_idx must be provided.")
        return self


class VisGymSeedSessionResponse(BaseSeedSessionResponse):
    env_id: str = Field(..., description="Server-issued session UUID.")
    obs: list[VisGymEnvStateEasyInputMessage]


class VisGymStepRequest(BaseModel):
    """Plain-text action extracted from the model's boxed final answer."""

    model_config = ConfigDict(extra="forbid")

    env_id: str
    action_string: str = Field(
        ...,
        description="Action string extracted by visgym_agent from \\boxed{...}.",
    )


class VisGymStepResponse(BaseModel):
    obs: list[VisGymEnvStateEasyInputMessage]
    reward: float
    done: bool
    horizon_terminated: bool = False


class VisGymCloseRequest(BaseModel):
    env_id: str
    discard_reward: bool = Field(
        default=False,
        description=(
            "Discard the accumulated training reward when aborting a rollout. "
            "Normal episode close leaves it available for the subsequent /verify drain."
        ),
    )


class VisGymCloseResponse(BaseModel):
    success: bool
    message: str = ""


class VisGymNeMoGymResponse(NeMoGymResponse):
    env_id: str
    group_id: str | None = None
    contains_transitions: bool = False
    seed_obs: list[VisGymEnvStateEasyInputMessage] | None = Field(default=None)
    output: list[NeMoGymResponseOutputItem] | list[list[NeMoGymResponseOutputItem]]


class VisGymAgentVerifyRequest(BaseVerifyRequest):
    """Verify request for a VisGym episode.

    Inherits ``responses_create_params`` from the base contract rather than
    declaring only ``response``. NeMo-RL reads that field straight off the
    rollout result when it rebuilds the initial prompt, so an environment that
    drops it fails postprocessing with ``KeyError: responses_create_params``
    after the episode has already been played.
    """

    response: VisGymNeMoGymResponse


class VisGymAgentVerifyResponse(BaseVerifyResponse):
    response: VisGymNeMoGymResponse
    task_success: bool | None = Field(
        default=None,
        description=(
            "True only when an audited VisGym family terminates with raw reward 1.0; null for "
            "unsupported IDs when strict success validation is disabled."
        ),
    )
