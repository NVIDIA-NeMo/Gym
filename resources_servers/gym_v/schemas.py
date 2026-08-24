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
from typing import Any, Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymEasyInputMessageForTraining,
    NeMoGymFunctionCallOutput,
    NeMoGymMessage,
    NeMoGymMessageForTraining,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseReasoningItemForTraining,
)


GymVNeMoGymResponseOutputItem: TypeAlias = Union[
    # Training variants must come first. Otherwise Pydantic validates a
    # token-bearing assistant message as the non-training base class and drops
    # prompt_token_ids/generation_token_ids/generation_log_probs.
    NeMoGymEasyInputMessageForTraining,
    NeMoGymMessageForTraining,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseReasoningItemForTraining,
    NeMoGymEasyInputMessage,
    NeMoGymMessage,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseFunctionToolCall,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseReasoningItem,
]


class GymVTaskRow(BaseModel):
    """One env-id-parametric task row loaded from a Gym-V JSONL file."""

    env_id: str = Field(..., description="Gym-V env ID, e.g. 'Games/FrozenLake-v0'.")
    env_kwargs: dict[str, Any] = Field(default_factory=dict)
    seed: int = Field(..., description="Seed passed to env.reset().")
    task_id: str | None = Field(
        default=None,
        description="Optional content-addressed identifier for reproducible lookup.",
    )
    horizon_cap: int | None = Field(
        default=None,
        ge=1,
        description="Optional server-enforced cap on rollout turns.",
    )
    task_metadata: dict[str, Any] = Field(default_factory=dict)
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        ...,
        description="OpenAI Responses-API request body; input is filled by /seed_session.",
    )


class GymVResourcesServerConfig(BaseResourcesServerConfig):
    """Server-level config for the Gym-V resources server."""

    task_jsonl_fpaths: list[str] = Field(
        default_factory=list,
        description=(
            "Deprecated v1 compatibility path: ordered JSONL files concatenated "
            "into the in-memory task table. New clients pass task_row to /seed_session."
        ),
    )
    image_format: Literal["PNG", "JPEG"] = "PNG"
    image_jpeg_quality: int = Field(default=90, ge=1, le=100)
    max_image_wh: int | None = Field(
        default=None,
        ge=1,
        description=(
            "If set, resize every rendered observation image so that "
            "max(W, H) == max_image_wh, preserving aspect ratio (both "
            "dimensions scaled by max_image_wh / max(W, H) via "
            "PIL.Image.thumbnail). Default null (no resize) — envs render "
            "at their native size. Set to a modest value (e.g. 512) to cap "
            "the visual-token budget on large-image envs and reduce vLLM "
            "context-overflow rejections."
        ),
    )
    skip_images: bool = Field(
        default=False,
        description=(
            "When True, observation messages omit all input_image content "
            "parts. Useful for ablation: if accuracy is unchanged without "
            "images, the model is not using the visual input."
        ),
    )
    enforce_horizon_cap: bool = True
    return_transitions: bool = Field(
        default=False,
        description="Pinned false for the flat trajectory shape consumed by the inspector.",
    )
    disable_text_feedback: bool = Field(
        default=False,
        description="Pass disable_text_feedback to gym_v.make() to strip post-step text and isolate visual reasoning. Per-row env_kwargs override.",
    )


class GymVEnvStateEasyInputMessage(NeMoGymEasyInputMessage):
    """User-role message with explicit server-side env metadata for inspection."""

    env_info: dict[str, Any] | None = Field(default=None)


class GymVSeedSessionRequest(BaseSeedSessionRequest):
    task_idx: int | None = Field(default=None, ge=0)
    task_row: GymVTaskRow | None = Field(
        default=None,
        description="Full task row for stateless server-side env instantiation.",
    )

    @model_validator(mode="after")
    def validate_task_selector(self) -> Self:
        if self.task_idx is None and self.task_row is None:
            raise ValueError("Either task_row or task_idx must be provided.")
        return self


class GymVSeedSessionResponse(BaseSeedSessionResponse):
    env_id: str = Field(..., description="Server-issued session UUID.")
    obs: list[GymVEnvStateEasyInputMessage]


class GymVStepRequest(BaseModel):
    """Path B action transport: a plain-text action string extracted from
    the model's `\\boxed{...}` output (Path A tool-call transport was
    removed)."""

    model_config = ConfigDict(extra="forbid")

    env_id: str
    action_string: str = Field(
        ...,
        description=(
            "Action string extracted by `gymv_agent` from the model's "
            "`\\boxed{...}` output. Must match the env's action grammar; the "
            "env's _score_answer / step path raises on invalid actions and the "
            "server returns a recovery message."
        ),
    )


class GymVStepResponse(BaseModel):
    obs: list[GymVEnvStateEasyInputMessage]
    reward: float
    done: bool
    horizon_terminated: bool = False


class GymVCloseRequest(BaseModel):
    env_id: str


class GymVCloseResponse(BaseModel):
    success: bool
    message: str = ""


class GymVNeMoGymResponse(NeMoGymResponse):
    """Response shape extended with the server-issued Gym-V session id.

    ``seed_obs`` carries the post-rules-injection initial observation that
    vLLM saw before the first model call. It is also prepended to
    ``output`` so the flat message log preserves chronological order
    (seed_obs → assistant_1 → step_obs_1 → …); the field is kept as a
    convenience mirror for consumers (e.g. the inspector) that want the
    initial observation without walking ``output``.
    """

    env_id: str
    group_id: str | None = None
    contains_transitions: bool = False
    seed_obs: list[GymVEnvStateEasyInputMessage] | None = Field(
        default=None,
        description=(
            "Post-rules-injection initial observation from /seed_session. "
            "Also mirrored at the head of ``output``. None for legacy "
            "responses or when return_transitions=True."
        ),
    )
    agent_input: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Effective `responses_create_params.input` after agent-side "
            "mutations (system-prompt prepend, string→message coercion) — "
            "the exact initial input vLLM saw on the first `/v1/responses` "
            "call. Downstream reconstructors (e.g. NeMo-RL's full-trajectory "
            "multimodal postprocess) must use this instead of the raw row's "
            "input so their token sequence matches vLLM's. None for legacy "
            "responses."
        ),
    )
    output: list[GymVNeMoGymResponseOutputItem] | list[list[GymVNeMoGymResponseOutputItem]]


class GymVAgentVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: GymVNeMoGymResponse


class GymVAgentVerifyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: GymVNeMoGymResponse
    reward: float
