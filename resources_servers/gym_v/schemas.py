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

from nemo_gym.core import (
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
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from resources_servers.gym_v.task_data import SuccessCriterion


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
    success_criterion: SuccessCriterion | None = Field(
        default=None,
        description=(
            "Explicit, environment-audited binary-success predicate. Reward-only legacy rows "
            "may omit it, but then verify returns task_success=null."
        ),
    )
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        ...,
        description="OpenAI Responses-API request body; input is filled by /seed_session.",
    )


class GymVResourcesServerConfig(BaseResourcesServerConfig):
    reward_mode: Literal["training", "task_success"] = Field(
        default="training",
        description="Training returns accumulated native reward; evaluation returns audited binary success.",
    )

    """Server-level config for the Gym-V resources server."""

    num_workers: Literal[1] | None = Field(
        default=None,
        description="Stateful sessions are process-local, so this server must run with at most one worker.",
    )

    task_jsonl_fpaths: list[str] = Field(
        default_factory=list,
        description=(
            "Deprecated v1 compatibility path: ordered JSONL files concatenated "
            "into the in-memory task table. New clients pass task_row to /seed_session."
        ),
    )
    success_criteria_by_env_id: dict[str, SuccessCriterion] = Field(
        default_factory=dict,
        description=(
            "Exact per-environment binary-success allowlist audited against the pinned Gym-V "
            "revision. A row-level success_criterion takes precedence."
        ),
    )
    require_success_criterion: bool = Field(
        default=False,
        description=(
            "Reject task rows that have neither a row-level criterion nor an allowlisted one. "
            "Enable this for training mixtures that report binary task accuracy."
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
    cleanup_tombstone_ttl_seconds: float = Field(
        default=600.0,
        ge=1.0,
        description=(
            "How long an aborting /close for a client-generated env_id prevents "
            "a delayed /seed_session with the same ID from creating an orphan."
        ),
    )
    return_transitions: Literal[False] = Field(
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
    discard_reward: bool = Field(
        default=False,
        description=(
            "Discard accumulated reward when aborting a rollout. Normal close "
            "leaves it available for the subsequent /verify drain."
        ),
    )


class GymVCloseResponse(BaseModel):
    success: bool
    message: str = ""


class GymVNeMoGymResponse(NeMoGymResponse):
    """Response shape extended with the server-issued Gym-V session id.

    ``seed_obs`` carries the post-rules-injection initial observation that
    vLLM saw before the first model call. It is kept separate from ``output``
    so consumers do not process the same initial media twice.
    """

    env_id: str
    group_id: str | None = None
    contains_transitions: bool = False
    seed_obs: list[GymVEnvStateEasyInputMessage] | None = Field(
        default=None,
        description=(
            "Post-rules-injection initial observation from /seed_session. "
            "Kept separate from ``output``. None for legacy responses or "
            "when return_transitions=True."
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


class GymVAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="forbid")

    response: GymVNeMoGymResponse


class GymVAgentVerifyResponse(GymVAgentVerifyRequest, BaseVerifyResponse):
    model_config = ConfigDict(extra="forbid")

    task_success: bool | None = Field(
        default=None,
        description=(
            "Binary outcome from the row's explicit success_criterion; null when the row did "
            "not declare a safe criterion."
        ),
    )
