# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared HTTP contract for CUBE-backed resources servers (seed_session / step / close / verify)
# and matching Responses API rollout agents. Domain-specific code lives in submodules (e.g. osworld).

from typing import Any, Literal

from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
)


class CubeResourcesServerConfig(BaseResourcesServerConfig):
    """Uses inherited ``domain`` (NeMo Gym :class:`~nemo_gym.config_types.Domain` for HF naming, etc.) and ``env_domain`` for the CUBE adapter."""

    model_config = ConfigDict(extra="allow")

    env_domain: Literal["osworld"] = Field(
        default="osworld",
        description="Which CUBE environment adapter to load (e.g. osworld). Distinct from ``domain`` on the base config.",
    )
    eager_benchmark_init: bool = Field(
        default=True,
        description="If True, load the CUBE benchmark during FastAPI startup (before accepting traffic). "
        "Surfaces OSWorld VM/image download errors at server boot and avoids doing that work on the first /seed_session.",
    )
    eager_osworld_vm_warmup: bool = Field(
        default=True,
        description="When True together with eager_benchmark_init and env_domain osworld, boot and tear down one "
        "disposable OSWorld task (see eager_osworld_warmup_task_idx) during startup so QEMU and the guest "
        "finish cold-start before the HTTP server accepts traffic. Each later /seed_session still creates a "
        "new task and launches a fresh VM.",
    )
    eager_osworld_warmup_task_idx: int = Field(
        default=0,
        ge=0,
        description="Which loaded task index to use for the startup VM warmup (must be in range after benchmark load).",
    )


class CubeSeedSessionRequest(BaseSeedSessionRequest):
    task_idx: int


class CubeEnvStateEasyInputMessage(NeMoGymEasyInputMessage):
    """Marks multimodal / desktop-style CUBE observations (optional context collapse in agents)."""

    is_env_state: Literal[True] = True


class CubeSeedSessionResponse(BaseSeedSessionResponse):
    env_id: str
    obs: list[NeMoGymEasyInputMessage | CubeEnvStateEasyInputMessage]
    tools: list[FunctionToolParam]


class CubeStepRequest(BaseModel):
    env_id: str
    action: list[NeMoGymResponseFunctionToolCall]


class CubeStepResponse(BaseModel):
    obs: list[NeMoGymFunctionCallOutput | NeMoGymEasyInputMessage | CubeEnvStateEasyInputMessage]
    reward: float
    done: bool


class CubeNeMoGymResponse(NeMoGymResponse):
    env_id: str
    group_id: str
    contains_transitions: bool
    output: list[NeMoGymResponseOutputItem] | list[list[NeMoGymResponseOutputItem]]


class CubeAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    response: CubeNeMoGymResponse
    debug_step_events: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-step policy/env debug when cube_agent runs with debug_each_step. Top-level so JSON round-trip "
        "to ng_collect_rollouts is reliable (nested OpenAI-shaped response payloads may omit extra fields).",
    )


class CubeAgentVerifyResponse(CubeAgentVerifyRequest, BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class CubeCloseRequest(BaseModel):
    env_id: str


class CubeCloseResponse(BaseModel):
    message: str
    success: bool
