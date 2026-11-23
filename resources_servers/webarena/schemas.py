# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Literal

from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionResponse,
    BaseVerifyResponse,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputItem,
)


DEFAULT_CREDENTIALS: dict[str, dict[str, str]] = {
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "classifieds": {
        "username": "blake.sullivan@gmail.com",
        "password": "Password.123",
    },
}


class WebArenaResourcesServerConfig(BaseResourcesServerConfig):
    site_urls: dict[str, str] = Field(default_factory=dict)
    credentials: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {key: dict(value) for key, value in DEFAULT_CREDENTIALS.items()}
    )
    viewport_width: int = Field(default=1920, ge=320)
    viewport_height: int = Field(default=1080, ge=240)
    screenshot_format: Literal["PNG", "JPEG"] = "PNG"
    screenshot_jpeg_quality: int = Field(default=85, ge=1, le=100)
    post_action_screenshot_delay_seconds: float = Field(default=4.0, ge=0)
    max_browser_workers: int = Field(default=16, ge=1)
    initial_browser_workers: int | None = Field(default=None, ge=0)
    browser_actor_num_cpus: float = Field(default=1.0, gt=0)
    browser_actor_resources: dict[str, float] = Field(default_factory=dict)
    browser_startup_concurrency: int = Field(default=1, ge=1)
    browser_startup_stagger_seconds: float = Field(default=30.0, ge=0)
    browser_startup_timeout_seconds: float = Field(default=120.0, gt=0)
    browser_pool_wait_timeout_seconds: float = Field(default=300.0, gt=0)
    browser_operation_timeout_seconds: float = Field(default=180.0, gt=0)
    browser_evaluation_timeout_seconds: float = Field(default=300.0, gt=0)
    browser_session_max_seconds: float = Field(default=1800.0, gt=0)
    judge_model: str = "us/azure/openai/gpt-4.1"
    judge_base_url: str = "https://inference-api.nvidia.com"
    judge_api_key: str = ""
    judge_timeout_seconds: float = Field(default=120.0, gt=0)

    @model_validator(mode="after")
    def validate_worker_and_pool_config(self) -> Self:
        if self.num_workers not in (None, 1):
            raise ValueError("WebArena requires num_workers=1 because session state is process-local")
        if self.initial_browser_workers is None:
            self.initial_browser_workers = self.max_browser_workers
        if self.initial_browser_workers > self.max_browser_workers:
            raise ValueError("initial_browser_workers cannot exceed max_browser_workers")
        return self


class WebArenaTaskRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    dataset: str = "webarena"
    dataset_version: str = "classic"
    dataset_hash: str = "6a6dae6809c84a08881ac37cadbfd4d50eac66e3578ebdb6ba8a308eebe1e0a3"
    sites: list[str]
    start_urls: list[str]
    intent: str
    eval: dict[str, Any]
    task_input_images: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_classic_evaluator_contract(self) -> Self:
        eval_types = self.eval.get("eval_types")
        if not isinstance(eval_types, list) or not eval_types:
            raise ValueError("task.eval.eval_types must be a non-empty list")
        unsupported = set(eval_types) - {"string_match", "url_match", "program_html"}
        if unsupported:
            raise ValueError(
                "Unsupported non-Classic WebArena evaluator(s): "
                + ", ".join(sorted(str(item) for item in unsupported))
            )
        return self


class WebArenaAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_payload: WebArenaTaskRow

    # Context-compaction caller identity (see nemo_gym.context_compaction).
    # NeMo-RL stamps these per logical rollout; they are optional for pure
    # Gym-side evaluation.
    context_compaction_rollout_id: str | None = None
    context_compaction_group_id: str | None = None
    context_compaction_task_id: str | None = None
    context_compaction_rollout_index: int | None = Field(default=None, ge=0)
    context_compaction_attempt_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def load_inline_task_from_verifier_metadata(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("task_payload") is not None:
            return value
        verifier_metadata = value.get("verifier_metadata")
        if isinstance(verifier_metadata, dict) and verifier_metadata.get("task") is not None:
            return {**value, "task_payload": verifier_metadata["task"]}
        return value


class WebArenaEnvStateEasyInputMessage(NeMoGymEasyInputMessage):
    is_env_state: Literal[True] = True
    is_browser_screenshot: bool = True
    step: int = 0
    tab_context: dict[str, Any] | None = None


class WebArenaSeedSessionRequest(BaseModel):
    request_id: str
    task: WebArenaTaskRow


class WebArenaSeedSessionResponse(BaseSeedSessionResponse):
    env_id: str
    obs: list[WebArenaEnvStateEasyInputMessage]
    tools: list[FunctionToolParam]


class WebArenaStepRequest(BaseModel):
    env_id: str
    action_request_id: str
    action: list[NeMoGymResponseFunctionToolCall]


class WebArenaStepResponse(BaseModel):
    obs: list[NeMoGymFunctionCallOutput | NeMoGymEasyInputMessage | WebArenaEnvStateEasyInputMessage]
    reward: float = 0.0
    done: bool = False
    termination_status: str | None = None
    answer: str | None = None
    evaluator_details: dict[str, Any] = Field(default_factory=dict)
    # Set when the tool raised while executing; the browser state is then undefined.
    tool_error: str | None = None


class WebArenaFinalizeRequest(BaseModel):
    env_id: str
    finalize_request_id: str


class WebArenaFinalizeResponse(BaseModel):
    env_id: str
    finalize_request_id: str
    reward: float
    evaluator_details: dict[str, Any] = Field(default_factory=dict)
    termination_status: str | None = None
    answer: str | None = None


class WebArenaCloseRequest(BaseModel):
    env_id: str
    close_request_id: str


class WebArenaCloseResponse(BaseModel):
    success: bool
    message: str = ""


class WebArenaNeMoGymResponse(NeMoGymResponse):
    env_id: str
    group_id: str
    contains_transitions: Literal[False] = False
    seed_obs: list[WebArenaEnvStateEasyInputMessage]
    output: list[NeMoGymResponseInputItem]


class WebArenaAgentVerifyRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verify_request_id: str
    response: WebArenaNeMoGymResponse


class WebArenaAgentVerifyResponse(WebArenaAgentVerifyRequest, BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    # With context management, run() strips raw image bytes from the echoed
    # response after verification; the trainer reads media from token capture.
    evaluator_details: dict[str, Any] = Field(default_factory=dict)
    termination_status: str | None = None
    answer: str | None = None
