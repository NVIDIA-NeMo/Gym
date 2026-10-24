# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the WebVoyager LLM-as-a-judge server."""

from typing import ClassVar, Literal

from pydantic import Field

from nemo_gym.base_resources_server import BaseResourcesServerConfig, ReverifyMode
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


class WebVoyagerJudgeConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    max_screenshots: int = Field(default=3, ge=1, le=200)
    require_screenshot: bool = True
    verifier_version: str = "webvoyager-gemini-v1"
    judge_call_timeout_secs: float = Field(default=270.0, gt=0.0)
    judge_parse_max_retries: int = Field(default=0, ge=0, le=3)
    # Keep the reference benchmark's historical default. Training or a new
    # evaluation recipe can explicitly mask judge failures instead of treating
    # them as measurements of the policy.
    unparseable_verdict_policy: Literal["zero", "mask"] = "zero"
