# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request and response models for WebVoyager judging."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.web.models import WebBenchmark, WebTask, WebVerifierResult


class WebVoyagerJudgeRequest(BaseModel):
    task: WebTask
    final_answer: str = ""
    screenshots: list[str] = Field(default_factory=list, max_length=20)
    page_urls: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def require_webvoyager(self) -> "WebVoyagerJudgeRequest":
        if self.task.benchmark != WebBenchmark.WEBVOYAGER:
            raise ValueError("the WebVoyager judge only accepts webvoyager tasks")
        return self


class WebVoyagerJudgeResponse(BaseModel):
    result: WebVerifierResult
    judge_text: str = ""


class WebVoyagerStandardVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    web_task: WebTask
    final_answer: str = ""
    screenshots: list[str] = Field(default_factory=list, max_length=20)
    page_urls: list[str] = Field(default_factory=list, max_length=20)


class WebVoyagerStandardVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    raw_score: float = 0.0
    task_success: bool = False
    mask_sample: bool = False
    failure_kind: str | None = None
    judge_text: str = ""
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)
