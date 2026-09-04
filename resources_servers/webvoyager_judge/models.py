# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request and response models for WebVoyager judging."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.web.judge_evidence import expand_webvoyager_judge_screenshots
from nemo_gym.web.models import WebBenchmark, WebTask, WebVerifierResult


MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS = 200


class WebVoyagerJudgeRequest(BaseModel):
    task: WebTask
    final_answer: str = ""
    screenshots: list[str] = Field(default_factory=list, max_length=MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS)
    page_urls: list[str] = Field(default_factory=list, max_length=MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS)

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
    screenshots: list[str] = Field(default_factory=list, max_length=MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS)
    page_urls: list[str] = Field(default_factory=list, max_length=MAX_WEBVOYAGER_JUDGE_EVIDENCE_ITEMS)

    @model_validator(mode="before")
    @classmethod
    def recover_reverify_evidence(cls, value: Any) -> Any:
        """Recover evidence stored in the rollout response during reverification."""

        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        response = normalized.get("response")
        if isinstance(response, BaseModel):
            response = response.model_dump(mode="json")
        evidence = response.get("webvoyager_judge_evidence") if isinstance(response, dict) else None
        if isinstance(evidence, dict):
            for key in ("final_answer", "page_urls"):
                if key not in normalized and key in evidence:
                    normalized[key] = evidence[key]
            if "screenshots" not in normalized:
                normalized["screenshots"] = expand_webvoyager_judge_screenshots(evidence, response)
        return normalized


class WebVoyagerStandardVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    raw_score: float = 0.0
    task_success: bool = False
    mask_sample: bool = False
    failure_kind: str | None = None
    judge_text: str = ""
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)
    verifier_version: str = "webvoyager-llm-judge-v1"
