# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared HealthBench rubric-scoring NeMo Gym resources server."""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.verifier_fixture import VerifierFixture


# Grader prompt from OpenAI simple-evals (MIT); see prompts/LICENSE and README.md.
GRADER_TEMPLATE = (Path(__file__).parent / "prompts" / "grader.txt").read_text(encoding="utf-8").strip()


class HealthBenchRubricResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    name: str = "healthbench_rubric"
    judge_model_server: ModelServerRef
    judge_chat_completion_params: NeMoGymChatCompletionCreateParamsNonStreaming
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None
    judge_system_prompt: str | None = "You are a helpful assistant."
    judge_endpoint_max_concurrency: int = Field(default=64, ge=1)
    judge_max_attempts: int = Field(default=3, ge=1)
    judge_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    length_adjustment_center: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    length_adjustment_penalty_per_500_chars: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    reward_metric: Literal["overall_score", "overall_score_length_adjusted"] = "overall_score"

    @model_validator(mode="after")
    def validate_length_adjustment(self) -> "HealthBenchRubricResourcesServerConfig":
        """Reject incomplete scoring settings before serving requests."""
        if (self.length_adjustment_center is None) != (self.length_adjustment_penalty_per_500_chars is None):
            raise ValueError("Length-adjustment settings must be set together")
        if self.reward_metric == "overall_score_length_adjusted" and self.length_adjustment_center is None:
            raise ValueError("Length-adjusted reward requires length-adjustment settings")
        return self


class HealthBenchRubricVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    source_id: str
    conversation: list[dict[str, str]]
    rubrics: list[dict[str, Any]] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_rubrics(self) -> "HealthBenchRubricVerifyRequest":
        """Reject malformed rubric data before any paid judge call."""
        for rubric in self.rubrics:
            if not isinstance(rubric.get("criterion"), str) or not rubric["criterion"].strip():
                raise ValueError("Each rubric requires non-empty criterion text")
            points = rubric.get("points")
            if isinstance(points, bool) or not isinstance(points, (int, float)) or not math.isfinite(points):
                raise ValueError("Rubric points must be finite numbers")
        if not any(rubric["points"] > 0 for rubric in self.rubrics):
            raise ValueError("Rubrics have no positive points")
        if not self.conversation or self.conversation[-1].get("role") != "user":
            raise ValueError("Conversation must end with a user message")
        return self


class HealthBenchRubricVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    source_id: str
    evaluation_completed: bool = True
    overall_score: float
    overall_score_length_adjusted: Optional[float] = None
    response_chars: int
    n_rubrics: int
    grades: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    for output in reversed(body.response.output):
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        return "".join(item.text for item in output.content if getattr(item, "type", None) == "output_text")
    return ""


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _conversation_text(prompt: list[dict[str, Any]], response_text: str) -> str:
    messages = prompt + [{"role": "assistant", "content": response_text}]
    return "\n\n".join(f"{message['role']}: {_message_text(message)}" for message in messages)


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match is not None:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Judge response did not contain a JSON object")


def _completion_content(response: NeMoGymChatCompletion | NeMoGymResponse) -> str:
    if isinstance(response, NeMoGymResponse):
        return response.output_text
    content = response.choices[0].message.content
    return content if isinstance(content, str) else ""


def _calculate_score(rubrics: list[dict[str, Any]], grades: list[dict[str, Any]]) -> float:
    total_possible_points = sum(float(rubric["points"]) for rubric in rubrics if float(rubric["points"]) > 0)
    if total_possible_points == 0:
        raise ValueError("Rubrics have no positive points")
    achieved_points = sum(
        float(rubric["points"]) for rubric, grade in zip(rubrics, grades, strict=True) if grade["criteria_met"]
    )
    return achieved_points / total_possible_points


def _finite_reward(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"Reward is not finite: {value}")
    return value


class HealthBenchRubricResourcesServer(SimpleResourcesServer):
    config: HealthBenchRubricResourcesServerConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._judge_endpoint_max_concurrency = asyncio.Semaphore(self.config.judge_endpoint_max_concurrency)

    async def verify(self, body: HealthBenchRubricVerifyRequest) -> HealthBenchRubricVerifyResponse:
        """Grade the final answer, preserving raw scores and masking unavailable judgements."""
        response_text = _extract_last_assistant_text(body)
        if not response_text.strip():
            return HealthBenchRubricVerifyResponse(
                **(
                    body.model_dump(exclude={"_ng_failure_class", "_ng_failure_judge_error"})
                    | {
                        "reward": 0.0,
                        "overall_score": 0.0,
                        "overall_score_length_adjusted": 0.0
                        if self.config.length_adjustment_center is not None
                        else None,
                        "response_chars": len(response_text),
                        "n_rubrics": len(body.rubrics),
                        "grades": [],
                        "evaluation_completed": True,
                        "mask_sample": False,
                        "failure_kind": "healthbench_rubric:empty_response",
                        "failure_reason": "No final assistant answer",
                    }
                )
            )
        conversation = _conversation_text(body.conversation, response_text)
        try:
            grades = [await self._grade_rubric(conversation, rubric) for rubric in body.rubrics]
        except JudgeError as error:
            reason = str(error)[:2000]
            return HealthBenchRubricVerifyResponse(
                **(
                    body.model_dump(exclude={"_ng_failure_class", "_ng_failure_judge_error"})
                    | {
                        "reward": 0.0,
                        "overall_score": 0.0,
                        "overall_score_length_adjusted": None,
                        "response_chars": len(response_text),
                        "n_rubrics": len(body.rubrics),
                        "grades": [],
                        "evaluation_completed": False,
                        "mask_sample": True,
                        "failure_kind": "judge_failed",
                        "failure_reason": reason,
                        "_ng_failure_class": "judge_failed",
                        "_ng_failure_judge_error": reason,
                    }
                )
            )

        score = _calculate_score(body.rubrics, grades)
        length_adjusted_score = self._calculate_length_adjusted_score(score, response_text)

        metrics = {
            "overall_score": score,
            "overall_score_length_adjusted": length_adjusted_score,
        }
        if self.config.reward_metric not in metrics:
            raise ValueError(
                f"Unknown reward metric {self.config.reward_metric!r}; available metrics: {sorted(metrics)}"
            )
        reward_value = metrics[self.config.reward_metric]
        if reward_value is None:
            raise ValueError(f"Reward metric {self.config.reward_metric!r} is not available")

        return HealthBenchRubricVerifyResponse(
            **(
                body.model_dump(exclude={"_ng_failure_class", "_ng_failure_judge_error"})
                | {
                    "reward": max(0.0, min(1.0, _finite_reward(float(reward_value)))),
                    "overall_score": score,
                    "overall_score_length_adjusted": length_adjusted_score,
                    "response_chars": len(response_text),
                    "n_rubrics": len(body.rubrics),
                    "grades": grades,
                    "evaluation_completed": True,
                    "mask_sample": False,
                    "failure_kind": None,
                    "failure_reason": None,
                }
            )
        )

    def _calculate_length_adjusted_score(self, score: float, response_text: str) -> float | None:
        center = self.config.length_adjustment_center
        penalty = self.config.length_adjustment_penalty_per_500_chars
        if center is None and penalty is None:
            return None
        if center is None or penalty is None:
            raise ValueError(
                "length_adjustment_center and length_adjustment_penalty_per_500_chars must be set together"
            )
        if center < 0 or penalty < 0:
            raise ValueError("Length-adjustment settings must be non-negative")
        return score - penalty * ((len(response_text) - center) / 500.0)

    async def _grade_rubric(self, conversation: str, rubric: dict[str, Any]) -> dict[str, Any]:
        rubric_text = f"[{rubric['points']}] {rubric['criterion']}"
        prompt = GRADER_TEMPLATE.replace("<<conversation>>", conversation).replace("<<rubric_item>>", rubric_text)

        messages = [{"role": "user", "content": prompt}]
        if self.config.judge_system_prompt is not None:
            messages.insert(0, {"role": "system", "content": self.config.judge_system_prompt})
        if self.config.judge_responses_create_params is not None:
            judge_params = self.config.judge_responses_create_params.model_copy(deep=True)
            judge_params.input = messages
            endpoint, response_model = "/v1/responses", NeMoGymResponse
        else:
            judge_params = self.config.judge_chat_completion_params.model_copy(deep=True)
            judge_params.messages = messages
            endpoint, response_model = "/v1/chat/completions", NeMoGymChatCompletion

        for attempt in range(self.config.judge_max_attempts):
            try:
                async with self._judge_endpoint_max_concurrency:
                    async with asyncio.timeout(self.config.judge_timeout_seconds):
                        judge_response = await call_judge(
                            self.server_client,
                            server_name=self.config.judge_model_server.name,
                            url_path=endpoint,
                            json=judge_params,
                            response_model=response_model,
                        )
            except TimeoutError as error:
                raise JudgeError("HealthBench judge call exceeded its timeout") from error

            try:
                parsed = _parse_json_object(_completion_content(judge_response))
                if not isinstance(parsed.get("criteria_met"), bool):
                    raise ValueError("Judge JSON did not include boolean criteria_met")
                return {
                    "criteria_met": parsed["criteria_met"],
                    "explanation": str(parsed.get("explanation", "")),
                    "judgement_id": judge_response.id,
                    "usage": judge_response.usage.model_dump() if judge_response.usage is not None else None,
                }
            except (ValueError, IndexError) as error:
                if attempt + 1 == self.config.judge_max_attempts:
                    raise JudgeError("HealthBench judge returned unusable grades after bounded retries") from error
        raise AssertionError("judge_max_attempts must be positive")

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Clamp the mean of raw scores, as in simple-evals, rather than each sample."""
        metrics = {}
        for key in ("overall_score", "overall_score_length_adjusted"):
            values = [
                row[key] for task in tasks for row in task if not row.get("mask_sample") and row.get(key) is not None
            ]
            if values:
                metrics[key] = max(0.0, min(1.0, math.fsum(values) / len(values)))
        return metrics

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Report the same benchmark metrics for each repeat."""
        return self.compute_metrics(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """Expose explicit benchmark metric names for comparison and certification."""
        return {
            key: agent_metrics[key]
            for key in ("overall_score", "overall_score_length_adjusted")
            if key in agent_metrics
        }


def _fixture_server() -> HealthBenchRubricResourcesServer:
    from resources_servers.healthbench_rubric.tests.fixture_support import fixture_server

    return fixture_server()


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_fixture_server,
    request_model=HealthBenchRubricVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    HealthBenchRubricResourcesServer.run_webserver()
