# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym verifier for custom 1–5 pointwise Indic AlpacaEval ratings."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from statistics import mean, stdev
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.prompt import PromptConfig, fill_prompt, load_prompt_config
from resources_servers.alpaca_pointwise_judge.task_data import AlpacaTask


Score = Annotated[int, Field(strict=True, ge=1, le=5)]


class DimensionScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction_following: Score
    correctness: Score
    usefulness: Score
    language_quality: Score


class PointwiseJudgment(BaseModel):
    """Strict rubric output; booleans and numeric strings are invalid."""

    model_config = ConfigDict(extra="forbid")

    overall_score: Score
    dimension_scores: DimensionScores
    rationale: str = Field(min_length=1)
    detected_response_language: str = Field(min_length=1)

    @field_validator("rationale", "detected_response_language")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must be nonempty")
        return value

    @field_validator("rationale")
    @classmethod
    def concise_rationale(cls, value: str) -> str:
        if len(value.split()) > 120:
            raise ValueError("Rationale must contain at most 120 words")
        return value


class AlpacaPointwiseJudgeConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef
    judge_chat_completions_create_params: NeMoGymChatCompletionCreateParamsNonStreaming
    judge_prompt_path: str = "resources_servers/alpaca_pointwise_judge/prompts/alpaca_pointwise_v1.yaml"
    max_judge_attempts: int = Field(default=3, ge=1, le=5)
    expected_counts: dict[str, int] = Field(default_factory=dict)


class AlpacaPointwiseVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: AlpacaTask


class AlpacaPointwiseVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: AlpacaTask
    judgment: PointwiseJudgment
    judge_completion: NeMoGymChatCompletion
    judge_attempts: int
    candidate_answer: str
    empty_final_answer: bool
    protocol: str = "indic_alpaca_pointwise_v1"


def final_answer_only(text: str) -> str:
    """Exclude tagged reasoning, including incomplete or orphaned think tags."""
    text = re.split(r"</(?:think|thinking)>", text)[-1]
    text = re.split(r"<(?:think|thinking)>", text, maxsplit=1)[0]
    return text.strip()


class AlpacaPointwiseJudgeServer(SimpleResourcesServer):
    """Reuse Gym's judge transport, response types, failure routing and aggregation."""

    ray_enabled = False
    config: AlpacaPointwiseJudgeConfig
    _prompt: PromptConfig = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._prompt = load_prompt_config(self.config.judge_prompt_path)

    async def verify(self, body: AlpacaPointwiseVerifyRequest) -> AlpacaPointwiseVerifyResponse:
        # output_text excludes structured reasoning. Preserve refusal content
        # so the rubric can distinguish justified and unnecessary refusals.
        answer = body.response.output_text
        if not answer:
            answer = "".join(
                content.refusal
                for item in body.response.output
                if item.type == "message"
                for content in item.content
                if content.type == "refusal"
            )
        answer = final_answer_only(answer)
        task = body.verifier_metadata
        evaluation_json = json.dumps(
            {
                "language_code": task.language_code,
                "language_name": task.language_name,
                "instruction": task.question,
                "response": answer,
            },
            ensure_ascii=False,
        )
        params = self.config.judge_chat_completions_create_params.model_copy(deep=True)
        params.messages = fill_prompt(self._prompt, {"evaluation_json": evaluation_json})
        params.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "indic_alpaca_pointwise",
                "strict": True,
                "schema": PointwiseJudgment.model_json_schema(),
            },
        }
        for attempt in range(1, self.config.max_judge_attempts + 1):
            completion = await call_judge(
                self.server_client,
                server_name=self.config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=params,
                response_model=NeMoGymChatCompletion,
            )
            try:
                choice = completion.choices[0]
                if choice.finish_reason != "stop" or not choice.message.content:
                    raise ValueError("Judge output is empty or truncated")
                judgment = PointwiseJudgment.model_validate_json(choice.message.content)
            except (IndexError, ValidationError, ValueError):
                continue
            # Re-verification can submit an already-scored or judge-failed row.
            # Replace the prior measurement while preserving task/rollout IDs.
            result = body.model_dump()
            result.pop("_ng_failure_class", None)
            result.pop("_ng_failure_judge_error", None)
            if isinstance(result.get("instance_config"), dict):
                result["instance_config"] = {**result["instance_config"], "mask_sample": False}
            result.update(
                # Graded evaluation: raw 1–5 scores remain the headline metric.
                reward=(judgment.overall_score - 1) / 4,
                mask_sample=False,
                failure_kind=None,
                failure_reason=None,
                judgment=judgment,
                judge_completion=completion,
                judge_attempts=attempt,
                candidate_answer=answer,
                empty_final_answer=not answer,
                protocol="indic_alpaca_pointwise_v1",
            )
            return AlpacaPointwiseVerifyResponse.model_validate(result)
        # Invalid judge JSON is a failed measurement, not a bad candidate.
        # judge_failsafe preserves the candidate in Gym's retry sidecar.
        raise JudgeError(f"Invalid or truncated pointwise judgment after {self.config.max_judge_attempts} attempts")

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Average repeats per task; report language coverage without best-of-N."""
        expected = dict(self.config.expected_counts)
        groups: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        evaluation_sets: set[str] = set()
        for task in tasks:
            if not task:
                continue
            metadata = AlpacaTask.model_validate(task[0]["verifier_metadata"])
            identity = (metadata.language_code, metadata.row_id)
            if identity in seen:
                raise ValueError(f"Duplicate task in metrics: {identity}")
            seen.add(identity)
            evaluation_sets.add(metadata.evaluation_set_id)
            if len(evaluation_sets) > 1:
                raise ValueError("Cannot combine different prepared evaluation sets")
            if expected and metadata.expected_counts != expected:
                raise ValueError("Inconsistent expected language coverage")
            expected = metadata.expected_counts
            if any(row["verifier_metadata"] != task[0]["verifier_metadata"] for row in task):
                raise ValueError("Repeated rollouts must describe the same task")
            groups[metadata.language_code].append(task)

        metrics: dict[str, Any] = {}
        language_means = []
        language_dimension_means: dict[str, list[float]] = defaultdict(list)
        total_judged = 0
        complete = bool(expected)
        for language in sorted(set(expected) | set(groups)):
            prefix = f"alpaca_pointwise/{language}"
            scores = []
            dimensions: dict[str, list[float]] = defaultdict(list)
            invalid = empty = pending_review = 0
            for task in groups[language]:
                valid = [row for row in task if not row.get("mask_sample") and row.get("judgment") is not None]
                invalid += len(task) - len(valid)
                if not valid:
                    continue
                judgments = [PointwiseJudgment.model_validate(row["judgment"]) for row in valid]
                scores.append(mean(j.overall_score for j in judgments))
                for dimension in DimensionScores.model_fields:
                    dimensions[dimension].append(mean(getattr(j.dimension_scores, dimension) for j in judgments))
                empty += sum(bool(row.get("empty_final_answer")) for row in valid)
                pending_review += bool(task[0]["verifier_metadata"].get("human_evaluation_pending", False))
            language_complete = len(scores) == expected.get(language, 0) and invalid == 0
            complete = complete and language_complete
            total_judged += len(scores)
            metrics.update(
                {
                    f"{prefix}/expected": expected.get(language, 0),
                    f"{prefix}/judged": len(scores),
                    f"{prefix}/complete": language_complete,
                    f"{prefix}/invalid_judgments": invalid,
                    f"{prefix}/empty_final_answers": empty,
                    f"{prefix}/translation_review_pending_judged": pending_review,
                }
            )
            if scores:
                language_means.append(mean(scores))
                metrics[f"{prefix}/mean_overall_score"] = mean(scores)
                if len(scores) > 1:
                    metrics[f"{prefix}/standard_error"] = stdev(scores) / len(scores) ** 0.5
                for dimension, values in dimensions.items():
                    metrics[f"{prefix}/mean_{dimension}"] = mean(values)
                    language_dimension_means[dimension].append(mean(values))
        metrics.update(
            {
                "alpaca_pointwise/expected": sum(expected.values()),
                "alpaca_pointwise/judged": total_judged,
                "alpaca_pointwise/complete": complete,
            }
        )
        if complete:
            metrics["alpaca_pointwise/macro_mean_overall_score"] = mean(language_means)
            for dimension, values in language_dimension_means.items():
                metrics[f"alpaca_pointwise/macro_mean_{dimension}"] = mean(values)
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in agent_metrics.items() if key.startswith("alpaca_pointwise/")}


if __name__ == "__main__":
    AlpacaPointwiseJudgeServer.run_webserver()
