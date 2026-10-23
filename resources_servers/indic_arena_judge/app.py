# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indic Arena Hard judging, defaulting to an isolated pointwise rubric.

Pairwise judging delegates to Gym's unchanged Arena Hard v2 server. Baseline
validation and reference-free scoring belong only to this Indic component.
"""

import re
from collections import defaultdict
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymChatCompletion
from nemo_gym.prompt import load_prompt_config
from resources_servers.arena_judge.app import (
    ArenaJudgeConfig,
    ArenaJudgeServer,
    ArenaJudgeVerifyRequest,
    ArenaJudgeVerifyResponse,
    sanitize_generation,
)


_POINTWISE_DIMENSIONS = ("correctness", "relevance", "completeness", "language_quality")
RubricScore = Annotated[int, Field(strict=True, ge=0, le=4)]


class PointwiseJudgement(BaseModel):
    """Strict rubric output; each dimension is an integer from zero to four."""

    model_config = ConfigDict(extra="forbid")

    correctness: RubricScore
    relevance: RubricScore
    completeness: RubricScore
    language_quality: RubricScore
    explanation: str = Field(min_length=1)


class IndicArenaJudgeConfig(ArenaJudgeConfig):
    """Indic-specific evaluation mode; English Arena configuration is unchanged."""

    evaluation_mode: Literal["pairwise", "pointwise"] = "pointwise"
    pointwise_prompt_path: str = "resources_servers/indic_arena_judge/prompts/pointwise.yaml"


class IndicArenaJudgeVerifyResponse(ArenaJudgeVerifyResponse):
    evaluation_mode: Literal["pairwise", "pointwise"] = "pointwise"
    judgement_pointwise: str | None = None
    pointwise_scores: dict[str, int] | None = None
    pointwise_score: float | None = None
    invalid_pointwise: bool = False


class IndicArenaJudgeServer(ArenaJudgeServer):
    """Reuse Arena pairwise judging and add the Indic pointwise rubric."""

    config: IndicArenaJudgeConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.config.evaluation_mode == "pointwise":
            self._pointwise_prompt = load_prompt_config(self.config.pointwise_prompt_path)

    async def verify(self, body: ArenaJudgeVerifyRequest) -> IndicArenaJudgeVerifyResponse:
        if self.config.evaluation_mode == "pairwise":
            baseline = body.baseline_answer or ""
            if self.config.sanitize_generations:
                baseline = sanitize_generation(baseline)
            if not baseline.strip():
                raise HTTPException(
                    status_code=422,
                    detail="Pairwise Indic Arena Hard requires a nonempty baseline_answer. "
                    "Supply a baseline or use pointwise mode.",
                )
            result = await super().verify(body)
            return IndicArenaJudgeVerifyResponse.model_validate(result.model_dump() | {"evaluation_mode": "pairwise"})

        candidate_answer = self._extract_response_output_text(body.response)
        if self.config.sanitize_generations:
            candidate_answer = sanitize_generation(candidate_answer)
        if not candidate_answer.strip():
            text = "Empty candidate answer; assigned zero without a judge call."
            scores = dict.fromkeys(_POINTWISE_DIMENSIONS, 0)
        else:
            text = await self._judge_pointwise(body.question or "", candidate_answer)
            parsed = self._parse_pointwise(text)
            scores = {name: getattr(parsed, name) for name in _POINTWISE_DIMENSIONS} if parsed is not None else None
        score = 100.0 * sum(scores.values()) / 16.0 if scores is not None else None
        return IndicArenaJudgeVerifyResponse.model_validate(
            body.model_dump()
            | {
                "evaluation_mode": "pointwise",
                "reward": score / 100.0 if score is not None else 0.0,
                "judgement_pointwise": text,
                "pointwise_scores": scores,
                "pointwise_score": score,
                "invalid_pointwise": scores is None,
            }
        )

    async def _judge_pointwise(self, question: str, answer: str) -> str:
        fill = {"question": question, "answer": answer}
        messages: list[dict[str, str]] = []
        if self._pointwise_prompt.system is not None:
            messages.append({"role": "system", "content": self._pointwise_prompt.system.format_map(fill)})
        messages.append({"role": "user", "content": self._pointwise_prompt.user.format_map(fill)})
        params = self.config.judge_chat_completions_create_params.model_copy(deep=True)
        params.messages = messages
        result = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/chat/completions",
            json=params,
            response_model=NeMoGymChatCompletion,
        )
        return self._extract_chat_completion_text(result)

    @staticmethod
    def _parse_pointwise(text: str) -> PointwiseJudgement | None:
        text = text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            return PointwiseJudgement.model_validate_json(text)
        except ValidationError:
            return None

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        if self.config.evaluation_mode == "pairwise":
            return super().compute_metrics(tasks)
        metrics = {f"pointwise/{key}": value for key, value in self._pointwise_metrics(tasks).items()}
        languages: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
        for task in tasks:
            if task and task[0].get("language"):
                languages[task[0]["language"]].append(task)
        for language, language_tasks in languages.items():
            for key, value in self._pointwise_metrics(language_tasks).items():
                metrics[f"pointwise/language/{language}/{key}"] = value
        return metrics

    @staticmethod
    def _pointwise_metrics(tasks: list[list[dict[str, Any]]]) -> dict[str, float]:
        # Average repeats within a task before averaging tasks. Count invalid
        # judge outputs separately and exclude them from rubric means.
        task_scores: list[float] = []
        dimensions: dict[str, list[float]] = {name: [] for name in _POINTWISE_DIMENSIONS}
        valid_count = invalid_count = 0
        for task in tasks:
            valid = [r for r in task if not r.get("invalid_pointwise", False) and r.get("pointwise_score") is not None]
            valid_count += len(valid)
            invalid_count += len(task) - len(valid)
            if not valid:
                continue
            task_scores.append(float(np.mean([r["pointwise_score"] for r in valid])))
            for name in dimensions:
                dimensions[name].append(float(np.mean([r["pointwise_scores"][name] for r in valid])))
        return {
            "score": float(np.mean(task_scores)) if task_scores else float("nan"),
            "valid_scores": valid_count,
            "invalid_scores": invalid_count,
            "scored_tasks": len(task_scores),
            **{name: float(np.mean(values)) if values else float("nan") for name, values in dimensions.items()},
        }

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        if self.config.evaluation_mode == "pairwise":
            return super().get_key_metrics(agent_metrics)
        return {
            name: value
            for name, value in agent_metrics.items()
            if name.startswith("pointwise/") or name in ("mean/input_tokens", "mean/output_tokens")
        }


if __name__ == "__main__":
    IndicArenaJudgeServer.run_webserver()
