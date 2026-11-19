# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VoiceBench open-response judge; actual returned choices determine legacy weighting."""

import asyncio
import hashlib
import math
import re
from pathlib import Path
from time import perf_counter

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.verification import call_judge
from resources_servers.voicebench.scorers.base import ScoringResult


class JudgeSettings(BaseModel):
    judge_model_server: ModelServerRef | None = None
    judge_prompt_file: str | None = None
    judge_temperature: float = 0.5
    judge_top_p: float = 0.95
    judge_max_tokens: int = Field(default=1024, gt=0)
    judge_n: int = Field(default=3, gt=0)
    judge_max_concurrency: int = Field(default=40, gt=0)


class JudgePrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: str | None = None
    user: str


def parse_score(text: str | None) -> float:
    """Match OpenEvaluator: accept any float, otherwise extract a boxed integer."""
    if text is None:
        raise AttributeError("legacy judge content is None and cannot be stripped")
    try:
        return float(text)
    except ValueError:
        match = re.search(r"\[\[(\d+)\]\]", text) or re.search(r"\\boxed\s*{\s*(\d+)\s*}", text)
        if match is None:
            raise NotImplementedError("legacy OpenEvaluator cannot extract a rating") from None
        return float(match.group(1))


def score_outputs(
    outputs: list[str | None], *, finish_reasons: list[str] | None = None, requested_n: int = 3
) -> ScoringResult:
    """Score saved grades with the source parser, including its failure behavior.

    Finish reasons remain diagnostic: the original evaluator never inspected them.
    The source's empty mean is NaN; the transport reward stays finite while the
    legacy score and sufficient statistics preserve nonfinite values.
    """
    finish_reasons = finish_reasons if finish_reasons is not None else ["stop"] * len(outputs)
    if len(finish_reasons) != len(outputs):
        raise ValueError("finish_reasons must match judge output count")
    scores = [parse_score(output) for output in outputs]
    score_sum = sum(scores)
    score_count = len(scores)
    mean = score_sum / score_count if score_count else float("nan")
    return ScoringResult(
        mean / 5 if math.isfinite(mean) else 0.0,
        {"score_sum": score_sum, "score_count": float(score_count)},
        {
            "judge_valid": True,
            "judge_outputs": outputs,
            "judge_scores": scores,
            "judge_finish_reasons": finish_reasons,
            "judge_requested_n": requested_n,
            "judge_actual_n": len(outputs),
            "judge_choice_count_mismatch": len(outputs) != requested_n,
            "judge_parse_errors": [],
            "judge_numeric_valid": bool(scores) and all(math.isfinite(score) and 1 <= score <= 5 for score in scores),
            "voicebench_score": mean,
            "voicebench_score_sum": score_sum,
            "voicebench_score_count": score_count,
        },
    )


def scoring_failure(error: Exception) -> ScoringResult:
    """Keep a failed row inspectable; aggregation must refuse a legacy score."""
    reason = f"{type(error).__name__}: {error}"
    return ScoringResult(
        0.0, {}, {"judge_valid": False, "judge_parse_errors": [reason], "legacy_scoring_error": reason}, False
    )


class OpenJudgeScorer:
    """One bounded judge client; caller chooses an independent versioned prompt file."""

    def __init__(self, settings: JudgeSettings, server_client: ServerClient) -> None:
        if settings.judge_model_server is None or settings.judge_prompt_file is None:
            raise ValueError("judge scoring requires judge_model_server and judge_prompt_file")
        self.settings = settings
        self.server_client = server_client
        prompt_bytes = Path(settings.judge_prompt_file).read_bytes()
        self.prompt = JudgePrompt.model_validate(yaml.safe_load(prompt_bytes))
        if not {"{prompt}", "{response}"}.issubset(set(re.findall(r"\{\w+\}", self.prompt.user))):
            raise ValueError("judge user prompt must contain {prompt} and {response}")
        self.prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        self._slots = asyncio.Semaphore(settings.judge_max_concurrency)
        self.inflight = 0
        self.peak_inflight = 0
        self.total_calls = 0

    async def collect(
        self, answer: str, metadata: dict[str, JsonValue]
    ) -> tuple[NeMoGymChatCompletion, dict[str, JsonValue]]:
        """Send text only to the judge through Gym's existing HTTP/retry/failsafe path."""
        # Sequential replace is intentional: tokens inserted by earlier replacements
        # are also expanded by the original VoiceBench generate() function.
        user = self.prompt.user.replace("{prompt}", str(metadata["prompt"]))
        if "{reference}" in self.prompt.user:
            user = user.replace("{reference}", str(metadata.get("reference", "")))
        user = user.replace("{response}", answer)
        messages = []
        if self.prompt.system is not None:
            messages.append({"role": "system", "content": self.prompt.system})
        messages.append({"role": "user", "content": user})
        params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
            {
                "messages": messages,
                "temperature": self.settings.judge_temperature,
                "top_p": self.settings.judge_top_p,
                "max_tokens": self.settings.judge_max_tokens,
                "n": self.settings.judge_n,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
            }
        )
        queued_at = perf_counter()
        async with self._slots:
            started_at = perf_counter()
            self.inflight += 1
            self.total_calls += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
            inflight_at_start = self.inflight
            try:
                judged = await call_judge(
                    self.server_client,
                    server_name=self.settings.judge_model_server.name,
                    url_path="/v1/chat/completions",
                    json=params,
                    response_model=NeMoGymChatCompletion,
                )
            finally:
                self.inflight -= 1
            return judged, {
                "judge_queue_seconds": started_at - queued_at,
                "judge_call_seconds": perf_counter() - started_at,
                "judge_inflight_at_start": inflight_at_start,
                "judge_peak_inflight": self.peak_inflight,
                "judge_configured_concurrency": self.settings.judge_max_concurrency,
            }

    def diagnostics(self, judged: NeMoGymChatCompletion) -> dict[str, JsonValue]:
        count = len(judged.choices)
        return {
            "judge_outputs": [choice.message.content for choice in judged.choices],
            "judge_finish_reasons": [choice.finish_reason for choice in judged.choices],
            "judge_requested_n": self.settings.judge_n,
            "judge_actual_n": count,
            "judge_choice_count_mismatch": count != self.settings.judge_n,
            "judge_diagnostics": [f"requested {self.settings.judge_n} choices, received {count}"]
            if count != self.settings.judge_n
            else [],
            "judge_prompt_sha256": self.prompt_sha256,
            "judge_response_id": judged.id,
            "judge_model": judged.model,
            "judge_response": judged.model_dump(mode="json"),
        }

    async def score(self, answer: str, metadata: dict[str, JsonValue]) -> ScoringResult:
        judged, timings = await self.collect(answer, metadata)
        try:
            # generate() strips every returned choice before OpenEvaluator sees it.
            outputs = [choice.message.content.strip() for choice in judged.choices]
            result = score_outputs(
                outputs,
                finish_reasons=[choice.finish_reason for choice in judged.choices],
                requested_n=self.settings.judge_n,
            )
        except (AttributeError, NotImplementedError) as error:
            result = scoring_failure(error)
        return ScoringResult(
            result.reward, result.metrics, self.diagnostics(judged) | result.details | timings, result.valid
        )
