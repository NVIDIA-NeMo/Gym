# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic scoring for the lm-eval GSM8K protocol."""

from __future__ import annotations

import re
from typing import Any

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics


STRICT_PATTERN = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEXIBLE_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
INVALID = "[invalid]"
IGNORE_PATTERNS = (r",", r"\$", r"(?s).*#### ", r"\.$")


def _regex_extract(text: str, pattern: re.Pattern[str], match_index: int) -> str:
    matches = pattern.findall(text)
    if not matches:
        return INVALID
    match = matches[match_index]
    if isinstance(match, tuple):
        match = next(part for part in match if part)
    return match.strip()


def strict_extract(text: str) -> str:
    return _regex_extract(text, STRICT_PATTERN, 0)


def flexible_extract(text: str) -> str:
    return _regex_extract(text, FLEXIBLE_PATTERN, -1)


def normalize_exact_match(text: str) -> str:
    for pattern in IGNORE_PATTERNS:
        text = re.sub(pattern, "", text)
    return text.lower()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_exact_match(prediction) == normalize_exact_match(reference))


def _assistant_text(body: BaseVerifyRequest) -> str:
    parts = []
    for output in body.response.output:
        if output.type == "message":
            parts.extend(item.text for item in output.content if item.type == "output_text")
    return "".join(parts)


class GSM8KReferenceConfig(BaseResourcesServerConfig):
    name: str = "gsm8k_reference"


class GSM8KReferenceRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    expected_answer: str
    language_code: str | None = None


class GSM8KReferenceVerifyRequest(GSM8KReferenceRunRequest, BaseVerifyRequest):
    pass


class GSM8KReferenceVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    expected_answer: str
    language_code: str | None = None
    strict_prediction: str
    flexible_prediction: str
    strict_match: float
    flexible_extract: float


class GSM8KReferenceResourcesServer(SimpleResourcesServer):
    config: GSM8KReferenceConfig

    async def verify(self, body: GSM8KReferenceVerifyRequest) -> GSM8KReferenceVerifyResponse:
        generation = _assistant_text(body)
        strict_prediction = strict_extract(generation)
        flexible_prediction = flexible_extract(generation)
        strict_score = exact_match(strict_prediction, body.expected_answer)
        flexible_score = exact_match(flexible_prediction, body.expected_answer)
        return GSM8KReferenceVerifyResponse(
            **body.model_dump(),
            reward=strict_score,
            strict_prediction=strict_prediction,
            flexible_prediction=flexible_prediction,
            strict_match=strict_score,
            flexible_extract=flexible_score,
        )

    @staticmethod
    def _score_fn(result: dict[str, Any]) -> dict[str, float]:
        return {
            "strict_match": float(result.get("strict_match", 0.0)),
            "flexible_extract": float(result.get("flexible_extract", 0.0)),
        }

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        if not tasks:
            return {}
        metrics = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="strict_prediction",
        )[0]
        metrics.update(
            compute_subset_metrics(
                tasks,
                subset_key="language_code",
                score_fn=self._score_fn,
                answer_key="strict_prediction",
            )
        )
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key = {
            name: agent_metrics[name] for name in ("mean/input_tokens", "mean/output_tokens") if name in agent_metrics
        }
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        return key


if __name__ == "__main__":
    GSM8KReferenceResourcesServer.run_webserver()
