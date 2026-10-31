# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a GenRM policy against chat preferences."""

import re
from importlib import import_module
from typing import Any, ClassVar, Literal

import json_repair
import tiktoken
from pydantic import BaseModel, Field, PrivateAttr

from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.genrm_compare.utils import GenRMOutputParseError, parse_genrm_output


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for name, value in metrics.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, key))
        else:
            flattened[key] = value
    return flattened


class GenRMEvalConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    scoring_module: str
    metric_key: str
    additional_metric_keys: list[str] = Field(default_factory=list)
    tokenizer_model: str = "gpt-4o"
    verdict_format: Literal["genrm", "brackets", "json_ab"] = "genrm"


class TaskData(BaseModel):
    id: str = Field(json_schema_extra={"consumed_by": ["provenance"]})
    verifier_metadata: dict[str, Any] = Field(json_schema_extra={"consumed_by": ["verify", "metrics"]})


class GenRMEvalVerifyRequest(TaskData, BaseVerifyRequest):
    pass


class GenRMEvalVerifyResponse(TaskData, BaseVerifyResponse):
    predicted_ranking: float | None
    request_error: bool
    response_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)


class GenRMEvalResourcesServer(SimpleResourcesServer):
    """A stateless verifier; inference is handled by simple_agent."""

    ray_enabled = False
    config: GenRMEvalConfig
    _encoding: tiktoken.Encoding = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._encoding = tiktoken.encoding_for_model(self.config.tokenizer_model)

    async def verify(self, body: GenRMEvalVerifyRequest) -> GenRMEvalVerifyResponse:
        text = "".join(
            content.text
            for item in body.response.output
            if item.type == "message"
            for content in item.content
            if content.type == "output_text"
        )
        reasoning = "".join(
            content.text
            for item in body.response.output
            if item.type == "reasoning"
            for content in item.summary or []
            if content.type == "summary_text"
        )
        if self.config.verdict_format != "genrm":
            # Published MMRB verdicts encode direction only; normalize to the GenRM scale.
            ranking = parse_ab_ranking(text, self.config.verdict_format)
        else:
            ranking = self._parse_genrm_ranking(text)
        return GenRMEvalVerifyResponse(
            **body.model_dump(),
            predicted_ranking=ranking,
            request_error=body.response.error is not None,
            response_tokens=len(self._encoding.encode(text, disallowed_special=())),
            reasoning_tokens=len(self._encoding.encode(reasoning, disallowed_special=())),
            reward=import_module(self.config.scoring_module).score(body.verifier_metadata, ranking),
        )

    @staticmethod
    def _parse_genrm_ranking(text: str) -> float | None:
        try:
            # A missing ranking must fail parsing rather than become a default verdict.
            _, _, ranking = parse_genrm_output(
                text, default_score=0.0, default_ranking=float("nan"), raise_on_fail=True
            )
            if ranking not in (1, 2, 3, 4, 5, 6):
                ranking = None
        except GenRMOutputParseError:
            # Missing rankings are scored, not skipped.
            ranking = None
        return ranking

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return flatten_metrics(import_module(self.config.scoring_module).compute_metrics(tasks))

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self.compute_metrics(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keys = (
            self.config.metric_key,
            *self.config.additional_metric_keys,
            "median/response_tokens",
            "median/reasoning_tokens",
        )
        return {name: agent_metrics[name] for name in keys if name in agent_metrics}


def parse_ab_ranking(text: str, verdict_format: Literal["brackets", "json_ab"]) -> float | None:
    """Parse the published MMRB A/B formats without inventing missing verdicts."""
    text = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.DOTALL)
    text = re.split(r"<(?:think|thinking)>", text, maxsplit=1)[0]
    if verdict_format == "brackets":
        # MMRB1's reference parser checks A first if both markers occur.
        label = "A" if "[[A]]" in text else "B" if "[[B]]" in text else None
    else:
        parsed = json_repair.loads(text)
        label = parsed.get("better_response") if isinstance(parsed, dict) else None
    return 3.0 if label == "A" else 4.0 if label == "B" else None


if __name__ == "__main__":
    GenRMEvalResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = GenRMEvalResourcesServer.run_webserver()
