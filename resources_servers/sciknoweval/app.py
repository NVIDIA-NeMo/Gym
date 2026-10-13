# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SciKnowEval V2: exact graders and task-specific LLM judge scores."""

import asyncio
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.sciknoweval.grading import grade_task, parse_judgement
from resources_servers.sciknoweval.task_data import TaskData
from resources_servers.sciknoweval.verifier_fixture import create_server


THINK_BLOCK = re.compile(r"<(think|thinking)\b[^>]*>.*?(?:</\1\s*>|$)", re.I | re.S)
THINK_END = re.compile(r"</(?:think|thinking)\s*>", re.I)


class SciKnowEvalResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef | None = None
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(
            input=[], temperature=0, max_output_tokens=2048
        )
    )
    judge_max_concurrency: int = Field(default=32, ge=1)


class SciKnowEvalVerifyRequest(BaseVerifyRequest):
    verifier_metadata: TaskData


class SciKnowEvalVerifyResponse(BaseVerifyResponse, SciKnowEvalVerifyRequest):
    predicted_answer: str | None = None
    symbolic_correct: bool | None = None
    symbolic_correct_normalized: bool | None = None
    extraction_ok: bool | None = None
    judgement: str | None = None
    judge_score: float | None = None
    judge_parse_ok: bool | None = None
    judge_response: NeMoGymResponse | None = None


class SciKnowEvalResourcesServer(SimpleResourcesServer):
    ray_enabled = False

    config: SciKnowEvalResourcesServerConfig
    _judge_semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)
        super().model_post_init(context)

    async def verify(self, body: SciKnowEvalVerifyRequest) -> SciKnowEvalVerifyResponse:
        text = THINK_END.split(THINK_BLOCK.sub("", body.response.output_text or ""))[-1].strip()
        metadata = body.verifier_metadata
        result = grade_task(metadata.model_dump(), text)
        if "symbolic_correct" in result:
            return SciKnowEvalVerifyResponse(**body.model_dump(), **result, reward=float(result["symbolic_correct"]))
        if not result["predicted_answer"]:
            return SciKnowEvalVerifyResponse(
                **body.model_dump(), **result, reward=0.0, judge_score=0.0, judge_parse_ok=False
            )
        if self.config.judge_model_server is None:
            raise JudgeError("Configure judge_model_server to evaluate SciKnowEval judged tasks")
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            self.config.judge_responses_create_params.model_dump(exclude_unset=True)
            | {
                "input": [
                    {"role": "system", "content": metadata.judge_system},
                    {
                        "role": "user",
                        "content": metadata.judge_prefix + result["predicted_answer"] + metadata.judge_suffix,
                    },
                ]
            }
        )
        async with self._judge_semaphore:
            judged = await call_judge(
                self.server_client,
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=params,
                response_model=NeMoGymResponse,
            )
        verdict = THINK_END.split(THINK_BLOCK.sub("", judged.output_text or ""))[-1].strip()
        score = parse_judgement(verdict, metadata.judge_scale)
        return SciKnowEvalVerifyResponse(
            **body.model_dump(),
            **result,
            reward=score if score is not None else 0.0,
            judgement=verdict,
            judge_score=score if score is not None else 0.0,
            judge_parse_ok=score is not None,
            judge_response=judged,
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, float]:
        groups = defaultdict(list)
        for rollouts in tasks:
            if not rollouts:
                continue
            mean = sum(row["reward"] for row in rollouts) / len(rollouts)
            metadata = rollouts[0]["verifier_metadata"]
            groups["overall_score"].append(mean)
            for key in ("level", "domain", "answer_type"):
                if metadata.get(key):
                    groups[f"by_{key}/{metadata[key]}/score"].append(mean)
        metrics = {key: sum(values) / len(values) for key, values in groups.items()}
        levels = [value for key, value in metrics.items() if key.startswith("by_level/")]
        if levels:
            metrics["level_macro_score"] = sum(levels) / len(levels)
        metrics.update(
            {
                f"mean/{level}": metrics[f"by_level/{level}/score"]
                for level in ("L1", "L2", "L3", "L4", "L5")
                if f"by_level/{level}/score" in metrics
            }
        )
        return metrics

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Place per-level headline scores with the opening mean metrics in saved JSON."""
        result = await super().aggregate_metrics(body)
        result.agent_metrics = {
            **{key: value for key, value in result.agent_metrics.items() if key.startswith("mean/")},
            **{key: value for key, value in result.agent_metrics.items() if not key.startswith("mean/")},
        }
        return result


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_server,
    request_model=SciKnowEvalVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "fixture_cases.jsonl",
)


if __name__ == "__main__":
    SciKnowEvalResourcesServer.run_webserver()
