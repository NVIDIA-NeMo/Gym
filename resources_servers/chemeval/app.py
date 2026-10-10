# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym ChemEval resource server with a bundled standalone scorer."""

import asyncio
import math
import re
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

from pydantic import Field, JsonValue, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.chemeval.english_judge import build_judge_messages, parse_judge_verdict
from resources_servers.chemeval.loader import load_grader
from resources_servers.chemeval.task_data import TaskData
from resources_servers.chemeval.verifier_fixture import create_server


THINK_BLOCK = re.compile(r"<(think|thinking)\b[^>]*>.*?(?:</\1\s*>|$)", re.I | re.S)
THINK_END = re.compile(r"</(?:think|thinking)\s*>", re.I)


def final_text(text: str | None) -> str:
    return THINK_END.split(THINK_BLOCK.sub("", text or ""))[-1].strip()


def finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


class ChemEvalResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    grader_path: str = str(Path(__file__).parent / "grading.py")
    grading_max_concurrency: int = Field(default=4, ge=1)
    judge_model_server: ModelServerRef | None = None
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(
            input=[], temperature=0, max_output_tokens=16384
        )
    )
    judge_max_concurrency: int = Field(default=32, ge=1)


class ChemEvalVerifyRequest(BaseVerifyRequest):
    verifier_metadata: TaskData


class ChemEvalVerifyResponse(BaseVerifyResponse, ChemEvalVerifyRequest):
    predicted_answer: str | None = None
    grading_metrics: dict[str, Any] = Field(default_factory=dict)
    scoring_error: str | None = None
    judgement: str | None = None
    judge_score: float | None = None
    judge_parse_ok: bool | None = None
    judge_v2: dict[str, JsonValue] | None = None
    judge_response: NeMoGymResponse | None = None


class ChemEvalResourcesServer(SimpleResourcesServer):
    ray_enabled = False

    config: ChemEvalResourcesServerConfig
    _grader: ModuleType = PrivateAttr()
    _grading_semaphore: asyncio.Semaphore = PrivateAttr()
    _judge_semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self._grader = load_grader(self.config.grader_path)
        self._grading_semaphore = asyncio.Semaphore(self.config.grading_max_concurrency)
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)
        super().model_post_init(context)

    async def verify(self, body: ChemEvalVerifyRequest) -> ChemEvalVerifyResponse:
        generation = final_text(body.response.output_text)
        if not generation:
            return ChemEvalVerifyResponse(**body.model_dump(), reward=0.0)
        metadata = body.verifier_metadata
        if metadata.family != "judged":
            sample = metadata.model_dump(exclude_none=True) | {"generation": generation}
            original_keys = set(sample)
            try:
                async with self._grading_semaphore:
                    await asyncio.to_thread(self._grader.grade, sample)
            except (ValueError, TypeError, OverflowError, ZeroDivisionError, RecursionError) as error:
                return ChemEvalVerifyResponse(
                    **body.model_dump(), reward=0.0, scoring_error=f"{type(error).__name__}: {error}"
                )
            metrics = finite_json({key: value for key, value in sample.items() if key not in original_keys})
            score = metrics.pop("score")
            predicted = metrics.pop("predicted_answer", None)
            return ChemEvalVerifyResponse(
                **body.model_dump(),
                reward=score if score is not None else 0.0,
                predicted_answer=predicted,
                grading_metrics=metrics,
            )
        if self.config.judge_model_server is None:
            raise JudgeError("Configure judge_model_server to evaluate ChemEval judged tasks")
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            self.config.judge_responses_create_params.model_dump(exclude_unset=True)
            | {
                "input": build_judge_messages(
                    rubric=metadata.judge_rubric,
                    question=metadata.judge_question,
                    candidate=generation,
                    reference=metadata.expected_answer,
                )
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
        verdict = final_text(judged.output_text)
        score, payload = parse_judge_verdict(verdict, scale=metadata.judge_scale)
        return ChemEvalVerifyResponse(
            **body.model_dump(),
            reward=score if score is not None else 0.0,
            predicted_answer=generation,
            judgement=verdict,
            judge_score=score if score is not None else 0.0,
            judge_parse_ok=score is not None,
            judge_response=judged,
            judge_v2=payload,
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, float | int]:
        by_task = defaultdict(list)
        labels = {}
        for rollouts in tasks:
            if not rollouts:
                continue
            metadata = rollouts[0]["verifier_metadata"]
            task = metadata["task"]
            by_task[task].append(sum(row["reward"] for row in rollouts) / len(rollouts))
            labels[task] = metadata
        groups = defaultdict(list)
        metrics = {}
        for task, scores in by_task.items():
            score = sum(scores) / len(scores)
            metrics[f"by_task/{task}/score"] = score
            groups["overall_score"].append(score)
            for key in ("level", "dimension", "family"):
                groups[f"by_{key}/{labels[task][key]}/score"].append(score)
        metrics.update({key: sum(scores) / len(scores) for key, scores in groups.items()})
        levels = [value for key, value in metrics.items() if key.startswith("by_level/")]
        if levels:
            metrics["level_macro_score"] = sum(levels) / len(levels)
            metrics["num_tasks"] = len(by_task)
        return metrics


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_server,
    request_model=ChemEvalVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "fixture_cases.jsonl",
)


if __name__ == "__main__":
    ChemEvalResourcesServer.run_webserver()
