# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym-native verification for VStarBench, HRBench8K, VTCBench and BabyVision."""

import asyncio
import os
from typing import Any, ClassVar

import aiohttp
from pydantic import Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymChatCompletion
from nemo_gym.server_utils import request
from resources_servers.opencode_multimodal_benchmarks.mcq import (
    MCQ_SCORER,
    infer_option,
    matcher_prompt,
    visible_answer,
)
from resources_servers.opencode_multimodal_benchmarks.metrics import average_repeats
from resources_servers.opencode_multimodal_benchmarks.reporting import benchmark_report
from resources_servers.opencode_multimodal_benchmarks.scoring import BABYVISION_JUDGE_PROMPT, boxed_answer
from resources_servers.opencode_multimodal_benchmarks.vtcbench import judge_prompt, judge_score, rule_score
from responses_api_agents.opencode_multimodal_agent.media import final_answer


class VisualBenchmarksConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    judge_endpoint: str | None = None
    judge_model: str | None = None
    # Keep the secret out of saved Gym configs and the OpenCode sandbox.
    judge_api_key_env: str = "OPENAI_API_KEY"
    # A model server enables judge-call correlation and capture.
    # None selects direct HTTP judging via judge_endpoint.
    judge_server: str | None = None
    judge_concurrency: int = Field(default=1024, gt=0)
    judge_timeout: float = Field(default=120, gt=0)
    judge_attempts: int = Field(default=3, gt=0)
    vtc_llm_fallback: bool = True


class VisualVerifyRequest(BaseVerifyRequest):
    verifier_metadata: dict[str, Any]


class VisualVerifyResponse(VisualVerifyRequest, BaseVerifyResponse):
    extracted: str | None = None
    scorer: str
    extraction_method: str | None = None
    judge_model: str | None = None


class VisualBenchmarksServer(SimpleResourcesServer):
    config: VisualBenchmarksConfig
    _judge_sem: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context):
        super().model_post_init(context)
        self._judge_sem = asyncio.Semaphore(self.config.judge_concurrency)

    def _judge_credentials(self) -> str | None:
        """Require explicit routing; never select a deployment on the caller's behalf."""
        if not self.config.judge_model:
            raise JudgeError("Configure judge_model before using LLM verification")
        if self.config.judge_server is not None:
            return None
        if not self.config.judge_endpoint:
            raise JudgeError("Configure judge_server or judge_endpoint before using LLM verification")
        key = os.environ.get(self.config.judge_api_key_env)
        if not key:
            raise JudgeError(f"Missing judge credential: {self.config.judge_api_key_env}")
        return key

    async def _judge(self, prompt: str, *, temperature: float = 0) -> str:
        key = self._judge_credentials()
        payload = {
            "model": self.config.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 128,
            "stream": False,
        }
        async with self._judge_sem:
            for attempt in range(self.config.judge_attempts):
                try:
                    if self.config.judge_server is not None:
                        async with asyncio.timeout(self.config.judge_timeout):
                            completion = await call_judge(
                                self.server_client,
                                server_name=self.config.judge_server,
                                url_path="/v1/chat/completions",
                                json=payload,
                                response_model=NeMoGymChatCompletion,
                            )
                        return completion.choices[0].message.content or ""
                    response = await request(
                        "POST",
                        self.config.judge_endpoint,
                        json=payload,
                        headers={"Authorization": f"Bearer {key}"},
                        timeout=aiohttp.ClientTimeout(total=self.config.judge_timeout),
                    )
                    async with response:
                        if response.status != 200:
                            # Do not include credential-bearing HTTP headers in diagnostics.
                            raise JudgeError(f"Visual judge HTTP {response.status}")
                        value = await response.json()
                    return value["choices"][0]["message"]["content"]
                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    JudgeError,
                    KeyError,
                    IndexError,
                    ValueError,
                ) as error:
                    if attempt + 1 == self.config.judge_attempts:
                        raise JudgeError(str(error) or type(error).__name__) from error
                    await asyncio.sleep(2 ** (attempt + 1))
        raise AssertionError("Judge attempts exhausted")

    async def verify(self, body: VisualVerifyRequest) -> VisualVerifyResponse:
        task = body.verifier_metadata
        prediction = visible_answer(final_answer(body.response.model_dump()["output"]))
        benchmark = task["benchmark"]
        if not prediction.strip():
            scored = {"hit": 0, "extracted": "", "scorer": "empty-answer-v1"}
        elif benchmark == "VTCBench":
            scored = {
                "hit": rule_score(task, prediction),
                "extracted": prediction,
                "scorer": "vtcbench-public-rules-v2",
                "extraction_method": "rule",
            }
            if self.config.vtc_llm_fallback:
                scored["scorer"] = "vtcbench-public-rules-llm-v2"
                if not scored["hit"]:
                    prompt = judge_prompt(task, prediction)
                    for attempt in range(5):
                        verdict = await self._judge(prompt, temperature=attempt * (0.1 if task["options"] else 0.5))
                        hit = judge_score(task, verdict)
                        if hit is not None:
                            scored.update(
                                hit=hit,
                                extracted=verdict,
                                extraction_method="llm",
                                judge_model=self.config.judge_model,
                            )
                            break
                    else:
                        raise JudgeError("Invalid VTCBench judge verdict after 5 attempts")
        elif benchmark in ("VStarBench", "HRBench8K"):
            extracted = infer_option(prediction, task["options"])
            method = "heuristic"
            if extracted is None:
                method = "llm"
                extracted = infer_option(await self._judge(matcher_prompt(task, prediction)), task["options"])
                if extracted is None:
                    raise JudgeError("Invalid MCQ judge verdict")
            scored = {
                "hit": int(extracted == task["answer"].strip().upper()),
                "extracted": extracted,
                "scorer": MCQ_SCORER,
                "extraction_method": method,
                "judge_model": self.config.judge_model if method == "llm" else None,
            }
        elif benchmark == "BabyVision":
            extracted = boxed_answer(prediction)
            extracted = prediction if extracted is None else extracted
            verdict = (
                (
                    await self._judge(
                        BABYVISION_JUDGE_PROMPT.format(
                            question=task["question"],
                            groundtruth=task["answer"],
                            modeloutput=extracted,
                        )
                    )
                )
                .strip()
                .lower()
            )
            if verdict not in ("true", "false"):
                raise JudgeError("Invalid BabyVision judge verdict")
            scored = {
                "hit": int(verdict == "true"),
                "extracted": extracted,
                "scorer": "babyvision-judge-v1",
                "judge_model": self.config.judge_model,
            }
        else:
            raise ValueError(f"Unsupported visual benchmark: {benchmark}")
        return VisualVerifyResponse(**body.model_dump(), reward=float(scored.pop("hit")), **scored)

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        grouped = {}
        for repeats in tasks:
            if repeats:
                benchmark = repeats[0]["verifier_metadata"]["benchmark"]
                grouped.setdefault(benchmark, []).append(repeats)
        metrics = {}
        for benchmark, benchmark_tasks in grouped.items():
            averaged, balanced = average_repeats(benchmark_tasks)
            rows = [row["verifier_metadata"] | {"hit": row["reward"]} for row in averaged]
            report = benchmark_report(benchmark, rows)
            value = sum(row["hit"] for row in rows) / len(rows)
            if benchmark == "HRBench8K":
                value = next(row["accuracy"] for row in report if row["cycle"] == "Average" and row["type"] == "all")
            expected_values = {row["expected_samples"] for row in rows if "expected_samples" in row}
            if len(expected_values) > 1:
                raise ValueError("Inconsistent prepared dataset sample counts")
            expected = next(iter(expected_values), len(rows))
            complete = balanced and len(rows) == expected
            metrics[f"{benchmark}/completed"] = len(rows)
            metrics[f"{benchmark}/expected"] = expected
            metrics[f"{benchmark}/complete"] = complete
            metrics[f"{benchmark}/" + ("accuracy" if complete else "partial_accuracy")] = value
            metrics[f"{benchmark}/report"] = report
        if len(grouped) == 1:
            benchmark = next(iter(grouped))
            if metrics[f"{benchmark}/complete"]:
                metrics["accuracy"] = metrics[f"{benchmark}/accuracy"]
        return metrics

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self.compute_metrics(tasks)

    def get_key_metrics(self, agent_metrics):
        return {key: value for key, value in agent_metrics.items() if key == "accuracy" or key.endswith("/accuracy")}


if __name__ == "__main__":
    VisualBenchmarksServer.run_webserver()
