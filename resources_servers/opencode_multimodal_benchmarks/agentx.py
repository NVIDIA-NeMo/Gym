# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-X Goal Accuracy with full-response rubric-based judging."""

import asyncio

import aiohttp

from nemo_gym.client import request
from nemo_gym.core import NeMoGymChatCompletion
from nemo_gym.verification import JUDGE_FAILURES, JudgeError, call_judge, is_judge_failure
from resources_servers.opencode_multimodal_benchmarks.agentx_rubric import (
    OFFICIAL_GOAL_SYSTEM_PROMPT,
    judge_prompt,
    parse_judge_score,
)
from resources_servers.opencode_multimodal_benchmarks.app import (
    VisualBenchmarksConfig,
    VisualBenchmarksServer,
)
from resources_servers.opencode_multimodal_benchmarks.metrics import average_repeats
from resources_servers.opencode_multimodal_benchmarks.verifier import AgentXVerifier
from resources_servers.opencode_multimodal_benchmarks.verifier import AgentXVerifyResponse as AgentXVerifyResponse
from resources_servers.opencode_multimodal_benchmarks.verifier_fixtures import agentx_fixture


VERIFIER_FIXTURE = agentx_fixture()


class AgentXConfig(VisualBenchmarksConfig):
    judge_concurrency: int = 4
    judge_timeout: float = 300
    judge_attempts: int = 10


class AgentXServer(AgentXVerifier, VisualBenchmarksServer):
    config: AgentXConfig

    async def _judge_answer(self, task: dict, prediction: str) -> tuple[float, str]:
        key = self._judge_credentials()
        payload = {
            "model": self.config.judge_model,
            "messages": [
                {"role": "system", "content": OFFICIAL_GOAL_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt(task, prediction)},
            ],
            "temperature": 0,
            "max_tokens": 2048,
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
                        value = completion.model_dump()
                    else:
                        response = await request(
                            "POST",
                            self.config.judge_endpoint,
                            json=payload,
                            headers={"Authorization": f"Bearer {key}"},
                            timeout=aiohttp.ClientTimeout(total=self.config.judge_timeout),
                        )
                        async with response:
                            if response.status != 200:
                                raise JudgeError(f"Agent-X judge HTTP {response.status}")
                            value = await response.json()
                    raw = value["choices"][0]["message"]["content"]
                    score = parse_judge_score(raw, task["query_type"])
                    if score is None:
                        raise JudgeError("Invalid Agent-X judge verdict")
                    return score, raw
                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    *JUDGE_FAILURES,
                    KeyError,
                    IndexError,
                    ValueError,
                    TypeError,
                ) as error:
                    if isinstance(error, JUDGE_FAILURES) and not is_judge_failure(error):
                        raise
                    if attempt + 1 == self.config.judge_attempts:
                        if is_judge_failure(error):
                            raise
                        raise JudgeError(str(error) or type(error).__name__) from error
                    await asyncio.sleep(min(30, 2 ** (attempt + 1)))
        raise AssertionError("Judge attempts exhausted")

    def compute_metrics(self, tasks: list) -> dict:
        rows, balanced = average_repeats(tasks)
        if not rows:
            return {}
        counts = {row["verifier_metadata"]["expected_samples"] for row in rows}
        if len(counts) != 1:
            raise ValueError("Inconsistent Agent-X sample counts")
        expected = next(iter(counts))
        complete = balanced and len(rows) == expected
        metrics = {
            "AgentX_GoalAcc/completed": len(rows),
            "AgentX_GoalAcc/expected": expected,
            "AgentX_GoalAcc/complete": complete,
        }
        score = sum(row["reward"] for row in rows) / len(rows)
        metrics["AgentX_GoalAcc/" + ("accuracy" if complete else "partial_accuracy")] = score
        if complete:
            metrics["accuracy"] = score
        for modality in ("Image", "Video"):
            subset = [row for row in rows if row["verifier_metadata"]["modality"] == modality]
            metrics[f"AgentX_GoalAcc/{modality}/completed"] = len(subset)
            if subset:
                metrics[f"AgentX_GoalAcc/{modality}/" + ("accuracy" if complete else "partial_accuracy")] = sum(
                    row["reward"] for row in subset
                ) / len(subset)
        return metrics


if __name__ == "__main__":
    AgentXServer.run_webserver()
