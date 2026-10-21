# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-X Goal Accuracy with full-response rubric-based judging."""

import asyncio

import aiohttp

from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymChatCompletion
from nemo_gym.server_utils import request
from resources_servers.opencode_multimodal_benchmarks.agentx_rubric import (
    OFFICIAL_GOAL_SYSTEM_PROMPT,
    final_response,
    judge_prompt,
    parse_judge_score,
)
from resources_servers.opencode_multimodal_benchmarks.app import (
    VisualBenchmarksConfig,
    VisualBenchmarksServer,
    VisualVerifyRequest,
    VisualVerifyResponse,
)
from resources_servers.opencode_multimodal_benchmarks.metrics import average_repeats
from responses_api_agents.opencode_multimodal_agent.media import final_answer


class AgentXConfig(VisualBenchmarksConfig):
    judge_concurrency: int = 4
    judge_timeout: float = 300
    judge_attempts: int = 10


class AgentXVerifyResponse(VisualVerifyResponse):
    judge_raw: str


class AgentXServer(VisualBenchmarksServer):
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
                    JudgeError,
                    KeyError,
                    IndexError,
                    ValueError,
                    TypeError,
                ) as error:
                    if attempt + 1 == self.config.judge_attempts:
                        raise JudgeError(str(error) or type(error).__name__) from error
                    await asyncio.sleep(min(30, 2 ** (attempt + 1)))
        raise AssertionError("Judge attempts exhausted")

    async def verify(self, body: VisualVerifyRequest) -> AgentXVerifyResponse:
        task = body.verifier_metadata
        if task["benchmark"] != "AgentX_GoalAcc":
            raise ValueError(f"Not an Agent-X task: {task['benchmark']}")
        prediction = final_response(final_answer(body.response.model_dump()["output"]))
        if prediction:
            score, raw = await self._judge_answer(task, prediction)
        else:
            score, raw = 0.0, "Score: 0; no final answer."
        return AgentXVerifyResponse(
            **body.model_dump(),
            reward=score,
            extracted=prediction,
            scorer="agentx-goal-full-response-v1",
            judge_model=self.config.judge_model,
            judge_raw=raw,
        )

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
