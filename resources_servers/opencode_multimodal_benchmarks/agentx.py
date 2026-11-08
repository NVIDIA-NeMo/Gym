# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-X Goal Accuracy with full-response rubric-based judging."""

from typing import Any

from pydantic import Field

from nemo_gym.verification import JudgeError, record_parsed_judge_verdict
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
    judge_concurrency: int = Field(default=4, gt=0)
    judge_timeout: float = Field(default=300, gt=0)
    judge_attempts: int = Field(default=10, gt=0)


class AgentXServer(AgentXVerifier, VisualBenchmarksServer):
    config: AgentXConfig

    async def _judge_answer(self, task: dict[str, Any], prediction: str) -> tuple[float, str]:
        self._judge_credentials()
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
        raw = await self._judge_completion(payload)
        score = parse_judge_score(raw, task["query_type"])
        if score is None:
            raise JudgeError("Invalid Agent-X judge verdict")
        record_parsed_judge_verdict()
        return score, raw

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
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
