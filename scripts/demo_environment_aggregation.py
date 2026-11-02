# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline design demo: Environment-owned aggregation with Resources-owned scoring policies.

Run from the repository root: python -m scripts.demo_environment_aggregation

The proposed change is DemoEnvironmentServer.aggregate_metrics. It subclasses the
real single_agent_turn server, keeping its Resources routing and verification flag.
Only the transport is replaced with an in-process Resources aggregation endpoint.
No production server/config is changed and no model or network service is started.

Input represents a persisted Collector batch: selected attempts, task/repeat IDs,
verifier fields, and ng_perf summaries already derived from execution observations.
Observation collection, model-call joins, and shard/attempt selection are upstream
prerequisites; this demo does not implement or exercise them.

Resources still receives ng_perf for existing per-repeat token statistics and
confidence intervals. Its shared reducer also calculates perf_summary today, so
this demo intentionally calculates that block twice. A production migration must
separate the complete execution reducer before removing ng_perf from Resources.
"""

import asyncio
import json
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, JsonValue

from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import (
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseServerConfig,
    ResourcesServerRef,
)
from nemo_gym.metrics import AggregateMetricsMixin, compute_aggregate_metrics, compute_perf_summary


class DemoEnvironmentServer(SingleAgentTurnEnvironmentServer):
    """Compose benchmark metrics and execution statistics from the same saved batch."""

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        # Preserve Resources' per-repeat token statistics and confidence intervals.
        # They consume ng_perf too; only final perf_summary ownership moves here.
        benchmark = await super().aggregate_metrics(body)
        # Masked samples still consumed compute. Missing observations stay missing,
        # and skipping verification does not suppress measured execution statistics.
        performance = compute_perf_summary(
            [row["ng_perf"] for row in body.verify_responses if isinstance(row.get("ng_perf"), dict)],
            total_rollouts=len(body.verify_responses),
        )
        return benchmark.model_copy(update={"perf_summary": performance})


class _StepCounts(BaseModel):
    passed: int = Field(ge=0)
    total: int = Field(gt=0)


class _PooledSubtaskMetrics(AggregateMetricsMixin):
    def compute_metrics(self, tasks: list[list[dict[str, JsonValue]]]) -> dict[str, float]:
        counts = [_StepCounts.model_validate(row) for task in tasks for row in task]
        return {"micro/subtask_accuracy": sum(row.passed for row in counts) / sum(row.total for row in counts)}


class _MeanTaskMetrics(AggregateMetricsMixin):
    def compute_metrics(self, tasks: list[list[dict[str, JsonValue]]]) -> dict[str, float]:
        # Equal task weight, even if tasks have different numbers of repeats.
        task_accuracies = []
        for task in tasks:
            counts = [_StepCounts.model_validate(row) for row in task]
            task_accuracies.append(sum(row.passed / row.total for row in counts) / len(counts))
        return {"macro/subtask_accuracy": sum(task_accuracies) / len(task_accuracies)}


class _InProcessResponse:
    ok = True

    def __init__(self, metrics: AggregateMetrics) -> None:
        self._body = metrics.model_dump_json().encode()

    async def read(self) -> bytes:
        return self._body


class _InProcessResourcesClient(ServerClient):
    """Run actual Gym reducers while replacing only the Resources HTTP transport."""

    policies: dict[str, AggregateMetricsMixin]
    requests: list[AggregateMetricsRequest] = Field(default_factory=list)

    async def post(self, server_name: str, url_path: str, *, json: AggregateMetricsRequest) -> _InProcessResponse:
        if url_path != "/aggregate_metrics":
            raise ValueError(f"The offline demo only supports /aggregate_metrics, got {url_path!r}")
        # Exercise the wire models, including JSON conversion, at the server boundary.
        body = AggregateMetricsRequest.model_validate_json(json.model_dump_json())
        self.requests.append(body)
        policy = self.policies[server_name]
        return _InProcessResponse(
            compute_aggregate_metrics(
                body.verify_responses,
                compute_metrics_fn=policy.compute_metrics,
                compute_repeat_metrics_fn=policy.compute_repeat_metrics,
                get_key_metrics_fn=policy.get_key_metrics,
            )
        )


def make_demo_environment(
    policy: Literal["pooled_subtasks", "mean_tasks"], *, skip_verification: bool = False
) -> DemoEnvironmentServer:
    """Bind one Env implementation to either Resources-owned benchmark policy."""
    client = _InProcessResourcesClient(
        head_server_config=BaseServerConfig(host="unused", port=1),
        global_config_dict=OmegaConf.create({"agent": {"responses_api_agents": {"demo": {}}}}),
        policies={"pooled_subtasks": _PooledSubtaskMetrics(), "mean_tasks": _MeanTaskMetrics()},
    )
    return DemoEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            name=f"{policy}_env",
            host="unused",
            port=8000,
            entrypoint="demo_environment_aggregation.py",
            agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
            resources_server=ResourcesServerRef(type="resources_servers", name=policy),
            cleanup_timeout_seconds=10,
            skip_verification=skip_verification,
        ),
        server_client=client,
    )


def example_batch() -> AggregateMetricsRequest:
    """Two synthetic scored tasks with Collector-shaped execution summaries."""
    return AggregateMetricsRequest(
        verify_responses=[
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "reward": 1.0,
                "passed": 1,
                "total": 1,
                "ng_perf": {"completion_tokens": 100, "num_turns": 1, "token_observability_coverage": 1.0},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "reward": 0.0,
                "passed": 0,
                "total": 9,
                "ng_perf": {"completion_tokens": 900, "num_turns": 3, "token_observability_coverage": 1.0},
            },
        ]
    )


async def main() -> None:
    """Print the different benchmark scores and identical execution statistics."""
    for policy in ("pooled_subtasks", "mean_tasks"):
        environment = make_demo_environment(policy)
        result = await environment.aggregate_metrics(example_batch())
        print(
            json.dumps(
                {
                    "environment": environment.config.name,
                    "benchmark_metrics": {
                        key: value for key, value in result.agent_metrics.items() if key.endswith("/subtask_accuracy")
                    },
                    "execution_metrics": {
                        key: value
                        for key, value in (result.perf_summary or {}).items()
                        if key in {"mean_completion_tokens", "total_completion_tokens", "mean_num_turns"}
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
