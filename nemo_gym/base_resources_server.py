# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import RewardProfiler
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


class BaseResourcesServerConfig(BaseRunServerInstanceConfig):
    pass


class AggregateMetricsRequest(BaseModel):
    """POST body for /aggregate_metrics.

    Each item is a stripped verify response dict containing at minimum:
    - TASK_INDEX_KEY_NAME: int
    - "reward": float
    """

    verify_responses: List[Dict[str, Any]]


class AggregateMetrics(BaseModel):
    """Response from /aggregate_metrics.

    Flat string keys for direct logging to W&B/MLflow.
    """

    group_level_metrics: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-task metrics (one dict per task) from RewardProfiler baseline stats.",
    )
    agent_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Overall metrics across all rollouts (RewardProfiler baseline + compute_metrics).",
    )
    key_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Headline metrics for this benchmark. Subset of agent_metrics.",
    )


class BaseResourcesServer(BaseServer):
    config: BaseResourcesServerConfig


class BaseRunRequest(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class BaseVerifyRequest(BaseRunRequest):
    response: NeMoGymResponse


class BaseVerifyResponse(BaseVerifyRequest):
    reward: float


class BaseSeedSessionRequest(BaseModel):
    pass


class BaseSeedSessionResponse(BaseModel):
    pass


def _group_by_task(verify_responses: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group verify responses by task index, returning a list of per-task rollout lists."""
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for vr in verify_responses:
        groups[vr.get(TASK_INDEX_KEY_NAME, 0)].append(vr)
    return [groups[k] for k in sorted(groups)]


def compute_aggregate_metrics(
    verify_responses: List[Dict[str, Any]],
    compute_metrics_fn=None,
    get_key_metrics_fn=None,
) -> AggregateMetrics:
    """Shared aggregation logic for /aggregate_metrics.

    RewardProfiler runs with defaults to produce baseline stats (mean/max/min/median/std)
    for both group-level (per-task) and agent-level metrics.

    Optionally accepts custom functions for benchmark-specific customization:
      - compute_metrics_fn: receives ALL verify responses grouped by task
        (List[List[Dict]]) for metrics that need the full dataset (e.g. confidence
        intervals, cross-task statistics, pass@k). Returned dict is merged into agent_metrics.
      - get_key_metrics_fn: select headline metrics from agent_metrics
    """
    if not verify_responses:
        return AggregateMetrics()

    rp = RewardProfiler()

    rows = []
    results = []
    for vr in verify_responses:
        rows.append(
            {
                TASK_INDEX_KEY_NAME: vr.get(TASK_INDEX_KEY_NAME, 0),
                ROLLOUT_INDEX_KEY_NAME: vr.get(ROLLOUT_INDEX_KEY_NAME, 0),
                "agent_ref": {"name": "agent"},
            }
        )
        results.append(vr if "response" in vr else {**vr, "response": {}})

    group_level_metrics, agent_level_metrics = rp.profile_from_data(rows, results)

    # Flatten agent_level_metrics (one entry since we use a single agent name)
    agent_metrics: Dict[str, Any] = {}
    for entry in agent_level_metrics:
        for k, v in entry.items():
            if k != "agent_ref":
                agent_metrics[k] = v

    serialized_group = rp.prepare_for_serialization(group_level_metrics)
    serialized_agent = rp.prepare_for_serialization([agent_metrics])[0] if agent_metrics else {}

    # Custom metrics computed from all raw verify responses grouped by task
    if compute_metrics_fn:
        tasks = _group_by_task(verify_responses)
        serialized_agent.update(compute_metrics_fn(tasks))

    if get_key_metrics_fn:
        key_metrics = get_key_metrics_fn(serialized_agent)
    else:
        key_metrics = {k: v for k, v in serialized_agent.items() if k.startswith("mean/")}

    return AggregateMetrics(
        group_level_metrics=serialized_group,
        agent_metrics=serialized_agent,
        key_metrics=key_metrics,
    )


class SimpleResourcesServer(BaseResourcesServer, SimpleServer):
    config: BaseResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        app.post("/seed_session")(self.seed_session)
        app.post("/verify")(self.verify)
        app.post("/aggregate_metrics")(self.aggregate_metrics)

        return app

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        return BaseSeedSessionResponse()

    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Override to compute custom metrics from all verify responses.

        Receives verify responses grouped by task: tasks[i] is a list of rollout
        dicts for task i. Each dict has at minimum reward, plus any custom fields
        from the verify response (e.g. symbolic_correct, judgement-gen-base).

        Use for metrics that need the full dataset at once:
        - Confidence intervals (ArenaMetrics)
        - Cross-task statistics (std_dev_across_runs)
        - pass@k with proper combinatorial computation

        The returned dict is merged into agent_metrics.
        Default: empty dict (no additional metrics).
        """
        return {}

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Override to select headline metrics for this benchmark.

        Default: all mean/* entries from agent_metrics.
        """
        return {k: v for k, v in agent_metrics.items() if k.startswith("mean/")}

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Compute aggregate metrics from verify responses.

        RewardProfiler provides baseline stats. Override compute_metrics() and/or
        get_key_metrics() for benchmark-specific customization.
        """
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )
