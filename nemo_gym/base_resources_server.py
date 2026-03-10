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
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
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
        description="Per-task metrics from describe_dataframe, one dict per task.",
    )
    agent_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Overall metrics from describe_dataframe across all rollouts.",
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


def compute_aggregate_metrics(
    verify_responses: List[Dict[str, Any]],
    describe_dataframe_fn=None,
    get_key_metrics_fn=None,
) -> AggregateMetrics:
    """Shared aggregation logic for /aggregate_metrics.

    Uses RewardProfiler to compute group-level (per-task) and agent-level metrics.
    Optionally accepts custom describe_dataframe and get_key_metrics functions
    for benchmark-specific customization.
    """
    from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
    from nemo_gym.reward_profile import RewardProfiler

    if not verify_responses:
        return AggregateMetrics()

    if describe_dataframe_fn:

        class _Profiler(RewardProfiler):
            def describe_dataframe(self, df):
                return describe_dataframe_fn(df)

        rp = _Profiler()
    else:
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

    def describe_dataframe(self, df: Any) -> Any:
        """Override to add custom per-group aggregation metrics.

        Called by /aggregate_metrics for each task group AND for the overall dataset.
        Default: delegates to RewardProfiler.describe_dataframe() (mean/max/min/median/std).

        Override pattern:
            result = super().describe_dataframe(df)
            # Add custom rows/columns (e.g. pass@k)
            return result
        """
        from nemo_gym.reward_profile import RewardProfiler

        return RewardProfiler().describe_dataframe(df)

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Override to select headline metrics for this benchmark.

        Default: all mean/* entries from agent_metrics.
        """
        return {k: v for k, v in agent_metrics.items() if k.startswith("mean/")}

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Compute aggregate metrics from verify responses using RewardProfiler.

        Uses self.describe_dataframe() as the aggregation function, so subclasses
        can add custom metrics by overriding describe_dataframe().
        """
        return compute_aggregate_metrics(
            body.verify_responses,
            describe_dataframe_fn=self.describe_dataframe,
            get_key_metrics_fn=self.get_key_metrics,
        )
