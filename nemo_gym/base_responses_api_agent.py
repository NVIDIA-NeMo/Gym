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
from typing import Any, Literal, Union

from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


class BaseResponsesAPIAgentConfig(BaseRunServerInstanceConfig):
    pass


class BaseResponsesAPIAgent(BaseServer):
    config: BaseResponsesAPIAgentConfig


class AgentArtifactRef(BaseModel):
    """Versioned reference to an Agent-owned artifact.

    Examples include prompts, skill directories, memory indexes, ACE playbooks,
    checkpoints, adapters, or model-routing config. The Processor records these
    refs for provenance; the Agent owns how to materialize and use them.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    hash: str
    uri: str | None = None
    metadata: dict[str, Any] = {}


class TrajectoryStep(BaseModel):
    """Typed trajectory step delivered to an Agent's `/observe` hook.

    The payload intentionally reuses existing Gym contracts instead of defining
    a parallel observation schema. Environment-specific events can use
    `kind="custom"` and carry their native serialized schema in `payload`.
    """

    model_config = ConfigDict(extra="allow")

    kind: Literal["response", "tool_result", "terminated", "truncated", "custom"]
    payload: Union[
        NeMoGymResponse, NeMoGymResponseOutputItem, NeMoGymFunctionCallOutput, BaseVerifyResponse, dict[str, Any]
    ]
    metadata: dict[str, Any] = {}


class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, AggregateMetricsMixin, SimpleServer):
    config: BaseResponsesAPIAgentConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        app.post("/v1/responses")(self.responses)
        app.post("/observe")(self.observe)
        app.post("/run")(self.run)
        app.post("/aggregate_metrics")(self.aggregate_metrics)

        return app

    # TODO: right now there is no validation on the TypedDict NeMoGymResponseCreateParamsNonStreaming
    # We should explicitly add validation at this server level or we should explicitly not validate so that there is flexibility in this API.
    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    @abstractmethod
    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        pass

    async def observe(self, body: TrajectoryStep = Body()) -> None:
        """Default Agent capability: observing existing Gym events is a no-op."""
        return None

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Default: same RewardProfiler aggregation as resources server. Override to proxy."""
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )
