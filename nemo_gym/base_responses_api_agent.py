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
from typing import Any, Optional

from fastapi import Body, FastAPI, Request

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.server_utils import (
    ROLLOUT_HEADER,
    BaseRunServerInstanceConfig,
    BaseServer,
    SimpleServer,
    apply_rollout_prefix,
    rollout_id_from_run_body,
)


class BaseResponsesAPIAgentConfig(BaseRunServerInstanceConfig):
    pass


class BaseResponsesAPIAgent(BaseServer):
    config: BaseResponsesAPIAgentConfig


class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, AggregateMetricsMixin, SimpleServer):
    config: BaseResponsesAPIAgentConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        app.post("/aggregate_metrics")(self.aggregate_metrics)

        return app

    def rollout_id_from_run(self, body: Any) -> Optional[str]:
        """Per-rollout capture id for a run-request (its task/rollout indices), or None."""
        return rollout_id_from_run_body(body)

    def resolve_model_base_url(self, model_server_name: str, request: Optional[Request] = None) -> str:
        """Resolve a model server's base_url (``.../v1``) with the per-rollout capture prefix applied
        when ``request`` carries a rollout id. Use instead of building base_url by hand."""
        server_config = get_first_server_config_dict(self.server_client.global_config_dict, model_server_name)
        base_url = f"{self.server_client._build_server_base_url(server_config)}/v1"
        rollout_id = request.headers.get(ROLLOUT_HEADER) if request is not None else None
        return apply_rollout_prefix(base_url, rollout_id)

    def rollout_call_kwargs(self, body: Any) -> dict[str, Any]:
        """``server_client.post(...)`` kwargs that forward a run-request's rollout id to a self-called
        ``/v1/responses`` (empty when there is no id). Spread with ``**``."""
        rollout_id = self.rollout_id_from_run(body)
        return {"headers": {ROLLOUT_HEADER: rollout_id}} if rollout_id else {}

    # TODO: right now there is no validation on the TypedDict NeMoGymResponseCreateParamsNonStreaming
    # We should explicitly add validation at this server level or we should explicitly not validate so that there is flexibility in this API.
    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    @abstractmethod
    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        pass

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Default: same RewardProfiler aggregation as resources server. Override to proxy."""
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )
