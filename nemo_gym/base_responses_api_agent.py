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
import atexit
from abc import abstractmethod
from typing import Any, Optional

from fastapi import Body, FastAPI
from pydantic import Field

from nemo_gym.adapters import AdapterProxyConfig, ProxyHandle, install_middleware, start_adapter_proxy
from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


class BaseResponsesAPIAgentConfig(BaseRunServerInstanceConfig):
    adapters: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Adapter middleware chain: list of {'name': ..., 'config': {...}}. None disables.",
    )
    adapter_proxy: Optional[AdapterProxyConfig] = Field(
        default=None,
        description=(
            "Optional localhost proxy in front of an external inference upstream. "
            "When set, the agent's SDK client should point its *_BASE_URL at "
            "``self._proxy_handle.url`` so model traffic flows through the chain."
        ),
    )


class BaseResponsesAPIAgent(BaseServer):
    config: BaseResponsesAPIAgentConfig


class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, AggregateMetricsMixin, SimpleServer):
    config: BaseResponsesAPIAgentConfig
    _proxy_handle: Optional[ProxyHandle] = None

    def setup_webserver(self) -> FastAPI:
        if self.config.adapter_proxy is not None:
            cfg = self.config.adapter_proxy
            self._proxy_handle = start_adapter_proxy(
                upstream_url=cfg.upstream_url,
                adapters=[spec.model_dump() for spec in cfg.adapters],
                host=cfg.host,
                port=cfg.port,
                request_timeout=cfg.request_timeout,
                unsafe_allow_remote=cfg.unsafe_allow_remote,
            )
            atexit.register(self._proxy_handle.stop)

        app = FastAPI()

        self.setup_session_middleware(app)

        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        app.post("/aggregate_metrics")(self.aggregate_metrics)

        install_middleware(app, self.config.adapters)

        return app

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
