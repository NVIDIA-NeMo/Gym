# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Server primitives for rollout protocols that coordinate agents and environments."""

from abc import abstractmethod
from functools import wraps
from typing import Any

from fastapi import Body, FastAPI

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, BaseRunServerInstanceConfig
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body, rollout_context
from nemo_gym.server_utils import BaseServer, SimpleServer


class BaseRolloutOrchestratorConfig(BaseRunServerInstanceConfig):
    pass


class BaseRolloutOrchestrator(BaseServer):
    config: BaseRolloutOrchestratorConfig


class SimpleRolloutOrchestrator(BaseRolloutOrchestrator, AggregateMetricsMixin, SimpleServer):
    """Expose a rollout protocol independently from any participant agent."""

    config: BaseRolloutOrchestratorConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        run = self.run

        @wraps(run)
        async def run_with_rollout_context(*args: Any, **kwargs: Any) -> BaseVerifyResponse:
            body = kwargs.get("body")
            if body is None:
                body = next((arg for arg in args if isinstance(arg, BaseRunRequest)), None)
            with rollout_context(maybe_rollout_id_from_run_body(body)):
                return await run(*args, **kwargs)

        app.post("/run")(run_with_rollout_context)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    @abstractmethod
    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        pass

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )
