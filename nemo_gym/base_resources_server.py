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
import importlib
from abc import abstractmethod

from fastapi import Body, FastAPI
from pydantic import BaseModel

from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


class BaseResourcesServerConfig(BaseRunServerInstanceConfig):
    pass


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


class SimpleResourcesServer(BaseResourcesServer, AggregateMetricsMixin, SimpleServer):
    config: BaseResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        app.post("/seed_session")(self.seed_session)

        # Auto-discover verify_request_cls from any model server in the global
        # config that specifies one (e.g. megatron_inference).
        verify_request_cls = self._discover_verify_request_cls()
        if verify_request_cls:
            async def verify_with_custom_cls(body: verify_request_cls = Body()) -> BaseVerifyResponse:
                return await self.verify(body)

            app.post("/verify")(verify_with_custom_cls)
        else:
            app.post("/verify")(self.verify)

        app.post("/aggregate_metrics")(self.aggregate_metrics)

        return app

    def _discover_verify_request_cls(self):
        """Look for a verify_request_cls in model server configs."""
        from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS, get_first_server_config_dict

        for key in self.server_client.global_config_dict:
            if key in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS:
                continue
            try:
                server_config = get_first_server_config_dict(self.server_client.global_config_dict, key)
                dotted_path = server_config.get("verify_request_cls")
                if dotted_path:
                    mod, cls_name = dotted_path.rsplit(".", 1)
                    return getattr(importlib.import_module(mod), cls_name)
            except (KeyError, TypeError, AttributeError):
                continue
        return None

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        return BaseSeedSessionResponse()

    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass

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
