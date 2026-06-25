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
import contextlib
import logging
from abc import abstractmethod
from typing import List, Optional
from uuid import uuid4

from fastapi import Body, FastAPI, Request
from pydantic import ConfigDict, TypeAdapter

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
    NeMoGymResponseInputItem,
    NeMoGymResponseOutputItem,
)
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.server_utils import (
    BaseRunServerInstanceConfig,
    BaseServer,
    SimpleServer,
    get_response_json,
    raise_for_status,
)


LOG = logging.getLogger(__name__)


class AgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class AgentRunVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


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

    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    async def run(self, request: Request, body: AgentRunRequest = Body()) -> AgentRunVerifyResponse:
        sem = getattr(self, "sem", None)
        async with sem if sem is not None else contextlib.nullcontext():
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            run_token = uuid4().hex if getattr(self.config, "model_server", None) else None
            run_header = {"headers": {"x-nemo-gym-run-token": run_token}} if run_token else {}

            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=body.responses_create_params,
                cookies=cookies,
                **run_header,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            if run_token:
                constructed = await self.get_monotonic_trajectory(self.config.model_server.name, run_token)
                if constructed:
                    gym_resp.output = constructed
                else:
                    LOG.warning("no constructed trajectory for run %s, falling back to harness output", run_token)
                agent_resp_json = gym_resp.model_dump()

            def _asst(it):
                return getattr(it, "type", None) == "message" and getattr(it, "role", None) == "assistant"

            turns = sum(_asst(it) for it in gym_resp.output)
            naturally = bool(gym_resp.output) and _asst(gym_resp.output[-1])

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            return AgentRunVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )

    @staticmethod
    def run_token_from_request(request: Request) -> Optional[str]:
        return request.headers.get("x-nemo-gym-run-token")

    def model_server_base_url(self) -> Optional[str]:
        ref = getattr(self.config, "model_server", None)
        if ref is None:
            return None
        cfg = get_first_server_config_dict(self.server_client.global_config_dict, ref.name)
        return self.server_client._build_server_base_url(cfg)

    def harness_base_url(self, request: Request, fallback: Optional[str] = None) -> Optional[str]:
        base = self.model_server_base_url() or fallback
        token = self.run_token_from_request(request)
        return f"{base.rstrip('/')}/runs/{token}" if base and token else base

    async def get_monotonic_trajectory(
        self, model_server_name: str, run_token: str
    ) -> List[NeMoGymResponseOutputItem]:
        try:
            resp = await self.server_client.get(
                server_name=model_server_name,
                url_path=f"/runs/{run_token}/trajectory",
            )
            await raise_for_status(resp)
            payload = await get_response_json(resp)
            return [TypeAdapter(NeMoGymResponseInputItem).validate_python(it) for it in payload.get("output", [])]
        except Exception as exc:
            LOG.warning("could not fetch constructed trajectory for run %s: %r", run_token, exc)
            return []
