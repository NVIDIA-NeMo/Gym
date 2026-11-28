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
from fastapi import Request
from pydantic import BaseModel

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesToolset,
    ToolsetResourcesServer,
    episode_tool,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


class StatefulCounterResourcesServerConfig(BaseResourcesServerConfig):
    pass


class IncrementCounterRequest(BaseModel):
    count: int


class IncrementCounterResponse(BaseModel):
    success: bool


class GetCounterValueRequest(BaseModel):
    pass


class GetCounterValueResponse(BaseModel):
    count: int


class StatefulCounterVerifyRequest(BaseVerifyRequest):
    expected_count: int


class StatefulCounterSeedSessionRequest(BaseSeedSessionRequest):
    initial_count: int


class CounterSessionState(BaseModel):
    """A session's counter: the seed sets it, the tools change it and the verifier reads it."""

    count: int


class StatefulCounterTools(ResourcesToolset):
    """The counter's tools, over one session's counter."""

    state: CounterSessionState

    @episode_tool
    async def increment_counter(self, arguments: IncrementCounterRequest) -> IncrementCounterResponse:
        """Add a number to the counter."""
        self.state.count += arguments.count
        return IncrementCounterResponse(success=True)

    @episode_tool
    async def get_counter_value(self, arguments: GetCounterValueRequest) -> GetCounterValueResponse:
        """Read the counter."""
        return GetCounterValueResponse(count=self.state.count)


class StatefulCounterResourcesServer(ToolsetResourcesServer):
    config: StatefulCounterResourcesServerConfig
    seed_request_model = StatefulCounterSeedSessionRequest
    session_state_model = CounterSessionState
    toolset = StatefulCounterTools

    async def seed_state(self, body: StatefulCounterSeedSessionRequest) -> CounterSessionState:
        return CounterSessionState(count=body.initial_count)

    async def verify(self, request: Request, body: StatefulCounterVerifyRequest) -> BaseVerifyResponse:
        state = self.session_state(request.session[SESSION_ID_KEY])
        reward = float(state is not None and state.count == body.expected_count)
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    StatefulCounterResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = StatefulCounterResourcesServer.run_webserver()  # noqa: F401
