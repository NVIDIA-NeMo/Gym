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

from asyncio import get_event_loop

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from tau2.runner.batch import run_single_task


class Tau2Config(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None


class Tau2RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class Tau2VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class Tau2VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class Tau2Agent(SimpleResponsesAPIAgent):
    config: Tau2Config

    async def run(self, request: Request, body: Tau2RunRequest) -> Tau2VerifyResponse:
        kwargs = None

        loop = get_event_loop()
        result = await loop.run_in_executor(None, run_single_task, **kwargs)

        return Tau2VerifyResponse.model_validate(result)


if __name__ == "__main__":
    Tau2Agent.run_webserver()
