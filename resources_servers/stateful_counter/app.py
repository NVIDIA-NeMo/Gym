# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict

from fastapi import FastAPI, Request
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class StatefulCounterResourcesServerConfig(BaseResourcesServerConfig):
    pass


class IncrementCounterRequest(BaseModel):
    count: int


class IncrementCounterResponse(BaseModel):
    success: bool


class GetCounterValueResponse(BaseModel):
    count: int


class StatefulCounterResourcesServer(SimpleResourcesServer):
    config: StatefulCounterResourcesServerConfig

    def model_post_init(self, context):
        res = super().model_post_init(context)

        self.session_id_to_counter: Dict[str, int] = dict()

        return res

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/increment_counter")(self.increment_counter)
        app.get("/get_counter_value")(self.get_counter_value)

        return app

    async def increment_counter(self, body: IncrementCounterRequest, request: Request) -> IncrementCounterResponse:
        session_id = request.session["session_id"]
        counter = self.session_id_to_counter.setdefault(session_id, 0)

        counter += body.count

        self.session_id_to_counter[session_id] = counter

        return IncrementCounterResponse(success=True)

    async def get_counter_value(self, request: Request) -> GetCounterValueResponse:
        session_id = request.session["session_id"]
        counter = self.session_id_to_counter.setdefault(session_id, 0)
        return counter

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


if __name__ == "__main__":
    StatefulCounterResourcesServer.run_webserver()
