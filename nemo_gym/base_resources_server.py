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
from abc import abstractmethod
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import BaseRunServerConfig, BaseServer, SimpleServer


class BaseResourcesServerConfig(BaseRunServerConfig):
    pass


class BaseResourcesServer(BaseServer):
    config: BaseResourcesServerConfig


class BaseRunRequest(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class BaseVerifyRequest(BaseRunRequest):
    response: NeMoGymResponse


class BaseVerifyResponse(BaseVerifyRequest):
    reward: float


class SimpleResourcesServer(BaseResourcesServer, SimpleServer):
    config: BaseResourcesServerConfig

    def get_session_middleware_secret_key(self) -> str:
        # This method is here to override in case we want to ever use an actual session middleware secret key.
        # e.g. for an actual product.
        return self.__class__.__name__

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        # The multiple middleware execution order described in https://fastapi.tiangolo.com/tutorial/middleware/#multiple-middleware-execution-order
        # Says that if you register middlewares A and then B,
        # - at request time: They execute B first then A
        # - at response time: They return to A first and then B
        # So for adding session IDs, that middleware must run after SessionMiddleware, so it must be registered before it.

        @app.middleware("http")
        async def add_session_id(request: Request, call_next):
            # If session_id not present, assign one
            if "session_id" not in request.session:
                request.session["session_id"] = str(uuid4())

            response: Response = await call_next(request)
            return response

        app.add_middleware(SessionMiddleware, secret_key=self.get_session_middleware_secret_key())

        app.post("/verify")(self.verify)

        return app

    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass
