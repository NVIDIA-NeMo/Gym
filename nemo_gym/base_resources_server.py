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
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, model_validator

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


class BaseResourcesServerConfig(BaseRunServerInstanceConfig):
    pass


class JudgeTruncationConfigMixin(BaseModel):
    """Mixin for resource servers that truncate judge inputs to fit a token budget.

    Both fields must be set together or both left as None.
    """

    max_judge_input_tokens: Optional[int] = None
    chars_per_token_estimate: Optional[float] = None

    @model_validator(mode="after")
    def _validate_truncation_params(self):
        has_max = self.max_judge_input_tokens is not None
        has_cpt = self.chars_per_token_estimate is not None
        if has_max != has_cpt:
            set_field = "max_judge_input_tokens" if has_max else "chars_per_token_estimate"
            missing_field = "chars_per_token_estimate" if has_max else "max_judge_input_tokens"
            raise ValueError(
                f"{set_field} is set but {missing_field} is not. "
                "Both must be set together to enable judge input truncation, "
                "or both must be None to disable it."
            )
        return self


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


class SimpleResourcesServer(BaseResourcesServer, SimpleServer):
    config: BaseResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        app.post("/seed_session")(self.seed_session)
        app.post("/verify")(self.verify)

        return app

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        return BaseSeedSessionResponse()

    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass
