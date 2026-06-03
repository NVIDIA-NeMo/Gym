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

from fastapi import Body, FastAPI
from pydantic import Field

from nemo_gym.adapters import install_middleware
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


class BaseResponsesAPIModelConfig(BaseRunServerInstanceConfig):
    adapters: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Adapter middleware chain: list of {'name': ..., 'config': {...}}. None disables.",
    )


class BaseResponsesAPIModel(BaseServer):
    config: BaseResponsesAPIModelConfig


class SimpleResponsesAPIModel(BaseResponsesAPIModel, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)

        app.post("/v1/chat/completions")(self.chat_completions)

        app.post("/v1/responses")(self.responses)

        install_middleware(app, self.config.adapters)

        return app

    @abstractmethod
    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        pass

    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass
