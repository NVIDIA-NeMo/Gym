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
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.verifier_fixture import VerifierFixture


class SimpleWeatherResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GetWeatherRequest(BaseModel):
    city: str


class GetWeatherResponse(BaseModel):
    city: str
    weather_description: str


class SimpleWeatherVerifier:
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        reward = float(
            any(item.type == "function_call" and item.name == "get_weather" for item in body.response.output)
        )
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


class SimpleWeatherResourcesServer(SimpleWeatherVerifier, SimpleResourcesServer):
    config: SimpleWeatherResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/get_weather")(self.get_weather)

        return app

    async def get_weather(self, body: GetWeatherRequest) -> GetWeatherResponse:
        return GetWeatherResponse(city=body.city, weather_description=f"The weather in {body.city} is cold.")


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=SimpleWeatherVerifier,
    request_model=BaseVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    SimpleWeatherResourcesServer.run_webserver()
