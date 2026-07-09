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
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
    gym_tool,
)


class SimpleWeatherResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GetWeatherResponse(BaseModel):
    city: str
    weather_description: str


class SimpleWeatherResourcesServer(SimpleResourcesServer):
    config: SimpleWeatherResourcesServerConfig

    @gym_tool
    async def get_weather(self, city: str) -> GetWeatherResponse:
        """Get the weather for a city."""
        return GetWeatherResponse(city=city, weather_description=f"The weather in {city} is cold.")

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


if __name__ == "__main__":
    SimpleWeatherResourcesServer.run_webserver()
