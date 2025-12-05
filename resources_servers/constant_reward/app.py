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
from typing import Any

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class ConstantRewardResourcesServerConfig(BaseResourcesServerConfig):
    reward: float = 1.0


class ConstantRewardRunRequest(BaseRunRequest):
    pass


class ConstantRewardVerifyRequest(ConstantRewardRunRequest, BaseVerifyRequest):
    pass


class ConstantRewardVerifyResponse(BaseVerifyResponse):
    pass


class ConstantRewardResourcesServer(SimpleResourcesServer):
    config: ConstantRewardResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        return super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: ConstantRewardVerifyRequest) -> ConstantRewardVerifyResponse:
        reward = self.config.reward
        return ConstantRewardVerifyResponse(
            **body.model_dump(),
            reward=reward,
        )


if __name__ == "__main__":
    ConstantRewardResourcesServer.run_webserver()
