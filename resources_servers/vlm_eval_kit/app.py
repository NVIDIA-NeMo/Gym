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
from subprocess import run

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class VlmEvalKitResourcesServerConfig(BaseResourcesServerConfig):
    pass


class VlmEvalKitResourcesServer(SimpleResourcesServer):
    config: VlmEvalKitResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        this_dir = Path(__file__).parent.absolute()
        # We freeze the commit SHA for now.
        # We pip install with no-deps since we have the deps in the pyproject.toml already.
        setup_command = f"""cd {this_dir} \
&& source .venv/bin/activate \
&& if [ ! -d VLMEvalKit ]; then git clone https://github.com/open-compass/VLMEvalKit/; fi \
&& cd VLMEvalKit \
&& git checkout 00804217f868058f871f5ff252a7b9623c3475d9 \
&& uv pip install '-e .' --no-deps \
&& sed -i '' 's/import clip/# import clip/' vlmeval/dataset/utils/SArena/FID.py
"""
        print(f"Running VLMEvalKit setup command: {setup_command}")
        run(setup_command, shell=True, check=True)

        return app

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


if __name__ == "__main__":
    VlmEvalKitResourcesServer.run_webserver()
