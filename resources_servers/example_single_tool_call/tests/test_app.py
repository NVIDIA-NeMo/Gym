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
import asyncio
import runpy
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI

import nemo_gym.server_utils
import resources_servers.example_single_tool_call.app as app_module
from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.example_single_tool_call.app import (
    VERIFIER_FIXTURE,
    SimpleWeatherResourcesServer,
    SimpleWeatherResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = SimpleWeatherResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        SimpleWeatherResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_uvicorn_worker_exports_app(self, monkeypatch) -> None:
        expected_app = FastAPI()
        monkeypatch.setattr(nemo_gym.server_utils, "is_nemo_gym_fastapi_entrypoint", lambda _file: True)
        monkeypatch.setattr(
            SimpleResourcesServer,
            "run_webserver",
            classmethod(lambda _cls: expected_app),
        )

        worker_module = runpy.run_path(str(Path(app_module.__file__)), run_name="uvicorn_worker")

        assert worker_module["app"] is expected_app

    def test_verifier_fixture(self) -> None:
        asyncio.run(
            exercise_verifier_fixture(
                VERIFIER_FIXTURE,
                reward_range=(0.0, 1.0),
                higher_is_better=True,
                determinism="unknown",
            )
        )
