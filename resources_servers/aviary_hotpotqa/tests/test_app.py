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

import warnings
from unittest.mock import MagicMock, patch

from nemo_gym.server_utils import ServerClient


class TestApp:
    def test_sanity(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*is not a Python type.*")

            with patch.dict(
                "sys.modules",
                {
                    "aviary.envs.hotpotqa.env": MagicMock(),
                    "aviary.core": MagicMock(),
                },
            ):
                from resources_servers.aviary_hotpotqa.app import AviaryHotpotqaResourcesServerConfig

                config = AviaryHotpotqaResourcesServerConfig(
                    name="",
                    host="0.0.0.0",
                    port=8080,
                    entrypoint="",
                    max_steps=10,
                )
                assert config is not None
                assert config.max_steps == 10

    def test_server_instantiation(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*is not a Python type.*")

            with patch.dict(
                "sys.modules",
                {
                    "aviary.envs.hotpotqa.env": MagicMock(),
                    "aviary.core": MagicMock(),
                },
            ):
                from resources_servers.aviary_hotpotqa.app import (
                    AviaryHotpotqaResourcesServer,
                    AviaryHotpotqaResourcesServerConfig,
                )

                config = AviaryHotpotqaResourcesServerConfig(
                    name="",
                    host="0.0.0.0",
                    port=8080,
                    entrypoint="",
                )
                server = AviaryHotpotqaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

                assert server is not None
                assert server.config.max_steps == 10

    def test_env_config_creation(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*is not a Python type.*")

            with patch.dict(
                "sys.modules",
                {
                    "aviary.envs.hotpotqa.env": MagicMock(),
                    "aviary.core": MagicMock(),
                },
            ):
                from resources_servers.aviary_hotpotqa.app import (
                    AviaryHotpotqaResourcesServer,
                    AviaryHotpotqaResourcesServerConfig,
                )

                config = AviaryHotpotqaResourcesServerConfig(
                    name="",
                    host="0.0.0.0",
                    port=8080,
                    entrypoint="",
                )
                server = AviaryHotpotqaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

                env_config = server._create_env_config("What is 2+2?", "4")
                assert env_config["question"] == "What is 2+2?"
                assert env_config["correct_answer"] == "4"
                assert env_config["question_id"] is None
