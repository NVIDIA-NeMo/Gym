# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import MagicMock

from app import VerifiersResourcesServer, VerifiersResourcesServerConfig
from schemas import (
    VerifiersGetExampleRequest,
    VerifiersVerifyRequest,
)

from nemo_gym.server_utils import ServerClient


class TestApp:
    def test_sanity(self) -> None:
        config = VerifiersResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        VerifiersResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_get_example(self) -> None:
        config = VerifiersResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = VerifiersResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        test_env_id = "test-env-123"
        server.env_id_to_dataset[test_env_id] = [
            {
                "prompt": [{"role": "user", "content": "Sort: b, a, c"}],
                "example_id": 0,
                "task": "sort",
                "answer": "a, b, c",
            },
            {
                "prompt": [{"role": "user", "content": "Sort: z, y, x"}],
                "example_id": 1,
                "task": "sort",
                "answer": "x, y, z",
            },
        ]

        request = MagicMock()
        body = VerifiersGetExampleRequest(env_id=test_env_id, task_idx=0)

        result = await server.get_example(request, body)
        assert result.prompt == [{"role": "user", "content": "Sort: b, a, c"}]
        assert result.example_id == 0
        assert result.task == "sort"
        assert result.answer == "a, b, c"

    async def test_get_example_second_item(self) -> None:
        config = VerifiersResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = VerifiersResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        test_env_id = "test-env-456"
        server.env_id_to_dataset[test_env_id] = [
            {"prompt": [{"role": "user", "content": "Q1"}], "example_id": 0, "task": "test"},
            {"prompt": [{"role": "user", "content": "Q2"}], "example_id": 1, "task": "test"},
        ]

        request = MagicMock()
        body = VerifiersGetExampleRequest(env_id=test_env_id, task_idx=1)

        result = await server.get_example(request, body)
        assert result.prompt == [{"role": "user", "content": "Q2"}]
        assert result.example_id == 1

    async def test_verify(self) -> None:
        config = VerifiersResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        server = VerifiersResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        request = MagicMock()
        body = VerifiersVerifyRequest(
            responses_create_params={"input": []},
            response={"reward": 0.75, "output": []},
        )

        result = await server.verify(request, body)
        assert result.reward == 0.75
