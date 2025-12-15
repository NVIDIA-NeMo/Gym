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
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from aviary.dataset_server import TaskDatasetServer
from aviary.envs.gsm8k import GSM8kDataset, GSM8kDatasetSplit
from fastapi import Request
from starlette.testclient import TestClient

from nemo_gym.integrations.aviary import AviaryCloseRequest, AviarySeedSessionRequest, AviaryStepRequest
from nemo_gym.openai_utils import NeMoGymResponseFunctionToolCall
from nemo_gym.server_utils import ServerClient
from resources_servers.aviary.app import AviaryResourcesServerConfig
from resources_servers.aviary.client_app import AviaryClientResourcesServer, AviaryClientResourcesServerConfig
from resources_servers.aviary.gsm8k_app import GSM8kResourcesServer
from resources_servers.aviary.notebook_app import BixBenchResourcesServer


class TestGSM8kApp:
    @pytest.mark.asyncio
    async def test_server_lifecycle(self) -> None:
        # Create the server
        config = AviaryResourcesServerConfig(
            name="",
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )
        server = GSM8kResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        # Start an environment
        mock_request = MagicMock(spec=Request)
        seed_resp = await server.seed_session(mock_request, AviarySeedSessionRequest(task_idx=0))

        assert seed_resp.obs, "Expected non-empty observations"
        assert seed_resp.tools, "Expected non-empty tools"

        # Take a step
        action = AviaryStepRequest(
            env_id=seed_resp.env_id,
            action=[
                NeMoGymResponseFunctionToolCall(
                    call_id="abc123",
                    name="calculator",
                    arguments=json.dumps({"expr": "1 + 1"}),
                )
            ],
        )
        step_resp = await server.step(mock_request, action)

        assert len(step_resp.obs) == 1, "Expected 1 observation"
        assert step_resp.obs[0].output == "2"


class TestNotebookApp:
    @pytest.mark.skip(reason="Skipping notebook app tests - requires data download and Docker")
    @pytest.mark.asyncio
    async def test_server_lifecycle(self) -> None:
        # Create the server
        config = AviaryResourcesServerConfig(
            name="",
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )
        server = BixBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        # Start an environment
        # NOTE: this spins up a docker container, so should probably skip in CI
        mock_request = MagicMock(spec=Request)
        seed_resp = await server.seed_session(mock_request, AviarySeedSessionRequest(task_idx=0))

        assert seed_resp.obs, "Expected non-empty observations"
        assert seed_resp.tools, "Expected non-empty tools"

        # Take a step
        action = AviaryStepRequest(
            env_id=seed_resp.env_id,
            action=[
                NeMoGymResponseFunctionToolCall(
                    call_id="abc123",
                    name="submit_solution",
                    arguments=json.dumps({"solution": "print('Hello, world!')"}),
                )
            ],
        )
        step_resp = await server.step(mock_request, action)

        assert len(step_resp.obs) == 2, "Expected 2 observations"
        assert step_resp.done, "Expected done"

        # Close the environment
        close_resp = await server.close(mock_request, AviaryCloseRequest(env_id=seed_resp.env_id))
        assert close_resp.success, "Expected success"
        assert not server.env_id_to_env, "Expected no environments"


class TestClientApp:
    @pytest.mark.asyncio
    async def test_client_server_lifecycle(self) -> None:
        dataset = GSM8kDataset(GSM8kDatasetSplit.train)
        task_server = TaskDatasetServer(dataset=dataset, port=8042)

        test_client = TestClient(task_server.app)

        def create_async_client(*args, **kwargs):
            return httpx.AsyncClient(transport=httpx.ASGITransport(app=task_server.app), base_url="http://testserver")

        with (
            patch("httpx.Client") as mock_client_class,
            patch("httpx_aiohttp.HttpxAiohttpClient", side_effect=create_async_client),
        ):
            mock_client_class.return_value = test_client

            config = AviaryClientResourcesServerConfig(
                name="",
                host="0.0.0.0",
                port=8081,
                entrypoint="",
                server_url="http://testserver",
            )
            client_server = AviaryClientResourcesServer(
                config=config,
                server_client=MagicMock(spec=ServerClient),
            )

            mock_request = MagicMock(spec=Request)
            seed_resp = await client_server.seed_session(mock_request, AviarySeedSessionRequest(task_idx=0))

            assert seed_resp.obs, "Expected non-empty observations"
            assert seed_resp.tools, "Expected non-empty tools"
            assert seed_resp.env_id, "Expected environment ID"

            assert len(task_server.envs) == 1, "Expected 1 environment in TaskDatasetServer"
            assert len(client_server.env_id_to_env) == 1, "Expected 1 environment in AviaryClientResourcesServer"
