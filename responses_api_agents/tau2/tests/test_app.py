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
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from pytest import raises

from nemo_gym.server_utils import ServerClient
from responses_api_agents.tau2.app import (
    ModelServerRef,
    Tau2Agent,
    Tau2Config,
)


class TestApp:
    def test_sanity(self) -> None:
        config = Tau2Config(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
            user_model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        Tau2Agent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_sanity_query_input(self) -> None:
        example_jsonl = Path(__file__).parent.parent / "data" / "example.jsonl"
        with example_jsonl.open() as f:
            data = list(map(json.loads, f))

        config = Tau2Config(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
            user_model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        server = Tau2Agent(config=config, server_client=MagicMock(spec=ServerClient))

        app = server.setup_webserver()
        client = TestClient(app)

        async_openai_mock = MagicMock()
        async_openai_mock.create_chat_completion = AsyncMock(return_value={})

        with raises(ValueError, match="UserMessage must have either content or tool_calls. Got UserMessage"):
            with (
                patch("responses_api_agents.tau2.app.get_server_url", return_value="dummy base url"),
                patch("tau2.utils.llm_utils.NeMoGymAsyncOpenAI", return_value=async_openai_mock),
            ):
                client.post("/run", json=data[0])
