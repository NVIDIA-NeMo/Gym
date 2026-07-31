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
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.server_utils import ServerClient
from resources_servers.swebench.app import SwebenchResourcesServer, SwebenchResourcesServerConfig


class TestApp:
    def test_sanity(self, monkeypatch: MonkeyPatch) -> None:
        config = SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            sandbox_provider="test",
            sandbox_config=dict(),
            is_verifying_golden_patch=True,
        )
        server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()

        client = TestClient(app)

        monkeypatch.setattr(
            "resources_servers.swebench.app.AsyncSandbox", lambda *args, **kwargs: MagicMock(start=AsyncMock())
        )
        monkeypatch.setattr(
            "resources_servers.swebench.app.run_instance",
            AsyncMock(return_value=dict(resolved=True, evaluation_completed=True)),
        )

        res = client.post(
            "/verify",
            json={
                "repo": "astropy/astropy",
                "instance_id": "my instance_id",
                "base_commit": "my base_commit",
                "patch": "my patch",
                "test_patch": "my test_patch",
                "problem_statement": "my problem_statement",
                "hints_text": "",
                "created_at": "my created_at",
                "version": "4.3",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
                "environment_setup_commit": "my environment_setup_commit",
                "difficulty": "my difficulty",
                "responses_create_params": {"input": []},
                "response": {
                    "output": [],
                    "id": "",
                    "created_at": 0,
                    "model": "",
                    "object": "response",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                },
                "subset": "my subset",
                "split": "my split",
            },
        )
        assert res.status_code == 200
