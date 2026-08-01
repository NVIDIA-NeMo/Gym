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

from nemo_gym.sandbox import SandboxResources
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.swebench.app import SwebenchResourcesServer, SwebenchResourcesServerConfig


class TestApp:
    async def test_create_sandbox_forwards_resource_requests(self, monkeypatch: MonkeyPatch) -> None:
        config = SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="swebench",
            sandbox_provider="test",
            sandbox_config={
                "resources": {"cpu": 2, "memory_mib": 8192, "disk_gib": 30},
                "resource_requests": {"cpu": 0.5, "memory_mib": 2048, "disk_gib": 10},
                "ports": [3000, "8080"],
            },
        )
        server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        provider = object()
        sandbox = MagicMock()
        sandbox.start = AsyncMock()
        async_sandbox = MagicMock(return_value=sandbox)
        monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
        monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_config", lambda *_args: provider)
        monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_metadata", lambda *_args: {})
        monkeypatch.setattr("resources_servers.swebench.app.AsyncSandbox", async_sandbox)
        test_spec = MagicMock(
            instance_image_key="swebench/image:latest",
            instance_id="owner__repo-1",
            repo="owner/repo",
        )

        result = await server._create_sandbox(test_spec)

        assert result is sandbox
        async_sandbox.assert_called_once_with(provider)
        sandbox.start.assert_awaited_once()
        spec = sandbox.start.await_args.args[0]
        assert spec.resources == SandboxResources(cpu=2, memory_mib=8192, disk_gib=30)
        assert spec.resource_requests == SandboxResources(cpu=0.5, memory_mib=2048, disk_gib=10)
        assert spec.ports == (3000, 8080)

    async def test_seed_session_forwards_sandbox_spec_override(self) -> None:
        config = SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="swebench",
            sandbox_provider="test",
            sandbox_config={},
        )
        server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        request_spec = {
            "resource_requests": {"cpu": 0.25, "memory_mib": 1024},
            "ports": [9000],
        }
        body = MagicMock(sandbox_spec=request_spec)
        test_spec = MagicMock()
        sandbox = MagicMock(_handle=MagicMock(sandbox_id="sandbox-1"))
        server._make_test_spec = MagicMock(return_value=test_spec)
        server._create_sandbox = AsyncMock(return_value=sandbox)
        request = MagicMock(session={SESSION_ID_KEY: "session-1"})

        response = await server.seed_session(request, body)

        server._create_sandbox.assert_awaited_once_with(test_spec, sandbox_spec=request_spec)
        assert response.sandbox_handle == "sandbox-1"
        assert server._session_id_to_sandbox["session-1"] is sandbox

    async def test_request_sandbox_spec_overrides_config(self, monkeypatch: MonkeyPatch) -> None:
        config = SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="swebench",
            sandbox_provider="test",
            sandbox_config={
                "ttl_s": 300,
                "resources": {"cpu": 2, "memory_mib": 8192},
                "resource_requests": {"cpu": 0.5, "memory_mib": 2048},
                "ports": [3000],
                "metadata": {"configured": "true", "overridden": "config"},
            },
        )
        server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        sandbox = MagicMock(start=AsyncMock())
        monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
        monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_config", lambda *_args: object())
        monkeypatch.setattr(
            "resources_servers.swebench.app.resolve_provider_metadata",
            lambda *_args: {"provider": "test"},
        )
        monkeypatch.setattr("resources_servers.swebench.app.AsyncSandbox", MagicMock(return_value=sandbox))
        test_spec = MagicMock(
            instance_image_key="swebench/image:latest",
            instance_id="owner__repo-1",
            repo="owner/repo",
        )

        await server._create_sandbox(
            test_spec,
            sandbox_spec={
                "image": "request/image:ignored",
                "ttl_s": 60,
                "resource_requests": {"cpu": 0.25, "memory_mib": 1024},
                "ports": [9000],
                "metadata": {"request": "true", "overridden": "request"},
            },
        )

        spec = sandbox.start.await_args.args[0]
        assert spec.image == "swebench/image:latest"
        assert spec.ttl_s == 60
        assert spec.resources == SandboxResources(cpu=2, memory_mib=8192)
        assert spec.resource_requests == SandboxResources(cpu=0.25, memory_mib=1024)
        assert spec.ports == (9000,)
        assert spec.metadata == {
            "provider": "test",
            "configured": "true",
            "overridden": "request",
            "request": "true",
            "nemo_gym_agent": "swebench",
            "instance_id": "owner__repo-1",
        }

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
            "resources_servers.swebench.app.SwebenchResourcesServer._create_sandbox", AsyncMock(start=AsyncMock())
        )
        monkeypatch.setattr(
            "resources_servers.swebench.app.run_instance",
            AsyncMock(return_value=dict(resolved=True, completed=True)),
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
