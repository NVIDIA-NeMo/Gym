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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.server_utils import ServerClient
from resources_servers.swebench_pro.app import (
    SWEBenchProInstanceRequest,
    SWEBenchProResourcesServer,
    SWEBenchProResourcesServerConfig,
)
from resources_servers.swebench_pro.verification import VerificationResult


def request_body() -> dict:
    return {
        "repo": "example/repo",
        "instance_id": "instance_example",
        "base_commit": "abc123",
        "patch": "gold patch",
        "test_patch": "",
        "problem_statement": "Fix it",
        "fail_to_pass": '["new_test"]',
        "pass_to_pass": '["old_test"]',
        "before_repo_set_cmd": "",
        "selected_test_files_to_run": '["tests"]',
        "dockerhub_tag": "example-tag",
        "run_script": "#!/bin/bash\nexit 0\n",
        "parser_script": "print('{}')\n",
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
    }


def make_server(*, golden: bool) -> SWEBenchProResourcesServer:
    config = SWEBenchProResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swebench_pro_resources_server",
        sandbox_provider="test",
        sandbox_config={},
        is_verifying_golden_patch=golden,
        prefetch_go_modules=True,
    )
    return SWEBenchProResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_golden_patch_verify_and_cleanup(monkeypatch: MonkeyPatch) -> None:
    server = make_server(golden=True)
    sandbox = SimpleNamespace(stop=AsyncMock())
    create = AsyncMock(return_value=sandbox)
    verify = AsyncMock(
        return_value=VerificationResult(
            completed=True,
            resolved=True,
            patch_applied=True,
            test_results={"tests": []},
        )
    )
    monkeypatch.setattr(server, "_create_sandbox", create)
    monkeypatch.setattr("resources_servers.swebench_pro.app.run_verification", verify)

    response = TestClient(server.setup_webserver()).post("/verify", json=request_body())

    assert response.status_code == 200
    assert response.json()["reward"] == 1.0
    assert response.json()["model_patch"] == "gold patch"
    assert response.json()["resolved"] is True
    assert verify.await_args.kwargs["inputs"].prefetch_go_modules is True
    create.assert_awaited_once()
    sandbox.stop.assert_awaited_once()


def test_normal_verify_extracts_agent_patch(monkeypatch: MonkeyPatch) -> None:
    server = make_server(golden=False)
    sandbox = SimpleNamespace(stop=AsyncMock())
    monkeypatch.setattr(server, "_extract_model_patch", AsyncMock(return_value="agent patch"))
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    verify = AsyncMock(
        return_value=VerificationResult(
            completed=True,
            resolved=False,
            patch_applied=True,
            test_results={"tests": []},
            test_output="test run output",
        )
    )
    monkeypatch.setattr("resources_servers.swebench_pro.app.run_verification", verify)

    response = TestClient(server.setup_webserver()).post("/verify", json=request_body())

    assert response.status_code == 200
    assert response.json()["model_patch"] == "agent patch"
    assert response.json()["reward"] == 0.0
    assert response.json()["test_output"] == "test run output"


def test_verify_reports_sandbox_failure(monkeypatch: MonkeyPatch) -> None:
    server = make_server(golden=True)
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(side_effect=RuntimeError("sandbox unavailable")))

    response = TestClient(server.setup_webserver()).post("/verify", json=request_body())

    assert response.status_code == 200
    assert response.json()["evaluation_completed"] is False
    assert response.json()["reward"] == 0.0
    assert "sandbox unavailable" in response.json()["error"]


def test_schema_rejects_missing_evaluator_asset() -> None:
    body = request_body()
    del body["parser_script"]

    response = TestClient(make_server(golden=True).setup_webserver()).post("/verify", json=body)

    assert response.status_code == 422


def test_image_digest_avoids_case_sensitive_tag_rewriting() -> None:
    body = request_body()
    body["image_digest"] = "sha256:abc123"
    instance = SWEBenchProInstanceRequest.model_validate(body)

    assert make_server(golden=True)._image(instance) == "docker.io/jefzda/sweap-images@sha256:abc123"


@pytest.mark.asyncio
async def test_extract_model_patch_includes_commits_and_untracked_files() -> None:
    server = make_server(golden=False)
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="complete patch", stderr="")),
        stop=AsyncMock(),
    )
    server._session_id_to_sandbox["session"] = sandbox

    patch = await server._extract_model_patch("session", "abc123")

    assert patch == "complete patch"
    command = sandbox.exec.await_args.args[0]
    assert "git -C /app add -N ." in command
    assert "git -C /app --no-pager diff abc123" in command
    sandbox.stop.assert_awaited_once()
    assert "session" not in server._session_id_to_sandbox


@pytest.mark.asyncio
async def test_shutdown_stops_abandoned_session_sandboxes() -> None:
    server = make_server(golden=False)
    first = SimpleNamespace(stop=AsyncMock())
    second = SimpleNamespace(stop=AsyncMock())
    server._session_id_to_sandbox = {"first": first, "second": second}

    await server.shutdown()

    first.stop.assert_awaited_once()
    second.stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {}
