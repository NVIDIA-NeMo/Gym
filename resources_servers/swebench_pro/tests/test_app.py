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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.swebench_pro.app import (
    SWEBenchProInstanceRequest,
    SWEBenchProResourcesServer,
    SWEBenchProResourcesServerConfig,
    SWEBenchProSeedSessionRequest,
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


def fake_pty(session_id: str = "pty-session") -> SimpleNamespace:
    """A stand-in for ``sandbox.pty``; seed_session opens a terminal for the agent."""
    session = SimpleNamespace(session_id=session_id, close=AsyncMock())
    return SimpleNamespace(create=AsyncMock(return_value=session))


def make_server(*, golden: bool, apply_anti_cheating: bool = True) -> SWEBenchProResourcesServer:
    config = SWEBenchProResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swebench_pro_resources_server",
        sandbox_provider="test",
        sandbox_config={},
        is_verifying_golden_patch=golden,
        apply_anti_cheating=apply_anti_cheating,
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
            test_results={
                "tests": [{"name": "new_test", "status": "PASSED"}, {"name": "old_test", "status": "PASSED"}]
            },
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
            test_results={
                "tests": [{"name": "new_test", "status": "FAILED"}, {"name": "old_test", "status": "PASSED"}]
            },
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
async def test_seed_session_applies_shared_anti_cheat_setup(monkeypatch: MonkeyPatch) -> None:
    server = make_server(golden=False)
    sandbox = SimpleNamespace(
        _handle=SimpleNamespace(sandbox_id="sandbox-id"),
        pty=fake_pty(),
        upload=AsyncMock(),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr="")),
    )
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})
    body = SWEBenchProSeedSessionRequest.model_validate(request_body())

    response = await server.seed_session(request, body)

    expected_script = Path(__file__).parents[2] / "swebench" / "anti_cheat_setup.sh"
    sandbox.upload.assert_awaited_once_with(expected_script, "/app/anti_cheat_setup.sh")
    # anti-cheat first, then normalize the container, then snapshot its untracked files
    assert sandbox.exec.await_count == 3
    assert sandbox.exec.await_args_list[0].args[0] == (
        "git reset --hard && WORKING_DIRECTORY=/app bash anti_cheat_setup.sh && rm anti_cheat_setup.sh"
    )
    assert sandbox.exec.await_args_list[0].kwargs["timeout_s"] == 600
    assert response.sandbox_handle == "sandbox-id"
    assert server._session_id_to_sandbox["session"] is sandbox


@pytest.mark.asyncio
async def test_seed_session_can_skip_anti_cheat_setup(monkeypatch: MonkeyPatch) -> None:
    server = make_server(golden=False, apply_anti_cheating=False)
    sandbox = SimpleNamespace(
        _handle=SimpleNamespace(sandbox_id="sandbox-id"),
        pty=fake_pty(),
        upload=AsyncMock(),
        exec=AsyncMock(),
    )
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})
    body = SWEBenchProSeedSessionRequest.model_validate(request_body())

    await server.seed_session(request, body)

    sandbox.upload.assert_not_awaited()
    # anti-cheat is skipped, but the container is still normalized and snapshotted
    assert sandbox.exec.await_count == 2


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
async def test_extract_model_patch_drops_untracked_files_the_image_already_shipped() -> None:
    """Artifacts the task image ships are not the agent's work and break `git apply`."""
    artifact = (
        "diff --git a/dump.rdb b/dump.rdb\nnew file mode 100644\n--- /dev/null\n+++ b/dump.rdb\n@@ -0,0 +1 @@\n+x\n"
    )
    fix = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n"
    server = make_server(golden=False)
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout=artifact + fix, stderr="")),
        stop=AsyncMock(),
    )
    server._session_id_to_sandbox["session"] = sandbox
    server._session_id_to_pristine_untracked["session"] = frozenset({"dump.rdb"})

    patch = await server._extract_model_patch("session", "abc123")

    assert patch == fix
    assert "session" not in server._session_id_to_pristine_untracked


@pytest.mark.asyncio
async def test_seed_session_normalizes_the_agent_environment_before_snapshotting() -> None:
    """The agent runs the same suites, so it needs the same repaired container the verifier gets.

    Order matters: normalization deletes stale files, so the untracked baseline must be
    taken afterwards or it records paths that no longer exist.
    """
    server = make_server(golden=False)
    calls: list[str] = []

    async def record(command, *args, **kwargs):
        calls.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    sandbox = SimpleNamespace(
        exec=record,
        upload=AsyncMock(),
        stop=AsyncMock(),
        _handle=SimpleNamespace(sandbox_id="sandbox-id"),
        pty=fake_pty(),
    )
    server._create_sandbox = AsyncMock(return_value=sandbox)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})

    await server.seed_session(request, SWEBenchProSeedSessionRequest.model_validate(request_body()))

    normalize = next(i for i, c in enumerate(calls) if "Xvfb" in c)
    snapshot = next(i for i, c in enumerate(calls) if "ls-files --others" in c)
    assert normalize < snapshot, calls


@pytest.mark.asyncio
async def test_seed_session_survives_a_container_it_cannot_normalize() -> None:
    server = make_server(golden=False)

    async def boom(command, *args, **kwargs):
        if "Xvfb" in command:
            raise RuntimeError("exec failed")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    sandbox = SimpleNamespace(
        exec=boom,
        upload=AsyncMock(),
        stop=AsyncMock(),
        _handle=SimpleNamespace(sandbox_id="sandbox-id"),
        pty=fake_pty(),
    )
    server._create_sandbox = AsyncMock(return_value=sandbox)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})

    # A container that cannot be normalized is still worth running.
    await server.seed_session(request, SWEBenchProSeedSessionRequest.model_validate(request_body()))
    assert server._session_id_to_sandbox["session"] is sandbox


@pytest.mark.asyncio
async def test_seed_session_returns_the_pty_session_the_agent_attaches_to() -> None:
    """The agent needs both ids; given only one it silently builds its own sandbox instead."""
    server = make_server(golden=False)
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr="")),
        upload=AsyncMock(),
        stop=AsyncMock(),
        _handle=SimpleNamespace(sandbox_id="sandbox-id"),
        pty=fake_pty("pty-id"),
    )
    server._create_sandbox = AsyncMock(return_value=sandbox)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})

    response = await server.seed_session(request, SWEBenchProSeedSessionRequest.model_validate(request_body()))

    assert response.sandbox_handle == "sandbox-id"
    assert response.pty_session_id == "pty-id"
    sandbox.pty.create.assert_awaited_once()
    assert server._session_id_to_pty["session"].session_id == "pty-id"


@pytest.mark.asyncio
async def test_extract_model_patch_closes_the_pty_before_stopping_the_sandbox() -> None:
    """A session that outlives its sandbox leaks its connection to the sandbox API."""
    server = make_server(golden=False)
    order: list[str] = []
    session = SimpleNamespace(session_id="pty-id", close=AsyncMock(side_effect=lambda: order.append("close")))
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr="")),
        stop=AsyncMock(side_effect=lambda: order.append("stop")),
    )
    server._session_id_to_sandbox["session"] = sandbox
    server._session_id_to_pty["session"] = session

    await server._extract_model_patch("session", "abc123")

    assert order == ["close", "stop"], "the terminal must be closed before its sandbox goes away"
    assert "session" not in server._session_id_to_pty


@pytest.mark.asyncio
async def test_pristine_untracked_files_lists_and_tolerates_failure() -> None:
    server = make_server(golden=False)
    listing = SimpleNamespace(return_code=0, stdout="dump.rdb\nappendonlydir/appendonly.aof.manifest\n\n", stderr="")
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=listing))

    assert await server.pristine_untracked_files(sandbox) == frozenset(
        {"dump.rdb", "appendonlydir/appendonly.aof.manifest"}
    )
    assert "ls-files --others --exclude-standard" in sandbox.exec.await_args.args[0]

    failing = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=1, stdout="", stderr="boom")))
    assert await server.pristine_untracked_files(failing) == frozenset()


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
