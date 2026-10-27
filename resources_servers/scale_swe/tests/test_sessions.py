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
"""Environment Server sessions and the legacy seed, through the server's HTTP routes with a fake sandbox."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.task_materialization import materialize_task
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.scale_swe import app as server_app
from resources_servers.scale_swe.app import (
    ScaleSWEInstanceRequest,
    ScaleSWEResourcesServer,
    ScaleSWEResourcesServerConfig,
    ScaleSWEVerifyRequest,
)
from resources_servers.scale_swe.verification import VerificationResult
from resources_servers.swebench.patch_capture import PatchCapture


# The first example row, without the prompt; an Environment Server sends the rest as task_data.
_ROW = json.loads((Path(__file__).parents[1] / "data" / "example.jsonl").read_text().splitlines()[0])
_ROW.pop("responses_create_params")
_TASK_ID = TaskId.model_validate(materialize_task(_ROW, taskset="scale_swe")["task_id"])
_PROMPT = {"input": "Fix it"}
_RESPONSE = {
    "output": [],
    "id": "",
    "created_at": 0,
    "model": "",
    "object": "response",
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}
_VERIFY = _ROW | {"responses_create_params": _PROMPT, "response": _RESPONSE}


def _server(monkeypatch: MonkeyPatch) -> tuple[ScaleSWEResourcesServer, list[MagicMock]]:
    """A server whose sandboxes are fakes, with the agent's patch capture stubbed, with the grading run stubbed."""
    sandboxes: list[MagicMock] = []

    async def exec_command(command: str, **kwargs) -> SandboxExecResult:
        stdout = "0123abcd\n" if "rev-parse HEAD" in command else ""
        return SandboxExecResult(return_code=0, stdout=stdout, stderr="")

    async def create_sandbox(self, body, files=None) -> MagicMock:
        sandbox = MagicMock()
        sandbox._handle = SandboxHandle(sandbox_id=f"sb-{len(sandboxes) + 1}", provider_name="test", raw=None)
        sandbox.exec = AsyncMock(side_effect=exec_command)
        sandbox.serialize = AsyncMock(return_value={"sandbox_id": sandbox._handle.sandbox_id})
        sandbox.stop = AsyncMock()
        sandboxes.append(sandbox)
        return sandbox

    monkeypatch.setattr(ScaleSWEResourcesServer, "_create_sandbox", create_sandbox)
    monkeypatch.setattr(
        server_app,
        "capture_model_patch",
        AsyncMock(return_value=PatchCapture.static("diff --git a/x b/x\n", "worktree", "worktree")),
    )
    monkeypatch.setattr(server_app, "prepare_git_for_commits", AsyncMock())
    monkeypatch.setattr(
        server_app,
        "run_verification",
        AsyncMock(
            return_value=VerificationResult(
                completed=True, resolved=True, patch_applied=True, test_results={}, test_output="ok"
            )
        ),
    )
    config = ScaleSWEResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="scale_swe",
        sandbox_provider="docker",
        sandbox_config={},
        apply_anti_cheating=False,
    )
    return ScaleSWEResourcesServer(config=config, server_client=MagicMock(spec=ServerClient)), sandboxes


def _seed(session_id: str = "resources-session") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=_TASK_ID,
        task_data=_ROW | {"responses_create_params": _PROMPT},
    )


def test_typed_episode_hands_over_the_task_sandbox_grades_it_and_closes(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    seed = _seed().model_dump(mode="json")
    close = {"resources_session_id": "resources-session", "episode_id": {"rollout_id": "rollout"}}

    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=seed)
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["sandbox_access"] == {
            "connection": {"kind": "direct", "provider_config_ref": "docker", "descriptor": {"sandbox_id": "sb-1"}},
            "workdir": _ROW["workdir"],
        }

        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0
        assert verified.json()["error"] is None, "the agent's patch must be captured from the task sandbox"
        sandboxes[0].stop.assert_awaited_once()
        replayed = client.post("/verify", json=_VERIFY)
        assert replayed.status_code == 200
        assert replayed.json() == verified.json()
        assert len(sandboxes) == 2, "replaying verification must not start another grading sandbox"

        assert client.post("/close_session", json=close).status_code == 200
        assert client.post("/seed_session", json=seed).status_code == 409

    sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {} and server._session_id_to_pristine_untracked == {}
    assert server._session_id_to_base_commit == {}


def test_legacy_seed_still_returns_the_sandbox_handle(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=_ROW)
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["sandbox_handle"] == "sb-1"
        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0
    sandboxes[0].stop.assert_awaited_once()


def test_session_conformance(monkeypatch: MonkeyPatch) -> None:
    server, _ = _server(monkeypatch)
    check_resources_session_contract(server.setup_webserver(), _seed("conformance-session"), keeps_state=True)


@pytest.mark.parametrize("sandbox_index", [0, 1])
def test_failed_close_retains_verdict_and_retries_cleanup(monkeypatch: MonkeyPatch, sandbox_index: int) -> None:
    server, sandboxes = _server(monkeypatch)
    original_create = server._create_sandbox

    async def create_with_failed_stop(self, body, files=None):
        sandbox = await original_create(body, files=files)
        if len(sandboxes) - 1 == sandbox_index:
            sandbox.stop.side_effect = [RuntimeError("provider unavailable"), RuntimeError("still unavailable"), None]
        return sandbox

    monkeypatch.setattr(ScaleSWEResourcesServer, "_create_sandbox", create_with_failed_stop)
    close = {"resources_session_id": "resources-session", "episode_id": {"rollout_id": "rollout"}}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        client.post("/seed_session", json=_seed().model_dump(mode="json")).raise_for_status()
        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200
        assert verified.json()["reward"] == 1.0
        assert client.post("/close_session", json=close).status_code == 500
        replayed = client.post("/verify", json=_VERIFY)
        assert replayed.status_code == 200
        assert replayed.json() == verified.json()
        assert len(sandboxes) == 2
        server_app.run_verification.assert_awaited_once()
        assert client.post("/close_session", json=close).status_code == 200
    assert sandboxes[sandbox_index].stop.await_count == 3
    assert not server._task_sessions and not server._session_id_to_sandbox and not server._verifier_cleanup


def test_verification_retry_rejects_changed_payload(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    with TestClient(server.setup_webserver()) as client:
        client.post("/seed_session", json=_seed().model_dump(mode="json")).raise_for_status()
        client.post("/verify", json=_VERIFY).raise_for_status()
        changed = _VERIFY | {"response": _RESPONSE | {"id": "different-response"}}
        assert client.post("/verify", json=changed).status_code == 409
        assert len(sandboxes) == 2
        server_app.run_verification.assert_awaited_once()


async def test_cancelled_http_waiter_does_not_repeat_or_cancel_grading(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    request = Request({"type": "http", "session": {SESSION_ID_KEY: "resources-session"}})
    await server.seed_task_sandbox_session(request, _seed(), ScaleSWEInstanceRequest)
    body = ScaleSWEVerifyRequest.model_validate(_VERIFY)
    started, finish = asyncio.Event(), asyncio.Event()
    result = server_app.run_verification.return_value

    async def grade(**kwargs):
        started.set()
        await finish.wait()
        return result

    server_app.run_verification.side_effect = grade
    first = asyncio.create_task(server.verify(request, body))
    await asyncio.wait_for(started.wait(), timeout=2)
    retry = asyncio.create_task(server.verify(request, body))
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    finish.set()
    verified = await asyncio.wait_for(retry, timeout=2)
    assert verified.reward == 1.0
    assert (await server.verify(request, body)).model_dump() == verified.model_dump()
    server_app.run_verification.assert_awaited_once()
    assert len(sandboxes) == 2
    await server.close_resources_session(
        request, ResourcesCloseSessionRequest(resources_session_id="resources-session", episode_id=_seed().episode_id)
    )
    assert all(sandbox.stop.await_count == 1 for sandbox in sandboxes)


async def test_cancelled_seed_retains_failed_cleanup_for_shutdown(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    preparing = asyncio.Event()

    async def prepare(*args, **kwargs):
        sandboxes[0].stop.side_effect = [RuntimeError("provider unavailable"), None]
        preparing.set()
        await asyncio.Future()

    monkeypatch.setattr(server_app, "prepare_git_for_commits", prepare)
    request = Request({"type": "http", "session": {SESSION_ID_KEY: "resources-session"}})
    seed = asyncio.create_task(server.seed_task_sandbox_session(request, _seed(), ScaleSWEInstanceRequest))
    await asyncio.wait_for(preparing.wait(), timeout=2)
    seed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await seed
    assert server._session_id_to_sandbox["resources-session"] is sandboxes[0]
    assert "resources-session" in server._task_sessions
    await server.shutdown()
    assert sandboxes[0].stop.await_count == 2
    assert not server._session_id_to_sandbox and not server._task_sessions
