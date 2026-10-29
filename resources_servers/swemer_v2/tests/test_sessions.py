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

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.server_utils import ServerClient
from nemo_gym.task_materialization import materialize_task
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.swebench.patch_capture import PatchCapture
from resources_servers.swemer_v2 import app as server_app
from resources_servers.swemer_v2.app import SwemerV2ResourcesServer, SwemerV2ResourcesServerConfig
from resources_servers.swemer_v2.verification import VerificationResult


# The first example row, without the prompt; an Environment Server sends the rest as task_data.
_ROW = json.loads((Path(__file__).parents[1] / "data" / "example.jsonl").read_text().splitlines()[0])
_ROW.pop("responses_create_params")
_TASK_ID = TaskId.model_validate(materialize_task(_ROW, taskset="swemer_v2")["task_id"])
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


def _server(monkeypatch: MonkeyPatch) -> tuple[SwemerV2ResourcesServer, list[MagicMock]]:
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

    monkeypatch.setattr(SwemerV2ResourcesServer, "_create_sandbox", create_sandbox)
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
    config = SwemerV2ResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swemer_v2",
        sandbox_provider="docker",
        sandbox_config={},
        apply_anti_cheating=False,
    )
    return SwemerV2ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient)), sandboxes


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


def test_sessions_keep_cookies_and_sandboxes_isolated(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    app = server.setup_webserver()
    with (
        TestClient(app, raise_server_exceptions=False) as first,
        TestClient(app, raise_server_exceptions=False) as second,
    ):
        assert first.post("/seed_session", json=_seed("one").model_dump(mode="json")).status_code == 200
        assert second.post("/seed_session", json=_seed("two").model_dump(mode="json")).status_code == 200
        assert first.cookies != second.cookies
        assert first.post("/verify", json=_VERIFY).json()["reward"] == 1.0
        assert server_app.capture_model_patch.await_args.args[0] is sandboxes[0]
        sandboxes[1].stop.assert_not_awaited()
        assert second.post("/verify", json=_VERIFY).json()["reward"] == 1.0
        assert server_app.capture_model_patch.await_args.args[0] is sandboxes[1]


def test_seed_retry_and_changed_verify_are_not_reexecuted(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    seed = _seed().model_dump(mode="json")
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        first = client.post("/seed_session", json=seed)
        assert client.post("/seed_session", json=seed).json() == first.json()
        assert len(sandboxes) == 1
        assert client.post("/verify", json=_VERIFY).status_code == 200
        changed = _VERIFY | {"responses_create_params": {"input": "changed"}}
        assert client.post("/verify", json=changed).status_code == 409
        server_app.capture_model_patch.assert_awaited_once()
        server_app.run_verification.assert_awaited_once()


def test_failed_setup_and_stop_remain_owned_until_close_retries(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)

    async def failed_setup(*args, **kwargs):
        sandboxes[0].stop.side_effect = RuntimeError("stop unavailable")
        raise RuntimeError("setup failed")

    monkeypatch.setattr(server_app, "prepare_git_for_commits", failed_setup)
    close = {"resources_session_id": "resources-session", "episode_id": {"rollout_id": "rollout"}}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=_seed().model_dump(mode="json"))
        assert seeded.status_code != 200
        assert server._session_id_to_sandbox["resources-session"] is sandboxes[0]
        assert client.post("/close_session", json=close).status_code != 200
        assert server._session_id_to_sandbox["resources-session"] is sandboxes[0]
        sandboxes[0].stop.side_effect = None
        assert client.post("/close_session", json=close).status_code == 200
        assert not server._session_id_to_sandbox
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 409


@pytest.mark.parametrize("failed_owner", ["task", "verifier"])
def test_failed_stop_preserves_verdict_for_replay_and_close(monkeypatch: MonkeyPatch, failed_owner: str) -> None:
    server, sandboxes = _server(monkeypatch)
    original = server_app.run_verification

    async def verify_with_failed_stop(**kwargs):
        if failed_owner == "verifier":
            kwargs["sandbox"].stop.side_effect = RuntimeError("stop unavailable")
        return await original(**kwargs)

    monkeypatch.setattr(server_app, "run_verification", verify_with_failed_stop)
    close = {"resources_session_id": "resources-session", "episode_id": {"rollout_id": "rollout"}}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        if failed_owner == "task":
            sandboxes[0].stop.side_effect = RuntimeError("stop unavailable")
        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200 and verified.json()["reward"] == 1.0
        assert client.post("/verify", json=_VERIFY).json() == verified.json()
        assert len(sandboxes) == 2
        assert client.post("/close_session", json=close).status_code != 200
        if failed_owner == "task":
            assert server._session_id_to_sandbox["resources-session"] is sandboxes[0]
        else:
            assert server._verifier_cleanup["resources-session"]
        for sandbox in sandboxes:
            sandbox.stop.side_effect = None
        assert client.post("/close_session", json=close).status_code == 200
        assert not server._verifier_cleanup


def test_failed_capture_is_incomplete_without_grading_empty_patch(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    server_app.capture_model_patch.side_effect = RuntimeError("capture failed")
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200
        assert verified.json()["evaluation_completed"] is False
        assert verified.json()["mask_sample"] is True
        assert "capture failed" in verified.json()["error"]
        server_app.run_verification.assert_not_awaited()
        assert len(sandboxes) == 1
        sandboxes[0].stop.assert_awaited_once()


def test_inconclusive_verification_retries_fresh_sandbox_then_replays(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    success = server_app.run_verification.return_value
    server_app.run_verification.side_effect = [
        VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=False,
            test_results=None,
            test_output="",
            error="provider error",
        ),
        success,
    ]
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200 and verified.json()["reward"] == 1.0
        assert client.post("/verify", json=_VERIFY).json() == verified.json()
        assert len(sandboxes) == 3
        assert server_app.run_verification.await_count == 2
        for sandbox in sandboxes:
            sandbox.stop.assert_awaited_once()


def test_lifespan_releases_unverified_task(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.assert_not_awaited()
    sandboxes[0].stop.assert_awaited_once()
    assert not server._session_id_to_sandbox


@pytest.mark.parametrize(
    "failure",
    [
        SandboxExecResult(return_code=1, stdout="", stderr="setup failed"),
        SandboxExecResult(return_code=None, stdout="", stderr="", error_type="timeout"),
    ],
)
async def test_setup_exec_failure_retains_handle_when_stop_fails(
    monkeypatch: MonkeyPatch, failure: SandboxExecResult
) -> None:
    from fastapi import Request

    server, sandboxes = _server(monkeypatch)
    original = server.__class__._create_sandbox

    async def create_with_failed_exec(self, body, files=None):
        sandbox = await original(self, body, files)
        sandbox.exec.return_value = failure
        sandbox.exec.side_effect = None
        sandbox.stop.side_effect = RuntimeError("stop failed")
        return sandbox

    monkeypatch.setattr(server.__class__, "_create_sandbox", create_with_failed_exec)
    request = Request({"type": "http", "session": {}})
    with pytest.raises(RuntimeError):
        await server.seed_session(request, _seed())
    assert server._session_id_to_sandbox["resources-session"] is sandboxes[0]
    sandboxes[0].stop.side_effect = None
    await server.shutdown()
    assert not server._session_id_to_sandbox


async def test_missing_seed_masks_without_creating_a_verifier(monkeypatch: MonkeyPatch) -> None:
    from fastapi import Request

    server, sandboxes = _server(monkeypatch)
    request = Request({"type": "http", "session": {"session_id": "missing"}})
    from nemo_gym.server_utils import SESSION_ID_KEY

    request.session[SESSION_ID_KEY] = "missing"
    body = server_app.SwemerV2VerifyRequest.model_validate(_VERIFY)
    response = await server.verify(request, body)
    assert response.mask_sample and not response.evaluation_completed
    assert "no base commit" in response.error
    assert not sandboxes


def test_golden_verification_preserves_reference_grading_inputs(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    server.config.is_verifying_golden_patch = True
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200 and verified.json()["reward"] == 1.0
        inputs = server_app.run_verification.await_args.kwargs["inputs"]
        assert inputs.test_patch == _ROW["test_patch"]
        assert inputs.workdir == _ROW["workdir"]
        assert inputs.patch == server_app.drop_test_patch_files(_ROW["patch"], _ROW["test_patch"])
        assert inputs.fail_to_pass == _ROW["FAIL_TO_PASS"]
        assert inputs.pass_to_pass == _ROW["PASS_TO_PASS"]
        assert inputs.test_framework == _ROW["test_framework"]
        assert len(sandboxes) == 1
        sandboxes[0].stop.assert_awaited_once()
