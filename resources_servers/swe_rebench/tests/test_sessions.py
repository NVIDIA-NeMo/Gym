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
"""Episode sessions and the agent-run seed, through the server's HTTP routes with a fake sandbox."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.server_runtime import ResourcesSeedSessionRequest, check_resources_session_contract
from resources_servers.swe_rebench import app as swe_rebench_app
from resources_servers.swe_rebench.app import SWERebenchResourcesServer, SWERebenchResourcesServerConfig
from resources_servers.swe_rebench.verification import TEST_OUTPUT_BEGIN, VerificationResult, run_verification
from resources_servers.swebench.patch_capture import CaptureCommandError, PatchCapture


_ROW = {
    "instance_id": "getmoto__moto-7365",
    "repo": "getmoto/moto",
    "version": "5.0",
    "base_commit": "abc",
    "image_name": "docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest",
    "FAIL_TO_PASS": ["tests/test_a.py::test_a"],
    "PASS_TO_PASS": [],
}
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
_VERIFY = _ROW | {"responses_create_params": {"input": "Fix it"}, "response": _RESPONSE}


def _server(monkeypatch: MonkeyPatch) -> tuple[SWERebenchResourcesServer, list[MagicMock]]:
    """A server whose sandboxes are fakes, with the agent's patch capture and the grading run stubbed."""
    sandboxes: list[MagicMock] = []

    async def create_sandbox(self, body, files=None, *, session_id=None, verifier_session_id=None) -> MagicMock:
        sandbox = MagicMock()
        sandbox._handle = SandboxHandle(sandbox_id=f"sb-{len(sandboxes) + 1}", provider_name="test", raw=None)
        sandbox.exec = AsyncMock(return_value=SandboxExecResult(return_code=0, stdout="", stderr=""))
        sandbox.serialize = AsyncMock(return_value={"sandbox_id": sandbox._handle.sandbox_id})
        sandbox.stop = AsyncMock()
        sandbox.initial_files = files or {}
        sandboxes.append(sandbox)
        if session_id is not None:
            self._session_id_to_sandbox[session_id] = sandbox
        if verifier_session_id is not None:
            self._verifier_sandboxes[verifier_session_id] = sandbox
        return sandbox

    monkeypatch.setattr(SWERebenchResourcesServer, "_create_sandbox", create_sandbox)
    monkeypatch.setattr(swe_rebench_app, "prepare_git_for_commits", AsyncMock())
    monkeypatch.setattr(
        swe_rebench_app,
        "capture_model_patch",
        AsyncMock(return_value=PatchCapture.static("diff --git a/x b/x\n", "worktree", "worktree")),
    )
    monkeypatch.setattr(
        swe_rebench_app, "run_verification", AsyncMock(return_value=VerificationResult(True, True, True, {}, "ok"))
    )
    monkeypatch.setattr(swe_rebench_app, "resolve_parser", lambda *args: lambda log: {})
    config = SWERebenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swe_rebench",
        sandbox_provider="docker",
        sandbox_config={},
        apply_anti_cheating=False,
    )
    return SWERebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient)), sandboxes


def _seed(session_id: str = "resources-session") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="swe_rebench", task_id=_ROW["instance_id"]),
        task_data=_ROW | {"responses_create_params": {"input": "Fix it"}},
    )


def test_episode_session_hands_over_the_task_sandbox_grades_it_and_closes(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    seed = _seed().model_dump(mode="json")
    close = {"resources_session_id": "resources-session", "episode_id": {"rollout_id": "rollout"}}

    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=seed)
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["sandbox_access"] == {
            "connection": {"kind": "direct", "provider_config_ref": "docker", "descriptor": {"sandbox_id": "sb-1"}},
            "workdir": "/moto",
        }

        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0
        assert verified.json()["model_patch"] == "diff --git a/x b/x\n"
        sandboxes[0].stop.assert_awaited_once()
        repeated = client.post("/verify", json=_VERIFY)
        assert repeated.status_code == 200 and repeated.json()["model_patch"] == verified.json()["model_patch"]
        assert len(sandboxes) == 2, "an identical retry must reuse the verdict"

        assert client.post("/close_session", json=close).status_code == 200
        assert client.post("/seed_session", json=seed).status_code == 409

    sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {} and server._session_id_to_pristine_untracked == {}


def test_agent_run_seed_still_returns_the_sandbox_handle(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=_ROW)
        assert seeded.status_code == 200, seeded.text
        assert seeded.json() == {"sandbox_handle": "sb-1"}
        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0
    sandboxes[0].stop.assert_awaited_once()


def test_session_conformance(monkeypatch: MonkeyPatch) -> None:
    server, _ = _server(monkeypatch)
    check_resources_session_contract(server.setup_webserver(), _seed("conformance-session"), keeps_state=True)


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("error_type", [None, "timeout", "sandbox"])
def test_git_failure_and_provider_failure_have_distinct_denominators(
    monkeypatch: MonkeyPatch, typed: bool, error_type: str | None
) -> None:
    server, sandboxes = _server(monkeypatch)
    monkeypatch.setattr(
        swe_rebench_app,
        "capture_model_patch",
        AsyncMock(side_effect=CaptureCommandError("git failed", return_code=128, error_type=error_type)),
    )
    monkeypatch.setattr(
        swe_rebench_app,
        "run_verification",
        AsyncMock(return_value=VerificationResult(True, False, True, {}, "failed")),
    )
    with TestClient(server.setup_webserver()) as client:
        seed = _seed().model_dump(mode="json") if typed else _ROW
        assert client.post("/seed_session", json=seed).status_code == 200
        response = client.post("/verify", json=_VERIFY).json()
        assert response["reward"] == 0.0
        assert response["evaluation_completed"] is (error_type is None)
        assert response["mask_sample"] is (error_type is not None)
        assert "git failed" in response["error"]
        if error_type is None:
            assert len(sandboxes) == 2
            assert swe_rebench_app.run_verification.await_args.kwargs["inputs"].patch == ""
        else:
            assert len(sandboxes) == 1
            swe_rebench_app.run_verification.assert_not_awaited()


def test_outer_task_identity_need_not_equal_the_benchmark_instance(monkeypatch: MonkeyPatch) -> None:
    server, _ = _server(monkeypatch)
    seed = _seed().model_copy(update={"task_id": TaskId(taskset="swe_rebench", task_id="caller-row-0")})
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json=seed.model_dump(mode="json"))
        assert response.status_code == 200, response.text
        assert client.post("/verify", json=_VERIFY).json()["reward"] == 1.0


@pytest.mark.parametrize("typed", [False, True])
def test_inconclusive_verification_retries_fresh_sandbox_with_same_patch(
    monkeypatch: MonkeyPatch, typed: bool
) -> None:
    server, sandboxes = _server(monkeypatch)
    verify = AsyncMock(
        side_effect=[
            VerificationResult(False, False, False, None, "", "provider unavailable"),
            VerificationResult(True, True, True, {}, "ok"),
        ]
    )
    monkeypatch.setattr(swe_rebench_app, "run_verification", verify)
    with TestClient(server.setup_webserver()) as client:
        seed = _seed().model_dump(mode="json") if typed else _ROW
        assert client.post("/seed_session", json=seed).status_code == 200
        response = client.post("/verify", json=_VERIFY)
        assert response.status_code == 200, response.text
        assert response.json()["reward"] == 1.0
        assert len(sandboxes) == 3
        assert sandboxes[1].initial_files == sandboxes[2].initial_files
        assert verify.await_args_list[0].kwargs["sandbox"] is sandboxes[1]
        assert verify.await_args_list[1].kwargs["sandbox"] is sandboxes[2]
        for sandbox in sandboxes:
            sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("typed", [False, True])
def test_signal_cutoff_retries_and_masks_through_real_verifier(
    monkeypatch: MonkeyPatch, tmp_path: Path, typed: bool
) -> None:
    server, sandboxes = _server(monkeypatch)
    server.config.verification_log_dir = tmp_path
    create = SWERebenchResourcesServer._create_sandbox

    async def interrupted(*args, **kwargs):
        sandbox = await create(*args, **kwargs)
        sandbox.exec.return_value = SandboxExecResult(
            stdout=f"{TEST_OUTPUT_BEGIN}\npartial test output\n", stderr="signal: killed", return_code=-1
        )
        return sandbox

    monkeypatch.setattr(SWERebenchResourcesServer, "_create_sandbox", interrupted)
    monkeypatch.setattr(swe_rebench_app, "run_verification", run_verification)
    with TestClient(server.setup_webserver()) as client:
        seed = _seed().model_dump(mode="json") if typed else _ROW
        assert client.post("/seed_session", json=seed).status_code == 200
        response = client.post("/verify", json=_VERIFY)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["reward"] == 0.0 and result["mask_sample"] is True
        assert result["evaluation_completed"] is False and result["verification_attempts"] == 2
        assert result["error"] == "evaluation ended without complete test output (exit -1; cause unknown)"
        assert len(sandboxes) == 3
        assert sandboxes[1].initial_files == sandboxes[2].initial_files
        assert not server._session_id_to_sandbox and not server._verifier_sandboxes
        for sandbox in sandboxes:
            sandbox.stop.assert_awaited_once()
    for attempt in (1, 2):
        logs = tmp_path / _ROW["instance_id"] / f"attempt-{attempt}"
        assert json.loads((logs / "verification.json").read_text())["sandbox_id"] == f"sb-{attempt + 1}"
        assert json.loads((logs / "exec_result.json").read_text())["return_code"] == -1


def test_retry_after_task_stop_failure_reuses_capture(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.side_effect = [RuntimeError("provider unavailable"), None]
        assert client.post("/verify", json=_VERIFY).status_code == 500
        assert server._session_id_to_sandbox["resources-session"] is sandboxes[0]
        response = client.post("/verify", json=_VERIFY)
        assert response.status_code == 200, response.text
        assert response.json()["reward"] == 1.0
        swe_rebench_app.capture_model_patch.assert_awaited_once()
        assert len(sandboxes) == 2


@pytest.mark.parametrize("typed", [False, True])
def test_capture_failure_is_incomplete_without_grading_an_empty_patch(monkeypatch: MonkeyPatch, typed: bool) -> None:
    server, sandboxes = _server(monkeypatch)
    monkeypatch.setattr(swe_rebench_app, "capture_model_patch", AsyncMock(side_effect=RuntimeError("capture failed")))
    with TestClient(server.setup_webserver()) as client:
        seed = _seed().model_dump(mode="json") if typed else _ROW
        assert client.post("/seed_session", json=seed).status_code == 200
        response = client.post("/verify", json=_VERIFY)
        assert response.status_code == 200, response.text
        assert response.json()["evaluation_completed"] is False
        assert response.json()["mask_sample"] is True
        assert "capture failed" in response.json()["error"]
        assert len(sandboxes) == 1
        swe_rebench_app.run_verification.assert_not_awaited()
    sandboxes[0].stop.assert_awaited_once()


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("completed", [False, True])
def test_only_incomplete_verification_masks_a_zero_reward(
    monkeypatch: MonkeyPatch, typed: bool, completed: bool
) -> None:
    server, sandboxes = _server(monkeypatch)
    monkeypatch.setattr(
        swe_rebench_app,
        "run_verification",
        AsyncMock(
            return_value=VerificationResult(completed, False, True, {}, "tests", None if completed else "error")
        ),
    )
    with TestClient(server.setup_webserver()) as client:
        seed = _seed().model_dump(mode="json") if typed else _ROW
        assert client.post("/seed_session", json=seed).status_code == 200
        response = client.post("/verify", json=_VERIFY)
        assert response.status_code == 200, response.text
        assert response.json()["reward"] == 0.0
        assert response.json()["evaluation_completed"] is completed
        assert response.json()["mask_sample"] is (not completed)
        assert len(sandboxes) == (2 if completed else 3)
        for sandbox in sandboxes:
            sandbox.stop.assert_awaited_once()
