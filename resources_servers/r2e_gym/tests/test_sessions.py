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

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.r2e_gym import app as r2e_gym_app
from resources_servers.r2e_gym.app import R2EGymResourcesServer, R2EGymResourcesServerConfig
from resources_servers.r2e_gym.verification import VerificationResult
from resources_servers.swebench.patch_capture import PatchCapture


_ROW = {
    "instance_id": "namanjain12__pillow_final__6e97dd5c",
    "repo_name": "pillow",
    "image_name": "namanjain12/pillow_final:6e97dd5cecc074a0d8622e4aa5658b4263324760",
    "expected_output_json": '{"test_a": "PASSED"}',
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


def _server(monkeypatch: MonkeyPatch) -> tuple[R2EGymResourcesServer, list[MagicMock]]:
    """A server whose sandboxes are fakes, with the agent's patch capture and the grading run stubbed."""
    sandboxes: list[MagicMock] = []

    async def create_sandbox(self, body, files=None) -> MagicMock:
        sandbox = MagicMock()
        sandbox._handle = SandboxHandle(sandbox_id=f"sb-{len(sandboxes) + 1}", provider_name="test", raw=None)
        sandbox.exec = AsyncMock(return_value=SandboxExecResult(return_code=0, stdout="", stderr=""))
        sandbox.serialize = AsyncMock(return_value={"sandbox_id": sandbox._handle.sandbox_id})
        sandbox.stop = AsyncMock()
        sandboxes.append(sandbox)
        return sandbox

    monkeypatch.setattr(R2EGymResourcesServer, "_create_sandbox", create_sandbox)
    monkeypatch.setattr(r2e_gym_app, "prepare_git_for_commits", AsyncMock())
    monkeypatch.setattr(
        r2e_gym_app,
        "capture_model_patch",
        AsyncMock(return_value=PatchCapture.static("diff --git a/x b/x\n", "worktree", "worktree")),
    )
    monkeypatch.setattr(
        r2e_gym_app, "run_verification", AsyncMock(return_value=VerificationResult(True, True, True, {}, "ok"))
    )
    config = R2EGymResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="r2e_gym",
        sandbox_provider="docker",
        sandbox_config={},
        apply_anti_cheating=False,
    )
    return R2EGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient)), sandboxes


def _seed(session_id: str = "resources-session") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="r2e_gym", task_id=_ROW["instance_id"]),
        task_data=_ROW | {"responses_create_params": {"input": "Fix it"}},
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
            "workdir": "/testbed",
        }

        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0
        assert verified.json()["model_patch"] == "diff --git a/x b/x\n"
        sandboxes[0].stop.assert_awaited_once()
        assert client.post("/verify", json=_VERIFY).status_code == 500, "a repeated verify must not grade again"

        assert client.post("/close_session", json=close).status_code == 200
        assert client.post("/seed_session", json=seed).status_code == 500

    sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {} and server._session_id_to_pristine_untracked == {}


def test_legacy_seed_still_returns_the_sandbox_handle(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=_ROW)
        assert seeded.status_code == 200, seeded.text
        assert seeded.json() == {"sandbox_handle": "sb-1", "workdir": "/testbed"}
        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0
    sandboxes[0].stop.assert_awaited_once()


def test_session_conformance(monkeypatch: MonkeyPatch) -> None:
    server, _ = _server(monkeypatch)
    check_resources_session_contract(server.setup_webserver(), _seed("conformance-session"), keeps_state=True)


def test_a_failed_test_scrub_refuses_the_seed_and_stops_the_sandbox(monkeypatch: MonkeyPatch) -> None:
    server, sandboxes = _server(monkeypatch)
    monkeypatch.setattr(
        R2EGymResourcesServer, "_hide_tests_from_agent", AsyncMock(side_effect=RuntimeError("scrub failed"))
    )
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 500
        assert client.post("/seed_session", json=_ROW).status_code == 500
    assert [sandbox.stop.await_count for sandbox in sandboxes] == [1, 1]
    assert server._session_id_to_sandbox == {}
