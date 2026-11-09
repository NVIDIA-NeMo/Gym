# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both routes share patch capture, owned cleanup and inconclusive grading."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.sandbox import SandboxExecResult
from resources_servers.swe_next import app
from resources_servers.swe_next.verification import VerificationResult
from resources_servers.swebench.patch_capture import PatchCapture


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("completed", [False, True])
def test_routes_capture_before_cleanup_and_mask_infrastructure(monkeypatch, tmp_path, typed, completed):
    sandboxes = []

    async def create(self, body, files=None, *, session_id=None, verifier_session_id=None):
        sandbox = MagicMock()
        sandbox.sandbox_id = f"sb-{len(sandboxes)}"
        sandbox.exec = AsyncMock(return_value=SandboxExecResult(return_code=0, stdout="seed-base\n", stderr=""))
        sandbox.serialize = AsyncMock(return_value={"sandbox_id": sandbox.sandbox_id})
        sandbox.stop = AsyncMock()
        sandboxes.append(sandbox)
        if session_id:
            self._session_id_to_sandbox[session_id] = sandbox
        if verifier_session_id:
            self._verifier_sandboxes[verifier_session_id] = sandbox
        return sandbox

    async def capture(sandbox, workdir, base_commit, **kwargs):
        sandbox.stop.assert_not_awaited()
        assert base_commit == "seed-base" and workdir == "/testbed"
        return PatchCapture.static("agent patch", "worktree", "worktree")

    monkeypatch.setattr(app.SWENextResourcesServer, "_create_sandbox", create)
    monkeypatch.setattr(app, "prepare_git_for_commits", AsyncMock())
    monkeypatch.setattr(app, "capture_model_patch", capture)
    grade = AsyncMock(
        return_value=VerificationResult(completed, completed, True, {}, "log", None if completed else "timeout")
    )
    monkeypatch.setattr(app, "run_verification", grade)
    server = app.SWENextResourcesServer(
        config=app.SWENextResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="swe_next",
            sandbox_provider="docker",
            sandbox_config={},
            apply_anti_cheating=False,
            verification_log_dir=tmp_path,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    row = {
        "instance_id": "example",
        "image_ref": "image",
        "workdir": "/testbed",
        "expected_output_json": '{"test": "PASSED"}',
    }
    seed = (
        {
            "resources_session_id": "session",
            "episode_id": {"rollout_id": "rollout"},
            "task_id": {"taskset": "swe_next", "task_id": "example"},
            "task_data": row,
        }
        if typed
        else row
    )
    response = {
        "output": [],
        "id": "",
        "created_at": 0,
        "model": "",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    verify = row | {"responses_create_params": {"input": "original prompt"}, "response": response}
    with TestClient(server.setup_webserver()) as client:
        seeded = client.post("/seed_session", json=seed)
        assert seeded.status_code == 200, seeded.text
        body = seeded.json()
        assert (body["sandbox_access"]["workdir"] if typed else body["workdir"]) == "/testbed"
        result = client.post("/verify", json=verify)
        assert result.status_code == 200, result.text
        assert result.json()["evaluation_completed"] is completed
        assert result.json()["reward"] == float(completed)
        if not completed:
            assert result.json()["mask_sample"] is True
        assert grade.await_count == (1 if completed else 2)
        assert grade.call_args.kwargs["inputs"].expected_output_json == row["expected_output_json"]
        if typed:
            assert client.post("/verify", json=verify).json() == result.json()
            assert grade.await_count == (1 if completed else 2)
    assert all(s.stop.await_count == 1 for s in sandboxes)
    assert not server._session_id_to_sandbox and not server._verifier_sandboxes
