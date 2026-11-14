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
        return_value=VerificationResult(
            completed, completed, True, {}, "log", None if completed else "provider unavailable"
        )
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


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize(
    "outcome",
    ["correct", "incorrect", "grading_timeout", "grading_provider_error", "capture_provider_error", "git_failure"],
)
def test_routes_execute_capture_and_grading(monkeypatch, tmp_path, typed, outcome):
    """Control only sandbox I/O; keep capture, verifier, retries and response accounting real."""
    from resources_servers.swe_next.verification import TEST_OUTPUT_BEGIN, TEST_OUTPUT_END

    patch = "diff --git a/code.py b/code.py\n--- a/code.py\n+++ b/code.py\n@@ -1 +1 @@\n-old\n+new\n"
    sandboxes = []
    verifier_files = []
    grading_calls = []
    capture_calls = []

    async def create(self, body, files=None, *, session_id=None, verifier_session_id=None):
        sandbox = MagicMock()
        sandbox.sandbox_id = f"sb-{len(sandboxes)}"
        sandbox.stop = AsyncMock()
        sandbox.serialize = AsyncMock(return_value={"sandbox_id": sandbox.sandbox_id})

        async def execute(command, timeout_s=None):
            sandbox.stop.assert_not_awaited()
            code, error_type, output = 0, None, ""
            if verifier_session_id is not None:
                assert command == "bash /tmp/nemo_gym_eval.sh"
                grading_calls.append(command)
                if outcome == "grading_timeout":
                    code, error_type = 124, "timeout"
                elif outcome == "grading_provider_error":
                    code, error_type = 125, "sandbox"
                status = "PASSED" if outcome == "correct" else "FAILED"
                output = f"{TEST_OUTPUT_BEGIN}\n{status} r2e_tests/test.py::test\n{TEST_OUTPUT_END}"
            elif "status --porcelain" in command:
                capture_calls.append(command)
                if outcome == "capture_provider_error":
                    # error_type takes precedence even if a provider also reports exit 0.
                    code, error_type = 0, "sandbox"
                elif outcome == "git_failure":
                    code = 128
                output = " M code.py\n"
            elif "HEAD_SHA" in command:
                output = "HEAD_SHA seed-base\n"
            elif "add -N" in command:
                output = patch
            elif "rev-parse HEAD" in command:
                output = "seed-base\n"
            return SandboxExecResult(return_code=code, error_type=error_type, stdout=output, stderr="")

        sandbox.exec = AsyncMock(side_effect=execute)
        sandboxes.append(sandbox)
        if session_id is not None:
            self._session_id_to_sandbox[session_id] = sandbox
        if verifier_session_id is not None:
            self._verifier_sandboxes[verifier_session_id] = sandbox
            verifier_files.append(files)
        return sandbox

    monkeypatch.setattr(app.SWENextResourcesServer, "_create_sandbox", create)
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
        # A scalar task_id remains an ordinary flat-row field.
        "task_id": "dataset-row-id",
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
    with TestClient(server.setup_webserver()) as client:
        seeded = client.post("/seed_session", json=seed)
        assert seeded.status_code == 200, seeded.text
        if typed:
            assert client.post("/seed_session", json=seed).json() == seeded.json()
            assert len(sandboxes) == 1
        verify = row | {"responses_create_params": {"input": "repair the code"}, "response": response}
        result = client.post("/verify", json=verify)
        assert result.status_code == 200, result.text
        body = result.json()
        masked = outcome in {"grading_provider_error", "capture_provider_error"}
        assert body["reward"] == float(outcome == "correct")
        assert body["evaluation_completed"] is (not masked)
        assert bool(body.get("mask_sample")) is masked
        assert len(capture_calls) == 1
        expected_attempts = (
            0 if outcome == "capture_provider_error" else 2 if outcome == "grading_provider_error" else 1
        )
        assert len(grading_calls) == expected_attempts
        for files in verifier_files:
            assert files.get("/tmp/nemo_gym_patch.diff", "") == ("" if outcome == "git_failure" else patch)
        if masked:
            assert body["failure_kind"] == "verifier_error"
        if typed:
            assert client.post("/verify", json=verify).json() == body
            assert len(capture_calls) == 1 and len(grading_calls) == expected_attempts
            close = {"resources_session_id": "session", "episode_id": {"rollout_id": "rollout"}}
        else:
            close = {}
        assert client.post("/close_session", json=close).status_code == 200
    assert not server._session_id_to_sandbox and not server._verifier_sandboxes
    assert all(sandbox.stop.await_count == 1 for sandbox in sandboxes)


@pytest.mark.parametrize(
    "partial",
    [
        {"resources_session_id": "session"},
        {"resources_session_id": "session", "episode_id": {"rollout_id": "rollout"}},
        {"episode_id": {"rollout_id": "rollout"}, "task_id": {"taskset": "swe_next", "task_id": "example"}},
    ],
)
def test_partial_episode_seed_is_rejected_before_allocation(monkeypatch, partial):
    create = AsyncMock()
    monkeypatch.setattr(app.SWENextResourcesServer, "_create_sandbox", create)
    server = app.SWENextResourcesServer(
        config=app.SWENextResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="swe_next",
            sandbox_provider="docker",
            sandbox_config={},
            apply_anti_cheating=False,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json={"instance_id": "example", "image_ref": "image"} | partial)
        assert response.status_code == 422, response.text
    create.assert_not_awaited()
