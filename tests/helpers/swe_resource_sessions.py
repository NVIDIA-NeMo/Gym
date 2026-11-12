# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise each SWE resources adapter through HTTP with controlled sandbox and grader outcomes."""

import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.client.server_client import ServerClient
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_runtime.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.server_runtime.testing.session_conformance import check_resources_session_contract
from resources_servers.swebench.patch_capture import CaptureCommandError, PatchCapture


RESPONSE = {
    "output": [],
    "id": "",
    "created_at": 0,
    "model": "",
    "object": "response",
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}


def check_swe_resource_episode(
    monkeypatch: MonkeyPatch, tmp_path: Path, module_name: str, prefix: str, scenario: str
) -> None:
    app = importlib.import_module(f"resources_servers.{module_name}.app")
    verification = importlib.import_module(f"resources_servers.{module_name}.verification")
    row_file = Path(app.__file__).parent / "data/example.jsonl"
    row = json.loads(row_file.read_text().splitlines()[0])
    row = row.get("verifier_metadata", row)
    server_type = getattr(app, prefix + "ResourcesServer")
    config_type = getattr(app, prefix + "ResourcesServerConfig")
    sandboxes = []
    completed = scenario != "incomplete"
    resolved = scenario not in {"incorrect", "incomplete", "git_failure"}

    async def create(self, body, files=None, *, session_id=None, verifier_session_id=None):
        sandbox = MagicMock()
        sandbox.sandbox_id = f"sb-{len(sandboxes)}"
        sandbox.exec = AsyncMock(return_value=SandboxExecResult(return_code=0, stdout="seed-base\n", stderr=""))
        sandbox.serialize = AsyncMock(return_value={"sandbox_id": sandbox.sandbox_id})
        sandbox.stop = AsyncMock()
        sandboxes.append(sandbox)
        if session_id is not None:
            self._session_id_to_sandbox[session_id] = sandbox
        if verifier_session_id is not None:
            self._verifier_sandboxes[verifier_session_id] = sandbox
        return sandbox

    capture = AsyncMock(return_value=PatchCapture.static("agent patch", "worktree", "worktree"))
    if scenario == "capture_unavailable":
        capture.side_effect = CaptureCommandError("provider timed out", return_code=None, error_type="timeout")
    elif scenario == "git_failure":
        capture.side_effect = CaptureCommandError("git failed", return_code=128, error_type=None)
    grade = AsyncMock(
        return_value=verification.VerificationResult(
            completed=completed,
            resolved=resolved,
            patch_applied=completed,
            test_results={},
            test_output="log",
            error=None if completed else "provider unavailable",
        )
    )
    monkeypatch.setattr(server_type, "_create_sandbox", create)
    monkeypatch.setattr(app, "prepare_git_for_commits", AsyncMock())
    monkeypatch.setattr(app, "capture_model_patch", capture)
    monkeypatch.setattr(app, "run_verification", grade)
    if hasattr(app, "resolve_parser"):
        monkeypatch.setattr(app, "resolve_parser", lambda *args: MagicMock())
    server = server_type(
        config=config_type(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name=module_name,
            sandbox_provider="docker",
            sandbox_config={},
            apply_anti_cheating=False,
            verification_log_dir=tmp_path,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    seed = {
        "resources_session_id": "session",
        "episode_id": {"rollout_id": "rollout"},
        "task_id": {"taskset": module_name, "task_id": row["instance_id"]},
        "task_data": row,
    }
    close = {"resources_session_id": "session", "episode_id": {"rollout_id": "rollout"}}
    verify = row | {"responses_create_params": {"input": "original prompt"}, "response": RESPONSE}
    if scenario == "conformance":
        check_resources_session_contract(
            server.setup_webserver(), ResourcesSeedSessionRequest.model_validate(seed), keeps_state=True
        )
        return
    legacy = scenario == "legacy"
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=row if legacy else seed)
        assert seeded.status_code == 200, seeded.text
        assert (
            seeded.json()["sandbox_handle"]
            if legacy
            else seeded.json()["sandbox_access"]["connection"]["descriptor"]["sandbox_id"]
        ) == "sb-0"
        if not legacy:
            assert client.post("/seed_session", json=seed).json() == seeded.json()
            assert len(sandboxes) == 1
        if scenario == "close_without_verify":
            sandboxes[0].stop.side_effect = [RuntimeError("stop unavailable"), None]
            assert client.post("/close_session", json=close).status_code == 500
            assert server._session_id_to_sandbox["session"] is sandboxes[0]
            assert client.post("/close_session", json=close).status_code == 200
            assert client.post("/seed_session", json=seed).status_code == 409
            grade.assert_not_awaited()
            return
        if scenario == "task_mismatch":
            invalid = verify | {"instance_id": "another-task"}
            assert client.post("/verify", json=invalid).status_code == 409
            capture.assert_not_awaited()
        if scenario == "stop_retry":
            sandboxes[0].stop.side_effect = [RuntimeError("stop unavailable"), None]
            failed = client.post("/verify", json=verify)
            assert failed.status_code == 500, failed.text
            capture.assert_awaited_once()
            grade.assert_not_awaited()
        result = client.post("/verify", json=verify)
        assert result.status_code == 200, result.text
        body = result.json()
        capture.assert_awaited_once()
        if scenario == "capture_unavailable":
            assert body["mask_sample"] is True and body["evaluation_completed"] is False
            grade.assert_not_awaited()
        else:
            assert body["evaluation_completed"] is completed
            assert body["reward"] == float(resolved)
            assert bool(body.get("mask_sample")) is (not completed)
            assert grade.await_count == (1 if completed else 2)
            assert grade.call_args.kwargs["inputs"].patch == ("" if scenario == "git_failure" else "agent patch")
        if not legacy:
            calls = grade.await_count
            assert client.post("/verify", json=verify).json() == body
            assert grade.await_count == calls
            assert client.post("/close_session", json=close).status_code == 200
            assert client.post("/close_session", json=close).status_code == 200
            assert client.post("/seed_session", json=seed).status_code == 409
    assert not server._session_id_to_sandbox and not server._verifier_sandboxes
    assert all(s.stop.await_count >= 1 for s in sandboxes)


async def check_swe_verifier_interruption(
    tmp_path: Path, module_name: str, prefix: str, error_type: str | None
) -> None:
    app = importlib.import_module(f"resources_servers.{module_name}.app")
    verification = importlib.import_module(f"resources_servers.{module_name}.verification")
    row = json.loads((Path(app.__file__).parent / "data/example.jsonl").read_text().splitlines()[0])
    server = getattr(app, prefix + "ResourcesServer")(
        config=getattr(app, prefix + "ResourcesServerConfig")(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name=module_name,
            sandbox_provider="docker",
            sandbox_config={},
            apply_anti_cheating=False,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    task = getattr(app, prefix + "InstanceRequest").model_validate(row)
    inputs = server._inputs(task, task.patch)
    sandbox = MagicMock()
    sandbox.exec = AsyncMock(
        return_value=SandboxExecResult(
            return_code=0 if error_type else None,
            error_type=error_type,
            stdout="partial passing-test output",
            stderr="interrupted",
        )
    )
    kwargs = {"parser": MagicMock()} if module_name == "swe_rebench" else {}
    result = await verification.run_verification(sandbox, inputs, log_dir=tmp_path, **kwargs)
    assert result.completed is False and result.resolved is False
    assert result.error == f"Test execution did not complete: {error_type or 'missing exit code'}"
    assert "partial passing-test output" in result.test_output
    assert (tmp_path / "test_output.log").read_text() == result.test_output
