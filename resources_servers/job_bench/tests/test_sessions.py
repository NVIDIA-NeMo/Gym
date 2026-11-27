# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Episode sessions and the agent-run seed, through the server's HTTP routes with a fake sandbox."""

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import SESSION_LOST, EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_runtime import ResourcesSeedSessionRequest, check_resources_session_contract
from nemo_gym.verification import JudgeError
from resources_servers.job_bench import app


_ROW = {"task_id": "jobs/task1", "task_dir": "dataset/jobs/task1", "rubrics_file": "dataset/jobs/task1/RUBRICS.json"}
_RESPONSE = {
    "id": "",
    "created_at": 0,
    "model": "",
    "object": "response",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}
_PARAMS = {"input": [{"role": "user", "content": "Do the task"}]}
_VERIFY = _ROW | {"responses_create_params": _PARAMS, "response": _RESPONSE}
_CLOSE = {"resources_session_id": "resources-session", "episode_id": {"rollout_id": "rollout"}}


def _output_archive() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        item = tarfile.TarInfo("./report.txt")
        item.size = 6
        tar.addfile(item, io.BytesIO(b"answer"))
    return stream.getvalue()


def _server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[app.JobBenchResourcesServer, list[MagicMock]]:
    """A server whose sandboxes are fakes and whose judge passes any output that holds report.txt."""
    task_dir = tmp_path / "dataset" / "jobs" / "task1"
    (task_dir / "task_folder").mkdir(parents=True)
    (task_dir / "task_folder" / "TASK_INSTRUCTIONS.txt").write_text("Write a report.")
    (task_dir / "RUBRICS.json").write_text('{"rubrics": []}')
    (tmp_path / "dataset" / "jobs" / "broken").mkdir()
    monkeypatch.setattr(app, "resolve_task_path", lambda path: tmp_path / path)
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *args: {})
    monkeypatch.setattr(app, "create_provider", lambda config: None)

    sandboxes: list[MagicMock] = []

    def new_sandbox(provider: object) -> MagicMock:
        sandbox_id = f"sb-{len(sandboxes) + 1}"

        async def serialize(*, scope: str | None = None) -> dict[str, str | None]:
            return {"sandbox_id": sandbox_id, "scope": scope}

        async def download(remote: str, local: Path) -> None:
            Path(local).write_bytes(_output_archive())

        sandbox = MagicMock()
        sandbox.start = AsyncMock()
        sandbox.upload = AsyncMock()
        sandbox.exec = AsyncMock(return_value=SandboxExecResult(return_code=0, stdout="", stderr=""))
        sandbox.download = AsyncMock(side_effect=download)
        sandbox.serialize = AsyncMock(side_effect=serialize)
        sandbox.stop = AsyncMock()
        sandboxes.append(sandbox)
        return sandbox

    monkeypatch.setattr(app, "AsyncSandbox", new_sandbox)

    def judge(self, output_dir: Path, rubrics_file: Path) -> tuple[dict, list]:
        passed = int((output_dir / "report.txt").is_file())
        scorecard = {
            "normalized_score": passed,
            "total_score": passed,
            "max_score": 1,
            "passed_count": passed,
            "total_count": 1,
        }
        return scorecard, []

    monkeypatch.setattr(app.JobBenchResourcesServer, "_judge", judge)
    config = app.JobBenchConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="job_bench",
        judge_base_url="https://judge.example",
        judge_api_key="test",  # pragma: allowlist secret
        judge_model="judge",
        sandbox_provider="sandbox",
        sandbox_config={"image": "python:3.13-bookworm", "ttl_s": 600},
    )
    return app.JobBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient)), sandboxes


def _seed(session_id: str = "resources-session", **task: str) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="job_bench", task_id=_ROW["task_id"]),
        task_data=_ROW | {"responses_create_params": _PARAMS} | task,
    )


def test_episode_session_lends_the_sandbox_grades_its_output_once_and_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes = _server(monkeypatch, tmp_path)
    seed = _seed().model_dump(mode="json")
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=seed)
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["sandbox_access"] == {
            "connection": {
                "kind": "direct",
                "provider_config_ref": "sandbox",
                "descriptor": {"sandbox_id": "sb-1", "scope": "operate"},
            },
            "workdir": "/workspace",
        }
        assert client.post("/seed_session", json=seed).json() == seeded.json(), "a repeated seed returns the sandbox"
        sandboxes[0].upload.assert_awaited_once()

        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0 and verified.json()["mask_sample"] is False
        sandboxes[0].stop.assert_awaited_once()
        repeated = client.post("/verify", json=_VERIFY)
        assert repeated.json() == verified.json()
        sandboxes[0].download.assert_awaited_once()
        other = _VERIFY | {"rubrics_file": "dataset/jobs/task1/OTHER.json"}
        assert client.post("/verify", json=other).status_code == 409

        assert client.post("/close_session", json=_CLOSE).status_code == 200
        assert client.post("/seed_session", json=seed).status_code == 409
        late = client.post("/verify", json=_VERIFY)
        assert late.json()["mask_sample"] is True and late.json()["failure_kind"] == SESSION_LOST

    assert len(sandboxes) == 1
    sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {}


def test_close_before_verify_stops_the_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server, sandboxes = _server(monkeypatch, tmp_path)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        assert client.post("/close_session", json=_CLOSE).status_code == 200
        assert client.post("/close_session", json=_CLOSE).status_code == 200
    sandboxes[0].stop.assert_awaited_once()
    sandboxes[0].download.assert_not_awaited()


def test_failed_close_keeps_the_sandbox_for_a_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server, sandboxes = _server(monkeypatch, tmp_path)
    seed = _seed().model_dump(mode="json")
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed).status_code == 200
        sandboxes[0].stop.side_effect = [RuntimeError("provider unavailable"), None]
        assert client.post("/close_session", json=_CLOSE).status_code == 500
        assert server._session_id_to_sandbox == {"resources-session": sandboxes[0]}
        assert client.post("/seed_session", json=seed).status_code == 200, "an unfenced session is still open"
        assert client.post("/close_session", json=_CLOSE).status_code == 200
    assert sandboxes[0].stop.await_count == 2
    assert server._session_id_to_sandbox == {}


def test_failed_output_collection_stops_the_sandbox_and_a_retry_is_masked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes = _server(monkeypatch, tmp_path)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        sandboxes[0].exec.return_value = SandboxExecResult(
            return_code=None, stdout="", stderr="", error_type="timeout"
        )
        assert client.post("/verify", json=_VERIFY).status_code == 500
        sandboxes[0].stop.assert_awaited_once()
        retried = client.post("/verify", json=_VERIFY)
        assert retried.json()["mask_sample"] is True and retried.json()["failure_kind"] == SESSION_LOST
        assert client.post("/close_session", json=_CLOSE).status_code == 200
    sandboxes[0].stop.assert_awaited_once()


def test_judge_failure_is_masked_and_a_retry_regrades_the_saved_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes = _server(monkeypatch, tmp_path)
    server.config.artifact_root = tmp_path / "artifacts"
    judge_calls = 0
    passing_judge = app.JobBenchResourcesServer._judge

    def judge_fails_once(self, output_dir: Path, rubrics_file: Path) -> tuple[dict, list]:
        nonlocal judge_calls
        judge_calls += 1
        if judge_calls == 1:
            raise JudgeError("judge rate-limited")
        return passing_judge(self, output_dir, rubrics_file)

    monkeypatch.setattr(app.JobBenchResourcesServer, "_judge", judge_fails_once)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        failed = client.post("/verify", json=_VERIFY)
        assert failed.status_code == 200, failed.text
        assert failed.json()["mask_sample"] is True and failed.json()["failure_kind"] == "judge_failed"
        sandboxes[0].stop.assert_awaited_once()
        other = _VERIFY | {"response": _RESPONSE | {"id": "other"}}
        assert client.post("/verify", json=other).status_code == 409
        retried = client.post("/verify", json=_VERIFY)
        assert retried.status_code == 200, retried.text
        assert retried.json()["reward"] == 1.0 and retried.json()["mask_sample"] is False
        sandboxes[0].download.assert_awaited_once()
        assert client.post("/verify", json=_VERIFY).json() == retried.json()
        assert client.post("/close_session", json=_CLOSE).status_code == 200
    (saved,) = (tmp_path / "artifacts").iterdir()
    assert json.loads((saved / "result.json").read_text())["reward"] == 1.0
    assert judge_calls == 2


def test_failed_stop_after_collection_keeps_the_sandbox_and_a_retry_grades_the_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes = _server(monkeypatch, tmp_path)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.side_effect = [RuntimeError("provider unavailable"), None]
        assert client.post("/verify", json=_VERIFY).status_code == 500
        assert server._session_id_to_sandbox == {"resources-session": sandboxes[0]}
        retried = client.post("/verify", json=_VERIFY)
        assert retried.status_code == 200, retried.text
        assert retried.json()["reward"] == 1.0
        sandboxes[0].download.assert_awaited_once()
        assert client.post("/close_session", json=_CLOSE).status_code == 200
    assert sandboxes[0].stop.await_count == 2
    assert server._session_id_to_sandbox == {}


@pytest.mark.parametrize(("return_code", "graded"), [(1, True), (2, False)])
def test_output_the_agent_left_changing_is_graded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, return_code: int, graded: bool
) -> None:
    """tar exits 1 when the agent's own process changed a file mid-archive; that output is the agent's result."""
    server, sandboxes = _server(monkeypatch, tmp_path)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        sandboxes[0].exec.return_value = SandboxExecResult(return_code=return_code, stdout="", stderr="file changed")
        verified = client.post("/verify", json=_VERIFY)
        assert client.post("/close_session", json=_CLOSE).status_code == 200
    collect = sandboxes[0].exec.await_args.args[0]
    assert "mkdir -p /workspace/output" in collect, "an output/ the agent removed is graded as empty"
    if graded:
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0 and verified.json()["mask_sample"] is False
    else:
        assert verified.status_code == 500
        sandboxes[0].download.assert_not_awaited()


def test_agent_run_verify_that_fails_before_collection_stops_the_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An agent-run session gets no close, so a verify that fails while saving its request still stops the sandbox."""
    server, sandboxes = _server(monkeypatch, tmp_path)
    server.config.artifact_root = tmp_path / "artifacts"
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_ROW | {"responses_create_params": _PARAMS}).status_code == 200

        def unreachable_snapshot(path: str) -> Path:
            raise ConnectionError("snapshot unavailable")

        monkeypatch.setattr(app, "resolve_task_path", unreachable_snapshot)
        assert client.post("/verify", json=_VERIFY).status_code == 500
        sandboxes[0].stop.assert_awaited_once()
        assert server._session_id_to_sandbox == {}
    sandboxes[0].download.assert_not_awaited()


def test_malformed_typed_seed_is_not_taken_for_an_agent_run_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes = _server(monkeypatch, tmp_path)
    malformed = _ROW | {"responses_create_params": _PARAMS, "resources_session_id": "resources-session"}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=malformed).status_code == 422
    assert sandboxes == []


@pytest.mark.parametrize(
    ("task", "starts"),
    [({"task_dir": "dataset/jobs/broken"}, 0), ({"rubrics_file": None}, 0)],
    ids=["missing-instructions", "invalid-task-data"],
)
def test_unusable_task_is_refused_without_a_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: dict, starts: int
) -> None:
    server, sandboxes = _server(monkeypatch, tmp_path)
    seed = _seed().model_dump(mode="json")
    seed["task_data"] |= task
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        response = client.post("/seed_session", json=seed)
    assert response.status_code == 422, response.text
    assert sum(sandbox.start.await_count for sandbox in sandboxes) == starts
    assert server._session_id_to_sandbox == {}


def test_seed_that_fails_after_start_stops_its_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server, sandboxes = _server(monkeypatch, tmp_path)
    original = app.AsyncSandbox

    def failing_upload(provider: object) -> MagicMock:
        sandbox = original(provider)
        sandbox.upload.side_effect = ConnectionError("upload lost")
        return sandbox

    monkeypatch.setattr(app, "AsyncSandbox", failing_upload)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 500
    sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {}


def test_shutdown_stops_a_sandbox_no_close_released(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server, sandboxes = _server(monkeypatch, tmp_path)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.assert_not_awaited()
    sandboxes[0].stop.assert_awaited_once()


def test_agent_run_seed_returns_the_full_descriptor_and_verify_stops_the_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes = _server(monkeypatch, tmp_path)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=_ROW | {"responses_create_params": _PARAMS})
        assert seeded.status_code == 200, seeded.text
        assert seeded.json() == {"sandbox_handle": "sb-1", "sandbox_descriptor": {"sandbox_id": "sb-1", "scope": None}}
        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0
        sandboxes[0].stop.assert_awaited_once()
        # An agent-run session holds no saved verdict: a second verify finds no sandbox to grade.
        assert client.post("/verify", json=_VERIFY).json()["mask_sample"] is True
    assert server._session_id_to_sandbox == {}


def test_agent_run_seed_cannot_replace_an_episode_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server, sandboxes = _server(monkeypatch, tmp_path)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        replaced = client.post("/seed_session", json=_ROW | {"responses_create_params": _PARAMS})
        assert replaced.status_code == 409, replaced.text
    assert len(sandboxes) == 1


def test_session_conformance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server, _ = _server(monkeypatch, tmp_path)
    check_resources_session_contract(
        server.setup_webserver(),
        _seed("conformance-session"),
        keeps_state=True,
        holds_state=lambda session_id: session_id in server._session_id_to_sandbox,
        verify_body=_VERIFY,
    )
