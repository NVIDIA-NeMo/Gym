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
from resources_servers.workspace_bench import app


_ROW = {"task_id": "3", "task_dir": "task_lite_clean_en/3"}
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


def _workspace_archive() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name, content in (("input/task.txt", b"edited by the agent"), ("output/answer.txt", b"answer")):
            item = tarfile.TarInfo(name)
            item.size = len(content)
            tar.addfile(item, io.BytesIO(content))
    return stream.getvalue()


# Built once: gzip stamps the time into the header, so two builds a second apart differ.
_WORKSPACE_ARCHIVE = _workspace_archive()


def _server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[app.WorkspaceBenchResourcesServer, list[MagicMock], list[Path]]:
    """A server whose sandboxes are fakes and whose judge passes output graded against the original input."""
    task_dir = tmp_path / "task_lite_clean_en" / "3"
    (task_dir / "source").mkdir(parents=True)
    (task_dir / "source" / "original.txt").write_text("original input")
    manifest = {"data_manifest": [{"stored_relpath": "source/original.txt", "filename": "task.txt"}]}
    (task_dir / "metadata.json").write_text(json.dumps(manifest))
    (tmp_path / "task_lite_clean_en" / "broken").mkdir()
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
            Path(local).write_bytes(_WORKSPACE_ARCHIVE)

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

    judged: list[Path] = []

    async def judge(self, directory: Path, session_id: str) -> tuple[list, dict]:
        judged.append(directory)
        passed = (directory / "output" / "answer.txt").is_file() and (
            directory / "data" / "task.txt"
        ).read_text() == "original input"
        return [{"passed": passed}], {}

    monkeypatch.setattr(app.WorkspaceBenchResourcesServer, "_judge", judge)
    config = app.WorkspaceBenchConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="workspace_bench",
        judge_base_url="https://judge.example",
        judge_api_key="test",  # pragma: allowlist secret
        judge_model="judge",
        sandbox_provider="sandbox",
        sandbox_config={"image": "workspace-bench", "ttl_s": 600},
    )
    server = app.WorkspaceBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return server, sandboxes, judged


def _seed(session_id: str = "resources-session") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="workspace_bench", task_id=_ROW["task_id"]),
        task_data=_ROW | {"responses_create_params": _PARAMS},
    )


def test_episode_session_lends_the_sandbox_judges_once_and_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes, judged = _server(monkeypatch, tmp_path)
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
        assert set(server._reference_dirs) == {"resources-session"}

        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0 and verified.json()["mask_sample"] is False
        sandboxes[0].stop.assert_awaited_once()
        assert server._reference_dirs == {}
        assert client.post("/verify", json=_VERIFY).json() == verified.json()
        assert len(judged) == 1

        assert client.post("/close_session", json=_CLOSE).status_code == 200
        assert client.post("/seed_session", json=seed).status_code == 409
        late = client.post("/verify", json=_VERIFY)
        assert late.json()["mask_sample"] is True and late.json()["failure_kind"] == SESSION_LOST

    assert len(sandboxes) == 1
    sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {}


def test_retried_verify_grades_the_capture_against_the_inputs_frozen_at_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes, judged = _server(monkeypatch, tmp_path)
    original_judge = app.WorkspaceBenchResourcesServer._judge
    calls = 0

    async def judge_sandbox_fails_once(self, directory: Path, session_id: str) -> tuple[list, dict]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Workspace-Bench judge failed: judge sandbox lost")
        return await original_judge(self, directory, session_id)

    monkeypatch.setattr(app.WorkspaceBenchResourcesServer, "_judge", judge_sandbox_fails_once)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        (tmp_path / "task_lite_clean_en" / "3" / "source" / "original.txt").write_text("changed after seed")
        assert client.post("/verify", json=_VERIFY).status_code == 500
        sandboxes[0].stop.assert_awaited_once()
        retried = client.post("/verify", json=_VERIFY)
        assert retried.status_code == 200, retried.text
        assert retried.json()["reward"] == 1.0
        sandboxes[0].download.assert_awaited_once()
        assert client.post("/close_session", json=_CLOSE).status_code == 200
    assert len(judged) == 1


def test_artifact_retention_survives_a_retried_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server, sandboxes, _ = _server(monkeypatch, tmp_path)
    server.config.artifact_root = tmp_path / "artifacts"
    original_judge = app.WorkspaceBenchResourcesServer._judge
    calls = 0

    async def judge_sandbox_fails_once(self, directory: Path, session_id: str) -> tuple[list, dict]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Workspace-Bench judge failed: judge sandbox lost")
        return await original_judge(self, directory, session_id)

    monkeypatch.setattr(app.WorkspaceBenchResourcesServer, "_judge", judge_sandbox_fails_once)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        assert client.post("/verify", json=_VERIFY).status_code == 500
        retried = client.post("/verify", json=_VERIFY)
        assert retried.status_code == 200, retried.text
    (saved,) = (tmp_path / "artifacts").iterdir()
    assert json.loads((saved / "result.json").read_text())["reward"] == 1.0
    with tarfile.open(saved / "input.tar.gz", "r:gz") as archive:
        assert archive.extractfile("task.txt").read() == b"original input"
    assert (saved / "workspace.tar.gz").read_bytes() == _WORKSPACE_ARCHIVE


def test_failed_collection_stops_the_sandbox_and_a_retry_is_masked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes, judged = _server(monkeypatch, tmp_path)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        sandboxes[0].exec.return_value = SandboxExecResult(
            return_code=None, stdout="", stderr="", error_type="timeout"
        )
        assert client.post("/verify", json=_VERIFY).status_code == 500
        sandboxes[0].stop.assert_awaited_once()
        assert server._reference_dirs == {}
        retried = client.post("/verify", json=_VERIFY)
        assert retried.json()["mask_sample"] is True and retried.json()["failure_kind"] == SESSION_LOST
        assert client.post("/close_session", json=_CLOSE).status_code == 200
    assert judged == []


def test_failed_collection_whose_stop_fails_keeps_the_inputs_for_the_retried_collection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes, judged = _server(monkeypatch, tmp_path)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        sandboxes[0].exec.return_value = SandboxExecResult(
            return_code=None, stdout="", stderr="", error_type="timeout"
        )
        sandboxes[0].stop.side_effect = [RuntimeError("provider unavailable"), None]
        assert client.post("/verify", json=_VERIFY).status_code == 500
        assert server._session_id_to_sandbox == {"resources-session": sandboxes[0]}
        assert set(server._reference_dirs) == {"resources-session"}
        sandboxes[0].exec.return_value = SandboxExecResult(return_code=0, stdout="", stderr="")
        retried = client.post("/verify", json=_VERIFY)
        assert retried.status_code == 200, retried.text
        assert retried.json()["reward"] == 1.0
        assert server._reference_dirs == {}
        assert client.post("/close_session", json=_CLOSE).status_code == 200
    assert sandboxes[0].stop.await_count == 2
    assert len(judged) == 1


@pytest.mark.parametrize(("return_code", "graded"), [(1, True), (2, False)])
def test_files_the_agent_left_changing_are_graded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, return_code: int, graded: bool
) -> None:
    """tar exits 1 when the agent's own process changed a file mid-archive; those files are the agent's result."""
    server, sandboxes, judged = _server(monkeypatch, tmp_path)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        sandboxes[0].exec.return_value = SandboxExecResult(return_code=return_code, stdout="", stderr="file changed")
        verified = client.post("/verify", json=_VERIFY)
        assert client.post("/close_session", json=_CLOSE).status_code == 200
    collect = sandboxes[0].exec.await_args.args[0]
    assert "mkdir -p /workspace/input /workspace/output" in collect, "folders the agent removed are graded as empty"
    if graded:
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0 and verified.json()["mask_sample"] is False
    else:
        assert verified.status_code == 500
        assert judged == []


def test_agent_run_verify_that_fails_before_collection_stops_the_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An agent-run session gets no close, so a verify that fails while saving its inputs still stops the sandbox."""
    server, sandboxes, _ = _server(monkeypatch, tmp_path)
    server.config.artifact_root = tmp_path / "artifacts"
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_ROW | {"responses_create_params": _PARAMS}).status_code == 200

        def full_disk(source: Path, destination: Path) -> None:
            raise OSError("no space left on device")

        monkeypatch.setattr(app.shutil, "copyfile", full_disk)
        assert client.post("/verify", json=_VERIFY).status_code == 500
        sandboxes[0].stop.assert_awaited_once()
        assert server._session_id_to_sandbox == {} and server._reference_dirs == {}
    sandboxes[0].download.assert_not_awaited()


def test_malformed_typed_seed_is_not_taken_for_an_agent_run_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes, _ = _server(monkeypatch, tmp_path)
    malformed = _ROW | {"responses_create_params": _PARAMS, "resources_session_id": "resources-session"}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=malformed).status_code == 422
    assert sandboxes == [] and server._reference_dirs == {}


@pytest.mark.parametrize(
    "metadata",
    ["{not json", json.dumps({"data_manifest": [{"stored_relpath": "source/original.txt"}]}), "[]"],
    ids=["invalid-json", "manifest-entry-without-filename", "not-an-object"],
)
def test_task_with_broken_metadata_is_refused_without_a_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, metadata: str
) -> None:
    server, sandboxes, _ = _server(monkeypatch, tmp_path)
    (tmp_path / "task_lite_clean_en" / "3" / "metadata.json").write_text(metadata)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        response = client.post("/seed_session", json=_seed().model_dump(mode="json"))
    assert response.status_code == 422, response.text
    assert sandboxes == [] and server._reference_dirs == {}


def test_close_before_verify_stops_the_sandbox_and_drops_the_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes, _ = _server(monkeypatch, tmp_path)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        (reference_dir,) = server._reference_dirs.values()
        assert client.post("/close_session", json=_CLOSE).status_code == 200
        assert client.post("/close_session", json=_CLOSE).status_code == 200
    sandboxes[0].stop.assert_awaited_once()
    assert not Path(reference_dir.name).exists()
    assert server._reference_dirs == {}


def test_unusable_task_is_refused_without_a_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server, sandboxes, _ = _server(monkeypatch, tmp_path)
    seed = _seed().model_dump(mode="json")
    seed["task_data"]["task_dir"] = "task_lite_clean_en/broken"
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        response = client.post("/seed_session", json=seed)
    assert response.status_code == 422, response.text
    assert sandboxes == [] and server._reference_dirs == {}


def test_agent_run_seed_returns_the_full_descriptor_and_rejects_a_repeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server, sandboxes, _ = _server(monkeypatch, tmp_path)
    row = _ROW | {"responses_create_params": _PARAMS}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=row)
        assert seeded.status_code == 200, seeded.text
        assert seeded.json() == {"sandbox_handle": "sb-1", "sandbox_descriptor": {"sandbox_id": "sb-1", "scope": None}}
        assert client.post("/seed_session", json=row).status_code == 409
        verified = client.post("/verify", json=_VERIFY)
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0
    assert len(sandboxes) == 1
    sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {} and server._reference_dirs == {}


def test_session_conformance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    server, _, _ = _server(monkeypatch, tmp_path)
    check_resources_session_contract(
        server.setup_webserver(),
        _seed("conformance-session"),
        keeps_state=True,
        holds_state=lambda session_id: session_id in server._session_id_to_sandbox
        or session_id in server._reference_dirs,
        verify_body=_VERIFY,
    )
