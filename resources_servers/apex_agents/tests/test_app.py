# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.apex_agents.app import (
    ApexResourcesServer,
    ApexResourcesServerConfig,
    ApexSeedSessionRequest,
    ApexVerifyRequest,
)


def _snapshot(content: str | None = "deliverable") -> str:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        if content is not None:
            archive.writestr("filesystem/final.txt", content)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _body(snapshot: str | None = None) -> ApexVerifyRequest:
    return ApexVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Do the task"}]},
            "response": {
                "id": "r1",
                "created_at": 0,
                "model": "policy",
                "object": "response",
                "output": [
                    {
                        "id": "m1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Done", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
            "task_id": "task-1",
            "world_id": "world-1",
            "verifier_metadata": {
                "rubric": [{"verifier_id": "v1", "criteria": "Correct result"}],
                "gold_response": "held out",
            },
            "initial_artifact_snapshot_b64": _snapshot(None),
            "artifact_snapshot_b64": snapshot if snapshot is not None else _snapshot(),
        }
    )


def _server(
    *,
    world_cache_dir: str = "benchmarks/apex_agents/data/world_cache",
    artifact_output_dir: str | None = None,
) -> ApexResourcesServer:
    config = ApexResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        name="apex_agents",
        entrypoint="app.py",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        world_cache_dir=world_cache_dir,
        artifact_output_dir=artifact_output_dir,
    )
    return ApexResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.asyncio
async def test_verify_scores_artifacts_without_returning_snapshot(monkeypatch) -> None:
    async def fake_judge(**kwargs):
        assert kwargs["artifact_changes"][0].path == "filesystem/final.txt"
        assert kwargs["final_root"].is_dir()
        assert kwargs["rubric"][0]["criteria"] == "Correct result"
        assert kwargs["server_client"] is not None
        assert kwargs["model_server_name"] == "judge"
        assert kwargs["judge_context_window_size"] == 32768
        return 1.0, {"v1": {"score": 1.0}}, {"ok": True, "grading_run_id": "run-1"}

    monkeypatch.setattr("resources_servers.apex_agents.app.grade_apex_output", fake_judge)
    result = await _server().verify(_body())

    assert result.reward == 1.0
    assert result.artifact_paths == ["filesystem/final.txt"]
    assert result.judge_response == {"ok": True, "grading_run_id": "run-1"}
    assert "artifact_snapshot_b64" not in result.model_dump()
    assert "initial_artifact_snapshot_b64" not in result.model_dump()


@pytest.mark.asyncio
async def test_verify_persists_snapshots_and_grading_result(monkeypatch, tmp_path: Path) -> None:
    async def fake_judge(**_kwargs):
        return 0.5, {"v1": {"score": 0.5}}, {"ok": True, "grading_run_id": "run-1"}

    monkeypatch.setattr("resources_servers.apex_agents.app.grade_apex_output", fake_judge)
    result = await _server(artifact_output_dir=str(tmp_path / "saved")).verify(_body())

    output_dir = Path(result.artifact_output_dir)
    assert Path(result.initial_snapshot_path) == output_dir / "initial_snapshot.zip"
    assert Path(result.final_snapshot_path) == output_dir / "final_snapshot.zip"
    assert Path(result.initial_snapshot_path).is_file()
    assert Path(result.final_snapshot_path).is_file()
    grading = json.loads((output_dir / "grading.json").read_text())
    assert grading == {
        "reward": 0.5,
        "rubric_scores": {"v1": {"score": 0.5}},
        "judge_response": {"ok": True, "grading_run_id": "run-1"},
    }


@pytest.mark.asyncio
async def test_verify_returns_saved_snapshots_when_grading_fails(monkeypatch, tmp_path: Path) -> None:
    async def failing_judge(**_kwargs):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr("resources_servers.apex_agents.app.grade_apex_output", failing_judge)
    result = await _server(artifact_output_dir=str(tmp_path / "saved")).verify(_body())

    assert result.reward == 0.0
    assert result.invalid_judge_response is True
    assert result.verifier_error == "judge unavailable"
    assert Path(result.initial_snapshot_path).is_file()
    assert Path(result.final_snapshot_path).is_file()
    grading = json.loads((Path(result.artifact_output_dir) / "grading.json").read_text())
    assert grading["error"] == "judge unavailable"


@pytest.mark.asyncio
async def test_verify_rejects_bad_snapshot() -> None:
    result = await _server().verify(_body("not-base64"))
    assert result.reward == 0.0
    assert result.invalid_judge_response is True
    assert "encoding" in result.verifier_error


@pytest.mark.asyncio
async def test_seed_session_binds_world_to_session(monkeypatch, tmp_path: Path) -> None:
    world = tmp_path / "world.zip"
    world.write_bytes(b"world")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(world)

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    # Avoid leaving Python's default executor alive in this unit test; production
    # deliberately keeps the blocking hub client off the event loop.
    monkeypatch.setattr("resources_servers.apex_agents.app.asyncio.to_thread", run_inline)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    server = _server(world_cache_dir=str(cache_dir))
    request = Request(scope={"type": "http", "session": {SESSION_ID_KEY: "session-1"}})
    body = ApexSeedSessionRequest(
        task_id="task-1",
        world_id="world_9797d81fa71c4dbfb192e89a0f2ac811",
    )

    response = await server.seed_session(request, body)
    download = await server.world(request)

    assert response.world_ready is True
    assert Path(download.path) == world
    assert calls == [
        {
            "repo_id": "mercor/apex-agents",
            "filename": "world_files_zipped/world_9797d81fa71c4dbfb192e89a0f2ac811.zip",
            "repo_type": "dataset",
            "cache_dir": str(cache_dir),
            "local_files_only": True,
        },
        {
            "repo_id": "mercor/apex-agents",
            "filename": "world_files_zipped/world_9797d81fa71c4dbfb192e89a0f2ac811.zip",
            "repo_type": "dataset",
            "cache_dir": str(cache_dir),
            "local_files_only": True,
        },
    ]
    with pytest.raises(HTTPException, match="seed_session"):
        await server.world(request)


def test_seed_session_rejects_invalid_world_id() -> None:
    with pytest.raises(ValueError):
        ApexSeedSessionRequest(task_id="task-1", world_id="../../grader")


def test_world_download_is_always_offline(monkeypatch, tmp_path: Path) -> None:
    world = tmp_path / "world.zip"
    world.write_bytes(b"world")
    download = MagicMock(return_value=str(world))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)

    cache_dir = tmp_path / "cache"
    result = _server(world_cache_dir=str(cache_dir))._download_world("world_9797d81fa71c4dbfb192e89a0f2ac811")

    assert result == str(world)
    download.assert_called_once_with(
        repo_id="mercor/apex-agents",
        filename="world_files_zipped/world_9797d81fa71c4dbfb192e89a0f2ac811.zip",
        repo_type="dataset",
        cache_dir=str(cache_dir),
        local_files_only=True,
    )


async def test_world_cache_preflight_is_registered(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    server = _server(world_cache_dir=str(cache_dir))
    app = server.setup_webserver()

    assert server._preflight_world_cache in app.router.on_startup
    await server._preflight_world_cache()


async def test_world_cache_preflight_has_actionable_failure(tmp_path: Path) -> None:
    server = _server(world_cache_dir=str(tmp_path / "missing"))

    with pytest.raises(RuntimeError, match="gym eval prepare --benchmark apex_agents"):
        await server._preflight_world_cache()


async def test_seed_session_reports_missing_preprocessed_world(monkeypatch, tmp_path: Path) -> None:
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("resources_servers.apex_agents.app.asyncio.to_thread", run_inline)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        MagicMock(side_effect=FileNotFoundError("not cached")),
    )
    server = _server(world_cache_dir=str(tmp_path))
    request = Request(scope={"type": "http", "session": {SESSION_ID_KEY: "session-1"}})
    body = ApexSeedSessionRequest(
        task_id="task-1",
        world_id="world_9797d81fa71c4dbfb192e89a0f2ac811",
    )

    with pytest.raises(HTTPException, match="gym eval prepare --benchmark apex_agents") as exc_info:
        await server.seed_session(request, body)

    assert exc_info.value.status_code == 503


def test_persist_submission_writes_task_artifacts_without_rubric(tmp_path: Path) -> None:
    extract_root = tmp_path / "extracted"
    artifact = extract_root / "filesystem" / "deliverable.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("finished work", encoding="utf-8")
    archive_path = tmp_path / "final.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(artifact, "filesystem/deliverable.txt")
    body = _body()
    body.__pydantic_extra__ = {
        **(body.__pydantic_extra__ or {}),
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 1,
    }

    initial_archive_path = tmp_path / "initial.zip"
    with zipfile.ZipFile(initial_archive_path, "w"):
        pass
    output_dir = _server(artifact_output_dir=str(tmp_path / "saved"))._persist_submission(
        body,
        initial_archive_path,
        archive_path,
        extract_root,
        ["filesystem/deliverable.txt"],
    )

    assert output_dir is not None
    assert output_dir.parent.name == "task-1"
    assert output_dir.name.startswith("rollout_2_attempt_1_")
    assert (output_dir / "initial_snapshot.zip").is_file()
    assert (output_dir / "final_snapshot.zip").is_file()
    assert (output_dir / "artifacts" / "filesystem" / "deliverable.txt").read_text() == "finished work"
    submission = json.loads((output_dir / "submission.json").read_text())
    assert submission["response"]["output"][0]["content"][0]["text"] == "Done"
    assert "verifier_metadata" not in submission
    assert "artifact_snapshot_b64" not in submission
