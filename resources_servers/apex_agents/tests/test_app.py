# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import json
import threading
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.apex_agents.app import (
    VERIFIER_FIXTURE,
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
    task_files_cache_dir: str | None = None,
    artifact_output_dir: str | None = None,
) -> ApexResourcesServer:
    config = ApexResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        name="apex_agents",
        entrypoint="app.py",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        world_cache_dir=world_cache_dir,
        task_files_cache_dir=task_files_cache_dir or world_cache_dir,
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
        assert kwargs["judge_context_window_size"] == 1_000_000
        assert kwargs["scoring_method"] == "all_pass"
        return 1.0, {"v1": {"score": 1.0}}, {"ok": True, "grading_run_id": "run-1"}

    monkeypatch.setattr("resources_servers.apex_agents.app.grade_apex_output", fake_judge)
    result = await _server().verify(_body())

    assert result.reward == 1.0
    assert result.artifact_paths == ["filesystem/final.txt"]
    assert result.judge_response == {"ok": True, "grading_run_id": "run-1"}
    assert "artifact_snapshot_b64" not in result.model_dump()
    assert "initial_artifact_snapshot_b64" not in result.model_dump()


@pytest.mark.asyncio
async def test_verify_passes_the_row_scoring_method_to_the_grader(monkeypatch) -> None:
    async def fake_judge(**kwargs):
        assert kwargs["scoring_method"] == "weighted_average"
        return 0.75, {"v1": {"score": 1.0}}, {"ok": True, "grading_run_id": "run-1"}

    monkeypatch.setattr("resources_servers.apex_agents.app.grade_apex_output", fake_judge)
    body = _body()
    body.verifier_metadata["scoring"] = "weighted_average"
    result = await _server().verify(body)

    assert result.reward == 0.75


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
    # The verifier failed, not the policy: masked, not a scored 0.
    assert result.mask_sample is True
    assert result.model_dump()["failure_kind"] == "verifier_error"
    assert result.model_dump()["evaluation_completed"] is False
    assert result.invalid_judge_response is True
    assert result.verifier_error == "judge unavailable"
    assert Path(result.initial_snapshot_path).is_file()
    assert Path(result.final_snapshot_path).is_file()
    grading = json.loads((Path(result.artifact_output_dir) / "grading.json").read_text())
    assert grading["error"] == "judge unavailable"


def _snapshot_with_files(count: int) -> str:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for index in range(count):
            archive.writestr(f"filesystem/file_{index}.txt", "x")
    return base64.b64encode(output.getvalue()).decode("ascii")


@pytest.mark.asyncio
@pytest.mark.parametrize(("over_limit", "masked"), [("final", False), ("initial", True)])
async def test_a_snapshot_over_its_limit_is_the_policy_s_zero_only_for_the_final_snapshot(
    monkeypatch, over_limit: str, masked: bool
) -> None:
    async def judge(**_kwargs):
        raise AssertionError("an over-limit snapshot is not graded")

    monkeypatch.setattr("resources_servers.apex_agents.app.grade_apex_output", judge)
    server = _server()
    server.config.max_artifact_files = 2
    body = _body()
    field = "artifact_snapshot_b64" if over_limit == "final" else "initial_artifact_snapshot_b64"
    setattr(body, field, _snapshot_with_files(3))

    result = (await server.verify(body)).model_dump()

    # The policy wrote the final snapshot; the world and the task files make the initial one.
    assert result["reward"] == 0.0
    assert result["mask_sample"] is masked
    assert result.get("failure_kind") == ("verifier_error" if masked else None)
    assert "limit is 2" in result["verifier_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["connection", "unreadable_verdict"])
async def test_verify_routes_a_judge_failure_to_the_judge_failed_row(monkeypatch, tmp_path: Path, fault: str) -> None:
    from nemo_gym.verification import JudgeError, judge_failsafe, reraise_judge_errors

    async def failing_judge(**_kwargs):
        if fault == "unreadable_verdict":
            raise JudgeError("judge returned invalid structured output after 3 attempts")

        async def call():
            raise ConnectionError("judge unreachable")

        await reraise_judge_errors(call())

    monkeypatch.setattr("resources_servers.apex_agents.app.grade_apex_output", failing_judge)
    server = _server(artifact_output_dir=str(tmp_path / "saved"))

    # The endpoint is registered behind judge_failsafe; call it the same way.
    response = await judge_failsafe(server.verify)(body=_body())

    row = json.loads(response.body)
    assert row["_ng_failure_class"] == "judge_failed"
    assert row["mask_sample"] is True
    assert "invalid_judge_response" not in row
    saved = next((tmp_path / "saved").rglob("grading.json"))
    assert json.loads(saved.read_text())["error"]
    assert (saved.parent / "final_snapshot.zip").is_file()


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


@pytest.mark.asyncio
async def test_seed_session_serves_prefetched_task_files_overlay(monkeypatch, tmp_path: Path) -> None:
    world = tmp_path / "world.zip"
    world.write_bytes(b"world")
    snapshot_root = tmp_path / "snapshot"
    task_root = snapshot_root / "task_files" / "task_a"
    task_root.mkdir(parents=True)
    (task_root / "source.docx").write_bytes(b"task input")

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    snapshot_download = MagicMock(return_value=str(snapshot_root))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", MagicMock(return_value=str(world)))
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr("resources_servers.apex_agents.app.asyncio.to_thread", run_inline)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    server = _server(world_cache_dir=str(cache_dir))
    request = Request(scope={"type": "http", "session": {SESSION_ID_KEY: "session-1"}})
    body = ApexSeedSessionRequest(
        task_id="task_a",
        world_id="world_9797d81fa71c4dbfb192e89a0f2ac811",
        task_input_files="snap_0123456789abcdef0123456789abcdef",
    )

    await server.seed_session(request, body)
    world_download = await server.world(request)
    task_download = await server.task_files(request)

    assert Path(world_download.path) == world
    with zipfile.ZipFile(task_download.path) as archive:
        assert archive.namelist() == ["filesystem/source.docx"]
        assert archive.read("filesystem/source.docx") == b"task input"
    snapshot_download.assert_called_once_with(
        repo_id="mercor/apex-agents",
        repo_type="dataset",
        cache_dir=str(cache_dir),
        allow_patterns=["task_files/task_a/**"],
        local_files_only=True,
    )
    with pytest.raises(HTTPException, match="seed_session"):
        await server.task_files(request)


def test_seed_session_rejects_invalid_world_id() -> None:
    with pytest.raises(ValueError):
        ApexSeedSessionRequest(task_id="task-1", world_id="../../grader")


def test_seed_session_rejects_invalid_task_attachment_id() -> None:
    with pytest.raises(ValueError):
        ApexSeedSessionRequest(
            task_id="task-1",
            world_id="world_9797d81fa71c4dbfb192e89a0f2ac811",
            task_input_files="../../attachment.zip",
        )


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


@pytest.mark.asyncio
@pytest.mark.parametrize("determinism", ["stochastic", "seeded"])
async def test_verifier_fixture_cases_match_the_scorer(determinism: str) -> None:
    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism=determinism
    )
    observed = [(result.name, result.kind, result.observed_rewards) for result in results]
    expected = [
        ("every_criterion_passes", "full_reward", (1.0,)),
        ("one_failed_criterion_fails_all_pass", "zero_reward", (0.0,)),
        ("weighted_average_gives_partial_credit", "determinism", (0.25, 0.25)),
        ("empty_rubric", "malformed", ()),
        ("missing_final_snapshot", "malformed", ()),
        ("unparseable_judge_reply", "malformed", ()),
    ]
    if determinism != "seeded":
        expected = [row for row in expected if row[1] != "determinism"]
    assert observed == expected


def _blocks_body(input_items) -> ApexVerifyRequest:
    payload = _body().model_dump()
    payload["responses_create_params"] = {"input": input_items}
    payload["initial_artifact_snapshot_b64"] = _snapshot(None)
    payload["artifact_snapshot_b64"] = _snapshot()
    return ApexVerifyRequest.model_validate(payload)


async def _judged_instruction(monkeypatch, body: ApexVerifyRequest) -> str:
    seen: dict = {}

    async def recording_judge(**kwargs):
        seen.update(kwargs)
        return 1.0, {}, {"ok": True}

    monkeypatch.setattr("resources_servers.apex_agents.app.grade_apex_output", recording_judge)
    result = await _server().verify(body)
    assert result.reward == 1.0
    assert "held out" not in seen["instruction"]
    return seen["instruction"]


@pytest.mark.asyncio
async def test_the_judge_receives_user_text_blocks_as_the_task_instruction(monkeypatch) -> None:
    blocks = [{"role": "user", "content": [{"type": "input_text", "text": "Use EUR, not USD"}]}]
    as_string = [{"role": "user", "content": "Use EUR, not USD"}]

    assert await _judged_instruction(monkeypatch, _blocks_body(blocks)) == "Use EUR, not USD"
    assert await _judged_instruction(monkeypatch, _blocks_body(as_string)) == "Use EUR, not USD"
    assert await _judged_instruction(monkeypatch, _blocks_body("Use EUR, not USD")) == "Use EUR, not USD"


@pytest.mark.asyncio
async def test_the_task_instruction_keeps_user_message_order_and_only_user_text(monkeypatch) -> None:
    image = {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "auto"}
    items = [
        {"role": "system", "content": "You are a careful analyst"},
        {"role": "user", "content": [{"type": "input_text", "text": "First: build the model"}, image]},
        {"role": "assistant", "content": "Understood"},
        {"role": "user", "content": "Second: report in EUR"},
    ]

    instruction = await _judged_instruction(monkeypatch, _blocks_body(items))

    assert instruction == "First: build the model\n\nSecond: report in EUR"


@pytest.mark.asyncio
async def test_document_preprocessing_does_not_block_the_event_loop(monkeypatch, tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    scratch: list[Path] = []

    def slow_changes(initial_root: Path, *_args, **_kwargs):
        scratch.append(initial_root.parent)
        started.set()
        # A document conversion that takes a long time, on whatever thread verify gave it.
        assert release.wait(30)
        return []

    async def judge(**_kwargs):
        return 1.0, {"v1": {"score": 1.0}}, {"ok": True}

    monkeypatch.setattr("resources_servers.apex_agents.app.snapshot_changes", slow_changes)
    monkeypatch.setattr("resources_servers.apex_agents.app.grade_apex_output", judge)
    server = _server(world_cache_dir=str(tmp_path), artifact_output_dir=str(tmp_path / "saved"))

    verifying = asyncio.ensure_future(server.verify(_body()))
    await asyncio.to_thread(started.wait, 30)
    # While the conversion is stuck the loop still turns, and an unrelated request is served.
    ticks = 0
    for _ in range(50):
        await asyncio.sleep(0)
        ticks += 1
    await server._preflight_world_cache()
    rejected = await server.verify(_body("not-base64"))
    assert ticks == 50
    assert "encoding" in rejected.verifier_error
    assert not verifying.done()
    assert scratch[0].is_dir()

    release.set()
    result = await asyncio.wait_for(verifying, 30)

    assert result.reward == 1.0
    assert Path(result.final_snapshot_path).is_file()
    # The scratch directory is gone once verify returns, and not before the worker was done with it.
    assert not scratch[0].exists()


@pytest.mark.asyncio
async def test_a_cancelled_verify_keeps_its_scratch_until_the_worker_returns(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()
    scratch: list[Path] = []
    survived: list[bool] = []

    def slow_changes(initial_root: Path, *_args, **_kwargs):
        scratch.append(initial_root.parent)
        started.set()
        assert release.wait(30)
        survived.append((initial_root.parent / "final" / "filesystem" / "final.txt").is_file())
        return []

    monkeypatch.setattr("resources_servers.apex_agents.app.snapshot_changes", slow_changes)
    server = _server()

    verifying = asyncio.ensure_future(server.verify(_body()))
    await asyncio.to_thread(started.wait, 30)
    verifying.cancel()
    await asyncio.sleep(0.05)
    # A thread cannot be cancelled: the verify keeps its slot and its scratch directory until the worker is done.
    assert not verifying.done()
    assert scratch[0].is_dir()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(verifying, 30)

    assert survived == [True]
    assert not scratch[0].exists()


def _judge_reply(passed: bool) -> MagicMock:
    content = json.dumps({"rationale": "checked", "is_criteria_true": passed})
    reply = MagicMock(ok=True)
    reply.read = AsyncMock(
        return_value=json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
    )
    return reply


def _rubric_body(snapshot: str, criteria: int) -> ApexVerifyRequest:
    payload = _body().model_dump()
    payload["verifier_metadata"] = {
        "rubric": [{"verifier_id": f"v{index}", "criteria": f"Criterion {index}"} for index in range(criteria)]
    }
    payload["initial_artifact_snapshot_b64"] = _snapshot(None)
    payload["artifact_snapshot_b64"] = snapshot
    return ApexVerifyRequest.model_validate(payload)


@pytest.mark.asyncio
async def test_verify_extracts_each_changed_file_once_for_all_criteria(monkeypatch) -> None:
    from resources_servers.apex_agents import artifacts

    reads: list[str] = []
    read_text = artifacts._extract_text

    def counting(path: Path, **converter) -> str:
        reads.append(path.name)
        return read_text(path, **converter)

    monkeypatch.setattr("resources_servers.apex_agents.artifacts._extract_text", counting)
    server = _server()
    server.server_client.post = AsyncMock(side_effect=lambda **_kwargs: _judge_reply(True))

    result = await server.verify(_rubric_body(_snapshot("the deliverable text"), criteria=10))

    assert result.reward == 1.0
    assert server.server_client.post.call_count == 10
    assert reads == ["final.txt"]
    for call in server.server_client.post.call_args_list:
        assert "the deliverable text" in call.kwargs["json"]["messages"][1]["content"]


def _workbook_snapshot() -> str:
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.active["A1"] = 2
    workbook.active["A2"] = "=A1*21"
    saved = io.BytesIO()
    workbook.save(saved)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("filesystem/model.xlsx", saved.getvalue())
    return base64.b64encode(output.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_a_converter_that_cannot_run_masks_the_verify_instead_of_grading_text(
    monkeypatch, tmp_path: Path
) -> None:
    image = tmp_path / "archipelago.sif"
    image.write_bytes(b"sif")
    # The converter image is configured, but this host has no apptainer to enter it with.
    monkeypatch.setattr("resources_servers.apex_agents.file_extraction.shutil.which", lambda _executable: None)
    server = _server()
    server.config.document_converter_image = str(image)
    server.server_client.post = AsyncMock(side_effect=lambda **_kwargs: _judge_reply(True))

    result = await server.verify(_rubric_body(_workbook_snapshot(), criteria=1))

    assert result.reward == 0.0
    assert result.mask_sample is True
    assert result.model_dump()["failure_kind"] == "verifier_error"
    assert "apptainer is not on PATH" in result.verifier_error
    # The workbook's formulas were never recalculated, so the judge was not asked to grade their blanks.
    assert server.server_client.post.call_count == 0

    # With no converter configured the same workbook is graded on its text, with the judge told what is missing.
    server.config.document_converter_image = None
    graded = await server.verify(_rubric_body(_workbook_snapshot(), criteria=1))
    assert graded.reward == 1.0
    assert (
        "formula cells could not be recalculated"
        in (server.server_client.post.call_args.kwargs["json"]["messages"][1]["content"])
    )


def _docx_snapshot() -> str:
    from docx import Document

    document = Document()
    document.add_paragraph("Quarterly memo")
    saved = io.BytesIO()
    document.save(saved)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("filesystem/memo.docx", saved.getvalue())
    return base64.b64encode(output.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_the_server_refuses_to_start_with_a_converter_it_cannot_run(monkeypatch, tmp_path: Path) -> None:
    which = "resources_servers.apex_agents.file_extraction.shutil.which"
    server = _server()
    assert server._preflight_document_converter in server.setup_webserver().router.on_startup

    # Nothing configured: nothing to check, whatever is installed.
    monkeypatch.setattr(which, lambda _executable: None)
    await server._preflight_document_converter()

    # The shipped default image, on a host that does not have it.
    monkeypatch.setattr(which, lambda executable: "/usr/bin/apptainer" if executable == "apptainer" else None)
    server.config.document_converter_image = str(tmp_path / "images" / "archipelago.sif")
    with pytest.raises(RuntimeError) as missing_image:
        await server._preflight_document_converter()
    message = str(missing_image.value)
    assert "image not found" in message and "archipelago.sif" in message
    # One message, with the three ways out.
    for way_out in (
        "`document_converter_image`",
        "`allow_host_document_tools: true`",
        "`document_converter_image: null`",
    ):
        assert way_out in message

    # A registry image needs apptainer, not a local file.
    server.config.document_converter_image = "docker://registry.example/apex/archipelago:1.0"
    await server._preflight_document_converter()
    monkeypatch.setattr(which, lambda _executable: None)
    with pytest.raises(RuntimeError, match="apptainer is not on PATH"):
        await server._preflight_document_converter()

    # Host tools, allowed and installed, are a way out without apptainer.
    monkeypatch.setattr(which, lambda executable: f"/usr/bin/{executable}" if executable != "apptainer" else None)
    with pytest.raises(RuntimeError, match="apptainer is not on PATH"):
        await server._preflight_document_converter()
    server.config.allow_host_document_tools = True
    await server._preflight_document_converter()


@pytest.mark.asyncio
async def test_without_a_converter_documents_are_graded_on_text_and_never_masked(monkeypatch) -> None:
    # No converter configured, and nothing installed that could convert: the null setting needs neither.
    monkeypatch.setattr("resources_servers.apex_agents.file_extraction.shutil.which", lambda _executable: None)
    server = _server()
    assert server.config.document_converter_image is None
    server.server_client.post = AsyncMock(side_effect=lambda **_kwargs: _judge_reply(True))

    for snapshot in (_docx_snapshot(), _workbook_snapshot()):
        result = await server.verify(_rubric_body(snapshot, criteria=2))

        assert result.reward == 1.0
        assert result.mask_sample is not True
        assert result.verifier_error is None
        assert "document_conversion_timeouts" not in result.judge_response
    request = server.server_client.post.call_args_list[0].kwargs["json"]
    # Text only: the document's words in the prompt, and no image blocks.
    assert isinstance(request["messages"][1]["content"], str)
    assert "Quarterly memo" in request["messages"][1]["content"]


@pytest.mark.asyncio
async def test_a_conversion_timeout_is_recorded_in_the_grading_result(monkeypatch, tmp_path: Path) -> None:
    import subprocess

    image = tmp_path / "archipelago.sif"
    image.write_bytes(b"sif")
    monkeypatch.setattr(
        "resources_servers.apex_agents.file_extraction.shutil.which",
        lambda executable: "/usr/bin/apptainer" if executable == "apptainer" else None,
    )

    def hangs(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("resources_servers.apex_agents.file_extraction.subprocess.run", hangs)
    server = _server(artifact_output_dir=str(tmp_path / "saved"))
    server.config.document_converter_image = str(image)
    server.server_client.post = AsyncMock(side_effect=lambda **_kwargs: _judge_reply(True))

    result = await server.verify(_rubric_body(_docx_snapshot(), criteria=3))

    # Graded on the document's text, not masked, with the timeout on record: once, though three criteria used it.
    assert result.reward == 1.0
    assert result.mask_sample is not True
    timeouts = [{"tool": "libreoffice", "file": "memo.docx", "timeout_seconds": 120}]
    assert result.judge_response["document_conversion_timeouts"] == timeouts
    grading = json.loads((Path(result.artifact_output_dir) / "grading.json").read_text())
    assert grading["judge_response"]["document_conversion_timeouts"] == timeouts

    # The next verify starts its own record.
    monkeypatch.setattr(
        "resources_servers.apex_agents.file_extraction.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, b"", b"Error"),
    )
    again = await server.verify(_rubric_body(_docx_snapshot(), criteria=1))
    assert "document_conversion_timeouts" not in again.judge_response
