# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Request

from nemo_gym.judge import JudgeError
from resources_servers.job_bench import app, task_data


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_seed_failure_preserves_original_error_and_closes_sandbox(
    monkeypatch, tmp_path, cancelled, cleanup_fails
):
    (tmp_path / "task_folder").mkdir()
    (tmp_path / "task_folder" / "TASK_INSTRUCTIONS.txt").write_text("task")
    error = asyncio.CancelledError("cancelled seed") if cancelled else RuntimeError("seed failed")
    sandbox = SimpleNamespace(
        start=AsyncMock(side_effect=error),
        stop=AsyncMock(side_effect=RuntimeError("cleanup failed") if cleanup_fails else None),
    )
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *args: {})
    monkeypatch.setattr(app, "create_provider", lambda *args: None)
    monkeypatch.setattr(app, "AsyncSandbox", lambda provider: sandbox)
    server = app.JobBenchResourcesServer.model_construct(
        config=app.JobBenchConfig.model_construct(sandbox_provider="sandbox", sandbox_config={"image": "test"})
    )
    request = Request({"type": "http", "session": {app.SESSION_ID_KEY: "session"}})
    body = app.JobBenchRequest(task_id="task", task_dir=str(tmp_path), rubrics_file="unused")
    with pytest.raises(type(error)) as caught:
        await server.seed_session(request, body)
    assert caught.value is error
    sandbox.stop.assert_awaited_once()
    assert not server._sandboxes


@pytest.mark.parametrize(
    "unsafe_name,mode",
    [
        (None, "temporary"),
        ("../escape.txt", "temporary"),
        (None, "persisted"),
        (None, "restart"),
        (None, "download_error"),
        (None, "exec_error"),
        (None, "invalid_grade"),
    ],
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_verify_collects_regular_outputs_without_runtime_links(
    monkeypatch, tmp_path, caplog, unsafe_name, mode, cleanup_fails
):
    rubrics_file = tmp_path / "rubrics.json"
    rubrics_file.write_text('{"rubrics": []}')
    collection_error = TimeoutError("archive exec failed") if mode == "exec_error" else OSError("download interrupted")

    async def download(remote, local):
        with tarfile.open(local, "w:gz") as tar:
            artifact = tarfile.TarInfo(unsafe_name or "answer.txt")
            artifact.size = 6
            tar.addfile(artifact, io.BytesIO(b"answer"))
            for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                link = tarfile.TarInfo(f"venv/python-{kind.decode()}")
                link.type = kind
                link.linkname = "/usr/bin/python3"
                tar.addfile(link)
        if mode == "download_error":
            raise collection_error

    cleanup_error = RuntimeError("cleanup failed")
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(return_code=0), side_effect=collection_error if mode == "exec_error" else None
        ),
        download=AsyncMock(side_effect=download),
        stop=AsyncMock(side_effect=cleanup_error if cleanup_fails else None),
    )
    server = app.JobBenchResourcesServer.model_construct(
        config=app.JobBenchConfig.model_construct(
            judge_model="judge", artifact_root=tmp_path / "saved" if mode != "temporary" else None
        )
    )
    server._sandboxes = {"session": sandbox}

    judge_calls = 0
    judge_error = JudgeError("judge interrupted")

    def judge(self, output, rubrics):
        nonlocal judge_calls
        judge_calls += 1
        assert (output / "answer.txt").read_text() == "answer"
        assert not (output / "venv").exists()
        assert json.loads(rubrics.read_text()) == {"rubrics": []}
        (output.parent / "judge-receipt.json").write_text(
            json.dumps(
                [
                    {
                        "result": {"index": 0},
                        "debug": {"api_exit_code": 2 if mode == "restart" and judge_calls == 1 else 0},
                    }
                ]
            )
        )
        if mode == "restart" and judge_calls == 1:
            raise judge_error
        return {
            "normalized_score": "invalid grade" if mode == "invalid_grade" else 1,
            "total_score": 1,
            "max_score": 1,
            "passed_count": 1,
            "total_count": 1,
        }, []

    monkeypatch.setattr(app.JobBenchResourcesServer, "_judge", judge)
    request = Request({"type": "http", "session": {app.SESSION_ID_KEY: "session"}})
    body = app.JobBenchVerifyRequest(
        task_id="task",
        task_dir=str(tmp_path),
        rubrics_file=str(rubrics_file),
        _ng_task_index=3,
        _ng_rollout_index=2,
        agent_error="original native failure" if mode in ("exec_error", "download_error") else None,
        responses_create_params={"input": []},
        response={
            "id": "test",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        },
    )
    if mode in ("exec_error", "download_error"):
        body.response.status = "failed"
    if unsafe_name:
        with pytest.raises(tarfile.OutsideDestinationError):
            await server.verify(request, body)
    elif mode in ("download_error", "exec_error"):
        with pytest.raises(type(collection_error)) as caught:
            await server.verify(request, body)
        assert caught.value is collection_error
        saved = server.config.artifact_root / body.artifact_id
        assert json.loads((saved / "request.json").read_text()) == body.model_dump(mode="json")
        assert (saved / "rubrics.json").read_bytes() == rubrics_file.read_bytes()
        assert not (saved / "request.partial").exists()
        assert not (saved / "result.json").exists()
        assert not (saved / "output.tar.gz").exists()
        assert judge_calls == 0
        if mode == "exec_error":
            sandbox.download.assert_not_awaited()
    elif mode == "restart":
        with pytest.raises(JudgeError, match="judge interrupted") as caught:
            await server.verify(request, body)
        assert caught.value is judge_error
        saved = server.config.artifact_root / body.artifact_id
        snapshot = (saved / "output.tar.gz").read_bytes()
        persisted = json.loads((saved / "request.json").read_text())
        assert persisted["_ng_task_index"] == 3 and persisted["_ng_rollout_index"] == 2
        assert not (saved / "result.json").exists()
        first_receipts = {path.name: path.read_bytes() for path in saved.glob("judge-receipt-*.json")}
        assert len(first_receipts) == 1
        rubrics_file.unlink()  # Replay must use the frozen rubric, not the original task directory.
        restarted = app.JobBenchResourcesServer.model_construct(config=server.config)
        recovered = await restarted.verify(request, body)
        assert recovered.reward == 1
        assert (saved / "output.tar.gz").read_bytes() == snapshot
        assert await restarted.verify(request, body) == recovered
        assert len(list(saved.glob("judge-receipt-*.json"))) == 2
        assert all((saved / name).read_bytes() == content for name, content in first_receipts.items())
        assert judge_calls == 2
        sandbox.download.assert_awaited_once()
    elif mode == "invalid_grade":
        with pytest.raises(ValueError, match="invalid grade"):
            await server.verify(request, body)
        assert not (server.config.artifact_root / body.artifact_id / "result.json").exists()
    else:
        result = await server.verify(request, body)
        assert result.reward == 1
        if mode == "persisted":
            saved = server.config.artifact_root / body.artifact_id
            snapshot = (saved / "output.tar.gz").read_bytes()
            assert json.loads((saved / "result.json").read_text()) == result.model_dump(mode="json")
            restarted = app.JobBenchResourcesServer.model_construct(config=server.config)
            assert await restarted.verify(request, body) == result
            assert len(list(saved.glob("judge-receipt-*.json"))) == 1
            assert (saved / "output.tar.gz").read_bytes() == snapshot
            assert judge_calls == 1
            sandbox.download.assert_awaited_once()
    sandbox.stop.assert_awaited_once()
    if cleanup_fails:
        assert "Job-Bench task sandbox cleanup failed" in caplog.text
        assert caplog.records[-1].exc_info[1] is cleanup_error


@pytest.mark.parametrize(
    "response",
    ['```json\n{"rubric_passed": true}\n```', 'prefix {"rubric_passed": true} suffix'],
)
def test_parse_judge_json_fallbacks(response: str) -> None:
    parsed, _ = app.judge.parse_judge_json(response)

    assert parsed["rubric_passed"] is True


def test_parse_judge_json_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        app.judge.parse_judge_json("not json")


@pytest.mark.parametrize(("passed", "expected_score"), [(True, 3), (False, 0)])
def test_judge_rubric_scores_model_verdict(monkeypatch, passed: bool, expected_score: int) -> None:
    content = json.dumps(
        {
            "criteria_results": [{"passed": passed, "reasoning": "reason", "evidence": "evidence"}],
            "rubric_passed": passed,
            "overall_reasoning": "reason",
        }
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    completions = SimpleNamespace(create=lambda **_kwargs: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(app.judge, "get_openai_client", lambda *_args: client)

    result, debug = app.judge.judge_rubric(
        0,
        {"rubric": "required", "weight": 3, "criterion": ["first", "second"]},
        "answer",
        "judge-model",
        "https://judge.example",
        "test",
        max_retries=1,
    )

    assert result["result"]["score"] == expected_score
    assert result["result"]["criteria_results"][1]["passed"] is False
    assert debug["api_exit_code"] == 0


@pytest.mark.parametrize("terminal_transport_error", [False, True])
def test_judge_retry_classification_uses_terminal_attempt(monkeypatch, tmp_path, terminal_transport_error):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not JSON"))])
    outcomes = [
        response,
        TimeoutError("judge timeout"),
        TimeoutError("judge timeout") if terminal_transport_error else response,
    ]
    create = Mock(side_effect=outcomes)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(app.judge, "get_openai_client", lambda *_: client)
    monkeypatch.setattr(app.judge.time, "sleep", lambda _: None)
    rubric = {"rubric": "required", "weight": 1, "criterion": ["answer"]}
    result, debug = app.judge.judge_rubric(0, rubric, "answer", "judge", "https://judge.example", "test")
    assert create.call_count == 3
    assert debug["api_exit_code"] == (2 if terminal_transport_error else 1)
    assert debug["raw_response"] == "not JSON"
    assert result["result"]["score"] == 0
    output = tmp_path / "output"
    output.mkdir()
    (output / "answer.txt").write_text("answer")
    rubrics = tmp_path / "rubrics.json"
    rubrics.write_text(json.dumps({"rubrics": [rubric]}))
    monkeypatch.setattr(app.judge, "judge_rubric", lambda *_: (result, debug))
    server = app.JobBenchResourcesServer.model_construct(
        config=app.JobBenchConfig.model_construct(
            judge_model="judge", judge_base_url="https://judge.example", judge_api_key="test", max_judge_workers=1
        )
    )
    if terminal_transport_error:
        with pytest.raises(JudgeError, match="judge timeout"):
            server._judge(output, rubrics)
    else:
        assert server._judge(output, rubrics)[0]["normalized_score"] == 0


def test_judge_uses_official_weighted_score(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "answer.txt").write_text("answer", encoding="utf-8")
    rubrics_file = tmp_path / "RUBRICS.json"
    rubrics_file.write_text(
        json.dumps({"rubrics": [{"rubric": "first", "weight": 3}, {"rubric": "second", "weight": 1}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(app.judge, "extract_all_file_contents", lambda _: "answer")
    monkeypatch.setattr(app.judge, "collect_image_attachments", lambda _: [])

    def fake_judge(index, rubric, *_args):
        passed = index == 0
        return {
            "index": index,
            "weight": rubric["weight"],
            "result": {"passed": passed, "score": rubric["weight"] if passed else 0},
        }, {"api_exit_code": 0}

    monkeypatch.setattr(app.judge, "judge_rubric", fake_judge)
    server = app.JobBenchResourcesServer.model_construct(
        config=app.JobBenchConfig.model_construct(
            judge_model="grok-4.3",
            judge_base_url="https://api.x.ai/v1",
            judge_api_key="test",
            max_judge_workers=2,
        )
    )

    scorecard, results = server._judge(output_dir, rubrics_file)

    assert scorecard["normalized_score"] == 0.75
    assert scorecard["passed_count"] == 1
    assert len(results) == 2


def test_empty_output_fails_without_calling_judge(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    rubrics_file = tmp_path / "RUBRICS.json"
    rubrics_file.write_text(json.dumps({"rubrics": [{"rubric": "required", "weight": 5}]}), encoding="utf-8")
    monkeypatch.setattr(app.judge, "judge_rubric", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    server = app.JobBenchResourcesServer.model_construct(
        config=app.JobBenchConfig.model_construct(max_judge_workers=1)
    )

    scorecard, _ = server._judge(output_dir, rubrics_file)

    assert scorecard["normalized_score"] == 0


@pytest.mark.parametrize("text", ["0123456789" * 30000, "漢字🙂" * 40000], ids=["numeric", "unicode"])
def test_extracted_text_budget_is_fair_and_utf8_safe(tmp_path, text):
    (tmp_path / "short.txt").write_text("Complete short report.")
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text(text)
    result = app.judge.extract_all_file_contents(tmp_path)
    assert len(result.encode("utf-8")) <= app.judge.MAX_EXTRACTED_BYTES
    assert "Complete short report." in result
    assert result.count("Middle omitted by job-bounded-utf8-v1") == 2
    assert "\ufffd" not in result
    assert result.index("a.txt") < result.index("b.txt") < result.index("short.txt")
    first = result.split("=== FILE: a.txt ===\n")[1].split("=== FILE: b.txt ===\n")[0]
    second = result.split("=== FILE: b.txt ===\n")[1].split("=== FILE: short.txt ===\n")[0]
    assert abs(len(first.encode("utf-8")) - len(second.encode("utf-8"))) < 16
    assert result == app.judge.extract_all_file_contents(tmp_path)


def test_sqlite_text_is_bounded_and_relative_names_remain_distinct(monkeypatch, tmp_path):
    for directory in ("first", "second"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "data.db").touch()
    monkeypatch.setattr(app.judge, "convert_file_to_text", lambda _: "HEAD" + "x" * 200000 + "TAIL")
    result = app.judge.extract_all_file_contents(tmp_path)
    assert len(result.encode("utf-8")) <= app.judge.MAX_EXTRACTED_BYTES
    assert "first/data.db" in result and "second/data.db" in result
    assert result.count("HEAD") == result.count("TAIL") == 2


def test_extracted_budget_keeps_small_files_and_rejects_filename_overflow(monkeypatch, tmp_path):
    (tmp_path / "report.txt").write_text("entire report")
    assert app.judge.extract_all_file_contents(tmp_path) == "=== FILE: report.txt ===\nentire report\n"
    monkeypatch.setattr(app.judge, "MAX_EXTRACTED_BYTES", 20)
    with pytest.raises(ValueError, match="Too many output filenames"):
        app.judge.extract_all_file_contents(tmp_path)


async def test_grade_view_excludes_runtime_files_without_changing_archive(monkeypatch, tmp_path):
    saved = tmp_path / "artifacts" / ("a" * 64)
    saved.mkdir(parents=True)
    files = {"reports/answer.txt": b"answer", "venv": b"a legitimate file named venv"}
    for directory in (".venv", "nested/venv", "node_modules", ".git", ".cache", "__pycache__"):
        files[f"{directory}/ignored.txt"] = b"runtime"
    with tarfile.open(saved / "output.tar.gz", "w:gz") as tar:
        for name, content in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(content)
            tar.addfile(item, io.BytesIO(content))
    archive = (saved / "output.tar.gz").read_bytes()
    (saved / "rubrics.json").write_text('{"rubrics": []}')
    body = app.JobBenchVerifyRequest(
        task_id="task",
        task_dir="/missing",
        rubrics_file="/missing",
        artifact_id=saved.name,
        responses_create_params={"input": []},
        response={
            "id": "test",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        },
    )
    (saved / "request.json").write_text(body.model_dump_json())

    def judge(self, output, rubrics):
        assert sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()) == [
            "reports/answer.txt",
            "venv",
        ]
        return {"normalized_score": 1, "total_score": 1, "max_score": 1, "passed_count": 1, "total_count": 1}, []

    monkeypatch.setattr(app.JobBenchResourcesServer, "_judge", judge)
    server = app.JobBenchResourcesServer.model_construct(
        config=app.JobBenchConfig.model_construct(artifact_root=saved.parent, judge_model="fixed")
    )
    result = await server.verify(Request({"type": "http", "session": {app.SESSION_ID_KEY: "replay"}}), body)
    assert result.grading_protocol == "job-bounded-utf8-v1"
    assert (saved / "output.tar.gz").read_bytes() == archive


@pytest.mark.parametrize("receipt_write_fails", [False, True])
def test_judge_failure_is_not_scored(monkeypatch, tmp_path: Path, caplog, receipt_write_fails) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "answer.txt").write_text("answer", encoding="utf-8")
    rubrics_file = tmp_path / "RUBRICS.json"
    rubrics_file.write_text(json.dumps({"rubrics": [{"rubric": "required", "weight": 1}] * 2}), encoding="utf-8")
    monkeypatch.setattr(app.judge, "extract_all_file_contents", lambda _: "answer")
    monkeypatch.setattr(app.judge, "collect_image_attachments", lambda _: [])
    monkeypatch.setattr(
        app.judge,
        "judge_rubric",
        lambda index, *_args: (
            {"index": index, "result": {"score": 1 if index == 0 else 0}},
            {
                "api_exit_code": 0 if index == 0 else 2,
                "error": "judge unavailable" if index else "",
                "api_base": "https://user:secret@example.test",  # pragma: allowlist secret
                "api_key": "secret",  # pragma: allowlist secret
                "raw_response": "verdict",
            },
        ),
    )
    server = app.JobBenchResourcesServer.model_construct(
        config=app.JobBenchConfig.model_construct(
            judge_model="grok-4.3",
            judge_base_url="https://api.x.ai/v1",
            judge_api_key="test",
            max_judge_workers=1,
        )
    )

    if receipt_write_fails:
        (tmp_path / "judge-receipt.json").mkdir()
    with pytest.raises(JudgeError, match="judge unavailable"):
        server._judge(output_dir, rubrics_file)
    if receipt_write_fails:
        assert "judge receipt retention failed (IsADirectoryError)" in caplog.text
        assert caplog.records[-1].exc_info is None
        return
    retained = (tmp_path / "judge-receipt.json").read_text()
    assert "secret" not in retained and "api_base" not in retained and "api_key" not in retained
    receipts = json.loads(retained)
    assert [row["debug"]["api_exit_code"] for row in receipts] == [0, 2]
    assert receipts[0]["result"]["result"]["score"] == 1
    assert all(row["debug"]["raw_response"] == "verdict" for row in receipts)


def test_unparseable_judge_response_is_scored_as_failure(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "answer.txt").write_text("answer", encoding="utf-8")
    rubrics_file = tmp_path / "RUBRICS.json"
    rubric = {"rubric": "required", "weight": 1}
    rubrics_file.write_text(json.dumps({"rubrics": [rubric]}), encoding="utf-8")
    monkeypatch.setattr(app.judge, "extract_all_file_contents", lambda _: "answer")
    monkeypatch.setattr(app.judge, "collect_image_attachments", lambda _: [])
    monkeypatch.setattr(
        app.judge,
        "judge_rubric",
        lambda *_args: (
            app.judge.build_failed_rubric_result(0, rubric, "unparseable response"),
            {"api_exit_code": 1, "error": "unparseable response"},
        ),
    )
    server = app.JobBenchResourcesServer.model_construct(
        config=app.JobBenchConfig.model_construct(
            judge_model="grok-4.3",
            judge_base_url="https://api.x.ai/v1",
            judge_api_key="test",
            max_judge_workers=1,
        )
    )

    scorecard, _ = server._judge(output_dir, rubrics_file)

    assert scorecard["normalized_score"] == 0


def test_relative_task_paths_resolve_against_pinned_snapshot(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    task_data.snapshot_root.cache_clear()
    try:
        assert task_data.resolve_task_path("dataset/jobs/task1") == tmp_path / "dataset/jobs/task1"
        assert task_data.resolve_task_path(str(tmp_path / "local")) == tmp_path / "local"
    finally:
        task_data.snapshot_root.cache_clear()
    assert calls == [
        (
            "JobBench/job-bench",
            {"repo_type": "dataset", "revision": task_data.DATASET_REVISION, "allow_patterns": "dataset/**"},
        )
    ]
