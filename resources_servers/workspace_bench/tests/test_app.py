# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException, Request

from nemo_gym.judge import JudgeError
from resources_servers.workspace_bench import app, dataset


@pytest.mark.parametrize(
    "mode",
    [
        "temporary",
        "persisted",
        "restart",
        "download_error",
        "exec_error",
        "invalid_grade",
        "parse_failure",
        "command_loss",
        "malformed_judge_output",
        "null_judge_output",
        "list_judge_output",
        "invalid_judge_metadata",
    ],
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_verify_collects_files_without_runtime_links(monkeypatch, tmp_path, caplog, mode, cleanup_fails):
    metadata = {"data_manifest": [{"stored_relpath": "source/original.txt", "filename": "task.txt"}]}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "source").mkdir()
    original = tmp_path / "source/original.txt"
    original.write_text("trusted original")
    (tmp_path / "output").mkdir()
    (tmp_path / "output/ground-truth.txt").write_text("must not reach judge")
    collection_error = TimeoutError("archive exec failed") if mode == "exec_error" else OSError("download interrupted")

    async def download(remote, local):
        with tarfile.open(local, "w:gz") as tar:
            for name in ("input/task.txt", "output/answer.txt"):
                item = tarfile.TarInfo(name)
                item.size = 6
                tar.addfile(item, io.BytesIO(b"answer"))
            link = tarfile.TarInfo("output/venv/python3")
            link.type = tarfile.SYMTYPE
            link.linkname = "/usr/bin/python3"
            tar.addfile(link)
        if mode == "download_error":
            raise collection_error

    cleanup_error = RuntimeError("cleanup failed")
    sandbox = SimpleNamespace(
        start=AsyncMock(),
        upload=AsyncMock(),
        serialize=AsyncMock(return_value={"sandbox_id": "test-sandbox"}),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0)),
        download=AsyncMock(side_effect=download),
        stop=AsyncMock(side_effect=cleanup_error if cleanup_fails else None),
    )
    server = app.WorkspaceBenchResourcesServer.model_construct(
        config=app.WorkspaceBenchConfig.model_construct(
            judge_model="judge",
            artifact_root=tmp_path / "saved" if mode != "temporary" else None,
            sandbox_provider="sandbox",
            sandbox_config={"image": "test-runtime"},
        )
    )
    monkeypatch.setattr(app, "resolve_provider_config", lambda *_: {})
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "create_provider", lambda _: "provider")
    monkeypatch.setattr(app, "AsyncSandbox", lambda _: sandbox)
    request = Request({"type": "http", "session": {app.SESSION_ID_KEY: "session"}})
    await server.seed_session(request, app.WorkspaceBenchRequest(task_id="task", task_dir=str(tmp_path)))
    if mode == "exec_error":
        sandbox.exec.side_effect = collection_error
    references = Path(server._reference_dirs["session"].name)
    input_snapshot = (references / "input.tar.gz").read_bytes()
    with tarfile.open(references / "input.tar.gz", "r:gz") as archive:
        assert archive.getnames() == ["task.txt"]
    original.write_text("changed after policy started")
    (tmp_path / "metadata.json").write_text("{}")

    judge_calls = 0
    invalid_outputs = {
        "malformed_judge_output": '{"partial": "receipt-secret',
        "null_judge_output": "null",
        "list_judge_output": "[]",
        "invalid_judge_metadata": '{"judge": "receipt-secret"}',
    }
    judge_error = JudgeError("judge interrupted: original-secret" if mode in invalid_outputs else "judge interrupted")

    async def judge(self, directory):
        nonlocal judge_calls
        judge_calls += 1
        assert (directory / "output/answer.txt").read_text() == "answer"
        assert (directory / "data/task.txt").read_text() == "trusted original"
        assert not (directory / "input").exists()
        assert not (directory / "workspace.tar.gz").exists()
        assert not (directory / "output/ground-truth.txt").exists()
        assert not (directory / "output/venv").exists()
        assert json.loads((directory / "metadata.json").read_text()) == metadata
        if mode == "command_loss":
            raise judge_error
        if mode in invalid_outputs:
            (directory / "rubrics_judge--gym-judge.json").write_text(invalid_outputs[mode])
            raise judge_error
        rubrics = [] if mode == "invalid_grade" else [{"passed": mode != "parse_failure"}]
        error = (
            "Judge output parse failed"
            if mode == "parse_failure"
            else "judge interrupted"
            if mode == "restart" and judge_calls == 1
            else None
        )
        (directory / "rubrics_judge--gym-judge.json").write_text(
            json.dumps(
                {
                    "taskId": "task",
                    "rubrics": rubrics,
                    "summary": {"total": len(rubrics)},
                    "apiKey": "top-level-secret",
                    "config": {"apiKey": "config-secret"},
                    "judge": {
                        "tries": 6 if mode == "parse_failure" else 1,
                        "error": error,
                        "rawResponseHead": "received verdict",
                        "usage": {"input_tokens": 10},
                        "baseUrl": "https://user:password@example.test/v1?key=secret",  # pragma: allowlist secret
                        "apiKey": "judge-secret",  # pragma: allowlist secret
                    },
                }
            )
        )
        (directory / "judge.yaml").write_text("apiKey: config-secret")
        if mode == "restart" and judge_calls == 1:
            raise judge_error
        return rubrics, {}

    monkeypatch.setattr(app.WorkspaceBenchResourcesServer, "_judge", judge)
    body = app.WorkspaceBenchVerifyRequest(
        task_id="task",
        task_dir=str(tmp_path),
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
    if mode in ("download_error", "exec_error"):
        body.response.status = "failed"
        with pytest.raises(type(collection_error)) as caught:
            await server.verify(request, body)
        assert caught.value is collection_error
        saved = server.config.artifact_root / body.artifact_id
        assert json.loads((saved / "request.json").read_text()) == body.model_dump(mode="json")
        assert json.loads((saved / "metadata.json").read_text()) == metadata
        assert (saved / "input.tar.gz").read_bytes() == input_snapshot
        assert not (saved / "request.partial").exists()
        assert not (saved / "result.json").exists()
        assert not (saved / "workspace.tar.gz").exists()
        assert judge_calls == 0
        if mode == "exec_error":
            sandbox.download.assert_not_awaited()
    elif mode == "command_loss" or mode in invalid_outputs:
        with pytest.raises(JudgeError, match="judge interrupted") as caught:
            await server.verify(request, body)
        assert caught.value is judge_error
        saved = server.config.artifact_root / body.artifact_id
        assert (saved / "request.json").exists()
        assert not list(saved.glob("judge-receipt-*.json"))
        assert not (saved / "result.json").exists()
        if mode in invalid_outputs:
            assert "judge receipt retention failed" in caplog.text
            retention_logs = [record for record in caplog.records if "receipt retention failed" in record.message]
            assert all(record.exc_info is None and "secret" not in record.message for record in retention_logs)
    elif mode == "restart":
        with pytest.raises(JudgeError, match="judge interrupted") as caught:
            await server.verify(request, body)
        assert caught.value is judge_error
        saved = server.config.artifact_root / body.artifact_id
        snapshot = (saved / "workspace.tar.gz").read_bytes()
        persisted = json.loads((saved / "request.json").read_text())
        assert persisted["_ng_task_index"] == 3 and persisted["_ng_rollout_index"] == 2
        assert not (saved / "result.json").exists()
        first_receipts = {path.name: path.read_bytes() for path in saved.glob("judge-receipt-*.json")}
        assert len(first_receipts) == 1
        assert json.loads(next(iter(first_receipts.values())))["judge"]["error"] == "judge interrupted"
        (tmp_path / "metadata.json").unlink()
        original.unlink()
        restarted = app.WorkspaceBenchResourcesServer.model_construct(config=server.config)
        recovered = await restarted.verify(request, body)
        assert recovered.reward == 1
        assert recovered.grading_protocol == "workspace-frozen-inputs-v1"
        assert (saved / "workspace.tar.gz").read_bytes() == snapshot
        assert (saved / "input.tar.gz").read_bytes() == input_snapshot
        with tarfile.open(saved / "workspace.tar.gz", "r:gz") as archive:
            assert archive.extractfile("input/task.txt").read() == b"answer"
        assert await restarted.verify(request, body) == recovered
        assert len(list(saved.glob("judge-receipt-*.json"))) == 2
        assert all((saved / name).read_bytes() == content for name, content in first_receipts.items())
        assert judge_calls == 2
        sandbox.download.assert_awaited_once()
    elif mode == "invalid_grade":
        with pytest.raises(ValueError, match="judge returned no rubrics"):
            await server.verify(request, body)
        assert not (server.config.artifact_root / body.artifact_id / "result.json").exists()
    else:
        result = await server.verify(request, body)
        assert result.reward == (0 if mode == "parse_failure" else 1)
        if mode in ("persisted", "parse_failure"):
            saved = server.config.artifact_root / body.artifact_id
            snapshot = (saved / "workspace.tar.gz").read_bytes()
            assert json.loads((saved / "result.json").read_text()) == result.model_dump(mode="json")
            restarted = app.WorkspaceBenchResourcesServer.model_construct(config=server.config)
            assert await restarted.verify(request, body) == result
            assert len(list(saved.glob("judge-receipt-*.json"))) == 1
            assert (saved / "workspace.tar.gz").read_bytes() == snapshot
            assert judge_calls == 1
            sandbox.download.assert_awaited_once()
            (saved / "input.tar.gz").unlink()
            with pytest.raises(ValueError, match="lacks frozen original inputs"):
                await restarted.verify(request, body)
            assert judge_calls == 1
            assert (saved / "workspace.tar.gz").read_bytes() == snapshot
    if mode not in ("temporary", "download_error", "exec_error", "command_loss") and mode not in invalid_outputs:
        receipts = list((server.config.artifact_root / body.artifact_id).glob("judge-receipt-*.json"))
        assert len(receipts) == (2 if mode == "restart" else 1)
        for path in receipts:
            text = path.read_text()
            assert "secret" not in text and "password" not in text and "baseUrl" not in text
            retained = json.loads(text)["judge"]
            assert retained["tries"] == (6 if mode == "parse_failure" else 1)
            assert retained["rawResponseHead"] == "received verdict"
            assert retained["usage"] == {"input_tokens": 10}
        assert not (receipts[0].parent / "judge.yaml").exists()
    sandbox.stop.assert_awaited_once()
    sandbox.start.assert_awaited_once()
    assert not references.exists()
    assert not server._reference_dirs
    if cleanup_fails:
        assert "Workspace-Bench task sandbox cleanup failed" in caplog.text
        assert caplog.records[-1].exc_info[1] is cleanup_error


@pytest.mark.parametrize("failure", [None, "error", "cancellation"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_seed_installs_native_harness_skills(monkeypatch, tmp_path, caplog, failure, cleanup_fails):
    (tmp_path / "metadata.json").write_text(json.dumps({"data_manifest": []}))
    primary_error = (
        app.asyncio.CancelledError("seed cancelled")
        if failure == "cancellation"
        else RuntimeError("seed failed")
        if failure
        else None
    )
    cleanup_error = RuntimeError("cleanup failed")
    sandbox = SimpleNamespace(
        start=AsyncMock(),
        upload=AsyncMock(side_effect=primary_error),
        stop=AsyncMock(side_effect=cleanup_error if cleanup_fails else None),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0)),
        serialize=AsyncMock(return_value={"sandbox_id": "test-sandbox"}),
    )
    monkeypatch.setattr(app, "resolve_provider_config", lambda *_: {})
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "create_provider", lambda _: "provider")
    monkeypatch.setattr(app, "AsyncSandbox", lambda _: sandbox)
    server = app.WorkspaceBenchResourcesServer.model_construct(
        config=app.WorkspaceBenchConfig.model_construct(
            sandbox_provider="sandbox", sandbox_config={"image": "runtime"}
        )
    )
    request = Request({"type": "http", "session": {app.SESSION_ID_KEY: "session"}})
    if primary_error is not None:
        with pytest.raises(type(primary_error)) as caught:
            await server.seed_session(request, app.WorkspaceBenchRequest(task_id="task", task_dir=str(tmp_path)))
        assert caught.value is primary_error
        assert not sandbox.upload.await_args.args[0].parent.exists()
        assert not server._reference_dirs
        sandbox.stop.assert_awaited_once()
        if cleanup_fails:
            assert "Workspace-Bench seed sandbox cleanup failed" in caplog.text
            assert caplog.records[-1].exc_info[1] is cleanup_error
        return
    started = app.asyncio.Event()
    proceed = app.asyncio.Event()

    async def start(_spec):
        started.set()
        await proceed.wait()

    sandbox.start.side_effect = start
    pending = app.asyncio.create_task(
        server.seed_session(request, app.WorkspaceBenchRequest(task_id="task", task_dir=str(tmp_path)))
    )
    await started.wait()
    with pytest.raises(HTTPException, match="session is already active"):
        await server.seed_session(request, app.WorkspaceBenchRequest(task_id="task", task_dir=str(tmp_path)))
    proceed.set()
    result = await pending
    assert result.sandbox_handle == "test-sandbox"
    command = sandbox.exec.await_args.args[0]
    assert "office-skills /workspace/.opencode/skills" in command
    assert "agent-homes/claude/.claude/skills /workspace/.claude/skills" in command
    assert sandbox.start.await_args.args[0].workdir == "/workspace"
    assert server._sandboxes["session"] is sandbox
    with pytest.raises(HTTPException, match="session is already active"):
        await server.seed_session(request, app.WorkspaceBenchRequest(task_id="task", task_dir=str(tmp_path)))
    sandbox.start.assert_awaited_once()


async def test_shutdown_cleans_frozen_reference_directories(monkeypatch):
    server = app.WorkspaceBenchResourcesServer.model_construct(config=app.WorkspaceBenchConfig.model_construct())
    directory = tempfile.TemporaryDirectory()
    server._reference_dirs["session"] = directory
    sandbox = SimpleNamespace(stop=AsyncMock())
    server._sandboxes["session"] = sandbox
    monkeypatch.setattr(app.SimpleResourcesServer, "setup_webserver", lambda _: FastAPI())
    web = server.setup_webserver()
    async with web.router.lifespan_context(web):
        assert Path(directory.name).exists()
    assert not Path(directory.name).exists()
    assert not server._reference_dirs
    sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("return_code", [0, 1])
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("judge_error", [None, "HTTP 503: judge unavailable", "Judge output parse failed"])
async def test_judge_uses_isolated_upstream_evaluator(
    monkeypatch, tmp_path, caplog, return_code, cleanup_fails, judge_error
):
    (tmp_path / "metadata.json").write_text(json.dumps({"__metadata_path": str(tmp_path / "metadata.json")}))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "reference.txt").write_text("original reference")
    server = app.WorkspaceBenchResourcesServer.model_construct(
        config=app.WorkspaceBenchConfig.model_construct(
            judge_base_url="https://judge.example",
            judge_api_key="test",
            judge_model="judge-model",
            sandbox_provider="sandbox",
            sandbox_config={"image": "test-runtime"},
        )
    )

    async def download(remote, local):
        assert remote == f"/judge/task/{local.name}"
        data = (
            {"rubrics": [{"index": 0, "passed": not judge_error}], "judge": {"error": judge_error}}
            if "rubrics" in remote
            else {"nodes": [], "edges": []}
        )
        local.write_text(json.dumps(data))

    async def upload(local, remote):
        assert remote == "/tmp/judge-task.tar.gz"
        with tarfile.open(local, "r:gz") as archive:
            metadata = json.load(archive.extractfile("task/metadata.json"))
            assert metadata["__metadata_path"] == "/judge/task/metadata.json"
            assert archive.extractfile("task/data/reference.txt").read() == b"original reference"

    cleanup_error = RuntimeError("cleanup failed")
    sandbox = SimpleNamespace(
        start=AsyncMock(),
        upload=AsyncMock(side_effect=upload),
        stop=AsyncMock(side_effect=cleanup_error if cleanup_fails else None),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=return_code, stderr="judge error", stdout="")),
        download=AsyncMock(side_effect=download),
    )
    monkeypatch.setattr(app, "resolve_provider_config", lambda *_: {})
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "create_provider", lambda _: "provider")
    monkeypatch.setattr(app, "AsyncSandbox", lambda _: sandbox)

    if return_code:
        with pytest.raises(RuntimeError, match="judge error"):
            await server._judge(tmp_path)
        sandbox.download.assert_not_awaited()
    elif judge_error and judge_error != "Judge output parse failed":
        with pytest.raises(JudgeError, match=judge_error):
            await server._judge(tmp_path)
        assert sandbox.download.await_count == 2
    else:
        rubrics, graph = await server._judge(tmp_path)
        assert rubrics == [{"index": 0, "passed": not judge_error}]
        assert graph == {"nodes": [], "edges": []}
    assert sandbox.start.await_args.args[0].image == "test-runtime"
    assert "agent_as_a_judge.py --task-dir /judge/task" in sandbox.exec.await_args.args[0]
    assert sandbox.exec.await_args.kwargs["timeout_s"] == 600
    sandbox.upload.assert_awaited_once()
    sandbox.stop.assert_awaited_once()
    if cleanup_fails:
        assert "Workspace-Bench judge sandbox cleanup failed" in caplog.text
        assert caplog.records[-1].exc_info[1] is cleanup_error


def test_relative_task_paths_resolve_against_pinned_snapshot(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    dataset.snapshot_root.cache_clear()
    try:
        assert dataset.resolve_task_path("task_lite_clean_en/3") == tmp_path / "task_lite_clean_en/3"
        assert dataset.resolve_task_path(str(tmp_path / "local")) == tmp_path / "local"
    finally:
        dataset.snapshot_root.cache_clear()
    assert calls == [
        (
            "Workspace-Bench/Workspace-Bench-Lite",
            {"repo_type": "dataset", "revision": dataset.DATASET_REVISION, "allow_patterns": "task_lite_clean_en/**"},
        )
    ]
