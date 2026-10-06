# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from benchmarks.gdpval.prepare_nooa import prepare_native
from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput
from resources_servers.gdpval import sandbox_app
from resources_servers.gdpval.app import GDPValResourcesServer, GDPValVerifyRequest, GDPValVerifyResponse
from resources_servers.gdpval.sandbox_app import GDPSandboxConfig, GDPSandboxResourcesServer
from resources_servers.gdpval.sandbox_tasks import INPUT_DIR, OUTPUT_DIR, GDPFileTask, prepare_row


def row(**extra):
    return {
        "task_id": "task-1",
        "prompt": "Write a financial summary.",
        "responses_create_params": {"input": []},
        "reference_files": [],
        "reference_file_urls": [],
        "rubric_pretty": "PRIVATE RUBRIC",
        **extra,
    }


def seed(**extra):
    return ResourcesSeedSessionRequest(
        resources_session_id="resources-1",
        episode_id=EpisodeId(rollout_id="rollout-1"),
        task_id=TaskId(taskset="gdp", task_id="task-1"),
        task_data=row(),
        **extra,
    )


def response():
    return NeMoGymResponse(
        id="response-1",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=False,
    )


class Sandbox:
    def __init__(self):
        self.files = {f"{OUTPUT_DIR}/report.csv": b"name,value\na,3\n"}
        self.start = AsyncMock()
        self.stop = AsyncMock()
        self.serialize = AsyncMock(return_value={"sandbox_id": "task-box"})
        self.exec = AsyncMock(side_effect=self.execute)
        self.upload = AsyncMock(side_effect=self.upload_file)
        self.download = AsyncMock(side_effect=self.download_file)

    async def execute(self, command, **kwargs):
        files = [
            {
                "name": key.removeprefix(f"{OUTPUT_DIR}/"),
                "size": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
            for key, value in self.files.items()
            if key.startswith(f"{OUTPUT_DIR}/")
        ]
        return SimpleNamespace(return_code=0, stdout=json.dumps(files), stderr="")

    async def upload_file(self, local, remote):
        self.files[remote] = Path(local).read_bytes()

    async def download_file(self, remote, local):
        Path(local).write_bytes(self.files[remote])


@pytest.fixture
def server(tmp_path, monkeypatch):
    box = Sandbox()
    monkeypatch.setattr(sandbox_app, "AsyncSandbox", lambda provider: box)
    monkeypatch.setattr(sandbox_app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(sandbox_app, "resolve_provider_config", lambda *args: {"docker": {}})
    config = GDPSandboxConfig(
        host="127.0.0.1",
        port=8000,
        name="resources",
        entrypoint="sandbox_app.py",
        image="test-only",
        deliverables_root=tmp_path,
        preconvert_office_to_pdf=False,
        judge_model_server={"type": "responses_api_models", "name": "judge"},
    )
    instance = GDPSandboxResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return instance, box, SimpleNamespace(session={})


def test_prepare_native_preserves_private_metadata_without_exposing_it():
    source = row(reference_files='["reference_files/a.xlsx"]', reference_file_urls='["https://example.com/a.xlsx"]')
    prepared = prepare_row(source)
    MaterializedTask[SingleAgentTurnTaskInput].model_validate(prepared)
    text = prepared["task_input"]["responses_create_params"]["input"][0]["content"]
    assert "PRIVATE RUBRIC" not in text
    assert "https://example.com" not in text
    assert f"{INPUT_DIR}/reference_files/a.xlsx" in text
    assert OUTPUT_DIR in text
    assert "`finish` tool" not in text
    assert prepared["task_input"]["task_data"]["rubric_pretty"] == "PRIVATE RUBRIC"
    assert prepared["task_input"]["task_data"]["reference_files"] == ["reference_files/a.xlsx"]
    assert source["responses_create_params"]["input"] == []
    params = prepared["task_input"]["responses_create_params"]
    assert (params["max_output_tokens"], params["temperature"], params["top_p"]) == (32768, 1.0, 1.0)


def test_prepare_file_keeps_all_tasks_and_checks_identity(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "native.jsonl"
    source.write_text("\n".join(json.dumps(row(task_id=f"task-{i}")) for i in range(3)))
    original = source.read_bytes()
    assert prepare_native(source=source, output=output) == output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [item["task_id"]["task_id"] for item in rows] == ["task-0", "task-1", "task-2"]
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="overwrite"):
        prepare_native(source=source, output=source)
    source.write_text(json.dumps(row()) + "\n" + json.dumps(row()))
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_native(source=source, output=output)


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_prepare_file_keeps_unicode_separators_inside_task_text(tmp_path, separator):
    prompt = f"Create{separator}a document"
    source = tmp_path / "canonical.jsonl"
    source.write_text(json.dumps(row(prompt=prompt), ensure_ascii=False) + "\n")
    output = prepare_native(source=source, output=tmp_path / "native.jsonl")
    with output.open("rb") as stream:
        prepared = [json.loads(line) for line in stream]
    assert len(prepared) == 1
    assert prepared[0]["task_input"]["task_data"]["prompt"] == prompt
    assert prompt in prepared[0]["task_input"]["responses_create_params"]["input"][0]["content"]


@pytest.mark.parametrize("name", ["../secret", "/etc/passwd", "a/../../x", "a\\b", "a//b", ".", "a/./b", "a\x00b"])
def test_unsafe_reference_paths_rejected(name):
    with pytest.raises(ValueError):
        GDPFileTask.model_validate(row(reference_files=[name], reference_file_urls=["https://example.com/file"]))


@pytest.mark.parametrize(
    "fields",
    [
        {"reference_files": ["x"], "reference_file_urls": []},
        {"reference_files": ["x", "reference_files/x"], "reference_file_urls": ["https://example.com/x"] * 2},
        {"reference_files": ["x"], "reference_file_urls": ["file:///secret"]},
        {
            "reference_files": ["x"],
            "reference_file_urls": ["https://user:secret@example.com/x"],  # pragma: allowlist secret
        },
    ],
)
def test_bad_reference_metadata_rejected(fields):
    with pytest.raises(ValueError):
        GDPFileTask.model_validate(row(**fields))


async def test_seed_is_idempotent_and_lends_resources_owned_sandbox(server):
    instance, box, request = server
    first, second = await asyncio.gather(
        instance.seed_session(request, seed()), instance.seed_session(request, seed())
    )
    assert first == second
    assert first.sandbox_access.workdir == "/workspace"
    assert first.sandbox_access.connection.descriptor == {"sandbox_id": "task-box"}
    assert request.session[SESSION_ID_KEY] == "resources-1"
    box.start.assert_awaited_once()
    box.stop.assert_not_awaited()
    assert "PRIVATE RUBRIC" not in str(box.start.call_args.args[0])


async def test_seed_conflict_rejected(server):
    instance, _, request = server
    body = seed()
    await instance.seed_session(request, body)
    body.task_data["prompt"] = "Different task"
    with pytest.raises(HTTPException, match="bound"):
        await instance.seed_session(request, body)


def fake_download(monkeypatch, content):
    async def chunks(*args):
        yield content

    download = SimpleNamespace(
        raise_for_status=MagicMock(), release=MagicMock(), content=SimpleNamespace(iter_chunked=chunks)
    )
    get = AsyncMock(return_value=download)
    monkeypatch.setattr(sandbox_app, "http_request", get)
    return download, get


async def test_pristine_references_are_preserved_for_judge_after_sandbox_mutation(server, monkeypatch):
    instance, box, request = server
    original = b"PK\x00\xffbinary spreadsheet\n"
    download, get = fake_download(monkeypatch, original)
    body = seed()
    body.task_data.update(reference_files=["reference_files/a.xlsx"], reference_file_urls=["https://example.com/a"])
    await instance.seed_session(request, body)
    remote = f"{INPUT_DIR}/reference_files/a.xlsx"
    assert box.files[remote] == original
    box.files[remote] = b"agent mutation"

    async def grade(self, payload):
        assert Path(payload.deliverables_dir, "reference_files/a.xlsx").read_bytes() == original
        box.stop.assert_awaited_once()
        return GDPValVerifyResponse(**payload.model_dump(), reward=0.5)

    monkeypatch.setattr(GDPValResourcesServer, "verify", grade)
    verdict = await instance.verify(request, GDPValVerifyRequest(**row(), response=response()))
    assert verdict.reward == 0.5
    download.release.assert_called_once()
    assert get.call_args.kwargs["headers"] == {}


@pytest.mark.parametrize("url,authenticated", [("https://huggingface.co/x", True), ("https://example.com/x", False)])
async def test_hf_auth_is_not_sent_to_other_reference_hosts(server, monkeypatch, url, authenticated):
    instance, _, request = server
    from pydantic import SecretStr

    instance.config.hf_token = SecretStr("test-credential")
    _, get = fake_download(monkeypatch, b"reference")
    body = seed()
    body.task_data.update(reference_files=["a"], reference_file_urls=[url])
    await instance.seed_session(request, body)
    assert ("Authorization" in get.call_args.kwargs["headers"]) == authenticated


async def test_reference_limit_stops_and_releases_failed_download(server, monkeypatch):
    instance, box, request = server
    instance.config.max_reference_bytes = 3
    download, _ = fake_download(monkeypatch, b"larger than allowed")
    body = seed()
    body.task_data.update(reference_files=["a"], reference_file_urls=["https://example.com/a"])
    with pytest.raises(RuntimeError, match="max_reference_bytes"):
        await instance.seed_session(request, body)
    download.release.assert_called_once()
    box.stop.assert_awaited_once()
    box.serialize.assert_not_awaited()


@pytest.mark.parametrize("stage", ["references", "serialize"])
async def test_seed_failure_and_cancellation_retain_ownership_until_cleanup(server, monkeypatch, stage):
    instance, box, request = server
    if stage == "references":
        monkeypatch.setattr(instance, "_stage_references", AsyncMock(side_effect=asyncio.CancelledError))
        error = asyncio.CancelledError
    else:
        box.serialize.side_effect = RuntimeError("serialization failed")
        error = RuntimeError
    box.stop.side_effect = [RuntimeError("provider unavailable"), None]
    with pytest.raises(error):
        await instance.seed_session(request, seed())
    assert instance._sessions["resources-1"].failed
    with pytest.raises(HTTPException, match="closed or failed"):
        await instance.seed_session(request, seed())
    await instance.close_resources_session(
        request, ResourcesCloseSessionRequest(resources_session_id="resources-1", episode_id=seed().episode_id)
    )
    assert box.stop.await_count == 2
    assert "resources-1" not in instance._sessions


async def test_verify_exports_bytes_uses_seeded_metadata_and_replays_identical_request(server, monkeypatch):
    instance, box, request = server
    await instance.seed_session(request, seed())

    async def grade(self, payload):
        assert Path(payload.deliverables_dir, "report.csv").read_bytes() == b"name,value\na,3\n"
        assert payload.rubric_pretty == "PRIVATE RUBRIC"
        box.stop.assert_awaited_once()
        return GDPValVerifyResponse(**payload.model_dump(), reward=0.75)

    grader = AsyncMock(side_effect=grade)
    monkeypatch.setattr(GDPValResourcesServer, "verify", lambda self, body: grader(self, body))
    body = GDPValVerifyRequest(**row(rubric_pretty="TAMPERED"), response=response(), deliverables_dir="/untrusted")
    first = await instance.verify(request, body)
    assert (await instance.verify(request, body)).reward == first.reward == 0.75
    assert grader.await_count == 1
    changed = body.model_copy(update={"reference_ids": ["other"]})
    with pytest.raises(HTTPException, match="changed"):
        await instance.verify(request, changed)
    directory = instance._sessions["resources-1"].directory
    assert json.loads((directory / "verdict.json").read_text())["reward"] == 0.75
    close = ResourcesCloseSessionRequest(resources_session_id="resources-1", episode_id=seed().episode_id)
    assert await instance.close_resources_session(request, close) == await instance.close_resources_session(
        request, close
    )
    box.stop.assert_awaited_once()
    assert Path(first.deliverables_dir, "report.csv").exists()


async def test_execute_only_persists_identity_and_verification_input_without_judge(server, monkeypatch):
    instance, box, request = server
    instance.config.execute_only = True
    grader = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "verify", grader)
    await instance.seed_session(request, seed())
    result = await instance.verify(request, GDPValVerifyRequest(**row(), response=response()))
    assert result.mask_sample is True and result.execute_only is True
    assert result.judge_response is None and result.total_wins is None
    saved = json.loads(Path(result.generation_manifest).read_text())
    assert saved["episode_id"] == {"rollout_id": "rollout-1", "attempt": 0}
    assert saved["task_id"] == {"taskset": "gdp", "task_id": "task-1"}
    payload = GDPValVerifyRequest.model_validate(saved["verify_request"])
    assert payload.task_id == "task-1"
    assert Path(payload.deliverables_dir, "report.csv").is_file()
    from resources_servers.gdpval.comparison import task_attempted

    assert task_attempted(payload.deliverables_dir)
    marker = json.loads(Path(payload.deliverables_dir, "finish_params.json").read_text())
    assert marker["submission_method"] == "nooa_final_response_output_directory"
    assert payload.response == response()
    box.stop.assert_awaited_once()
    grader.assert_not_awaited()
    metrics = await instance.aggregate_metrics(AggregateMetricsRequest(verify_responses=[result.model_dump()]))
    assert metrics.agent_metrics == {"generation/exported": 1}
    assert metrics.key_metrics == {}


async def test_invalid_judge_is_retryable_without_reexport_or_new_generation(server, monkeypatch):
    instance, box, request = server
    await instance.seed_session(request, seed())
    attempts = 0

    async def grade(self, body):
        nonlocal attempts
        attempts += 1
        return GDPValVerifyResponse(**body.model_dump(), reward=0.0, invalid_judge_response=attempts == 1)

    monkeypatch.setattr(GDPValResourcesServer, "verify", grade)
    body = GDPValVerifyRequest(**row(), response=response())
    with pytest.raises(HTTPException) as error:
        await instance.verify(request, body)
    assert error.value.status_code == 503
    result = await instance.verify(request, body)
    assert result.invalid_judge_response is False
    box.download.assert_awaited_once()
    box.stop.assert_awaited_once()


async def test_close_before_seed_fences_delayed_seed_and_wrong_episode(server):
    instance, box, request = server
    close = ResourcesCloseSessionRequest(resources_session_id="resources-1", episode_id=seed().episode_id)
    await instance.close_resources_session(request, close)
    with pytest.raises(HTTPException, match="closed"):
        await instance.seed_session(request, seed())
    with pytest.raises(HTTPException, match="episode"):
        await instance.close_resources_session(
            request, close.model_copy(update={"episode_id": EpisodeId(rollout_id="x")})
        )
    box.start.assert_not_awaited()


async def test_wrong_session_or_task_cannot_verify(server):
    instance, _, request = server
    await instance.seed_session(request, seed())
    for bad_request, task_id in [(SimpleNamespace(session={}), "task-1"), (request, "other")]:
        with pytest.raises(HTTPException, match="does not match"):
            await instance.verify(bad_request, GDPValVerifyRequest(**row(task_id=task_id), response=response()))


async def test_download_changes_prevent_grading(server, monkeypatch):
    instance, box, request = server
    await instance.seed_session(request, seed())

    async def corrupt(remote, local):
        Path(local).write_bytes(b"x" * len(box.files[remote]))

    box.download.side_effect = corrupt
    grader = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "verify", grader)
    with pytest.raises(HTTPException, match="changed during export"):
        await instance.verify(request, GDPValVerifyRequest(**row(), response=response()))
    grader.assert_not_awaited()


@pytest.mark.parametrize("kind", ["file", "symlink", "directory", "hardlink", "too_large"])
def test_actual_export_listing_checks_files_and_limits(tmp_path, kind):
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "report.csv"
    outside = tmp_path / "not-a-deliverable"
    outside.write_bytes(b"private data")
    if kind in {"file", "too_large"}:
        candidate.write_bytes(b"a,b\n1,2\n")
    elif kind == "symlink":
        candidate.symlink_to(outside)
    elif kind == "directory":
        candidate.mkdir()
    else:
        candidate.hardlink_to(outside)
    script = (
        sandbox_app._LIST_OUTPUTS.replace("__OUTPUT_DIR__", repr(str(output)))
        .replace("__MAX_FILES__", "100")
        .replace("__MAX_BYTES__", "3" if kind == "too_large" else "100")
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    if kind == "file":
        assert result.returncode == 0
        assert json.loads(result.stdout) == [
            {"name": "report.csv", "size": 8, "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}
        ]
    elif kind == "directory":
        assert result.returncode == 0
        assert json.loads(result.stdout) == []
    else:
        assert result.returncode != 0


def test_actual_recursive_export_preserves_relative_names_and_authored_zip(tmp_path):
    output = tmp_path / "output"
    expected = {"project/a/report.csv": b"first", "project/b/report.csv": b"second", "project.zip": b"authored ZIP"}
    for name, data in expected.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    script = (
        sandbox_app._LIST_OUTPUTS.replace("__OUTPUT_DIR__", repr(str(output)))
        .replace("__MAX_FILES__", "100")
        .replace("__MAX_BYTES__", "1000")
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    exported = {item["name"]: item for item in json.loads(result.stdout)}
    assert set(exported) == set(expected)
    for name, data in expected.items():
        assert exported[name]["size"] == len(data)
        assert exported[name]["sha256"] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("kind", ["symlink_directory", "nested_symlink", "nested_hardlink", "fifo", "count", "bytes"])
def test_actual_recursive_export_rejects_unsafe_entries_and_aggregate_limits(tmp_path, kind):
    output = tmp_path / "output"
    nested = output / "project"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "private.txt"
    secret.write_bytes(b"private")
    if kind == "symlink_directory":
        (nested / "escape").symlink_to(outside, target_is_directory=True)
    elif kind == "nested_symlink":
        (nested / "escape").symlink_to(secret)
    elif kind == "nested_hardlink":
        (nested / "escape").hardlink_to(secret)
    elif kind == "fifo":
        os.mkfifo(nested / "fifo")
    else:
        (output / "first.txt").write_bytes(b"123")
        (nested / "second.txt").write_bytes(b"456")
    script = (
        sandbox_app._LIST_OUTPUTS.replace("__OUTPUT_DIR__", repr(str(output)))
        .replace("__MAX_FILES__", "1" if kind == "count" else "100")
        .replace("__MAX_BYTES__", "5" if kind == "bytes" else "1000")
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode != 0
    assert result.stdout == ""


def test_actual_recursive_export_does_not_silently_skip_unreadable_directory(tmp_path):
    output = tmp_path / "output"
    (output / "project").mkdir(parents=True)
    (output / "project/report.txt").write_text("must not disappear")
    script = (
        sandbox_app._LIST_OUTPUTS.replace("__OUTPUT_DIR__", repr(str(output)))
        .replace("__MAX_FILES__", "100")
        .replace("__MAX_BYTES__", "1000")
    )
    injection = """
import os
original_scandir = os.scandir
def failing_scandir(path):
    if str(path).endswith('/project'):
        raise PermissionError('injected directory read failure')
    return original_scandir(path)
os.scandir = failing_scandir
"""
    result = subprocess.run([sys.executable, "-c", injection + script], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "PermissionError: injected directory read failure" in result.stderr
    assert result.stdout == ""


async def test_execute_only_preserves_nested_tree_and_original_zip_bytes(server, monkeypatch):
    instance, box, request = server
    instance.config.execute_only = True
    expected = {"project/src/report.csv": b"nested report", "project.zip": b"original authored archive"}
    box.files = {f"{OUTPUT_DIR}/{name}": data for name, data in expected.items()}
    grader = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "verify", grader)
    await instance.seed_session(request, seed())
    result = await instance.verify(request, GDPValVerifyRequest(**row(), response=response()))
    directory = Path(result.deliverables_dir)
    for name, data in expected.items():
        assert (directory / name).read_bytes() == data
    artifacts = json.loads((Path(result.generation_manifest).parent / "artifacts.json").read_text())
    assert {entry["name"]: entry["sha256"] for entry in artifacts} == {
        name: hashlib.sha256(data).hexdigest() for name, data in expected.items()
    }
    finish = json.loads((directory / "finish_params.json").read_text())
    assert {Path(name).relative_to(directory).as_posix() for name in finish["paths"]} == set(expected)
    assert result.execute_only and result.mask_sample
    grader.assert_not_awaited()
    box.stop.assert_awaited_once()


@pytest.mark.parametrize(
    "name", ["reference_files/override.txt", "finish_params.json/child", "../escape", "a/../../escape"]
)
async def test_recursive_export_cannot_replace_verifier_owned_paths(server, monkeypatch, name):
    instance, box, request = server
    await instance.seed_session(request, seed())
    entry = {"name": name, "size": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
    box.exec.side_effect = None
    box.exec.return_value = SimpleNamespace(return_code=0, stdout=json.dumps([entry]), stderr="")
    grader = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "verify", grader)
    with pytest.raises((HTTPException, ValueError)):
        await instance.verify(request, GDPValVerifyRequest(**row(), response=response()))
    grader.assert_not_awaited()
    box.download.assert_not_awaited()


def test_config_rejects_relative_output_and_multiple_workers(server):
    instance, _, _ = server
    for change in ({"deliverables_root": "relative"}, {"num_workers": 2}, {"max_reference_bytes": 0}):
        with pytest.raises(ValidationError):
            GDPSandboxConfig.model_validate(instance.config.model_dump() | change)


def test_native_endpoints_preserve_generation_receipt_and_cleanup(server):
    instance, box, _ = server
    instance.config.execute_only = True
    with TestClient(instance.setup_webserver()) as client:
        seeded = client.post("/seed_session", json=seed().model_dump(mode="json"))
        assert seeded.status_code == 200
        assert seeded.json()["sandbox_access"]["workdir"] == "/workspace"
        verified = client.post(
            "/verify", json=GDPValVerifyRequest(**row(), response=response()).model_dump(mode="json")
        )
        assert verified.status_code == 200
        result = verified.json()
        assert result["mask_sample"] is True
        assert result["execute_only"] is True
        assert Path(result["generation_manifest"]).is_file()
        closed = client.post(
            "/close_session",
            json=ResourcesCloseSessionRequest(
                resources_session_id="resources-1", episode_id=seed().episode_id
            ).model_dump(mode="json"),
        )
        assert closed.status_code == 200
    box.stop.assert_awaited_once()


def test_nooa_recipe_resolves_with_explicit_runtime_and_artifact_paths(monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
    from nemo_gym.global_config import GlobalConfigDictParser

    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("GDPVAL_CONTAINER_PATH", "/images/audited-gdp.sif")
    monkeypatch.setenv("PERSIST_DELIVERABLES_DIR", str(tmp_path))
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(root / "benchmarks/gdpval/nooa.yaml")])
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    environment = SingleAgentTurnEnvironmentServerConfig(
        name="gdpval_nooa",
        host="localhost",
        port=8000,
        **OmegaConf.to_container(config.gdpval_nooa.environment_servers.single_agent_turn, resolve=True),
    )
    agent = config.gdpval_nooa_agent.responses_api_agents.nooa_agent
    resources = GDPSandboxConfig(
        name=environment.resources_server.name,
        host="localhost",
        port=8002,
        **OmegaConf.to_container(config.gdpval_nooa_resources_server.resources_servers.gdpval, resolve=True),
    )
    assert resources.execute_only and not resources.preconvert_office_to_pdf
    assert resources.image == "/images/audited-gdp.sif"
    assert resources.deliverables_root == tmp_path
    assert environment.max_concurrent_episodes == config.num_samples_in_parallel == 8
    assert environment.default_episode_timeout_seconds == 21600
    assert environment.cleanup_timeout_seconds == 180
    assert environment.queue_timeout_seconds == 300
    config.num_samples_in_parallel = 1
    assert (
        OmegaConf.to_container(config.gdpval_nooa.environment_servers.single_agent_turn, resolve=True)[
            "max_concurrent_episodes"
        ]
        == 1
    )
    assert agent.context_window == 262144 and agent.max_policy_calls == 250
    assert agent.nooa.execution_mode == "sandboxed"
    assert "--userns" in config.sandbox.apptainer.create.extra_start_args
    assert "--fakeroot" not in config.sandbox.apptainer.create.extra_start_args
    assert "--fakeroot" not in config.sandbox.apptainer.exec.extra_exec_args
    assert config.sandbox.apptainer.exec.fakeroot_for_root is False
    assert config.sandbox.apptainer.exec.default_timeout_s is None
    assert environment.resources_server.name == "gdpval_nooa_resources_server"
    assert environment.agent_server.name == "gdpval_nooa_agent"
    assert "rollout_collection_driver" not in config


@pytest.mark.parametrize("failure", ["command", "listing", "entry", "reserved", "limit", "size"])
async def test_invalid_export_never_reaches_judge(server, monkeypatch, failure):
    instance, box, request = server
    await instance.seed_session(request, seed())
    original = box.files[f"{OUTPUT_DIR}/report.csv"]
    entry = {"name": "report.csv", "size": len(original), "sha256": hashlib.sha256(original).hexdigest()}
    value = [entry]
    if failure == "listing":
        value = {}
    elif failure == "entry":
        value = ["not a file"]
    elif failure == "reserved":
        entry["name"] = "finish_params.json"
    elif failure == "limit":
        instance.config.max_deliverable_bytes = 1
    elif failure == "size":
        entry["size"] += 1
    box.exec.side_effect = None
    box.exec.return_value = SimpleNamespace(
        return_code=1 if failure == "command" else 0, stdout=json.dumps(value), stderr="listing failed"
    )
    grader = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "verify", grader)
    with pytest.raises(HTTPException) as error:
        await instance.verify(request, GDPValVerifyRequest(**row(), response=response()))
    assert error.value.status_code == 503
    grader.assert_not_awaited()
    assert instance._sessions["resources-1"].deliverables is None


def test_shutdown_stops_an_abandoned_owned_sandbox(server):
    instance, box, _ = server
    with TestClient(instance.setup_webserver()) as client:
        assert client.post("/seed_session", json=seed().model_dump(mode="json")).status_code == 200
        box.stop.assert_not_awaited()
    box.stop.assert_awaited_once()


async def test_generation_is_durable_when_owner_stop_needs_retry(server, monkeypatch):
    instance, box, request = server
    instance.config.execute_only = True
    grader = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "verify", grader)
    await instance.seed_session(request, seed())
    box.stop.side_effect = [RuntimeError("provider temporarily unavailable"), None]
    body = GDPValVerifyRequest(**row(), response=response())
    with pytest.raises(RuntimeError):
        await instance.verify(request, body)
    session = instance._sessions["resources-1"]
    assert Path(session.directory, "generation.json").is_file()
    assert Path(session.deliverables, "finish_params.json").is_file()
    assert not session.stopped
    result = await instance.verify(request, body)
    assert result.execute_only
    assert box.start.await_count == 1 and box.stop.await_count == 2
    box.download.assert_awaited_once()
    grader.assert_not_awaited()
    finish = json.loads(Path(session.deliverables, "finish_params.json").read_text())
    assert [Path(path).name for path in finish["paths"]] == ["report.csv"]
