# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task ownership and artifact isolation without external infrastructure."""

import asyncio
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
)
from resources_servers.legal_agent_bench import session_app as lab


def seed(session_id="s", task="domain__task"):
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=session_id),
        task_id=TaskId(taskset="lab", task_id=f"legal_agent_bench::{task}"),
        task_data={"instance_id": f"legal_agent_bench::{task}"},
    )


def request():
    return Request({"type": "http", "session": {}})


@pytest.fixture
def server(tmp_path, monkeypatch):
    task = tmp_path / "tasks/domain__task"
    (task / "documents").mkdir(parents=True)
    (task / "documents" / "public.txt").write_text("Public input")
    (task / "task.json").write_text(
        json.dumps(
            {
                "title": "Task",
                "instructions": "Write answer.txt",
                "criteria": [{"match_criteria": "hidden verifier rubric"}],
                "metadata": {"lab_source_revision": lab.LAB_SOURCE_REVISION},
            }
        )
    )
    skills = tmp_path / "skills/docx"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("Skill instructions")
    box = SimpleNamespace(stop=AsyncMock(), serialize=AsyncMock(return_value={"id": "owned"}), exec=AsyncMock())

    async def start(spec, setup):
        box.spec = spec
        await setup(box)

    box.exec.return_value = SimpleNamespace(error_type=None, return_code=0)

    async def upload(path, target):
        with tarfile.open(path) as archive:
            box.uploaded_names = archive.getnames()

    box.upload = AsyncMock(side_effect=upload)
    box.start_with_setup = AsyncMock(side_effect=start)
    config = lab.LabSessionResourcesConfig(
        name="lab",
        entrypoint="session_app.py",
        judges=[{"model_name": "claude-sonnet-4-6"}, {"model_name": "gpt-5.5"}],
        host="127.0.0.1",
        port=0,
        harbor_tasks_cache_dir=str(task.parent),
        harness_skills_dir=str(skills.parent),
        sandbox_image="lab:test",
        results_dir=str(tmp_path / "results"),
    )
    result = lab.LabSessionResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(lab.LabSessionResourcesServer, "_provider", lambda self: ({"docker": {}}, {}))
    monkeypatch.setattr(lab, "AsyncSandbox", lambda provider: box)
    return result, box


@pytest.mark.asyncio
async def test_seed_is_idempotent_and_hides_rubric_and_judge_credentials(server):
    resource, box = server
    resource.config.judge_api_key = "verifier-only-test-value"
    req = request()
    first, second = await resource.seed_session(req, seed()), await resource.seed_session(req, seed())
    assert first == second
    assert "Public input" not in first.observation  # documents are staged, not pasted into prompt
    assert "Skill instructions" in first.observation and "hidden verifier rubric" not in first.observation
    assert "verifier-only-test-value" not in json.dumps(box.spec.env)
    assert first.sandbox_access.env["OUTPUT_DIR"] == "/workspace/output"
    assert "LAB_JUDGE_API_KEY" not in first.sandbox_access.env
    assert req.session[SESSION_ID_KEY] == "s"
    assert box.start_with_setup.await_count == 1
    assert set(box.uploaded_names) == {"vdr", "vdr/public.txt", "skills", "skills/docx", "skills/docx/SKILL.md"}
    await resource.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id="s", episode_id=seed().episode_id)
    )
    assert box.stop.await_count == 1


@pytest.mark.asyncio
async def test_close_fences_unknown_seed_and_rejects_wrong_episode(server):
    resource, box = server
    await resource.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id="s", episode_id=seed().episode_id)
    )
    with pytest.raises(ResourcesSessionConflictError):
        await resource.seed_session(request(), seed())
    with pytest.raises(ResourcesSessionConflictError):
        await resource.close_resources_session(
            ResourcesCloseSessionRequest(resources_session_id="s", episode_id=EpisodeId(rollout_id="other"))
        )
    assert box.start_with_setup.await_count == 0


@pytest.mark.asyncio
async def test_cleanup_failure_retains_owner_and_slot_for_retry(server):
    resource, box = server
    await resource.seed_session(request(), seed())
    close = ResourcesCloseSessionRequest(resources_session_id="s", episode_id=seed().episode_id)
    box.stop.side_effect = [RuntimeError("temporary provider failure"), None]
    with pytest.raises(RuntimeError):
        await resource.close_resources_session(close)
    assert "s" in resource._sessions and resource._slots.locked()
    await resource.close_resources_session(close)
    assert "s" not in resource._sessions and not resource._slots.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True], ids=["upload-error", "cancelled"])
async def test_failed_setup_and_cleanup_cannot_be_reused_by_seed_retry(server, cancelled):
    resource, box = server
    setup_error = asyncio.CancelledError() if cancelled else RuntimeError("upload failed")
    box.upload.side_effect = setup_error
    box.stop.side_effect = [RuntimeError("cleanup unavailable"), None]
    req = request()

    with pytest.raises(type(setup_error)) as failure:
        await resource.seed_session(req, seed())
    assert failure.value is setup_error
    assert resource._sessions["s"].sandbox is box
    assert resource._slots.locked()
    assert SESSION_ID_KEY not in req.session
    box.serialize.assert_not_awaited()

    with pytest.raises(ResourcesSessionConflictError, match="setup did not complete"):
        await resource.seed_session(req, seed())
    assert box.start_with_setup.await_count == 1
    assert box.stop.await_count == 1
    assert SESSION_ID_KEY not in req.session
    box.serialize.assert_not_awaited()

    await resource.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id="s", episode_id=seed().episode_id)
    )
    assert box.stop.await_count == 2
    assert "s" not in resource._sessions and not resource._slots.locked()
    with pytest.raises(ResourcesSessionConflictError):
        await resource.seed_session(request(), seed())
    assert box.start_with_setup.await_count == 1


@pytest.mark.asyncio
async def test_concurrency_slot_blocks_second_seed_until_close(server):
    resource, box = server
    await resource.seed_session(request(), seed())
    waiting = asyncio.create_task(resource.seed_session(request(), seed("second")))
    await asyncio.sleep(0)
    assert not waiting.done() and box.start_with_setup.await_count == 1
    await resource.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id="s", episode_id=seed().episode_id)
    )
    await asyncio.wait_for(waiting, 2)
    await resource.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id="second", episode_id=seed("second").episode_id)
    )


@pytest.mark.parametrize(
    "name, kind",
    [
        ("output/link", tarfile.SYMTYPE),
        ("output/link", tarfile.LNKTYPE),
        ("../outside", tarfile.REGTYPE),
        ("/output/absolute", tarfile.REGTYPE),
        ("other/file", tarfile.REGTYPE),
        ("output/device", tarfile.CHRTYPE),
        ("output/sparse", tarfile.GNUTYPE_SPARSE),
    ],
)
async def test_unsafe_output_archive_is_rejected_before_any_publication(tmp_path, name, kind):
    source = tmp_path / "archive.tgz"
    with tarfile.open(source, "w:gz") as archive:
        okay = tarfile.TarInfo("output/first.txt")
        okay.size = 4
        archive.addfile(okay, io.BytesIO(b"safe"))
        bad = tarfile.TarInfo(name)
        bad.type, bad.linkname = kind, "/etc/passwd"
        archive.addfile(bad)
    destination = tmp_path / "results"
    with pytest.raises(ValueError, match="Unsafe LAB"):
        await lab.extract_outputs(source, destination, prefix="output", limits=lab.ArchiveLimits())
    assert not destination.exists()


async def test_safe_output_archive_retains_nested_documents(tmp_path):
    document = tmp_path / "work/output/nested/result.txt"
    document.parent.mkdir(parents=True)
    document.write_text("Work product")
    archive = tmp_path / "archive.tgz"
    lab._archive_inputs(archive, [(document.parent.parent, "output")])
    await lab.extract_outputs(archive, tmp_path / "result", prefix="output", limits=lab.ArchiveLimits())
    assert (tmp_path / "result/output/nested/result.txt").read_text() == "Work product"


@pytest.mark.asyncio
async def test_verify_stops_owner_before_grading_and_preserves_evidence_on_wire(server, tmp_path, monkeypatch):
    from nemo_gym.core import NeMoGymResponse

    resource, box = server
    monkeypatch.setattr(lab, "get_global_config_dict", lambda: {})
    req = request()
    await resource.seed_session(req, seed())
    output = tmp_path / "agent/output/answer.txt"
    output.parent.mkdir(parents=True)
    output.write_text("Legal work product")
    transfer = tmp_path / "output.tgz"
    lab._archive_inputs(transfer, [(output.parent, "output")])

    async def download(remote, destination):
        destination.write_bytes(transfer.read_bytes())

    async def download_archive(sandbox, remote, destination, **kwargs):
        assert sandbox is box
        await download(remote, destination)

    monkeypatch.setattr(lab, "download_archive", download_archive)

    async def grade(self, state):
        assert state.sandbox is None and box.stop.await_count == 1
        assert (state.artifact_dir / "output/answer.txt").read_text() == "Legal work product"
        return {"reward": 0.5, "judge_error_count": 0}, {"judge_profile": "official_dual", "score": 0.5}

    grading = AsyncMock(side_effect=grade)
    monkeypatch.setattr(lab.LabSessionResourcesServer, "_grade", lambda self, state: grading(self, state))
    response = NeMoGymResponse(
        id="r",
        created_at=0,
        model="policy",
        object="response",
        output=[],
        status="completed",
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=False,
    )
    body = lab.LabSessionVerifyRequest(
        instance_id=seed().task_data["instance_id"], responses_create_params={"input": []}, response=response
    )
    result = await resource.verify(req, body)
    wire = result.model_dump(mode="json")
    assert wire["instance_id"] == body.instance_id and wire["reward"] == 0.5
    assert wire["lab_metrics"]["judge_error_count"] == 0 and wire["lab_scores"]["score"] == 0.5
    assert wire["source_revision"] == lab.LAB_SOURCE_REVISION and Path(wire["artifact_dir"]).is_dir()
    assert await resource.verify(req, body) == result and grading.await_count == 1
    changed = body.model_copy(update={"response": response.model_copy(update={"id": "different"})})
    with pytest.raises(ResourcesSessionConflictError, match="another agent response"):
        await resource.verify(req, changed)


def test_http_seed_preserves_cookie_and_lifespan_releases_owner(server, monkeypatch):
    from fastapi.testclient import TestClient
    from omegaconf import OmegaConf

    resource, box = server
    resource.server_client.global_config_dict = OmegaConf.create({})
    monkeypatch.setattr(lab, "ensure_assets", lambda **kwargs: {})
    app = resource.setup_webserver()
    with TestClient(app) as client:
        first = client.post("/seed_session", json=seed().model_dump(mode="json"))
        first.raise_for_status()
        assert client.cookies
        assert first.json()["resources_session_id"] == "s"
        assert first.json()["sandbox_access"]["env"]["OUTPUT_DIR"] == "/workspace/output"
        second = client.post("/seed_session", json=seed().model_dump(mode="json"))
        assert second.status_code == 200 and second.json() == first.json()
        assert box.start_with_setup.await_count == 1
        schema = client.get("/openapi.json").json()
        assert schema["paths"]["/verify"]["post"]["responses"]["200"]["content"]["application/json"]["schema"][
            "$ref"
        ].endswith("LabSessionVerifyResponse")
    assert box.stop.await_count == 1 and not resource._sessions


@pytest.mark.parametrize("extra_args", [[], ["--fakeroot"]])
def test_apptainer_sessions_disable_implicit_host_mounts_without_mutating_provider_config(monkeypatch, extra_args):
    from omegaconf import OmegaConf

    shared = OmegaConf.create({"sandbox": {"apptainer": {"create": {"extra_start_args": extra_args}}}})
    before = OmegaConf.to_container(shared, resolve=True)
    monkeypatch.setattr(lab, "get_global_config_dict", lambda: shared)
    server = lab.LabSessionResourcesServer(
        config=lab.LabSessionResourcesConfig(
            name="lab", host="127.0.0.1", port=0, entrypoint="session_app.py", judges=[{"model_name": "gpt-5.5"}]
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    provider, _ = server._provider()
    flags = provider["apptainer"]["create"]["extra_start_args"]
    assert "--containall" in flags and "--no-home" in flags
    assert flags[flags.index("--no-mount") + 1] == "cwd,hostfs"
    assert all(argument in flags for argument in extra_args)
    assert OmegaConf.to_container(shared, resolve=True) == before


async def test_verifier_teardown_failure_retains_downloaded_reports_for_cleanup_retry(server, tmp_path, monkeypatch):
    resource, box = server
    await resource.seed_session(request(), seed())
    state = resource._sessions["s"]
    await resource._stop_task(state)
    (state.task_dir / "tests").mkdir()
    (state.task_dir / "tests/task.json").write_text("{}")
    (state.artifact_dir / "output").mkdir(parents=True)
    (state.artifact_dir / "config.json").write_text("{}")
    reports = tmp_path / "verification"
    reports.mkdir()
    (reports / "scores.json").write_text('{"score": 0.5}')
    archive = tmp_path / "reports.gz"
    lab._archive_inputs(archive, [(reports, "verification")])

    async def download(sandbox, remote, destination, **kwargs):
        assert sandbox is box
        destination.write_bytes(archive.read_bytes())

    monkeypatch.setattr(lab, "download_archive", download)
    box.stop.side_effect = [RuntimeError("teardown failed"), None]
    with pytest.raises(RuntimeError, match="teardown failed"):
        await resource._grade(state)
    assert (state.artifact_dir / ".verifier-transfer.tar.gz").read_bytes() == archive.read_bytes()
    assert state.verifier is box and state.result is None
    await resource.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id="s", episode_id=seed().episode_id)
    )
    assert state.verifier is None and "s" not in resource._sessions


async def test_two_sessions_can_prepare_concurrently_and_release_both_slots(server, monkeypatch):
    template, _ = server
    resource = lab.LabSessionResourcesServer(
        config=template.config.model_copy(update={"concurrency": 2}), server_client=template.server_client
    )
    boxes = []
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = 0

    def new_box(provider):
        box = SimpleNamespace(
            stop=AsyncMock(),
            upload=AsyncMock(),
            exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None)),
        )
        boxes.append(box)
        box.serialize = AsyncMock(return_value={"id": str(len(boxes))})

        async def start(spec, setup):
            nonlocal started
            await setup(box)
            started += 1
            if started == 2:
                both_started.set()
            await release.wait()

        box.start_with_setup = start
        return box

    monkeypatch.setattr(lab, "AsyncSandbox", new_box)
    seeds = [seed("first"), seed("second")]
    tasks = [asyncio.create_task(resource.seed_session(request(), body)) for body in seeds]
    try:
        await asyncio.wait_for(both_started.wait(), 2)
        assert all(not task.done() for task in tasks)
        assert len(resource._sessions) == 2
        release.set()
        first, second = await asyncio.gather(*tasks)
        assert first.sandbox_access.connection.descriptor != second.sandbox_access.connection.descriptor
        for box in boxes:
            box.stop.assert_not_awaited()
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for body in seeds:
            await resource.close_resources_session(
                ResourcesCloseSessionRequest(
                    resources_session_id=body.resources_session_id, episode_id=body.episode_id
                )
            )
    assert not resource._sessions and resource._slots._value == 2
    for box in boxes:
        box.stop.assert_awaited_once()
