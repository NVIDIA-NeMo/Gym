# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agentic IF Resources Server owns the episode: seed lends the sandbox, verify grades, close releases.

The task backend and the judge are a ServerClient double, the task sandbox a scripted transport, and the plugin bundle
a double of its staging and collection; everything else is the real server.
"""

import asyncio
import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientConnectionError
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from nemo_gym import PARENT_DIR
from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY
from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.judge import JudgeError
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.sandbox.providers import SandboxExecResult
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.task_data import validate_jsonl_rows
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.agentic_if import app as app_module
from resources_servers.agentic_if import opencode_bundle
from resources_servers.agentic_if.app import (
    AgenticIFResourcesServer,
    AgenticIFResourcesServerConfig,
    AgenticIFVerifyRequest,
)
from resources_servers.agentic_if.episode import resources
from resources_servers.agentic_if.episode.artifacts import load_episode_record
from resources_servers.agentic_if.episode.judging import JudgeUsage, judge_prompt, vote_key
from resources_servers.agentic_if.if_grading import judge as judge_module
from resources_servers.agentic_if.if_grading.configuration import load_config
from resources_servers.agentic_if.task_data import TaskData
from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, seal_runtime_row, validate_runtime
from resources_servers.agentic_if.task_variants.tasks import backend_task_digest
from resources_servers.agentic_if.tests.support import (
    JUDGED,
    WORKSPACE_BASE,
    Backends,
    LocalTransport,
    Transport,
    bundle_outputs,
    call,
    export,
    final_call_record,
    finished_session,
    http_error,
    message,
    presented,
    response,
    swe_row,
    task_data,
    tb21_row,
    text,
    verify_body,
)


EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
PROFILE = "p" * 64


@dataclass(frozen=True)
class BundleReceipt:
    profile_sha256: str
    files: dict


def seed_body(row, *, session="resources-1"):
    return ResourcesSeedSessionRequest(
        resources_session_id=session,
        episode_id=EPISODE,
        task_id=TaskId(taskset="agentic_if:tests", task_id=row["task_id"]),
        task_data=task_data(row),
    )


def caller(session_id=None):
    return SimpleNamespace(session={} if session_id is None else {SESSION_ID_KEY: session_id})


@pytest.fixture
def world(tmp_path, monkeypatch):
    backends = Backends()
    sandbox = Transport()
    timeline = []
    state = SimpleNamespace(
        backends=backends,
        sandbox=sandbox,
        timeline=timeline,
        staged=[],
        outputs=bundle_outputs([finished_session()], call_records=[final_call_record()]),
        stage_error=None,
        expected=[],
        artifacts=tmp_path / "artifacts",
    )

    async def connect(descriptor, *, provider):
        return sandbox

    async def stage_bundle(io, *, spec, agent_user):
        timeline.append("stage")
        if state.stage_error is not None:
            raise state.stage_error
        state.staged.append(SimpleNamespace(io=io, spec=spec, agent_user=agent_user))
        return BundleReceipt(profile_sha256=PROFILE, files={"if-profile.mjs": "0" * 64})

    async def collect_bundle(io, *, expected=None):
        timeline.append("collect")
        state.expected.append(expected)
        if isinstance(state.outputs, Exception):
            raise state.outputs
        return state.outputs

    post = backends.post

    async def recorded(**kwargs):
        timeline.append(f"{kwargs['server_name']}{kwargs['url_path']}")
        return await post(**kwargs)

    backends.post = recorded
    monkeypatch.setattr(resources.AsyncSandbox, "connect", connect)
    monkeypatch.setattr(resources, "create_provider", lambda config: SimpleNamespace())
    monkeypatch.setattr(opencode_bundle, "stage_bundle", stage_bundle)
    monkeypatch.setattr(opencode_bundle, "collect_bundle", collect_bundle)
    monkeypatch.setattr(judge_module.time, "sleep", lambda seconds: None)
    state.server = make_server(state.artifacts, backends)
    return state


def make_server(artifacts, backends, **settings):
    config = AgenticIFResourcesServerConfig(
        **{
            "host": "127.0.0.1",
            "port": 12345,
            "entrypoint": "app.py",
            "name": "agentic_if",
            "judge_model_server": {"type": "responses_api_models", "name": "judge"},
            "task_resources": {
                "swe": {"type": "resources_servers", "name": "tasks"},
                "tb21": {"type": "resources_servers", "name": "tasks"},
            },
            "artifacts_dir": artifacts,
            "max_concurrent_grades": 1,
        }
        | settings
    )
    client = MagicMock(spec=ServerClient)
    client.post = backends.post
    client.global_config_dict = backends.global_config_dict
    return AgenticIFResourcesServer(config=config, server_client=client)


async def seed(world, row, **options):
    request = caller()
    reply = await world.server.seed_session(request, seed_body(row, **options))
    return reply, request.session[SESSION_ID_KEY]


async def verify(world, row, session_id, *, answer_response=None, **extra):
    body = AgenticIFVerifyRequest.model_validate(verify_body(row, answer_response, **extra))
    return await world.server.verify(caller(session_id), body)


# ------------------------------------------------------------------------------------------------------- seed


@pytest.mark.asyncio
async def test_seed_lends_the_backend_sandbox_after_preparing_the_workspace_and_plugins(world):
    row = swe_row()
    reply, session_id = await seed(world, row)
    assert session_id == "resources-1" and reply.resources_session_id == "resources-1"
    (backend_seed,) = world.backends.to("/seed_session")
    seeded = ResourcesSeedSessionRequest.model_validate(backend_seed["json"])
    # The task backend seeds its own task: no IF configuration, identity field or actor request.
    assert seeded.task_data == {"instance_id": "task-a", "workdir": "/app", "image_url": "registry/task-a:1"}
    assert seeded.resources_session_id.startswith("agentic-if-") and seeded.episode_id == EPISODE
    assert reply.sandbox_access.connection.descriptor == {"sandbox_id": seeded.resources_session_id}
    assert reply.sandbox_access.workdir == "/app" and reply.sandbox_access.user is None
    assert reply.sandbox_access.env == {}
    # The baseline is taken as the task identity before the bundle is staged.
    assert any("git rev-parse --is-inside-work-tree" in call["command"] for call in world.sandbox.calls)
    assert world.timeline[-1] == "stage" and world.staged[0].agent_user is None
    # The profile plugin's workspace diffs take the base tree seed observed.
    (episode,) = world.server._episodes.values()
    assert episode.seed["workspace_base"] == WORKSPACE_BASE
    assert json.loads(world.sandbox.uploads[opencode_bundle.WORKSPACE_BASE_PATH]) == {"base_tree": WORKSPACE_BASE}
    assert world.staged[0].spec == validate_runtime({k: v for k, v in row.items() if k != "task_id"})
    again, _ = await seed(world, row)
    assert again == reply and len(world.backends.to("/seed_session")) == 1
    with pytest.raises(HTTPException) as different:
        await seed(world, swe_row(text="Another task"))
    assert different.value.status_code == 409


@pytest.mark.asyncio
async def test_seed_stages_the_configured_runtime_files_for_the_agent(world, tmp_path):
    rg, archive = tmp_path / "rg", tmp_path / "plugin-dependencies.tar.gz"
    rg.write_bytes(b"ripgrep")
    archive.write_bytes(b"dependencies")
    world.server = make_server(world.artifacts, world.backends, opencode_runtime_files=[rg, archive])

    await seed(world, swe_row())

    root = opencode_bundle.BUNDLE_ROOT
    assert world.sandbox.uploads[f"{root}/rg"] == b"ripgrep"
    assert world.sandbox.uploads[f"{root}/plugin-dependencies.tar.gz"] == b"dependencies"
    chmods = [call for call in world.sandbox.calls if call["command"].startswith(f"chmod 0644 {root}/")]
    # The runtime files and the workspace base the profile plugin reads.
    assert len(chmods) == 3 and all(call["user"] == "root" for call in chmods)


def test_runtime_files_resolve_under_the_gym_root_and_must_exist(world):
    relative = Path("benchmarks/agentic_if/data/not-prepared-for-this-test/rg")
    with pytest.raises(ValueError, match="gym eval prepare") as missing:
        make_server(world.artifacts, world.backends, opencode_runtime_files=[relative])
    assert str(PARENT_DIR / relative) in str(missing.value)
    with pytest.raises(ValueError, match="distinct file names"):
        make_server(world.artifacts, world.backends, opencode_runtime_files=[Path("/a/rg"), Path("/b/rg")])


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["task_receipt", "identity", "backend_disabled", "backend_rejects", "cwd"])
async def test_an_invalid_row_is_a_terminal_422_and_leaves_nothing_allocated(world, tmp_path, defect):
    row = swe_row()
    if defect == "task_receipt":
        row["instance_id"] = "another-task"
    elif defect == "identity":
        row["task_id"] = "0" * 64
    elif defect == "backend_disabled":
        world.server = make_server(
            tmp_path / "other",
            world.backends,
            task_resources={"tb21": {"type": "resources_servers", "name": "tasks"}},
        )
    elif defect == "backend_rejects":
        world.backends.seed_error = http_error(422)
    else:
        row = swe_row(provenance={"materialized_cwd": "/elsewhere"})
    if defect == "identity":
        body = seed_body(row).model_copy(update={"task_id": TaskId(taskset="agentic_if:tests", task_id="1" * 64)})
        with pytest.raises(HTTPException) as rejected:
            await world.server.seed_session(caller(), body)
    else:
        with pytest.raises(HTTPException) as rejected:
            await seed(world, row)
    assert rejected.value.status_code == 422
    assert not world.server._episodes
    seeded = len(world.backends.to("/seed_session"))
    assert len(world.backends.to("/close_session")) == seeded
    assert world.sandbox.disconnects == (1 if defect == "cwd" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["backend_unavailable", "staging", "sandbox"])
async def test_an_infrastructure_failure_is_retryable_and_releases_the_backend_session(world, failure, caplog):
    if failure == "backend_unavailable":
        world.backends.seed_error = http_error(503)
        # Gym's raise_for_status keeps the backend's reply on the error.
        world.backends.seed_error.response_content = b'{"detail": "sandbox create failed"}'
    elif failure == "staging":
        world.stage_error = RuntimeError("upload failed")
    else:
        world.sandbox.fail["git rev-parse --is-inside-work-tree"] = SandboxExecResult(
            stdout="", stderr="transport", return_code=1, error_type="transport"
        )
    with caplog.at_level("WARNING"), pytest.raises(HTTPException) as failed:
        await seed(world, swe_row())
    assert failed.value.status_code == 503
    # The cause reaches both the caller and the server log.
    assert failed.value.detail in caplog.text
    if failure == "backend_unavailable":
        assert "sandbox create failed" in failed.value.detail
    assert not world.server._episodes
    assert len(world.backends.to("/close_session")) == 1
    # A retry of the episode seeds a fresh session.
    world.backends.seed_error, world.stage_error, world.sandbox.fail = None, None, {}
    await seed(world, swe_row(), session="resources-2")


@pytest.mark.asyncio
async def test_a_failed_release_after_a_failed_seed_is_retried_by_close(world):
    world.stage_error = RuntimeError("upload failed")
    world.backends.close_error = http_error(503)
    with pytest.raises(HTTPException):
        await seed(world, swe_row())
    assert "resources-1" in world.server._episodes
    with pytest.raises(HTTPException) as retried_seed:
        await seed(world, swe_row())
    assert retried_seed.value.status_code == 409
    world.backends.close_error = None
    await world.server.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id="resources-1", episode_id=EPISODE)
    )
    assert not world.server._episodes and len(world.backends.to("/close_session")) == 2


@pytest.mark.asyncio
async def test_a_non_root_agent_user_is_gated_and_lent_with_its_home(world):
    row = tb21_row(fields={"agent_user": "agent", "workdir": "/app"})
    reply, _ = await seed(world, row)
    assert reply.sandbox_access.user == "agent"
    assert reply.sandbox_access.env == {"HOME": "/home/agent", "SHELL": "/bin/bash"}
    assert world.staged[0].agent_user == "agent"
    gate = [call for call in world.sandbox.calls if call["command"] in {"id -u", "id -u && id -g"}]
    assert [(call["command"], call["user"]) for call in gate] == [("id -u", "root"), ("id -u && id -g", "agent")]
    baseline = next(call for call in world.sandbox.calls if "git rev-parse --is-inside-work-tree" in call["command"])
    assert baseline["user"] == "agent" and baseline["env"] == {"HOME": "/home/agent", "SHELL": "/bin/bash"}


@pytest.mark.asyncio
async def test_a_root_resolving_agent_user_fails_the_gate_and_releases_the_sandbox(world):
    world.sandbox.uid = "0"
    with pytest.raises(HTTPException) as failed:
        await seed(world, tb21_row(fields={"agent_user": "agent"}))
    assert failed.value.status_code == 503
    assert not world.staged and world.sandbox.disconnects == 1
    assert len(world.backends.to("/close_session")) == 1


@pytest.mark.asyncio
async def test_workspace_files_and_cwd_are_prepared_as_the_actor_before_the_bundle(world):
    row = swe_row(workspace_files={"AGENTS.md": "Literal\n"}, provenance={"materialized_cwd": "/app"})
    await seed(world, row)
    commands = [call["command"] for call in world.sandbox.calls]
    cwd = next(i for i, command in enumerate(commands) if command.startswith("pwd -P &&"))
    staged = next(i for i, command in enumerate(commands) if command.startswith("python3 -c"))
    baseline = next(i for i, command in enumerate(commands) if "git rev-parse --is-inside-work-tree" in command)
    assert cwd < staged < baseline
    assert world.timeline.index("stage") > world.timeline.index("tasks/seed_session")


# ------------------------------------------------------------------------------------------------------ close


def test_session_lifecycle_meets_the_resources_session_contract(world):
    check_resources_session_contract(world.server.setup_webserver(), seed_body(swe_row()), keeps_state=True)


@pytest.mark.asyncio
async def test_close_releases_the_backend_session_and_fences_the_id(world):
    await seed(world, swe_row())
    close = ResourcesCloseSessionRequest(resources_session_id="resources-1", episode_id=EPISODE)
    with pytest.raises(HTTPException) as mismatched:
        await world.server.close_resources_session(close.model_copy(update={"episode_id": EpisodeId(rollout_id="x")}))
    assert mismatched.value.status_code == 409
    world.backends.close_error = http_error(503)
    with pytest.raises(Exception):
        await world.server.close_resources_session(close)
    assert "resources-1" in world.server._episodes  # a failed release stays retryable
    world.backends.close_error = None
    await world.server.close_resources_session(close)
    await world.server.close_resources_session(close)
    assert len(world.backends.to("/close_session")) == 2 and world.sandbox.disconnects == 1
    assert world.sandbox.stops == 0  # the task backend stops its own sandbox
    with pytest.raises(HTTPException) as reseeded:
        await seed(world, swe_row())
    assert reseeded.value.status_code == 409


@pytest.mark.asyncio
async def test_close_during_seed_fences_the_seed_which_releases_its_own_sandbox(world):
    entered, release = asyncio.Event(), asyncio.Event()
    staged = opencode_bundle.stage_bundle

    async def slow_stage(io, *, spec, agent_user):
        entered.set()
        await release.wait()
        return await staged(io, spec=spec, agent_user=agent_user)

    opencode_bundle.stage_bundle = slow_stage
    seeding = asyncio.create_task(seed(world, swe_row()))
    await asyncio.wait_for(entered.wait(), 5)
    closing = asyncio.create_task(
        world.server.close_resources_session(
            ResourcesCloseSessionRequest(resources_session_id="resources-1", episode_id=EPISODE)
        )
    )
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(HTTPException) as fenced:
        await asyncio.wait_for(seeding, 5)
    await asyncio.wait_for(closing, 5)
    assert fenced.value.status_code == 409
    assert not world.server._episodes and "resources-1" in world.server._closed
    assert len(world.backends.to("/close_session")) == 1


@pytest.mark.asyncio
async def test_shutdown_releases_live_sessions(world):
    await seed(world, swe_row())
    await world.server._shutdown()
    assert len(world.backends.to("/close_session")) == 1
    with pytest.raises(HTTPException) as refused:
        await seed(world, swe_row(), session="resources-2")
    assert refused.value.status_code == 503


# ----------------------------------------------------------------------------------------------------- verify


@pytest.mark.asyncio
async def test_verify_reads_the_bundle_before_the_backend_then_persists_and_grades(world):
    row = swe_row()
    _, session_id = await seed(world, row)
    result = await verify(world, row, session_id)
    assert world.timeline[-2:] == ["collect", "tasks/verify"]
    (backend_verify,) = world.backends.to("/verify")
    # The backend's own flat verify: its seeded task fields, the actor's request and the response.
    assert set(backend_verify["json"]) == {
        "instance_id",
        "workdir",
        "image_url",
        "responses_create_params",
        "response",
    }
    assert backend_verify["cookies"] == {"backend": "cookie"}
    assert result.reward == 1.0 and result.mask_sample is False and result.evaluation_completed is True
    assert result.failure_kind is None and result.execution_status == "completed"
    assert result.if_result.if_all == 1 and result.evidence.status == "complete"
    assert result.instance_id == "task-a"  # the backend's fields stay at the top level
    directory = Path(result.deliverables_dir)
    assert directory.parent == world.artifacts and (directory / "result.json").is_file()
    record = load_episode_record(directory)
    assert record.backend.result["reward"] == 1.0 and record.verdict.reward == 1.0
    assert record.evidence.facts["workspace_initial_diff"] == "" and record.evidence.facts["workdir"] == "/app"
    assert record.seed["profile_sha256"] == PROFILE
    # The bundle's files are kept as read, and the evidence names them.
    calls = directory / "bundle" / "receipts" / "model-calls.jsonl"
    assert json.loads(calls.read_text().splitlines()[0])["assistant_message_id"] == "a1"
    assert result.evidence.artifacts["model_calls"] == str(calls)
    # The raw OpenCode export beside its receipt-enriched copy, which the evidence was built from.
    assert result.evidence.artifacts["raw_export:ses_root"] == str(directory / "bundle" / "exports" / "ses_root.json")
    assert result.evidence.artifacts["ses_root"] == str(directory / "export-ses_root-with-if-receipts.json")
    assert world.expected == [record.seed["bundle"]]
    # An identical retry replays the result; another body for the same session conflicts.
    assert (await verify(world, row, session_id)) == result and len(world.backends.to("/verify")) == 1
    with pytest.raises(HTTPException) as conflict:
        await verify(world, row, session_id, answer_response=response(status="incomplete"))
    assert conflict.value.status_code == 409


@pytest.mark.asyncio
async def test_verify_refuses_another_task_and_rejects_a_broken_receipt(world):
    row = swe_row()
    _, session_id = await seed(world, row)
    with pytest.raises(HTTPException) as other_task:
        await verify(world, row | {"instance_id": "other"}, session_id)
    assert other_task.value.status_code == 409
    tampered = row | {"responses_create_params": {"input": [{"role": "user", "content": "Tampered"}]}}
    with pytest.raises(HTTPException) as receipt:
        await verify(world, tampered, session_id)
    assert receipt.value.status_code == 422
    assert not world.backends.to("/verify")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case, kind, reward, completed",
    [
        ("collection_failed", "session_lost", 1.0, False),
        ("backend_failed", "verifier_error", 0.0, False),
        ("backend_incomplete", "verifier_error", 0.0, False),
        ("backend_masked", "scale_swe:patch_capture_failed", 0.0, False),
        ("no_load_receipt", "agentic_if:evidence_incomplete", 1.0, False),
        ("bundle_gap", "agentic_if:evidence_incomplete", 1.0, False),
        ("agent_error", "model_request_error", 1.0, True),
    ],
)
async def test_infrastructure_failures_are_masked_scored_rows(world, case, kind, reward, completed):
    row = swe_row()
    answer = None
    if case == "collection_failed":
        world.outputs = OSError("sandbox is gone")
    elif case == "backend_failed":
        world.backends.verify_error = http_error(503)
    elif case == "backend_incomplete":
        world.backends.verdict = {"reward": 0.0, "evaluation_completed": False, "error": "tests did not run"}
    elif case == "backend_masked":
        world.backends.verdict = {
            "reward": 0.0,
            "evaluation_completed": False,
            "mask_sample": True,
            "failure_kind": "scale_swe:patch_capture_failed",
            "failure_reason": "could not extract the patch",
        }
    elif case == "no_load_receipt":
        world.outputs.hook_receipts = []
    elif case == "bundle_gap":
        world.outputs.gaps = ["receipts_unreadable"]
    else:
        outcome = json.dumps({"failure_kind": "model_request_error", "status_code": 400})
        answer = response(status="failed", metadata={AGENT_OUTCOME_METADATA_KEY: outcome})
    _, session_id = await seed(world, row)
    result = await verify(world, row, session_id, answer_response=answer)
    assert result.mask_sample is True and result.failure_kind == kind and result.failure_reason
    assert result.reward == reward and result.evaluation_completed is completed
    assert load_episode_record(Path(result.deliverables_dir)).verdict.failure_kind == kind
    if case in ("no_load_receipt", "bundle_gap", "collection_failed"):
        assert result.if_result.if_all is None  # incomplete evidence is never graded as a pass or a failure
    # Only a measured result is the session's verification result. A retry after a missing measurement gets a 503:
    # the first attempt used the agent's sandbox, so the episode itself must be retried.
    retry = verify(world, row, session_id, answer_response=answer)
    if completed:
        assert (await retry) == result
    else:
        with pytest.raises(HTTPException) as refused:
            await retry
        assert refused.value.status_code == 503
        assert refused.value.detail == "Previous verification did not complete; retry the episode"
    with pytest.raises(HTTPException) as conflict:
        await verify(world, row, session_id, answer_response=response(status="incomplete"))
    assert conflict.value.status_code == 409
    assert len(world.backends.to("/verify")) == 1 and world.timeline.count("collect") == 1


@pytest.mark.asyncio
async def test_a_verify_that_failed_before_recording_its_episode_is_not_repeated(world, monkeypatch):
    def unwritable(path, payload):
        raise OSError("No space left on device")

    monkeypatch.setattr(app_module, "save_json", unwritable)
    row = swe_row()
    _, session_id = await seed(world, row)
    with pytest.raises(OSError):
        await verify(world, row, session_id)
    # The bundle was read and the task verified: the agent's sandbox is spent, so the episode is retried instead.
    with pytest.raises(HTTPException) as refused:
        await verify(world, row, session_id)
    assert refused.value.status_code == 503
    assert len(world.backends.to("/verify")) == 1 and world.timeline.count("collect") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", ["unfinished", "finished"])
async def test_a_harness_timeout_is_scored_with_agent_timeout(world, tail):
    session = finished_session(finish="tool-calls")
    if tail == "unfinished":
        session["messages"][1]["parts"].append(call("c9", "bash", status="running", input={"command": "sleep 99"}))
    world.outputs = bundle_outputs([session], call_records=[final_call_record()])
    row = swe_row()
    _, session_id = await seed(world, row)
    result = await verify(world, row, session_id, answer_response=response(status="incomplete"))
    assert result.execution_status == "timeout"
    assert result.mask_sample is False and result.evaluation_completed is True and result.reward == 1.0
    assert result.failure_kind == "agent_timeout"
    assert (result.if_result.if_all is None) is (tail == "unfinished")
    # A scored timeout is a measured result: an identical retry replays it.
    assert (await verify(world, row, session_id, answer_response=response(status="incomplete"))) == result
    assert len(world.backends.to("/verify")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing, issue",
    [
        ("profile_load_receipt", "hook_receipt_error:missing live hook load receipt"),
        ("boundary_load_receipt", "bundle_gap:boundary_load_receipt_missing"),
        ("system_prompt", "system_prompt_not_presented:ses_root:a1"),
        ("literal_task", "literal_task_not_presented:ses_root"),
    ],
)
@pytest.mark.parametrize("stop", ["timeout", "context_limit"])
async def test_a_limit_stop_without_the_plugins_receipts_or_the_row_as_written_is_masked(world, missing, issue, stop):
    error = {"name": "ContextOverflowError"} if stop == "context_limit" else None
    finish = "tool-calls" if stop == "timeout" else "stop"
    session = finished_session(finish=finish, error=error)
    row = swe_row(system="Answer in one word.")
    outputs = presented(row)
    calls = outputs.call_records
    if missing == "profile_load_receipt":
        world.outputs = bundle_outputs([session], hook_receipts=[], call_records=calls)
    elif missing == "boundary_load_receipt":
        world.outputs = bundle_outputs([session], call_records=calls, gaps=["boundary_load_receipt_missing"])
    elif missing == "system_prompt":
        world.outputs = bundle_outputs([session], call_records=[final_call_record()])
    else:
        session = finished_session(finish=finish, error=error, task="Fix something else")
        world.outputs = bundle_outputs([session], call_records=calls, boundary_receipts=outputs.boundary_receipts)
    _, session_id = await seed(world, row)
    result = await verify(world, row, session_id, answer_response=response(status="incomplete"))
    assert result.mask_sample is True and result.failure_kind == "agentic_if:evidence_incomplete"
    assert issue in result.evidence.issues


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, finish, error, expected",
    [
        ("completed", "stop", None, "completed"),
        ("incomplete", "length", None, "length"),
        ("incomplete", "content-filter", None, "completed"),
        ("incomplete", "stop", {"name": "ContextOverflowError"}, "error"),
        ("failed", "stop", None, "error"),
    ],
)
async def test_execution_status_reads_the_response_and_the_root_export(world, status, finish, error, expected):
    world.outputs = bundle_outputs([finished_session(finish=finish, error=error)], call_records=[final_call_record()])
    row = swe_row()
    _, session_id = await seed(world, row)
    result = await verify(world, row, session_id, answer_response=response(status=status))
    assert result.execution_status == expected
    # A context-window stop is the model's own limit: scored, and named.
    assert result.mask_sample is (expected == "error" and error is None)
    if error is not None:
        assert result.failure_kind == "model_context_limit" and result.evaluation_completed is True
    # Each of these is measured, the harness error included: an identical retry replays it.
    assert result.evaluation_completed is True
    assert (await verify(world, row, session_id, answer_response=response(status=status))) == result
    assert len(world.backends.to("/verify")) == 1


@pytest.mark.asyncio
async def test_a_lost_session_is_a_masked_row(world):
    row = swe_row()
    result = await verify(world, row, "never-seeded")
    assert result.mask_sample is True and result.failure_kind == "session_lost" and result.reward == 0.0
    assert result.evaluation_completed is False and result.if_result is None
    assert not world.backends.to("/verify")


# ------------------------------------------------------------------------------------------------------ judge


@pytest.mark.asyncio
async def test_an_unreachable_judge_raises_judge_error_after_persisting_the_episode(world):
    world.backends.judge_error = ClientConnectionError("judge unreachable")
    row = swe_row(checks=JUDGED)
    _, session_id = await seed(world, row)
    body = AgenticIFVerifyRequest.model_validate(verify_body(row))
    with pytest.raises(JudgeError, match="could not be reached"):
        await world.server.verify(caller(session_id), body)
    # judge_failsafe answers with this request: it names the persisted episode, which re-verification re-grades.
    assert body.deliverables_dir is not None
    record = load_episode_record(Path(body.deliverables_dir))
    assert record.verdict.reward == 1.0
    # The calls that got no reply are kept with the episode: no reply ID, no tokens, and why.
    attempts = [call for call in world.backends.calls if call["server_name"] == "judge"]
    assert len(record.judge_calls) == len(attempts) > 0
    assert all(call.response_id is None and call.total_tokens is None for call in record.judge_calls)
    assert {call.error for call in record.judge_calls} == {"ClientConnectionError"}
    world.backends.judge_error = None
    regraded = await world.server.verify(
        caller(), AgenticIFVerifyRequest.model_validate(verify_body(row, deliverables_dir=body.deliverables_dir))
    )
    assert regraded.if_result.if_all == 1 and regraded.reward == 1.0
    assert len(world.backends.to("/verify")) == 1  # the task is never verified again
    # A session-less re-grade reports its own calls, and leaves the episode record as the live verify left it.
    assert regraded.judge_usage.calls == len(regraded.judge_calls) > 0 and regraded.judge_usage.failed_calls == 0
    assert load_episode_record(Path(body.deliverables_dir)).judge_calls == record.judge_calls


@pytest.mark.asyncio
async def test_a_verify_retried_after_a_judge_outage_grades_the_episode_again(world):
    world.backends.judge_error = ClientConnectionError("judge unreachable")
    row = swe_row(checks=JUDGED)
    _, session_id = await seed(world, row)
    with pytest.raises(JudgeError):
        await verify(world, row, session_id)
    unanswered = len([call for call in world.backends.calls if call["server_name"] == "judge"])
    world.backends.judge_error = None
    result = await verify(world, row, session_id)
    assert result.if_result.if_all == 1 and result.mask_sample is False
    assert len(world.backends.to("/verify")) == 1 and world.timeline.count("collect") == 1
    # The result reports this grade's calls; the record keeps both grades' calls, in order.
    assert result.judge_usage.failed_calls == 0 and all(call.response_id for call in result.judge_calls)
    record = load_episode_record(Path(result.deliverables_dir))
    assert [call.error for call in record.judge_calls] == ["ClientConnectionError"] * unanswered + [None] * len(
        result.judge_calls
    )
    assert (await verify(world, row, session_id)) == result  # measured now, so a retry replays it


@pytest.mark.asyncio
async def test_each_judge_call_s_reply_id_and_usage_are_reported_and_kept_with_the_episode(world):
    row = swe_row(checks=JUDGED)
    _, session_id = await seed(world, row)
    result = await verify(world, row, session_id)
    assert result.if_result.if_all == 1
    sent = [call for call in world.backends.calls if call["server_name"] == "judge"]
    n = len(sent)
    assert n > 0 and [call.response_id for call in result.judge_calls] == [f"judge-reply-{i}" for i in range(1, n + 1)]
    row_view = result.model_dump(mode="json")
    # What lands in rollouts.jsonl: each call's reply ID, the model that answered and tokens, and their totals.
    assert row_view["judge_calls"][0] == {
        "response_id": "judge-reply-1",
        "model": "azure/openai/gpt-6-sol",
        "input_tokens": 1200,
        "output_tokens": 300,
        "reasoning_tokens": 120,
        "total_tokens": 1500,
    }
    assert result.judge_usage == JudgeUsage(
        calls=n, input_tokens=1200 * n, output_tokens=300 * n, reasoning_tokens=120 * n, total_tokens=1500 * n
    )
    # The episode record keeps them with their questions, and result.json with the result.
    directory = Path(result.deliverables_dir)
    record = load_episode_record(directory)
    assert [(call.response_id, call.prompt_sha256) for call in record.judge_calls] == [
        (f"judge-reply-{i}", hashlib.sha256(judge_prompt(call["json"]).encode()).hexdigest())
        for i, call in enumerate(sent, start=1)
    ]
    assert json.loads((directory / "result.json").read_text())["judge_usage"]["total_tokens"] == 1500 * n
    # A replayed result reports the calls of the grade behind it, without asking the judge again.
    assert (await verify(world, row, session_id)) == result
    assert len([call for call in world.backends.calls if call["server_name"] == "judge"]) == n


@pytest.mark.asyncio
async def test_a_call_that_got_no_reply_and_a_reply_without_usage_are_reported_as_such(world):
    world.backends.judge_error = [ClientConnectionError("judge unreachable")]  # the first call only
    world.backends.judge_usage = None
    row = swe_row(checks=JUDGED)
    _, session_id = await seed(world, row)
    result = await verify(world, row, session_id)
    assert result.if_result.if_all == 1 and result.mask_sample is False
    assert [call.response_id for call in result.judge_calls][:2] == [None, "judge-reply-1"]
    usage = result.judge_usage
    assert (usage.failed_calls, usage.calls_without_usage, usage.total_tokens) == (1, usage.calls - 1, 0)
    record = load_episode_record(Path(result.deliverables_dir))
    assert record.judge_calls[0].error == "ClientConnectionError" and record.judge_calls[1].error is None


def test_the_verify_route_records_a_judge_outage_as_judge_failed(world):
    world.backends.judge_error = ClientConnectionError("judge unreachable")
    row = swe_row(checks=JUDGED)
    client = TestClient(world.server.setup_webserver())
    assert client.post("/seed_session", json=seed_body(row).model_dump(mode="json")).status_code == 200
    reply = client.post("/verify", json=verify_body(row))
    assert reply.status_code == 200
    failed = reply.json()
    assert failed["failure_kind"] == "judge_failed" and failed["mask_sample"] is True
    assert failed["_ng_failure_class"] == "judge_failed"
    assert Path(failed["deliverables_dir"]).parent == world.artifacts
    record = load_episode_record(Path(failed["deliverables_dir"]))
    assert record.judge_calls and {call.error for call in record.judge_calls} == {"ClientConnectionError"}


@pytest.mark.asyncio
async def test_an_unparseable_judge_reply_stays_a_graded_check_error(world):
    world.backends.judge_text = "I cannot decide."
    row = swe_row(checks=JUDGED)
    _, session_id = await seed(world, row)
    result = await verify(world, row, session_id)
    assert result.mask_sample is False and result.if_result.checks[0].status == "error"
    # Each attempt got a reply, with its ID and tokens.
    assert result.judge_usage.calls == load_config()["model"]["attempts"] and result.judge_usage.failed_calls == 0
    assert all(call.response_id and call.total_tokens for call in result.judge_calls)


@pytest.mark.asyncio
async def test_a_masked_row_is_not_turned_into_judge_failed(world):
    world.backends.judge_error = ClientConnectionError("judge unreachable")
    world.backends.verify_error = http_error(503)
    row = swe_row(checks=JUDGED)
    _, session_id = await seed(world, row)
    result = await verify(world, row, session_id)
    assert result.failure_kind == "verifier_error" and result.mask_sample is True


@pytest.mark.asyncio
async def test_a_judged_check_asks_the_configured_judge_model_server(world):
    row = swe_row(checks=JUDGED)
    _, session_id = await seed(world, row)
    result = await verify(world, row, session_id)
    assert result.if_result.if_all == 1
    judged = [call for call in world.backends.calls if call["server_name"] == "judge"]
    model = load_config()["model"]
    assert judged and all(call["url_path"] == "/v1/responses" for call in judged)
    assert judged[0]["json"]["model"] == model["name"] and judged[0]["json"]["temperature"] == model["temperature"]


@pytest.mark.asyncio
async def test_judge_calls_stay_out_of_the_rollouts_model_call_capture(world):
    # A rollout's captured model calls each need an agent invocation owner; the judge is the verifier's model.
    row = swe_row(checks=JUDGED)
    with rollout_context("rollout-1"):
        _, session_id = await seed(world, row)
        result = await verify(world, row, session_id)
    assert result.if_result.if_all == 1
    judged = [call for call in world.backends.calls if call["server_name"] == "judge"]
    assert judged and all(call["rollout_id"] is None for call in judged)
    # The task backend's calls stay attributed to the rollout.
    assert {call["rollout_id"] for call in world.backends.to("/verify")} == {"rollout-1"}


def test_the_judge_outage_ledger_uses_the_judge_s_own_vote_key():
    judge = judge_module.Judge(lambda body: None)
    assert vote_key(judge.hash, "question", 2) == judge._key("question", 2)


# --------------------------------------------------------------------------------------------- stateless re-verify


@pytest.mark.asyncio
async def test_a_session_less_verify_regrades_persisted_evidence_and_keeps_the_task_verdict(world):
    row = swe_row()
    _, session_id = await seed(world, row)
    live = await verify(world, row, session_id)
    await world.server.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id=session_id, episode_id=EPISODE)
    )
    regraded = await world.server.verify(
        caller(), AgenticIFVerifyRequest.model_validate(verify_body(row, deliverables_dir=live.deliverables_dir))
    )
    assert regraded.reward == live.reward and regraded.if_result.if_all == live.if_result.if_all
    assert regraded.evidence == live.evidence and regraded.instance_id == "task-a"
    assert len(world.backends.to("/verify")) == 1 and world.timeline.count("collect") == 1
    with pytest.raises(HTTPException) as closed:
        await verify(world, row, session_id)
    assert closed.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deliverables, row_change, status",
    [("../outside", {}, 422), ("missing", {}, 404), (None, {"instance_id": "other"}, 409)],
)
async def test_a_session_less_verify_only_reads_its_own_persisted_episode(world, deliverables, row_change, status):
    row = swe_row()
    _, session_id = await seed(world, row)
    live = await verify(world, row, session_id)
    body = verify_body(row | row_change, deliverables_dir=deliverables or live.deliverables_dir)
    with pytest.raises(HTTPException) as refused:
        await world.server.verify(caller(), AgenticIFVerifyRequest.model_validate(body))
    assert refused.value.status_code == status


# --------------------------------------------------------------------------------------------- configuration


def test_a_session_server_refuses_more_than_one_worker(tmp_path):
    with pytest.raises(ValueError, match="num_workers"):
        make_server(tmp_path, Backends(), num_workers=2)


def test_the_server_declares_stateless_reverification_and_no_ray():
    assert AgenticIFResourcesServerConfig.REVERIFY_MODE.value == "stateless"
    assert AgenticIFResourcesServer.ray_enabled is False


def test_the_verify_request_keeps_the_backend_fields_and_requires_the_contract_fields():
    row = tb21_row()
    body = AgenticIFVerifyRequest.model_validate(verify_body(row))
    assert body.model_extra == {
        "task_name": "terminal-bench/a",
        "docker_image": "registry/a:1",
        "task_folder": "tasks/a",
    }
    for missing in ("response", "agentic_if", "responses_create_params"):
        with pytest.raises(ValidationError, match=missing):
            AgenticIFVerifyRequest.model_validate(
                {key: value for key, value in verify_body(row).items() if key != missing}
            )


# ---------------------------------------------------------------------------------------------- metrics, data


def test_metrics_report_if_rates_beside_the_task_reward(world):
    def rollout(reward, if_all, if_frac, statuses):
        checks = [{"id": str(index), "status": status, "explanation": ""} for index, status in enumerate(statuses)]
        return {"reward": reward, "if_result": {"checks": checks, "if_all": if_all, "if_frac": if_frac}}

    metrics = world.server.compute_metrics(
        [
            [rollout(1.0, 1, 1.0, ["pass"]), rollout(0.0, 1, 1.0, ["pass"])],
            [rollout(1.0, 0, 0.5, ["pass", "fail"])],
            [rollout(1.0, None, None, ["not_applicable"]), rollout(0.0, None, None, ["error"])],
        ]
    )
    assert metrics["if/graded"] == 3 and metrics["if/not_applicable"] == 1 and metrics["if/error"] == 1
    assert metrics["mean/if_all"] == pytest.approx(2 / 3)
    assert metrics["mean/if_frac"] == pytest.approx(2.5 / 3)
    assert metrics["mean/task_and_if"] == pytest.approx(1 / 3)


def test_task_data_selects_the_rows_own_backend_schema_and_requires_its_digest_identity():
    schema = TypeAdapter(TaskData)
    checks = [{"id": "end", "kind": "exact", "parameters": {"value": "DONE"}, "scope": {"places": ["final"]}}]
    swe = seal_runtime_row(
        {"instance_id": "task-a", "workdir": "/testbed", "image_url": "registry/task-a:1"}
        | {"responses_create_params": {"input": [{"role": "user", "content": "Fix it"}]}},
        RuntimeSpec(harness="opencode", task_backend="swe", checks=checks),
    )
    tb21 = seal_runtime_row(
        {"task_name": "terminal-bench/a", "docker_image": "registry/a:1", "task_folder": "tasks/a"}
        | {"workdir": "/app", "responses_create_params": {"input": [{"role": "user", "content": "Fix it"}]}},
        RuntimeSpec(harness="opencode", task_backend="tb21", checks=checks),
    )
    assert type(schema.validate_python({"task_id": digest(swe)} | swe)).__name__ == "SWETask"
    assert type(schema.validate_python({"task_id": digest(tb21)} | tb21)).__name__ == "TerminalBench21Task"
    with pytest.raises(ValidationError, match="task_folder"):
        schema.validate_python({"task_id": digest(swe)} | swe | {"agentic_if": tb21["agentic_if"]})
    with pytest.raises(ValidationError, match="task_id"):
        schema.validate_python({"task_id": "not-a-digest"} | swe)


def test_committed_example_rows_validate_clean_and_carry_their_digest():
    example = Path(__file__).parents[1] / "data" / "example.jsonl"
    lines = example.read_text().splitlines()
    report = validate_jsonl_rows("agentic_if", TypeAdapter(TaskData), str(example), lines)
    assert report.rows == len(lines) and report.clean, report.summary()
    for row in map(json.loads, lines):
        content = {key: value for key, value in row.items() if key != "task_id"}
        assert row["task_id"] == digest(content)
        validate_runtime(content)


@pytest.mark.asyncio
async def test_every_example_row_seeds_and_verifies_as_single_agent_turn_sends_it(world):
    example = Path(__file__).parents[1] / "data" / "example.jsonl"
    for index, row in enumerate(map(json.loads, example.read_text().splitlines())):
        reply, session_id = await seed(world, row, session=f"example-{index}")
        assert reply.sandbox_access is not None
        world.outputs = presented(row)
        result = await verify(world, row, session_id)
        assert result.agentic_if["task_backend"] == row["agentic_if"]["task_backend"]
        assert result.mask_sample is False and result.evidence.status == "complete", (index, result.evidence.issues)


# --------------------------------------------------------------------------------------- what the model was shown


@pytest.mark.asyncio
@pytest.mark.parametrize("system", [None, "Answer in one word."])
async def test_a_row_shown_to_the_model_as_written_is_graded(world, system):
    row = swe_row(system=system)
    _, session_id = await seed(world, row)
    world.outputs = presented(row)
    result = await verify(world, row, session_id)
    assert result.mask_sample is False and result.evidence.status == "complete", result.evidence.issues
    assert result.if_result.if_all == 1


def _sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change, issue",
    [
        # The agent added its own system prompt to the row's, so the boundary was configured with another text.
        ("agent_system_prompt", "boundary_system_mismatch"),
        ("call_without_the_system_prompt", "system_prompt_not_presented:ses_root:a1"),
        ("call_with_a_second_system_message", "system_prompt_not_presented:ses_root:a1"),
        ("another_task", "literal_task_not_presented:ses_root"),
    ],
)
async def test_a_model_not_shown_the_row_as_written_masks_the_episode(world, change, issue):
    row = swe_row(system="Answer in one word.")
    _, session_id = await seed(world, row)
    outputs = presented(row)
    if change == "agent_system_prompt":
        outputs.boundary_receipts = [
            {"kind": "configured", "system_sha256": _sha256("Be kind.\n\nAnswer in one word.")}
        ]
    elif change == "call_without_the_system_prompt":
        outputs.call_records = [final_call_record()]
    elif change == "call_with_a_second_system_message":
        messages = outputs.call_records[0]["request"]["messages"]
        outputs.call_records[0]["request"]["messages"] = (
            messages[:1] + [{"role": "developer", "content": "Hi"}] + messages[1:]
        )
    else:
        outputs = bundle_outputs(
            [finished_session(task="Fix something else")],
            call_records=outputs.call_records,
            boundary_receipts=outputs.boundary_receipts,
        )
    world.outputs = outputs
    result = await verify(world, row, session_id)
    assert result.mask_sample is True and result.failure_kind == "agentic_if:evidence_incomplete"
    assert result.evidence.status == "incomplete" and issue in result.evidence.issues
    assert result.if_result.if_all is None


@pytest.mark.asyncio
async def test_a_row_without_a_system_message_needs_an_agent_without_its_own(world):
    row = swe_row()
    _, session_id = await seed(world, row)
    world.outputs = presented(row)
    world.outputs.boundary_receipts = [{"kind": "configured", "system_sha256": _sha256("Be kind.")}]
    result = await verify(world, row, session_id)
    assert result.mask_sample is True and "boundary_system_mismatch" in result.evidence.issues


# ------------------------------------------------------------------------------------------------- continuation


@pytest.mark.asyncio
async def test_a_continuation_row_replays_its_prefix_at_seed_and_grades_from_the_fresh_turn(
    world, tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (repo / "file").write_text("before")
    git("add", ".")
    git("commit", "-qm", "base")
    (repo / "file").write_text("changed by the prefix")
    diff = git("diff", "--binary", "--full-index")
    git("checkout", "--", ".")
    recorded = call(
        "p-call",
        "bash",
        input={"command": "printf replayed"},
        output="replayed",
        metadata={"nemo_gym_profile": {"workspace_diff": diff}},
    )
    prefix = export(
        message("p-user", "user", text("Fix it"), session="ses_prefix"),
        message("p-reply", "assistant", recorded, session="ses_prefix", created=2),
        session="ses_prefix",
    )
    fields = {"instance_id": "task-a", "workdir": str(repo), "image_url": "registry/task-a:1"}
    tools = RuntimeSpec(harness="opencode", task_backend="swe").tools.model_dump(mode="json")
    continuation = {
        "mode": "restore",
        "prefix_export": prefix,
        "workspace_diff": diff,
        "source_task_sha256": backend_task_digest(fields),
        "source_tools_sha256": digest(tools),
    }
    checks = [
        {
            "id": "end",
            "kind": "exact",
            "parameters": {"value": "DONE"},
            "scope": {"places": ["final"], "activation": "continuation_start"},
        }
    ]
    row = swe_row(fields=fields, continuation=continuation, checks=checks, text="Continue")
    local = LocalTransport()

    async def connect(descriptor, *, provider):
        return local

    monkeypatch.setattr(resources.AsyncSandbox, "connect", connect)
    monkeypatch.setattr(opencode_bundle, "SESSION_IMPORT_PATH", str(tmp_path / "bundle" / "prefix-export.json"))
    reply, session_id = await seed(world, row)
    assert reply.sandbox_access.workdir == str(repo)
    assert (repo / "file").read_text() == "changed by the prefix"
    assert json.loads((tmp_path / "bundle" / "prefix-export.json").read_text()) == prefix
    # The bundle is staged after the replay, and the import file after the bundle.
    assert world.timeline.index("stage") > world.timeline.index("tasks/seed_session")

    root = deepcopy(prefix)
    root["messages"] += [
        message("fresh-user", "user", text("Continue"), session="ses_prefix", created=10),
        message("fresh-reply", "assistant", text("DONE"), session="ses_prefix", created=11),
    ]
    fresh = final_call_record(session="ses_prefix", message_id="fresh-reply")
    world.outputs = bundle_outputs([root], call_records=[fresh])
    result = await verify(world, row, session_id)
    assert result.mask_sample is False and result.evidence.status == "complete", result.evidence.issues
    assert result.evidence.continuation_start == 3
    assert result.evidence.facts["replay"] == {
        "mode": "restore",
        "replayed_commands": 1,
        "command_output_mismatch_indices": [],
        "workspace_matches_diff": True,
        "prefix_messages": 2,
        "tool_outputs_match": True,
    }
    assert result.if_result.if_all == 1


# ------------------------------------------------------------------------------------- with the real plugin bundle


@pytest.fixture
def real_bundle(world, monkeypatch):
    """The plugin bundle's own staging and collection, over the in-memory task sandbox."""
    from resources_servers.agentic_if.opencode_bundle import bundle as implementation

    monkeypatch.setattr(opencode_bundle, "stage_bundle", implementation.stage_bundle)
    monkeypatch.setattr(opencode_bundle, "collect_bundle", implementation.collect_bundle)
    return world


def leave_bundle_outputs(sandbox, staged, *, boundary_loaded=True, system=None, configured_with=None):
    """Write what OpenCode and the two plugins leave in the bundle when the run ends, as the real ones do.

    ``system`` is the system prompt the model calls carried; ``configured_with`` the instructions the boundary read
    (``system`` unless given).
    """
    root = opencode_bundle.BUNDLE_ROOT
    files = staged["files"]
    session = finished_session()
    hooks = [
        {"kind": "loaded", "profile_sha256": staged["profile_sha256"]},
        {"kind": "plugin", "plugin": "if-profile", "plugin_sha256": files[f"{root}/if-profile.mjs"]},
    ]
    configured_with = system if configured_with is None else configured_with
    record = final_call_record()
    if system is not None:
        record["request"]["messages"].insert(0, {"role": "system", "content": system})
    calls = [
        {
            "kind": "configured",
            "provider": "nemo_gym",
            "base_url": "http://policy",
            "denied": [],
            "instructions": [] if configured_with is None else ["/tmp/session/instructions.md"],
            "system_sha256": None if configured_with is None else _sha256(configured_with),
        },
        {"kind": "request", "call": "1", "observation": {k: v for k, v in record.items() if k != "response"}},
        {"kind": "response", "call": "1", "response": record["response"]},
    ]
    if boundary_loaded:
        loaded = {"kind": "loaded", "plugin": "if-boundary", "config_sha256": staged["boundary_sha256"]}
        calls.insert(0, loaded | {"plugin_sha256": files[f"{root}/if-boundary.mjs"], "pid": 1})
    sandbox.uploads |= {
        f"{root}/exports/sessions.json": json.dumps([{"id": "ses_root", "parent_id": None}]).encode(),
        f"{root}/exports/ses_root.json": json.dumps(session).encode(),
        f"{root}/receipts/hook-applied.jsonl": "".join(json.dumps(row) + "\n" for row in hooks).encode(),
        f"{root}/receipts/hook-events-fired.jsonl": b"",
        f"{root}/receipts/model-calls.jsonl": "".join(json.dumps(row) + "\n" for row in calls).encode(),
    }


@pytest.mark.asyncio
async def test_the_real_bundle_is_staged_at_seed_and_graded_at_verify(real_bundle):
    world = real_bundle
    row = swe_row()
    _, session_id = await seed(world, row)
    staged = world.server._episodes[session_id].seed["bundle"]
    root = opencode_bundle.BUNDLE_ROOT
    assert set(staged["files"]) >= {f"{root}/if-profile.mjs", f"{root}/if-boundary.mjs", f"{root}/profile.json"}
    assert staged["owner"] == "1001:1001"
    leave_bundle_outputs(world.sandbox, staged)
    result = await verify(world, row, session_id)
    assert result.evidence.status == "complete", result.evidence.issues
    assert result.mask_sample is False and result.if_result.if_all == 1
    raw = Path(result.deliverables_dir) / "bundle" / "receipts" / "model-calls.jsonl"
    assert raw.read_bytes() == world.sandbox.uploads[f"{root}/receipts/model-calls.jsonl"]


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_with, masked", [(None, False), ("Be kind.\n\nAnswer in one word.", True)])
async def test_the_real_bundle_checks_the_system_prompt_the_boundary_read(real_bundle, configured_with, masked):
    world = real_bundle
    row = swe_row(system="Answer in one word.")
    _, session_id = await seed(world, row)
    staged = world.server._episodes[session_id].seed["bundle"]
    leave_bundle_outputs(world.sandbox, staged, system="Answer in one word.", configured_with=configured_with)
    result = await verify(world, row, session_id)
    assert result.mask_sample is masked, result.evidence.issues
    assert ("boundary_system_mismatch" in result.evidence.issues) is masked


@pytest.mark.asyncio
async def test_a_boundary_that_did_not_load_masks_the_episode(real_bundle):
    world = real_bundle
    row = swe_row()
    _, session_id = await seed(world, row)
    leave_bundle_outputs(world.sandbox, world.server._episodes[session_id].seed["bundle"], boundary_loaded=False)
    result = await verify(world, row, session_id)
    assert result.mask_sample is True and result.failure_kind == "agentic_if:evidence_incomplete"
    assert "bundle_gap:boundary_load_receipt_missing" in result.evidence.issues
