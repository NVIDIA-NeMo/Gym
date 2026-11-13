# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session ownership and grading retries with an in-process sandbox stub."""

import asyncio
import json
import shlex
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
)
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.pdf_analyst_qa.app import (
    VERIFIER_FIXTURE,
    PdfAnalystQaResourcesServer,
    PdfAnalystQaResourcesServerConfig,
    PdfAnalystQaVerifyRequest,
    _SessionState,
)
from resources_servers.pdf_analyst_qa.documents import MountedDocumentsConfig


ROOT = Path(__file__).resolve().parents[1] / "data/example_bundle"
TASK_ID = "task_111111111111111111111111"
WORKSPACE_ID = "workspace_111111111111111111111111"
START_SANDBOX = PdfAnalystQaResourcesServer._start_sandbox


def request():
    return MagicMock(session={})


def seed(session_id="session-1", rollout_id="rollout-1"):
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=rollout_id),
        task_id=TaskId(taskset="pdf_analyst_qa:example", task_id=TASK_ID),
        task_data={"task_id": TASK_ID, "workspace_id": WORKSPACE_ID, "prompt_version": "example-document-v1"},
    )


def verify_body(answer="23"):
    response = {
        "id": "response-1",
        "object": "response",
        "created_at": 0,
        "model": "policy_model",
        "output": [
            {
                "id": "message-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": answer, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    return PdfAnalystQaVerifyRequest.model_validate(
        {
            "task_id": TASK_ID,
            "workspace_id": WORKSPACE_ID,
            "prompt_version": "example-document-v1",
            "responses_create_params": {"input": []},
            "response": response,
        }
    )


@pytest.fixture
def server(monkeypatch):
    config = PdfAnalystQaResourcesServerConfig(
        host="127.0.0.1",
        port=9000,
        name="pdf_analyst_qa",
        entrypoint="app.py",
        dataset_root=str(ROOT),
        document_image="example@sha256:" + "a" * 64,
        model_server={"type": "responses_api_models", "name": "policy_model"},
        mounted_documents=MountedDocumentsConfig(prefix="example"),
        max_active_sandboxes=1,
    )
    client = MagicMock()
    client.global_config_dict = {}
    result = PdfAnalystQaResourcesServer.model_construct(config=config, server_client=client)
    sandboxes = []

    async def start(server, state, workspace_id):
        assert workspace_id == WORKSPACE_ID
        sandbox = MagicMock()
        sandbox.exec = AsyncMock(return_value=SandboxExecResult(stdout="", stderr="", return_code=0))
        sandbox.serialize = AsyncMock(return_value={"sandbox_id": str(len(sandboxes))})
        sandbox.stop = AsyncMock()
        state.sandbox = sandbox
        sandboxes.append(sandbox)

    monkeypatch.setattr(PdfAnalystQaResourcesServer, "_start_sandbox", start)
    monkeypatch.setattr(
        "resources_servers.pdf_analyst_qa.app.sandbox_server_url", lambda *_args, **_kwargs: "http://127.0.0.1:8000"
    )
    return result, sandboxes


@pytest.mark.asyncio
async def test_seed_retries_reuse_one_sandbox_and_close_without_cookie(server):
    resource, sandboxes = server
    first = await resource.seed_session(request(), seed())
    again = await resource.seed_session(request(), seed())
    assert first == again
    assert len(sandboxes) == 1
    assert first.sandbox_access.workdir == "/workspace/task"
    assert first.sandbox_access.user == 65534
    preflight = shlex.split(sandboxes[0].exec.await_args.args[0])
    assert preflight == [
        "/opt/venv/bin/python",
        "/tmp/pdf-analyst-qa-scripts/preflight.py",
        resource.config.mounted_documents.mount_root,
        "http://127.0.0.1:8000",
    ]
    sandboxes[0].serialize.assert_awaited_with(scope="operate")
    close = ResourcesCloseSessionRequest(resources_session_id="session-1", episode_id=seed().episode_id)
    await resource.close_resources_session(close)
    await resource.close_resources_session(close)
    sandboxes[0].stop.assert_awaited_once()
    with pytest.raises(Exception):
        await resource.seed_session(request(), seed())


@pytest.mark.asyncio
async def test_correct_wrong_and_retry_verification(server):
    resource, sandboxes = server
    req = request()
    await resource.seed_session(req, seed())
    correct = await resource.verify(req, verify_body())
    retry = await resource.verify(req, verify_body())
    assert (correct.reward, correct.evaluation_completed, correct.mask_sample) == (1.0, True, False)
    assert retry.reward == correct.reward
    sandboxes[0].stop.assert_awaited_once()
    await resource.seed_session(request(), seed("session-2", "rollout-2"))
    wrong_req = request()
    wrong_req.session[SESSION_ID_KEY] = "session-2"
    wrong = await resource.verify(wrong_req, verify_body("24"))
    assert (wrong.reward, wrong.evaluation_completed, wrong.mask_sample) == (0.0, True, False)
    missing = await resource.verify(request(), verify_body("23"))
    assert (missing.reward, missing.evaluation_completed, missing.mask_sample) == (0.0, False, True)


@pytest.mark.asyncio
async def test_failed_agent_response_is_not_a_scored_policy_example(server):
    resource, _ = server
    req = request()
    await resource.seed_session(req, seed())
    body = verify_body("23")
    body.response.status = "failed"
    result = await resource.verify(req, body)
    assert (result.reward, result.evaluation_completed, result.mask_sample) == (0.0, False, True)


@pytest.mark.asyncio
async def test_failed_stop_keeps_capacity_and_close_can_retry(server):
    resource, sandboxes = server
    await resource.seed_session(request(), seed())
    sandboxes[0].stop.side_effect = [RuntimeError("stop failed"), None]
    close = ResourcesCloseSessionRequest(resources_session_id="session-1", episode_id=seed().episode_id)
    with pytest.raises(RuntimeError, match="stop failed"):
        await resource.close_resources_session(close)
    assert resource.has_session_state("session-1")
    assert resource._sandbox_slots.locked()
    await resource.close_resources_session(close)
    assert not resource.has_session_state("session-1")
    assert not resource._sandbox_slots.locked()


@pytest.mark.asyncio
async def test_cancelled_initialization_releases_partial_allocation(server, monkeypatch):
    resource, sandboxes = server
    started = asyncio.Event()
    wait = asyncio.Event()

    async def hang(server, state, workspace_id):
        sandbox = MagicMock()
        sandbox.stop = AsyncMock()
        state.sandbox = sandbox
        sandboxes.append(sandbox)
        started.set()
        await wait.wait()

    monkeypatch.setattr(PdfAnalystQaResourcesServer, "_start_sandbox", hang)
    pending = asyncio.create_task(resource.seed_session(request(), seed()))
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not resource.has_session_state("session-1")
    sandboxes[0].stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_malformed_reference_is_rejected_before_sandbox_start(server, tmp_path):
    from environments.pdf_analyst_qa.prepare_example import prepare_example

    resource, sandboxes = server
    root = tmp_path / "runtime"
    prepare_example(root)
    task_file = root / "tasks" / f"{TASK_ID}.json"
    task = json.loads(task_file.read_text())
    task["final_answer"] = ""
    task_file.write_text(json.dumps(task))
    resource.config.dataset_root = str(root)
    with pytest.raises(ResourcesSeedRejectedError, match="reference answer is invalid"):
        await resource.seed_session(request(), seed())
    assert not sandboxes


@pytest.mark.asyncio
async def test_seed_rejects_task_id_mismatch_before_allocation(server):
    resource, sandboxes = server
    body = seed()
    body.task_id = TaskId(taskset="pdf_analyst_qa:example", task_id="task_other")
    with pytest.raises(ResourcesSeedRejectedError, match="TaskId differs"):
        await resource.seed_session(request(), body)
    assert not sandboxes


@pytest.mark.asyncio
async def test_verifier_fixture_covers_reference_and_format_variants():
    cases = await exercise_verifier_fixture(VERIFIER_FIXTURE, reward_range=(0, 1), determinism="stochastic")
    assert {case.name for case in cases} >= {
        "correct",
        "wrong",
        "empty",
        "wrapped-format",
        "tagged",
        "unavailable-reference",
        "malformed-identity",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("use_cache", [True, False])
async def test_provider_spec_stages_as_root_and_lends_selected_view(server, monkeypatch, tmp_path, use_cache):
    from environments.pdf_analyst_qa.prepare_example import prepare_example
    from resources_servers.pdf_analyst_qa import app

    resource, _ = server
    root = tmp_path / "runtime"
    prepare_example(root)
    resource.config.dataset_root = str(root)
    resource.config.sandbox_request_cpu = 0.75
    resource.config.sandbox_request_memory_mib = 2048
    resource.config.mounted_documents = MountedDocumentsConfig(
        prefix="example",
        node_cache_host_path="/var/lib/pdf-analyst-qa-cache/release-one" if use_cache else None,
    )
    resource.server_client.global_config_dict = {
        "sandbox": {"default_metadata": {"team": "test"}, "opensandbox": {"connection": {"domain": "sandbox.example"}}}
    }
    created = []

    class Sandbox:
        def __init__(self, provider):
            self.provider = provider
            self.start = AsyncMock()
            self.exec = AsyncMock(return_value=SandboxExecResult(stdout="staged", stderr="", return_code=0))
            self.stop = AsyncMock()
            self.set_default_user = MagicMock()
            created.append(self)

    monkeypatch.setattr(app, "create_provider", lambda config: config)
    monkeypatch.setattr(app, "AsyncSandbox", Sandbox)
    monkeypatch.setattr(app, "sandbox_server_url", lambda *_args, **_kwargs: "http://10.2.3.4:8000")
    state = _SessionState(resource._load_task(resource.seed_request_model(**seed().task_data)), "example-document-v1")
    await START_SANDBOX(resource, state, WORKSPACE_ID)
    sandbox = created[0]
    spec = sandbox.start.await_args.args[0]
    assert spec.metadata == {"team": "test", "workspace-id": WORKSPACE_ID}
    assert spec.provider_options["network_policy"] == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "10.2.3.4"}],
    }
    if use_cache:
        assert spec.provider_options["volumes"][1]["host"]["path"].endswith("/views/" + WORKSPACE_ID)
        assert spec.provider_options["volumes"][1]["readOnly"] is True
        assert spec.provider_options["volumes"][0]["readOnly"] is False
    assert spec.provider_options["resource_requests"] == {"cpu": 0.75, "memory_mib": 2048, "disk_gib": 10}
    assert spec.files and spec.env["OPENCODE_CONFIG"] == "/opt/cli/opencode-model.json"
    assert sandbox.exec.await_args.kwargs["user"] == 0
    sandbox.set_default_user.assert_called_once_with(65534)
    command = shlex.split(sandbox.exec.await_args.args[0])
    assert command[1] == f"{app.SANDBOX_SCRIPTS_ROOT}/documents.py"
    assert command[1] in spec.files
    assert f"{app.SANDBOX_SCRIPTS_ROOT}/document_cache.py" in spec.files
    assert f"{app.SANDBOX_SCRIPTS_ROOT}/preflight.py" in spec.files
    assert (command[-1] == WORKSPACE_ID) == use_cache


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("option", "value"),
    [("network_policy", {"defaultAction": "allow"}), ("resource_requests", {"cpu": 1})],
)
async def test_preconfigured_managed_sandbox_option_is_rejected(server, monkeypatch, tmp_path, option, value):
    from environments.pdf_analyst_qa.prepare_example import prepare_example
    from resources_servers.pdf_analyst_qa import app

    resource, _ = server
    root = tmp_path / "runtime"
    prepare_example(root)
    resource.config.dataset_root = str(root)
    resource.config.sandbox_config = {"provider_options": {option: value}}
    resource.server_client.global_config_dict = {
        "sandbox": {"opensandbox": {"connection": {"domain": "sandbox.example"}}}
    }
    monkeypatch.setattr(app, "sandbox_server_url", lambda *_args, **_kwargs: "http://10.2.3.4:8000")
    state = _SessionState(resource._load_task(resource.seed_request_model(**seed().task_data)), "example-document-v1")
    with pytest.raises(ValueError, match=f"{option}.*managed"):
        await START_SANDBOX(resource, state, WORKSPACE_ID)
