# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox, DirectSandboxConnection, SandboxAccess
from nemo_gym.server_runtime import (
    LentSandboxes,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.deepcad.app import DeepCADResourcesServer
from resources_servers.deepcad.mixed_iou_config import MixedIoUConfig
from resources_servers.deepcad.opencode_app import DeepCADOpenCodeConfig, DeepCADOpenCodeResourcesServer


CODE = "import cadquery as cq\nresult = cq.Workplane('XY').box(1,2,3)"


def image_url() -> str:
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def seed(session: str = "cad-session") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session,
        episode_id=EpisodeId(rollout_id=f"rollout-{session}"),
        task_id=TaskId(taskset="cad", task_id="box"),
        task_data={
            "verifier_metadata": {
                "task_id": "box",
                "reference_step": "private-reference.step",
                "view_mode": "diagonal",
                "foreground_color": "teal",
            },
            "responses_create_params": {
                "input": [
                    {"role": "user", "content": [{"type": "input_image", "image_url": image_url(), "detail": "high"}]}
                ]
            },
        },
    )


def verify_body(status: str = "completed") -> dict:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "target images"}]},
        "verifier_metadata": {"task_id": "box", "reference_step": "private-reference.step"},
        "response": {
            "id": "resp",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "status": status,
            "output": [
                {
                    "type": "message",
                    "id": "msg",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "```python\nraise Exception('prose')\n```", "annotations": []}
                    ],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }


def close_body(session: str = "cad-session") -> dict:
    return ResourcesCloseSessionRequest(resources_session_id=session, episode_id=seed(session).episode_id).model_dump(
        mode="json"
    )


@pytest.fixture
def runtime(monkeypatch):
    workspaces, files = {}, {}
    lending = MagicMock(spec=LentSandboxes)

    async def lend(session_id, task_id, *, files=None):
        workspace = MagicMock(spec=AsyncSandbox)
        workspace.exec = AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None, stdout="True"))

        async def download(remote, local):
            local.write_bytes(workspace.source)

        workspace.download = AsyncMock(side_effect=download)
        workspace.source = CODE.encode()
        workspaces[session_id] = workspace
        runtime_files[session_id] = dict(files or {})
        return SandboxAccess(
            connection=DirectSandboxConnection(provider_config_ref="agent_sandbox", descriptor={}),
            workdir="/workspace",
        )

    async def release(session_id):
        workspaces.pop(session_id, None)

    async def release_all():
        workspaces.clear()

    runtime_files = files
    lending.lend = AsyncMock(side_effect=lend)
    lending.get.side_effect = workspaces.get
    lending.release = AsyncMock(side_effect=release)
    lending.release_all = AsyncMock(side_effect=release_all)
    monkeypatch.setattr("resources_servers.deepcad.opencode_app.LentSandboxes", lambda *a, **k: lending)
    evaluate = AsyncMock(return_value={"reward": 0.75})
    monkeypatch.setattr(DeepCADResourcesServer, "_evaluate", evaluate)
    config = DeepCADOpenCodeConfig(name="deepcad_opencode", host="127.0.0.1", port=8080, entrypoint="opencode_app.py")
    server = DeepCADOpenCodeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    client = TestClient(server.setup_webserver())
    return SimpleNamespace(
        server=server, client=client, lending=lending, workspaces=workspaces, files=files, evaluate=evaluate
    )


def start(runtime, session="cad-session"):
    result = runtime.client.post("/seed_session", json=seed(session).model_dump(mode="json"))
    assert result.status_code == 200, result.text
    return result


def test_seed_lends_targets_and_settings_only_and_is_idempotent(runtime):
    first = start(runtime)
    assert first.json()["sandbox_access"]["workdir"] == "/workspace"
    assert set(runtime.files["cad-session"]) == {"/workspace/target-0.png", "/workspace/.cad/settings.json"}
    assert json.loads(runtime.files["cad-session"]["/workspace/.cad/settings.json"]) == {
        "view_mode": "diagonal",
        "color": "teal",
    }
    assert start(runtime).json() == first.json()
    assert runtime.lending.lend.await_count == 1


def test_final_file_is_authoritative_and_workspace_stops_before_grading(runtime):
    start(runtime)

    async def grade(code, *args, **kwargs):
        assert not runtime.workspaces
        assert code == CODE
        return {"reward": 0.75}

    runtime.evaluate.side_effect = grade
    result = runtime.client.post("/verify", json=verify_body())
    assert result.status_code == 200, result.text
    assert result.json()["reward"] == 0.75
    assert result.json()["final_file_status"] == "collected"
    assert "prose" in result.json()["response"]["output"][0]["content"][0]["text"]


@pytest.mark.parametrize("stdout", ["False", ""])
def test_missing_file_never_uses_prose_or_caller_final_code(runtime, stdout):
    start(runtime)
    runtime.workspaces["cad-session"].exec.return_value.stdout = stdout
    result = runtime.client.post("/verify", json=verify_body() | {"final_code": CODE})
    assert result.status_code == 200, result.text
    assert result.json()["reward"] == 0 and result.json()["mask_sample"] is False
    runtime.evaluate.assert_not_awaited()


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_incomplete_agent_is_unmasked_zero_without_collecting_file(runtime, status):
    start(runtime)
    workspace = runtime.workspaces["cad-session"]
    result = runtime.client.post("/verify", json=verify_body(status))
    assert result.status_code == 200, result.text
    assert result.json()["reward"] == 0 and result.json()["mask_sample"] is False
    assert result.json()["failure_kind"] == "deepcad:agent_incomplete"
    workspace.download.assert_not_awaited()


@pytest.mark.parametrize("source", [b"\xff", b"x" * 65537])
def test_invalid_final_source_is_zero(runtime, source):
    start(runtime)
    runtime.workspaces["cad-session"].source = source
    result = runtime.client.post("/verify", json=verify_body())
    assert result.status_code == 200, result.text
    assert result.json()["reward"] == 0 and result.json()["mask_sample"] is False
    runtime.evaluate.assert_not_awaited()


def test_provider_collection_failure_is_masked_and_released(runtime):
    start(runtime)
    runtime.workspaces["cad-session"].exec.return_value.error_type = "timeout"
    result = runtime.client.post("/verify", json=verify_body())
    assert result.status_code == 200, result.text
    assert result.json()["mask_sample"] is True
    assert result.json()["failure_kind"] == "verifier_error"
    assert not runtime.workspaces
    runtime.evaluate.assert_not_awaited()


def test_verify_retry_uses_frozen_source_after_workspace_is_gone(runtime):
    start(runtime)
    workspace = runtime.workspaces["cad-session"]
    runtime.evaluate.side_effect = [
        {"reward": 0, "mask_sample": True, "failure_kind": "deepcad:infrastructure"},
        {"reward": 0.75},
    ]
    assert runtime.client.post("/verify", json=verify_body()).json()["mask_sample"]
    workspace.source = b"changed after collection"
    retried = runtime.client.post("/verify", json=verify_body())
    assert retried.status_code == 200, retried.text
    assert retried.json()["reward"] == 0.75
    assert all(call.args[0] == CODE for call in runtime.evaluate.await_args_list)
    assert workspace.download.await_count == 1


def test_close_is_idempotent_and_closed_verify_is_masked(runtime):
    start(runtime)
    for _ in range(2):
        assert runtime.client.post("/close_session", json=close_body()).status_code == 200
    result = runtime.client.post("/verify", json=verify_body())
    assert result.json()["mask_sample"] is True
    runtime.evaluate.assert_not_awaited()


def test_failed_stop_retains_ownership_for_close_retry(runtime):
    start(runtime)
    release = runtime.lending.release.side_effect
    runtime.lending.release.side_effect = RuntimeError("provider down")
    with pytest.raises(RuntimeError, match="provider down"):
        runtime.client.post("/close_session", json=close_body())
    assert runtime.server.has_session_state("cad-session")
    runtime.lending.release.side_effect = release
    assert runtime.client.post("/close_session", json=close_body()).status_code == 200
    assert not runtime.workspaces


@pytest.mark.parametrize("body", [{}, {"resources_session_id": "partial"}])
def test_cookie_and_partial_typed_seeds_are_rejected(runtime, body):
    assert runtime.client.post("/seed_session", json=body).status_code == 422
    runtime.lending.lend.assert_not_awaited()


def test_invalid_target_is_rejected_before_lending(runtime):
    body = seed().model_dump(mode="json")
    body["task_data"]["responses_create_params"]["input"][0]["content"][0]["image_url"] = (
        "https://target.invalid/a.png"
    )
    assert runtime.client.post("/seed_session", json=body).status_code == 422
    runtime.lending.lend.assert_not_awaited()


def test_shutdown_releases_unclosed_workspaces(runtime):
    with runtime.client:
        start(runtime)
        assert runtime.workspaces
    runtime.lending.release_all.assert_awaited_once()
    assert not runtime.workspaces


def test_resources_session_conformance(runtime):
    check_resources_session_contract(
        runtime.server.setup_webserver(),
        seed("contract"),
        keeps_state=True,
        holds_state=runtime.server.has_session_state,
        verify_body=verify_body(),
    )


def test_lost_mixed_metric_session_keeps_component_schema(runtime):
    runtime.server.config.verification_metric = "mixed_iou"
    runtime.server.config.mixed_iou = MixedIoUConfig()
    result = runtime.client.post("/verify", json=verify_body())
    assert result.status_code == 200, result.text
    assert result.json()["mask_sample"] is True
    assert result.json()["reward_components"] == {"brep_iou": 0.0, "octree_iou": 0.0, "boundary_iou": 0.0}


def test_failed_seed_releases_partial_state_and_can_retry(runtime):
    lend = runtime.lending.lend.side_effect
    runtime.lending.lend.side_effect = RuntimeError("failed seed")
    with pytest.raises(RuntimeError, match="failed seed"):
        start(runtime)
    runtime.lending.release.assert_awaited_with("cad-session")
    assert not runtime.server.has_session_state("cad-session")
    runtime.lending.lend.side_effect = lend
    start(runtime)
