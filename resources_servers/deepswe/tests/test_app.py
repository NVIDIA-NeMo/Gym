# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.deepswe.app import (
    WORKSPACE_HELPER_LOCAL_PATH,
    WORKSPACE_HELPER_REMOTE_PATH,
    DeepSWEResourcesServer,
    DeepSWEResourcesServerConfig,
    DeepSWESeedSessionRequest,
    DeepSWEVerifyRequest,
    VerifierResult,
    WorkspacePatchMetadata,
    _resolve_task,
    _resolve_task_id,
)


UPSTREAM_IMAGE = "public.example/project/example-task:v1.1"


def _config(
    tasks_dir: Path,
    *,
    golden: bool = True,
) -> DeepSWEResourcesServerConfig:
    return DeepSWEResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="deepswe_resources_server",
        tasks_dir=tasks_dir,
        expected_task_count=1,
        is_verifying_golden_patch=golden,
        sandbox_provider="test",
        sandbox_config={},
    )


def _request() -> dict:
    return {
        "task_id": "example-task",
        "image": UPSTREAM_IMAGE,
        "verifier_metadata": {"task_id": "example-task"},
        "responses_create_params": {"input": [{"role": "user", "content": "test"}]},
        "response": {
            "output": [],
            "id": "response",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }


def test_model_endpoint_is_the_only_added_egress_target(task_assets: Path, tmp_path: Path) -> None:
    config = _config(task_assets)
    config.sandbox_model_base_url = "http://model.internal:8000"
    config.sandbox_config = {
        "provider_options": {
            "network_policy": {
                "defaultAction": "deny",
                "egress": [{"action": "deny", "target": "example.com"}],
            }
        }
    }
    server = DeepSWEResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    assert server._provider_options(phase="agent")["network_policy"] == {
        "defaultAction": "deny",
        "egress": [
            {"action": "deny", "target": "example.com"},
            {"action": "allow", "target": "model.internal"},
        ],
    }


def test_loopback_model_endpoint_is_rejected(task_assets: Path, tmp_path: Path) -> None:
    config = _config(task_assets)
    config.sandbox_model_base_url = "http://127.0.0.1:8000"
    server = DeepSWEResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    with pytest.raises(ValueError, match="loopback model host"):
        server._provider_options(phase="agent")


def test_network_policy_is_scoped_to_agent_sandbox(task_assets: Path, tmp_path: Path) -> None:
    config = _config(task_assets)
    config.sandbox_config = {"provider_options": {"network_policy": {"defaultAction": "allow", "egress": []}}}
    server = DeepSWEResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )

    assert server._provider_options(phase="agent")["network_policy"] == {
        "defaultAction": "allow",
        "egress": [],
    }
    assert server._provider_options(phase="verifier") == {}


async def test_golden_verify_passes_structured_result(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets),
        server_client=MagicMock(spec=ServerClient),
    )
    fake_sandbox = AsyncMock()
    create_sandbox = AsyncMock(return_value=fake_sandbox)
    monkeypatch.setattr(server, "_create_sandbox", create_sandbox)
    monkeypatch.setattr(
        server,
        "_run_verifier",
        AsyncMock(
            return_value=VerifierResult(
                evaluation_completed=True,
                reward=1.0,
                f2p_total=2,
                f2p_passed=2,
                p2p_total=1,
                p2p_passed=1,
                f2p=1.0,
                p2p=1.0,
                partial=1.0,
            )
        ),
    )

    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}
    response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))

    body = response.model_dump()
    assert body["evaluation_completed"] is True
    assert body["reward"] == 1.0
    assert body["f2p_passed"] == body["f2p_total"] == 2
    assert body["model_patch"] == "golden patch\n"
    assert create_sandbox.await_args.args[0].image == UPSTREAM_IMAGE
    assert create_sandbox.await_args.kwargs == {"phase": "golden-verifier"}
    fake_sandbox.stop.assert_awaited_once()


async def test_rollout_captures_workspace_and_verifies_in_fresh_sandbox(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False),
        server_client=MagicMock(spec=ServerClient),
    )
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox", "workdir": "/app"}
    verifier_sandbox = AsyncMock()
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(side_effect=[agent_sandbox, verifier_sandbox]))
    monkeypatch.setattr(server, "_capture_initial_tree", AsyncMock(return_value="1" * 40))
    capture_model_patch = AsyncMock(
        return_value=(
            "agent patch\n",
            WorkspacePatchMetadata(
                initial_tree="1" * 40,
                final_tree="2" * 40,
                synthetic_tree="3" * 40,
                changed_paths=4,
                patch_bytes=12,
            ),
        )
    )
    monkeypatch.setattr(server, "_capture_model_patch", capture_model_patch)
    monkeypatch.setattr(
        server,
        "_run_verifier",
        AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1.0)),
    )
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    seed = await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))
    response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))

    assert seed.sandbox_handle == "agent-sandbox"
    assert seed.sandbox_descriptor == {"sandbox_id": "agent-sandbox", "workdir": "/app"}
    assert seed.initial_tree == "1" * 40
    captured_task = capture_model_patch.await_args.args[1]
    assert captured_task.image == UPSTREAM_IMAGE
    assert capture_model_patch.await_args.args == (agent_sandbox, captured_task, "1" * 40)
    assert [call.args[0].image for call in server._create_sandbox.await_args_list] == [
        UPSTREAM_IMAGE,
        UPSTREAM_IMAGE,
    ]
    assert response.reward == 1.0
    assert response.evaluation_completed is True
    assert response.model_patch == "agent patch\n"
    assert response.changed_paths == 4
    assert response.initial_tree == "1" * 40
    assert response.final_tree == "2" * 40
    assert response.synthetic_tree == "3" * 40
    agent_sandbox.stop.assert_awaited_once()
    verifier_sandbox.stop.assert_awaited_once()
    assert server._agent_sessions == {}


async def test_workspace_exclusions_are_forwarded_to_patch_capture(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False),
        server_client=MagicMock(spec=ServerClient),
    )
    server.config.workspace_excluded_paths = ("export.json", ".opencode")
    initial_tree = "1" * 40
    helper = AsyncMock(
        return_value={
            "initial_tree": initial_tree,
            "final_tree": "2" * 40,
            "synthetic_tree": "3" * 40,
            "changed_paths": 0,
            "patch_bytes": 0,
        }
    )
    monkeypatch.setattr(server, "_workspace_helper_payload", helper)
    sandbox = AsyncMock()

    async def download_empty_patch(_remote_path: str, local_path: Path) -> None:
        local_path.write_bytes(b"")

    sandbox.download.side_effect = download_empty_patch

    await server._capture_model_patch(sandbox, server._task_store.get("example-task"), initial_tree)

    arguments = helper.await_args.args[1]
    assert arguments[-4:] == ["--exclude-path", "export.json", "--exclude-path", ".opencode"]


async def test_workspace_helper_uses_isolated_python(
    task_assets: Path,
    tmp_path: Path,
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False),
        server_client=MagicMock(spec=ServerClient),
    )
    sandbox = AsyncMock()
    sandbox.exec.return_value = MagicMock(return_code=0, stdout='{"initial_tree": "tree"}', stderr="")

    payload = await server._workspace_helper_payload(sandbox, ["snapshot", "--repo", "/app"])

    assert payload == {"initial_tree": "tree"}
    sandbox.upload.assert_awaited_once_with(WORKSPACE_HELPER_LOCAL_PATH, WORKSPACE_HELPER_REMOTE_PATH)
    sandbox.exec.assert_awaited_once_with(
        f"python3 -I {WORKSPACE_HELPER_REMOTE_PATH} snapshot --repo /app",
        cwd="/app",
        timeout_s=server.config.workspace_capture_timeout_s,
    )


async def test_rollout_workspace_capture_failure_is_structured_and_cleans_up(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False),
        server_client=MagicMock(spec=ServerClient),
    )
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    create_sandbox = AsyncMock(return_value=agent_sandbox)
    monkeypatch.setattr(server, "_create_sandbox", create_sandbox)
    monkeypatch.setattr(server, "_capture_initial_tree", AsyncMock(return_value="1" * 40))
    monkeypatch.setattr(server, "_capture_model_patch", AsyncMock(side_effect=RuntimeError("broken git repo")))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))
    response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0
    assert response.evaluation_completed is False
    assert response.verifier_error == "RuntimeError: broken git repo"
    assert response.model_patch_bytes == 0
    assert create_sandbox.await_count == 1
    agent_sandbox.stop.assert_awaited_once()


async def test_rollout_without_seed_returns_incomplete_result(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False),
        server_client=MagicMock(spec=ServerClient),
    )
    create_sandbox = AsyncMock()
    monkeypatch.setattr(server, "_create_sandbox", create_sandbox)
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "missing-session"}

    response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0
    assert response.evaluation_completed is False
    assert "No DeepSWE agent sandbox" in (response.verifier_error or "")
    create_sandbox.assert_not_awaited()


def test_conflicting_task_ids_fail() -> None:
    request = _request()
    request["verifier_metadata"] = {"task_id": "different-task"}

    with pytest.raises(ValueError, match="Conflicting"):
        _resolve_task_id(DeepSWEVerifyRequest.model_validate(request))


def test_request_image_matches_pinned_task_image(task_assets: Path) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets),
        server_client=MagicMock(spec=ServerClient),
    )

    task = _resolve_task(DeepSWEVerifyRequest.model_validate(_request()), server._task_store)

    assert task.task_id == "example-task"
    assert task.image == UPSTREAM_IMAGE


def test_request_image_must_match_pinned_task_image(task_assets: Path) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets),
        server_client=MagicMock(spec=ServerClient),
    )
    image = "registry.example/project/deepswe.example-task:v2"

    with pytest.raises(ValueError, match="does not match the pinned image"):
        _resolve_task(DeepSWEVerifyRequest.model_validate(_request() | {"image": image}), server._task_store)
