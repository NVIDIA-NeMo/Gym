# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

import resources_servers.nl2repobench.app as app_module
from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.nl2repobench.app import (
    NL2RepoBenchResourcesServer,
    NL2RepoBenchResourcesServerConfig,
    NL2RepoBenchSeedSessionRequest,
    NL2RepoBenchVerifyRequest,
    VerifierResult,
    _compute_reward,
    _parse_pytest_summary,
    _resolve_task,
    _resolve_task_id,
)
from resources_servers.nl2repobench.task_store import task_id, task_image


UPSTREAM_IMAGE = "ghcr.io/multimodal-art-projection/nl2repobench/example-task:1.0"


def _config(tasks_dir: Path) -> NL2RepoBenchResourcesServerConfig:
    return NL2RepoBenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="nl2repobench_resources_server",
        tasks_dir=tasks_dir,
        expected_task_count=1,
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


# --- pytest-summary regex parsing -------------------------------------------------------------


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ("12 passed, 3 failed in 4.21s", (12, 3, 0)),
        ("0 passed, 1 error in 0.02s", (0, 0, 1)),
        ("5 passed in 1.10s", (5, 0, 0)),
        ("no summary line at all", (0, 0, 0)),
        ("=== 2 failed, 1 passed, 1 error in 0.5s ===", (1, 2, 1)),
    ],
)
def test_parse_pytest_summary(summary: str, expected: tuple[int, int, int]) -> None:
    assert _parse_pytest_summary(summary) == expected


def test_parse_pytest_summary_takes_last_occurrence() -> None:
    # A command before the final pytest run (e.g. pip install output) might also emit numbers
    # that look like a summary; only the last occurrence should be used.
    combined_output = "1 passed in the install step\n...\n7 passed, 2 failed in 3.00s"
    assert _parse_pytest_summary(combined_output) == (7, 2, 0)


# --- reward formula ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tests_passed", "test_case_count", "expected"),
    [
        (5, 10, 0.5),
        (10, 10, 1.0),
        (15, 10, 1.0),  # clamped to 1.0
        (0, 10, 0.0),
        (5, 0, 0.0),  # divide-by-zero guard
    ],
)
def test_compute_reward(tests_passed: int, test_case_count: int, expected: float) -> None:
    assert _compute_reward(tests_passed, test_case_count) == expected


# --- task resolution -----------------------------------------------------------------------------


def test_conflicting_task_ids_fail() -> None:
    request = _request()
    request["verifier_metadata"] = {"task_id": "different-task"}

    with pytest.raises(ValueError, match="Conflicting"):
        _resolve_task_id(NL2RepoBenchVerifyRequest.model_validate(request))


def test_request_image_matches_pinned_task_image(task_assets: Path) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))

    task = _resolve_task(NL2RepoBenchVerifyRequest.model_validate(_request()), server._task_store)

    assert task_id(task) == "example-task"
    assert task_image(task) == UPSTREAM_IMAGE


def test_request_image_must_match_pinned_task_image(task_assets: Path) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    image = "ghcr.io/multimodal-art-projection/nl2repobench/example-task:v2"

    with pytest.raises(ValueError, match="does not match the pinned image"):
        _resolve_task(NL2RepoBenchVerifyRequest.model_validate(_request() | {"image": image}), server._task_store)


# --- sandbox network policy ----------------------------------------------------------------------


def test_model_endpoint_is_the_only_added_egress_target(monkeypatch: MonkeyPatch, task_assets: Path) -> None:
    config = _config(task_assets)
    config.sandbox_model_server = ModelServerRef(type="responses_api_models", name="policy_model")
    monkeypatch.setattr(
        "resources_servers.nl2repobench.app.get_global_config_dict",
        lambda: {"policy_model": {"responses_api_agents": {"model": {"host": "model.internal", "port": 8000}}}},
    )
    config.sandbox_config = {
        "provider_options": {
            "network_policy": {"defaultAction": "deny", "egress": [{"action": "deny", "target": "example.com"}]}
        }
    }
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    assert server._provider_options(phase="agent")["network_policy"] == {
        "defaultAction": "deny",
        "egress": [
            {"action": "deny", "target": "example.com"},
            {"action": "allow", "target": "model.internal"},
        ],
    }


def test_loopback_model_endpoint_is_rejected(monkeypatch: MonkeyPatch, task_assets: Path) -> None:
    config = _config(task_assets)
    config.sandbox_model_server = ModelServerRef(type="responses_api_models", name="policy_model")
    monkeypatch.setattr(
        "resources_servers.nl2repobench.app.get_global_config_dict",
        lambda: {"policy_model": {"responses_api_agents": {"model": {"host": "127.0.0.1", "port": 8000}}}},
    )
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    with pytest.raises(ValueError, match="loopback model host"):
        server._provider_options(phase="agent")


def test_network_policy_is_scoped_to_agent_sandbox(task_assets: Path) -> None:
    config = _config(task_assets)
    config.sandbox_config = {"provider_options": {"network_policy": {"defaultAction": "allow", "egress": []}}}
    server = NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    assert server._provider_options(phase="agent")["network_policy"] == {"defaultAction": "allow", "egress": []}
    assert server._provider_options(phase="verifier") == {}


# --- seed_session / verify flow ------------------------------------------------------------------


async def test_seed_session_creates_agent_sandbox(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox", "workdir": "/workspace"}
    monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: MagicMock())
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    response = await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))

    assert response.sandbox_handle == "agent-sandbox"
    assert response.sandbox_descriptor == {"sandbox_id": "agent-sandbox", "workdir": "/workspace"}
    assert "test-session" in server._agent_sessions
    spec = sandbox.start.await_args.args[0]
    assert spec.image == UPSTREAM_IMAGE
    assert spec.workdir == "/workspace"


async def test_verify_without_seed_returns_incomplete_result(task_assets: Path) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "missing-session"}

    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0
    assert response.evaluation_completed is False
    assert "no agent session found" in (response.verifier_error or "")


async def test_verify_collects_workspace_and_runs_tests_in_fresh_sandbox(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    verifier_sandbox = AsyncMock()
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(side_effect=[agent_sandbox, verifier_sandbox]))
    collect_workspace = AsyncMock(return_value=(b"tar-bytes", None))
    monkeypatch.setattr(server, "_collect_workspace", collect_workspace)
    monkeypatch.setattr(
        server,
        "_run_verifier",
        AsyncMock(
            return_value=VerifierResult(
                evaluation_completed=True,
                reward=1.0,
                tests_passed=5,
                tests_failed=0,
                tests_error=0,
                test_case_count=5,
                success_rate=1.0,
                test_output="5 passed in 1.0s",
            )
        ),
    )
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))
    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 1.0
    assert response.evaluation_completed is True
    assert response.tests_passed == 5
    assert response.test_case_count == 5
    assert response.workspace_sha256 is not None
    assert response.workspace_bytes == len(b"tar-bytes")
    agent_sandbox.stop.assert_awaited_once()
    verifier_sandbox.stop.assert_awaited_once()
    assert server._agent_sessions == {}


async def test_verify_workspace_collection_failure_short_circuits(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server = NL2RepoBenchResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    create_sandbox = AsyncMock(return_value=agent_sandbox)
    monkeypatch.setattr(server, "_create_sandbox", create_sandbox)
    monkeypatch.setattr(server, "_collect_workspace", AsyncMock(return_value=(None, "tar command failed")))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    await server.seed_session(request, NL2RepoBenchSeedSessionRequest.model_validate(_request()))
    response = await server.verify(request, NL2RepoBenchVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0
    assert response.evaluation_completed is False
    assert response.verifier_error == "tar command failed"
    assert create_sandbox.await_count == 1
    agent_sandbox.stop.assert_awaited_once()


def test_collect_command_excludes_precede_positional_args() -> None:
    # GNU tar requires --exclude flags to precede positional args (-C DIR .);
    # once an --exclude follows them it's rejected (exit code 2) or silently
    # ignored, depending on tar version. Caught via a real (non-mocked) tar
    # invocation during a live smoke test — mocked sandbox tests can't catch
    # this since they never actually shell out to tar.
    command = app_module._COLLECT_COMMAND
    exclude_idx = command.index("--exclude")
    positional_idx = command.index("-C /workspace")
    assert exclude_idx < positional_idx

    result = subprocess.run(
        "cd /tmp && rm -rf nl2repobench_tar_test && mkdir -p nl2repobench_tar_test/workspace/.git "
        "nl2repobench_tar_test/workspace/src && echo x > nl2repobench_tar_test/workspace/src/a.py "
        "&& echo y > nl2repobench_tar_test/workspace/.git/HEAD && "
        + command.replace("/tmp/workspace.tar.gz", "/tmp/nl2repobench_tar_test/workspace.tar.gz").replace(
            "-C /workspace", "-C /tmp/nl2repobench_tar_test/workspace"
        ),
        shell=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    listing = subprocess.run(
        "tar tzf /tmp/nl2repobench_tar_test/workspace.tar.gz",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "src/a.py" in listing
    assert ".git" not in listing
