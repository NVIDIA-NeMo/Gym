# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

import resources_servers.deepswe.app as app_module
from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import ENVIRONMENT_SERVER_FAILED, SESSION_LOST, VERIFIER_ERROR, RequestFailureLog
from nemo_gym.sandbox import SandboxResources
from resources_servers.deepswe.app import (
    DeepSWEResourcesServer,
    DeepSWEResourcesServerConfig,
    DeepSWESeedSessionRequest,
    DeepSWEVerifyRequest,
    VerifierResult,
    _resolve_task,
    _resolve_task_id,
)
from resources_servers.deepswe.task_store import task_collect_hook, task_id, task_image


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


def test_model_endpoint_is_the_only_added_egress_target(
    monkeypatch: MonkeyPatch, task_assets: Path, tmp_path: Path
) -> None:
    config = _config(task_assets)
    config.sandbox_model_server = ModelServerRef(type="responses_api_models", name="policy_model")

    monkeypatch.setattr(
        "resources_servers.deepswe.app.get_global_config_dict",
        lambda: {"policy_model": {"responses_api_agents": {"model": {"host": "model.internal", "port": 8000}}}},
    )
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


def test_loopback_model_endpoint_is_rejected(monkeypatch: MonkeyPatch, task_assets: Path, tmp_path: Path) -> None:
    config = _config(task_assets)
    config.sandbox_model_server = ModelServerRef(type="responses_api_models", name="policy_model")

    monkeypatch.setattr(
        "resources_servers.deepswe.app.get_global_config_dict",
        lambda: {"policy_model": {"responses_api_agents": {"model": {"host": "127.0.0.1", "port": 8000}}}},
    )
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


@pytest.mark.parametrize(
    ("phase", "expected_resources"),
    [
        ("agent", SandboxResources(cpu=4, memory_mib=16384, disk_gib=20)),
        ("verifier", SandboxResources(cpu=6, memory_mib=24576, disk_gib=25)),
    ],
)
async def test_create_sandbox_scales_phase_limits_from_task_toml(
    task_assets: Path,
    monkeypatch: MonkeyPatch,
    phase: str,
    expected_resources: SandboxResources,
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets),
        server_client=MagicMock(spec=ServerClient),
    )
    sandbox = AsyncMock()
    monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: MagicMock())
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))

    created = await server._create_sandbox(server._task_store.get("example-task"), phase=phase)

    assert created is sandbox
    spec = sandbox.start_with_setup.await_args.args[0]
    assert spec.resources == expected_resources
    assert spec.image == UPSTREAM_IMAGE


async def test_create_sandbox_allows_resource_multiplier_and_explicit_overrides(
    task_assets: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    config = _config(task_assets)
    config.task_cpu_multiplier = 1.5
    config.task_memory_multiplier = 1.25
    config.sandbox_config = {"resources": {"memory_mib": 20000}}
    server = DeepSWEResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )
    sandbox = AsyncMock()
    monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: MagicMock())
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))

    await server._create_sandbox(server._task_store.get("example-task"), phase="agent")

    spec = sandbox.start_with_setup.await_args.args[0]
    assert spec.resources == SandboxResources(cpu=3, memory_mib=20000, disk_gib=20)


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
    assert body["mask_sample"] is False
    assert body["failure_kind"] is None
    assert body["failure_reason"] is None
    assert body["f2p_passed"] == body["f2p_total"] == 2
    assert body["model_patch"] == "golden patch\n"
    assert task_image(create_sandbox.await_args.args[0]) == UPSTREAM_IMAGE
    assert create_sandbox.await_args.kwargs == {"phase": "golden-verifier"}
    fake_sandbox.stop.assert_awaited_once()


async def test_rollout_collects_committed_patch_and_verifies_in_fresh_sandbox(
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
    collect_model_patch = AsyncMock(return_value=b"agent patch\n")
    monkeypatch.setattr(server, "_collect_model_patch", collect_model_patch)
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
    captured_task = collect_model_patch.await_args.args[1]
    assert task_image(captured_task) == UPSTREAM_IMAGE
    assert collect_model_patch.await_args.args == (agent_sandbox, captured_task)
    assert [task_image(call.args[0]) for call in server._create_sandbox.await_args_list] == [
        UPSTREAM_IMAGE,
        UPSTREAM_IMAGE,
    ]
    assert response.reward == 1.0
    assert response.evaluation_completed is True
    assert response.model_patch == "agent patch\n"
    agent_sandbox.stop.assert_awaited_once()
    verifier_sandbox.stop.assert_awaited_once()
    assert server._agent_sessions == {}


async def test_collect_model_patch_executes_upstream_hook(task_assets: Path) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False),
        server_client=MagicMock(spec=ServerClient),
    )
    sandbox = AsyncMock()
    sandbox.exec.return_value = MagicMock(return_code=0, stdout="", stderr="")
    expected_patch = b"binary patch\x00bytes"

    async def download_patch(remote_path: str, local_path: Path) -> None:
        assert remote_path == "/logs/artifacts/model.patch"
        local_path.write_bytes(expected_patch)

    sandbox.download.side_effect = download_patch
    task = server._task_store.get("example-task")

    model_patch = await server._collect_model_patch(sandbox, task)

    assert model_patch == expected_patch
    collect = task_collect_hook(task)
    sandbox.exec.assert_awaited_once_with(collect.command, timeout_s=collect.timeout_sec)
    assert collect.command.endswith(
        "git diff --binary 0123456789abcdef0123456789abcdef01234567 HEAD > /logs/artifacts/model.patch"
    )


async def test_rollout_collect_failure_is_structured_and_cleans_up(
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
    monkeypatch.setattr(server, "_collect_model_patch", AsyncMock(side_effect=RuntimeError("broken git repo")))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))
    response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0
    assert response.evaluation_completed is False
    assert response.mask_sample is True
    assert response.failure_kind == ENVIRONMENT_SERVER_FAILED
    assert response.verifier_error == "RuntimeError: broken git repo"
    assert response.failure_reason == "RuntimeError: broken git repo"
    assert response.model_patch is None
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
    assert response.mask_sample is True
    assert response.failure_kind == SESSION_LOST
    assert "No DeepSWE agent sandbox" in (response.verifier_error or "")
    assert response.model_patch is None
    create_sandbox.assert_not_awaited()


async def test_incomplete_verification_is_a_typed_masked_failure_that_keeps_the_patch(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=AsyncMock()))
    monkeypatch.setattr(
        server,
        "_run_verifier",
        AsyncMock(
            return_value=VerifierResult(
                evaluation_completed=False,
                reward=0.0,
                verifier_exit_code=0,
                verifier_error="Verifier reward.json is missing its reward field",
            )
        ),
    )
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    with RequestFailureLog.capture() as log:
        response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0
    assert response.evaluation_completed is False
    assert response.mask_sample is True
    assert response.failure_kind == VERIFIER_ERROR
    assert response.failure_reason == "Verifier reward.json is missing its reward field"
    assert response.model_patch == "golden patch\n"
    [event] = log.report.events.values()
    assert (event.boundary, event.failure_kind, event.operation, event.stage) == (
        "resources_server",
        VERIFIER_ERROR,
        "verify",
        "verification",
    )
    assert event.failure_reason == "Verifier reward.json is missing its reward field"


async def test_long_verifier_diagnostic_is_bounded_and_keeps_the_patch(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    """A diagnostic longer than the failure event allows is truncated there; the result and patch survive."""
    server = DeepSWEResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=AsyncMock()))
    long_error = "provider said: " + "x" * 5000
    monkeypatch.setattr(server, "_run_verifier", AsyncMock(side_effect=RuntimeError(long_error)))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    with RequestFailureLog.capture() as log:
        response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))

    assert response.evaluation_completed is False
    assert response.mask_sample is True
    assert response.model_patch == "golden patch\n"
    assert response.verifier_error == f"RuntimeError: {long_error}"
    # The verifier raised on every attempt: the two attempts run again were observed, the last one is the failure.
    *attempts, event = sorted(log.report.events.values(), key=lambda e: e.attempt_id)
    assert [e.operation for e in attempts] == ["verify", "verify"]
    assert all(e.exception_type == "RuntimeError" for e in attempts)
    assert len(event.failure_reason) <= 2000
    assert event.failure_reason.startswith("RuntimeError: provider said: xxx")
    assert event.failure_reason.endswith("[truncated]")


@pytest.mark.parametrize(
    ("reward_json", "expected_reward", "expected_error"),
    [
        ('{"reward": 0}', 0.0, None),
        ('{"reward": 1}', 1.0, None),
        ('{"reward": 1.0}', 1.0, None),
        ('{"reward": 0.0, "f2p_total": 3, "f2p_passed": 1}', 0.0, None),
        ("{}", 0.0, "missing its reward field"),
        ("[1]", 0.0, "missing its reward field"),
        ('{"reward": "1"}', 0.0, "non-binary reward"),
        ('{"reward": true}', 0.0, "non-binary reward"),
        ('{"reward": 2}', 0.0, "non-binary reward"),
        ('{"reward": 0.5}', 0.0, "non-binary reward"),
        ('{"reward": null}', 0.0, "non-binary reward"),
        ("{", 0.0, "Invalid verifier reward.json"),
        (None, 0.0, "did not produce /logs/verifier/reward.json"),
    ],
)
async def test_verifier_reward_json_must_hold_a_binary_reward(
    task_assets: Path,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    reward_json: str | None,
    expected_reward: float,
    expected_error: str | None,
) -> None:
    """Only an object with a real 0 or 1 reward is a verdict; anything else leaves the evaluation incomplete."""
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False), server_client=MagicMock(spec=ServerClient)
    )
    monkeypatch.setattr(server, "_stage_verifier", AsyncMock())
    sandbox = AsyncMock()
    artifacts = {"ctrf.json": "{}", "run.log": "log"}
    if reward_json is not None:
        artifacts["reward.json"] = reward_json

    async def execute(command: str, **_kwargs: object) -> SimpleNamespace:
        if command == "bash /tests/test.sh":
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        assert command.startswith("test -f /logs/verifier/")
        return SimpleNamespace(return_code=0 if command.rsplit("/", 1)[1] in artifacts else 1)

    async def download(remote_path: str, local_path: Path) -> None:
        local_path.write_text(artifacts[Path(remote_path).name], encoding="utf-8")

    sandbox.exec.side_effect = execute
    sandbox.download.side_effect = download

    result = await server._run_verifier(sandbox, server._task_store.get("example-task"), b"patch", tmp_path)

    assert result.evaluation_completed is (expected_error is None)
    assert result.reward == expected_reward
    assert result.verifier_exit_code == 0
    if expected_error is None:
        assert result.verifier_error is None
    else:
        assert expected_error in (result.verifier_error or "")
    assert sandbox.download.await_count == len(artifacts)


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

    assert task_id(task) == "example-task"
    assert task_image(task) == UPSTREAM_IMAGE


def test_request_image_must_match_pinned_task_image(task_assets: Path) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets),
        server_client=MagicMock(spec=ServerClient),
    )
    image = "registry.example/project/deepswe.example-task:v2"

    with pytest.raises(ValueError, match="does not match the pinned image"):
        _resolve_task(DeepSWEVerifyRequest.model_validate(_request() | {"image": image}), server._task_store)


# --- Inconclusive verification is run again on a fresh sandbox, within a bound --------------------------------------


NO_VERDICT = "Verifier did not produce /logs/verifier/reward.json"


def _rollout_server(task_assets: Path, monkeypatch: MonkeyPatch, *, verifier_results, sandboxes=None, **overrides):
    """A server past its agent phase: the committed patch is collected, and each verifier attempt gets a fresh sandbox
    from ``sandboxes`` (or a new mock) and the next of ``verifier_results`` (a result or an exception)."""
    config = _config(task_assets, golden=False).model_copy(update=overrides)
    server = DeepSWEResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox", "workdir": "/app"}
    created = [agent_sandbox, *(sandboxes if sandboxes is not None else [AsyncMock() for _ in verifier_results])]
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(side_effect=created))
    monkeypatch.setattr(server, "_collect_model_patch", AsyncMock(return_value=b"agent patch\n"))
    monkeypatch.setattr(server, "_run_verifier", AsyncMock(side_effect=list(verifier_results)))
    return server, created


async def _verify(server) -> tuple:
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}
    await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))
    with RequestFailureLog.capture() as log:
        response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))
    return response, log


def _inconclusive() -> VerifierResult:
    return VerifierResult(evaluation_completed=False, reward=0.0, verifier_exit_code=1, verifier_error=NO_VERDICT)


async def test_a_verifier_run_without_a_verdict_is_run_again_on_a_fresh_sandbox_and_the_verdict_counts(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, created = _rollout_server(
        task_assets,
        monkeypatch,
        verifier_results=[_inconclusive(), VerifierResult(evaluation_completed=True, reward=1.0)],
    )

    response, log = await _verify(server)

    assert (response.reward, response.evaluation_completed, response.mask_sample) == (1.0, True, False)
    assert response.failure_kind is None and response.failure_reason is None
    assert response.verification_attempts == 2 and response.inconclusive_attempts == [NO_VERDICT]
    # Each attempt verifies the same collected patch on its own fresh sandbox, and every sandbox is stopped.
    assert [call.args[2] for call in server._run_verifier.await_args_list] == [b"agent patch\n", b"agent patch\n"]
    assert [call.args[0] for call in server._run_verifier.await_args_list] == created[1:]
    for sandbox in created:
        sandbox.stop.assert_awaited_once()
    assert response.model_patch == "agent patch\n"
    # The attempt run again is observed as its own failure; the verdict stands and the row reports no failure.
    [attempt] = log.report.events.values()
    assert (attempt.failure_kind, attempt.operation, attempt.failure_reason, attempt.stage) == (
        VERIFIER_ERROR,
        "verify",
        "Verification produced no conclusive verdict",
        "verification",
    )
    assert attempt.attempt_id == f"{attempt.operation_id}:1"


async def test_a_verifier_that_never_produces_a_verdict_keeps_the_masked_outcome_after_the_bound(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, created = _rollout_server(task_assets, monkeypatch, verifier_results=[_inconclusive()] * 3)

    response, log = await _verify(server)

    assert server._run_verifier.await_count == 3 and len(created) == 4
    assert (response.reward, response.evaluation_completed, response.mask_sample) == (0.0, False, True)
    assert response.failure_kind == VERIFIER_ERROR and response.failure_reason == NO_VERDICT
    assert response.verification_attempts == 3 and response.inconclusive_attempts == [NO_VERDICT, NO_VERDICT]
    assert response.model_patch == "agent patch\n"
    # Two attempts were run again and observed as such; the last one is the row's failure, with its attempt id.
    first, second, final = sorted(log.report.events.values(), key=lambda e: e.attempt_id)
    assert [e.failure_reason for e in (first, second)] == ["Verification produced no conclusive verdict"] * 2
    assert (final.failure_kind, final.operation, final.failure_reason) == (VERIFIER_ERROR, "verify", NO_VERDICT)
    assert [e.attempt_id for e in (first, second, final)] == [f"{final.operation_id}:{n}" for n in (1, 2, 3)]
    for sandbox in created:
        sandbox.stop.assert_awaited_once()


async def test_a_conclusive_failing_verdict_is_not_run_again(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, created = _rollout_server(
        task_assets, monkeypatch, verifier_results=[VerifierResult(evaluation_completed=True, reward=0.0)]
    )
    response, _ = await _verify(server)
    assert server._run_verifier.await_count == 1 and len(created) == 2
    assert (response.reward, response.evaluation_completed, response.mask_sample) == (0.0, True, False)
    assert response.verification_attempts == 1 and response.inconclusive_attempts == []


async def test_a_verifier_sandbox_that_fails_to_start_or_run_is_tried_again(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    # The first verifier sandbox is refused by the provider before it exists; the second runs the verifier.
    verifier_sandbox = AsyncMock()
    server, _ = _rollout_server(
        task_assets,
        monkeypatch,
        verifier_results=[VerifierResult(evaluation_completed=True, reward=1.0)],
        sandboxes=[RuntimeError("provider refused the sandbox"), verifier_sandbox],
    )
    response, log = await _verify(server)
    assert response.reward == 1.0 and response.evaluation_completed is True
    assert response.verification_attempts == 2
    assert response.inconclusive_attempts == ["RuntimeError: provider refused the sandbox"]
    verifier_sandbox.stop.assert_awaited_once()
    # Only an exception before a sandbox was assigned names allocation as the failing operation.
    [attempt] = log.report.events.values()
    assert (attempt.failure_kind, attempt.operation, attempt.failure_reason, attempt.exception_type) == (
        VERIFIER_ERROR,
        "verify.sandbox.allocate",
        "Verifier sandbox allocation failed",
        "RuntimeError",
    )
    assert attempt.attempt_id == f"{attempt.operation_id}:1"


@pytest.mark.parametrize(
    "error",
    [
        "Verifier reward.json is missing its reward field",
        "Verifier returned non-binary reward: 2",
        "Invalid verifier reward.json: Expecting value",
        "Verifier timed out after 1800 seconds",
    ],
)
async def test_a_malformed_verdict_or_a_timeout_is_not_run_again(
    task_assets: Path, monkeypatch: MonkeyPatch, error: str
) -> None:
    server, created = _rollout_server(
        task_assets,
        monkeypatch,
        verifier_results=[
            VerifierResult(evaluation_completed=False, reward=0.0, verifier_exit_code=0, verifier_error=error)
        ],
    )
    response, log = await _verify(server)
    assert server._run_verifier.await_count == 1 and len(created) == 2
    assert (
        response.mask_sample is True and response.failure_kind == VERIFIER_ERROR and response.failure_reason == error
    )
    assert response.verification_attempts == 1 and response.inconclusive_attempts == []
    assert len(log.report.events) == 1


async def test_the_retry_bound_is_configurable_and_zero_keeps_a_single_attempt(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, created = _rollout_server(
        task_assets, monkeypatch, verifier_results=[_inconclusive()], inconclusive_verification_retries=0
    )
    response, _ = await _verify(server)
    assert server._run_verifier.await_count == 1 and len(created) == 2
    assert response.verification_attempts == 1 and response.failure_reason == NO_VERDICT


@pytest.mark.parametrize("intact", [True, False], ids=["intact", "digest-mismatch"])
async def test_a_saved_patch_is_reverified_on_a_fresh_sandbox_without_its_agent_session(
    task_assets: Path, monkeypatch: MonkeyPatch, intact: bool
) -> None:
    import hashlib

    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False),
        server_client=MagicMock(spec=ServerClient),
    )
    verifier_sandbox = AsyncMock()
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=verifier_sandbox))
    collect_model_patch = AsyncMock()
    monkeypatch.setattr(server, "_collect_model_patch", collect_model_patch)
    run_verifier = AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1.0))
    monkeypatch.setattr(server, "_run_verifier", run_verifier)
    request = MagicMock()
    request.session = {}
    patch = "saved patch\n"
    digest = hashlib.sha256(patch.encode("utf-8")).hexdigest() if intact else "0" * 64
    body = _request() | {"saved_model_patch": patch, "saved_model_patch_sha256": digest}

    response = await server.verify(request, DeepSWEVerifyRequest.model_validate(body))

    collect_model_patch.assert_not_awaited()
    dumped = response.model_dump()
    assert "saved_model_patch" not in dumped
    if intact:
        assert response.evaluation_completed is True and response.reward == 1.0
        assert response.replayed_saved_patch is True
        assert response.model_patch == patch and response.model_patch_sha256 == digest
        verifier_sandbox.stop.assert_awaited_once()
    else:
        assert response.evaluation_completed is False and response.mask_sample is True
        assert response.failure_kind == "session_lost"
        assert response.replayed_saved_patch is False
        run_verifier.assert_not_awaited()


async def test_without_a_saved_patch_a_missing_agent_session_is_still_lost(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False),
        server_client=MagicMock(spec=ServerClient),
    )
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock())
    request = MagicMock()
    request.session = {}
    response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))
    assert response.failure_kind == "session_lost" and response.replayed_saved_patch is False
