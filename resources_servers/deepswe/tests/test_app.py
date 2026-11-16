# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import resources_servers.deepswe.app as app_module
from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    ENVIRONMENT_SERVER_FAILED,
    PROVIDER_OOM_KILLED,
    SESSION_LOST,
    VERIFIER_ERROR,
    EpisodeId,
    FailureBoundary,
    FailureClass,
    NgFailureEvent,
    OutcomeAction,
    RequestFailureLog,
    SandboxFailureDetails,
    TaskId,
    failure_kinds,
    failure_report_for,
)
from nemo_gym.sandbox import SandboxResources
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from nemo_gym.trajectory import NG_AGENT_SANDBOX_OBSERVATION_KEY, SandboxObservation
from resources_servers.deepswe.app import (
    DeepSWEResourcesServer,
    DeepSWEResourcesServerConfig,
    DeepSWESeedSessionRequest,
    DeepSWEVerifyRequest,
    DeepSWEVerifyResponse,
    VerifierResult,
    _resolve_task,
    _resolve_task_id,
)
from resources_servers.deepswe.task_store import task_collect_hook, task_id, task_image


UPSTREAM_IMAGE = "public.example/project/example-task:v1.1"


@pytest.fixture(autouse=True)
def sandbox_provider_config(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {"test": {"opensandbox": {}}})


def _config(
    tasks_dir: Path,
    *,
    golden: bool = True,
    agent_policy: dict | None = None,
    classification: dict | None = None,
) -> DeepSWEResourcesServerConfig:
    return DeepSWEResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="deepswe_resources_server",
        tasks_dir=tasks_dir,
        expected_task_count=1,
        is_verifying_golden_patch=golden,
        outcome_policy={
            **({"by_stage": {"agent": agent_policy}} if agent_policy else {}),
            **({"classification": classification} if classification else {}),
        },
        sandbox_provider="test",
        sandbox_config={},
    )


def test_agent_failure_policy_uses_shared_defaults_and_accepts_partial_rules(task_assets: Path) -> None:
    default = _config(task_assets).outcome_policy.by_stage["agent"]
    assert default.default_action == OutcomeAction.FAIL
    assert default.by_failure_kind == {PROVIDER_OOM_KILLED: OutcomeAction.COUNT_AS_ZERO}
    assert default.by_status_code == {}
    override = _config(
        task_assets, agent_policy={"by_failure_kind": {PROVIDER_OOM_KILLED: "count_as_zero"}}
    ).outcome_policy.by_stage["agent"]
    assert override.default_action == OutcomeAction.FAIL
    assert override.by_failure_kind == {PROVIDER_OOM_KILLED: OutcomeAction.COUNT_AS_ZERO}


@pytest.mark.parametrize("action", ["verify", "retry"])
@pytest.mark.parametrize("rule", ["default_action", "by_failure_kind", "by_status_code", "by_failure_class"])
def test_agent_failure_policy_rejects_actions_this_boundary_cannot_execute(
    task_assets: Path, action: str, rule: str
) -> None:
    key = {"by_status_code": 503, "by_failure_class": "model"}.get(rule, PROVIDER_OOM_KILLED)
    value = action if rule == "default_action" else {key: action}
    with pytest.raises(ValueError, match="supports only count_as_zero, fail"):
        _config(task_assets, agent_policy={rule: value})


@pytest.mark.parametrize(
    "stage,action",
    [
        ("verification", "verify"),
        ("verification", "count_as_zero"),
        (None, "retry"),
        (None, "count_as_zero"),
        (None, "verify"),
    ],
)
def test_stage_policy_rejects_unsupported_class_actions(task_assets: Path, stage: str | None, action: str) -> None:
    policy = {"by_failure_class": {"evaluation": action}}
    override = {"default_policy": policy} if stage is None else {"by_stage": {stage: policy}}
    with pytest.raises(ValueError, match="policy supports only"):
        DeepSWEResourcesServerConfig.model_validate(_config(task_assets).model_dump() | {"outcome_policy": override})


def test_agent_failure_policy_rejects_unregistered_failure_kind(task_assets: Path) -> None:
    with pytest.raises(ValueError, match="Unregistered failure kinds"):
        _config(task_assets, agent_policy={"by_failure_kind": {"typo_oom": "count_as_zero"}})


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
    sandbox.exec.return_value = MagicMock(return_code=0, error_type=None, stdout="", stderr="")
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
            return SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")
        assert command.startswith("test -f /logs/verifier/")
        return SimpleNamespace(return_code=0 if command.rsplit("/", 1)[1] in artifacts else 1, error_type=None)

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


def _agent_oom_request(*, sandbox_id: str = "agent-sandbox") -> dict:
    error = _sandbox_failure(
        PROVIDER_OOM_KILLED, f"OpenSandbox terminal lifecycle OOMKilled; sandbox_id={sandbox_id!r}"
    )
    error.details = SandboxFailureDetails(provider="opensandbox", sandbox_id=sandbox_id)
    observation = SandboxObservation(
        role="agent",
        stage="agent",
        outcome="oom",
        failure_id=error.event_id,
    )
    body = _request()
    body["response"]["metadata"] = {NG_AGENT_SANDBOX_OBSERVATION_KEY: observation.model_dump_json()}
    report = failure_report_for(error).model_dump(mode="json")
    report["failure_id"] = None
    body["failure_report"] = report
    return body


async def test_empty_agent_response_can_still_receive_a_completed_verifier_zero(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False), server_client=MagicMock(spec=ServerClient)
    )
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    verifier_sandbox = AsyncMock()
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(side_effect=[agent_sandbox, verifier_sandbox]))
    collect_model_patch = AsyncMock(return_value=b"")
    run_verifier = AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=0.0, verifier_exit_code=0))
    monkeypatch.setattr(server, "_collect_model_patch", collect_model_patch)
    monkeypatch.setattr(server, "_run_verifier", run_verifier)
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "agent-timeout"}

    await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))
    response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))

    assert response.reward == 0.0
    assert response.evaluation_completed is True
    assert response.mask_sample is False
    assert response.failure_kind is None
    collect_model_patch.assert_awaited_once()
    run_verifier.assert_awaited_once()
    agent_sandbox.stop.assert_awaited_once()
    verifier_sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize(
    "policy,http_status,expected_action",
    [
        (None, None, OutcomeAction.COUNT_AS_ZERO),
        (
            {"by_failure_kind": {PROVIDER_OOM_KILLED: "count_as_zero"}, "by_status_code": {503: "fail"}},
            None,
            OutcomeAction.COUNT_AS_ZERO,
        ),
        (
            {"by_failure_kind": {PROVIDER_OOM_KILLED: "count_as_zero"}, "by_status_code": {503: "fail"}},
            503,
            OutcomeAction.FAIL,
        ),
        (
            {"default_action": "count_as_zero", "by_failure_kind": {PROVIDER_OOM_KILLED: "fail"}},
            None,
            OutcomeAction.FAIL,
        ),
    ],
)
@pytest.mark.parametrize("failure_class", [FailureClass.MODEL, FailureClass.EVALUATION])
async def test_confirmed_agent_sandbox_oom_applies_policy_without_collecting_or_verifying(
    task_assets: Path,
    monkeypatch: MonkeyPatch,
    policy: dict | None,
    http_status: int | None,
    expected_action: OutcomeAction,
    failure_class: FailureClass,
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(
            task_assets,
            golden=False,
            agent_policy=policy,
            classification={"by_stage": {"agent": {"by_failure_kind": {PROVIDER_OOM_KILLED: failure_class}}}},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    create_sandbox = AsyncMock(return_value=agent_sandbox)
    collect_model_patch = AsyncMock()
    run_verifier = AsyncMock()
    monkeypatch.setattr(server, "_create_sandbox", create_sandbox)
    monkeypatch.setattr(server, "_collect_model_patch", collect_model_patch)
    monkeypatch.setattr(server, "_run_verifier", run_verifier)
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))
    raw_body = _agent_oom_request()
    selected = next(iter(raw_body["failure_report"]["events"].values()))
    selected["http_status"] = http_status
    # A status on a cause must never override the selected OOM event's policy.
    cause = _sandbox_failure("provider_unavailable", "Prior provider request failed")
    cause.http_status = 503
    raw_body["failure_report"]["events"].update(failure_report_for(cause).model_dump(mode="json")["events"])
    selected["cause_id"] = cause.event_id
    body = DeepSWEVerifyRequest.model_validate(raw_body)
    with RequestFailureLog.capture() as log:
        response = await server.verify(request, body)
    assert log.report == body.failure_report
    assert log.report.failure_id is None
    counted = expected_action == OutcomeAction.COUNT_AS_ZERO
    assert response.failure_outcome_attribution is failure_class
    restored = DeepSWEVerifyResponse.model_validate_json(response.model_dump_json())
    assert restored.failure_outcome_attribution is failure_class
    assert restored.failure_outcome_action == expected_action
    assert restored.failure_report == response.failure_report
    assert response.failure_outcome_action == expected_action
    assert response.model_dump(mode="json")["failure_outcome_action"] == expected_action.value
    original_id = next(iter(body.failure_report.events))
    assert response.failure_report.events == body.failure_report.events
    assert response.failure_report.failure_id == (None if counted else original_id)
    fields = response.model_dump(mode="json")
    from nemo_gym.core import FailureReportRow
    from nemo_gym.orchestration.recovery import _scored_kind, status_fault_kind

    # Persist the applied decision without changing canonical facts or inferring model fault.
    saved_row = fields | FailureReportRow.fields(response.failure_report)
    assert _scored_kind(saved_row) == status_fault_kind(PROVIDER_OOM_KILLED, http_status)
    if counted:
        assert "_ng_failure_class" not in fields
    else:
        assert fields["_ng_failure_class"] == PROVIDER_OOM_KILLED
        assert fields["_ng_failure_terminal"] is True
        assert fields["_ng_failure_counts_as_zero"] is False
        assert fields["_ng_failure_id"] == original_id
    observation = SandboxObservation.model_validate_json(response.response.metadata[NG_AGENT_SANDBOX_OBSERVATION_KEY])
    assert observation.failure_id in response.failure_report.events

    assert response.reward == 0.0
    assert response.mask_sample is (not counted)
    assert response.failure_kind == PROVIDER_OOM_KILLED
    assert response.response.metadata[NG_AGENT_SANDBOX_OBSERVATION_KEY]
    assert response.model_patch_bytes == 0
    assert response.evaluation_completed is False
    assert "no patch could be collected" in (response.failure_reason or "")
    assert "sandbox_id='agent-sandbox'" in (response.failure_reason or "")
    create_sandbox.assert_awaited_once()
    collect_model_patch.assert_not_awaited()
    run_verifier.assert_not_awaited()
    agent_sandbox.stop.assert_awaited_once()
    assert server._agent_sessions == {}


@pytest.mark.parametrize("saved_patch", [False, True])
async def test_agent_sandbox_oom_zero_requires_a_seeded_session(task_assets: Path, saved_patch: bool) -> None:
    server = DeepSWEResourcesServer(
        config=_config(
            task_assets,
            golden=False,
            agent_policy={
                "default_action": "fail",
                "by_failure_kind": {PROVIDER_OOM_KILLED: "count_as_zero"},
            },
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "missing-session"}

    body = _agent_oom_request()
    if saved_patch:
        import hashlib

        body |= {
            "saved_model_patch": "saved patch",
            "saved_model_patch_sha256": hashlib.sha256(b"saved patch").hexdigest(),
        }
    with pytest.raises(RuntimeError, match="without its seeded DeepSWE session"):
        await server.verify(request, DeepSWEVerifyRequest.model_validate(body))


async def test_agent_sandbox_oom_zero_cannot_grade_a_golden_patch(task_assets: Path) -> None:
    server = DeepSWEResourcesServer(
        config=_config(
            task_assets,
            golden=True,
            agent_policy={
                "default_action": "fail",
                "by_failure_kind": {PROVIDER_OOM_KILLED: "count_as_zero"},
            },
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "golden"}

    with pytest.raises(ValueError, match="non-golden run"):
        await server.verify(request, DeepSWEVerifyRequest.model_validate(_agent_oom_request()))


async def test_agent_sandbox_oom_zero_requires_matching_sandbox(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server = DeepSWEResourcesServer(
        config=_config(
            task_assets,
            golden=False,
            agent_policy={
                "default_action": "fail",
                "by_failure_kind": {PROVIDER_OOM_KILLED: "count_as_zero"},
            },
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=agent_sandbox))
    collect_model_patch = AsyncMock()
    monkeypatch.setattr(server, "_collect_model_patch", collect_model_patch)
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}
    await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))

    with pytest.raises(ValueError, match="does not match the seeded"):
        await server.verify(request, DeepSWEVerifyRequest.model_validate(_agent_oom_request(sandbox_id="other")))

    collect_model_patch.assert_not_awaited()
    agent_sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("failure_phase", ["patch_collection", "verification"])
@pytest.mark.parametrize(
    "failure_kind",
    [PROVIDER_OOM_KILLED, "provider_unavailable", "provider_command_outcome_unknown", "provider_io_timeout"],
)
async def test_sandbox_loss_without_agent_execution_proof_fails_instead_of_scoring(
    task_assets: Path, monkeypatch: MonkeyPatch, failure_phase: str, failure_kind: str
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(
            task_assets,
            golden=False,
            agent_policy={"by_failure_kind": {PROVIDER_OOM_KILLED: "count_as_zero"}},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    verifier_sandboxes = [AsyncMock() for _ in range(3)]
    create_sandbox = AsyncMock(side_effect=[agent_sandbox, *verifier_sandboxes])
    monkeypatch.setattr(server, "_create_sandbox", create_sandbox)
    collect_model_patch = AsyncMock(return_value=b"valid patch\n")
    if failure_phase == "patch_collection":
        collect_model_patch.side_effect = _sandbox_failure(failure_kind, "sandbox loss during patch collection")
    monkeypatch.setattr(server, "_collect_model_patch", collect_model_patch)
    run_verifier = AsyncMock(side_effect=_sandbox_failure(failure_kind, "sandbox loss during verification"))
    monkeypatch.setattr(server, "_run_verifier", run_verifier)
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))
    with RequestFailureLog.capture() as log:
        response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))
    assert response.mask_sample is True
    assert response.evaluation_completed is False
    assert any(event.failure_kind == failure_kind for event in log.report.events.values())

    agent_sandbox.stop.assert_awaited_once()
    assert create_sandbox.await_count == (1 if failure_phase == "patch_collection" else 4)
    if failure_phase == "verification":
        assert response.verification_attempts == 3
        for sandbox in verifier_sandboxes:
            sandbox.stop.assert_awaited_once()
    else:
        run_verifier.assert_not_awaited()


async def test_verifier_status_timeout_propagates_without_scoring(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False), server_client=MagicMock(spec=ServerClient)
    )
    stage_verifier = AsyncMock()
    monkeypatch.setattr(server, "_stage_verifier", stage_verifier)
    task = server._task_store.get("example-task")
    error = _sandbox_failure("provider_command_outcome_unknown", "command status unavailable after recovery deadline")
    sandbox = AsyncMock()
    sandbox.exec.side_effect = error

    with pytest.raises(NgFailureEvent) as caught:
        await server._run_verifier(sandbox, task, b"valid patch\n", tmp_path)

    assert caught.value is error
    sandbox.exec.assert_awaited_once_with(
        "bash /tests/test.sh", cwd="/app", timeout_s=task.config.verifier.timeout_sec
    )


async def test_unclassified_verifier_timeout_preserves_its_exception(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False), server_client=MagicMock(spec=ServerClient)
    )
    monkeypatch.setattr(server, "_stage_verifier", AsyncMock())
    task = server._task_store.get("example-task")
    sandbox = AsyncMock()
    error = TimeoutError("verifier request timed out")
    sandbox.exec.side_effect = error

    with pytest.raises(TimeoutError) as caught:
        await server._run_verifier(sandbox, task, b"valid patch\n", tmp_path)
    assert caught.value is error
    sandbox.exec.assert_awaited_once_with(
        "bash /tests/test.sh", cwd="/app", timeout_s=task.config.verifier.timeout_sec
    )


@pytest.mark.parametrize("failure", ["verifier_timeout", "missing_reward", "verifier_setup"])
async def test_non_oom_verifier_failure_never_returns_an_unmasked_main_row(
    task_assets: Path, monkeypatch: MonkeyPatch, failure: str
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False), server_client=MagicMock(spec=ServerClient)
    )
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    verifier_sandbox = AsyncMock()
    create_sandbox = AsyncMock(side_effect=[agent_sandbox, verifier_sandbox])
    if failure == "verifier_setup":
        create_sandbox.side_effect = [agent_sandbox, RuntimeError("verifier sandbox unavailable")]
    monkeypatch.setattr(server, "_create_sandbox", create_sandbox)
    monkeypatch.setattr(server, "_collect_model_patch", AsyncMock(return_value=b"patch"))
    monkeypatch.setattr(server, "_stage_verifier", AsyncMock())
    if failure == "verifier_timeout":
        verifier_sandbox.exec.side_effect = TimeoutError("command deadline")
    else:

        async def execute(command: str, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                return_code=0 if command == "bash /tests/test.sh" else 1,
                error_type=None,
                stdout="",
                stderr="",
            )

        verifier_sandbox.exec.side_effect = execute
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}

    await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))
    response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))
    assert response.mask_sample is True
    assert response.evaluation_completed is False

    agent_sandbox.stop.assert_awaited_once()
    if failure != "verifier_setup":
        verifier_sandbox.stop.assert_awaited_once()
    assert server._agent_sessions == {}


@pytest.mark.parametrize(
    ("return_code", "error_type", "reward_json", "ctrf_json", "expected_error"),
    [
        (0, None, '{"reward": 0}', "{}", None),
        (0, None, "{}", "{}", "missing its reward field"),
        (0, None, '{"reward": true}', "{}", "non-binary reward"),
        (0, None, '{"reward": 2}', "{}", "non-binary reward"),
        (1, None, '{"reward": 0}', "{}", "Verifier command failed with code 1"),
        (0, "timeout", '{"reward": 0}', "{}", "error_type='timeout'"),
        (0, None, '{"reward": 0}', "{", "Invalid verifier ctrf.json"),
    ],
)
async def test_verifier_requires_completed_command_and_valid_artifacts(
    task_assets: Path,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    return_code: int,
    error_type: str | None,
    reward_json: str,
    ctrf_json: str,
    expected_error: str | None,
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False), server_client=MagicMock(spec=ServerClient)
    )
    monkeypatch.setattr(server, "_stage_verifier", AsyncMock())
    sandbox = AsyncMock()

    async def execute(command: str, **_kwargs: object) -> SimpleNamespace:
        if command == "bash /tests/test.sh":
            return SimpleNamespace(return_code=return_code, error_type=error_type, stdout="", stderr="")
        assert command.startswith("test -f /logs/verifier/")
        return SimpleNamespace(return_code=0, error_type=None)

    async def download(remote_path: str, local_path: Path) -> None:
        contents = {"reward.json": reward_json, "ctrf.json": ctrf_json, "run.log": "log"}
        local_path.write_text(contents[Path(remote_path).name])

    sandbox.exec.side_effect = execute
    sandbox.download.side_effect = download

    result = await server._run_verifier(sandbox, server._task_store.get("example-task"), b"patch", tmp_path)

    assert result.evaluation_completed is (expected_error is None)
    assert result.reward == 0.0
    if expected_error is None:
        assert result.verifier_exit_code == 0
        assert result.verifier_error is None
    else:
        assert expected_error in (result.verifier_error or "")
    assert sandbox.download.await_count == (0 if error_type == "timeout" else 3)


@pytest.mark.parametrize("status", [(1, None), (124, "timeout"), (2, None)])
async def test_verifier_artifact_probe_distinguishes_missing_file_from_failure(
    task_assets: Path, tmp_path: Path, status: tuple[int, str | None]
) -> None:
    server = DeepSWEResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(return_code=status[0], error_type=status[1])
    artifact = tmp_path / "reward.json"

    if status == (1, None):
        assert await server._download_if_present(sandbox, "/logs/verifier/reward.json", artifact) is False
    else:
        with pytest.raises(RuntimeError, match="Could not check DeepSWE verifier artifact"):
            await server._download_if_present(sandbox, "/logs/verifier/reward.json", artifact)

    sandbox.download.assert_not_awaited()


def _sandbox_failure(kind: str, reason: str) -> NgFailureEvent:
    return NgFailureEvent(
        boundary=FailureBoundary.SANDBOX,
        failure_kind=kind,
        failure_reason=reason,
        operation="exec",
        details=SandboxFailureDetails(provider="opensandbox", sandbox_id="agent-sandbox"),
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_report",
        "missing_failure_id",
        "terminal_report",
        "wrong_details",
        "wrong_provider",
        "observation_provider",
        "observation_sandbox_id",
        "omitted_report",
        "missing_event",
        "mismatched_id",
        "wrong_boundary",
        "wrong_kind",
        "missing_details",
        "missing_provider",
        "missing_sandbox_id",
        "golden",
        "nonempty_response",
        "setup_stage",
        "missing_stage",
        "event_stage",
        "wrong_task",
        "wrong_image",
    ],
)
@pytest.mark.parametrize("action", ["fail", "count_as_zero"])
async def test_agent_oom_scoring_requires_matching_canonical_execution_evidence(
    task_assets: Path, monkeypatch: MonkeyPatch, invalid: str, action: str
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False, agent_policy={"by_failure_kind": {PROVIDER_OOM_KILLED: action}}),
        server_client=MagicMock(spec=ServerClient),
    )
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    collect = AsyncMock()
    monkeypatch.setattr(server, "_collect_model_patch", collect)
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}
    await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))
    body = _agent_oom_request()
    report = body["failure_report"]
    [event] = report["events"].values()
    if invalid == "missing_report":
        body.pop("failure_report")
    elif invalid == "missing_failure_id":
        observation = SandboxObservation(role="agent", stage="agent", outcome="oom")
        body["response"]["metadata"][NG_AGENT_SANDBOX_OBSERVATION_KEY] = observation.model_dump_json()
    elif invalid == "terminal_report":
        report["failure_id"] = event["event_id"]
    elif invalid == "wrong_details":
        event["details"] = {"model": "test"}
    elif invalid == "wrong_provider":
        event["details"]["provider"] = "other"
    elif invalid in ("observation_provider", "observation_sandbox_id"):
        observation = SandboxObservation(
            role="agent",
            stage="agent",
            outcome="oom",
            failure_id=event["event_id"],
            **{invalid.removeprefix("observation_"): "other"},
        )
        body["response"]["metadata"][NG_AGENT_SANDBOX_OBSERVATION_KEY] = observation.model_dump_json()
    elif invalid == "omitted_report":
        report["omitted"] = True
    elif invalid == "missing_event":
        report["events"] = {}
    elif invalid == "mismatched_id":
        observation = SandboxObservation(role="agent", stage="agent", outcome="oom", failure_id="other")
        body["response"]["metadata"][NG_AGENT_SANDBOX_OBSERVATION_KEY] = observation.model_dump_json()
    elif invalid == "wrong_boundary":
        event["boundary"] = "resources_server"
        event["details"] = None
    elif invalid == "wrong_kind":
        event["failure_kind"] = "provider_unavailable"
    elif invalid == "missing_details":
        event["details"] = None
    elif invalid == "missing_provider":
        event["details"]["provider"] = None
    elif invalid == "missing_sandbox_id":
        event["details"]["sandbox_id"] = None
    elif invalid == "golden":
        server.config.is_verifying_golden_patch = True
    elif invalid == "nonempty_response":
        body["response"]["output"] = [
            {"type": "message", "id": "message", "role": "assistant", "status": "completed", "content": []}
        ]
    elif invalid == "setup_stage":
        observation = SandboxObservation(role="agent", stage="seed", outcome="oom", failure_id=event["event_id"])
        body["response"]["metadata"][NG_AGENT_SANDBOX_OBSERVATION_KEY] = observation.model_dump_json()
    elif invalid == "missing_stage":
        observation = SandboxObservation(role="agent", outcome="oom", failure_id=event["event_id"])
        body["response"]["metadata"][NG_AGENT_SANDBOX_OBSERVATION_KEY] = observation.model_dump_json()
    elif invalid == "event_stage":
        event["stage"] = "seed"
    elif invalid == "wrong_task":
        server._agent_sessions["test-session"].task_id = "other"
    elif invalid == "wrong_image":
        server._agent_sessions["test-session"].image = "other"
    with pytest.raises((ValueError, RuntimeError)):
        await server.verify(request, DeepSWEVerifyRequest.model_validate(body))
    collect.assert_not_awaited()


@pytest.mark.parametrize("return_code,error_type", [(1, None), (0, "timeout")])
async def test_failed_agent_git_setup_cannot_start_a_session(
    task_assets: Path, monkeypatch: MonkeyPatch, return_code: int, error_type: str | None
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False), server_client=MagicMock(spec=ServerClient)
    )
    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(return_code=return_code, error_type=error_type, stderr="setup failed")
    monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: MagicMock())
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=sandbox))
    await server._create_sandbox(server._task_store.get("example-task"), phase="agent")
    setup = sandbox.start_with_setup.await_args.args[1]
    with pytest.raises(RuntimeError, match="DeepSWE agent git setup failed"):
        await setup(sandbox)
    assert "|| true" not in sandbox.exec.await_args.args[0]


@pytest.mark.parametrize("return_code,error_type", [(1, None), (0, "timeout")])
async def test_failed_patch_capture_is_never_downloaded(
    task_assets: Path, return_code: int, error_type: str | None
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False), server_client=MagicMock(spec=ServerClient)
    )
    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(
        return_code=return_code, error_type=error_type, stdout="", stderr="failed"
    )
    with pytest.raises(RuntimeError, match="DeepSWE collect hook"):
        await server._collect_model_patch(sandbox, server._task_store.get("example-task"))
    sandbox.download.assert_not_awaited()


async def test_verifier_timeout_cause_survives_the_masked_failure_report(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    sandbox = AsyncMock()
    error = TimeoutError("verifier request timed out")
    sandbox.exec.side_effect = error
    observed = []
    observe = app_module.observe_failure

    def record_failure(failure: NgFailureEvent) -> None:
        observed.append(failure)
        observe(failure)

    monkeypatch.setattr(app_module, "observe_failure", record_failure)
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(server, "_stage_verifier", AsyncMock())
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}
    with RequestFailureLog.capture() as log:
        response = await server.verify(request, DeepSWEVerifyRequest.model_validate(_request()))
    assert response.evaluation_completed is False
    assert response.mask_sample is True
    assert response.verification_attempts == 3
    assert len(log.report.events) == 3
    assert all(event.foreign_error.exception_type == "TimeoutError" for event in log.report.events.values())
    assert all(event.__cause__ is error for event in observed)
    assert response.verifier_error == "TimeoutError: verifier request timed out"
    assert sandbox.stop.await_count == 3


async def test_diagnostic_oom_event_without_execution_observation_does_not_choose_a_score(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(
            task_assets,
            golden=False,
            agent_policy={
                "default_action": "fail",
                "by_failure_kind": {PROVIDER_OOM_KILLED: "count_as_zero"},
            },
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    agent_sandbox = AsyncMock()
    agent_sandbox.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    verifier_sandbox = AsyncMock()
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(side_effect=[agent_sandbox, verifier_sandbox]))
    collect = AsyncMock(return_value=b"valid patch")
    monkeypatch.setattr(server, "_collect_model_patch", collect)
    verifier = AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1.0))
    monkeypatch.setattr(server, "_run_verifier", verifier)
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test-session"}
    await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))
    body = _agent_oom_request()
    body["response"].pop("metadata")
    verify_body = DeepSWEVerifyRequest.model_validate(body)
    with RequestFailureLog.capture() as log:
        response = await server.verify(request, verify_body)
    assert log.report == verify_body.failure_report
    assert response.reward == 1.0
    assert response.evaluation_completed is True
    assert response.mask_sample is False
    assert response.failure_report == verify_body.failure_report
    collect.assert_awaited_once()
    verifier.assert_awaited_once()


@pytest.mark.parametrize("kind", [PROVIDER_OOM_KILLED, "provider_unavailable"])
async def test_recovered_verifier_keeps_original_sandbox_failure_without_changing_its_score(
    task_assets: Path, monkeypatch: MonkeyPatch, kind: str
) -> None:
    error = _sandbox_failure(kind, "first verifier sandbox failed")
    server, created = _rollout_server(
        task_assets,
        monkeypatch,
        verifier_results=[error, VerifierResult(evaluation_completed=True, reward=1.0)],
    )
    response, log = await _verify(server)

    assert (response.reward, response.evaluation_completed, response.mask_sample) == (1.0, True, False)
    assert response.failure_kind is None and response.verification_attempts == 2
    assert response.failure_outcome_action is None and response.failure_outcome_attribution is None
    restored = DeepSWEVerifyResponse.model_validate_json(response.model_dump_json())
    assert restored.reward == 1.0 and restored.mask_sample is False
    assert "failure_outcome_action" not in restored.model_dump(mode="json")
    assert "failure_outcome_attribution" not in restored.model_dump(mode="json")
    assert log.report.failure_id is None and not log.report.omitted
    assert log.report.events[error.event_id] == failure_report_for(error).events[error.event_id]
    [attempt] = [event for event in log.report.events.values() if event.boundary == FailureBoundary.RESOURCES_SERVER]
    assert attempt.cause_id == error.event_id
    assert attempt.attempt_id == f"{attempt.operation_id}:1"
    for sandbox in created:
        sandbox.stop.assert_awaited_once()


async def test_registered_verifier_command_deadline_is_not_retried(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    error = _sandbox_failure(failure_kinds.PROVIDER_COMMAND_TIMEOUT, "verifier command deadline elapsed")
    server, created = _rollout_server(task_assets, monkeypatch, verifier_results=[error])

    response, log = await _verify(server)

    assert (response.reward, response.evaluation_completed, response.mask_sample) == (0.0, False, True)
    assert response.verification_attempts == 1 and response.inconclusive_attempts == []
    assert server._run_verifier.await_count == 1
    [failure] = [event for event in log.report.events.values() if event.boundary == FailureBoundary.RESOURCES_SERVER]
    assert failure.cause_id == error.event_id
    assert log.report.events[error.event_id].failure_kind == failure_kinds.PROVIDER_COMMAND_TIMEOUT
    for sandbox in created:
        sandbox.stop.assert_awaited_once()


async def test_nonzero_verifier_without_a_verdict_remains_retryable(
    task_assets: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(config=_config(task_assets), server_client=MagicMock(spec=ServerClient))
    monkeypatch.setattr(server, "_stage_verifier", AsyncMock())
    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(return_code=1, error_type=None, stdout="", stderr="no verdict")

    result = await server._run_verifier(sandbox, server._task_store.get("example-task"), b"patch", tmp_path)

    assert result.evaluation_completed is False and result.reward == 0.0
    assert result.verifier_exit_code == 1 and result.verifier_error == NO_VERDICT
    assert app_module._inconclusive_reason(result, None) == NO_VERDICT
    sandbox.download.assert_not_awaited()


async def test_saved_patch_recovery_keeps_diagnostic_failure_without_grading_from_it(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    import hashlib

    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False), server_client=MagicMock(spec=ServerClient)
    )
    sandbox = AsyncMock()
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    collect = AsyncMock()
    monkeypatch.setattr(server, "_collect_model_patch", collect)
    monkeypatch.setattr(
        server, "_run_verifier", AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1.0))
    )
    body = _agent_oom_request()
    body["response"].pop("metadata")
    body |= {
        "saved_model_patch": "saved patch",
        "saved_model_patch_sha256": hashlib.sha256(b"saved patch").hexdigest(),
    }
    request = MagicMock()
    request.session = {}
    verify_body = DeepSWEVerifyRequest.model_validate(body)

    with RequestFailureLog.capture() as log:
        response = await server.verify(request, verify_body)

    assert (response.reward, response.evaluation_completed, response.mask_sample) == (1.0, True, False)
    assert response.replayed_saved_patch is True and response.verification_attempts == 1
    assert response.failure_report == verify_body.failure_report == log.report
    assert log.report.failure_id is None and not log.report.omitted
    collect.assert_not_awaited()
    sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("action", ["fail", "count_as_zero"])
@pytest.mark.parametrize("invalid", ["missing_report", "wrong_provider"])
async def test_rejected_oom_evidence_stops_only_its_owned_seeded_session(
    task_assets: Path, monkeypatch: MonkeyPatch, invalid: str, action: str
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=False, agent_policy={"by_failure_kind": {PROVIDER_OOM_KILLED: action}}),
        server_client=MagicMock(spec=ServerClient),
    )
    owned, other = AsyncMock(), AsyncMock()
    owned.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    other.serialize.return_value = {"sandbox_id": "other-sandbox"}
    create = AsyncMock(side_effect=[owned, other])
    monkeypatch.setattr(server, "_create_sandbox", create)
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "owned"}
    await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))
    other_request = MagicMock()
    other_request.session = {SESSION_ID_KEY: "other"}
    await server.seed_session(other_request, DeepSWESeedSessionRequest.model_validate(_request()))
    body = _agent_oom_request() | {"sandbox_handle": "foreign-borrowed-handle"}
    if invalid == "missing_report":
        body.pop("failure_report")
    elif invalid == "wrong_provider":
        next(iter(body["failure_report"]["events"].values()))["details"]["provider"] = "other"

    with pytest.raises(ValueError):
        await server.verify(request, DeepSWEVerifyRequest.model_validate(body))

    owned.stop.assert_awaited_once()
    other.stop.assert_not_awaited()
    assert "owned" not in server._agent_sessions
    assert server._agent_sessions["other"].sandbox is other
    assert create.await_count == 2


async def test_rejected_oom_evidence_keeps_its_original_error_when_owned_cleanup_fails(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server = DeepSWEResourcesServer(
        config=_config(
            task_assets,
            golden=False,
            agent_policy={
                "default_action": "fail",
                "by_failure_kind": {PROVIDER_OOM_KILLED: "count_as_zero"},
            },
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    owned = AsyncMock()
    owned.serialize.return_value = {"sandbox_id": "agent-sandbox"}
    owned.stop.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=owned))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "owned"}
    await server.seed_session(request, DeepSWESeedSessionRequest.model_validate(_request()))
    original = ValueError("invalid OOM evidence")
    cause = RuntimeError("original evidence cause")
    original.__cause__ = cause
    monkeypatch.setattr(app_module.SandboxObservation, "model_validate_json", MagicMock(side_effect=original))

    with pytest.raises(ValueError) as caught:
        await server.verify(request, DeepSWEVerifyRequest.model_validate(_agent_oom_request()))

    assert caught.value is original and caught.value.__cause__ is cause
    owned.stop.assert_awaited_once()
    assert "owned" not in server._agent_sessions


@pytest.mark.parametrize("action,attempts", [("retry", 3), ("fail", 1)])
async def test_verification_stage_policy_controls_bounded_retry_and_retains_evaluation_fault(
    task_assets: Path, monkeypatch: MonkeyPatch, action: str, attempts: int
) -> None:
    errors = [_sandbox_failure(PROVIDER_OOM_KILLED, f"verifier container OOM {i}") for i in range(3)]
    server, created = _rollout_server(task_assets, monkeypatch, verifier_results=errors)
    server.config.outcome_policy.by_stage["verification"].by_failure_kind[PROVIDER_OOM_KILLED] = action
    response, log = await _verify(server)
    assert response.verification_attempts == attempts
    assert response.evaluation_completed is False and response.mask_sample is True
    assert response.failure_outcome_action == OutcomeAction.FAIL
    assert response.failure_outcome_attribution == "evaluation"
    for error in errors[:attempts]:
        assert log.report.events[error.event_id] == failure_report_for(error).events[error.event_id]
    assert server._create_sandbox.await_count == attempts + 1
    for sandbox in created[: attempts + 1]:
        sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("stage", ["agent", "seed"])
async def test_verifier_does_not_reassign_a_known_failure_origin_or_claim_its_ownership(
    task_assets: Path, monkeypatch: MonkeyPatch, stage: str
) -> None:
    error = _sandbox_failure(PROVIDER_OOM_KILLED, "failure forwarded from another execution stage")
    error.stage = stage
    server, _ = _rollout_server(task_assets, monkeypatch, verifier_results=[error])
    response, log = await _verify(server)
    assert response.verification_attempts == 1
    assert response.failure_outcome_action == OutcomeAction.FAIL
    assert response.failure_outcome_attribution == "unknown"
    assert response.mask_sample is True and response.evaluation_completed is False
    assert log.report.events[error.event_id].stage == stage


@pytest.mark.parametrize("action,attempts", [("retry", 3), ("fail", 1)])
async def test_verifier_uses_shared_class_override_for_actions_and_saved_result(
    task_assets: Path, monkeypatch: MonkeyPatch, action: str, attempts: int
) -> None:
    errors = [_sandbox_failure("provider_unavailable", f"verifier unavailable {i}") for i in range(3)]
    server, _ = _rollout_server(task_assets, monkeypatch, verifier_results=errors)
    server.config = DeepSWEResourcesServerConfig.model_validate(
        server.config.model_dump()
        | {
            "outcome_policy": {
                "classification": {
                    "by_stage": {"verification": {"by_failure_kind": {"provider_unavailable": "model"}}}
                },
                "by_stage": {"verification": {"default_action": "fail", "by_failure_class": {"model": action}}},
            },
        }
    )
    response, log = await _verify(server)
    restored = DeepSWEVerifyResponse.model_validate_json(response.model_dump_json())
    assert response.verification_attempts == attempts
    assert restored.failure_outcome_attribution is FailureClass.MODEL
    assert restored.failure_outcome_action == OutcomeAction.FAIL
    assert restored.mask_sample is True and restored.evaluation_completed is False
    assert response.failure_report == restored.failure_report
    assert all(log.report.events[error.event_id].failure_kind == "provider_unavailable" for error in errors[:attempts])


def _environment_seed(
    session_id: str = "resources-session-1", task: str = "example-task"
) -> ResourcesSeedSessionRequest:
    """The seed single_agent_turn_legacy sends for a prepared DeepSWE row."""
    row = _request()
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="rollout-1"),
        task_id=TaskId(taskset="deepswe_opencode_resources_server", task_id=task),
        task_data={key: row[key] for key in ("task_id", "image", "verifier_metadata")} | {"split": "test"},
    )


def _environment_verify_body(seed: ResourcesSeedSessionRequest) -> dict:
    """The flat verify body single_agent_turn_legacy sends with the seed's session cookie."""
    return seed.task_data | {key: _request()[key] for key in ("responses_create_params", "response")}


def _close_body(seed: ResourcesSeedSessionRequest) -> dict:
    return ResourcesCloseSessionRequest(
        resources_session_id=seed.resources_session_id, episode_id=seed.episode_id
    ).model_dump(mode="json")


def _session_server(
    task_assets: Path, monkeypatch: MonkeyPatch, *, golden: bool = False
) -> tuple[DeepSWEResourcesServer, list[AsyncMock], AsyncMock]:
    """A server whose sandboxes are fakes, registered with their session as the real start does."""
    server = DeepSWEResourcesServer(
        config=_config(task_assets, golden=golden), server_client=MagicMock(spec=ServerClient)
    )
    sandboxes: list[AsyncMock] = []

    async def create_sandbox(task, *, phase: str, session_id: str | None = None) -> AsyncMock:
        sandbox = AsyncMock()
        sandbox.serialize.return_value = {"sandbox_id": f"{phase}-sandbox-{len(sandboxes) + 1}", "workdir": "/app"}
        sandboxes.append(sandbox)
        if session_id is not None:
            server._session_id_to_sandbox[session_id] = sandbox
        return sandbox

    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(side_effect=create_sandbox))
    collect_model_patch = AsyncMock(return_value=b"agent patch\n")
    monkeypatch.setattr(server, "_collect_model_patch", collect_model_patch)
    monkeypatch.setattr(
        server, "_run_verifier", AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1.0))
    )
    return server, sandboxes, collect_model_patch


def test_environment_session_follows_the_resources_session_contract(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, sandboxes, _ = _session_server(task_assets, monkeypatch)
    seed = _environment_seed()

    check_resources_session_contract(
        server.setup_webserver(),
        seed,
        keeps_state=True,
        holds_state=server.keeps_session_state,
        verify_body=_environment_verify_body(seed),
    )

    # The repeated seed lent the same sandbox, and close stopped it once.
    assert server._create_sandbox.await_args_list[0].kwargs == {
        "phase": "agent",
        "session_id": seed.resources_session_id,
    }
    sandboxes[0].stop.assert_awaited_once()
    assert seed.resources_session_id not in server._episode_agent_sandboxes


def test_environment_session_lends_the_agent_sandbox_and_verify_grades_its_capture(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, sandboxes, collect_model_patch = _session_server(task_assets, monkeypatch)
    seed = _environment_seed()
    verify_body = _environment_verify_body(seed)

    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=seed.model_dump(mode="json"))
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["sandbox_access"] == {
            "connection": {
                "kind": "direct",
                "provider_config_ref": "test",
                "descriptor": {"sandbox_id": "agent-sandbox-1", "workdir": "/app"},
            },
            "workdir": "/app",
        }
        # The agent may operate the sandbox but not destroy it: verify still collects the patch from it.
        sandboxes[0].serialize.assert_any_await(scope="operate")
        sandboxes[0].stop.assert_not_awaited()

        verified = client.post("/verify", json=verify_body)
        assert verified.status_code == 200, verified.text
        assert verified.json()["reward"] == 1.0
        assert verified.json()["model_patch"] == "agent patch\n"
        assert verified.json()["sandbox_handle"] == "agent-sandbox-1"
        assert collect_model_patch.await_args.args[0] is sandboxes[0]
        sandboxes[0].stop.assert_awaited_once()

        # An identical retry replays the verdict instead of grading an empty patch.
        repeated = client.post("/verify", json=verify_body)
        assert repeated.status_code == 200, repeated.text
        assert repeated.json() == verified.json()
        assert len(sandboxes) == 2 and collect_model_patch.await_count == 1

        # A sandbox that verify consumed cannot be lent again.
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 409
        assert client.post("/close_session", json=_close_body(seed)).status_code == 200
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 409
        lost = client.post("/verify", json=verify_body)
        assert lost.status_code == 200, lost.text
        assert lost.json()["mask_sample"] is True and lost.json()["failure_kind"] == SESSION_LOST

    sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {} and server._episode_agent_sandboxes == {}


def test_failed_stop_after_capture_is_retried_with_the_captured_patch(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, sandboxes, collect_model_patch = _session_server(task_assets, monkeypatch)
    seed = _environment_seed()
    verify_body = _environment_verify_body(seed)

    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.side_effect = [RuntimeError("provider unavailable"), None]

        assert client.post("/verify", json=verify_body).status_code == 500
        assert server._session_id_to_sandbox[seed.resources_session_id] is sandboxes[0]

        retried = client.post("/verify", json=verify_body)
        assert retried.status_code == 200, retried.text
        assert retried.json()["reward"] == 1.0
        assert retried.json()["model_patch"] == "agent patch\n"
        assert retried.json()["sandbox_handle"] == "agent-sandbox-1"

    collect_model_patch.assert_awaited_once()
    assert sandboxes[0].stop.await_count == 2


def test_failed_close_keeps_the_agent_sandbox_for_a_retry(task_assets: Path, monkeypatch: MonkeyPatch) -> None:
    server, sandboxes, _ = _session_server(task_assets, monkeypatch)
    seed = _environment_seed()

    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.side_effect = [RuntimeError("provider unavailable"), None]

        assert client.post("/close_session", json=_close_body(seed)).status_code == 500
        assert server._session_id_to_sandbox[seed.resources_session_id] is sandboxes[0]
        assert client.post("/close_session", json=_close_body(seed)).status_code == 200
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 409

    assert sandboxes[0].stop.await_count == 2
    assert server._session_id_to_sandbox == {}


def test_an_episode_session_that_is_never_closed_is_stopped_at_shutdown(
    task_assets: Path, monkeypatch: MonkeyPatch
) -> None:
    server, sandboxes, _ = _session_server(task_assets, monkeypatch)

    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_environment_seed().model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.assert_not_awaited()

    sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {}


@pytest.mark.parametrize("golden", [False, True])
def test_environment_seed_that_names_another_task_or_runs_in_golden_mode_is_rejected(
    task_assets: Path, monkeypatch: MonkeyPatch, golden: bool
) -> None:
    server, _, _ = _session_server(task_assets, monkeypatch, golden=golden)
    seed = _environment_seed() if golden else _environment_seed(task="other-task")

    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        response = client.post("/seed_session", json=seed.model_dump(mode="json"))

    assert response.status_code == 422, response.text
    server._create_sandbox.assert_not_awaited()
