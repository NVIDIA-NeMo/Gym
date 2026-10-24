# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check policy selection at Resources-owned creation, not live network isolation."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.swebench_pro.app import (
    SWEBenchProSeedSessionRequest,
    SWEBenchProTaskNetworkPolicy,
    SWEBenchProVerifyRequest,
)
from resources_servers.swebench_pro.tests.test_app import make_server, request_body
from resources_servers.swebench_pro.verification import VerificationResult


MODEL_POLICY = {"model_server": {"type": "responses_api_models", "name": "policy_model"}}


def sandbox_double() -> SimpleNamespace:
    return SimpleNamespace(
        _handle=SimpleNamespace(sandbox_id="sandbox-id"),
        start=AsyncMock(),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr="")),
        serialize=AsyncMock(return_value={"sandbox_id": "sandbox-id"}),
        stop=AsyncMock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", [False, True])
async def test_task_only_policy_is_present_before_handoff_and_does_not_change_grading(
    monkeypatch: MonkeyPatch, typed: bool
) -> None:
    server = make_server(golden=False, apply_anti_cheating=False, task_network_policy=MODEL_POLICY)
    original_options = {
        "platform": {"os": "linux", "architecture": "amd64"},
        "network_policy": {"defaultAction": "allow", "egress": []},
    }
    server.config.sandbox_config["provider_options"] = deepcopy(original_options)
    monkeypatch.setattr(
        "resources_servers.swebench_pro.app.get_global_config_dict", lambda: {"test": {"opensandbox": {}}}
    )
    endpoint = MagicMock(return_value="http://model.example:8000/rollouts/one/v1")
    monkeypatch.setattr("nemo_gym.sandbox.agent_tools.get_server_url", endpoint)
    task_sandbox, grading_sandbox = sandbox_double(), sandbox_double()
    create = MagicMock(side_effect=[task_sandbox, grading_sandbox])
    monkeypatch.setattr("resources_servers.swebench_pro.app.AsyncSandbox", create)
    monkeypatch.setattr(server, "_extract_model_patch", AsyncMock(return_value="agent patch"))
    verify = AsyncMock(
        return_value=VerificationResult(completed=True, resolved=True, patch_applied=True, test_results={"tests": []})
    )
    monkeypatch.setattr("resources_servers.swebench_pro.app.run_verification", verify)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})
    if typed:
        body = ResourcesSeedSessionRequest(
            resources_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="swebench_pro", task_id="instance_example"),
            task_data=request_body(),
        )
    else:
        body = SWEBenchProSeedSessionRequest.model_validate(request_body())

    response = await server.seed_session(request, body)
    task_sandbox.start.assert_awaited_once()
    task_spec = task_sandbox.start.await_args.args[0]
    assert task_spec.provider_options == original_options | {
        "network_policy": {
            "defaultAction": "deny",
            "egress": [{"action": "allow", "target": "model.example"}],
        }
    }
    assert task_spec.image == "docker.io/jefzda/sweap-images:example-tag"
    assert server._session_id_to_sandbox["session"] is task_sandbox
    if typed:
        assert response.sandbox_access.connection.descriptor == {"sandbox_id": "sandbox-id"}
    else:
        assert response.sandbox_handle == "sandbox-id"
    endpoint.assert_called_once_with("policy_model")

    result = await server.verify(request, SWEBenchProVerifyRequest.model_validate(request_body()))
    grading_sandbox.start.assert_awaited_once()
    grading_spec = grading_sandbox.start.await_args.args[0]
    assert grading_spec.provider_options == original_options
    assert grading_spec.image == task_spec.image
    assert grading_spec.files  # Fresh verifier inputs, not the agent workspace.
    assert verify.await_args.kwargs["sandbox"] is grading_sandbox
    assert result.reward == 1
    assert server.config.sandbox_config["provider_options"] == original_options
    endpoint.assert_called_once_with("policy_model")
    grading_sandbox.stop.assert_awaited_once()
    await server.shutdown()
    task_sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["docker", "opensandbox"])
@pytest.mark.parametrize("options", [{}, None, {"network_policy": {"defaultAction": "allow", "egress": []}}])
async def test_default_task_and_grading_networking_is_unchanged(
    monkeypatch: MonkeyPatch, provider: str, options: dict[str, object] | None
) -> None:
    server = make_server(golden=True, apply_anti_cheating=False)
    assert server.config.task_network_policy is None
    server.config.sandbox_config["provider_options"] = deepcopy(options)
    monkeypatch.setattr("resources_servers.swebench_pro.app.get_global_config_dict", lambda: {"test": {provider: {}}})
    endpoint = MagicMock(side_effect=AssertionError("Unrestricted recipes must not resolve a model endpoint"))
    monkeypatch.setattr("nemo_gym.sandbox.agent_tools.get_server_url", endpoint)
    task_sandbox, grading_sandbox = sandbox_double(), sandbox_double()
    monkeypatch.setattr(
        "resources_servers.swebench_pro.app.AsyncSandbox", MagicMock(side_effect=[task_sandbox, grading_sandbox])
    )
    monkeypatch.setattr(
        "resources_servers.swebench_pro.app.run_verification",
        AsyncMock(
            return_value=VerificationResult(
                completed=True, resolved=True, patch_applied=True, test_results={"tests": []}
            )
        ),
    )
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})

    await server.seed_session(request, SWEBenchProSeedSessionRequest.model_validate(request_body()))
    await server.verify(request, SWEBenchProVerifyRequest.model_validate(request_body()))

    assert task_sandbox.start.await_args.args[0].provider_options == options
    assert grading_sandbox.start.await_args.args[0].provider_options == options
    endpoint.assert_not_called()
    await server.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,endpoint,error",
    [
        ("docker", "http://model.example:8000", "requires the OpenSandbox"),
        ("e2b", "http://model.example:8000", "requires the OpenSandbox"),
        ("opensandbox", "http://localhost:8000", "sandbox-reachable host"),
        ("opensandbox", "http://127.0.0.1:8000", "sandbox-reachable host"),
        ("opensandbox", "http://0.0.0.0:8000", "sandbox-reachable host"),
        ("opensandbox", "http://[::1]:8000", "sandbox-reachable host"),
    ],
)
async def test_unsupported_policy_fails_before_creation_or_handoff(
    monkeypatch: MonkeyPatch, provider: str, endpoint: str, error: str
) -> None:
    server = make_server(golden=False, task_network_policy=MODEL_POLICY)
    monkeypatch.setattr("resources_servers.swebench_pro.app.get_global_config_dict", lambda: {"test": {provider: {}}})
    monkeypatch.setattr("nemo_gym.sandbox.agent_tools.get_server_url", lambda name: endpoint)
    create = MagicMock()
    monkeypatch.setattr("resources_servers.swebench_pro.app.AsyncSandbox", create)

    with pytest.raises(ValueError, match=error):
        await server._seed_session("session", SWEBenchProSeedSessionRequest.model_validate(request_body()))

    create.assert_not_called()
    assert not server._session_id_to_sandbox


def test_unknown_task_policy_settings_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SWEBenchProTaskNetworkPolicy.model_validate(MODEL_POLICY | {"allow_internet": True})


@pytest.mark.asyncio
async def test_policy_rejection_from_provider_never_retries_unrestricted(monkeypatch: MonkeyPatch) -> None:
    server = make_server(golden=False, task_network_policy=MODEL_POLICY)
    monkeypatch.setattr(
        "resources_servers.swebench_pro.app.get_global_config_dict", lambda: {"test": {"opensandbox": {}}}
    )
    monkeypatch.setattr("nemo_gym.sandbox.agent_tools.get_server_url", lambda name: "http://model.example:8000")
    sandbox = sandbox_double()
    sandbox.start.side_effect = RuntimeError("OpenSandbox has no egress image configured")
    create = MagicMock(return_value=sandbox)
    monkeypatch.setattr("resources_servers.swebench_pro.app.AsyncSandbox", create)

    with pytest.raises(RuntimeError, match="no egress image"):
        await server._seed_session("session", SWEBenchProSeedSessionRequest.model_validate(request_body()))

    create.assert_called_once()
    sandbox.start.assert_awaited_once()
    assert sandbox.start.await_args.args[0].provider_options["network_policy"]["defaultAction"] == "deny"
    assert not server._session_id_to_sandbox
