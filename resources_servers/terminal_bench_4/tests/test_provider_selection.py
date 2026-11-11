# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider selection: a named sandbox block, and tasks that the selected provider cannot host."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.sandbox import SupportsSandboxNetwork
from resources_servers.terminal_bench_4 import environment as module
from resources_servers.terminal_bench_4.app import TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.environment import (
    ProviderCapabilityError,
    resolve_environment_provider,
)
from resources_servers.terminal_bench_4.models import SandboxedVerifyRequest
from resources_servers.terminal_bench_4.task import TaskSettings
from resources_servers.terminal_bench_4.tests.test_environment import environment_config, make_environment
from resources_servers.terminal_bench_4.tests.test_resources_sessions import Harness


class PlainProvider:
    """A provider without direct networking between sandboxes."""


class NetworkedProvider(SupportsSandboxNetwork):
    pass


NAMED = {"sandbox": {"plain": {"endpoint": "example.invalid"}, "default_metadata": {"pool": "a"}}}


@pytest.fixture
def named_blocks(monkeypatch):
    monkeypatch.setattr(module, "get_global_config_dict", lambda: NAMED)
    monkeypatch.setattr(module, "get_provider_class", lambda name: PlainProvider)
    return NAMED


def test_environment_config_accepts_a_name_or_an_inline_block():
    assert environment_config(sandbox_provider="sandbox").sandbox_provider == "sandbox"
    assert isinstance(environment_config().sandbox_provider, dict)


def test_an_inline_block_resolves_to_a_copy():
    inline = {"docker": {"image": "x"}}
    resolved = resolve_environment_provider(inline)
    assert resolved == inline and resolved is not inline


def test_a_named_block_resolves_with_its_default_metadata_and_leaves_the_global_config_alone(named_blocks):
    resolved = resolve_environment_provider("sandbox")
    assert resolved == {"plain": {"endpoint": "example.invalid"}, "default_metadata": {"pool": "a"}}
    resolved["plain"]["connection"] = "changed"
    assert named_blocks["sandbox"]["plain"] == {"endpoint": "example.invalid"}


def test_an_environment_uses_the_named_block(tmp_path, monkeypatch, named_blocks):
    env, *_ = make_environment(tmp_path, monkeypatch, config={"sandbox_provider": "sandbox"})
    assert env.provider_config == resolve_environment_provider("sandbox")
    assert env.build_spec().metadata["pool"] == "a"


def test_a_missing_name_fails_with_the_defined_blocks_listed(monkeypatch):
    monkeypatch.setattr(module, "get_global_config_dict", lambda: NAMED)
    with pytest.raises(ValueError, match=r"'absent' is not defined.*'sandbox'"):
        resolve_environment_provider("absent")


def test_the_server_refuses_to_start_on_a_missing_name(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "get_global_config_dict", lambda: NAMED)
    with pytest.raises(ValueError, match="'absent' is not defined"):
        Harness(tmp_path, monkeypatch, environment={"sandbox_provider": "absent"})
    Harness(tmp_path, monkeypatch, environment={"sandbox_provider": "sandbox"})


def test_compose_needs_a_provider_with_sandbox_networking(tmp_path, monkeypatch, named_blocks):
    with pytest.raises(ProviderCapabilityError, match="Compose needs direct networking.*'plain'"):
        make_environment(tmp_path, monkeypatch, compose=True, config={"sandbox_provider": "sandbox"})
    monkeypatch.setattr(module, "get_provider_class", lambda name: NetworkedProvider)
    env, *_ = make_environment(tmp_path, monkeypatch, compose=True, config={"sandbox_provider": "sandbox"})
    assert env.uses_compose


@pytest.mark.parametrize("compose", [False, True])
def test_offline_verification_needs_opensandbox_or_docker(tmp_path, monkeypatch, named_blocks, compose):
    monkeypatch.setattr(module, "get_provider_class", lambda name: NetworkedProvider)
    offline = {"verifier": {"environment": {"docker_image": "public/verifier", "network_mode": "no-network"}}}
    with pytest.raises(ProviderCapabilityError, match="Offline verification.*'plain'"):
        make_environment(
            tmp_path,
            monkeypatch,
            compose=compose,
            verifier=True,
            config={"sandbox_provider": "sandbox"},
            task_config=offline,
        )


def test_a_capability_error_is_a_value_error():
    assert issubclass(ProviderCapabilityError, ValueError)


@pytest.fixture
def server(tmp_path, monkeypatch):
    original = TerminalBench4ResourcesServer._prepare_session
    harness = Harness(tmp_path, monkeypatch, sandbox_provider_ref="task_sandbox")
    monkeypatch.setattr(TerminalBench4ResourcesServer, "_prepare_session", original)
    task = SimpleNamespace(
        name=harness.row["task_name"],
        config=TaskSettings.model_validate({"environment": {"docker_image": "a"}, "verifier": {"environment": {}}}),
    )
    monkeypatch.setattr(harness.server._loader, "load", AsyncMock(return_value=task))
    return harness


@pytest.mark.parametrize(
    ("error", "status"),
    [(ProviderCapabilityError("task cannot run here"), 422), (RuntimeError("image pull failed"), 503)],
)
async def test_setup_errors_map_to_a_deterministic_422_only_for_capability_errors(server, monkeypatch, error, status):
    from resources_servers.terminal_bench_4 import lifecycle

    build = MagicMock(side_effect=error)
    monkeypatch.setattr(lifecycle, "Environment", build)
    seed = server.seed()
    for _ in range(2):
        with pytest.raises(HTTPException) as raised:
            await server.server.seed_session(SimpleNamespace(session={}), seed)
        assert raised.value.status_code == status
        assert str(error) in raised.value.detail
    # The failure is replayed, never provisioned again.
    assert build.call_count == 1


async def test_a_capability_error_is_a_terminal_infrastructure_failure_in_the_verified_response(server, monkeypatch):
    from nemo_gym.core import NeMoGymResponse
    from nemo_gym.rollouts import NG_TERMINAL_KEY
    from resources_servers.terminal_bench_4 import lifecycle
    from resources_servers.terminal_bench_4.models import TerminalBench4RunRequest

    build = MagicMock(side_effect=ProviderCapabilityError("not here"))
    monkeypatch.setattr(lifecycle, "Environment", build)
    row = server.row
    seeded = await server.server.seed_session(
        SimpleNamespace(session={SESSION_ID_KEY: "cookie"}),
        TerminalBench4RunRequest(
            responses_create_params={"input": []},
            task_name=row["task_name"],
            task_ref=row["task_ref"],
            dataset_ref=row["dataset_ref"],
            rollout_id="rollout-1",
        ),
    )
    assert seeded.termination.reason == "infrastructure_error"
    assert seeded.termination.detail == "ProviderCapabilityError: not here"
    session = server.server._sessions[seeded.session_id]
    session.verify_body = SandboxedVerifyRequest(
        session_id=session.session_id,
        termination=seeded.termination,
        responses_create_params={"input": []},
        response=NeMoGymResponse(
            id="resp",
            created_at=0,
            model="policy",
            object="response",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        ),
        agent_started=False,
    )
    await server.server._finalize_session(session)
    verified = session.verified_response.model_dump()
    assert verified["_ng_failure_class"] == "infrastructure_error"
    assert verified["_ng_failure_type"] == "ProviderCapabilityError"
    assert verified[NG_TERMINAL_KEY] is True
    assert verified["reward"] == 0.0 and verified["evaluation_completed"] is False
