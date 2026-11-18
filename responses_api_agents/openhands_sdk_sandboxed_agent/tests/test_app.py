# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.core import AGENT_ROUTE_FAILED, NgFailureEvent
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.openhands_sdk_sandboxed_agent import app as app_module
from responses_api_agents.openhands_sdk_sandboxed_agent.app import (
    OpenHandsSDKSandboxedAgent,
    OpenHandsSDKSandboxedAgentConfig,
)


class _Request:
    """The parts of a Request that run() reads and rebinds: `run` sets `_cookies`, `responses` reads `cookies`."""

    def __init__(self) -> None:
        self.session = {SESSION_ID_KEY: "session"}
        self.state = SimpleNamespace()
        self._cookies: dict = {}

    @property
    def cookies(self) -> dict:
        return self._cookies


def _agent() -> OpenHandsSDKSandboxedAgent:
    config = OpenHandsSDKSandboxedAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        model_server=ModelServerRef(type="responses_api_models", name="model"),
    )
    return OpenHandsSDKSandboxedAgent(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))


def _exec(return_code: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(return_code=return_code, stdout=stdout, stderr=stderr, error_type=None)


@pytest.fixture
def world(tmp_path, monkeypatch: MonkeyPatch):
    """An agent whose seed returns a sandbox handle and whose sandbox is a mock; /verify calls are recorded."""
    agent = _agent()
    sandbox = AsyncMock()
    sandbox.exec.return_value = _exec(0, "OPENHANDS_INSTALLED")
    monkeypatch.setattr(app_module, "__file__", str(tmp_path / "app.py"))  # results/ goes under tmp_path
    monkeypatch.setattr(OpenHandsSDKSandboxedAgent, "_connect_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(OpenHandsSDKSandboxedAgent, "rollout_id_from_run", lambda self, body: "rollout")
    monkeypatch.setattr(
        OpenHandsSDKSandboxedAgent, "resolve_model_base_url", lambda self, name, rollout_id: "http://model/v1"
    )
    monkeypatch.setattr(app_module, "raise_for_status", AsyncMock())
    seed = AsyncMock()
    seed.cookies = {}
    seed.json.return_value = {"sandbox_handle": "sandbox-1"}
    posts: list[str] = []

    async def post(*, url_path: str, **kwargs):
        posts.append(url_path)
        if url_path == "/seed_session":
            return seed
        raise AssertionError(f"unexpected call to {url_path}")

    agent.server_client.post = post
    request = _Request()
    body = MagicMock()
    body.model_dump.return_value = {}
    body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "build it"}]
    )
    return SimpleNamespace(agent=agent, sandbox=sandbox, posts=posts, request=request, body=body)


async def test_install_failure_is_a_failed_setup_and_is_never_verified(world) -> None:
    world.sandbox.exec.return_value = _exec(1, "", "no network")

    with pytest.raises(NgFailureEvent) as caught:
        await world.agent.run(world.request, world.body)

    assert caught.value.failure_kind == AGENT_ROUTE_FAILED
    assert caught.value.operation == "openhands.setup"
    assert "install failed" in caught.value.failure_reason
    assert world.posts == ["/seed_session"]  # no /verify: an untouched workspace must not score as a 0
    world.sandbox.stop.assert_awaited_once()
    assert world.agent._sandbox_id_to_sandbox == {}


async def test_install_exec_error_is_also_a_failed_setup(world) -> None:
    world.sandbox.exec.side_effect = TimeoutError("install timed out")

    with pytest.raises(NgFailureEvent) as caught:
        await world.agent.run(world.request, world.body)

    assert caught.value.operation == "openhands.setup"
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert world.posts == ["/seed_session"]
    world.sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("error", [RuntimeError("verify failed"), asyncio.CancelledError()])
async def test_sandbox_is_stopped_when_verification_raises_or_is_cancelled(
    world, monkeypatch: MonkeyPatch, error: BaseException
) -> None:
    monkeypatch.setattr(
        OpenHandsSDKSandboxedAgent, "responses", AsyncMock(return_value=MagicMock(model_dump=lambda **kw: {}))
    )
    monkeypatch.setattr(
        app_module,
        "OpenHandsSDKSandboxedAgentVerifyRequest",
        MagicMock(model_validate=MagicMock(return_value=MagicMock(model_dump=lambda: {}))),
    )
    seed_post = world.agent.server_client.post

    async def post(*, url_path: str, **kwargs):
        if url_path == "/verify":
            world.posts.append(url_path)
            raise error
        return await seed_post(url_path=url_path, **kwargs)

    world.agent.server_client.post = post
    world.agent._sandbox_id_to_run_result["session"] = {"left": "over"}

    with pytest.raises(type(error)):
        await world.agent.run(world.request, world.body)

    assert world.posts == ["/seed_session", "/verify"]
    world.sandbox.stop.assert_awaited_once()
    assert world.agent._sandbox_id_to_sandbox == {}
    assert world.agent._sandbox_id_to_run_result == {}
