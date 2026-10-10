# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.base_responses_api_agent import AGENT_SESSION_COOKIE_KEY
from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseOutputMessage
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig, OpenCodeAgentRunRequest
from responses_api_agents.opencode_agent.runtime import OPENCODE_VERSION


def make_agent() -> OpenCodeAgent:
    return OpenCodeAgent(
        config=OpenCodeAgentConfig(
            host="localhost",
            port=8001,
            name="opencode",
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "policy"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def test_constructing_agent_never_installs_on_host() -> None:
    with patch("responses_api_agents.opencode_agent.app.ensure_opencode") as install:
        agent = make_agent()
    install.assert_not_called()
    assert agent._local_setup_task is None
    assert not (
        {"execution_mode", "opencode_max_context_window", "sandbox_timeout", "debug", "setup_timeout"}
        & type(agent.config).model_fields.keys()
    )


async def test_local_runtime_is_lazy_once_and_retries_failed_setup() -> None:
    with patch("responses_api_agents.opencode_agent.app.ensure_opencode") as install:
        agent = make_agent()
        install.assert_not_called()
        install.side_effect = RuntimeError("install failed")
        with pytest.raises(RuntimeError, match="install failed"):
            await agent._run_opencode("task", None)
        assert agent._local_setup_task is None
        install.side_effect = None
        await asyncio.gather(agent._ensure_local_runtime(), agent._ensure_local_runtime())
        await agent._ensure_local_runtime()
    assert install.call_count == 2
    install.assert_called_with(OPENCODE_VERSION)


async def test_local_install_does_not_block_loop_or_cancel_shared_attempt() -> None:
    started, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()

    def install(version: str) -> None:
        assert version == OPENCODE_VERSION
        assert threading.get_ident() != loop_thread
        started.set()
        assert release.wait(3), "event loop could not release the installer"

    with patch("responses_api_agents.opencode_agent.app.ensure_opencode", side_effect=install) as ensure:
        agent = make_agent()
        first = asyncio.create_task(agent._ensure_local_runtime())
        try:
            async with asyncio.timeout(2):
                while not started.is_set():
                    await asyncio.sleep(0.01)
            second = asyncio.create_task(agent._ensure_local_runtime())
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not agent._local_setup_task.cancelled()
        finally:
            release.set()
        await second
        ensure.assert_called_once()


async def test_failed_install_retries_after_all_waiters_cancel() -> None:
    started, release = threading.Event(), threading.Event()

    def fail_install(version: str) -> None:
        started.set()
        assert release.wait(3)
        raise RuntimeError("install failed after cancellation")

    with patch("responses_api_agents.opencode_agent.app.ensure_opencode", side_effect=fail_install) as install:
        agent = make_agent()
        waiter = asyncio.create_task(agent._ensure_local_runtime())
        try:
            async with asyncio.timeout(2):
                while not started.is_set():
                    await asyncio.sleep(0.01)
            setup = agent._local_setup_task
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="after cancellation"):
            await setup
        install.side_effect = None
        await agent._ensure_local_runtime()
        assert install.call_count == 2


@pytest.mark.parametrize("shipped_config", [False, True])
@pytest.mark.parametrize("path", ["/v1/responses", "/ng-rollout/local-smoke/v1/responses"])
def test_unseeded_responses_run_local_cli(shipped_config: bool, path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent()
    if shipped_config:
        monkeypatch.chdir(Path(__file__).resolve().parents[3])
        _, configs = GlobalConfigDictParser().load_extra_config_paths(
            [str(Path(__file__).resolve().parents[1] / "configs/opencode_agent.yaml")]
        )
        config = OmegaConf.merge(*configs, {"policy_model_name": "test-model"})
        agent.config = OpenCodeAgentConfig.model_validate(
            dict(config.opencode_agent.responses_api_agents.opencode_agent)
            | {"host": "localhost", "port": 8001, "name": "opencode"}
        )
    message = NeMoGymResponseOutputMessage(
        id="msg-local",
        role="assistant",
        type="message",
        status="completed",
        content=[{"type": "output_text", "text": "local result", "annotations": []}],
    )
    with (
        patch.object(
            agent,
            "_run_opencode",
            AsyncMock(
                return_value=(
                    [message],
                    {"input_tokens": 3, "output_tokens": 2},
                    "local-model",
                    AgentObservationBundle(source="opencode"),
                )
            ),
        ) as local,
        patch("responses_api_agents.opencode_agent.app.create_provider") as provider,
        TestClient(agent.setup_webserver()) as client,
    ):
        response = client.post(path, json={"input": "task", "temperature": 0.7})
    assert response.status_code == 200, response.text
    assert response.json()["output"][0]["content"][0]["text"] == "local result"
    assert response.json()["usage"]["total_tokens"] == 5
    local.assert_awaited_once()
    assert local.call_args.args == ("task", None)
    provider.assert_not_called()


async def test_unseeded_run_requires_resources_not_a_session() -> None:
    agent = make_agent()
    with pytest.raises(HTTPException) as error:
        await agent.run(
            Request({"type": "http", "session": {}}),
            OpenCodeAgentRunRequest(responses_create_params={"input": "task"}),
        )
    assert error.value.status_code == 422
    assert "resources_server" in error.value.detail
    agent.server_client.post.assert_not_called()


@pytest.mark.parametrize("marker", [None, "closed-session"])
async def test_invalid_session_never_falls_back_to_local(marker: str | None) -> None:
    agent = make_agent()
    request = Request({"type": "http", "session": {AGENT_SESSION_COOKIE_KEY: marker}})
    with patch.object(agent, "_create_episode", AsyncMock()) as local:
        with pytest.raises(HTTPException) as error:
            await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert error.value.status_code == 409
    local.assert_not_awaited()
