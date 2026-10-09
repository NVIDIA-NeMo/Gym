# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hermes agent sessions that send sandboxed model calls through a configured sandbox session capture."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi import HTTPException
from pydantic import ValidationError

import responses_api_agents.hermes_agent.app as hermes_app
from nemo_gym.agent_utils.sandbox_session import HARNESS_NOT_RUN_GAP
from nemo_gym.agent_utils.sandbox_session_capture import SandboxSessionCapture, SandboxSessionCaptureConfig
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    ModelEndpoint,
    TokenCapture,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig


# Lifecycle calls of FakeCapture, in order, for the current test.
EVENTS: list[str] = []
CAPTURE_URL = "http://127.0.0.1:4321/v1"
CAPTURED = TokenCapture(atif_trajectories=[{"steps": []}], metrics={"calls": 1})


class FakeCapture(SandboxSessionCapture):
    """A sandbox session capture whose start succeeds or raises, and which may dictate the model name."""

    def __init__(self, *, start: str = "ok", model: str | None = None):
        self.start_mode, self.model = start, model

    async def start(self, sandbox: AsyncSandbox) -> ModelEndpoint:
        EVENTS.append("capture.start")
        if self.start_mode == "raise":
            raise RuntimeError("capture port in use")
        return ModelEndpoint(base_url=CAPTURE_URL, model=self.model)

    async def collect(self, sandbox: AsyncSandbox) -> TokenCapture:
        EVENTS.append("capture.collect")
        return CAPTURED

    async def abort(self, sandbox: AsyncSandbox) -> None:
        EVENTS.append("capture.abort")


def _capture(**options) -> SandboxSessionCaptureConfig:
    return SandboxSessionCaptureConfig(implementation=f"{__name__}:FakeCapture", options=options)


def _config(**kwargs) -> HermesAgentConfig:
    fields = {
        "host": "127.0.0.1",
        "port": 8080,
        "name": "hermes",
        "entrypoint": "app.py",
        "model": "policy",
        "model_server": None,
        "sandbox_session_capture": _capture(),
    }
    return HermesAgentConfig(**(fields | kwargs))


class _Sandbox:
    """A borrowed task sandbox whose Hermes runner completes with one model call."""

    def __init__(self) -> None:
        self.uploaded: dict[str, str] = {}
        self.commands: list[str] = []
        self.disconnect = AsyncMock()

    async def upload(self, local_path, remote_path) -> None:
        self.uploaded[remote_path] = Path(local_path).read_text()

    # Mirrors AsyncSandbox.exec so an unsupported argument fails here too.
    async def exec(
        self, command, *, cwd=None, env=None, timeout_s=180, user=None, preserve_background_services=False
    ) -> SandboxExecResult:
        self.commands.append(command)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def download(self, remote_path, local_path) -> None:
        if remote_path.endswith("/cleanup.json"):
            payload = {"cleanup_confirmed": True, "error": None}
        else:
            messages = [{"role": "user", "content": "fix bug"}, {"role": "assistant", "content": "done"}]
            invocation = {
                "invocation_id": "root",
                "status": "completed",
                "messages": messages,
                "model_response_ids": ["resp-from-capture"],
            }
            payload = {
                "result": {"messages": messages, "final_response": "done"},
                "runtime": {"hostname": "sandbox", "pid": 1, "python": "python"},
                "observations": {"invocations": [invocation]},
            }
        Path(local_path).write_text(json.dumps(payload), encoding="utf-8")

    def runner_input(self) -> dict:
        return json.loads(next(text for path, text in self.uploaded.items() if path.endswith("/input.json")))


@pytest.fixture
def seeded(monkeypatch):
    """Seed a borrowed-sandbox session through the agent's own setup; returns the agent, sandbox and seed."""
    EVENTS.clear()
    sandbox = _Sandbox()
    factory = MagicMock()
    factory.connect = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(hermes_app, "AsyncSandbox", factory)
    monkeypatch.setattr(hermes_app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(hermes_app, "resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(hermes_app, "create_provider", lambda config: AsyncMock())
    seed = AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id=EpisodeId(rollout_id="episode", attempt=1),
        task_id=TaskId(taskset="test", task_id="task"),
        sandbox_access={
            "connection": {"kind": "direct", "provider_config_ref": "provider", "descriptor": {"sandbox_id": "box"}},
            "workdir": "/app",
        },
    )

    async def seed_with(config: HermesAgentConfig) -> HermesAgent:
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        hermes = HermesAgent(config=config, server_client=server_client)
        await hermes.seed_agent_session(SimpleNamespace(session={}), seed)
        return hermes

    request = SimpleNamespace(
        session={"agent_session_id": "session"}, path_params={"rollout_id": seed.episode_id.capture_key}
    )
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=seed.episode_id)
    return SimpleNamespace(seed_with=seed_with, sandbox=sandbox, request=request, close=close)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"model_server": {"type": "responses_api_models", "name": "policy_model"}}, "set model_server: null"),
        ({"sandbox_session_capture": None}, "needs model_server or sandbox_session_capture"),
        ({"model": None}, "needs model when model_server is null"),
    ],
)
def test_config_requires_exactly_one_model_endpoint(fields, message):
    with pytest.raises(ValidationError, match=message):
        _config(**fields)


def test_config_accepts_model_server_without_capture():
    config = _config(
        sandbox_session_capture=None, model_server={"type": "responses_api_models", "name": "policy_model"}
    )
    assert config.sandbox_session_capture is None


@pytest.mark.parametrize(("capture_model", "harness_model"), [(None, "policy"), ("served", "served")])
async def test_activation_calls_the_capture_endpoint_and_close_returns_its_capture(
    seeded, capture_model, harness_model
):
    hermes = await seeded.seed_with(_config(sandbox_session_capture=_capture(model=capture_model)))
    # The capture starts at seed, after the runtime is installed and before any harness input exists.
    assert EVENTS == ["capture.start"]
    assert not any(path.endswith("/input.json") for path in seeded.sandbox.uploaded)

    response = await hermes.responses(seeded.request, NeMoGymResponseCreateParamsNonStreaming(input="fix bug"))

    runner_input = seeded.sandbox.runner_input()
    assert runner_input["model_base_url"] == CAPTURE_URL
    assert runner_input["model"] == harness_model
    assert yaml.safe_load(runner_input["config_yaml"])["model"] == harness_model
    assert response.status == "completed" and response.model == harness_model

    result = await hermes.close_agent_session(SimpleNamespace(session={}), seeded.close)
    assert EVENTS == ["capture.start", "capture.collect"]
    assert result.token_capture == CAPTURED
    # Response IDs from the capture do not identify Gym model server calls.
    (invocation,) = result.agent_observations.records
    assert invocation.model_calls == []
    assert [gap.code for gap in result.agent_observations.gaps] == ["model_call_join_key_unavailable"]


async def test_failed_capture_start_answers_without_running_hermes(seeded):
    hermes = await seeded.seed_with(_config(sandbox_session_capture=_capture(start="raise")))
    commands_after_seed = list(seeded.sandbox.commands)

    response = await hermes.responses(seeded.request, NeMoGymResponseCreateParamsNonStreaming(input="fix bug"))

    # A failed response, not an error, so the environment server still verifies the episode.
    assert response.status == "failed"
    assert "capture did not start: RuntimeError: capture port in use" in response.error.message
    assert not any(path.endswith("/input.json") for path in seeded.sandbox.uploaded)
    assert seeded.sandbox.commands == commands_after_seed

    result = await hermes.close_agent_session(SimpleNamespace(session={}), seeded.close)
    assert EVENTS == ["capture.start", "capture.abort"]
    assert result.token_capture is not None and result.token_capture.masked
    assert "capture port in use" in result.token_capture.mask_reason
    (gap,) = result.agent_observations.gaps
    assert gap.code == HARNESS_NOT_RUN_GAP and "capture port in use" in gap.detail
    seeded.sandbox.disconnect.assert_awaited_once()


async def test_host_execution_needs_a_model_server():
    hermes = HermesAgent(config=_config(), server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    with pytest.raises(HTTPException, match="host execution needs model_server") as raised:
        await hermes._create_response(NeMoGymResponseCreateParamsNonStreaming(input="fix bug"))
    assert raised.value.status_code == 422
