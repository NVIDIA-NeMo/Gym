# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M5 local-CLI conformance; external CLI execution is a separate smoke gate."""

import asyncio
import importlib
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.episode import AgentCloseSessionRequest, AgentSeedSessionRequest, EpisodeId, TaskId
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


HARNESSES = [
    ("opencode_agent", "OpenCodeAgent", "ensure_opencode", "_run_opencode"),
    ("openclaw_agent", "OpenClawAgent", "ensure_openclaw", "_run_openclaw"),
    ("pi_agent", "PiAgent", "ensure_pi", "_run_pi"),
    ("codex_agent", "CodexAgent", "ensure_codex", "_run_codex"),
    ("claude_code_agent", "ClaudeCodeAgent", "ensure_claude_code", "_run_claude_code"),
    ("cline_agent", "ClineAgent", "ensure_cline", "_run_cline"),
    ("kilocode_agent", "KiloCodeAgent", "ensure_kilo", "_run_kilo"),
]


@pytest.fixture(params=HARNESSES, ids=[case[0] for case in HARNESSES])
def cli(request: pytest.FixtureRequest, tmp_path: Path):
    module_name, class_name, installer, runner = request.param
    module = importlib.import_module(f"responses_api_agents.{module_name}.app")
    cls = getattr(module, class_name)
    config_cls = cls.model_fields["config"].annotation
    settings = {
        "host": "127.0.0.1",
        "port": 8080,
        "name": "agent",
        "entrypoint": "app.py",
        "resources_server": {"type": "resources_servers", "name": "verifier"},
        "model_server": {"type": "responses_api_models", "name": "policy"},
        "model": "test-model",
        "concurrency": 1,
    }
    if "workspace_root" in config_cls.model_fields:
        settings["workspace_root"] = str(tmp_path / "workspaces")
    if module_name == "codex_agent":
        settings["codex_version"] = "0.144.4"
    if module_name == "openclaw_agent":
        settings["openclaw_version"] = "2026.6.11"
    server = MagicMock(spec=ServerClient)
    server.global_config_dict = OmegaConf.create(
        {
            "observability_enabled": True,
            "policy": {"responses_api_models": {"vllm_model": {"host": "model", "port": 9000}}},
        }
    )
    server._build_server_base_url.return_value = "http://model:9000"
    with ExitStack() as patches:
        patches.enter_context(patch.object(module, installer))
        patches.enter_context(patch("subprocess.run", return_value=SimpleNamespace(stdout="test-version")))
        agent = cls(config=config_cls(**settings), server_client=server)
    return agent, module, runner


def params() -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(input="Solve the task", model="test-model")


def seed(rollout: str = "rollout") -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        episode_id=EpisodeId(rollout_id=rollout, attempt=1),
        task_id=TaskId(task_source="verifier", task_id="task"),
        resources_access={"kind": "direct_http", "base_url": "http://verifier", "cookies": {"res": "private"}},
    )


def request_for(session_id: str = "session", rollout: str = "rollout") -> Request:
    key = seed(rollout).episode_id.capture_key
    return Request(
        {
            "type": "http",
            "path": f"/ng-rollout/{key}/training-token-capture/v1/responses",
            "headers": [],
            "session": {SESSION_ID_KEY: session_id},
            "path_params": {"rollout_id": key},
        }
    )


def mock_runner(agent, runner: str) -> AsyncMock:
    output = [
        NeMoGymResponseOutputMessage(
            id="answer",
            type="message",
            role="assistant",
            status="completed",
            content=[NeMoGymResponseOutputText(type="output_text", text="42", annotations=[])],
        )
    ]
    usage = {"input_tokens": 7, "output_tokens": 3}
    if runner == "_run_opencode":
        result = (output, usage, "test-model", AgentObservationBundle(source="opencode"))
    elif runner == "_run_pi":
        result = (output, usage, "test-model", [])
    elif runner == "_run_claude_code":
        result = (output, "test-model", usage)
    elif runner == "_run_codex":
        result = (
            "\n".join(
                json.dumps(event)
                for event in [
                    {"type": "item.completed", "item": {"type": "agent_message", "id": "answer", "text": "42"}},
                    {"type": "turn.completed", "usage": usage},
                ]
            ),
            "test-model",
        )
    else:
        result = (output, usage, "test-model")
    mocked = AsyncMock(return_value=result)
    setattr(agent, runner, mocked)
    return mocked


@pytest.mark.parametrize("capture_segment", ["", "/training-token-capture"])
def test_http_seed_activate_close_preserves_response_usage_and_observations(cli, capture_segment):
    agent, _, runner = cli
    mocked = mock_runner(agent, runner)
    with TestClient(agent.setup_webserver()) as client:
        seeded = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert seeded.status_code == 200
        session_id = seeded.json()["agent_session_id"]
        key = seed().episode_id.capture_key
        response = client.post(
            f"/ng-rollout/{key}{capture_segment}/v1/responses", json=params().model_dump(mode="json")
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["output"][-1]["content"][0]["text"] == "42"
        assert body["usage"]["input_tokens"] == 7
        assert body["usage"]["output_tokens"] == 3
        assert body["usage"]["total_tokens"] == 10
        assert "_ng_agent_observations" not in body
        assert key in [*mocked.call_args.args, *mocked.call_args.kwargs.values()]
        closed = client.post("/v1/agent_sessions/close", json={"agent_session_id": session_id})
        assert closed.status_code == 200, closed.text
        bundle = AgentObservationBundle.model_validate(closed.json()["agent_observations"])
        assert bundle.source == agent.observation_source
        if runner == "_run_codex":
            invocation = next(record for record in bundle.records if isinstance(record, AgentInvocation))
            assert invocation.conversation[-1].content[0].text == "42"
            assert invocation.model_calls == []
            assert "model_call_ownership_unavailable" in {gap.code for gap in bundle.gaps}
        assert session_id not in agent._agent_sessions
        agent.server_client.post.assert_not_called()  # Native agent does no seed/verify orchestration.


def test_http_cookie_isolation_and_single_activation(cli):
    agent, _, runner = cli
    mocked = mock_runner(agent, runner)
    app = agent.setup_webserver()
    with TestClient(app) as first, TestClient(app) as second:
        a = first.post("/v1/agent_sessions", json=seed("a").model_dump(mode="json")).json()["agent_session_id"]
        b = second.post("/v1/agent_sessions", json=seed("b").model_dump(mode="json")).json()["agent_session_id"]
        assert a != b
        route = f"/ng-rollout/{seed('a').episode_id.capture_key}/v1/responses"
        assert second.post(route, json=params().model_dump(mode="json")).status_code == 409
        assert first.post("/v1/responses", json=params().model_dump(mode="json")).status_code == 409
        mocked.assert_not_called()
        assert first.post(route, json=params().model_dump(mode="json")).status_code == 200
        assert first.post(route, json=params().model_dump(mode="json")).status_code == 409
        mocked.assert_awaited_once()
        assert first.post("/v1/agent_sessions/close", json={"agent_session_id": a}).status_code == 200
        assert b in agent._agent_sessions
        assert second.post("/v1/agent_sessions/close", json={"agent_session_id": b}).status_code == 200


@pytest.mark.parametrize("access", ["sandbox", "mcp"])
def test_unsupported_access_fails_before_activation(cli, access):
    agent, _, runner = cli
    mocked = mock_runner(agent, runner)
    body = seed().model_dump(mode="json")
    if access == "sandbox":
        body["sandbox_access"] = {
            "workdir": "/app",
            "connection": {"kind": "direct", "provider_config_ref": "owner", "descriptor": {"sandbox_id": "owned"}},
        }
    else:
        body["resources_access"] = {"kind": "mcp", "metadata": {"server_name": "owner", "headers": {}}}
    with TestClient(agent.setup_webserver()) as client:
        result = client.post("/v1/agent_sessions", json=body)
        assert result.status_code == 422
        assert "does not support" in result.text or "do not support" in result.text
    assert agent._agent_sessions == {}
    mocked.assert_not_called()


def test_multiworker_legacy_config_rejects_only_native_seed(cli):
    agent, _, runner = cli
    mock_runner(agent, runner)
    agent.config.num_workers = 2
    with TestClient(agent.setup_webserver()) as client:
        native = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert native.status_code == 422 and "num_workers=1" in native.text
        legacy = client.post("/v1/responses", json=params().model_dump(mode="json"))
        assert legacy.status_code == 200
    assert agent._agent_sessions == {}


def test_persistent_workspace_is_legacy_only(cli, tmp_path):
    agent, _, runner = cli
    options = [name for name in ("cwd", "repo_dir") if name in type(agent.config).model_fields]
    if not options:
        pytest.skip("Harness only exposes temporary workspaces")
    setattr(agent.config, options[0], str(tmp_path))
    mock_runner(agent, runner)
    with TestClient(agent.setup_webserver()) as client:
        native = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert native.status_code == 422 and "isolated workspace" in native.text
        assert client.post("/v1/responses", json=params().model_dump(mode="json")).status_code == 200


@pytest.mark.asyncio
async def test_queued_activation_can_close_without_starting_cli(cli):
    agent, _, runner = cli
    mocked = mock_runner(agent, runner)
    request = request_for()
    await agent.seed_agent_session(request, seed())
    await agent.sem.acquire()
    activation = asyncio.create_task(agent.responses(request, params()))
    try:
        await asyncio.sleep(0)
        assert agent._agent_sessions["session"].activation_started
        result = await agent.close_agent_session(request, AgentCloseSessionRequest(agent_session_id="session"))
        assert result.agent_observations.records[0].status == "incomplete"
        with pytest.raises(asyncio.CancelledError):
            await activation
        mocked.assert_not_called()
    finally:
        agent.sem.release()
        if not activation.done():
            activation.cancel()
        await asyncio.gather(activation, return_exceptions=True)


@pytest.mark.asyncio
async def test_caller_cancellation_leaves_cleanup_for_close(cli):
    agent, _, _ = cli
    request = request_for()
    await agent.seed_agent_session(request, seed())
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def block(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    object.__setattr__(agent, "legacy_responses", AsyncMock(side_effect=block))
    activation = asyncio.create_task(agent.responses(request, params()))
    await entered.wait()
    activation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await activation
    await cleaning.wait()
    assert not agent._agent_sessions["session"].state.task.done()
    release.set()
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(agent_session_id="session"))
    assert closed.agent_observations.records[0].status == "incomplete"
    assert agent._agent_sessions == {}


@pytest.mark.asyncio
async def test_close_waits_for_cancelled_activation_and_can_be_retried(cli):
    agent, _, _ = cli
    request = request_for()
    await agent.seed_agent_session(request, seed())
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def block(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    object.__setattr__(agent, "legacy_responses", AsyncMock(side_effect=block))
    activation = asyncio.create_task(agent.responses(request, params()))
    await entered.wait()
    closing = asyncio.create_task(
        agent.close_agent_session(request, AgentCloseSessionRequest(agent_session_id="session"))
    )
    await cleaning.wait()
    assert not closing.done()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert "session" in agent._agent_sessions
    release.set()
    result = await agent.close_agent_session(request, AgentCloseSessionRequest(agent_session_id="session"))
    assert result.agent_observations.records[0].status == "incomplete"
    with pytest.raises(asyncio.CancelledError):
        await activation
    assert agent.sem._value == 1
    assert agent._agent_sessions == {}


@pytest.mark.asyncio
async def test_cancelled_cleanup_failure_does_not_close_session(cli):
    agent, _, _ = cli
    request = request_for()
    await agent.seed_agent_session(request, seed())
    entered = asyncio.Event()

    async def block(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            raise RuntimeError("cleanup failed")

    object.__setattr__(agent, "legacy_responses", AsyncMock(side_effect=block))
    activation = asyncio.create_task(agent.responses(request, params()))
    await entered.wait()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await agent.close_agent_session(request, AgentCloseSessionRequest(agent_session_id="session"))
        assert "session" in agent._agent_sessions
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await activation


@pytest.mark.asyncio
async def test_activation_error_is_preserved_and_close_collects_failure(cli):
    agent, _, _ = cli
    request = request_for()
    await agent.seed_agent_session(request, seed())
    object.__setattr__(agent, "legacy_responses", AsyncMock(side_effect=RuntimeError("model transport failed")))
    with pytest.raises(RuntimeError, match="model transport failed"):
        await agent.responses(request, params())
    result = await agent.close_agent_session(request, AgentCloseSessionRequest(agent_session_id="session"))
    assert result.agent_observations.records[0].status == "failed"
    assert agent._agent_sessions == {}


@pytest.mark.asyncio
async def test_native_path_preserves_token_ids_and_existing_observation_bundle(cli):
    agent, _, _ = cli
    request = request_for()
    await agent.seed_agent_session(request, seed())
    response = NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "id": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": "42", "annotations": []}],
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3, 4],
                "generation_log_probs": [-0.1, -0.2],
            }
        ],
        tool_choice="auto",
        tools=[],
        parallel_tool_calls=True,
    )
    observations = AgentObservationBundle(
        source=agent.observation_source,
        records=[
            AgentInvocation(invocation_id="root", model_calls=[{"model_call_id": "call-1"}]),
            AgentInvocation(invocation_id="child", parent_invocation_id="root"),
        ],
    )
    attached = response.model_copy(update={"_ng_agent_observations": observations.model_dump(mode="json")})
    object.__setattr__(agent, "legacy_responses", AsyncMock(return_value=attached))
    actual = await agent.responses(request, params())
    assert actual.model_dump(mode="json") == response.model_dump(mode="json")
    assert "_ng_agent_observations" in attached.model_extra  # No mutation of the producer result.
    closed = await agent.close_agent_session(request, AgentCloseSessionRequest(agent_session_id="session"))
    assert closed.agent_observations == observations


@pytest.mark.asyncio
async def test_legacy_run_preserves_verifier_fields_cookie_and_rollout_path(cli):
    agent, module, runner = cli
    mock_runner(agent, runner)
    calls = []

    class Response:
        ok = True
        cookies = {"resource-cookie": "owner"}

        def __init__(self, payload):
            self.payload = payload

        async def read(self):
            return json.dumps(self.payload).encode()

    async def post(**kwargs):
        calls.append(kwargs)
        if kwargs["url_path"] == "/seed_session":
            return Response({})
        if kwargs["url_path"] == "/verify":
            return Response(kwargs["json"] | {"reward": 0.0, "mask_sample": True, "benchmark_field": "kept"})
        assert kwargs["url_path"].endswith("/v1/responses")
        request = request_for(rollout="legacy")
        # Self-calls have no native-session flag and must use the legacy path.
        response = await agent.responses(
            request, NeMoGymResponseCreateParamsNonStreaming.model_validate(kwargs["json"])
        )
        return Response(response.model_dump(mode="json"))

    agent.server_client.post = AsyncMock(side_effect=post)
    run_request = getattr(module, f"{type(agent).__name__}RunRequest")(
        responses_create_params=params(), _ng_task_index=0, _ng_rollout_index=0, _ng_rollout_id="legacy"
    )
    result = await agent.run(request_for(), run_request)
    assert result.reward == 0.0 and result.mask_sample is True
    assert result.model_extra["benchmark_field"] == "kept"
    assert result.response.usage.total_tokens == 10
    assert result.turns_used == 1 and result.finished_naturally is True
    assert [call["url_path"] for call in calls][0] == "/seed_session"
    assert calls[-1]["url_path"] == "/verify"
    assert calls[-1]["cookies"] == {"resource-cookie": "owner"}
    assert "_ng_agent_observations" not in calls[-1]["json"]["response"]
    assert agent._agent_sessions == {}


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="CLI process groups require POSIX")
async def test_real_subprocess_cancellation_cleans_up(cli, monkeypatch, tmp_path):
    agent, _, runner = cli
    if runner not in {"_run_opencode", "_run_pi", "_run_codex", "_run_kilo"}:
        pytest.skip("Other harnesses have existing cancellation tests")
    real_exec = asyncio.create_subprocess_exec
    # Codex stages an isolated home under Path.home(); keep this test entirely
    # inside pytest's temporary directory, not the operator's home directory.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    started = asyncio.Event()
    processes = []

    async def execute(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        process = await real_exec(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", execute)
    task = asyncio.create_task(getattr(agent, runner)("test", None))
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert processes[0].returncode is not None
        if hasattr(agent.config, "workspace_root"):
            assert not list(Path(agent.config.workspace_root).glob("*"))
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
