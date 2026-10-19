# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from nemo_gym.base_responses_api_agent import AGENT_SESSION_COOKIE_KEY, AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import ServerClient
from responses_api_agents.claude_code_agent import sandbox as adapter
from responses_api_agents.claude_code_agent.app import ClaudeCodeAgent, ClaudeCodeAgentConfig
from responses_api_agents.claude_code_agent.install_claude_runtime import install


def seed(*, borrowed=True, workdir="/app"):
    return AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id={"rollout_id": "rollout"},
        task_id={"taskset": "fixture", "task_id": "task"},
        sandbox_access={
            "connection": {"provider_config_ref": "sandbox", "descriptor": {"id": "borrowed"}},
            "workdir": workdir,
        }
        if borrowed
        else None,
    )


def agent(**kwargs):
    config = ClaudeCodeAgentConfig(
        name="claude",
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        claude_code_version="2.1.108",
        **kwargs,
    )
    with (
        patch("responses_api_agents.claude_code_agent.app.ensure_claude_code"),
        patch("responses_api_agents.claude_code_agent.app.subprocess.run"),
    ):
        return ClaudeCodeAgent(
            config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={"sandbox": {"local": {}}})
        )


def result(stdout="", *, code=0):
    return SimpleNamespace(stdout=stdout, stderr="diagnostic" if code else "", return_code=code, error_type=None)


@pytest.mark.parametrize("input", ["Question", [{"role": "user", "content": "Question"}]])
def test_text_prompt(input):
    assert adapter.extract_prompt(NeMoGymResponseCreateParamsNonStreaming(input=input, instructions="System")) == (
        "Question",
        "System",
    )


@pytest.mark.parametrize(
    "input",
    [
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        [{"role": "assistant", "content": "answer"}],
        [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.org/a.png"}]}],
        [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}],
    ],
)
def test_reject_input_that_would_be_silently_dropped(input):
    with pytest.raises(ValueError):
        adapter.extract_prompt(NeMoGymResponseCreateParamsNonStreaming(input=input))


def test_system_and_developer_instructions():
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "system", "content": "First"},
            {"role": "developer", "content": "Second"},
            {"role": "user", "content": "Question"},
        ],
        instructions="Third",
        stream=False,
        background=False,
    )
    assert adapter.extract_prompt(body) == ("Question", "First\n\nSecond\n\nThird")


@pytest.mark.parametrize("kwargs", [{"max_output_tokens": 10}, {"temperature": 0.5}, {"model": "other"}])
def test_reject_unsupported_request_controls(kwargs):
    with pytest.raises(ValueError):
        adapter.extract_prompt(NeMoGymResponseCreateParamsNonStreaming(input="Question", **kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [True, False])
async def test_seed_installs_inside_sandbox_and_preserves_ownership(monkeypatch, borrowed):
    remote = SimpleNamespace(
        exec=AsyncMock(side_effect=([result()] if not borrowed else []) + [result(), result("2.1.108 (Claude Code)")]),
        upload=AsyncMock(),
        start=AsyncMock(),
        stop=AsyncMock(),
        disconnect=AsyncMock(),
    )
    constructor = MagicMock(return_value=remote)
    constructor.connect = AsyncMock(return_value=remote)
    monkeypatch.setattr(adapter, "AsyncSandbox", constructor)
    provider = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(adapter, "create_provider", lambda config: provider)
    instance = agent(sandbox_provider="sandbox", sandbox_config={"workdir": "/app"})
    state = await instance._seed_agent_session_state(seed(borrowed=borrowed))
    commands = [call.args[0] for call in remote.exec.await_args_list]
    assert "install.py" in commands[-1] and "2.1.108" in commands[-1]
    assert len(remote.upload.await_args_list) == 2
    assert state.executable.startswith(state.session.session_dir)
    assert state.session.owns_sandbox is not borrowed
    assert remote.start.await_count == (0 if borrowed else 1)
    if borrowed:
        constructor.connect.assert_awaited_once_with({"id": "borrowed"}, provider=provider)
        assert not any(command.startswith("mkdir -p -- /app") for command in commands)


@pytest.mark.asyncio
@pytest.mark.parametrize("workdir", ["/", "/tmp", "relative", "/app/../tmp", "/tmp/nemo-gym-claude-x"])
async def test_bad_workdir_rejected_before_connect(monkeypatch, workdir):
    connector = AsyncMock()
    monkeypatch.setattr(adapter.AsyncSandbox, "connect", connector)
    with pytest.raises(HTTPException, match="separate from session storage"):
        await agent()._seed_agent_session_state(seed(workdir=workdir))
    connector.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_seed_releases_and_failed_release_retains_state(monkeypatch):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result(code=1)), upload=AsyncMock())
    monkeypatch.setattr(adapter.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(adapter, "create_provider", lambda config: object())
    close = AsyncMock()
    monkeypatch.setattr(adapter.SandboxSession, "close", close)
    with pytest.raises(RuntimeError, match="diagnostic"):
        await agent()._seed_agent_session_state(seed())
    close.assert_awaited_once()
    close.side_effect = RuntimeError("release failed")
    with pytest.raises(adapter.AgentSessionSetupError) as error:
        await agent()._seed_agent_session_state(seed())
    assert error.value.state.request == seed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timed_out,code,status,reason",
    [
        (False, 0, "completed", None),
        (False, 7, "failed", "process_exit_7"),
        (True, -15, "incomplete", "timeout"),
    ],
)
async def test_supervised_output_uses_cleanup_outcome_and_shared_builders(
    monkeypatch, timed_out, code, status, reason
):
    files = {}

    async def upload(remote, *, path, text):
        files[path] = text

    monkeypatch.setattr(adapter, "upload_text", upload)
    instance = agent(anthropic_base_url="https://model.example", max_turns=4, thinking="adaptive")
    observed = {}
    raw = "\n".join(
        map(
            json.dumps,
            [
                {"type": "assistant", "message": {"id": "served", "content": [{"type": "text", "text": "partial"}]}},
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
            ],
        )
    )

    class Session:
        session_dir = "/tmp/session"
        sandbox = SimpleNamespace(exec=AsyncMock(return_value=result(code=1)))
        cleanup = {"timed_out": timed_out, "return_code": code, "cleanup_confirmed": True}
        closing = False
        read_output_log = AsyncMock(return_value=raw)

        async def execute(self, **kwargs):
            command = await kwargs["stage_activation"]()
            observed.update(kwargs, command=command)
            return await kwargs["collect"]()

    state = adapter.ClaudeCodeSandboxedState(request=seed(), session=Session(), executable="/runtime/claude")
    body = NeMoGymResponseCreateParamsNonStreaming(input="Question", instructions="System")
    response = await instance._execute(state, body, instance._execution_input(body))
    launch = json.loads(files["/tmp/session/launch.json"])
    assert launch["argv"] == [
        "/runtime/claude",
        *instance._build_command(instance.config.model, "Question", "System")[1:],
    ]
    assert launch["env"]["ANTHROPIC_BASE_URL"] == "https://model.example"
    assert "HOME" not in launch["env"]
    assert launch["env"]["CLAUDE_CONFIG_DIR"] == "/tmp/session/config"
    assert observed["command"].python == "python3"
    assert observed["timeout"] == instance.config.timeout
    assert response.output[0].content[0].text == "partial"
    assert response.status == status
    assert response.metadata.get("stop_reason") == reason
    assert response.usage.total_tokens == 12
    assert state.observations.gaps[0].code == "agent_transcript_unavailable"


@pytest.mark.asyncio
async def test_close_waits_for_cleanup_and_returns_observations():
    instance = agent()
    session = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("unconfirmed")))
    observations = AgentObservationBundle(source="claude_code")
    state = adapter.ClaudeCodeSandboxedState(request=seed(), session=session, observations=observations)
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await instance._close_agent_session_state(state)
    session.close.side_effect = None
    response = await instance._close_agent_session_state(state)
    assert response.agent_observations == observations


def request(session_id="session", rollout="rollout"):
    return Request(
        {"type": "http", "session": {AGENT_SESSION_COOKIE_KEY: session_id}, "path_params": {"rollout_id": rollout}}
    )


@pytest.mark.asyncio
async def test_retry_shares_execution_and_rejects_changed_body(monkeypatch):
    instance = agent()
    state = adapter.ClaudeCodeSandboxedState(request=seed(), session=SimpleNamespace())
    async with instance._locked_agent_session("session") as record:
        record.state = state
    response = MagicMock()
    response.model_copy.return_value = response
    execute = AsyncMock(return_value=response)
    monkeypatch.setattr(ClaudeCodeAgent, "_execute", execute)
    body = NeMoGymResponseCreateParamsNonStreaming(input="Question")
    assert await instance.responses(request(), body) is response
    assert await instance.responses(request(), body) is response
    assert execute.await_count == 1
    with pytest.raises(HTTPException) as error:
        await instance.responses(request(), NeMoGymResponseCreateParamsNonStreaming(input="Changed"))
    assert error.value.status_code == 409
    with pytest.raises(HTTPException):
        await instance.responses(request(rollout="wrong"), body)


@pytest.mark.asyncio
async def test_invalid_session_cannot_fall_back_to_local(monkeypatch):
    instance = agent()
    local = AsyncMock()
    monkeypatch.setattr(ClaudeCodeAgent, "_create_response", local)
    with pytest.raises(HTTPException) as error:
        await instance.responses(request("unknown"), NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert error.value.status_code == 409
    local.assert_not_awaited()


def test_offline_installer_checks_checksum(tmp_path, monkeypatch):
    import hashlib

    source = tmp_path / "source"
    source.write_bytes(b"native executable")
    destination = tmp_path / "claude"
    run = MagicMock()
    monkeypatch.setattr("responses_api_agents.claude_code_agent.install_claude_runtime.subprocess.run", run)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    install("2.1.108", destination, str(source), checksum)
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o700
    run.assert_called_once_with([str(destination), "--version"], check=True, timeout=30)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        install("2.1.108", destination, str(source), "incorrect")
    assert not destination.exists()
