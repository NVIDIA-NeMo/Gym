# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
import subprocess
import tarfile
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.agents import AGENT_SESSION_COOKIE_KEY, AgentSeedRejectedError, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.trajectory import SandboxObservation
from responses_api_agents.claude_code_agent import app
from responses_api_agents.claude_code_agent import sandbox as adapter


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
    return app.ClaudeCodeAgent(
        config=app.ClaudeCodeAgentConfig(
            name="claude", host="127.0.0.1", port=8080, entrypoint="app.py", claude_code_version="2.1.108", **kwargs
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={"sandbox": {"local": {}}}),
    )


def result(stdout="", *, code=0):
    return SimpleNamespace(stdout=stdout, stderr="diagnostic" if code else "", return_code=code, error_type=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [True, False])
async def test_seed_installs_inside_sandbox_and_preserves_ownership(monkeypatch, borrowed):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    constructor = MagicMock(return_value=remote)
    constructor.connect = AsyncMock(return_value=remote)
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    provider = SimpleNamespace(name="local", aclose=AsyncMock())
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    install = AsyncMock()
    monkeypatch.setattr(app, "install_sandbox_runtime", install)
    instance = agent(sandbox_provider="sandbox", sandbox_config={"workdir": "/app"})
    state = await instance._seed_agent_session_state(seed(borrowed=borrowed))
    commands = [call.args[0] for call in remote.exec.await_args_list]
    assert install.await_args.args == (remote,)
    assert install.await_args.kwargs["version"] == "2.1.108"
    assert install.await_args.kwargs["runtime"] == state.runtime
    assert state.runtime == "/tmp/nemo-gym-claude-runtime-2.1.108"
    assert state.runtime not in state.session.session_dir
    assert len(remote.upload.await_args_list) == 2
    assert state.session.owns_sandbox is not borrowed
    assert remote.start.await_count == (0 if borrowed else 1)
    if borrowed:
        constructor.connect.assert_awaited_once_with({"id": "borrowed"}, provider=provider)
        assert not any(command.startswith("mkdir -p -- /app") for command in commands)


@pytest.mark.asyncio
@pytest.mark.parametrize("network_access", ["inherit", "model_only"])
async def test_owned_sandbox_egress_allows_only_the_model_endpoint(monkeypatch, network_access):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    monkeypatch.setattr(app, "AsyncSandbox", MagicMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: SimpleNamespace(name="opensandbox", aclose=AsyncMock()))
    monkeypatch.setattr(app, "install_sandbox_runtime", AsyncMock())
    instance = agent(
        sandbox_provider="sandbox",
        sandbox_config={"workdir": "/app"},
        anthropic_base_url="http://model.example:9000",
        network_access=network_access,
    )
    await instance._seed_agent_session_state(seed(borrowed=False))
    policy = remote.start.await_args.args[0].provider_options.get("network_policy")
    if network_access == "inherit":
        assert policy is None
    else:
        assert policy == {"defaultAction": "deny", "egress": [{"action": "allow", "target": "model.example"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "allowed"),
    [
        # Neither a Gym model server nor a direct URL: the CLI calls Anthropic's own API.
        ({"anthropic_api_key": "sk-test"}, "api.anthropic.com"),  # pragma: allowlist secret
        ({"anthropic_base_url": "https://direct.example/anthropic"}, "direct.example"),
        # A Gym model server wins over a direct URL, as it does for the CLI's own calls.
        (
            {
                "model_server": {"type": "responses_api_models", "name": "policy_model"},
                "anthropic_base_url": "https://direct.example/anthropic",
            },
            "gym-model.example",
        ),
    ],
)
async def test_model_only_allows_the_endpoint_the_cli_calls(monkeypatch, settings, allowed):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    monkeypatch.setattr(app, "AsyncSandbox", MagicMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: SimpleNamespace(name="opensandbox", aclose=AsyncMock()))
    monkeypatch.setattr(app, "install_sandbox_runtime", AsyncMock())
    instance = agent(
        sandbox_provider="sandbox", sandbox_config={"workdir": "/app"}, network_access="model_only", **settings
    )
    instance.server_client.global_config_dict["policy_model"] = {"responses_api_models": {"vllm_model": {}}}
    instance.server_client._build_server_base_url = MagicMock(return_value="http://gym-model.example:8000")
    await instance._seed_agent_session_state(seed(borrowed=False))
    policy = remote.start.await_args.args[0].provider_options["network_policy"]
    assert policy == {"defaultAction": "deny", "egress": [{"action": "allow", "target": allowed}]}
    if "anthropic_api_key" in settings:
        # The fallback applies only to the allowlist: the CLI still gets no ANTHROPIC_BASE_URL.
        assert instance._resolve_call_base_url("rollout") == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "allowed"),
    [
        ("https://Direct.Example/anthropic", "direct.example"),
        ("https://user:secret@direct.example/anthropic", "direct.example"),  # pragma: allowlist secret
        ("http://[2001:db8::1]:9000/anthropic", "2001:db8::1"),
    ],
    ids=["uppercase", "userinfo", "ipv6_literal"],
)
async def test_model_only_allows_the_host_however_the_url_writes_it(monkeypatch, base_url, allowed):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    monkeypatch.setattr(app, "AsyncSandbox", MagicMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: SimpleNamespace(name="opensandbox", aclose=AsyncMock()))
    monkeypatch.setattr(app, "install_sandbox_runtime", AsyncMock())
    instance = agent(
        sandbox_provider="sandbox",
        sandbox_config={"workdir": "/app"},
        network_access="model_only",
        anthropic_base_url=base_url,
    )
    await instance._seed_agent_session_state(seed(borrowed=False))
    policy = remote.start.await_args.args[0].provider_options["network_policy"]
    assert policy == {"defaultAction": "deny", "egress": [{"action": "allow", "target": allowed}]}


@pytest.mark.asyncio
async def test_inherit_keeps_a_loopback_model_endpoint_working(monkeypatch):
    # A local Gym model server must keep working by default: inherit adds no policy and refuses nothing.
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    monkeypatch.setattr(app, "AsyncSandbox", MagicMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: SimpleNamespace(name="local", aclose=AsyncMock()))
    monkeypatch.setattr(app, "install_sandbox_runtime", AsyncMock())
    instance = agent(
        sandbox_provider="sandbox", sandbox_config={"workdir": "/app"}, anthropic_base_url="http://127.0.0.1:8000"
    )
    await instance._seed_agent_session_state(seed(borrowed=False))
    assert "network_policy" not in (remote.start.await_args.args[0].provider_options or {})


def test_an_unknown_network_access_value_is_refused_at_load():
    with pytest.raises(ValidationError):
        app.ClaudeCodeAgentConfig(
            name="claude", host="127.0.0.1", port=8080, entrypoint="app.py", network_access="open"
        )


@pytest.mark.parametrize(
    ("provider_name", "provider_options", "reason"),
    [
        ("opensandbox", {"network_policy": {}}, "not both"),
        ("docker", {}, "requires the OpenSandbox network-policy provider"),
    ],
)
def test_unenforceable_network_access_is_a_terminal_seed_refusal(monkeypatch, provider_name, provider_options, reason):
    constructor = MagicMock()
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    provider = SimpleNamespace(name=provider_name, aclose=AsyncMock())
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    instance = agent(
        sandbox_provider="sandbox",
        sandbox_config={"workdir": "/app", "provider_options": provider_options},
        anthropic_base_url="http://model.example:9000",
        network_access="model_only",
    )
    body = seed(borrowed=False)
    with TestClient(instance.setup_webserver()) as client:
        refused = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
    # A configuration another attempt cannot fix: HTTP 422 fails the episode once instead of a retried 500.
    assert refused.status_code == 422, refused.text
    assert reason in refused.json()["detail"]
    constructor.assert_not_called()
    provider.aclose.assert_awaited_once()
    assert body.agent_session_id not in instance._session_records


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("borrowed", "network_access", "message"),
    [(True, "model_only", "not a borrowed one"), (False, "model_and_tools", "no tool grants")],
)
async def test_unusable_network_access_is_rejected_before_connecting(monkeypatch, borrowed, network_access, message):
    constructor = MagicMock()
    constructor.connect = AsyncMock()
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    create_provider = MagicMock()
    monkeypatch.setattr(app, "create_provider", create_provider)
    instance = agent(sandbox_provider="sandbox", sandbox_config={"workdir": "/app"}, network_access=network_access)
    with pytest.raises(AgentSeedRejectedError, match=message):
        await instance._seed_agent_session_state(seed(borrowed=borrowed))
    create_provider.assert_not_called()
    constructor.assert_not_called()
    constructor.connect.assert_not_awaited()


def test_setup_without_bash_names_it(tmp_path):
    # A non-root image without bash must fail at setup with a message, not with a bare 127 later.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "id").write_text("#!/bin/sh\necho 1000\n")
    (bindir / "id").chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", "-c", app._BASH_BOOTSTRAP], env={"PATH": str(bindir)}, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 1
    assert "require bash" in result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("workdir", ["/", "/tmp", "relative", "/app/../tmp", "/tmp/nemo-gym-claude-runtime-2.1.108"])
async def test_bad_workdir_rejected_before_connect(monkeypatch, workdir):
    connector = AsyncMock()
    monkeypatch.setattr(app.AsyncSandbox, "connect", connector)
    with pytest.raises(HTTPException, match="separate from runtime/session storage"):
        await agent()._seed_agent_session_state(seed(workdir=workdir))
    connector.assert_not_awaited()


@pytest.mark.asyncio
async def test_seed_passes_trusted_binary_checksum_to_installer(monkeypatch):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock())
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: object())
    checksum = "a" * 64
    state = await agent(
        sandbox_executable_path="/opt/claude", sandbox_executable_sha256=checksum
    )._seed_agent_session_state(seed())
    command = remote.exec.await_args.args[0]
    assert "--source /opt/claude" in command
    assert "--sha256 " + checksum in command
    assert "--runtime " + state.runtime in command


@pytest.mark.asyncio
async def test_preinstalled_binary_requires_checksum():
    with pytest.raises(HTTPException, match="sandbox_executable_sha256"):
        await agent(sandbox_executable_path="/usr/bin/claude")._seed_agent_session_state(seed())


@pytest.mark.asyncio
async def test_failed_seed_releases_and_failed_release_retains_state(monkeypatch):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result(code=1)), upload=AsyncMock())
    monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(return_value=remote))
    monkeypatch.setattr(app, "create_provider", lambda config: object())
    close = AsyncMock()
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "close", close)
    with pytest.raises(RuntimeError, match="diagnostic"):
        await agent()._seed_agent_session_state(seed())
    close.assert_awaited_once()
    close.side_effect = RuntimeError("release failed")
    with pytest.raises(app.AgentSessionSetupError) as error:
        await agent()._seed_agent_session_state(seed())
    assert error.value.state.request == seed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timed_out,code,status", [(False, 0, "completed"), (False, 7, "failed"), (True, -15, "incomplete")]
)
async def test_native_events_outcome_and_shared_command_options(monkeypatch, timed_out, code, status):
    instance = agent(anthropic_base_url="https://model.example", max_turns=4, thinking="adaptive")
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
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(
            session_dir="/tmp/session",
            workdir="/app",
            cleanup={"timed_out": timed_out, "return_code": code, "cleanup_confirmed": True},
            artifacts=None,
        ),
    )
    execute = AsyncMock(return_value=adapter.ClaudeCodeArtifacts(raw))
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "execute", execute)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": "System"}, {"role": "user", "content": "Question"}]
    )
    response = await instance._sandbox_response(state, body, prompt="Question", system="System")
    payload = execute.await_args.args[0]
    assert payload["command"] == [
        "/runtime/claude",
        *instance._build_command(instance.config.model, "Question", "System")[1:],
    ]
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "https://model.example"
    assert "HOME" not in payload["env"]
    # A backgrounded command would keep the CLI, and so the run, alive after the final answer.
    assert payload["env"]["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert response.output[0].content[0].text == "partial"
    assert response.status == status
    assert response.usage.total_tokens == 12
    assert response.metadata is None  # no adapter-specific runtime policy fields
    assert "model_call_ownership_unavailable" in [gap.code for gap in state.observations.gaps]


@pytest.mark.asyncio
async def test_artifact_capture_reads_separate_stdout_and_process_info(monkeypatch):
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(
            session_dir="/tmp/session",
            sandbox=SimpleNamespace(exec=AsyncMock(return_value=result(code=1))),
            read_output_log=AsyncMock(return_value='{"type":"result","subtype":"error_fake"}'),
        ),
    )
    read = AsyncMock(
        side_effect=lambda name: {
            "events.jsonl": '{"type":"result","subtype":"success"}\n',
            "stderr.log": "x" * 20000,
            "runtime.json": '{"hostname":"worker","pid":123}',
        }[name]
    )
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "read_text", read)
    artifacts = await state.collect_artifacts()
    assert artifacts.events == '{"type":"result","subtype":"success"}\n'
    assert state.runtime_info.pid == 123
    assert len(state.stderr) == 16000
    state.session.read_output_log.assert_not_awaited()


@pytest.mark.asyncio
async def test_artifact_capture_copies_the_transcript_before_the_session_directory_goes(monkeypatch):
    exec_ = AsyncMock(return_value=result(code=0))
    written = {}

    async def upload_text(sandbox, *, path, text):
        written[path] = text

    monkeypatch.setattr(adapter, "upload_text", upload_text)
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(session_dir="/tmp/session", sandbox=SimpleNamespace(exec=exec_)),
        transcript_path="/logs/agent/claude-code.txt",
    )
    read = AsyncMock(side_effect=lambda name: {"events.jsonl": "{}\n", "stderr.log": "", "runtime.json": "{}"}[name])
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "read_text", read)

    await state.collect_artifacts()

    assert exec_.await_args_list[0].args[0] == "mkdir -p /logs/agent && rm -rf -- /logs/agent/claude-code.txt"
    assert written == {"/logs/agent/claude-code.txt": "{}\n"}


@pytest.mark.asyncio
async def test_close_returns_gap_without_activation_and_honors_cleanup_error():
    instance = agent()
    session = SimpleNamespace(
        close=AsyncMock(side_effect=RuntimeError("unconfirmed")), artifacts=None, cleanup_confirmed=True
    )
    state = adapter.ClaudeCodeSandboxSession(request=seed(), runtime="/runtime", session=session)
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await instance._close_agent_session_state(state)
    session.close.side_effect = None
    response = await instance._close_agent_session_state(state)
    assert response.agent_observations.gaps[0].code == "agent_activation_interrupted"


@pytest.mark.asyncio
async def test_close_bounds_a_stubborn_activation():
    started = asyncio.Event()
    release = asyncio.Event()

    async def activation():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()

    task = asyncio.create_task(activation())
    await started.wait()
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(), runtime="/runtime", session=SimpleNamespace(close=AsyncMock(), artifacts=None), task=task
    )
    try:
        with pytest.raises(TimeoutError):
            await state.close(0.01)
        assert task.cancelling()
    finally:
        release.set()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("parser_name", ["parse_stream_json", "parse_artifact_observations"])
async def test_close_joins_captured_observations_during_parsing(monkeypatch, parser_name):
    entered = threading.Event()
    release = threading.Event()
    original = getattr(app, parser_name)

    def paused_parser(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(app, parser_name, paused_parser)
    native = json.dumps(
        {
            "type": "assistant",
            "sessionId": "captured",
            "uuid": "answer",
            "timestamp": "2026-10-20T10:00:00Z",
            "message": {"role": "assistant", "id": "served", "content": [{"type": "text", "text": "Answer"}]},
        }
    ).encode()
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        info = tarfile.TarInfo("projects/work/captured.jsonl")
        info.size = len(native)
        archive.addfile(info, io.BytesIO(native))
    artifacts = adapter.ClaudeCodeArtifacts(
        '{"type":"result","subtype":"success","is_error":false}', archive_bytes.getvalue()
    )
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(
            session_dir="/tmp/session",
            workdir="/app",
            artifacts=artifacts,
            cleanup={"timed_out": False, "return_code": 0, "cleanup_confirmed": True},
            close=AsyncMock(),
            cleanup_confirmed=True,
        ),
    )
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "execute", AsyncMock(return_value=artifacts))
    instance = agent()
    body = NeMoGymResponseCreateParamsNonStreaming(input="Question")
    state.task = asyncio.create_task(instance._sandbox_response(state, body, prompt="Question", system=None))
    close = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        close = asyncio.create_task(instance._close_agent_session_state(state))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(close), timeout=0.02)
        assert not state.task.cancelling()
        release.set()
        receipt = await asyncio.wait_for(close, timeout=5)
        assert any(record.kind == "agent_invocation" for record in receipt.agent_observations.records)
        assert any(record.kind == "sandbox" for record in receipt.agent_observations.records)
        assert "agent_activation_interrupted" not in [gap.code for gap in receipt.agent_observations.gaps]
    finally:
        release.set()
        await asyncio.gather(state.task, *([close] if close else []), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata,expected",
    [
        (None, 300),
        ({}, 300),
        ({"agent_timeout_sec": "12.5"}, 12.5),
        ({"agent_timeout_sec": "900"}, 300),
        ({"agent_timeout_sec": "0"}, 300),
        ({"agent_timeout_sec": "NaN"}, 300),
        ({"agent_timeout_sec": "bad"}, 300),
    ],
)
async def test_sandbox_honors_task_timeout_and_server_ceiling(monkeypatch, metadata, expected):
    instance = agent(timeout=300)
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(session_dir="/tmp/session", workdir="/app", cleanup=None),
    )
    execute = AsyncMock(return_value=adapter.ClaudeCodeArtifacts(""))
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "execute", execute)
    body = NeMoGymResponseCreateParamsNonStreaming(input="Question", metadata=metadata)
    instance._validate_sandbox_request(body)
    await instance._sandbox_response(state, body, prompt="Question", system=None)
    assert execute.await_args.kwargs["timeout"] == expected


@pytest.mark.parametrize("metadata", [{"other": "value"}, {"agent_timeout_sec": "10", "other": "value"}])
def test_sandbox_rejects_unsupported_metadata(metadata):
    with pytest.raises(HTTPException) as error:
        agent()._validate_sandbox_request(NeMoGymResponseCreateParamsNonStreaming(input="Question", metadata=metadata))
    assert error.value.status_code == 422


def request(session_id="session", rollout="rollout"):
    return Request(
        {"type": "http", "session": {AGENT_SESSION_COOKIE_KEY: session_id}, "path_params": {"rollout_id": rollout}}
    )


@pytest.mark.asyncio
async def test_retry_shares_execution_and_rejects_changed_body(monkeypatch):
    instance = agent()
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(), runtime="/runtime", session=SimpleNamespace(closing=False)
    )
    async with instance._locked_agent_session("session") as record:
        record.state = state
    response = MagicMock()
    response.model_copy.return_value = response
    execute = AsyncMock(return_value=response)
    monkeypatch.setattr(app.ClaudeCodeAgent, "_sandbox_response", execute)
    body = NeMoGymResponseCreateParamsNonStreaming(input="Question")
    assert await instance.responses(request(), body) is response
    assert await instance.responses(request(), body) is response
    assert execute.await_count == 1
    with pytest.raises(HTTPException) as error:
        await instance.responses(request(), NeMoGymResponseCreateParamsNonStreaming(input="Changed"))
    assert error.value.status_code == 409
    with pytest.raises(HTTPException) as error:
        await instance.responses(request(rollout="wrong"), body)
    assert error.value.status_code == 409
    assert execute.await_count == 1


@pytest.mark.asyncio
async def test_invalid_session_cannot_fall_back_to_local(monkeypatch):
    instance = agent()
    local = AsyncMock()
    monkeypatch.setattr(app.ClaudeCodeAgent, "_create_response", local)
    with pytest.raises(HTTPException) as error:
        await instance.responses(request("unknown"), NeMoGymResponseCreateParamsNonStreaming(input="Question"))
    assert error.value.status_code == 409
    local.assert_not_awaited()


@pytest.mark.parametrize(
    "input,expected",
    [
        ("Question", ("Question", "Configured")),
        ([{"role": "user", "content": "old"}, {"role": "user", "content": "new"}], ("new", "Configured")),
        (
            [{"role": "developer", "content": "Ignored"}, {"role": "user", "content": "Question"}],
            ("Question", "Configured"),
        ),
        (
            [{"role": "system", "content": "System"}, {"role": "user", "content": "Question"}],
            ("Question", "Configured\n\nSystem"),
        ),
    ],
)
def test_local_prompt_conversion_preserves_existing_contract(input, expected):
    instance = agent(system_prompt="Configured")
    body = NeMoGymResponseCreateParamsNonStreaming(input=input, instructions="Legacy ignores request instructions")
    assert instance._conversation_input(body) == expected


@pytest.mark.asyncio
async def test_empty_sandbox_output_uses_local_padding(monkeypatch):
    instance = agent()
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(
            session_dir="/tmp/session", workdir="/app", artifacts=None, cleanup={"timed_out": False, "return_code": 0}
        ),
    )
    monkeypatch.setattr(
        adapter.ClaudeCodeSandboxSession, "execute", AsyncMock(return_value=adapter.ClaudeCodeArtifacts(""))
    )
    body = NeMoGymResponseCreateParamsNonStreaming(input="Question")
    response = await instance._sandbox_response(state, body, prompt="Question", system=None)
    assert response.output[0].content[0].text == ""
    assert response.output[0].role == "assistant"
    assert response.status == "incomplete"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_server", [None, {"type": "responses_api_models", "name": "policy"}])
async def test_sandbox_only_receives_direct_credentials_without_gym_model_server(monkeypatch, model_server):
    credential = "direct-provider-credential"
    instance = agent(model_server=model_server, anthropic_api_key=credential, anthropic_base_url="http://direct")
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(session_dir="/tmp/session", workdir="/app", cleanup=None),
    )
    execute = AsyncMock(return_value=adapter.ClaudeCodeArtifacts(""))
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "execute", execute)
    monkeypatch.setattr(app.ClaudeCodeAgent, "_resolve_call_base_url", lambda *_: "http://model-server")
    await instance._sandbox_response(
        state, NeMoGymResponseCreateParamsNonStreaming(input="Question"), prompt="Question", system=None
    )
    payload = execute.await_args.args[0]
    expected = "gym" if model_server else credential
    assert payload["env"]["ANTHROPIC_API_KEY"] == expected
    assert payload["env"]["ANTHROPIC_AUTH_TOKEN"] == expected
    if model_server:
        assert credential not in json.dumps(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("timed_out,expected", [(False, "cancelled"), (True, "timeout")])
async def test_cancelled_activation_reports_sandbox_outcome(monkeypatch, timed_out, expected):
    instance = agent()
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(
            session_dir="/tmp/session",
            workdir="/app",
            artifacts=adapter.ClaudeCodeArtifacts(""),
            cleanup={"timed_out": timed_out, "return_code": 143},
        ),
    )
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "execute", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await instance._sandbox_response(
            state, NeMoGymResponseCreateParamsNonStreaming(input="Question"), prompt="Question", system=None
        )
    record = next(record for record in state.observations.records if record.kind == "sandbox")
    assert record.outcome == expected
    assert record.exit_code == 143


@pytest.mark.asyncio
async def test_skills_are_staged_before_launch_and_disable_bare(monkeypatch, tmp_path):
    skill = tmp_path / "skills" / "example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: example\n---\nUse the helper.\n")
    (skill / "helper.sh").write_text("echo helper\n")
    files = {}

    async def upload(source, target):
        files[target] = source.read_bytes()

    remote = SimpleNamespace(upload=upload)
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(), runtime="/runtime", session=SimpleNamespace(session_dir="/tmp/session", sandbox=remote)
    )
    payload = {
        "command": ["/runtime/claude"],
        "cwd": "/app",
        "env": {},
        "settings": {},
        "mcp_config": None,
        "skills_path": str(tmp_path / "skills"),
    }
    staged = AsyncMock()
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "upload_json", staged)
    command = await state.stage_activation(payload)
    assert files["/tmp/session/config/skills/example/helper.sh"] == b"echo helper\n"
    assert "SKILL.md" in " ".join(files)
    assert command.argv == ["bash", "/tmp/session/harness_runner.sh", "/tmp/session/params.sh"]
    assert "/runtime/claude" in files["/tmp/session/params.sh"].decode()
    runner_payload = staged.await_args.args[1]
    assert "skills_path" not in runner_payload
    instance = agent()
    instance.server_client.global_config_dict["skills"] = {"path": str(tmp_path / "skills")}
    assert instance._configured_skills_path() == str(tmp_path / "skills")
    assert "--bare" not in instance._build_command("model", "prompt", skills_active=True)


@pytest.mark.asyncio
async def test_host_install_is_lazy_shared_and_retryable(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def install(version):
        entered.set()
        release.wait(timeout=5)

    installer = MagicMock(side_effect=install)
    monkeypatch.setattr(app, "ensure_claude_code", installer)
    instance = agent()
    installer.assert_not_called()
    waiters = [asyncio.create_task(instance._ensure_local_runtime()) for _ in range(2)]
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        waiters[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiters[0]
        assert not instance._local_setup_task.cancelled()
    finally:
        release.set()
        await asyncio.gather(*waiters, return_exceptions=True)
    assert installer.call_count == 1
    instance = agent()
    installer.side_effect = RuntimeError("install failed")
    with pytest.raises(RuntimeError, match="install failed"):
        await instance._ensure_local_runtime()
    installer.side_effect = None
    await instance._ensure_local_runtime()
    assert installer.call_count == 3


@pytest.mark.parametrize(
    "input",
    [
        [],
        [{"role": "assistant", "content": "Earlier answer"}],
        [{"role": "user", "content": "Earlier"}, {"role": "user", "content": "Latest"}],
        [{"role": "developer", "content": "Instructions"}, {"role": "user", "content": "Question"}],
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up"},
        ],
    ],
)
def test_sandbox_rejects_history_instead_of_dropping_it(input):
    with pytest.raises(HTTPException) as error:
        agent()._validate_sandbox_request(NeMoGymResponseCreateParamsNonStreaming(input=input))
    assert error.value.status_code == 422


def test_sandbox_rejects_request_instructions_that_local_mode_ignores():
    with pytest.raises(HTTPException, match="instructions"):
        agent()._validate_sandbox_request(
            NeMoGymResponseCreateParamsNonStreaming(input="Question", instructions="Use me")
        )


@pytest.mark.parametrize(
    "input",
    [
        "Question",
        [{"role": "user", "content": "Question"}],
        [{"role": "system", "content": "Instructions"}, {"role": "user", "content": "Question"}],
    ],
)
def test_sandbox_accepts_one_text_turn_with_optional_system_message(input):
    agent()._validate_sandbox_request(NeMoGymResponseCreateParamsNonStreaming(input=input))


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [True, False])
async def test_close_distinguishes_borrowed_release_from_confirmed_termination(confirmed):
    from nemo_gym.agents import SandboxSession, harness_termination_failure

    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), disconnect=AsyncMock())
    session = SandboxSession(sandbox=remote, session_dir="/tmp/session", workdir="/app", harness="Claude Code")
    session.launch_started = True
    receipt = {
        "cleanup_confirmed": confirmed,
        "timed_out": False,
        "return_code": 0,
        "error": None if confirmed else "process survived",
    }
    if confirmed:
        session.cleanup = receipt
    else:
        session.termination_error = harness_termination_failure(harness="Claude Code", receipt=receipt)
    state = adapter.ClaudeCodeSandboxSession(request=seed(), runtime="/runtime", session=session)
    close = await agent()._close_agent_session_state(state)
    remote.disconnect.assert_awaited_once()
    assert close.cleanup_confirmed is confirmed
    assert close.model_dump(mode="json")["cleanup_confirmed"] is confirmed


def _events(*events: dict) -> str:
    return "\n".join(map(json.dumps, events))


def refused(session_id: str = "sess-1", message_id: str = "msg-refused") -> str:
    """Claude Code's stream when the provider's safety filter ends a response: its error frame keeps stop_reason."""
    return _events(
        {"type": "system", "subtype": "init", "session_id": session_id},
        {
            "type": "assistant",
            "message": {
                "id": message_id,
                "stop_reason": "refusal",
                "content": [{"type": "text", "text": "API Error: the model's safeguards flagged this message"}],
            },
        },
        {"type": "result", "subtype": "success", "is_error": True, "usage": {"input_tokens": 5, "output_tokens": 1}},
    )


def finished(session_id: str = "sess-1") -> str:
    return _events(
        {"type": "system", "subtype": "init", "session_id": session_id},
        {
            "type": "assistant",
            "message": {"id": "msg-done", "stop_reason": "end_turn", "content": [{"type": "text", "text": "done"}]},
        },
        {"type": "result", "subtype": "success", "is_error": False, "usage": {"input_tokens": 7, "output_tokens": 2}},
    )


def test_a_resumed_activation_that_timed_out_still_counts_its_tokens():
    """F22: the first activation's result covers its own messages only; a resume killed before its result adds its
    assistant messages' usage."""
    resumed_without_result = _events(
        {"type": "system", "subtype": "init", "session_id": "sess-1"},
        {
            "type": "assistant",
            "message": {
                "id": "msg-long",
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1000, "output_tokens": 300},
                "content": [{"type": "text", "text": "working"}],
            },
        },
    )

    _, metadata = app.parse_stream_json(refused() + "\n" + resumed_without_result)

    assert (metadata["input_tokens"], metadata["output_tokens"]) == (5 + 1000, 1 + 300)


def test_the_parser_counts_provider_refusals_and_whether_one_ended_the_session():
    _, metadata = app.parse_stream_json(refused())
    assert metadata["safety_refusals"] == 1
    assert metadata["ended_on_safety_refusal"] is True
    assert metadata["session_id"] == "sess-1"
    _, metadata = app.parse_stream_json(refused() + "\n" + finished())
    assert metadata["safety_refusals"] == 1 and metadata["ended_on_safety_refusal"] is False
    _, metadata = app.parse_stream_json(finished())
    assert "safety_refusals" not in metadata and "ended_on_safety_refusal" not in metadata


def resumable_state() -> adapter.ClaudeCodeSandboxSession:
    return adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(
            troxy_failed=False,
            troxy_base_url=None,
            token_capture=lambda: None,
            session_dir="/tmp/session",
            workdir="/app",
            cleanup={"timed_out": False, "return_code": 0, "cleanup_confirmed": True},
            artifacts=None,
            cleanup_confirmed=True,
            closing=False,
        ),
    )


async def _counting_resume(self, *, timeout):
    self.resumes += 1


@pytest.mark.asyncio
@pytest.mark.parametrize("resumes", [0, 1])
async def test_a_refusal_resumes_the_conversation_only_when_configured(monkeypatch, resumes):
    instance = agent(anthropic_base_url="https://model.example", refusal_resumes=resumes)
    state = resumable_state()
    execute = AsyncMock(side_effect=[adapter.ClaudeCodeArtifacts(refused()), adapter.ClaudeCodeArtifacts(finished())])
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "execute", execute)
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "begin_resume", _counting_resume)
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Question"}])

    response = await instance._sandbox_response(state, body, prompt="Question", system=None)

    if not resumes:
        # Off by default: Claude Code's exit stands, and the response still says it ended on a refusal (F23).
        assert execute.await_count == 1
        assert response.metadata == {
            "safety_refusals": "1",
            "safety_refusal_resumes": "0",
            "ended_on_safety_refusal": "true",
        }
        return
    assert execute.await_count == 2
    first, second = (call.args[0]["command"] for call in execute.await_args_list)
    assert "--resume" not in first
    assert second[second.index("--resume") + 1] == "sess-1"
    assert second[-1] == instance.config.refusal_resume_prompt
    assert execute.await_args_list[1].kwargs["timeout"] <= execute.await_args_list[0].kwargs["timeout"]
    # The response covers both activations and says the conversation went on.
    assert response.output[-1].content[0].text == "done"
    assert response.metadata == {
        "safety_refusals": "1",
        "safety_refusal_resumes": "1",
        "ended_on_safety_refusal": "false",
    }
    assert state.claude_config_dir() == "/tmp/session/config"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_activation_s, timed_out, resumed_timeout",
    [(200.0, False, 400.0), (560.0, False, None), (100.0, True, None)],
    ids=["resume-gets-the-time-left", "too-little-time-left", "activation-timed-out"],
)
async def test_a_refusal_resume_runs_only_within_the_activations_time_left(
    monkeypatch, first_activation_s, timed_out, resumed_timeout
):
    """The resume shares the activation's one deadline: it gets exactly the time left, and none when under a minute
    is left or the refused activation itself timed out."""
    instance = agent(anthropic_base_url="https://model.example", refusal_resumes=1, timeout=600)
    state = resumable_state()
    state.session.cleanup["timed_out"] = timed_out
    clock = [1000.0]
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock[0])
    outcomes = iter([refused(), finished()])

    async def execute(self, payload, *, timeout, close_timeout):
        clock[0] += first_activation_s
        return adapter.ClaudeCodeArtifacts(next(outcomes))

    timeouts = []

    async def recording(self, payload, *, timeout, close_timeout):
        timeouts.append(timeout)
        return await execute(self, payload, timeout=timeout, close_timeout=close_timeout)

    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "execute", recording)
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "begin_resume", _counting_resume)
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Question"}])

    response = await instance._sandbox_response(state, body, prompt="Question", system=None)

    if resumed_timeout is None:
        assert timeouts == [600]
        assert response.metadata["ended_on_safety_refusal"] == "true"
    else:
        assert timeouts == [600, resumed_timeout]
        assert response.metadata["ended_on_safety_refusal"] == "false"


@pytest.mark.asyncio
async def test_spent_resumes_leave_the_session_ended_on_a_refusal(monkeypatch):
    instance = agent(anthropic_base_url="https://model.example", refusal_resumes=2)
    state = resumable_state()
    execute = AsyncMock(
        side_effect=[adapter.ClaudeCodeArtifacts(refused(message_id=f"msg-{index}")) for index in range(3)]
    )
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "execute", execute)
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "begin_resume", _counting_resume)
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Question"}])

    response = await instance._sandbox_response(state, body, prompt="Question", system=None)

    assert execute.await_count == 3
    assert response.metadata == {
        "safety_refusals": "3",
        "safety_refusal_resumes": "2",
        "ended_on_safety_refusal": "true",
    }


@pytest.mark.asyncio
async def test_a_resume_rearms_the_harness_and_keeps_the_conversation_directory():
    session = SimpleNamespace(
        session_dir="/tmp/session",
        begin_activation=AsyncMock(),
        sandbox=SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock()),
    )
    session.begin_activation.side_effect = lambda *, session_dir, timeout: setattr(session, "session_dir", session_dir)
    state = adapter.ClaudeCodeSandboxSession(request=seed(), runtime="/runtime", session=session)
    assert state.claude_config_dir() == "/tmp/session/config"

    await state.begin_resume(timeout=5)

    assert session.session_dir == "/tmp/session/resume-1" and state.resumes == 1
    assert state.claude_config_dir() == "/tmp/session/config"
    assert session.sandbox.upload.await_args.args[1] == "/tmp/session/resume-1/harness_runner.sh"


@pytest.mark.asyncio
async def test_each_activation_rewrites_the_verifier_transcript_from_the_events_gym_holds(monkeypatch):
    """F24: the agent can rewrite the transcript during a resumed activation; Gym's copy replaces it each time."""
    written = []

    async def upload_text(sandbox, *, path, text):
        written.append(text)

    monkeypatch.setattr(adapter, "upload_text", upload_text)
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()))
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(session_dir="/tmp/session", sandbox=remote),
    )

    await state.copy_transcript("/logs/agent/claude-code.txt", '{"first":1}')
    state.earlier_events = '{"first":1}\n'
    await state.copy_transcript("/logs/agent/claude-code.txt", '{"resumed":2}\n')

    assert written == ['{"first":1}\n', '{"first":1}\n{"resumed":2}\n']
    assert all("rm -rf -- /logs/agent/claude-code.txt" in call.args[0] for call in remote.exec.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["rm", "upload"])
async def test_a_transcript_gym_could_not_replace_is_reported_unavailable(monkeypatch, failing):
    """S1: when the rm or the upload fails, the file at the path may be the agent's, so the response says so."""
    instance = agent(anthropic_base_url="https://model.example")

    async def upload_text(sandbox, *, path, text):
        if failing == "upload":
            raise RuntimeError("upload failed")

    monkeypatch.setattr(adapter, "upload_text", upload_text)
    exec_result = result(code=1) if failing == "rm" else result()
    state = resumable_state()
    state.session.sandbox = SimpleNamespace(exec=AsyncMock(return_value=exec_result))

    async def execute(self, payload, *, timeout, close_timeout):
        await self.copy_transcript("/logs/agent/claude-code.txt", finished())
        return adapter.ClaudeCodeArtifacts(finished())

    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "execute", execute)
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Question"}])

    response = await instance._sandbox_response(state, body, prompt="Question", system=None)

    assert response.metadata == {"transcript_unavailable": "true"}


STUB_CLAUDE = """#!/bin/bash
# Stand-in for the claude CLI: records its argv and the config it saw, then replays one prepared activation.
echo "$*" >> "$CLAUDE_CONFIG_DIR/argv.log"
n=$(ls "$CLAUDE_CONFIG_DIR"/activation.* 2>/dev/null | wc -l)
touch "$CLAUDE_CONFIG_DIR/activation.$n"
mkdir -p "$CLAUDE_CONFIG_DIR/projects"
if [ -f "$(dirname "$0")/sleep.$n" ]; then sleep 60; fi
cat "$(dirname "$0")/events.$n.jsonl"
"""


async def _real_resume_session(tmp_path, *activations: str, sleep_on: int | None = None):
    from nemo_gym.agents import HARNESS_RUNNER_FILE, HARNESS_RUNNER_SCRIPT, SandboxSession
    from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, create_provider

    work = tmp_path / "work"
    work.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "claude").write_text(STUB_CLAUDE)
    (runtime / "claude").chmod(0o755)
    for index, events in enumerate(activations):
        (runtime / f"events.{index}.jsonl").write_text(events + "\n")
    if sleep_on is not None:
        (runtime / f"sleep.{sleep_on}").write_text("")
    sandbox = AsyncSandbox(create_provider({"local": {}}))
    await sandbox.start(SandboxSpec(workdir=str(work)))
    directory = str(tmp_path / "session")
    (tmp_path / "session" / "config").mkdir(parents=True)
    await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
    session = SandboxSession(
        sandbox=sandbox, session_dir=directory, workdir=str(work), harness="Claude Code", owns_sandbox=True
    )
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(workdir=str(work)),
        runtime=str(runtime),
        session=session,
        transcript_path=str(tmp_path / "logs" / "claude-code.txt"),
    )
    instance = agent(
        anthropic_base_url="https://model.example", refusal_resumes=2, timeout=600, session_close_timeout_seconds=20
    )
    return instance, state, session


@pytest.mark.asyncio
async def test_a_refused_session_resumes_through_the_real_sandbox_session(tmp_path):
    instance, state, session = await _real_resume_session(tmp_path, refused(), finished())
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Question"}])

    response = await instance._sandbox_response(state, body, prompt="Question", system=None)

    config = tmp_path / "session" / "config"
    argv = (config / "argv.log").read_text().splitlines()
    assert len(argv) == 2 and "--resume" not in argv[0]
    assert "--resume sess-1" in argv[1] and argv[1].endswith(instance.config.refusal_resume_prompt)
    assert response.metadata == {
        "safety_refusals": "1",
        "safety_refusal_resumes": "1",
        "ended_on_safety_refusal": "false",
    }
    # The resumed activation kept Claude's config: its native transcripts were collected from there.
    assert session.artifacts is not None and session.artifacts.transcripts is not None
    transcript = (tmp_path / "logs" / "claude-code.txt").read_text().splitlines()
    assert len(transcript) == len(refused().splitlines()) + len(finished().splitlines())
    await state.close(20)
    assert session.cleanup_confirmed


@pytest.mark.asyncio
async def test_a_resume_that_cannot_start_keeps_the_refused_activation_as_the_result(tmp_path, monkeypatch):
    """F21: a rearm that fails after begin_resume reset the session (here the runner upload) is not a failed
    rollout, and it is not counted as a resume."""
    instance, state, session = await _real_resume_session(tmp_path, refused(), finished())
    upload = session.sandbox.upload

    async def failing_resume_upload(source, destination, *args, **kwargs):
        if "/resume-" in destination:
            raise RuntimeError("upload failed")
        return await upload(source, destination, *args, **kwargs)

    monkeypatch.setattr(session.sandbox, "upload", failing_resume_upload)
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Question"}])

    response = await instance._sandbox_response(state, body, prompt="Question", system=None)

    assert len((tmp_path / "session" / "config" / "argv.log").read_text().splitlines()) == 1
    # As with resumes off: the refusal's own outcome, not a sandbox error.
    assert response.status == "failed"
    assert response.metadata == {
        "safety_refusals": "1",
        "safety_refusal_resumes": "0",
        "ended_on_safety_refusal": "true",
    }
    # One activation's usage, not the refused events counted twice, and the refused activation's sandbox receipt.
    assert response.usage.input_tokens == 5 and response.usage.output_tokens == 1
    (observation,) = [record for record in state.observations.records if isinstance(record, SandboxObservation)]
    assert observation.exit_code == 0
    await state.close(20)


@pytest.mark.asyncio
async def test_a_resume_refused_because_the_session_is_closing_fails_the_call(monkeypatch):
    instance = agent(anthropic_base_url="https://model.example", refusal_resumes=1)
    state = resumable_state()
    monkeypatch.setattr(
        adapter.ClaudeCodeSandboxSession, "execute", AsyncMock(return_value=adapter.ClaudeCodeArtifacts(refused()))
    )

    async def closing_resume(self, *, timeout):
        self.session.closing = True
        raise RuntimeError("Claude Code sandbox session is closing")

    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "begin_resume", closing_resume)
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Question"}])

    with pytest.raises(RuntimeError, match="closing"):
        await instance._sandbox_response(state, body, prompt="Question", system=None)


@pytest.mark.asyncio
async def test_a_cancelled_resume_keeps_the_first_activation_and_stops_the_harness(tmp_path):
    instance, state, session = await _real_resume_session(tmp_path, refused(), finished(), sleep_on=1)
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Question"}])
    state.task = asyncio.create_task(instance._sandbox_response(state, body, prompt="Question", system=None))
    resumed = tmp_path / "session" / "config" / "activation.1"
    for _ in range(100):
        if resumed.exists():
            break
        await asyncio.sleep(0.1)
    # The resume is running (its stub sleeps) when close cancels it.
    assert resumed.exists() and session.session_dir.endswith("/resume-1")

    await state.close(20)

    assert state.task.cancelled() or isinstance(state.task.exception(), BaseException)
    # The verifier transcript still holds the first activation, refusal frame included, for the task's judges.
    assert "msg-refused" in (tmp_path / "logs" / "claude-code.txt").read_text()
    assert state.observations is not None and session.cleanup_confirmed
