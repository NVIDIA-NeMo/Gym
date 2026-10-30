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

from nemo_gym.base_responses_api_agent import AGENT_SESSION_COOKIE_KEY, AgentSeedSessionRequest
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
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
    provider = SimpleNamespace(aclose=AsyncMock())
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
    from nemo_gym.agent_utils.sandbox_session import SandboxSession
    from nemo_gym.agent_utils.supervisor_client import HarnessTerminationError

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
        session.termination_error = HarnessTerminationError(harness="Claude Code", receipt=receipt)
    state = adapter.ClaudeCodeSandboxSession(request=seed(), runtime="/runtime", session=session)
    close = await agent()._close_agent_session_state(state)
    remote.disconnect.assert_awaited_once()
    assert close.cleanup_confirmed is confirmed
    assert close.model_dump(mode="json")["cleanup_confirmed"] is confirmed
