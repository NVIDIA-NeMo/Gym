# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle, ObservationGap
from responses_api_agents.claude_code_agent import sandbox as app
from responses_api_agents.claude_code_agent.app import ClaudeCodeAgent, ClaudeCodeAgentConfig


def seed(**kwargs):
    return AgentSeedSessionRequest(
        agent_session_id="test",
        episode_id={"rollout_id": "test"},
        task_id={"taskset": "test", "task_id": "fixture"},
        sandbox_access={
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"id": "s"}},
            "workdir": "/tmp",
        },
        **kwargs,
    )


def agent():
    return ClaudeCodeAgent.model_construct(
        config=ClaudeCodeAgentConfig(
            name="judge",
            host="127.0.0.1",
            port=18313,
            entrypoint="app.py",
            claude_code_version="2.1.108",
            model="claude-opus-4-6",
            timeout=1200,
            max_turns=50,
            model_server={"type": "responses_api_models", "name": "judge_model"},
        )
    )


@pytest.mark.parametrize("input", ["Judge fixture", [{"role": "user", "content": "Judge fixture"}]])
def test_single_text_input(input):
    assert app.extract_prompt(NeMoGymResponseCreateParamsNonStreaming(input=input, instructions="Frozen")) == (
        "Judge fixture",
        "Frozen",
    )


def test_system_and_developer_remain_system_instructions():
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": "System"}]},
            {"role": "developer", "content": "Developer"},
            {"role": "user", "content": "Question"},
        ],
        instructions="Request instructions",
    )
    assert app.extract_prompt(body) == ("Question", "System\n\nDeveloper\n\nRequest instructions")


@pytest.mark.parametrize(
    "input",
    [
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.org/image.png"}]}],
        [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}],
    ],
)
def test_unsupported_input_is_rejected_before_launch(input):
    with pytest.raises(ValueError):
        app.extract_prompt(NeMoGymResponseCreateParamsNonStreaming(input=input))


@pytest.mark.asyncio
async def test_seed_installs_runtime_inside_borrowed_sandbox(monkeypatch):
    def result(stdout="", code=0):
        return SimpleNamespace(return_code=code, stdout=stdout, stderr="", error_type=None)

    sandbox = SimpleNamespace(
        upload=AsyncMock(), exec=AsyncMock(side_effect=[result(), result("2.1.108 (Claude Code)")])
    )
    connector = AsyncMock(return_value=sandbox)
    provider = object()
    monkeypatch.setattr(app.AsyncSandbox, "connect", connector)
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    instance = agent()
    instance.server_client = SimpleNamespace(global_config_dict={"sandbox": {"opensandbox": {}}})
    state = await instance._seed_agent_session_state(seed(timeout_seconds=30))
    assert connector.await_args.kwargs["provider"] is provider
    commands = [call.args[0] for call in sandbox.exec.await_args_list]
    assert "install.py" in commands[-1] and "2.1.108" in commands[-1]
    assert len(sandbox.upload.await_args_list) == 2
    assert not state.session.owns_sandbox
    assert state.version == "2.1.108 (Claude Code)"


@pytest.mark.asyncio
async def test_pinned_argv_uses_supervisor_return_code_and_deadline(monkeypatch):
    files = {}

    async def upload(sandbox, *, path, text):
        files[path] = text

    monkeypatch.setattr(app, "upload_text", upload)
    instance = agent()
    monkeypatch.setattr(ClaudeCodeAgent, "_resolve_call_base_url", lambda self, *args: "http://judge.example:80")
    captured = {}

    class Session:
        session_dir = "/tmp/private-judge"
        sandbox = object()
        cleanup = {"cleanup_confirmed": True, "timed_out": False, "return_code": 7}

        async def execute(self, **kwargs):
            await kwargs["stage_activation"]()
            captured.update(kwargs)
            return "\n".join(
                map(
                    json.dumps,
                    [
                        {
                            "type": "assistant",
                            "message": {
                                "content": [{"type": "text", "text": "written"}],
                                "usage": {"input_tokens": 10, "output_tokens": 2},
                            },
                        },
                        {
                            "type": "result",
                            "subtype": "success",
                            "is_error": False,
                            "usage": {"input_tokens": 10, "output_tokens": 2},
                        },
                    ],
                )
            )

    state = app.ClaudeCodeSandboxedState(
        request=seed(), session=Session(), executable="/tmp/runtime/claude", version="2.1.108 (Claude Code)"
    )
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="Judge fixture", instructions="Frozen system", timeout_seconds=600
    )
    response = await instance._execute(state, body, instance._execution_input(body))
    launch = json.loads(files["/tmp/private-judge/launch.json"])
    assert launch["argv"][-2:] == ["--", "Judge fixture"]
    assert launch["argv"][launch["argv"].index("--append-system-prompt") + 1] == "Frozen system"
    assert "--bare" in launch["argv"]
    assert "--max-turns" in launch["argv"] and "50" in launch["argv"]
    assert launch["env"]["ANTHROPIC_BASE_URL"] == "http://judge.example:80"
    assert 0 < captured["timeout"] <= 600 and captured["deadline"] > app.monotonic()
    assert response.status == "failed" and response.metadata["stop_reason"] == "process_exit_7"
    assert response.usage.total_tokens == 12


@pytest.mark.asyncio
async def test_close_requires_cleanup_and_returns_generic_observations():
    instance = agent()
    session = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("unconfirmed cleanup")))
    observations = AgentObservationBundle(source="claude_code", gaps=[ObservationGap(code="test_gap")])
    state = app.ClaudeCodeSandboxedState(request=seed(), session=session, observations=observations)
    with pytest.raises(RuntimeError, match="unconfirmed cleanup"):
        await instance._close_agent_session_state(state)
    session.close.side_effect = None
    session.cleanup_confirmed = True
    receipt = await instance._close_agent_session_state(state)
    assert receipt.cleanup_confirmed and receipt.agent_observations == observations


@pytest.mark.parametrize(
    "control",
    [
        {"tool_choice": "none"},
        {"parallel_tool_calls": False},
        {"background": True},
        {"temperature": 0.5},
        {"max_output_tokens": 100},
        {"reasoning": {"effort": "low"}},
        {"model": "different-model"},
        {"custom_control": "ignored"},
    ],
)
def test_unsupported_controls_are_rejected_before_launch(control):
    with pytest.raises(ValueError):
        app.extract_prompt(NeMoGymResponseCreateParamsNonStreaming(input="Judge", **control), model="fixed-model")


def test_matching_configured_model_is_supported():
    assert app.extract_prompt(
        NeMoGymResponseCreateParamsNonStreaming(input="Judge", model="fixed-model"), model="fixed-model"
    ) == ("Judge", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [{"tool_choice": "none"}, {"temperature": 0.5}])
async def test_invalid_request_can_be_corrected_in_the_same_session(monkeypatch, invalid):
    instance = agent()
    state = app.ClaudeCodeSandboxedState(request=seed(), session=SimpleNamespace())

    @asynccontextmanager
    async def locked(self, session_id):
        assert session_id == "test"
        yield SimpleNamespace(closing=False, state=state)

    monkeypatch.setattr(ClaudeCodeAgent, "_locked_agent_session", locked)
    monkeypatch.setattr(ClaudeCodeAgent, "_agent_session_id_from_request", lambda *args: "test")
    execute = AsyncMock(return_value=SimpleNamespace(status="completed"))
    monkeypatch.setattr(ClaudeCodeAgent, "_execute", execute)
    with pytest.raises(app.HTTPException) as error:
        await instance.responses(None, NeMoGymResponseCreateParamsNonStreaming(input="Judge", **invalid))
    assert error.value.status_code == 422
    assert state.body is state.execution is None
    body = NeMoGymResponseCreateParamsNonStreaming(input="Judge")
    result = await instance.responses(None, body)
    assert result.status == "completed" and state.body == body
    execute.assert_awaited_once()


@pytest.mark.parametrize(
    ("terminal", "timed_out", "expected_status", "expected_reason"),
    [
        ({"subtype": "success", "is_error": False}, False, "completed", None),
        ({"subtype": "success", "is_error": True, "stop_reason": "refusal"}, False, "failed", "refusal"),
        ({"subtype": "success", "is_error": True}, False, "failed", "agent_error"),
        ({"subtype": "error_max_turns", "is_error": True}, False, "incomplete", "error_max_turns"),
        ({"subtype": "success", "is_error": False}, True, "incomplete", "timeout"),
    ],
)
async def test_semantic_outcome_reaches_response_and_native_observations(
    monkeypatch, tmp_path, terminal, timed_out, expected_status, expected_reason
):
    import tarfile

    from nemo_gym.agent_utils.sandbox_session import SandboxSession
    from nemo_gym.rollout_observability import AgentInvocation

    root = tmp_path / "projects/work"
    root.mkdir(parents=True)
    (root / "root.jsonl").write_text(
        json.dumps(
            {
                "type": "assistant",
                "sessionId": "root",
                "uuid": "answer",
                "timestamp": "2026-10-14T00:00:00Z",
                "message": {"id": "native-answer", "role": "assistant", "content": [{"type": "text", "text": "done"}]},
            }
        )
    )
    archive = tmp_path / "transcripts.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(tmp_path / "projects", arcname="projects")

    async def download(source, destination):
        destination.write_bytes(archive.read_bytes())

    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None)),
        download=download,
    )
    session = SandboxSession(sandbox=sandbox, session_dir="/session", workdir="/app", harness="claude_code")
    session.cleanup = {"cleanup_confirmed": True, "timed_out": timed_out, "return_code": 0}
    raw = json.dumps({"type": "result", "usage": {"input_tokens": 10, "output_tokens": 2}, **terminal})
    monkeypatch.setattr(SandboxSession, "read_output_log", AsyncMock(return_value=raw))

    async def execute(self, **kwargs):
        return await kwargs["collect"]()

    monkeypatch.setattr(SandboxSession, "execute", execute)
    state = app.ClaudeCodeSandboxedState(request=seed(), session=session)
    instance = agent()
    body = NeMoGymResponseCreateParamsNonStreaming(input="solve")
    response = await instance._execute(state, body, instance._execution_input(body))
    invocation = next(item for item in state.observations.records if isinstance(item, AgentInvocation))
    assert response.status == invocation.status == expected_status
    assert response.metadata.get("stop_reason") == invocation.error_type == expected_reason
    assert not state.observations.gaps


@pytest.mark.asyncio
async def test_connect_timeout_releases_provider(monkeypatch):
    import asyncio

    provider = SimpleNamespace(aclose=AsyncMock())
    entered = asyncio.Event()

    async def connect(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    monkeypatch.setattr(app.AsyncSandbox, "connect", connect)
    instance = agent()
    instance.server_client = SimpleNamespace(global_config_dict={"sandbox": {"opensandbox": {}}})
    with pytest.raises(TimeoutError):
        await instance._seed_agent_session_state(seed(timeout_seconds=0.01))
    assert entered.is_set()
    provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_owned_session_installs_and_stops_sandbox(monkeypatch):
    from nemo_gym.sandbox import SandboxSpec

    sandbox = SimpleNamespace(
        start=AsyncMock(),
        stop=AsyncMock(),
        upload=AsyncMock(),
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None),
                SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None),
                SimpleNamespace(return_code=0, stdout="2.1.108 (Claude Code)", stderr="", error_type=None),
            ]
        ),
    )
    monkeypatch.setattr(app, "AsyncSandbox", lambda provider: sandbox)
    monkeypatch.setattr(app, "create_provider", lambda config: object())
    instance = agent()
    instance.server_client = SimpleNamespace(global_config_dict={"sandbox": {"opensandbox": {}}})
    instance.config.sandbox_provider = "sandbox"
    instance.config.sandbox_spec = SandboxSpec(image="fixture", workdir="/work")
    body = seed().model_copy(update={"sandbox_access": None})
    state = await instance._seed_agent_session_state(body)
    assert state.session.owns_sandbox
    sandbox.start.assert_awaited_once_with(instance.config.sandbox_spec)
    receipt = await instance._close_agent_session_state(state)
    assert receipt.cleanup_confirmed
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_install_is_lazy_shared_and_retryable(monkeypatch):
    import asyncio

    from responses_api_agents.claude_code_agent import app as local_app

    calls = []

    def install(version):
        calls.append(version)
        if len(calls) == 1:
            raise RuntimeError("download failed")

    monkeypatch.setattr(local_app, "ensure_claude_code", install)
    instance = agent()
    assert calls == []
    with pytest.raises(RuntimeError, match="download failed"):
        await instance._ensure_local_runtime()
    await asyncio.gather(instance._ensure_local_runtime(), instance._ensure_local_runtime())
    assert calls == ["2.1.108", "2.1.108"]
