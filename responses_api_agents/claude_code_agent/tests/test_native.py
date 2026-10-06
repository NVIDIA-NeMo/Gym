# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.claude_code_agent import native


def seed():
    return AgentSeedSessionRequest(
        agent_session_id="test",
        episode_id={"rollout_id": "test"},
        task_id={"taskset": "test", "task_id": "fixture"},
        sandbox_access={
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"id": "s"}},
            "workdir": "/tmp",
        },
    )


def agent():
    return native.NativeClaudeCodeAgent.model_construct(
        config=SimpleNamespace(
            model_server=SimpleNamespace(name="judge_model"),
            model="claude-opus-4-6",
            timeout=1200,
            max_turns=50,
            close_timeout=90,
            debug_log=False,
        )
    )


@pytest.mark.asyncio
async def test_native_argv_pin_and_supervision(tmp_path, monkeypatch):
    files = {}

    async def upload(sandbox, *, path, text):
        files[path] = text

    monkeypatch.setattr(native, "upload_text", upload)
    config = SimpleNamespace(global_config_dict={})
    instance = agent()
    instance.server_client = config
    monkeypatch.setattr(
        native.NativeClaudeCodeAgent, "resolve_model_base_url", lambda self, *args: "http://judge.example:80/v1"
    )
    captured = {}

    class Sandbox:
        session_dir = "/tmp/private-judge"
        sandbox = object()
        cleanup = {"cleanup_confirmed": True, "timed_out": False}

        async def read_output_log(self):
            return ""

        async def execute(self, **kwargs):
            command = await kwargs["stage_activation"]()
            captured.update(kwargs)
            captured["command"] = command
            return (
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": "verdict written"}],
                            "usage": {"input_tokens": 10, "output_tokens": 2},
                        },
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                    }
                )
            )

    state = native.NativeClaudeState(
        request=seed(), session=Sandbox(), executable="/tmp/runtime/claude", version="2.1.108 (Claude Code)"
    )
    response = await instance._execute(
        state,
        NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Judge fixture"}],
            instructions="Frozen system",
            metadata={"timeout_seconds": "600"},
        ),
    )
    launch = json.loads(files["/tmp/private-judge/launch.json"])
    assert launch["argv"][:7] == [
        "/tmp/runtime/claude",
        "--print",
        "--max-turns",
        "50",
        "--model",
        "claude-opus-4-6",
        "--dangerously-skip-permissions",
    ]
    assert "--setting-sources" in launch["argv"] and "--bare" not in launch["argv"]
    assert launch["argv"][-3:] == ["--append-system-prompt", "Frozen system", "Judge fixture"]
    assert launch["env"]["ANTHROPIC_BASE_URL"] == "http://judge.example:80"
    assert launch["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-opus-4-6"
    assert response.status == "completed"
    assert "verdict written" in response.metadata["raw_log"]
    assert launch["env"]["HOME"] == "/tmp/private-judge/home"
    assert captured["timeout"] == 600 and captured["close_timeout"] == 90
    assert response.usage.total_tokens == 12 and response.metadata["runtime_version"] == "2.1.108 (Claude Code)"


@pytest.mark.asyncio
async def test_native_close_requires_supervisor_cleanup_before_receipt():
    instance = agent()
    sandbox = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("unconfirmed cleanup")))
    state = native.NativeClaudeState(request=seed(), session=sandbox)
    with pytest.raises(RuntimeError, match="unconfirmed cleanup"):
        await instance._close_agent_session_state(state)
    sandbox.close.side_effect = None
    receipt = await instance._close_agent_session_state(state)
    assert receipt.cleanup_confirmed


@pytest.mark.asyncio
async def test_native_seed_refuses_missing_sandbox():
    instance = agent()
    body = seed().model_copy(update={"sandbox_access": None})
    with pytest.raises(ValueError, match="SandboxAccess"):
        await instance._seed_agent_session_state(body)


@pytest.mark.asyncio
async def test_native_seed_reconnects_provider_and_reuses_verified_binary(monkeypatch):
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout=""),
                SimpleNamespace(return_code=0, stdout="/usr/local/bin/claude\n"),
                SimpleNamespace(return_code=0, stdout="2.1.108 (Claude Code)\n"),
                SimpleNamespace(return_code=0, stdout="2.1.108 (Claude Code)\n"),
            ]
        ),
        upload=AsyncMock(),
    )
    provider = object()
    connector = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(native, "ensure_python", AsyncMock(return_value="python3"))
    monkeypatch.setattr(native.AsyncSandbox, "connect", connector)
    monkeypatch.setattr(native, "create_provider", lambda config: provider)
    instance = agent()
    instance.config = native.NativeClaudeCodeConfig(
        name="judge",
        host="127.0.0.1",
        port=18313,
        entrypoint="native.py",
        model_server={"type": "responses_api_models", "name": "judge_model"},
        runtime_archive="/unused/pinned.tar.gz",
    )
    instance.server_client = SimpleNamespace(global_config_dict={"sandbox": {"opensandbox": {}}})
    state = await instance._seed_agent_session_state(seed())
    assert connector.await_args.kwargs["provider"] is provider
    assert state.executable == "/usr/local/bin/claude" and state.version == "2.1.108 (Claude Code)"
    assert not sandbox.upload.called
