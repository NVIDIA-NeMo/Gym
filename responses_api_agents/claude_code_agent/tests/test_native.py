# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

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


def runtime_config(**kwargs):
    return native.NativeClaudeCodeConfig(
        name="judge",
        host="127.0.0.1",
        port=18313,
        entrypoint="native.py",
        model_server={"type": "responses_api_models", "name": "judge_model"},
        **kwargs,
    )


@pytest.mark.parametrize(
    "values",
    [
        {"runtime_archive_url": "https://runtime.example/claude.tar.gz"},
        {"runtime_archive_sha256": "0" * 64},
        {"runtime_archive_url": "file:///tmp/runtime.tar.gz", "runtime_archive_sha256": "0" * 64},
        {"runtime_archive_url": "https://runtime.example/runtime.tar.gz", "runtime_archive_sha256": "bad"},
    ],
)
def test_native_remote_runtime_requires_http_url_and_sha256(values):
    with pytest.raises(ValidationError):
        runtime_config(**values)


@pytest.mark.parametrize("case", ["valid", "digest_mismatch", "traversal", "symlink", "extra_file"])
def test_runtime_download_checks_digest_and_archive_paths(tmp_path, monkeypatch, case):
    binary = b"#!/bin/sh\nprintf '2.1.108 (Claude Code)\\n'\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        entry = tarfile.TarInfo("../claude" if case == "traversal" else "claude")
        if case == "symlink":
            entry.type = tarfile.SYMTYPE
            entry.linkname = "/usr/bin/false"
        else:
            entry.size = len(binary)
        archive.addfile(entry, io.BytesIO(binary))
        if case == "extra_file":
            archive.addfile(tarfile.TarInfo("unexpected"))
    payload = buffer.getvalue()
    expected = "0" * 64 if case == "digest_mismatch" else hashlib.sha256(payload).hexdigest()
    target = tmp_path / "runtime" / "claude"
    target.parent.mkdir()
    download = tmp_path / "runtime.tar.gz"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    monkeypatch.setattr(
        sys, "argv", ["download", "https://runtime.example/archive", expected, str(download), str(target)]
    )
    if case == "valid":
        exec(native.DOWNLOAD_RUNTIME, {})
        assert target.read_bytes() == binary
        assert target.stat().st_mode & 0o777 == 0o700
        assert not download.exists()
    else:
        with pytest.raises(ValueError, match="SHA-256 mismatch|must contain only"):
            exec(native.DOWNLOAD_RUNTIME, {})
        assert not target.exists()
        assert not (tmp_path / "claude").exists()


@pytest.mark.asyncio
async def test_remote_runtime_avoids_upload_and_still_checks_version(monkeypatch):
    def result(stdout="", return_code=0):
        return SimpleNamespace(return_code=return_code, stdout=stdout, stderr="", error_type=None)

    sandbox = SimpleNamespace(
        exec=AsyncMock(side_effect=[result(), result(return_code=1), result(), result("2.1.108 (Claude Code)")]),
        upload=AsyncMock(),
    )
    monkeypatch.setattr(native, "ensure_python", AsyncMock(return_value="/portable/python3"))
    monkeypatch.setattr(native.AsyncSandbox, "connect", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(native, "create_provider", lambda config: object())
    instance = agent()
    instance.config = runtime_config(
        runtime_archive="/local/fallback.tar.gz",
        runtime_archive_url="https://runtime.example/claude.tar.gz",
        runtime_archive_sha256="1" * 64,
        install_runtime=False,
    )
    instance.server_client = SimpleNamespace(global_config_dict={"sandbox": {"opensandbox": {}}})
    state = await instance._seed_agent_session_state(seed())
    assert state.version == "2.1.108 (Claude Code)"
    assert state.executable.endswith("/runtime/claude")
    assert not sandbox.upload.called
    assert "/portable/python3" in sandbox.exec.await_args_list[2].args[0]
    assert "https://runtime.example/claude.tar.gz" in sandbox.exec.await_args_list[2].args[0]
    assert "--version" in sandbox.exec.await_args_list[3].args[0]

    sandbox.exec.side_effect = [result(), result(return_code=1), result(), result("2.1.107 (Claude Code)")]
    monkeypatch.setattr(native.SandboxSession, "close", AsyncMock())
    with pytest.raises(RuntimeError, match="version mismatch"):
        await instance._seed_agent_session_state(seed())
