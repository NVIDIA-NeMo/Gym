# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local safety checks for the opt-in real-OpenSandbox DeepSWE OOM probe."""

import hashlib
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from scripts.deepswe_oom_probe import (
    AgentExecGate,
    ProbeNotValidated,
    allocator_command,
    cleanup_probe_sandboxes,
    load_probe_config,
    task_tree_sha256,
)

from nemo_gym.sandbox import SandboxExecResult, SandboxOOMKilledError


SETUP = 'echo "Shell: $SHELL" && echo "Installed OpenCode" && rm -f /tmp/nemo-gym-mcp-setup-error'


def _sandbox(*, result: SandboxExecResult | None = None, error: Exception | None = None) -> SimpleNamespace:
    execution = AsyncMock(side_effect=error) if error is not None else AsyncMock(return_value=result)
    return SimpleNamespace(sandbox_id="seeded-sandbox", exec=execution)


def test_allocator_is_shell_quoted_python_without_running_it() -> None:
    argv = shlex.split(allocator_command())

    assert argv[:3] == ["exec", "python3", "-c"]
    assert len(argv) == 4
    compile(argv[3], "<bounded-allocator>", "exec")


@pytest.mark.asyncio
async def test_unknown_primary_command_cannot_reach_real_sandbox() -> None:
    sandbox = _sandbox(result=SandboxExecResult(stdout="", stderr="", return_code=0))
    gate = AgentExecGate(sandbox, seeded_sandbox_id="seeded-sandbox", seeded_at=0)

    await gate.exec(SETUP)
    with pytest.raises(ProbeNotValidated, match="refusing model inference"):
        await gate.exec('opencode run --title "changed title" -- "prompt"')

    sandbox.exec.assert_awaited_once_with(SETUP)
    assert gate.agent_exec_started is False


@pytest.mark.asyncio
async def test_setup_containing_model_command_is_rejected_before_real_exec() -> None:
    sandbox = _sandbox(result=SandboxExecResult(stdout="", stderr="", return_code=0))
    gate = AgentExecGate(sandbox, seeded_sandbox_id="seeded-sandbox", seeded_at=0)

    with pytest.raises(ProbeNotValidated, match="refusing model inference"):
        await gate.exec(SETUP.replace("&& echo", '&& opencode run --title "unexpected" && echo'))

    sandbox.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_child_exit_137_does_not_validate_provider_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _sandbox(result=SandboxExecResult(stdout="", stderr="", return_code=0))
    gate = AgentExecGate(sandbox, seeded_sandbox_id="seeded-sandbox", seeded_at=0)
    monkeypatch.setattr("scripts.deepswe_oom_probe.monotonic", lambda: 0)

    await gate.exec(SETUP)
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=137)
    with pytest.raises(ProbeNotValidated, match="child exit 137 is not validation"):
        await gate.exec('opencode run --title "NG dummy title" -- "prompt"')

    assert gate.agent_exec_started is True
    assert gate.provider_oom is None
    assert sandbox.exec.await_count == 2


@pytest.mark.asyncio
async def test_provider_oom_must_name_seeded_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _sandbox(result=SandboxExecResult(stdout="", stderr="", return_code=0))
    gate = AgentExecGate(sandbox, seeded_sandbox_id="seeded-sandbox", seeded_at=0)
    monkeypatch.setattr("scripts.deepswe_oom_probe.monotonic", lambda: 0)
    await gate.exec(SETUP)
    sandbox.exec.side_effect = SandboxOOMKilledError("provider OOMKilled", sandbox_id="other-sandbox")

    with pytest.raises(ProbeNotValidated, match="different sandbox"):
        await gate.exec('opencode run --title "NG dummy title" -- "prompt"')

    assert gate.provider_oom is None


@pytest.mark.asyncio
async def test_cleanup_requires_provider_confirmed_stop_of_seeded_sandbox() -> None:
    seeded = SimpleNamespace(sandbox_id="seeded-sandbox", _stopped=False, _handle=object())

    async def stop() -> None:
        seeded._stopped = True

    seeded.stop = AsyncMock(side_effect=stop)
    cleanup = await cleanup_probe_sandboxes(seeded=seeded, connected=None, expected_id="seeded-sandbox")

    assert cleanup["same_sandbox_stop_confirmed"] is True
    assert cleanup["seeded_stop_confirmed"] is True
    assert cleanup["errors"] == []
    seeded.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_fails_closed_when_stop_does_not_confirm_teardown() -> None:
    seeded = SimpleNamespace(
        sandbox_id="seeded-sandbox",
        _stopped=False,
        _handle=object(),
        stop=AsyncMock(side_effect=RuntimeError("close failed")),
    )
    cleanup = await cleanup_probe_sandboxes(seeded=seeded, connected=None, expected_id="seeded-sandbox")

    assert cleanup["same_sandbox_stop_confirmed"] is False
    assert cleanup["seeded_stop_confirmed"] is False
    assert cleanup["errors"] == ["seeded stop failed: RuntimeError: close failed"]


def test_probe_disables_sandbox_create_and_command_retries(tmp_path: Path) -> None:
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(
        "sandbox:\n  opensandbox:\n    create:\n      retries: 2\n"
        "    operations:\n      command_retries: 3\n"
        "deepswe_opencode_resources_server: {}\n"
        "deepswe_opencode_sandboxed_agent: {}\n",
        encoding="utf-8",
    )
    config = load_probe_config(config_path, expected_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())

    assert config.sandbox.opensandbox.create.retries == 0
    assert config.sandbox.opensandbox.create.renew_interval_s is None
    assert config.sandbox.opensandbox.operations.command_retries == 0


def test_frozen_task_tree_hash_changes_with_source_bytes(tmp_path: Path) -> None:
    task_file = tmp_path / "task" / "instruction.md"
    task_file.parent.mkdir()
    task_file.write_text("first", encoding="utf-8")
    first_hash = task_tree_sha256(tmp_path)
    task_file.write_text("second", encoding="utf-8")

    assert task_tree_sha256(tmp_path) != first_hash
