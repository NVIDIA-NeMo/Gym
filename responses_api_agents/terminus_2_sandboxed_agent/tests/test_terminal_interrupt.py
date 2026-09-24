# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from harbor.agents.terminus_2.terminus_2 import Command
from harbor.agents.terminus_2.tmux_session import TmuxSession

from responses_api_agents.terminus_2_sandboxed_agent.app import NeMoGymLLM, NeMoGymTerminus2


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or any(shutil.which(tool) is None for tool in ("tmux", "stty", "timeout", "dd")),
    reason="Terminal recovery integration tests require Linux, tmux, and coreutils",
)


class LocalEnvironment:
    """Run real terminal operations against an isolated tmux server."""

    def __init__(self):
        self.socket = "gym-interrupt-test-" + uuid4().hex

    async def exec(self, command, *, timeout_sec=10, user=None):
        wrapped = f'tmux() {{ command tmux -L {shlex.quote(self.socket)} -f /dev/null "$@"; }}; {command}'
        process = await asyncio.create_subprocess_exec(
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            wrapped,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "TMUX": ""},
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return SimpleNamespace(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )


@pytest.fixture
async def terminal(tmp_path):
    environment = LocalEnvironment()
    try:
        result = await environment.exec("tmux new-session -d -s probe 'bash --noprofile --norc'")
        assert result.return_code == 0, result.stderr
        session = TmuxSession(
            session_name="probe",
            environment=environment,
            logging_path=tmp_path / "pane.log",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
        )
        await asyncio.sleep(0.1)
        yield session
    finally:
        await environment.exec("tmux kill-server")


def agent(tmp_path, *, recovery=True):
    return NeMoGymTerminus2(
        logs_dir=tmp_path,
        model_name="test",
        llm=MagicMock(spec=NeMoGymLLM),
        dump_trajectory=False,
        recover_stalled_interrupts=recovery,
    )


async def start_sleep(session):
    await session.send_keys("sleep 300\n", min_timeout_sec=0.2)
    result = await session.environment.exec(
        "pane_pid=$(tmux display-message -p -t probe '#{pane_pid}'); pgrep -P \"$pane_pid\" -x sleep"
    )
    assert result.return_code == 0, result.stderr
    return int(result.stdout)


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("lines", [110, 1100])
async def test_backlogged_interrupt_discards_stale_input_and_preserves_next_command(
    terminal, tmp_path, recovery, lines
):
    sleep_pid = await start_sleep(terminal)
    stale = tmp_path / "stale-command-ran"
    fresh = tmp_path / "fresh-command-ran"
    payload = ("# filler " + "x" * 80 + "\n") * lines + f"touch {shlex.quote(str(stale))}\n"
    await terminal.send_keys(payload, min_timeout_sec=0.2)

    instance = agent(tmp_path, recovery=recovery)
    timeout, _ = await instance._execute_commands(
        [Command("C-c", 0.2), Command(f"echo recovered > {shlex.quote(str(fresh))}\n", 0.2)], terminal
    )

    assert not timeout
    assert instance._terminal_interrupt_drains == int(recovery)
    assert instance._terminal_interrupt_drain_errors == 0
    assert Path(f"/proc/{sleep_pid}").exists() is not recovery
    assert not stale.exists(), "A queued command must not execute during recovery"
    assert fresh.exists() is recovery
    if recovery:
        assert fresh.read_text() == "recovered\n"


async def test_normal_interrupt_and_multiline_commands(terminal, tmp_path):
    output = tmp_path / "content"
    instance = agent(tmp_path)
    expected = "first line\nC-c\nlast line\n"
    await instance._execute_commands(
        [Command(f"cat > {shlex.quote(str(output))} <<'EOF'\n{expected}EOF\n", 0.2)], terminal
    )
    sleep_pid = await start_sleep(terminal)
    timeout, observation = await instance._execute_commands(
        [Command("C-c", 0.2), Command("echo HEALTHY_TERMINAL\n", 0.2)], terminal
    )
    assert not timeout
    assert not Path(f"/proc/{sleep_pid}").exists()
    assert output.read_text() == expected
    assert "HEALTHY_TERMINAL" in observation


async def test_malformed_interrupt_stays_literal(terminal, tmp_path):
    sleep_pid = await start_sleep(terminal)
    instance = agent(tmp_path)
    instance._recover_stalled_interrupt = AsyncMock()
    await instance._execute_commands([Command("C-c\n", 0.2)], terminal)
    assert Path(f"/proc/{sleep_pid}").exists()
    instance._recover_stalled_interrupt.assert_not_awaited()


@pytest.mark.parametrize("modes", ["-icanon", "-isig", "noflsh", "intr '^X'"])
async def test_nonstandard_terminal_modes_preserve_pending_input(terminal, tmp_path, modes):
    await start_sleep(terminal)
    result = await terminal.environment.exec(
        f"pane_tty=$(tmux display-message -p -t probe '#{{pane_tty}}'); stty -F \"$pane_tty\" {modes}"
    )
    assert result.return_code == 0, result.stderr
    await terminal.send_keys("PRESERVE_INPUT\n", min_timeout_sec=0.1)
    instance = agent(tmp_path)
    await instance._recover_stalled_interrupt(terminal)
    assert instance._terminal_interrupt_drains == 0
    assert instance._terminal_interrupt_drain_errors == 0
    result = await terminal.environment.exec(
        "pane_tty=$(tmux display-message -p -t probe '#{pane_tty}'); "
        'timeout 1 dd if="$pane_tty" bs=1 count=15 status=none'
    )
    assert "PRESERVE_INPUT" in result.stdout


async def test_recovery_failure_still_returns_terminal_observation(tmp_path, caplog):
    instance = agent(tmp_path)
    session = SimpleNamespace(
        _session_name="probe",
        _user=None,
        environment=SimpleNamespace(exec=AsyncMock(side_effect=TimeoutError("control channel unavailable"))),
        send_keys=AsyncMock(),
        get_incremental_output=AsyncMock(return_value="Current terminal state"),
    )
    timeout, observation = await instance._execute_commands([Command("C-c", 0.1)], session)
    assert not timeout
    assert observation == "Current terminal state"
    assert "input drain failed" in caplog.text
    assert instance._terminal_interrupt_drain_errors == 1


async def test_command_timeout_preserves_harbor_feedback(tmp_path):
    session = SimpleNamespace(
        send_keys=AsyncMock(side_effect=TimeoutError()),
        get_incremental_output=AsyncMock(return_value="SLOW_COMMAND_OUTPUT"),
    )
    timeout, observation = await agent(tmp_path)._execute_commands([Command("sleep 100\n", 0.1)], session)
    assert timeout
    assert "SLOW_COMMAND_OUTPUT" in observation
    assert "sleep 100" in observation
