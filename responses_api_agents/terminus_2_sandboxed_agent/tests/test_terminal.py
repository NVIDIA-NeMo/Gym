# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.core import NgFailureEvent
from responses_api_agents.terminus_2_sandboxed_agent.terminal import (
    TerminusJSONParser,
    TerminusTmuxSession,
    TerminusXMLParser,
)


@pytest.mark.parametrize("parser", [TerminusJSONParser, TerminusXMLParser])
@pytest.mark.parametrize("keys", ["C-c", "C-c\n", "C-c\r\n", "C-d", "C-z\n", "Enter"])
def test_control_keys_do_not_request_a_newline(parser, keys):
    if parser is TerminusJSONParser:
        response = json.dumps(
            {
                "analysis": "interrupt",
                "plan": "check",
                "commands": [{"keystrokes": keys, "duration": 1}, {"keystrokes": "pwd\n", "duration": 1}],
            }
        )
    else:
        response = (
            "<response><analysis>interrupt</analysis><plan>check</plan><commands>"
            f'<keystrokes duration="1">{keys}</keystrokes>'
            '<keystrokes duration="1">pwd\n</keystrokes></commands></response>'
        )
    result = parser().parse_response(response)
    assert result.error == ""
    assert result.commands[0].keystrokes == keys.rstrip("\r\n")
    assert "should end with newline" not in result.warning


def test_shell_text_and_unrelated_warnings_are_preserved():
    keys = ["echo C-c\n", "printf 'C-c\\n'\n", "C-c\necho after\n", "echo incomplete", "pwd\n"]
    result = TerminusJSONParser().parse_response(
        json.dumps({"analysis": "", "plan": "", "commands": [{"keystrokes": key} for key in keys]})
    )
    assert [command.keystrokes for command in result.commands] == keys
    assert "Command 4 should end with newline" in result.warning
    assert "Missing duration" in result.warning


def session(environment):
    return TerminusTmuxSession("test", environment, Path("/tmp/test.pane"), None, None)


@pytest.mark.asyncio
async def test_identical_buffer_is_not_reported_as_new_output():
    async def execute(command, **kwargs):
        return SimpleNamespace(return_code=0, stdout="still running\n", stderr="")

    terminal = session(SimpleNamespace(exec=execute))
    terminal._previous_buffer = "old output\nstill running\n"
    assert await terminal._find_new_content(terminal._previous_buffer) == ""
    assert "fresh output" in await terminal._find_new_content(terminal._previous_buffer + "fresh output\n")


@pytest.mark.asyncio
async def test_failed_input_drain_does_not_claim_interrupt_was_delivered():
    calls = []

    async def execute(command, **kwargs):
        calls.append(command)
        if "pane_dead" in command:
            return SimpleNamespace(return_code=0, stdout="0\n", stderr="")
        return SimpleNamespace(return_code=1, stdout="", stderr="terminal disappeared")

    with pytest.raises(RuntimeError, match="terminal disappeared"):
        await session(SimpleNamespace(exec=execute)).send_keys("C-c\n")
    assert sum("timeout 0.2 cat" in command for command in calls) == 1
    assert not any(command.startswith("tmux send-keys") for command in calls)


@pytest.mark.asyncio
async def test_controls_preserve_command_order():
    calls = []

    async def execute(command, **kwargs):
        if "pane_dead" not in command:
            calls.append(command)
        return SimpleNamespace(return_code=0, stdout="0\n", stderr="")

    await session(SimpleNamespace(exec=execute)).send_keys(["sleep 60\n", "C-c\n", "echo alive\n"])
    assert len(calls) == 4
    assert "sleep 60" in calls[0]
    assert "timeout 0.2 cat" in calls[1]
    assert calls[2].endswith(" C-c")
    assert "echo alive" in calls[3]


@pytest.mark.asyncio
async def test_missing_session_is_not_normal_completion():
    async def execute(command, **kwargs):
        return SimpleNamespace(return_code=1, stdout="", stderr="no server running")

    with pytest.raises(RuntimeError, match="session disappeared"):
        await session(SimpleNamespace(exec=execute)).is_session_alive()


@pytest.mark.asyncio
async def test_shell_exit_during_send_is_distinguished_from_transport_failure():
    from responses_api_agents.terminus_2_sandboxed_agent.terminal import ShellExitedError

    inspected = 0

    async def execute(command, **kwargs):
        nonlocal inspected
        if "pane_dead" in command:
            inspected += 1
            return SimpleNamespace(return_code=0, stdout="0" if inspected == 1 else "1", stderr="")
        return SimpleNamespace(return_code=1, stdout="", stderr="target pane has exited")

    with pytest.raises(ShellExitedError):
        await session(SimpleNamespace(exec=execute, session_id="test")).send_keys("echo after\n")


async def eventually(read, done, *, timeout: float = 30):
    """Poll ``read`` until ``done`` accepts its value and return that value; raise once ``timeout`` passes."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not done(value := await read()):
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"condition not met within {timeout} s; last value: {str(value)[-2000:]!r}")
        await asyncio.sleep(0.1)
    return value


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("tmux") is None, reason="requires Linux tmux")
@pytest.mark.asyncio
@pytest.mark.parametrize("exit_trigger", ["failure", "interrupt"])
async def test_exited_shell_recovery_discards_pending_input_and_reports_lost_state(tmp_path, caplog, exit_trigger):
    from responses_api_agents.terminus_2_sandboxed_agent.terminal import ShellExitedError

    caplog.set_level(logging.INFO, logger="harbor.utils.logger")

    class Environment:
        session_id = "recovery"

        async def exec(self, command, **kwargs):
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=tmp_path,
                env={**os.environ, "TMUX_TMPDIR": str(tmp_path)},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return SimpleNamespace(
                return_code=process.returncode,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )

    environment = Environment()
    terminal = TerminusTmuxSession(
        "test", environment, tmp_path / "pane.log", None, None, extra_env={"TERMINUS_CONFIGURED": "preserved"}
    )

    def pane_format(field):
        async def read():
            return (await environment.exec(f"tmux display-message -p -t test '{field}'")).stdout.strip()

        return read

    try:
        await terminal.start()
        await asyncio.sleep(0.3)
        initial = (await environment.exec("tmux display-message -p -t test '#{pane_current_path}'")).stdout.strip()
        pending = tmp_path / "queued.txt"
        forbidden = tmp_path / "must-not-execute"
        pending.write_text(f"touch {forbidden}\n" * 1024)
        await environment.exec(f"tmux load-buffer -b queued {pending}")
        failing_command = "sleep 2; false" if exit_trigger == "failure" else "sleep 120"
        # With PROMPT_COMMAND set, bash returns to the prompt after Ctrl-C instead of exiting under `set -e`.
        # Login profiles can set it (systemd's 80-systemd-osc-context.sh on Ubuntu 26.04), so clear it here.
        await terminal.send_keys(
            f"touch {tmp_path}/persisted; export TERMINUS_EPHEMERAL=lost; "
            f"mkdir {tmp_path}/sub; cd {tmp_path}/sub; unset PROMPT_COMMAND; set -e; {failing_command}\n",
            min_timeout_sec=0.2,
        )
        if exit_trigger == "interrupt":
            # Ctrl-C sent before sleep starts reaches bash instead, which survives it, so wait for sleep to run.
            await eventually(pane_format("#{pane_current_command}"), lambda command: command == "sleep")
        await environment.exec("tmux paste-buffer -d -b queued -t test")
        if exit_trigger == "interrupt":
            await terminal.send_keys("C-c", min_timeout_sec=0.3)
        await eventually(pane_format("#{pane_dead}"), lambda dead: dead == "1")
        assert await terminal.is_session_alive()
        with pytest.raises(ShellExitedError):
            await terminal.send_keys(f"touch {forbidden}\n")
        with pytest.raises(ShellExitedError):
            await terminal.get_incremental_output()
        observation = await terminal.recover_shell()
        assert "shell variables, options, and the working directory have reset" in observation
        assert "remaining commands in the previous response were skipped" in observation
        await terminal.send_keys(
            'printf \'RECOVERED=%s:%s:%s\\n\' "$TERMINUS_CONFIGURED" "${TERMINUS_EPHEMERAL-unset}" "$PWD"\n',
            min_timeout_sec=0.3,
        )
        expected = f"RECOVERED=preserved:unset:{initial}"
        output = ""

        async def accumulated_output():
            # get_incremental_output returns only new content, so keep what earlier polls returned.
            nonlocal output
            output += await terminal.get_incremental_output()
            return output

        await eventually(accumulated_output, lambda text: expected in text)
        assert (tmp_path / "persisted").exists()
        assert not forbidden.exists()

        # pipe-pane appends through a separate process, so the log can lag the screen.
        async def pane_log():
            return (tmp_path / "pane.log").read_text()

        await eventually(pane_log, lambda text: expected in text)
    finally:
        await environment.exec("tmux kill-server")


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("tmux") is None, reason="requires Linux tmux")
@pytest.mark.asyncio
@pytest.mark.parametrize("queued_commands", [160, 16384])
async def test_ctrl_c_recovers_a_full_pty_input_queue_without_executing_pending_text(
    tmp_path, caplog, queued_commands
):
    caplog.set_level(logging.INFO, logger="harbor.utils.logger")

    class Environment:
        session_id = "test"

        async def exec(self, command, **kwargs):
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                env={**os.environ, "TMUX_TMPDIR": str(tmp_path)},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return SimpleNamespace(
                return_code=process.returncode,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )

    environment = Environment()
    terminal = session(environment)
    marker = tmp_path / "should-not-execute"
    try:
        result = await environment.exec(terminal._tmux_start_session)
        assert result.return_code == 0, result.stderr
        await asyncio.sleep(0.3)
        await terminal.send_keys("sleep 120\n", min_timeout_sec=0.3)

        async def current_command() -> str:
            shown = await environment.exec("tmux display-message -p -t test '#{pane_current_command}'")
            return shown.stdout.strip()

        # Polled rather than slept on, so a slow runner cannot read the pane before sleep has started.
        await eventually(current_command, lambda command: command == "sleep")
        await terminal.send_keys(f"touch {marker}\n" * queued_commands, min_timeout_sec=0.3)
        # Unpatched tmux queues Ctrl-C after the paste and leaves sleep running.
        await environment.exec("tmux send-keys -t test C-c")
        # Time for the raw Ctrl-C to land; a slow runner can only make sleep look more stuck, never less.
        await asyncio.sleep(0.2)
        assert await current_command() == "sleep"
        await terminal.send_keys("C-c\n", min_timeout_sec=0.3)
        await terminal.send_keys("echo TERMINAL_RECOVERED\n", min_timeout_sec=0.3)
        await eventually(terminal.capture_pane, lambda pane: "\nTERMINAL_RECOVERED\n" in pane)
        assert not marker.exists()
        await terminal.send_keys("stty -echo -icanon -isig; head -c 4 | od -An -tx1; stty sane\n", min_timeout_sec=0.3)
        await terminal.send_keys(["xy", "C-c", "z"], min_timeout_sec=0.3)
        await eventually(terminal.capture_pane, lambda pane: "78 79 03 7a" in pane)
    finally:
        await environment.exec("tmux kill-server")


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["alive", "inspect", "drain", "recover"])
async def test_ordinary_nonzero_exits_stay_terminal_failures_not_provider_failures(stage):
    """Exit 127 (a missing stty), a killed tmux server or a failed respawn is the task's state on a responding sandbox."""

    async def execute(command, **kwargs):
        if "pane_dead" in command:
            if stage == "inspect":
                return SimpleNamespace(return_code=1, stdout="", stderr="no server")
            return SimpleNamespace(return_code=0, stdout="0\n", stderr="")
        return SimpleNamespace(return_code=127, stdout="", stderr="stty: command not found")

    terminal = session(SimpleNamespace(exec=execute))
    if stage == "alive":
        call = terminal.is_session_alive()
    elif stage == "inspect":
        call = terminal.get_incremental_output()
    elif stage == "recover":
        call = terminal.recover_shell()
    else:
        call = terminal.send_keys("C-c\n")
    with pytest.raises(RuntimeError) as error:
        await call
    assert not isinstance(error.value, NgFailureEvent)


@pytest.mark.asyncio
async def test_provider_failure_from_the_exec_layer_passes_through_the_terminal_unchanged():
    event = NgFailureEvent(
        boundary="sandbox", failure_kind="provider_unavailable", failure_reason="sandbox gone", http_status=503
    )

    async def execute(command, **kwargs):
        if "pane_dead" in command:
            return SimpleNamespace(return_code=0, stdout="0\n", stderr="")
        raise event

    with pytest.raises(NgFailureEvent) as error:
        await session(SimpleNamespace(exec=execute)).send_keys("C-c\n")
    assert error.value is event
