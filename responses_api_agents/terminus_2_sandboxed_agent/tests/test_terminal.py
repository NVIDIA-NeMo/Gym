# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        return SimpleNamespace(return_code=1, stdout="", stderr="terminal disappeared")

    with pytest.raises(RuntimeError, match="terminal disappeared"):
        await session(SimpleNamespace(exec=execute)).send_keys("C-c\n")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_controls_preserve_command_order():
    calls = []

    async def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    await session(SimpleNamespace(exec=execute)).send_keys(["sleep 60\n", "C-c\n", "echo alive\n"])
    assert len(calls) == 4
    assert "sleep 60" in calls[0]
    assert "timeout 0.2 cat" in calls[1]
    assert calls[2].endswith(" C-c")
    assert "echo alive" in calls[3]


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("tmux") is None, reason="requires Linux tmux")
@pytest.mark.asyncio
async def test_ctrl_c_recovers_a_full_pty_input_queue_without_executing_pending_text(tmp_path):
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
        await terminal.send_keys(f"touch {marker}\n" * 160, min_timeout_sec=0.3)
        # Unpatched tmux queues Ctrl-C after the paste and leaves sleep running.
        await environment.exec("tmux send-keys -t test C-c")
        await asyncio.sleep(0.2)
        before = await environment.exec("tmux display-message -p -t test '#{pane_current_command}'")
        assert before.stdout.strip() == "sleep"
        await terminal.send_keys("C-c\n", min_timeout_sec=0.3)
        await terminal.send_keys("echo TERMINAL_RECOVERED\n", min_timeout_sec=0.3)
        assert "\nTERMINAL_RECOVERED\n" in await terminal.capture_pane()
        assert not marker.exists()
        await terminal.send_keys("stty -echo -icanon -isig; head -c 4 | od -An -tx1; stty sane\n", min_timeout_sec=0.3)
        await terminal.send_keys(["xy", "C-c", "z"], min_timeout_sec=0.3)
        assert "78 79 03 7a" in await terminal.capture_pane()
    finally:
        await environment.exec("tmux kill-server")
