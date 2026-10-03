# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import time
from pathlib import Path
from typing import AsyncIterator

import pytest

from resources_servers.codex_tools.exec_runtime import (
    UNIFIED_EXEC_ENV,
    ExecError,
    ExecRuntime,
    HeadTailBuffer,
    _truncate_middle,
    format_exec_response,
)


_HEADER = re.compile(
    r"Chunk ID: [0-9a-f]{6}\nWall time: \d+\.\d{4} seconds\n"
    r"(?:Process exited with code (?P<exit>-?\d+)\n)?(?:Process running with session ID (?P<session>\d+)\n)?"
    r"Original token count: (?P<tokens>\d+)\nOutput:\n(?P<output>.*)\Z",
    re.S,
)


def parse(response: str) -> dict:
    match = _HEADER.match(response)
    assert match, response
    return match.groupdict()


@pytest.fixture
async def runtime(tmp_path: Path) -> AsyncIterator[ExecRuntime]:
    runtime = ExecRuntime(cwd=str(tmp_path), shell="/bin/bash", login=False, env={**os.environ, **UNIFIED_EXEC_ENV})
    yield runtime
    runtime.close()


class TestTruncation:
    # Expected strings from codex-rs utils/string/src/truncate/tests.rs.
    @pytest.mark.parametrize(
        "text, max_tokens, expected",
        [
            ("short output", 100, "short output"),
            ("abcdef", 0, "…2 tokens truncated…"),
            ("😀😀😀😀😀😀😀😀😀😀\nsecond line with text\n", 8, "😀😀😀😀…8 tokens truncated… line with text\n"),
        ],
    )
    def test_truncate_middle_matches_upstream(self, text: str, max_tokens: int, expected: str) -> None:
        assert _truncate_middle(text, max_tokens) == expected

    def test_large_output_is_truncated_with_warning(self) -> None:
        buffer = HeadTailBuffer()
        buffer.push("".join(f"{i}\n" for i in range(1, 100_001)).encode())
        response = format_exec_response(
            output=buffer, wall_time_s=0.5, exit_code=0, session_id=None, max_output_tokens=1_000
        )
        fields = parse(response)
        assert fields["tokens"] == str((len(buffer.text()) + 3) // 4)
        assert fields["output"].startswith(
            f"Warning: truncated output (original token count: {fields['tokens']})\nTotal output lines: 100000\n\n1\n2\n"
        )
        assert fields["output"].endswith("99999\n100000\n")
        assert "tokens truncated…" in fields["output"]
        # The whole response fits the 20% serialization allowance over the token budget.
        assert len(response.encode()) <= 1_000 * 4 * 1.2

    def test_head_tail_buffer_caps_retained_bytes(self) -> None:
        buffer = HeadTailBuffer()
        buffer.push(b"a" * (1024 * 1024))
        buffer.push(b"b" * 1000)
        assert buffer.omitted == 1000
        assert buffer.text().startswith("a" * 10) and buffer.text().endswith("b" * 1000)
        assert "... 1000 bytes omitted ..." in buffer.text()


class TestExecCommand:
    async def test_completed_command(self, runtime: ExecRuntime) -> None:
        fields = parse(await runtime.exec_command({"cmd": "echo out; echo err >&2; exit 3"}))
        assert fields == {"exit": "3", "session": None, "tokens": "2", "output": "out\nerr\n"}
        assert runtime.processes == {}

    async def test_upstream_environment(self, runtime: ExecRuntime) -> None:
        output = parse(await runtime.exec_command({"cmd": "echo $CODEX_CI $TERM $PAGER $GIT_PAGER"}))["output"]
        assert output == "1 dumb cat cat\n"

    async def test_workdir_and_invalid_arguments(self, runtime: ExecRuntime, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        assert parse(await runtime.exec_command({"cmd": "pwd", "workdir": "sub"}))["output"] == f"{tmp_path}/sub\n"
        with pytest.raises(ExecError, match="missing field `cmd`"):
            await runtime.exec_command({})
        with pytest.raises(ExecError, match="is not a directory"):
            await runtime.exec_command({"cmd": "true", "workdir": "missing"})
        with pytest.raises(ExecError, match="expected a boolean"):
            await runtime.exec_command({"cmd": "true", "tty": "yes"})

    async def test_long_running_command_yields_session_then_finishes(self, runtime: ExecRuntime) -> None:
        started = time.monotonic()
        first = parse(await runtime.exec_command({"cmd": "echo start; sleep 1; echo done", "yield_time_ms": 300}))
        assert time.monotonic() - started < 0.9
        assert first["exit"] is None and first["output"] == "start\n"

        second = parse(await runtime.write_stdin({"session_id": int(first["session"]), "yield_time_ms": 10_000}))
        # The poll returns as soon as the process exits rather than waiting out the yield time.
        assert second == {"exit": "0", "session": None, "tokens": "2", "output": "done\n"}
        with pytest.raises(ExecError, match=f"Unknown process id {first['session']}"):
            await runtime.write_stdin({"session_id": int(first["session"])})

    async def test_tty_session_accepts_input(self, runtime: ExecRuntime) -> None:
        first = parse(await runtime.exec_command({"cmd": "read x; echo got:$x", "tty": True, "yield_time_ms": 300}))
        reply = parse(await runtime.write_stdin({"session_id": int(first["session"]), "chars": "hello\n"}))
        assert reply["exit"] == "0"
        assert reply["output"].replace("\r\n", "\n") == "hello\ngot:hello\n"

    async def test_pipe_session_rejects_input_but_accepts_interrupt(self, runtime: ExecRuntime) -> None:
        session = int(parse(await runtime.exec_command({"cmd": "sleep 30", "yield_time_ms": 250}))["session"])
        with pytest.raises(ExecError, match="^write_stdin failed: stdin is closed for this session"):
            await runtime.write_stdin({"session_id": session, "chars": "x"})
        reply = parse(await runtime.write_stdin({"session_id": session, "chars": "\x03", "yield_time_ms": 2000}))
        assert reply["exit"] == "130"

    async def test_close_kills_background_jobs(self, runtime: ExecRuntime, tmp_path: Path) -> None:
        reply = parse(await runtime.exec_command({"cmd": "sleep 60 & echo $!"}))
        assert reply["exit"] == "0"
        pid = int(reply["output"])
        os.kill(pid, 0)  # still running after its parent command exited
        runtime.close()
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.02)
        pytest.fail("background job survived ExecRuntime.close()")

    async def test_write_stdin_poll_yield_bounds(self, runtime: ExecRuntime) -> None:
        session = int(parse(await runtime.exec_command({"cmd": "sleep 30", "yield_time_ms": 250}))["session"])
        started = time.monotonic()
        # Empty polls wait at least 5 s even when asked for less.
        reply = parse(await runtime.write_stdin({"session_id": session, "yield_time_ms": 10}))
        assert 4.9 < time.monotonic() - started < 6 and reply["session"] == str(session)
