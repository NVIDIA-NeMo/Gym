# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from responses_api_agents.nooa_agent.sandbox_shell import (
    _CWD_MARKER,
    SandboxBashSession,
    SandboxShellTools,
    _extract_cwd_marker,
    _wrap_with_cwd_capture,
)


WRAPPER_TAIL = "\n}\n__nooa_rc=$?\n"


class FakeExecResult:
    def __init__(self, *, stdout: str = "", stderr: str = "", return_code: int = 0, error_type: str | None = None):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.error_type = error_type


class FakeSandbox:
    """Emulates the wrapper contract, not a shell.

    Scripted entries map an inner user command to
    ``(stdout, stderr_without_marker, return_code, final_dir_or_None)``; a fifth
    optional element overrides ``error_type``.
    """

    def __init__(self, *, workdir: str = "/image-root", scripted: dict[str, tuple] | None = None) -> None:
        self.workdir = workdir
        self.scripted = scripted or {}
        self.calls: list[tuple[str, str | None]] = []

    async def exec(self, command: str, *, cwd: str | None = None, timeout_s: float = 180) -> Any:
        self.calls.append((command, cwd))
        if command == "pwd":
            return FakeExecResult(stdout=self.workdir + "\n")
        if command == "command -v base64":
            return FakeExecResult(stdout="/usr/bin/base64\n")
        if command == "command -v rg":
            return FakeExecResult(return_code=1)
        assert command.startswith("{\n") and WRAPPER_TAIL in command, command
        inner = command[len("{\n") : command.index(WRAPPER_TAIL)]
        entry = self.scripted[inner]
        stdout, stderr, return_code, final_dir = entry[:4]
        error_type = entry[4] if len(entry) > 4 else None
        if final_dir is not None:
            stderr = f"{stderr}{_CWD_MARKER}{final_dir}\n"
        return FakeExecResult(stdout=stdout, stderr=stderr, return_code=return_code, error_type=error_type)


def test_wrapper_shape() -> None:
    wrapped = _wrap_with_cwd_capture("ls -la")
    assert wrapped.startswith("{\nls -la\n}\n__nooa_rc=$?\n")
    assert f"printf '{_CWD_MARKER}%s\\n' \"$PWD\" >&2\n" in wrapped
    assert wrapped.endswith("exit $__nooa_rc\n")


def test_extract_cwd_marker() -> None:
    reported, cleaned = _extract_cwd_marker("out\n__nooa_cwd__/opt\n")
    assert reported == "/opt"
    assert cleaned == "out\n"
    reported, cleaned = _extract_cwd_marker("nothing\n")
    assert reported is None
    assert cleaned == "nothing\n"


def test_extract_cwd_marker_requires_line_start_and_takes_last() -> None:
    stderr = "note __nooa_cwd__/not-a-marker\n__nooa_cwd__/real\n"
    reported, cleaned = _extract_cwd_marker(stderr)
    assert reported == "/real"
    assert cleaned == "note __nooa_cwd__/not-a-marker\n"


@pytest.mark.asyncio
async def test_cd_persists_across_calls() -> None:
    sandbox = FakeSandbox(scripted={"cd /opt": ("", "", 0, "/opt"), "pwd": ("/opt\n", "", 0, "/opt")})
    session = SandboxBashSession(sandbox, cwd="/repo")

    stdout, stderr, code, timed_out = await session.run_with_timeout_flag("cd /opt")

    assert (code, timed_out) == (0, False)
    assert stdout == ""
    assert stderr == ""  # marker stripped before the caller sees it
    assert session._cwd == "/opt"
    await session.run_with_timeout_flag("pwd")
    assert sandbox.calls[-1][1] == "/opt"  # next call resumes from the reported directory


@pytest.mark.asyncio
async def test_exit_code_and_stderr_are_preserved() -> None:
    sandbox = FakeSandbox(scripted={"printf boom >&2; false": ("", "boom\n", 3, "/kept")})
    session = SandboxBashSession(sandbox, cwd="/repo")

    stdout, stderr, code, _ = await session.run_with_timeout_flag("printf boom >&2; false")

    assert (stdout, stderr, code) == ("", "boom\n", 3)
    assert session._cwd == "/kept"


@pytest.mark.asyncio
async def test_process_replacing_command_leaves_cwd_unchanged() -> None:
    sandbox = FakeSandbox(scripted={"exit 0": ("", "", 0, None)})
    session = SandboxBashSession(sandbox, cwd="/repo")

    _, _, code, _ = await session.run_with_timeout_flag("exit 0")

    assert code == 0
    assert session._cwd == "/repo"


@pytest.mark.asyncio
async def test_timeout_never_adopts_partial_marker() -> None:
    sandbox = FakeSandbox(scripted={"sleep 999": ("", _CWD_MARKER + "/par", 0, None, "timeout")})
    session = SandboxBashSession(sandbox, cwd="/repo")

    _, stderr, code, timed_out = await session.run_with_timeout_flag("sleep 999")

    assert timed_out is True
    assert session._cwd == "/repo"
    assert _CWD_MARKER not in stderr


@pytest.mark.asyncio
async def test_start_trusts_absolute_cwd() -> None:
    sandbox = FakeSandbox()
    session = SandboxBashSession(sandbox, cwd="/repo")

    await session.start()

    assert session._cwd == "/repo"  # not reset to the image workdir


@pytest.mark.asyncio
async def test_start_adopts_relative_cwd() -> None:
    sandbox = FakeSandbox(workdir="/image-root")
    session = SandboxBashSession(sandbox, cwd=".")

    await session.start()

    assert session._cwd == "/image-root"


@pytest.mark.asyncio
async def test_init_command_cwd_is_adopted() -> None:
    sandbox = FakeSandbox(scripted={"cd /opt/init": ("", "", 0, "/opt/init")})
    session = SandboxBashSession(sandbox, cwd="/repo", init_command="cd /opt/init")

    await session.start()

    assert session._cwd == "/opt/init"


@pytest.mark.asyncio
async def test_shell_relative_paths_follow_session_cwd() -> None:
    shell = SandboxShellTools(FakeSandbox(scripted={"cd /opt": ("", "", 0, "/opt")}), cwd="/repo")
    assert shell._resolve_path("file.py") == "/repo/file.py"

    await shell._session.run("cd /opt")

    assert shell._resolve_path("file.py") == "/opt/file.py"
