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

"""Docker PTY sessions.

The session plumbing (host pty, pipes, resize, exit, EOF, the marker protocol) runs against a local ``sh``
spawned the way ``docker exec`` would be, so it is exercised for real without a daemon; the provider's
``docker exec`` command line, signal and cleanup execs are checked against a recorded fake CLI.
"""

import asyncio
import contextlib
import hashlib
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.sandbox import AsyncSandbox, SandboxHandle
from nemo_gym.sandbox.providers.base import (
    SandboxPtyError,
    SandboxPtySession,
    SandboxPtySpec,
    SupportsSandboxPty,
    SupportsSandboxPtyAttach,
)
from nemo_gym.sandbox.providers.docker import provider as docker_provider
from nemo_gym.sandbox.providers.docker import pty as docker_pty


pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(sys.platform != "linux" or shutil.which("sh") is None, reason="needs a Linux host with sh"),
]

SH = shutil.which("sh") or "/bin/sh"
ENV = shutil.which("env") or "/usr/bin/env"
BUSYBOX = shutil.which("busybox")
# The shells that run the provider's in-container lines (pid read, signal, cleanup): this host's sh, and busybox's
# when it is installed, which is what ``sh`` is in Alpine-based task images.
KILL_SHELLS = [
    pytest.param([SH], id="sh"),
    pytest.param(
        [BUSYBOX or "busybox", "sh"], id="busybox", marks=pytest.mark.skipif(BUSYBOX is None, reason="no busybox")
    ),
]
SESSION_ID = "sess01"
# A ``PID_READ_COMMAND`` record as the shell prints it: pid and the matched marker line, then ``/proc/<pid>/stat``
# (parent 0: injected by the runtime; session 4242: its own leader).
PID_RECORD = f"4242 {docker_pty.session_marker(SESSION_ID)}\n4242 (ba sh) S 0 4242 4242 34816 4300 4194560 0 0 0 0\n"


def pid_record_for(command: str) -> str:
    """``PID_RECORD`` for the session whose marker the rendered ``PID_READ_COMMAND`` carries."""
    session_id = re.search(rf"{docker_pty.SESSION_ENV}=([0-9a-f]+)", command).group(1)
    return PID_RECORD.replace(SESSION_ID, session_id)


def pid_read_command(session) -> str:
    return docker_pty.PID_READ_COMMAND.format(
        pid_file=docker_pty.pid_file_for(session.session_id),
        fallback=docker_pty.fallback_pid_file_for(session.session_id),
        tries=docker_pty.PID_READ_TRIES,
        marker=docker_pty.session_marker(session.session_id),
        session_id=session.session_id,
    )


def pid_files(session) -> str:
    """The session's pid file and its ``/dev/shm`` fallback as the cleanup lines name them."""
    return f"{docker_pty.pid_file_for(session.session_id)} {docker_pty.fallback_pid_file_for(session.session_id)}"


def rm_command(session) -> str:
    """The one exec ``close()`` runs for a session that exited or whose pid it could not trust."""
    return f"rm -f -- {pid_files(session)}"


def fallback_pid_path(pid_dir) -> Path:
    """The local stand-in for the ``/dev/shm`` pid file of a session whose ``/tmp`` one is ``pid_dir / "pid"``."""
    (Path(pid_dir) / "shm").mkdir(parents=True, exist_ok=True)
    return Path(pid_dir) / "shm" / "pid"


async def read_until(session, needle: bytes, *, timeout_s: float = 10.0) -> bytes:
    buffer = bytearray()
    async with asyncio.timeout(timeout_s):
        while needle not in buffer:
            chunk = await session.read()
            if not chunk:
                break
            buffer.extend(chunk)
    return bytes(buffer)


# --------------------------------------------------------------------------- #
# The session against a real local process
# --------------------------------------------------------------------------- #
async def test_pty_session_runs_an_interactive_shell_under_a_sized_terminal() -> None:
    session = await docker_pty.spawn(in_container_terminal([SH]), use_pty=True, rows=31, cols=121)
    async with session:
        assert isinstance(session, SandboxPtySession)  # the runtime-checkable protocol the facade relies on
        assert session.mode == "pty" and not session.closed
        # The sentinel is split in the command so the terminal's echo of the line cannot match it.
        await session.write(b"stty size; echo __do''ne__\n")
        out = await read_until(session, b"__done__\r\n")
        # The terminal was sized before the shell started, and PTY mode delivers output with CRLF.
        assert b"31 121\r\n" in out
        await session.resize(40, 100)
        await session.write(b"stty size; echo __ag''ain__\n")
        out = await read_until(session, b"__again__\r\n")
        assert b"40 100\r\n" in out
        await session.write(b"exit 7\n")
        assert await session.wait_exit(timeout_s=10) == 7
        # The stream drains to EOF after the exit, and stays at EOF.
        async with asyncio.timeout(10):
            while await session.read():
                pass
        assert await session.read() == b"" and await session.read_stderr() == b""
        assert session.closed


# A docker client that is slow to come up: half a second later than ``create_pty`` returned it puts its terminal in
# raw mode as the CLI does (``TCSANOW``, the CLI's ``tcset``: nothing pending is flushed) and reads what has been
# written to it meanwhile, reporting the byte count and digest of one line.
LATE_RAW_CLIENT = (
    "import hashlib, os, sys, termios, time, tty\n"
    "time.sleep(0.5)\n"
    "tty.setraw(0, termios.TCSANOW)\n"
    "data = b''\n"
    "while b'\\n' not in data:\n"
    "    data += os.read(0, 65536)\n"
    "sys.stdout.write(f'GOT {len(data)} {hashlib.sha256(data).hexdigest()}\\n')\n"
    "sys.stdout.flush()\n"
)


async def test_an_immediate_write_reaches_a_late_client_byte_for_byte() -> None:
    """``create_pty`` returns before the docker client has set its terminal up, and a write may follow at once. The
    host pty is raw from the start, so those bytes wait in it for the client instead of passing through a cooked line
    discipline, which would echo them back as session output, act on control characters and drop everything past
    its 4095-byte line: a client that comes up half a second late and reads its terminal raw, as the CLI does,
    receives all 10005 bytes of an immediate write as written, control characters included, and the session reads
    none of them back."""
    session = await docker_pty.spawn([sys.executable, "-c", LATE_RAW_CLIENT], use_pty=True, rows=24, cols=80)
    async with session:
        # ``^U`` would erase the line and ``^C`` raise the host's SIGINT under a cooked discipline.
        payload = b"\x15" + b"x" * 10002 + b"\x03\n"
        assert len(payload) == 10005
        await session.write(payload)
        out = await read_until(session, b"\n")
        digest = hashlib.sha256(payload).hexdigest().encode()
        assert out == b"GOT 10005 " + digest + b"\n", out[:120]
        assert await session.wait_exit(timeout_s=10) == 0


async def test_pipe_mode_splits_stderr_and_ignores_resize() -> None:
    session = await docker_pty.spawn(
        [SH, "-c", "echo out; echo err >&2; read line; echo got:$line; exit 3"], use_pty=False, rows=24, cols=80
    )
    async with session:
        assert session.mode == "pipe"
        await session.resize(10, 10)  # no terminal: accepted and ignored
        assert await read_until(session, b"out\n") == b"out\n"
        assert await asyncio.wait_for(session.read_stderr(), timeout=10) == b"err\n"
        await session.write(b"hello\n")
        assert b"got:hello\n" in await read_until(session, b"got:hello\n")
        assert await session.wait_exit(timeout_s=10) == 3
        assert [chunk async for chunk in session] == []


# The docker client's part in a resize, stood in for by a program that does what the Go CLI does and no more: it
# never makes the host pty its controlling terminal and it re-reads the terminal's size on ``SIGWINCH``, reporting
# each signal with the size it read. A shell would not do: bash under ``-c`` takes the terminal (``tpgid`` its own
# group) where dash does not, so with bash as ``sh`` the kernel's own window-change signal reaches the program too.
WINCH_LISTENER = (
    "import os, signal, sys, time\n"
    "n = 0\n"
    "def report(*_):\n"
    "    global n\n"
    "    n += 1\n"
    "    size = os.get_terminal_size(1)\n"
    "    sys.stdout.write(f'WINCH{n} {size.lines}x{size.columns}\\n')\n"
    "    sys.stdout.flush()\n"
    "signal.signal(signal.SIGWINCH, report)\n"
    "sys.stdout.write('ready\\n')\n"
    "sys.stdout.flush()\n"
    "while True:\n"
    "    time.sleep(0.05)\n"
)


async def test_resize_signals_the_client_with_sigwinch() -> None:
    """The docker CLI re-reads its terminal's size only on ``SIGWINCH``, and the kernel sends none to a process that
    never made the host pty its controlling terminal (``spawn`` gives the client none and the Go CLI takes none), so
    the one that matters is the signal ``resize()`` sends itself: a listener that behaves like the client in this
    (``WINCH_LISTENER``, independent of which shell ``sh`` is) sees exactly one per resize, and reads the new size at
    each."""
    session = await docker_pty.spawn([sys.executable, "-c", WINCH_LISTENER], use_pty=True, rows=24, cols=80)
    async with session:
        await read_until(session, b"ready")
        # No controlling terminal (``tpgid`` -1): a window change on the pty alone would never reach the program.
        assert terminal_groups(str(session._proc.pid))[1] == "-1"
        await session.resize(40, 100)
        out = await read_until(session, b"WINCH1 40x100")
        await session.resize(50, 120)
        out += await read_until(session, b"WINCH2 50x120")
        assert out.count(b"WINCH") == 2, out


async def test_close_before_exit_kills_the_client_and_fails_waiters() -> None:
    session = await docker_pty.spawn([SH, "-c", "sleep 30"], use_pty=True, rows=24, cols=80)
    waiter = asyncio.create_task(session.wait_exit())
    await asyncio.sleep(0.05)
    await session.close()
    await session.close()  # idempotent
    with pytest.raises(SandboxPtyError, match="closed before process exit"):
        await waiter
    with pytest.raises(SandboxPtyError, match="closed"):
        await session.write(b"x")
    with pytest.raises(SandboxPtyError, match="closed"):
        await session.read()
    assert session.returncode is not None  # the local client is gone


async def test_a_write_waiting_for_room_is_failed_before_the_terminal_is_released() -> None:
    """A write blocked on a full pty (the client not reading) when the session is abandoned or closed must not
    outlive the terminal: the master's descriptor number is free for the next session's pty at once, and a writer
    resuming on it would push the old session's input into the new one, or its readiness callback, left registered
    on the number, would break the new session's reads and never let the write return. The write fails with
    ``SandboxPtyError`` before the fd closes, and a session opened right after, on the reused number, runs normally
    and sees nothing of the old input."""
    # The client puts its terminal raw and never reads it; a background child keeps the slave open past the
    # client's death, so the write stays blocked across the release instead of failing on the hangup.
    session = await docker_pty.spawn(
        [SH, "-c", "stty raw -echo; sleep 300 & echo bg=$!; exec sleep 300"], use_pty=True, rows=24, cols=80
    )
    holder = int(re.search(rb"bg=(\d+)", await read_until(session, b"\n")).group(1))
    try:
        old_fd = session._master_fd
        writer = asyncio.create_task(session.write(b"echo LEAKED_INPUT\n" * 20000))
        await asyncio.sleep(0.2)
        assert not writer.done(), "the write should be waiting for room in the pty"
        await session.abandon()
        assert session._master_fd is None
        with pytest.raises(SandboxPtyError, match="closed before the write completed"):
            await asyncio.wait_for(writer, timeout=5)
        fresh = await docker_pty.spawn(in_container_terminal([SH]), use_pty=True, rows=24, cols=80)
        async with fresh:
            assert fresh._master_fd == old_fd, "the new session should get the released descriptor number"
            await fresh.write(b"echo fresh''_ok\n")
            out = await read_until(fresh, b"fresh_ok\r\n")
            assert b"fresh_ok\r\n" in out and b"LEAKED_INPUT" not in out, out
            await fresh.write(b"exit 0\n")
            assert await fresh.wait_exit(timeout_s=10) == 0
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(holder, signal.SIGKILL)


# A docker client that reads nothing until told to (``SIGUSR1``): its terminal raw as the CLI's, it then drains its
# input to the first newline and reports the byte count and digest of everything it read.
HELD_RAW_CLIENT = (
    "import hashlib, os, signal, sys, termios, time, tty\n"
    "go = False\n"
    "def release(*_):\n"
    "    global go\n"
    "    go = True\n"
    "signal.signal(signal.SIGUSR1, release)\n"
    "tty.setraw(0, termios.TCSANOW)\n"
    "sys.stdout.write('ready\\n')\n"
    "sys.stdout.flush()\n"
    "while not go:\n"
    "    time.sleep(0.02)\n"
    "data = b''\n"
    "while b'\\n' not in data:\n"
    "    data += os.read(0, 1 << 16)\n"
    "sys.stdout.write(f'GOT {len(data)} {hashlib.sha256(data).hexdigest()}\\n')\n"
    "sys.stdout.flush()\n"
)


async def test_two_writes_waiting_for_room_both_complete_once_the_client_reads() -> None:
    """Two concurrent writes that both run out of room in the pty while the client is not reading. The loop keeps
    one readiness callback per descriptor, so unless the session serializes its writes the second writer's
    registration replaces the first's and the first never wakes once the client drains: it stays blocked until the
    session closes. Both must complete within a bounded time once the client reads, one after the other, and the
    client receives exactly A then B."""
    session = await docker_pty.spawn([sys.executable, "-c", HELD_RAW_CLIENT], use_pty=True, rows=24, cols=80)
    async with session:
        await read_until(session, b"ready")
        a = b"a" * (3 << 20)
        b = b"b" * (1 << 20) + b"\n"
        writer_a = asyncio.create_task(session.write(a))
        await asyncio.sleep(0.2)
        assert not writer_a.done(), "write A should be waiting for room in the pty"
        writer_b = asyncio.create_task(session.write(b))
        await asyncio.sleep(0.2)
        assert not writer_b.done(), "write B should be waiting behind A"
        os.kill(session._proc.pid, signal.SIGUSR1)
        await asyncio.wait_for(asyncio.gather(writer_a, writer_b), timeout=60)
        out = await read_until(session, b"\n", timeout_s=20)
        digest = hashlib.sha256(a + b).hexdigest().encode()
        assert out == b"GOT " + str(len(a) + len(b)).encode() + b" " + digest + b"\n", out[:120]
        assert await session.wait_exit(timeout_s=10) == 0


async def test_two_writes_waiting_for_room_are_both_failed_before_the_terminal_is_released() -> None:
    """Abandoned with two writes pending, one waiting for room in the pty and one queued behind it: both fail with
    ``SandboxPtyError`` before the fd closes, and a session opened right after, on the reused descriptor number,
    sees nothing of either."""
    session = await docker_pty.spawn(
        [SH, "-c", "stty raw -echo; sleep 300 & echo bg=$!; exec sleep 300"], use_pty=True, rows=24, cols=80
    )
    holder = int(re.search(rb"bg=(\d+)", await read_until(session, b"\n")).group(1))
    try:
        old_fd = session._master_fd
        writer_a = asyncio.create_task(session.write(b"echo LEAKED_A\n" * 20000))
        await asyncio.sleep(0.2)
        assert not writer_a.done(), "write A should be waiting for room in the pty"
        writer_b = asyncio.create_task(session.write(b"echo LEAKED_B\n" * 20000))
        await asyncio.sleep(0.2)
        assert not writer_b.done(), "write B should be waiting behind A"
        await session.abandon()
        assert session._master_fd is None
        with pytest.raises(SandboxPtyError, match="closed"):
            await asyncio.wait_for(writer_a, timeout=5)
        with pytest.raises(SandboxPtyError, match="closed"):
            await asyncio.wait_for(writer_b, timeout=5)
        fresh = await docker_pty.spawn(in_container_terminal([SH]), use_pty=True, rows=24, cols=80)
        async with fresh:
            assert fresh._master_fd == old_fd, "the new session should get the released descriptor number"
            await fresh.write(b"echo fresh''_ok\n")
            out = await read_until(fresh, b"fresh_ok\r\n")
            assert b"fresh_ok\r\n" in out and b"LEAKED" not in out, out
            await fresh.write(b"exit 0\n")
            assert await fresh.wait_exit(timeout_s=10) == 0
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(holder, signal.SIGKILL)


async def test_run_detached_runs_attached_with_the_marker_protocol() -> None:
    async with await docker_pty.spawn(in_container_terminal([SH]), use_pty=True, rows=24, cols=80) as session:
        output, code = await session.run_detached("printf 'a\\nb'; sh -c 'exit 4'", poll_interval_s=0.1)
        assert output == b"a\r\nb" and code == 4
        # The session survives the command and keeps its shell state.
        output, code = await session.run_detached("X=1; echo $X")
        assert output == b"1\r\n" and code == 0
    async with await docker_pty.spawn([SH], use_pty=False, rows=24, cols=80) as session:
        output, code = await session.run_detached("echo out; echo err >&2; false")
        assert code == 1 and output == b"out\nerr\n"


async def test_run_detached_raises_when_the_shell_exits_first() -> None:
    async with await docker_pty.spawn([SH], use_pty=False, rows=24, cols=80) as session:
        with pytest.raises(SandboxPtyError, match="ended before the command finished"):
            await session.run_detached("exit 0")


# What runc does for `docker exec -t` before the program starts, and what the docker client never does: the process
# is a session leader (``spawn`` starts it in a new session) and takes the terminal as its controlling tty, so a
# shell in it has job control; and the terminal it gets is a fresh console in the cooked discipline (Linux's
# ``tty_std_termios``: canonical input, echo, CRLF output, job-control characters), where ``spawn`` has put the
# host pty raw as the docker CLI would, and its stdout and stderr are both that console (the client's own stderr is
# a pipe of the session's, see ``spawn``). Without a terminal (pipe mode) there is nothing to take or restore, and
# the streams stay split.
CONTAINER_TERMINAL = (
    "import contextlib, fcntl, os, sys, termios\n"
    "if os.isatty(0):\n"
    "    with contextlib.suppress(OSError):\n"
    "        fcntl.ioctl(0, termios.TIOCSCTTY, 0)\n"
    "    mode = termios.tcgetattr(0)\n"
    "    mode[0] |= termios.ICRNL | termios.IXON\n"
    "    mode[1] |= termios.OPOST | termios.ONLCR\n"
    "    mode[3] |= (termios.ISIG | termios.ICANON | termios.ECHO | termios.ECHOE | termios.ECHOK\n"
    "                | termios.ECHOCTL | termios.ECHOKE | termios.IEXTEN)\n"
    "    termios.tcsetattr(0, termios.TCSANOW, mode)\n"
    "    os.dup2(1, 2)\n"
    "os.execvp(sys.argv[1], sys.argv[1:])\n"
)
# The shell line that does the same for a program that cannot be wrapped in this interpreter (inside the stand-in
# container): the cooked discipline back, then the program.
COOKED_TERMINAL_SH = 'if [ -t 0 ]; then stty sane 2>/dev/null; exec 2>&1; fi; exec "$@"'


def in_container_terminal(argv: list[str]) -> list[str]:
    """``argv`` run locally as the program of a ``docker exec -t`` runs in the container (``CONTAINER_TERMINAL``)."""
    return [sys.executable, "-c", CONTAINER_TERMINAL, *argv]


def local_program(argv: list[str], container: str, pid_dir) -> list[str]:
    """Everything up to the container name is docker's; the launcher and its program run locally instead, under the
    terminal as their controlling tty like an exec in the container, with the exec's ``--env`` pairs (the session
    marker among them) in their environment and the pid file under ``pid_dir`` rather than the host's /tmp."""
    docker_part, program = argv[: argv.index(container)], argv[argv.index(container) + 1 :]
    env_pairs = [docker_part[i + 1] for i, token in enumerate(docker_part) if token == "--env"]
    assert program[:3] == [program[0], "-c", docker_pty.LAUNCHER]
    assert program[3:5] == [
        docker_pty.pid_file_for(program[3].removeprefix(docker_pty.PID_FILE_PREFIX).removesuffix(".pid")),
        docker_pty.fallback_pid_file_for(program[3].removeprefix(docker_pty.PID_FILE_PREFIX).removesuffix(".pid")),
    ]
    return [
        ENV,
        *env_pairs,
        *in_container_terminal(
            [SH, *program[1:3], str(pid_dir / "pid"), str(fallback_pid_path(pid_dir)), *program[5:]]
        ),
    ]


async def test_facade_pty_exec_over_a_real_terminal(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """``sandbox.pty.exec`` through the Docker provider: the facade's marker protocol over the real plumbing."""
    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")
    provider = docker_provider.DockerProvider(exec={"exec_shell": "sh"})
    argvs: list[list[str]] = []

    async def spawn_local(argv, **kwargs):
        argvs.append(argv)
        return await docker_pty.spawn(local_program(argv, "nemo-gym-x", tmp_path), **kwargs)

    monkeypatch.setattr(provider, "_spawn_pty", spawn_local)
    inst = docker_provider._DockerContainer(name="nemo-gym-x", image="img", shell="sh")
    handle = SandboxHandle(sandbox_id="nemo-gym-x", provider_name="docker", raw=inst)
    sandbox = AsyncSandbox(provider)
    sandbox._handle, sandbox._stopped = handle, False
    # One-shot: a private session runs `sh -c command` to completion in pipe mode with split streams.
    result = await sandbox.pty.exec("echo out; echo err >&2; exit 5", pty=False, timeout_s=10)
    assert (result.stdout, result.stderr, result.return_code) == ("out\n", "err\n", 5)
    # Session mode: the default shell session stays open across commands.
    session = await sandbox.pty.create()
    try:
        first = await sandbox.pty.exec("Y=42; echo $Y", timeout_s=10)
        second = await sandbox.pty.exec("echo $Y", timeout_s=10)
        assert (first.stdout, first.return_code) == ("42\r\n", 0)
        assert (second.stdout, second.return_code) == ("42\r\n", 0)
    finally:
        await session.close()
    assert all(argv[:2] == ["/usr/bin/docker", "exec"] for argv in argvs)
    assert not any(
        session for sessions in provider._pty_sessions.values() for session in sessions if not session.closed
    )


# --------------------------------------------------------------------------- #
# The provider: the docker command line, signals, cleanup
# --------------------------------------------------------------------------- #
class FakeSession:
    def __init__(self, argv: list[str], kwargs: dict[str, Any]) -> None:
        self.argv, self.kwargs = argv, kwargs
        self.session_id = kwargs["session_id"]
        self.closed = False
        self.abandoned = False
        self.run_in_container = kwargs["run_in_container"]
        self.pid_file = kwargs["pid_file"]

    async def close(self) -> None:
        self.closed = True

    async def abandon(self) -> None:
        self.closed = self.abandoned = True


def _provider(monkeypatch: pytest.MonkeyPatch, responder=None, **kwargs):
    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")
    kwargs.setdefault("exec", {"exec_shell": "bash"})
    provider = docker_provider.DockerProvider(**kwargs)
    runs: list[list[str]] = []

    async def run(argv, *, timeout_s, stdin=None):
        runs.append(list(argv))
        return responder(argv) if responder else (0, "", "")

    monkeypatch.setattr(provider, "_run", run)
    spawned: list[FakeSession] = []

    async def spawn(argv, **kwargs):
        session = FakeSession(list(argv), kwargs)
        spawned.append(session)
        return session

    monkeypatch.setattr(provider, "_spawn_pty", spawn)
    return provider, runs, spawned


def _handle(env=None, shell: str = "bash") -> SandboxHandle:
    inst = docker_provider._DockerContainer(name="nemo-gym-abc", image="img", shell=shell, env=env or {})
    return SandboxHandle(sandbox_id="nemo-gym-abc", provider_name="docker", raw=inst)


def exec_user(argv: list[str]) -> str | None:
    """The ``--user`` of a recorded ``docker exec``; ``None`` when it runs as the container's default user."""
    return argv[argv.index("--user") + 1] if "--user" in argv else None


def test_docker_provider_offers_pty_but_not_attach(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _, _ = _provider(monkeypatch)
    assert isinstance(provider, SupportsSandboxPty)
    # Sessions live in this process: there is nothing to re-attach to by id.
    assert not isinstance(provider, SupportsSandboxPtyAttach)


async def test_create_pty_builds_the_exec_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _, spawned = _provider(monkeypatch, exec={"exec_shell": "bash", "extra_exec_args": ["--privileged"]})
    handle = _handle(env={"BASE": "1", "SHARED": "spec"})
    session = await provider.create_pty(
        handle,
        SandboxPtySpec(command="htop", cwd="/work", env={"SHARED": "pty", "X": "2"}, rows=50, cols=200, user="root"),
    )
    (fake,) = spawned
    assert session is fake
    pid_file = docker_pty.pid_file_for(fake.session_id)
    fallback = docker_pty.fallback_pid_file_for(fake.session_id)
    assert pid_file == f"/tmp/.nemo-gym-pty-{fake.session_id}.pid"
    assert fallback == f"/dev/shm/.nemo-gym-pty-{fake.session_id}.pid"
    assert fake.argv == [
        "/usr/bin/docker",
        "exec",
        "-i",
        "-t",
        "-w",
        "/work",
        "--env",
        "BASE=1",
        "--env",
        "SHARED=pty",
        "--env",
        "X=2",
        "--env",
        f"NEMO_GYM_PTY_SESSION={fake.session_id}",
        "--user",
        "0",
        "--privileged",
        "nemo-gym-abc",
        "bash",
        "-c",
        docker_pty.LAUNCHER,
        pid_file,
        fallback,
        "bash",
        "-c",
        "htop",
    ]
    assert (fake.kwargs["use_pty"], fake.kwargs["rows"], fake.kwargs["cols"]) == (True, 50, 200)
    assert provider._pty_sessions == {"nemo-gym-abc": {fake}}
    # The marker is the session's identity for the root-side kill: a spec env cannot set or override it.
    forged = await provider.create_pty(handle, SandboxPtySpec(env={"NEMO_GYM_PTY_SESSION": "forged"}))
    markers = [v for k, v in zip(forged.argv, forged.argv[1:]) if k == "--env" and v.startswith("NEMO_GYM_PTY_")]
    assert markers == [f"NEMO_GYM_PTY_SESSION={forged.session_id}"]
    spawned.remove(forged)

    # No command: the session's program is the image's interactive shell (bash if present, else sh); pipe mode
    # drops -t; a named user and a uid pass through as docker takes them.
    await provider.create_pty(handle, SandboxPtySpec(pty=False, user="agent"))
    await provider.create_pty(handle, SandboxPtySpec(user=1000))
    pipe, uid = spawned[1:]
    assert "-t" not in pipe.argv and pipe.kwargs["use_pty"] is False
    assert pipe.argv[pipe.argv.index("--user") + 1] == "agent"
    assert pipe.argv[-3:] == ["sh", "-c", docker_pty.DEFAULT_SHELL_SNIPPET]
    assert uid.argv[uid.argv.index("--user") + 1] == "1000"
    with pytest.raises(ValueError, match="not a bool"):
        await provider.create_pty(handle, SandboxPtySpec(user=True))


@pytest.mark.parametrize("kill_shell", KILL_SHELLS)
async def test_launcher_records_the_pid_and_execs_the_program(tmp_path, kill_shell: list[str]) -> None:
    """The launcher line the provider hands docker, run by a local sh and by busybox's: the pid file holds the
    program's pid; a pid file whose directory refuses the write (``/tmp`` on a read-only root filesystem) sends the
    record to the fallback (``/dev/shm``); and when neither can be written the program is not started: the launcher
    says so on stderr and exits 1, so a session that could be neither signalled nor cleaned up never runs."""
    if os.geteuid() == 0:
        pytest.skip("root writes into a 0555 directory")
    pid_dir = tmp_path / "ro"
    fallback = fallback_pid_path(pid_dir)
    pid_file = pid_dir / "pid"

    async def launch(command: str) -> tuple[bytes, bytes, int]:
        argv = [*kill_shell[:-1], *docker_pty.launcher_argv(kill_shell[-1], str(pid_file), str(fallback), command)]
        async with await docker_pty.spawn(argv, use_pty=False, rows=24, cols=80) as session:
            code = await session.wait_exit(timeout_s=10)
            out, err = bytearray(), bytearray()
            while chunk := await session.read(timeout_s=5):
                out.extend(chunk)
            while chunk := await session.read_stderr(timeout_s=5):
                err.extend(chunk)
        return bytes(out), bytes(err), code

    try:
        out, err, code = await launch('echo "pid=$$"; exit 9')
        recorded = pid_file.read_text().strip()
        # `exec` keeps the launcher's pid for the program, so `$$` inside it is the recorded one.
        assert (out.strip(), err, code) == (f"pid={recorded}".encode(), b"", 9) and recorded.isdigit()
        assert not fallback.exists()
        pid_file.unlink()

        pid_dir.chmod(0o555)
        out, err, code = await launch('echo "pid=$$"; exit 9')
        recorded = fallback.read_text().strip()
        assert (out.strip(), err, code) == (f"pid={recorded}".encode(), b"", 9) and recorded.isdigit()
        assert not pid_file.exists()
        fallback.unlink()

        fallback.parent.chmod(0o555)
        out, err, code = await launch('echo "pid=$$"; exit 9')
        assert (out, code) == (b"", 1)
        assert err.decode().strip() == (
            f"nemo-gym: cannot record the pid of this PTY session in {pid_file} or {fallback}"
        )
        assert not pid_file.exists() and not fallback.exists()
    finally:
        fallback.parent.chmod(0o755)
        pid_dir.chmod(0o755)


async def test_send_signal_and_close_go_through_docker_exec(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def responder(argv):
        command = argv[-1]
        if command.startswith("f="):
            return (0, pid_record_for(command), "")
        return (0, "", "")

    provider, runs, _ = _provider(monkeypatch, responder)
    handle = _handle()

    async def spawn_local(argv, **kwargs):
        # The real session plumbing with a local shell, keeping the provider's in-container hooks.
        return await docker_pty.spawn(local_program(argv, "nemo-gym-abc", tmp_path), **kwargs)

    monkeypatch.setattr(provider, "_spawn_pty", spawn_local)
    session = await provider.create_pty(handle, SandboxPtySpec())
    pid_file = docker_pty.pid_file_for(session.session_id)
    with pytest.raises(ValueError, match="supports only"):
        await session.send_signal("SIGUSR1")
    await session.send_signal("SIGINT")
    await session.send_signal("SIGTERM")
    # The pid is read and checked once through the provider, as root; each signal is one exec that kills the
    # terminal's foreground group when the recorded process has a terminal, else the process's group and the
    # process, with the ``kill -NAME -pgid`` spelling dash, bash and busybox all accept.
    read, interrupt, terminate = [argv[-1] for argv in runs]
    # The pid read runs as the session's user (no --user here: the container's default user, as the session itself
    # was started), the kills as root: /proc/<pid>/environ is readable by the process's own uid and not by a root
    # without CAP_SYS_PTRACE, which Docker does not grant by default.
    assert [exec_user(argv) for argv in runs] == [None, "0", "0"]
    assert read == pid_read_command(session)
    # The read waits up to two seconds (whole ones under an integer-only sleep) for the launcher's record in the
    # pid file or its /dev/shm fallback: a signal or close right after create_pty may reach the container first.
    # Either is read only while it is a regular file, through PID_RECORD_READ (the open pinned and tested again on
    # its descriptor, 32 bytes at most), and a path that exists as anything else is refused at once.
    fallback = docker_pty.fallback_pid_file_for(session.session_id)
    assert read.startswith(
        f'f={pid_file}; g={fallback}; n=0; rd() {{ [ -f "$1" ] && {docker_pty.PID_RECORD_READ}; }} 2>/dev/null; '
        'until pid=$(rd "$f") && [ -n "$pid" ]; do if pid=$(rd "$g") && [ -n "$pid" ]; then break; fi; '
        'for q in "$f" "$g"; do if [ -e "$q" ] && ! [ -f "$q" ]; then '
        "printf 'pid record %s is not a regular file\\n' \"$q\" >&2; exit 1; fi; done; "
        '[ "$n" -lt 20 ] || break; sleep 0.1 2>/dev/null || { sleep 1; n=$((n+9)); }; n=$((n+1)); done; '
        "case \"$pid\" in ''|0|1|*[!0-9]*) "
    )
    assert 'cat "$f"' not in read and 'cat "$g"' not in read  # nothing reads the record unbounded
    assert f'grep -m1 -x -F NEMO_GYM_PTY_SESSION={session.session_id}); [ -n "$m" ] || ' in read
    assert "ls " not in read  # the file's owner is no identity
    for command, name in ((interrupt, "INT"), (terminate, "TERM")):
        assert command.startswith("s=$(cat /proc/4242/stat 2>/dev/null); ")
        assert f'then kill -{name} -"$tp"; ' in command
        assert command.endswith(f"else kill -{name} -4242 2>/dev/null; kill -{name} 4242; fi")
        assert "kill -s " not in command and " -- -" not in command
    assert all(argv[:2] == ["/usr/bin/docker", "exec"] for argv in runs)
    # Closing a live session ends its process in the container and drops the pid file in one exec, by the pid
    # already checked (the file is not read again: it could have been rewritten since).
    await session.close()
    assert len(runs) == 4
    cleanup = runs[-1][-1]
    assert cleanup == docker_pty.CLEANUP_COMMAND.format(pid_files=pid_files(session), pid="4242")
    assert exec_user(runs[-1]) == "0"
    assert cleanup.startswith(f'rm -f -- {pid_file} {fallback}; pid=4242; kill -HUP -"$pid" "$pid" 2>/dev/null; ')
    assert '[ "$4" = "$pid" ] && kill -KILL "${p#/proc/}"' in cleanup and cleanup.endswith("; true")
    assert "cat " + pid_file not in cleanup and "kill -s " not in cleanup and " -- -" not in cleanup

    # A live session that was never signalled reads and checks the pid at close, then cleans up: two execs.
    runs.clear()
    session = await provider.create_pty(handle, SandboxPtySpec())
    await session.close()
    assert [argv[-1] for argv in runs] == [
        pid_read_command(session),
        docker_pty.CLEANUP_COMMAND.format(pid_files=pid_files(session), pid="4242"),
    ]

    # A session that already exited only loses its pid files.
    runs.clear()
    session = await provider.create_pty(handle, SandboxPtySpec(command="exit 0"))
    assert await session.wait_exit(timeout_s=10) == 0
    await session.close()
    assert [argv[-1] for argv in runs] == [rm_command(session)]

    # A session with a user of its own (what Harbor's default user gives a session on a non-root image):
    # the pid read runs as that user, by name or uid as the session was started, the kill and the cleanup as root.
    for user, flag in (("agent", "agent"), (1000, "1000")):
        runs.clear()
        session = await provider.create_pty(handle, SandboxPtySpec(user=user))
        await session.send_signal("SIGINT")
        await session.close()
        assert [exec_user(argv) for argv in runs] == [flag, "0", "0"], runs
        assert runs[0][-1] == pid_read_command(session) and "kill" in runs[1][-1] and "kill" in runs[2][-1]


async def test_signal_failure_surfaces_unless_the_process_exited(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def responder(argv):
        command = argv[-1]
        if command.startswith("f="):
            return (1, "", "no usable pid in /tmp/x: ")
        return (0, "", "")

    provider, runs, _ = _provider(monkeypatch, responder)

    async def spawn_local(argv, **kwargs):
        return await docker_pty.spawn(local_program(argv, "nemo-gym-abc", tmp_path), **kwargs)

    monkeypatch.setattr(provider, "_spawn_pty", spawn_local)
    session = await provider.create_pty(_handle(), SandboxPtySpec(command="sleep 30"))
    try:
        with pytest.raises(SandboxPtyError, match="no usable recorded pid.*no usable pid in /tmp/x"):
            await session.send_signal("SIGTERM")
    finally:
        await session.close()
    # Without a trusted pid, close() only drops the files: nothing is killed on an unverified target.
    assert [argv[-1] for argv in runs][-1] == rm_command(session)
    assert not any("kill" in argv[-1] for argv in runs)
    with pytest.raises(SandboxPtyError, match="closed"):
        await session.send_signal("SIGTERM")


async def test_a_failed_kill_surfaces_unless_the_process_exits_within_a_second(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A ``kill`` that fails (rc 1) is reported as ``Could not signal`` once the process is still there a second
    later; a kill that races the exit of its target fails although the signal did its work, and is no failure."""
    target: dict[str, Any] = {}

    def responder(argv):
        command = argv[-1]
        if command.startswith("f="):
            return (0, pid_record_for(command), "")
        if command.startswith("s=$(cat /proc/4242/stat"):
            if target:
                os.kill(target["pid"], signal.SIGTERM)  # the target goes down as the kill reports a failure
            return (1, "", "sh: 1: kill: No such process")
        return (0, "", "")

    provider, runs, _ = _provider(monkeypatch, responder)

    async def spawn_local(argv, **kwargs):
        return await docker_pty.spawn(local_program(argv, "nemo-gym-abc", tmp_path), **kwargs)

    monkeypatch.setattr(provider, "_spawn_pty", spawn_local)
    session = await provider.create_pty(_handle(), SandboxPtySpec(command="exec sleep 30", pty=False))
    async with session:
        started = time.monotonic()
        with pytest.raises(SandboxPtyError, match="Could not signal PTY session .*No such process"):
            await session.send_signal("SIGTERM")
        assert 0.9 <= time.monotonic() - started < 5.0
        assert not session.closed and session.returncode is None
    assert runs[-1][-1].startswith("rm -f -- ")  # close() cleaned up by the checked pid

    session = await provider.create_pty(_handle(), SandboxPtySpec(command="exec sleep 30", pty=False))
    target["pid"] = session._proc.pid
    await session.send_signal("SIGTERM")  # no raise: the exit settled within the second
    assert await session.wait_exit(timeout_s=10) in (-signal.SIGTERM, 128 + signal.SIGTERM)
    assert runs[-1][-1].startswith("s=$(cat /proc/4242/stat")


async def test_container_close_abandons_a_live_session_without_reaching_into_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``provider.close(handle)`` with a live session: the session is abandoned (local teardown only, no exec into
    a container that is going away), a pending ``wait_exit()`` fails as on ``close()``, the client is gone, and the
    one docker call is the ``rm -f``; ``close()`` of the abandoned session afterwards does nothing more."""
    provider, runs, _ = _provider(monkeypatch)

    async def spawn_local(argv, **kwargs):
        return await docker_pty.spawn(local_program(argv, "nemo-gym-abc", tmp_path), **kwargs)

    monkeypatch.setattr(provider, "_spawn_pty", spawn_local)
    handle = _handle()
    session = await provider.create_pty(handle, SandboxPtySpec(command="exec sleep 30", pty=False))
    waiter = asyncio.create_task(session.wait_exit())
    await asyncio.sleep(0.05)
    assert isinstance(session, docker_pty.DockerPtySession) and session.returncode is None
    await provider.close(handle)
    assert runs == [["/usr/bin/docker", "rm", "-f", "nemo-gym-abc"]]
    with pytest.raises(SandboxPtyError, match="closed before process exit"):
        await waiter
    assert session.closed and session.returncode is not None and provider._pty_sessions == {}
    await session.close()
    assert len(runs) == 1
    with pytest.raises(SandboxPtyError, match="closed"):
        await session.send_signal("SIGTERM")


MARKER = docker_pty.session_marker(SESSION_ID)


@pytest.mark.parametrize(
    ("record", "mode", "problem"),
    [
        (PID_RECORD, "pty", None),
        (PID_RECORD, "pipe", None),
        # A session leader is required only where the exec has a terminal (pipe mode may lead none).
        (PID_RECORD.replace("S 0 4242 4242", "S 0 4242 7"), "pty", "does not lead its session"),
        (PID_RECORD.replace("S 0 4242 4242", "S 0 4242 7"), "pipe", None),
        # The process must carry this session's marker: a file rewritten (or relinked) by the session's user may
        # only lead to a process this session's exec started.
        (PID_RECORD.replace(SESSION_ID, "other0"), "pty", "pid 4242 is not a process of PTY session sess01"),
        (PID_RECORD.replace(MARKER, "0"), "pipe", "not a process of PTY session sess01 [(]marker '0'[)]"),
        # ... and have been injected by the runtime (parent outside the pid namespace), not started inside: the
        # agent's own children inherit the marker but have a parent in the container.
        (
            PID_RECORD.replace("S 0 4242", "S 17 4242"),
            "pipe",
            "not started by the runtime [(]parent 17, expected 0[)]",
        ),
        (f"1 {MARKER}\n1 (init) S 0 1 1 0 -1\n", "pipe", "not a usable pid: '1'"),
        ("0 x\nx\n", "pipe", "not a usable pid: '0'"),
        ("\u00b24 x\nx\n", "pipe", "not a usable pid"),
        (f"4242 {MARKER}\n", "pty", "unexpected pid record"),
        ("4242\nx\n", "pty", "unexpected pid record"),
        (f"4242 {MARKER} extra\nx\n", "pty", "unexpected pid record"),
        (f"4242 {MARKER}\n\n", "pty", "gone or unreadable"),
        (f"4242 {MARKER}\n4242 (bash) S 0 4242 4242 34816\n", "pty", "gone or unreadable"),
    ],
)
def test_check_recorded_pid(record: str, mode: str, problem: str | None) -> None:
    if problem is None:
        assert docker_pty.check_recorded_pid(record, mode=mode, session_id=SESSION_ID) == "4242"
    else:
        with pytest.raises(ValueError, match=problem):
            docker_pty.check_recorded_pid(record, mode=mode, session_id=SESSION_ID)


def _live_kill_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    kill_shell: list[str] | None = None,
    *,
    pid_dir: Path | None = None,
    launcher_delay_s: float = 0.0,
):
    """A provider whose sessions run locally and whose in-container execs are real: the pid read, the ``kill`` of
    a signal and ``close()``'s cleanup run on this host through ``kill_shell`` (this host's sh by default), with the
    container's pid file paths rewritten to local ones: ``pid_dir / "pid"`` for the ``/tmp`` file and
    ``pid_dir / "shm" / "pid"`` for its ``/dev/shm`` fallback (``pid_dir`` is ``tmp_path`` by default).
    ``launcher_delay_s`` holds the launcher back, as a session exec slower than a following one would. The cleanup
    is safe to run here: it ends the recorded process's own session, which ``spawn`` starts in a session of its
    own."""
    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")
    # The local stand-in for the exec'd process is a child of this test process, where the runtime's exec in a
    # container has no parent in the pid namespace (0).
    monkeypatch.setattr(docker_pty, "EXEC_PARENT_PID", str(os.getpid()))
    provider = docker_provider.DockerProvider(exec={"exec_shell": "sh"})
    commands: list[str] = []
    shell = kill_shell or [SH]
    pid_dir = Path(pid_dir or tmp_path)
    fallback = fallback_pid_path(pid_dir)

    async def run(argv, *, timeout_s, stdin=None):
        command = argv[-1]
        commands.append(command)
        local = re.sub(r"/tmp/\.nemo-gym-pty-[0-9a-f]+\.pid", str(pid_dir / "pid"), command)
        local = re.sub(r"/dev/shm/\.nemo-gym-pty-[0-9a-f]+\.pid", str(fallback), local)
        proc = await asyncio.create_subprocess_exec(
            *shell, "-c", local, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        return (proc.returncode, out.decode(), err.decode())

    monkeypatch.setattr(provider, "_run", run)

    async def spawn_local(argv, **kwargs):
        local = local_program(argv, "nemo-gym-abc", pid_dir)
        if launcher_delay_s:
            local = [SH, "-c", f'sleep {launcher_delay_s}; exec "$@"', "sh", *local]
        return await docker_pty.spawn(local, **kwargs)

    monkeypatch.setattr(provider, "_spawn_pty", spawn_local)
    return provider, commands


async def recorded_pid(pid_path: Path) -> str:
    async with asyncio.timeout(10):
        while not pid_path.exists() or not pid_path.read_text().strip():
            await asyncio.sleep(0.02)
    return pid_path.read_text().strip()


def terminal_groups(pid: str) -> tuple[str, str]:
    """``(pgrp, tpgid)`` of a process: its own process group and the foreground group of its terminal (-1 without
    one), the fields ``send_signal`` relies on, read the way it reads them."""
    fields = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()
    return fields[2], fields[5]


async def until_foreground(pid: str, *, job: bool) -> None:
    """Wait until a job other than the shell (``job=True``) or the shell itself owns the terminal's foreground."""
    async with asyncio.timeout(10):
        while True:
            pgrp, tpgid = terminal_groups(pid)
            assert tpgid != "-1", "the shell has no controlling terminal, so no job control"
            if (tpgid != pgrp) == job:
                return
            await asyncio.sleep(0.02)


@pytest.mark.parametrize("kill_shell", KILL_SHELLS)
@pytest.mark.parametrize("command", [None, "exec sh -i"], ids=["default-shell", "sh"])
async def test_send_signal_reaches_the_foreground_job_of_an_interactive_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path, command: str | None, kill_shell: list[str]
) -> None:
    """In PTY mode the signal goes to the terminal's foreground process group, as a Ctrl-C would: the job an
    interactive shell is running ends with it (status 130) and the shell stays usable. The default session is the
    image's interactive shell (bash here when installed, else sh); ``exec sh -i`` pins the POSIX sh (dash on
    Debian), which also runs each job in a group of its own. The pid read and the ``kill`` run through this host's
    sh and through busybox's, which must both accept them (busybox rejects ``kill -s NAME -- -pgid``)."""
    provider, commands = _live_kill_provider(monkeypatch, tmp_path, kill_shell)
    session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec(command=command))
    async with session:
        pid = await recorded_pid(tmp_path / "pid")
        await session.write(b"sleep 30\n")
        await until_foreground(pid, job=True)
        pgrp, foreground = terminal_groups(pid)
        assert foreground not in (pgrp, "-1"), "the shell has job control: the sleep runs in a group of its own"
        await session.send_signal("SIGINT")
        # The job is gone and the shell is back in the foreground, its status that of a SIGINT-ed job.
        await until_foreground(pid, job=False)
        await session.write(b"echo rc=$? __do''ne__\n")
        out = await read_until(session, b"__done__")
        assert b"rc=130" in out, out
        assert not session.closed
        await session.write(b"exit 0\n")
        assert await session.wait_exit(timeout_s=10) == 0
    assert any(c.startswith(f"s=$(cat /proc/{pid}/stat") and "kill -INT" in c for c in commands)
    # The session's own pid passed the check: it leads its session and runs as the owner of its pid file.
    assert session._pid == pid


async def test_send_signal_without_a_terminal_falls_back_to_the_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A pipe session has no terminal (``tpgid`` is -1), so the signal goes to the recorded process's own group and
    the process, which ends a command session."""
    provider, _ = _live_kill_provider(monkeypatch, tmp_path)
    session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec(command="sleep 30", pty=False))
    async with session:
        pid = await recorded_pid(tmp_path / "pid")
        assert terminal_groups(pid)[1] == "-1"
        await session.send_signal("SIGTERM")
        # The local client is the process itself, so its status is the signal; `docker exec` would report 143.
        assert await session.wait_exit(timeout_s=10) in (-signal.SIGTERM, 128 + signal.SIGTERM)
        with pytest.raises(SandboxPtyError, match="closed"):
            await session.send_signal("SIGTERM")


async def background_job(session, pid_path: Path) -> tuple[str, str]:
    """``(shell pid, background job pid)`` of an interactive session after ``sleep 300 &``."""
    pid = await recorded_pid(pid_path)
    await session.write(b"sleep 300 & echo bg=$! __do''ne__\n")
    out = await read_until(session, b"__done__")
    match = re.search(rb"bg=(\d+)", out)
    assert match, out
    return pid, match.group(1).decode()


def process_state(pid: str) -> str | None:
    """The state letter of a process, ``None`` once it is gone (a reaped zombie included)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()[0]
    except FileNotFoundError:
        return None


async def until_gone(pid: str, *, timeout_s: float = 10.0) -> None:
    async with asyncio.timeout(timeout_s):
        while process_state(pid) not in (None, "Z", "X"):
            await asyncio.sleep(0.02)


@pytest.mark.parametrize("kill_shell", KILL_SHELLS)
async def test_close_of_a_live_shell_session_ends_its_background_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path, kill_shell: list[str]
) -> None:
    """``close()`` of a live interactive session hangs the shell up and waits for it to relay the hangup before the
    SIGKILL, then sweeps the session: a background job (its own process group under job control, out of reach of a
    signal to the shell's group) does not survive the session."""
    provider, commands = _live_kill_provider(monkeypatch, tmp_path, kill_shell)
    session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec())
    pid, job = await background_job(session, tmp_path / "pid")
    assert process_state(job) not in (None, "Z", "X")
    await session.close()
    await until_gone(job, timeout_s=5.0)
    await until_gone(pid, timeout_s=5.0)
    assert commands[-1] == docker_pty.CLEANUP_COMMAND.format(pid_files=pid_files(session), pid=pid)
    assert not (tmp_path / "pid").exists()


async def test_a_rewritten_pid_file_is_never_a_kill_target(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The pid file belongs to the session's user, who can rewrite it: a file that names pid 1 (``kill -1`` would
    reach every process in the container) is refused before any kill, by ``send_signal`` and by ``close()``, which
    then only drops the file. A file naming a process of another user (here kthreadd, uid 0, a session leader) is
    refused because that process carries no marker of this session; one naming a process of the session's own tree
    (its background job: the marker inherited, the parent inside the container) is refused because the runtime did
    not start it."""
    provider, commands = _live_kill_provider(monkeypatch, tmp_path)
    session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec())
    real_pid, job = await background_job(session, tmp_path / "pid")
    try:
        (tmp_path / "pid").write_text("1\n")
        with pytest.raises(SandboxPtyError, match="no usable recorded pid.*no usable pid in .*: 1"):
            await session.send_signal("SIGTERM")
        assert session._pid is None and not session.closed and process_state(real_pid) not in (None, "Z", "X")
        (tmp_path / "pid").write_text("2\n")
        with pytest.raises(SandboxPtyError, match="no usable recorded pid.*pid 2 is not a process of PTY session"):
            await session.send_signal("SIGTERM")
        (tmp_path / "pid").write_text(f"{job}\n")
        with pytest.raises(SandboxPtyError, match=f"process {job} was not started by the runtime"):
            await session.send_signal("SIGTERM")
        assert session._pid is None and process_state(job) not in (None, "Z", "X")
        (tmp_path / "pid").write_text("1\n")
        await session.close()
        assert not any("kill" in c for c in commands)
        assert commands[-1] == rm_command(session)
        assert not (tmp_path / "pid").exists()
    finally:
        # Nothing cleaned the session up in the "container": the test ends the job itself.
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(job), signal.SIGKILL)


async def test_a_pid_file_replaced_by_a_link_is_never_a_kill_target(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The owner of the pid file's inode is no identity: in a world-writable ``/tmp`` the session's user can unlink
    its pid file and put a hard link to a file it does not own (but may write) in its place, or a symlink, so that
    the path names content of its choosing under another owner. The pid such a file names is followed only when
    its process carries this session's marker and was started by the runtime; a linked file naming a root process
    (pid 2 here) is refused before any kill, and ``close()`` then only drops the link."""
    provider, commands = _live_kill_provider(monkeypatch, tmp_path)
    session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec())
    real_pid = await recorded_pid(tmp_path / "pid")
    other = tmp_path / "someone_elses_file"  # stands in for a root-owned file the session's user can write
    other.write_text("2\n")
    pid_path = tmp_path / "pid"
    pid_path.unlink()
    os.link(other, pid_path)
    assert pid_path.stat().st_nlink == 2
    with pytest.raises(SandboxPtyError, match="no usable recorded pid.*pid 2 is not a process of PTY session"):
        await session.send_signal("SIGTERM")
    pid_path.unlink()
    pid_path.symlink_to(other)
    with pytest.raises(SandboxPtyError, match="no usable recorded pid.*pid 2 is not a process of PTY session"):
        await session.send_signal("SIGTERM")
    assert session._pid is None and not session.closed and process_state(real_pid) not in (None, "Z", "X")
    await session.close()
    assert not any("kill" in c for c in commands)
    assert commands[-1] == rm_command(session)
    assert not pid_path.exists() and not pid_path.is_symlink() and other.read_text() == "2\n"


def integer_only_sleep(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Path:
    """Put a ``sleep`` that rejects fractions (busybox built without fractional sleep) first on PATH; the file
    returned records every argument it was called with."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "sleep_calls"
    fake_sleep = bin_dir / "sleep"
    fake_sleep.write_text(
        f'#!/bin/sh\necho "$1" >> {calls}\ncase "$1" in *.*) echo "sleep: invalid number \'$1\'" >&2; exit 1;; esac\n'
        f'exec {shutil.which("sleep")} "$@"\n'
    )
    fake_sleep.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return calls


async def test_close_hangup_grace_is_one_second_with_an_integer_only_sleep(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``close()`` gives the hangup up to a second whatever the image's ``sleep``: where it rejects fractions (busybox
    built without fractional sleep) the loop sleeps the whole second once and stops, instead of ten times. A process
    that ignores ``SIGHUP`` under such a ``sleep`` is swept by the ``SIGKILL`` after about a second, not ten."""
    calls = integer_only_sleep(monkeypatch, tmp_path)
    provider, commands = _live_kill_provider(monkeypatch, tmp_path)
    session = await provider.create_pty(
        _handle(shell="sh"), SandboxPtySpec(command="trap '' HUP; sleep 300", pty=False)
    )
    pid = await recorded_pid(tmp_path / "pid")
    await asyncio.sleep(0.2)  # the command's own `sleep 300` is under way
    started = time.monotonic()
    await session.close()
    elapsed = time.monotonic() - started
    await until_gone(pid, timeout_s=5.0)
    assert "sleep 0.1 2>/dev/null || { sleep 1; n=9; }; n=$((n+1)); done" in commands[-1]
    # One rejected fractional sleep, one whole second, and the command's own sleep: no second iteration.
    assert sorted(calls.read_text().split()) == ["0.1", "1", "300"], calls.read_text()
    assert elapsed < 5.0, elapsed


@pytest.mark.parametrize("kill_shell", KILL_SHELLS)
async def test_close_right_after_create_waits_for_the_launchers_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path, kill_shell: list[str]
) -> None:
    """``create_pty()`` returns as soon as the client is spawned, so a ``close()`` or ``send_signal()`` issued right
    away reads the pid file through a second exec that may reach the container before the launcher has written it.
    The read waits for the record: with the launcher held back half a second, ``close()`` of a fresh interactive
    session still ends its shell through the checked pid instead of falling back to ``rm -f``, and ``send_signal()``
    right after ``create_pty()`` still reaches a command session."""
    provider, commands = _live_kill_provider(monkeypatch, tmp_path, kill_shell, launcher_delay_s=0.5)
    session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec())
    started = time.monotonic()
    await session.close()
    elapsed = time.monotonic() - started
    pid = session._pid
    assert pid is not None, "the pid read gave up before the launcher recorded the pid"
    assert 0.4 <= elapsed < 5.0, elapsed
    assert commands[-2:] == [
        pid_read_command(session),
        docker_pty.CLEANUP_COMMAND.format(pid_files=pid_files(session), pid=pid),
    ]
    await until_gone(pid, timeout_s=5.0)
    assert not (tmp_path / "pid").exists()

    session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec(command="exec sleep 30", pty=False))
    await session.send_signal("SIGTERM")
    assert await session.wait_exit(timeout_s=10) in (-signal.SIGTERM, 128 + signal.SIGTERM)
    await session.close()


async def test_the_wait_for_a_missing_record_is_bounded_with_an_integer_only_sleep(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A live session whose pid file is gone and does not come back is refused once the wait is over, about two
    seconds, and under a ``sleep`` that rejects fractions the loop sleeps whole seconds and stops after the same two
    (two rejected fractional sleeps, two whole ones) instead of twenty. Nothing is killed on the way, and a record
    put back is found at once by ``close()``."""
    calls = integer_only_sleep(monkeypatch, tmp_path)
    provider, commands = _live_kill_provider(monkeypatch, tmp_path)
    session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec(command="exec sleep 30", pty=False))
    pid = await recorded_pid(tmp_path / "pid")
    (tmp_path / "pid").unlink()
    started = time.monotonic()
    pid_path, fallback = tmp_path / "pid", fallback_pid_path(tmp_path)
    with pytest.raises(
        SandboxPtyError, match=f"no usable pid in {re.escape(str(pid_path))} or {re.escape(str(fallback))}:$"
    ):
        await session.send_signal("SIGTERM")
    elapsed = time.monotonic() - started
    assert 1.9 <= elapsed < 5.0, elapsed
    assert session._pid is None and not session.closed and process_state(pid) not in (None, "Z", "X")
    assert [c for c in calls.read_text().split() if c != "30"] == ["0.1", "1", "0.1", "1"], calls.read_text()
    assert not any("kill" in c for c in commands)
    # The read tests the path before it opens it, so a missing record is not created by the attempts.
    assert not pid_path.exists() and not fallback.exists()
    pid_path.write_text(f"{pid}\n")
    started = time.monotonic()
    await session.close()
    assert time.monotonic() - started < 1.9
    assert commands[-1] == docker_pty.CLEANUP_COMMAND.format(pid_files=pid_files(session), pid=pid)
    await until_gone(pid, timeout_s=5.0)


@pytest.mark.parametrize("kill_shell", KILL_SHELLS)
async def test_the_record_open_never_blocks_whatever_the_path_turns_into(tmp_path, kill_shell: list[str]) -> None:
    """``PID_RECORD_READ`` is what runs once the record's path tested as a regular file, and the session's user can
    swap the path between that test and the open, so the open itself must never wait and the open file is tested
    again on its own descriptor: a FIFO (opened read-write, so no wait for a writer, with or without one attached),
    a symlink to one, a device and a directory all come back at once, empty and failed, and a regular file gives at
    most 32 bytes, however large it is."""
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    link = tmp_path / "link"
    link.symlink_to(fifo)
    zero = tmp_path / "zero"
    zero.symlink_to("/dev/zero")
    big = tmp_path / "big"
    big.write_bytes(b"7" * 1_000_000)
    record = tmp_path / "record"
    record.write_text("4242\n")

    async def attempt(path: Path) -> tuple[int | None, bytes, float]:
        proc = await asyncio.create_subprocess_exec(
            *kill_shell,
            "-c",
            docker_pty.PID_RECORD_READ,
            "sh",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        started = time.monotonic()
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return proc.returncode, out, time.monotonic() - started

    for path in (fifo, link, zero, tmp_path):
        rc, out, elapsed = await attempt(path)
        assert rc not in (None, 0) and out == b"" and elapsed < 1.0, (path, rc, out, elapsed)
    # A FIFO with a writer attached opens read-only without waiting but its read would wait forever; the pinned
    # test refuses it before any read.
    held = os.open(fifo, os.O_RDWR)
    try:
        rc, out, elapsed = await attempt(fifo)
        assert rc not in (None, 0) and out == b"" and elapsed < 1.0, (rc, out, elapsed)
    finally:
        os.close(held)
    rc, out, _ = await attempt(big)
    assert rc == 0 and out == b"7" * 32
    rc, out, _ = await attempt(record)
    assert rc == 0 and out == b"4242\n"


@pytest.mark.parametrize("kill_shell", KILL_SHELLS)
async def test_a_pid_record_that_is_not_a_regular_file_is_refused_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path, kill_shell: list[str]
) -> None:
    """The session's user can replace its pid record, once it sees it, by something a plain read would wait on
    forever: a FIFO, a symlink to one at the fallback path, a link to ``/dev/zero``. The read refuses a record that
    is not a regular file at once, reads a regular one 32 bytes at most, and neither ``send_signal()`` nor
    ``close()`` stalls on it: both come back within the two-second wait, nothing is killed, the process stays with
    the container and ``close()`` only removes the record."""
    provider, commands = _live_kill_provider(monkeypatch, tmp_path, kill_shell)
    session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec(command="exec sleep 300", pty=False))
    pid = await recorded_pid(tmp_path / "pid")
    pid_path, fallback = tmp_path / "pid", fallback_pid_path(tmp_path)
    real_fifo = tmp_path / "fifo"
    os.mkfifo(real_fifo)

    def clear() -> None:
        for path in (pid_path, fallback):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()

    try:
        for label, plant, culprit in (
            ("a FIFO in place of the pid file", lambda: os.mkfifo(pid_path), pid_path),
            ("the pid file gone, the fallback a symlink to a FIFO", lambda: fallback.symlink_to(real_fifo), fallback),
            ("the fallback a symlink to /dev/zero", lambda: fallback.symlink_to("/dev/zero"), fallback),
        ):
            clear()
            plant()
            started = time.monotonic()
            with pytest.raises(
                SandboxPtyError,
                match=f"no usable recorded pid.*pid record {re.escape(str(culprit))} is not a regular file",
            ):
                await session.send_signal("SIGTERM")
            assert time.monotonic() - started < 1.5, label
            assert session._pid is None and not session.closed, label
            assert process_state(pid) not in (None, "Z", "X"), label
        # A regular file is read 32 bytes at most: a huge one is refused at once as garbage, not slurped.
        clear()
        pid_path.write_bytes(b"x" * 1_000_000)
        started = time.monotonic()
        # The shell's report is clipped to 200 characters, so under a long temporary path (an xdist worker adds a
        # folder) only the start of the junk record survives; that is enough to tell it apart.
        with pytest.raises(SandboxPtyError, match="no usable recorded pid.*no usable pid in .*: x{8,}") as excinfo:
            await session.send_signal("SIGTERM")
        assert time.monotonic() - started < 1.5
        assert "x" * 33 not in str(excinfo.value)
        assert not any("kill" in c for c in commands)
        # close() does not wait on such a record either: it only removes it.
        clear()
        os.mkfifo(pid_path)
        started = time.monotonic()
        await session.close()
        assert time.monotonic() - started < 1.5
        assert commands[-1] == rm_command(session) and not any("kill" in c for c in commands)
        assert not pid_path.exists() and not fallback.exists() and not fallback.is_symlink()
    finally:
        # Nothing cleaned the session up in the "container": the test ends the process itself.
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(pid), signal.SIGKILL)


async def test_the_pid_read_exec_has_a_timeout_of_its_own(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The exec that reads the record runs under ``PID_READ_TIMEOUT_S`` (the two-second wait plus the round trip),
    never under the provider's ``exec.default_timeout_s`` when that is longer (180 s shipped), and never above the
    default when that is shorter; the ``kill`` and the cleanup keep the default. The session asks for the bound on
    the read alone, and the provider's hook applies it."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def hook(command: str, **kwargs: Any):
        calls.append((command.split(";")[0], kwargs))
        from nemo_gym.sandbox import SandboxExecResult

        stdout = PID_RECORD if command.startswith("f=") else ""
        return SandboxExecResult(stdout=stdout, stderr="", return_code=0, error_type=None)

    session = await docker_pty.spawn(
        [SH, "-c", "exec sleep 30"],
        use_pty=False,
        rows=24,
        cols=80,
        session_id=SESSION_ID,
        run_in_container=hook,
        pid_file=docker_pty.pid_file_for(SESSION_ID),
    )
    async with session:
        await session.send_signal("SIGTERM")
        await session.send_signal("SIGTERM")
    assert [kwargs for _, kwargs in calls] == [
        {"as_session_user": True, "timeout_s": docker_pty.PID_READ_TIMEOUT_S},
        {},
        {},
        {},
    ]
    assert [c.startswith("f=") for c, _ in calls] == [True, False, False, False]

    for default, expected in ((180, [30.0, 180, 180]), (3, [3, 3, 3]), (None, [30, None, None])):
        provider, _, spawned = _provider(monkeypatch, exec={"exec_shell": "bash", "default_timeout_s": default})
        timeouts: list[float | None] = []

        async def run(argv, *, timeout_s, stdin=None):
            timeouts.append(timeout_s)
            return (0, "", "")

        monkeypatch.setattr(provider, "_run", run)
        fake = await provider.create_pty(_handle(), SandboxPtySpec(user="agent"))
        await fake.run_in_container("f=read", as_session_user=True, timeout_s=docker_pty.PID_READ_TIMEOUT_S)
        await fake.run_in_container("kill")
        await fake.run_in_container("rm", timeout_s=None)
        assert timeouts == expected, (default, timeouts)
        assert docker_pty.PID_READ_TIMEOUT_S == 30


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a 0555 directory")
async def test_a_pid_tmp_refuses_is_recorded_under_dev_shm(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Under ``create.read_only: true`` without a tmpfs on ``/tmp`` the launcher's write to ``/tmp`` fails and the
    pid goes to ``/dev/shm`` (stood in for by an unwritable ``pid_dir`` with a writable ``shm/``). ``send_signal()``
    and ``close()`` work from that record as from the ``/tmp`` one: the interrupt reaches the foreground job of the
    interactive shell, ``close()`` sweeps the shell and its background job and removes both pid paths, and a command
    session on pipes ends on ``SIGTERM``."""
    pid_dir = tmp_path / "ro"
    fallback = fallback_pid_path(pid_dir)
    pid_dir.chmod(0o555)
    try:
        provider, commands = _live_kill_provider(monkeypatch, tmp_path, pid_dir=pid_dir)
        session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec())
        pid, job = await background_job(session, fallback)
        assert not (pid_dir / "pid").exists()
        await session.write(b"sleep 30\n")
        await until_foreground(pid, job=True)
        await session.send_signal("SIGINT")
        await until_foreground(pid, job=False)
        await session.write(b"echo rc=$? __do''ne__\n")
        assert b"rc=130" in await read_until(session, b"__done__")
        await session.close()
        await until_gone(job, timeout_s=5.0)
        await until_gone(pid, timeout_s=5.0)
        assert commands[-1] == docker_pty.CLEANUP_COMMAND.format(pid_files=pid_files(session), pid=pid)
        assert commands[-1].startswith(
            f"rm -f -- /tmp/.nemo-gym-pty-{session.session_id}.pid /dev/shm/.nemo-gym-pty-{session.session_id}.pid; "
        )
        assert not fallback.exists()

        session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec(command="exec sleep 30", pty=False))
        pid = await recorded_pid(fallback)
        await session.send_signal("SIGTERM")
        assert await session.wait_exit(timeout_s=10) in (-signal.SIGTERM, 128 + signal.SIGTERM)
        await session.close()
        assert commands[-1] == rm_command(session) and not fallback.exists()
    finally:
        pid_dir.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a 0555 directory")
async def test_a_session_whose_pid_can_be_recorded_nowhere_ends_at_once_with_a_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Neither ``/tmp`` nor ``/dev/shm`` writable (a read-only root filesystem under ``--ipc=none``, say): a session
    that could be neither signalled nor cleaned up is not started. The launcher says why on stderr (the terminal in
    PTY mode) and exits 1, so the session ends at once with ``wait_exit()`` 1 and the message in its output,
    ``send_signal()`` finds it closed, and ``close()`` only removes the pid paths."""
    pid_dir = tmp_path / "ro"
    fallback = fallback_pid_path(pid_dir)
    fallback.parent.chmod(0o555)
    pid_dir.chmod(0o555)
    try:
        provider, commands = _live_kill_provider(monkeypatch, tmp_path, pid_dir=pid_dir)
        message = f"nemo-gym: cannot record the pid of this PTY session in {pid_dir / 'pid'} or {fallback}"
        session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec(command="exec sleep 30", pty=False))
        assert await session.wait_exit(timeout_s=10) == 1
        err = bytearray()
        while chunk := await session.read_stderr(timeout_s=5):
            err.extend(chunk)
        assert err.decode().strip() == message and await session.read(timeout_s=5) == b""
        with pytest.raises(SandboxPtyError, match="closed"):
            await session.send_signal("SIGTERM")
        await session.close()
        assert commands == [rm_command(session)]

        commands.clear()
        session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec())
        assert await session.wait_exit(timeout_s=10) == 1
        assert message.encode() in await read_until(session, b"\n")
        await session.close()
        assert commands == [rm_command(session)]
    finally:
        fallback.parent.chmod(0o755)
        pid_dir.chmod(0o755)


def fake_docker_cli(tmp_path, script: str) -> str:
    """An executable standing in for the docker CLI: every invocation runs ``script`` under sh, whatever the
    arguments, as the client of a session and of the provider's execs alike."""
    fake = Path(tmp_path) / "docker"
    fake.write_text(f"#!/bin/sh\n{script}\n")
    fake.chmod(0o755)
    return str(fake)


# What the daemon answers an exec in a paused container; the CLI prints it on its stderr and exits 1.
REFUSED = "Error response from daemon: container 3f2a is paused, unpause the container before exec"


@pytest.mark.parametrize("pty_mode", [True, False], ids=["pty", "pipe"])
async def test_a_docker_exec_that_cannot_start_the_program_ends_the_session_with_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path, pty_mode: bool
) -> None:
    """The daemon refusing the exec (a paused or removed container, an unreachable daemon) makes the client exit 1
    with the refusal on its stderr, a code a program could exit with too. It is told apart by that stderr, with the
    provider's own ``exec()`` classification (the same refusal gives ``exec()`` its runtime sentinel and
    ``error_type="sandbox"``), and ends the session with ``SandboxPtyError`` from ``wait_exit()`` and ``read()``
    rather than passing 1 off as the program's exit, so ``sandbox.pty.exec()`` raises instead of returning what
    looks like a failed command."""
    monkeypatch.setattr(
        docker_provider, "_require_docker", lambda: fake_docker_cli(tmp_path, f"echo '{REFUSED}' >&2; exit 1")
    )
    provider = docker_provider.DockerProvider(exec={"exec_shell": "sh"})
    handle = _handle()
    result = await provider.exec(handle, "true")
    assert (result.return_code, result.error_type) == (125, "sandbox")
    session = await provider.create_pty(handle, SandboxPtySpec(command="true", pty=pty_mode))
    async with session:
        with pytest.raises(
            SandboxPtyError, match=f"could not be started .docker exec exited 1.: {re.escape(REFUSED)}"
        ):
            await session.wait_exit(timeout_s=10)
        assert session.closed and session.returncode == 1
        with pytest.raises(SandboxPtyError, match="could not be started"):
            while await session.read(timeout_s=10):
                pass
    sandbox = AsyncSandbox(provider)
    sandbox._handle, sandbox._stopped = handle, False
    with pytest.raises(SandboxPtyError, match="could not be started"):
        await sandbox.pty.exec("true", pty=pty_mode, timeout_s=10)


@pytest.mark.parametrize("pty_mode", [True, False], ids=["pty", "pipe"])
async def test_a_programs_nonzero_exit_stays_the_programs(
    monkeypatch: pytest.MonkeyPatch, tmp_path, pty_mode: bool
) -> None:
    """A client that ran the program reports the program's exit code, and the session keeps it: ``wait_exit()``
    returns 7 whatever was written to stderr (a complaint of the program's own is no refusal of the exec), and
    ``sandbox.pty.exec()`` returns it as a plain result. In PTY mode the client's own stderr is kept out of the
    terminal's output (the program's stderr arrives through the terminal); in pipe mode stderr is the program's,
    split as before."""
    monkeypatch.setattr(
        docker_provider,
        "_require_docker",
        lambda: fake_docker_cli(tmp_path, "echo out; echo 'program complaint' >&2; exit 7"),
    )
    provider = docker_provider.DockerProvider(exec={"exec_shell": "sh"})
    handle = _handle()
    session = await provider.create_pty(handle, SandboxPtySpec(command="false", pty=pty_mode))
    async with session:
        assert await session.wait_exit(timeout_s=10) == 7
        out, err = bytearray(), bytearray()
        while chunk := await session.read(timeout_s=10):
            out.extend(chunk)
        while chunk := await session.read_stderr(timeout_s=10):
            err.extend(chunk)
        assert (bytes(out), bytes(err)) == (b"out\n", b"" if pty_mode else b"program complaint\n")
    sandbox = AsyncSandbox(provider)
    sandbox._handle, sandbox._stopped = handle, False
    result = await sandbox.pty.exec("false", pty=pty_mode, timeout_s=10)
    assert (result.return_code, result.error_type, result.stdout) == (7, None, "out\n")
    assert result.stderr == (None if pty_mode else "program complaint\n")


async def test_container_close_abandons_sessions_and_aclose_closes_them(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, runs, spawned = _provider(monkeypatch)
    first, second = (
        _handle(),
        SandboxHandle(
            sandbox_id="nemo-gym-other",
            provider_name="docker",
            raw=docker_provider._DockerContainer(name="nemo-gym-other", image="img", shell="bash"),
        ),
    )
    await provider.create_pty(first, SandboxPtySpec())
    await provider.create_pty(first, SandboxPtySpec())
    await provider.create_pty(second, SandboxPtySpec())
    # The container is removed: its sessions are released without reaching into it, before `docker rm -f`.
    await provider.close(first)
    assert [s.abandoned for s in spawned] == [True, True, False]
    assert runs[-1][:3] == ["/usr/bin/docker", "rm", "-f"] and "nemo-gym-abc" not in provider._pty_sessions
    # Provider shutdown closes what is left properly (the containers still belong to their handles).
    await provider.aclose()
    assert spawned[2].closed and not spawned[2].abandoned and provider._pty_sessions == {}


async def test_exited_sessions_are_released_when_retired_on_the_next_create(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session whose process exited counts as closed, but its host pty master stays open and its pid record stays
    in the container until ``close()``, which a caller that only read the session to EOF never issued. The next
    ``create_pty`` of the container retires such sessions and releases them (the master closed, the record removed
    through the one ``rm -f`` exec a session that exited gets), instead of only forgetting them: forgotten, they
    would hold their descriptors until the container's close or ``aclose()``, which no longer reach them."""
    provider, runs, _ = _provider(monkeypatch)

    async def spawn_local(argv, **kwargs):
        return await docker_pty.spawn([SH, "-c", "exit 0"], **kwargs)

    monkeypatch.setattr(provider, "_spawn_pty", spawn_local)
    handle = _handle()
    exited = [await provider.create_pty(handle, SandboxPtySpec()) for _ in range(3)]
    for session in exited:
        assert await session.wait_exit(timeout_s=10) == 0 and session.closed
    assert all(session._master_fd is not None for session in exited) and runs == []
    live = await provider.create_pty(handle, SandboxPtySpec(command="sleep 30"))
    assert provider._pty_sessions == {"nemo-gym-abc": {live}}
    assert all(session._master_fd is None for session in exited)
    assert sorted(argv[-1] for argv in runs) == sorted(rm_command(session) for session in exited)
    # A session closed by its caller is retired without a second release.
    runs.clear()
    assert await live.wait_exit(timeout_s=10) == 0
    await live.close()
    assert [argv[-1] for argv in runs] == [rm_command(live)]
    await provider.create_pty(handle, SandboxPtySpec())
    assert [argv[-1] for argv in runs] == [rm_command(live)] and live not in provider._pty_sessions["nemo-gym-abc"]
    await provider.aclose()


@pytest.mark.parametrize("shutdown", ["close", "close_then_aclose"])
async def test_a_session_spawned_while_its_container_closes_is_abandoned_and_refused(
    monkeypatch: pytest.MonkeyPatch, shutdown: str
) -> None:
    """``create_pty`` awaits the stale releases and the spawn; ``close(handle)`` arriving meanwhile (the sandbox's
    TTL or teardown racing a harness that opens a session) pops the container's session set and abandons what it
    held, and the facade's ``stop()`` follows it with ``aclose()`` at once. A session that comes back after either
    would leak its master and client: added to the popped set it is out of ``aclose()``'s reach, added to the
    provider after ``aclose()`` it sits in a provider nothing closes again. Its container is gone: the session is
    abandoned like its siblings (the client ended, the master closed) and the caller gets ``SandboxPtyError``."""
    provider, runs, _ = _provider(monkeypatch)
    handle = _handle()
    gate = asyncio.Event()
    spawned: list[docker_pty.DockerPtySession] = []

    async def spawn_after_gate(argv, **kwargs):
        await gate.wait()
        spawned.append(await docker_pty.spawn([SH, "-c", "sleep 30"], **kwargs))
        return spawned[-1]

    monkeypatch.setattr(provider, "_spawn_pty", spawn_after_gate)
    gate.set()
    earlier = await provider.create_pty(handle, SandboxPtySpec())  # a sibling the close abandons
    gate.clear()
    creating = asyncio.create_task(provider.create_pty(handle, SandboxPtySpec()))
    await asyncio.sleep(0.01)
    assert not creating.done()
    await provider.close(handle)
    assert earlier._abandoned and earlier._master_fd is None and runs[-1][:3] == ["/usr/bin/docker", "rm", "-f"]
    if shutdown == "close_then_aclose":
        await provider.aclose()
    gate.set()
    with pytest.raises(SandboxPtyError, match="closed while its PTY session was being opened"):
        await creating
    [session] = spawned[1:]
    assert session._abandoned and session._master_fd is None and session.returncode is not None
    assert provider._pty_sessions == {}
    # Abandoned, not closed: the container is gone, so no cleanup exec was attempted against it.
    assert [argv[1] for argv in runs] == ["rm"]
    await provider.aclose()
    assert provider._pty_sessions == {}


# --------------------------------------------------------------------------- #
# A non-root session in a stand-in container: the uids and capabilities Docker gives
# --------------------------------------------------------------------------- #
NS_UID = 1000  # the session's uid in the stand-in container, a subordinate uid of this user through /etc/subuid
UNSHARE, NSENTER, SETPRIV, SETSID = (shutil.which(t) for t in ("unshare", "nsenter", "setpriv", "setsid"))


class StandInContainer:
    """A user, pid and mount namespace standing in for a container, as far as the pid read, the ``kill`` and the
    cleanup are concerned: its own pid numbering (a process entered from outside has ``PPid`` 0 inside, as the
    runtime's execs do in a container), a private tmpfs ``/tmp`` for the pid file at its real path, uid 0 mapped to
    this user and ``NS_UID`` to a subordinate uid, so a session run as ``NS_UID`` and the root-side execs run as
    uid 0 are different uids for real; the root-side execs drop ``CAP_SYS_PTRACE``, which Docker's default
    capability set does not grant either."""

    def __init__(self) -> None:
        self._holder: asyncio.subprocess.Process | None = None
        self.init_pid: int | None = None

    async def __aenter__(self) -> "StandInContainer":
        if not all((UNSHARE, NSENTER, SETPRIV, SETSID)):
            pytest.skip("needs util-linux unshare, nsenter, setpriv and setsid")
        self._holder = await asyncio.create_subprocess_exec(
            UNSHARE,
            *("--user", "--map-root-user", "--map-auto", "--fork", "--pid", "--mount-proc"),
            *(SH, "-c", "mount -t tmpfs tmpfs /tmp && : > /tmp/.ready && exec sleep 600"),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        children = Path(f"/proc/{self._holder.pid}/task/{self._holder.pid}/children")
        try:
            async with asyncio.timeout(15):
                while True:
                    with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                        await asyncio.wait_for(self._holder.wait(), timeout=0.05)
                    if self._holder.returncode is not None:
                        err = (await self._holder.stderr.read()).decode(errors="replace").strip()
                        pytest.skip(f"cannot start a user+pid namespace here: {err or self._holder.returncode}")
                    try:
                        inner = children.read_text().split()
                    except OSError:
                        inner = []
                    if inner:
                        self.init_pid = int(inner[0])
                        if (await self.sh("test -e /tmp/.ready"))[0] == 0 and (await self.sh("true", user=NS_UID))[
                            0
                        ] == 0:
                            return self
        except (asyncio.TimeoutError, TimeoutError):
            await self.__aexit__(None, None, None)
            pytest.skip("the stand-in container did not come up")

    async def __aexit__(self, *exc: Any) -> None:
        if self.init_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.init_pid, signal.SIGKILL)  # pid 1 of the namespace: the kernel ends the rest
        if self._holder is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._holder.wait(), timeout=10)

    def enter(self, *, user: int | None) -> list[str]:
        """The command prefix that runs a program inside: as ``user`` (no capabilities), or as the namespace's root
        without ``CAP_SYS_PTRACE``, the capability set a root exec has in a container Docker ran with its defaults."""
        base = [NSENTER, "-t", str(self.init_pid), "-U", "-p", "-m"]
        if user is not None:
            return [*base, "-S", str(user), "-G", str(user), "--"]
        return [*base, "--", SETPRIV, "--bounding-set=-sys_ptrace", "--inh-caps=-sys_ptrace", "--"]

    async def sh(self, command: str, *, user: int | None = None) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *self.enter(user=user),
            SH,
            "-c",
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")

    async def stat_fields(self, pid: str) -> list[str]:
        """The fields of ``/proc/<pid>/stat`` after ``comm``, as seen inside; ``[]`` once the process is gone."""
        rc, out, _ = await self.sh(f"cat /proc/{pid}/stat")
        return out.rpartition(") ")[2].split() if rc == 0 else []

    async def until_gone(self, pid: str, *, timeout_s: float = 10.0) -> None:
        async with asyncio.timeout(timeout_s):
            while (fields := await self.stat_fields(pid)) and fields[0] not in ("Z", "X"):
                await asyncio.sleep(0.05)

    async def until_foreground(self, pid: str, *, job: bool) -> None:
        async with asyncio.timeout(10):
            while True:
                fields = await self.stat_fields(pid)
                assert fields and fields[5] != "-1", "the shell has no controlling terminal, so no job control"
                if (fields[5] != fields[2]) == job:
                    return
                await asyncio.sleep(0.05)


def _stand_in_provider(monkeypatch: pytest.MonkeyPatch, ns: StandInContainer):
    """A provider whose session runs inside ``ns`` as ``NS_UID`` (``setsid -c``: a session of its own with the pty
    as controlling tty, as runc gives an exec) and whose in-container execs run inside ``ns`` as the user their
    ``--user`` names: the root-side ones as the namespace's root without ``CAP_SYS_PTRACE``. Nothing is rewritten:
    the pid file is at its real path in the namespace's ``/tmp`` and ``EXEC_PARENT_PID`` is the real 0."""
    monkeypatch.setattr(docker_provider, "_require_docker", lambda: "/usr/bin/docker")
    provider = docker_provider.DockerProvider(exec={"exec_shell": "sh"})
    execs: list[list[str]] = []

    async def run(argv, *, timeout_s, stdin=None):
        execs.append(list(argv))
        user = exec_user(argv)
        return await ns.sh(argv[-1], user=None if user == "0" else int(user))

    monkeypatch.setattr(provider, "_run", run)

    async def spawn_inside(argv, **kwargs):
        container = "nemo-gym-abc"
        docker_part, program = argv[: argv.index(container)], argv[argv.index(container) + 1 :]
        env_pairs = [docker_part[i + 1] for i, token in enumerate(docker_part) if token == "--env"]
        assert exec_user(docker_part) == str(NS_UID) and program[:3] == ["sh", "-c", docker_pty.LAUNCHER]
        ctty = ["-c"] if kwargs["use_pty"] else []
        cooked = [SH, "-c", COOKED_TERMINAL_SH, "sh"]
        inside = [*ns.enter(user=NS_UID), SETSID, *ctty, *cooked, ENV, *env_pairs, *program]
        return await docker_pty.spawn(inside, **kwargs)

    monkeypatch.setattr(provider, "_spawn_pty", spawn_inside)
    return provider, execs


async def ns_recorded_pid(ns: StandInContainer, pid_file: str) -> str:
    async with asyncio.timeout(10):
        while not (pid := (await ns.sh(f"cat {pid_file} 2>/dev/null"))[1].strip()):
            await asyncio.sleep(0.05)
    return pid


async def test_a_non_root_session_is_signalled_and_closed_under_dockers_default_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session of a non-root user (what Harbor's default user gives a session on a non-root image), in a
    stand-in container where the session runs as uid 1000, the root-side execs as uid 0 without ``CAP_SYS_PTRACE``
    (Docker's default capability set) and the session's process has ``PPid`` 0 for real. The pid read the provider
    used to run as root is refused there by the kernel (``/proc/<pid>/environ`` of another uid is ptrace-gated), so
    it runs as the session's user: ``send_signal`` reaches the foreground job of the interactive shell, ``close()``
    ends the shell and its background job through the root-side cleanup and removes the pid file, and on pipes a
    command session ends on ``SIGTERM``."""
    async with StandInContainer() as ns:
        provider, execs = _stand_in_provider(monkeypatch, ns)
        session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec(user=NS_UID))
        pid_file = docker_pty.pid_file_for(session.session_id)
        async with session:
            pid = await ns_recorded_pid(ns, pid_file)
            _, status, _ = await ns.sh(f"grep -E '^(PPid|Uid):' /proc/{pid}/status")
            assert "PPid:\t0" in status and f"Uid:\t{NS_UID}\t{NS_UID}" in status, status
            # The read as root under Docker's capabilities, what the provider ran before: refused by the kernel,
            # so every signal of a non-root session failed and close() left its shell running. Where the host sets
            # ``fs.protected_regular`` (Ubuntu: 2) root is already refused the read-write open of another user's
            # file in a sticky directory it owns, so the record reads as missing; where it is 0 the open succeeds
            # and the ptrace gate refuses the marker read. The gate itself is shown on the raw environ either way.
            rc, out, err = await ns.sh(pid_read_command(session))
            assert rc == 1 and (f"pid {pid} is not a process of PTY session" in err or "no usable pid in" in err), (
                out,
                err,
            )
            assert (await ns.sh(f"tr '\\0' '\\n' < /proc/{pid}/environ"))[0] != 0
            # The same read as the session's user: the record check_recorded_pid accepts, with the real PPid 0.
            rc, out, _ = await ns.sh(pid_read_command(session), user=NS_UID)
            assert rc == 0 and docker_pty.check_recorded_pid(out, mode="pty", session_id=session.session_id) == pid

            await session.write(b"sleep 300 & echo bg=$! __do''ne__\n")
            match = re.search(rb"bg=(\d+)", await read_until(session, b"__done__"))
            assert match, "no background job"
            job = match.group(1).decode()
            await session.write(b"sleep 30\n")
            await ns.until_foreground(pid, job=True)
            await session.send_signal("SIGINT")
            await ns.until_foreground(pid, job=False)
            await session.write(b"echo rc=$? __do''ne__\n")
            out = await read_until(session, b"__done__")
            assert b"rc=130" in out, out
            assert session._pid == pid and not session.closed
        # close(): the root-side cleanup ends the shell and its background job inside, and drops the pid file.
        await ns.until_gone(job)
        await ns.until_gone(pid)
        assert (await ns.sh(f"test -e {pid_file}"))[0] == 1
        reads = [argv for argv in execs if argv[-1].startswith("f=")]
        assert [exec_user(argv) for argv in reads] == [str(NS_UID)]
        kills = [argv for argv in execs if "kill" in argv[-1]]
        assert len(kills) == 2 and all(exec_user(argv) == "0" for argv in kills)

        # Pipe mode: no terminal, the signal goes to the process's group and the process; the command session ends.
        session = await provider.create_pty(
            _handle(shell="sh"), SandboxPtySpec(command="exec sleep 30", pty=False, user=NS_UID)
        )
        async with session:
            pid = await ns_recorded_pid(ns, docker_pty.pid_file_for(session.session_id))
            assert (await ns.stat_fields(pid))[5] == "-1"
            await session.send_signal("SIGTERM")
            assert await session.wait_exit(timeout_s=10) in (-signal.SIGTERM, 128 + signal.SIGTERM)
        await ns.until_gone(pid)


async def test_a_non_root_session_under_a_read_only_tmp_records_its_pid_under_dev_shm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stand-in container with a read-only ``/tmp`` (a read-only tmpfs mounted over it: what ``create.read_only:
    true`` without a tmpfs on ``/tmp`` gives a session): the uid-1000 session's launcher cannot write
    ``/tmp/.nemo-gym-pty-<id>.pid`` and records the pid under ``/dev/shm`` (the host's here, a world-writable tmpfs
    as a container's is), the read as the session's user finds it there, ``send_signal`` reaches the foreground job,
    and ``close()`` ends the shell and its background job through the root-side cleanup and removes the record."""
    async with StandInContainer() as ns:
        rc, _, err = await ns.sh("mount -t tmpfs -o ro tmpfs /tmp && ! touch /tmp/.probe 2>/dev/null")
        if rc != 0:
            pytest.skip(f"cannot mount a read-only /tmp in the stand-in container: {err.strip()}")
        assert (await ns.sh("touch /dev/shm/.nemo-gym-probe && rm /dev/shm/.nemo-gym-probe", user=NS_UID))[0] == 0
        provider, execs = _stand_in_provider(monkeypatch, ns)
        session = await provider.create_pty(_handle(shell="sh"), SandboxPtySpec(user=NS_UID))
        pid_file = docker_pty.pid_file_for(session.session_id)
        fallback = docker_pty.fallback_pid_file_for(session.session_id)
        async with session:
            pid = await ns_recorded_pid(ns, fallback)
            assert (await ns.sh(f"test -e {pid_file}"))[0] == 1
            _, status, _ = await ns.sh(f"grep -E '^(PPid|Uid):' /proc/{pid}/status")
            assert "PPid:\t0" in status and f"Uid:\t{NS_UID}\t{NS_UID}" in status, status
            await session.write(b"sleep 300 & echo bg=$! __do''ne__\n")
            match = re.search(rb"bg=(\d+)", await read_until(session, b"__done__"))
            assert match, "no background job"
            job = match.group(1).decode()
            await session.write(b"sleep 30\n")
            await ns.until_foreground(pid, job=True)
            await session.send_signal("SIGINT")
            await ns.until_foreground(pid, job=False)
            await session.write(b"echo rc=$? __do''ne__\n")
            assert b"rc=130" in await read_until(session, b"__done__")
            assert session._pid == pid
        await ns.until_gone(job)
        await ns.until_gone(pid)
        assert (await ns.sh(f"test -e {fallback}"))[0] == 1 and (await ns.sh(f"test -e {pid_file}"))[0] == 1
        reads = [argv for argv in execs if argv[-1].startswith("f=")]
        assert [exec_user(argv) for argv in reads] == [str(NS_UID)]
        kills = [argv for argv in execs if "kill" in argv[-1]]
        assert len(kills) == 2 and all(exec_user(argv) == "0" for argv in kills)
        assert kills[-1][-1].startswith(f"rm -f -- {pid_file} {fallback}; ")
