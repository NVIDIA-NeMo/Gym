# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive PTY sessions for the Docker provider.

A session is one ``docker exec -i [-t]`` client process. In PTY mode the client runs under a host
pseudo-terminal of its own, so the docker CLI sees a terminal, allocates one in the container
(``-t``), puts it in raw mode and forwards bytes both ways; the session reads and writes the master
side of that host pty, which :func:`spawn` sizes and puts in raw mode before the client starts, so that
what is written before the client is up waits for it unchanged rather than passing through a cooked line
discipline. The client reads its terminal size at start and again on ``SIGWINCH``, which
is how ``rows``/``cols`` and ``resize()`` reach the container's terminal. Pipe mode (``pty=False``)
runs the client with plain pipes, so stdout and stderr arrive separately.

Sessions live in this process, not in the sandbox: there is nothing to re-attach to from another
process, and ``run_detached`` runs attached (there is no proxy connection to spare). Signals are
delivered with ``kill`` through a second ``docker exec``: in PTY mode to the foreground process group of
the session's terminal (what a key press would reach), otherwise to the process group of the session's
process, whose pid the session's launcher writes under ``/tmp`` inside the container, or under ``/dev/shm`` when
``/tmp`` refuses the write (a read-only root filesystem, ``create.read_only: true``, without a tmpfs on ``/tmp``;
``/dev/shm`` is a writable tmpfs in every container that does not run with ``--ipc=none``). A session whose pid can
be recorded in neither place could be neither signalled nor cleaned up, so its launcher says so on stderr (the
terminal in PTY mode) and exits 1 instead of starting the program: ``create_pty`` itself returns as soon as the
client is spawned, without a round trip, so the failure shows as the session ending at once with that message.

Every in-container command here is one ``<shell> -c`` line run by the image's shell (bash when it has one, else sh,
which is busybox in Alpine-based images), so the commands stay POSIX and busybox-compatible. In particular ``kill``
is spelled ``kill -NAME -pgid``, the one form dash, bash and busybox all accept: busybox rejects
``kill -s NAME -- -pgid`` (``invalid number '--'``, exit 1 although the signal is delivered) and dash rejects
``kill -s NAME -pgid`` (``Illegal option``).

The pid file is written by the session's own user, so anything running as that user can rewrite it, or replace it
by a link to a file it does not own; nothing about the file identifies the process that wrote it. Its content is a
hint, never a kill target by itself: the pid it names is accepted only when the process provably is the one this
session's ``docker exec`` started, by two properties no process in the container can give another: its environment
carries this session's marker (``NEMO_GYM_PTY_SESSION=<session id>``, set by ``docker exec --env`` and inherited by
the program the launcher runs) and its parent is not in the container (``PPid`` 0 in the container's pid
namespace, true of a process the runtime injected and of nothing the agent can start). The file is read once
(:data:`PID_READ_COMMAND`, which waits up to two seconds for the launcher to write it: ``create_pty`` returns as
soon as the session's client is spawned, and a ``send_signal`` or ``close`` issued right away reaches the container
through a second ``docker exec`` that races the session's own to the daemon; a record that is not a regular file,
a FIFO or a device the session's user put at the path, is refused at once rather than opened, the file is opened on
a pinned descriptor and checked again there, so a swap of the path after the test cannot hand the read something
that blocks, at most 32 bytes are read, and the exec that reads it has a timeout of its own,
:data:`PID_READ_TIMEOUT_S`, so the session's user can make the record useless but cannot stall ``send_signal`` or
``close`` on it), as the session's own user and not as
root: ``/proc/<pid>/environ`` is ptrace-gated,
readable by the process's own uid or with ``CAP_SYS_PTRACE`` in its user namespace, which Docker's default capability
set does not grant, so a root-side read fails for every non-root session, while a read as the session's user reads
the session's process and can vouch for no process of another uid. The record is checked by
:func:`check_recorded_pid` and cached, and ``close()`` reuses the checked pid. The ``kill`` and the cleanup run as
root, so they reach the process whatever user the session runs as; no added capability is needed.
"""

import asyncio
import contextlib
import fcntl
import logging
import os
import pty
import re
import shlex
import signal
import struct
import termios
import tty
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxPtyError


LOGGER = logging.getLogger(__name__)

# The signals ``send_signal`` accepts: the same set execd's PTY supports, so a harness written against one provider
# behaves the same on the other.
SUPPORTED_SIGNALS = frozenset({"SIGINT", "SIGTERM", "SIGKILL", "SIGQUIT", "SIGHUP"})
PID_FILE_PREFIX = "/tmp/.nemo-gym-pty-"
# Where the launcher records the pid when ``/tmp`` refuses the write, as it does under a read-only root filesystem
# (``create.read_only: true``) without a tmpfs on ``/tmp``: ``/dev/shm`` is a writable tmpfs in every container Docker
# starts, ``--read-only`` or not, unless it runs with ``--ipc=none``.
FALLBACK_PID_FILE_PREFIX = "/dev/shm/.nemo-gym-pty-"
# How many tenths of a second the pid read waits for the launcher's record (two seconds, in whole seconds too where
# ``sleep`` takes integers only). ``create_pty()`` returns as soon as the session's ``docker exec`` client is spawned,
# and a ``send_signal()`` or ``close()`` issued right away reaches the container through a second exec that races the
# first to the daemon, so the record may not be written yet when the read starts.
PID_READ_TRIES = 20
# The timeout of the exec that reads the pid record: the two-second wait plus the exec's round trip, with room for a
# slow daemon, instead of ``exec.default_timeout_s`` (180 s shipped). Nothing the session's user can do to the record
# makes the read block (see ``PID_RECORD_READ``), so this is a backstop for the docker side, and the per-call cost of
# a session whose record is unusable stays at the two-second wait.
PID_READ_TIMEOUT_S = 30
# The environment variable ``docker exec --env`` sets on a session's process, its value the session id: the mark by
# which the recorded pid is known to be this session's (see the module docstring).
SESSION_ENV = "NEMO_GYM_PTY_SESSION"
# ``PPid`` of a process the runtime injected into the container: its parent lives outside the pid namespace. Tests
# that run the session's program locally point this at the test process.
EXEC_PARENT_PID = "0"
# The in-container launcher: record the shell's pid (kept by ``exec``) where ``send_signal`` and ``close`` find it,
# then become the session's program. ``$0`` is the pid file, ``$1`` the fallback pid file (under ``/dev/shm``, for a
# ``/tmp`` that refuses the write), the rest the program. A session whose pid can be recorded nowhere could be
# neither signalled nor cleaned up, so the launcher says so on stderr (the terminal in PTY mode) and exits 1 instead
# of starting the program.
LAUNCHER = (
    'f=$0; p=$1; shift; { printf "%s\\n" "$$" > "$f" || printf "%s\\n" "$$" > "$p"; } 2>/dev/null || '
    '{ printf "nemo-gym: cannot record the pid of this PTY session in %s or %s\\n" "$f" "$p" >&2; exit 1; }; '
    'exec "$@"'
)
# The shell a session without a command spawns: bash when the image has it, else sh (execd's own pick).
DEFAULT_SHELL_SNIPPET = 'exec "$(command -v bash || echo sh)"'
# The in-container ``kill`` of ``send_signal``. In PTY mode the recorded process leads the terminal's session and
# field 8 of ``/proc/<pid>/stat`` (``tpgid``) is the terminal's foreground process group: the program of a command
# session, or the job an interactive shell is running (the shell itself while idle), since a shell with job control
# puts each job in a group of its own, out of reach of a signal to the shell's group. Without a terminal (pipe mode)
# ``tpgid`` is -1 and the signal goes to the process's own group, best-effort (the process may lead none), then to
# the process. The ``comm`` field may contain spaces, so the line is cut after its closing parenthesis. ``{pid}`` is
# the pid :func:`check_recorded_pid` accepted.
SIGNAL_COMMAND = (
    "s=$(cat /proc/{pid}/stat 2>/dev/null); s=${{s##*) }}; set -- $s; tp=${{6:-0}}; "
    'if [ "$tp" -gt 0 ] 2>/dev/null; then kill -{name} -"$tp"; '
    "else kill -{name} -{pid} 2>/dev/null; kill -{name} {pid}; fi"
)
# One attempt at reading a record, ``$1``, in a way the session's user cannot make block: the path is opened
# read-write on a pinned descriptor (an ``O_RDWR`` open of a FIFO returns at once where a read-only one waits for a
# writer; a regular file of the session's own user is writable by it), the open file is tested on that descriptor
# (``/proc/self/fd/3`` names the file itself, so a FIFO, a device or a directory swapped in after the path was
# tested is still caught, and a regular file never blocks a read), and at most 32 bytes are read (a link to a huge
# file, or to a file that never ends, cannot grow the read without bound; a pid fits in far fewer). A redirection
# that fails on a brace group is not fatal to the shell, unlike one on ``exec``. The caller tests the path first
# (``[ -f "$1" ]``), so a missing record is never created by the open and a record that is not a regular file is
# refused before this runs; the pinned test closes the window between that test and the open.
PID_RECORD_READ = '{ [ -f /proc/self/fd/3 ] && head -c 32 <&3; } 3<>"$1"'
# The one read of the pid file, run as the session's user, not as root: ``/proc/<pid>/environ`` is readable by the
# process's own uid or with ``CAP_SYS_PTRACE`` in its user namespace, which Docker's default capabilities omit, so a
# root-side read of a non-root session's process fails with EACCES (and a read as the session's user cannot vouch
# for a process of another uid). The shell waits for a non-empty pid file, ``{pid_file}`` or ``{fallback}``, for up
# to ``{tries}`` tenths of a second (the launcher may not have written it yet; where ``sleep`` takes integers only
# the loop sleeps whole seconds and counts ten tries each), reading either only while it is a regular file and
# through ``PID_RECORD_READ`` (a path that exists as anything else, a FIFO or a link to a device say, is refused at
# once: the session's user can make its record useless but not make the read wait on it), refuses anything but a
# plain pid above 1 (``kill -1`` would reach every process in the container) and a pid whose process does not carry
# this session's marker in its environment (NUL-separated, matched whole), then prints the pid with the marker line,
# followed by ``/proc/<pid>/stat`` of that process for :func:`check_recorded_pid` (parent, session). ``{marker}`` is
# the quoted ``NEMO_GYM_PTY_SESSION=<session id>``.
PID_READ_COMMAND = (
    "f={pid_file}; g={fallback}; n=0; "
    'rd() {{ [ -f "$1" ] && ' + PID_RECORD_READ.replace("{", "{{").replace("}", "}}") + "; }} 2>/dev/null; "
    'until pid=$(rd "$f") && [ -n "$pid" ]; do '
    'if pid=$(rd "$g") && [ -n "$pid" ]; then break; fi; '
    'for q in "$f" "$g"; do if [ -e "$q" ] && ! [ -f "$q" ]; then '
    "printf 'pid record %s is not a regular file\\n' \"$q\" >&2; exit 1; fi; done; "
    '[ "$n" -lt {tries} ] || break; sleep 0.1 2>/dev/null || {{ sleep 1; n=$((n+9)); }}; n=$((n+1)); done; '
    'case "$pid" in \'\'|0|1|*[!0-9]*) printf \'no usable pid in %s or %s: %s\\n\' "$f" "$g" "$pid" >&2; exit 1;; esac; '
    "m=$(tr '\\0' '\\n' 2>/dev/null < \"/proc/$pid/environ\" | grep -m1 -x -F {marker}); "
    '[ -n "$m" ] || {{ printf \'pid %s is not a process of PTY session %s\\n\' "$pid" {session_id} >&2; exit 1; }}; '
    'printf \'%s %s\\n\' "$pid" "$m"; cat "/proc/$pid/stat"'
)
# ``close()`` of a live session, with the pid :func:`check_recorded_pid` accepted: drop the pid files, hang up the
# recorded process's group (a shell relays the hangup to its jobs, running and stopped, in their own groups), give
# it up to a second to do so (back to back, the SIGKILL below would end the shell before it relays anything and its
# background jobs would survive), then SIGKILL every process left in the recorded process's session (whatever the
# shell did not reach) and, for a process that leads no session, its group and itself. ``read`` rather than ``cat``
# keeps the sweep free of forks. Where ``sleep`` takes integers only (busybox without fractional sleep) the first
# iteration sleeps the whole second and ends the loop, so the grace is bounded by one second either way.
CLEANUP_COMMAND = (
    "rm -f -- {pid_files}; pid={pid}; "
    'kill -HUP -"$pid" "$pid" 2>/dev/null; '
    'n=0; while [ "$n" -lt 10 ] && {{ IFS= read -r s < "/proc/$pid/stat"; }} 2>/dev/null; do '
    's=${{s##*) }}; set -- $s; case "$1" in Z|X) break;; esac; '
    "sleep 0.1 2>/dev/null || {{ sleep 1; n=9; }}; n=$((n+1)); done; "
    'for p in /proc/[0-9]*; do {{ IFS= read -r s < "$p/stat"; }} 2>/dev/null || continue; s=${{s##*) }}; set -- $s; '
    '[ "$4" = "$pid" ] && kill -KILL "${{p#/proc/}}" 2>/dev/null; done; '
    'kill -KILL -"$pid" "$pid" 2>/dev/null; true'
)
READ_CHUNK = 65536
# How much of the docker client's own stderr a session keeps, to tell a client that could not start the program (the
# daemon's refusal is a line) from a program that exited; see ``DockerPtySession._startup_failure``.
CLIENT_STDERR_KEEP = 16384

# The in-container commands run through the provider at close and on a signal: as root by default, so the ``kill``
# and the cleanup reach a process of any user, and with ``as_session_user=True`` as the session's own user, which the
# pid read needs (``/proc/<pid>/environ`` is readable by the process's uid, not by a root without ``CAP_SYS_PTRACE``);
# ``timeout_s`` caps the exec's timeout below the provider's default (the pid read passes
# :data:`PID_READ_TIMEOUT_S`). The pid file is read once and cached by the session.
RunInContainer = Callable[..., Awaitable[SandboxExecResult]]


def pid_file_for(session_id: str) -> str:
    return f"{PID_FILE_PREFIX}{session_id}.pid"


def fallback_pid_file_for(session_id: str) -> str:
    """Where the launcher records the pid when :func:`pid_file_for`'s ``/tmp`` refuses the write."""
    return f"{FALLBACK_PID_FILE_PREFIX}{session_id}.pid"


def session_marker(session_id: str) -> str:
    """The ``KEY=value`` a session's ``docker exec`` sets in its process's environment."""
    return f"{SESSION_ENV}={session_id}"


def launcher_argv(shell: str, pid_file: str, fallback_pid_file: str, command: str | None) -> list[str]:
    """The program ``docker exec`` runs: the launcher under ``shell`` with its pid file and the fallback, then the
    shell itself or ``shell -c command``."""
    program = [shell, "-c", command] if command is not None else ["sh", "-c", DEFAULT_SHELL_SNIPPET]
    return [shell, "-c", LAUNCHER, pid_file, fallback_pid_file, *program]


def check_recorded_pid(record: str, *, mode: str, session_id: str) -> str:
    """The pid ``PID_READ_COMMAND`` printed, once it is safe to signal as root.

    The pid file belongs to the session's user, who can rewrite it or replace it by a link to any file, so a kill as
    root may only follow it to the process this session's ``docker exec`` started, told by what no process in the
    container can give another: a plain pid above 1 whose environment carries this session's marker (the line the
    shell matched, ``NEMO_GYM_PTY_SESSION=<session id>``; the read ran as the session's user, so the process is of
    that user's uid too), whose parent is outside the container (field 4 of
    ``/proc/<pid>/stat`` is :data:`EXEC_PARENT_PID`: a process the runtime injected, not one started inside, so not
    a process of the agent's own tree either) and, in PTY mode, which leads its session (field 6 is the pid: the
    session leader runc created for this exec). ``ValueError`` names what is wrong otherwise.
    """
    lines = record.splitlines()
    head = lines[0].split() if lines else []
    if len(head) != 2 or len(lines) < 2:
        raise ValueError(f"unexpected pid record {record.strip()[:200]!r}")
    pid, marker = head
    if not re.fullmatch(r"[0-9]+", pid) or int(pid) <= 1:
        raise ValueError(f"not a usable pid: {pid!r}")
    if marker != session_marker(session_id):
        raise ValueError(f"pid {pid} is not a process of PTY session {session_id} (marker {marker!r})")
    stat = lines[1].rpartition(") ")[2].split()
    if len(stat) < 6:
        raise ValueError(f"process {pid} is gone or unreadable")
    if stat[1] != EXEC_PARENT_PID:
        raise ValueError(
            f"process {pid} was not started by the runtime (parent {stat[1]}, expected {EXEC_PARENT_PID})"
        )
    if mode == "pty" and stat[3] != pid:
        raise ValueError(f"process {pid} does not lead its session (session {stat[3]})")
    return pid


def _winsize(rows: int, cols: int) -> bytes:
    return struct.pack("HHHH", rows, cols, 0, 0)


async def _read_fd(fd: int) -> bytes:
    """One chunk from a non-blocking fd; ``b""`` at EOF (a pty master reports EIO once every slave closed)."""
    loop = asyncio.get_running_loop()
    ready: asyncio.Future[bytes] = loop.create_future()

    def on_ready() -> None:
        try:
            data = os.read(fd, READ_CHUNK)
        except BlockingIOError:
            return
        except OSError:
            data = b""
        loop.remove_reader(fd)
        if not ready.done():
            ready.set_result(data)

    loop.add_reader(fd, on_ready)
    try:
        return await ready
    finally:
        loop.remove_reader(fd)


async def _write_fd(fd: int, data: bytes, *, released: asyncio.Future[None]) -> None:
    """Write all of ``data`` to a non-blocking fd without blocking the loop while the pty buffer is full.

    ``released`` is set when the session starts releasing the fd (``DockerPtySession._end_local``): a write
    waiting for room then stops with ``SandboxPtyError`` as soon as it runs again and never touches the fd after
    that, and its readiness callback is gone from the loop before the fd is closed. Otherwise the descriptor
    number, free once closed, goes to the next session's pty, and the write would resume into that session's
    terminal, or its stale registration would break that session's reads.
    """
    loop = asyncio.get_running_loop()
    view = memoryview(data)
    while view:
        if released.done():
            raise SandboxPtyError("PTY session closed before the write completed")
        try:
            view = view[os.write(fd, view) :]
            continue
        except BlockingIOError:
            pass
        writable: asyncio.Future[None] = loop.create_future()

        def on_writable() -> None:
            loop.remove_writer(fd)
            if not writable.done():
                writable.set_result(None)

        def on_released(_: asyncio.Future[None]) -> None:
            if not writable.done():
                writable.set_result(None)

        loop.add_writer(fd, on_writable)
        released.add_done_callback(on_released)
        try:
            await writable
        finally:
            loop.remove_writer(fd)
            released.remove_done_callback(on_released)


class DockerPtySession:
    """One live terminal: a ``docker exec`` client under a host pty (or on pipes). Created via :func:`spawn`."""

    def __init__(
        self,
        *,
        proc: asyncio.subprocess.Process,
        session_id: str,
        mode: str,
        master_fd: int | None,
        run_in_container: RunInContainer | None,
        pid_file: str | None,
        fallback_pid_file: str | None = None,
        runtime_failure: Callable[[str], bool] | None = None,
        client_stderr_fd: int | None = None,
    ) -> None:
        self._proc = proc
        self.session_id = session_id
        self.mode: str | None = mode
        self._master_fd = master_fd
        self._client_stderr_fd = client_stderr_fd
        self._run_in_container = run_in_container
        self._pid_file = pid_file
        self._fallback_pid_file = fallback_pid_file or pid_file
        self._pid: str | None = None
        self._output: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._stderr: asyncio.Queue[bytes | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        self._exit: asyncio.Future[int] = loop.create_future()
        # Set when ``_end_local`` starts releasing the master; the writes still waiting for room, each tracked by a
        # future that resolves once it has left ``_write_fd``, are failed through it and joined before the fd closes.
        self._released: asyncio.Future[None] = loop.create_future()
        self._writes: set[asyncio.Future[None]] = set()
        # One write at a time on the descriptor: the loop keeps a single readiness callback per fd, so a second
        # writer waiting for room would replace the first's registration and the first would never wake.
        self._write_lock = asyncio.Lock()
        self._closed = False
        self._abandoned = False
        self._runtime_failure = runtime_failure
        # The client's own stderr, its head kept (``CLIENT_STDERR_KEEP``) for ``_startup_failure``, and the end of it.
        self._client_stderr = bytearray()
        self._client_stderr_eof: asyncio.Future[None] = loop.create_future()
        self._tasks: list[asyncio.Task[None]] = [asyncio.create_task(self._wait())]
        if master_fd is not None:
            self._tasks.append(asyncio.create_task(self._pump_fd(master_fd, self._output, close_stderr=True)))
            # The terminal carries the program's output, its stderr included; the client's stderr is its own.
            if client_stderr_fd is not None:
                self._tasks.append(asyncio.create_task(self._pump_fd(client_stderr_fd, None, keep=True)))
        else:
            self._tasks.append(asyncio.create_task(self._pump_stream(proc.stdout, self._output)))
            self._tasks.append(asyncio.create_task(self._pump_stream(proc.stderr, self._stderr, keep=True)))

    @property
    def closed(self) -> bool:
        """True once the session can no longer run commands: after ``close()`` or once the process exited."""
        return self._closed or self._exit.done()

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    async def _wait(self) -> None:
        code = await self._proc.wait()
        # The client's stderr ends with the client, and what it said tells a refused exec from a program's exit.
        # Bounded: a child of a local stand-in program may hold the pipe open (the docker client has no children).
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(self._client_stderr_eof), timeout=1.0)
        if self._exit.done():
            return
        failure = self._startup_failure(code)
        if failure is None:
            self._exit.set_result(code)
        else:
            self._exit.set_exception(SandboxPtyError(failure))
            self._exit.exception()  # retrieved; silences never-retrieved warnings

    def _startup_failure(self, code: int) -> str | None:
        """Why the client exited without having run the program, or ``None`` when ``code`` is the program's exit.

        ``docker exec`` exits 1 with the daemon's refusal on its stderr when the container is paused or gone or the
        daemon is unreachable, a code a program may exit with too, so the exit is told by what the client wrote: the
        lines the provider's ``exec()`` recognises as a runtime failure (``runtime_failure``), which make that call
        return its runtime sentinel with ``error_type="sandbox"``. Here they end the session with ``SandboxPtyError``
        from ``wait_exit()`` and ``read()`` rather than pass the client's code off as the program's. In PTY mode the
        client's stderr is its own (the program's goes to the terminal); in pipe mode it is the program's stderr
        too, so a program's complaint that reads like the daemon's is classified as ``exec()`` would classify it.
        """
        if code == 0 or self._runtime_failure is None or not self._client_stderr:
            return None
        text = self._client_stderr.decode(errors="replace")
        if not self._runtime_failure(text):
            return None
        return f"PTY session {self.session_id} could not be started (docker exec exited {code}): {text.strip()[:200]}"

    def _keep_client_stderr(self, chunk: bytes) -> None:
        """Retain the head of the client's own stderr (``CLIENT_STDERR_KEEP``) for :meth:`_startup_failure`."""
        if len(self._client_stderr) < CLIENT_STDERR_KEEP:
            self._client_stderr += chunk[: CLIENT_STDERR_KEEP - len(self._client_stderr)]

    async def _pump_fd(
        self, fd: int, queue: asyncio.Queue[bytes | None] | None, *, close_stderr: bool = False, keep: bool = False
    ) -> None:
        """Feed the non-blocking ``fd`` (the pty master, or the pipe of the client's own stderr in PTY mode, which
        no reader sees: ``queue`` is ``None`` and ``keep`` retains its head and marks its end) into ``queue``."""
        try:
            while chunk := await _read_fd(fd):
                if keep:
                    self._keep_client_stderr(chunk)
                if queue is not None:
                    await queue.put(chunk)
        finally:
            if keep and not self._client_stderr_eof.done():
                self._client_stderr_eof.set_result(None)
            if queue is not None:
                # The pty is gone: whatever the process was, it cannot write any more; let the exit settle first so
                # a reader that sees EOF also sees the exit code.
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.shield(self._exit), timeout=5.0)
                await queue.put(None)
                if close_stderr:
                    await self._stderr.put(None)

    async def _pump_stream(
        self, stream: asyncio.StreamReader | None, queue: asyncio.Queue[bytes | None], *, keep: bool = False
    ) -> None:
        """Feed ``stream`` into ``queue``; ``keep`` (the client's stderr in pipe mode, the program's too) retains
        its head for :meth:`_startup_failure` and marks its end."""
        try:
            if stream is not None:
                while chunk := await stream.read(READ_CHUNK):
                    if keep:
                        self._keep_client_stderr(chunk)
                    await queue.put(chunk)
        finally:
            if keep and not self._client_stderr_eof.done():
                self._client_stderr_eof.set_result(None)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self._exit), timeout=5.0)
            await queue.put(None)

    def _close_error(self) -> SandboxPtyError:
        return SandboxPtyError("PTY session closed before process exit")

    async def _read_stream(self, queue: asyncio.Queue[bytes | None], timeout_s: float | None) -> bytes:
        chunk = await asyncio.wait_for(queue.get(), timeout=timeout_s)
        if chunk is None:
            # Keep the EOF observable by subsequent reads and iterators.
            await queue.put(None)
            if self._exit.done() and self._exit.exception() is None:
                return b""
            raise self._exit.exception() if self._exit.done() else self._close_error()
        return chunk

    async def read(self, *, timeout_s: float | None = None) -> bytes:
        return await self._read_stream(self._output, timeout_s)

    async def read_stderr(self, *, timeout_s: float | None = None) -> bytes:
        return await self._read_stream(self._stderr, timeout_s)

    def __aiter__(self) -> AsyncIterator[bytes]:
        async def _iterate() -> AsyncIterator[bytes]:
            while chunk := await self.read():
                yield chunk

        return _iterate()

    async def write(self, data: bytes) -> None:
        async with self._write_lock:
            await self._write_unlocked(data)

    async def _write_unlocked(self, data: bytes) -> None:
        # The closed check comes after the lock: a write queued behind one that the release failed must not touch
        # the descriptor, which may already be the next session's.
        if self.closed:
            raise SandboxPtyError("PTY session is closed")
        try:
            if self._master_fd is not None:
                pending: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self._writes.add(pending)
                try:
                    await _write_fd(self._master_fd, data, released=self._released)
                finally:
                    self._writes.discard(pending)
                    pending.set_result(None)
            else:
                assert self._proc.stdin is not None
                self._proc.stdin.write(data)
                await self._proc.stdin.drain()
        except (OSError, ConnectionError) as e:
            raise SandboxPtyError(f"PTY connection lost: {e}") from e

    async def resize(self, rows: int, cols: int) -> None:
        """Resize the terminal: the host pty's size, then ``SIGWINCH`` to the docker client, which resizes the
        container's. Pipe mode has no terminal and ignores it."""
        if self._master_fd is None:
            return
        if self.closed:
            raise SandboxPtyError("PTY session is closed")
        fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, _winsize(rows, cols))
        with contextlib.suppress(ProcessLookupError):
            os.kill(self._proc.pid, signal.SIGWINCH)

    def _pid_files(self) -> str:
        """The pid file and its fallback, quoted for one shell line (the one file when the two are the same)."""
        assert self._pid_file is not None
        files = [self._pid_file]
        if self._fallback_pid_file not in (None, self._pid_file):
            files.append(self._fallback_pid_file)
        return " ".join(shlex.quote(f) for f in files)

    async def _session_pid(self) -> str:
        """The pid the launcher recorded, read once through the provider as the session's user (waiting up to two
        seconds for the launcher to record it, refusing at once a record that is not a regular file, under an exec
        timeout of :data:`PID_READ_TIMEOUT_S`) and checked (:func:`check_recorded_pid`)."""
        if self._pid is not None:
            return self._pid
        if self._run_in_container is None or self._pid_file is None:
            raise SandboxPtyError("This PTY session cannot signal its process: it was spawned without a sandbox")
        command = PID_READ_COMMAND.format(
            pid_file=shlex.quote(self._pid_file),
            fallback=shlex.quote(self._fallback_pid_file or self._pid_file),
            tries=PID_READ_TRIES,
            marker=shlex.quote(session_marker(self.session_id)),
            session_id=shlex.quote(self.session_id),
        )
        result = await self._run_in_container(command, as_session_user=True, timeout_s=PID_READ_TIMEOUT_S)
        problem = (result.stderr or result.stdout or "").strip()[:200] if result.return_code != 0 else None
        if problem is None:
            try:
                self._pid = check_recorded_pid(
                    result.stdout or "", mode=self.mode or "pipe", session_id=self.session_id
                )
            except ValueError as e:
                problem = str(e)
        if problem is not None:
            raise SandboxPtyError(
                f"PTY session {self.session_id} has no usable recorded pid ({self._pid_file}): {problem}"
            )
        return self._pid

    async def send_signal(self, signal: str) -> None:
        if signal not in SUPPORTED_SIGNALS:
            raise ValueError(f"Docker PTY supports only {sorted(SUPPORTED_SIGNALS)}, got {signal!r}")
        if self.closed:
            raise SandboxPtyError("PTY session is closed")
        pid = await self._session_pid()
        # The terminal's foreground process group in PTY mode (the running job of an interactive shell included),
        # the process's own group then the process otherwise; see ``SIGNAL_COMMAND``. The process decides the exit.
        result = await self._run_in_container(SIGNAL_COMMAND.format(pid=pid, name=signal.removeprefix("SIG")))
        if result.return_code != 0:
            # A kill that races the exit of its target fails although the signal did its work: let the exit settle.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self._exit), timeout=1.0)
            if not self._exit.done():
                raise SandboxPtyError(
                    f"Could not signal PTY session {self.session_id}: "
                    f"{(result.stderr or result.stdout or '').strip()[:200]}"
                )

    async def wait_exit(self, *, timeout_s: float | None = None) -> int:
        # shield: the future is shared; a timed-out waiter must not cancel it.
        return await asyncio.wait_for(asyncio.shield(self._exit), timeout=timeout_s)

    async def run_detached(self, command: str, *, poll_interval_s: float = 15.0) -> tuple[bytes, int | None]:
        """Run one command and return ``(merged output, exit code or None)``.

        A Docker session is a local client process, so there is no connection to spare between polls: the
        command runs attached, with the same marker discipline as session exec. ``poll_interval_s`` is accepted
        for the protocol and unused.
        """
        token = f"NGPTY{uuid.uuid4().hex[:12]}"
        needle = f"{token}:".encode()
        # Markers split across two printf arguments so the shell's echo can never match them; both printfs
        # inside the brace group so the slice between them is the command's output alone; stdin at EOF.
        await self.write(
            f"{{ printf '%s%s\\n' '{token[:5]}' '{token[5:]}S'\n"
            f"{command}\n"
            f"printf '%s%s:%s\\n' '{token[:5]}' '{token[5:]}' \"$?\"\n"
            f"}} </dev/null\n".encode()
        )
        buffer = bytearray()
        while needle not in buffer:
            chunk = await self.read()
            if not chunk:
                raise SandboxPtyError("PTY session ended before the command finished")
            buffer.extend(chunk)
        output, _, trailing = bytes(buffer).partition(needle)
        _, seen_start, after_start = output.partition(f"{token}S".encode())
        if seen_start:
            output = after_start.partition(b"\n")[2]
        while b"\n" not in trailing:
            chunk = await self.read(timeout_s=5.0)
            if not chunk:
                break
            trailing += chunk
        exit_text = trailing.split(b"\n", 1)[0].strip()
        stderr = bytearray()
        try:
            while chunk := await self.read_stderr(timeout_s=0.05):
                stderr.extend(chunk)
        except (TimeoutError, asyncio.TimeoutError):
            pass
        return bytes(output + stderr), int(exit_text) if exit_text.isdigit() else None

    async def _end_local(self) -> None:
        """Stop the docker client and release the pty; the in-container process is the caller's business.

        A write still waiting for room in the pty is failed first and given the turn it needs to leave (see
        :func:`_write_fd`), so the master is closed with no writer and no readiness callback left on its descriptor
        number: a session opened right after may get that number for its own pty.
        """
        if not self._released.done():
            self._released.set_result(None)
        if self._writes:
            await asyncio.wait(list(self._writes), timeout=5.0)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._proc.wait(), timeout=10.0)
        if self._master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._master_fd)
            self._master_fd = None
            if self._client_stderr_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(self._client_stderr_fd)
                self._client_stderr_fd = None
        elif self._proc.stdin is not None:
            with contextlib.suppress(Exception):
                self._proc.stdin.close()
        await self._output.put(None)
        await self._stderr.put(None)

    async def abandon(self) -> None:
        """Release the session when its container is going away: no in-container cleanup is attempted."""
        if self._closed:
            return
        self._closed = True
        self._abandoned = True
        if not self._exit.done():
            self._exit.set_exception(self._close_error())
            self._exit.exception()  # retrieved; silences never-retrieved warnings
        await self._end_local()

    async def close(self) -> None:
        """Idempotent: end the process in the container when it still runs, stop the client, drop the pid files.

        A docker client that goes away does not end the process it exec'd (the daemon keeps it until it exits),
        so a live session's process is ended through a second exec (``CLEANUP_COMMAND``: ``SIGHUP`` to the group of
        the recorded pid, up to a second for a shell to relay it to its jobs, then ``SIGKILL`` to every process of
        that pid's session), by the pid its launcher recorded and :func:`check_recorded_pid` accepted; a session that
        exited only has its pid files removed. Both are best-effort: a sandbox that is already gone cannot be reached
        and needs no cleanup, and a session whose recorded pid cannot be trusted is left to the container's removal.
        """
        if self._closed:
            return
        self._closed = True
        running = self._proc.returncode is None
        if not self._exit.done():
            self._exit.set_exception(self._close_error())
            self._exit.exception()  # retrieved; silences never-retrieved warnings
        try:
            if self._run_in_container is not None and self._pid_file is not None:
                pid_files = self._pid_files()
                cleanup = f"rm -f -- {pid_files}"
                if running:
                    try:
                        cleanup = CLEANUP_COMMAND.format(pid_files=pid_files, pid=await self._session_pid())
                    except Exception:
                        LOGGER.warning(
                            "PTY session %s: no usable recorded pid, its process is left to the container",
                            self.session_id,
                            exc_info=True,
                        )
                try:
                    await self._run_in_container(cleanup)
                except Exception:
                    LOGGER.debug("PTY session %s: in-container cleanup failed", self.session_id, exc_info=True)
        finally:
            await self._end_local()

    async def __aenter__(self) -> "DockerPtySession":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


async def spawn(
    argv: list[str],
    *,
    use_pty: bool,
    rows: int,
    cols: int,
    session_id: str | None = None,
    run_in_container: RunInContainer | None = None,
    pid_file: str | None = None,
    fallback_pid_file: str | None = None,
    runtime_failure: Callable[[str], bool] | None = None,
) -> DockerPtySession:
    """Start ``argv`` as a PTY session: under a fresh host pty sized ``rows``x``cols`` in PTY mode, on pipes otherwise.

    The Docker provider passes a ``docker exec`` argv; a test may pass any local program, which exercises the same
    terminal plumbing without a daemon. ``runtime_failure`` tells, from the client's own stderr, a client that
    exited without having started the program (the provider passes its ``exec()`` classifier): such an exit ends
    the session with ``SandboxPtyError`` instead of standing as the program's exit code (``_startup_failure``).
    """
    session_id = session_id or uuid.uuid4().hex
    if not use_pty:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        return DockerPtySession(
            proc=proc,
            session_id=session_id,
            mode="pipe",
            master_fd=None,
            run_in_container=run_in_container,
            pid_file=pid_file,
            fallback_pid_file=fallback_pid_file,
            runtime_failure=runtime_failure,
        )
    master_fd, slave_fd = pty.openpty()
    # The client's own stderr: a plain pipe read like the master (not an asyncio pipe of the process, whose
    # ``wait()`` would then also wait for the pipe, which a child of a local stand-in program may hold open).
    stderr_read, stderr_write = os.pipe()
    try:
        # Sized before the client starts: the docker CLI reads its terminal's size at start and applies it to the
        # container's terminal, so a command that reads the size immediately sees the requested one.
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, _winsize(rows, cols))
        # Raw before the client starts, too. The docker CLI puts its terminal in raw mode itself once it runs, but
        # ``create_pty`` returns as soon as the client is spawned and the session takes writes at once; until the
        # client got that far the host pty's line discipline would be the cooked default, and bytes written in that
        # window would be echoed back into the session's output, acted on by the host side (``^C``, ``^U``, ``^D``)
        # instead of reaching the container, and dropped past the canonical 4095-byte line. Raw, the pty holds them
        # for the client, byte for byte (the master's write waits while the buffer is full, see ``_write_fd``). The
        # CLI's own raw-mode switch and its restore at exit are no-ops on a terminal that is raw already.
        tty.setraw(slave_fd, termios.TCSANOW)
        os.set_blocking(master_fd, False)
        os.set_blocking(stderr_read, False)
        # The client's stdin and stdout are the terminal; its stderr is a pipe of the session's, since with ``-t``
        # the docker CLI writes the program's whole output, stderr included, to its stdout and keeps its own
        # messages, the daemon's refusal of the exec among them, for stderr (see ``_startup_failure``).
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=stderr_write,
            start_new_session=True,
        )
    except BaseException:
        for fd in (master_fd, slave_fd, stderr_read, stderr_write):
            os.close(fd)
        raise
    # Only the child holds the slave and the pipe's write end now: once it exits, reads of the master report EOF.
    os.close(slave_fd)
    os.close(stderr_write)
    return DockerPtySession(
        proc=proc,
        session_id=session_id,
        mode="pty",
        master_fd=master_fd,
        run_in_container=run_in_container,
        pid_file=pid_file,
        fallback_pid_file=fallback_pid_file,
        runtime_failure=runtime_failure,
        client_stderr_fd=stderr_read,
    )
