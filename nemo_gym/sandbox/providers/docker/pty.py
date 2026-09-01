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

"""Interactive PTY sessions for the docker provider.

A session is a tiny in-container python3 "broker" started with ``docker exec
-d``. The broker allocates a real PTY (or plain pipes in pipe mode), runs the
interactive shell / command on it, and mirrors everything through files under
``/tmp/.nemo-gym-pty/<session_id>/`` inside the container:

- ``ngpty.py``     this module's helper program, staged at create time
- ``broker.pid`` / ``child.pid``  the broker and its shell/command process
- ``stdin`` / ``control``  FIFOs carrying input and resize/signal requests
- ``output.log``   append-only merged terminal output (stdout in pipe mode)
- ``stderr.log``   append-only stderr (pipe mode only)
- ``exit_code``    written once when the process exits
- ``owner``        the current holder's token (empty = released)
- ``meta.json`` / ``ready`` / ``broker.log``  bookkeeping and diagnostics

Because all state lives in the container, sessions survive the creating
process: any process that shares the docker daemon can ``connect()`` to the
container and ``attach_pty()`` by session id. The client side drives the
session with short-lived ``docker exec`` calls: ``write()`` appends to the
stdin FIFO, ``read()`` polls ``output.log`` from a tracked byte offset (which
is also what makes ``attach(since=...)`` replay work), ``resize()`` and
``send_signal()`` go through the control FIFO, and ``wait_exit()`` polls the
exit-code file.

Requirements and simplifications vs the OpenSandbox execd backend:

- ``python3`` must exist in the sandbox image; ``create_pty`` fails with a
  clear error otherwise (SWE-bench task images all carry python3). No tmux or
  other terminal multiplexer is required.
- Retained output is the session's *entire* output log (bounded only by
  container disk), not execd's ~1 MiB ring. ``attach(since=0)`` therefore
  replays everything, and ``run_detached`` can never lose output between
  polls, so the "retained window exceeded" error path of the protocol never
  triggers here.
- There is no live connection, so "attached" is a token, not a socket: the
  ``owner`` file holds the current holder's token. ``attach(takeover=True)``
  rotates it and the evicted client's next ``read``/``write``/``poll`` sees
  the rotation and raises ``SandboxPtyError``. Without takeover, attaching to
  a held session raises; it succeeds only after the holder released it (an
  attached — non-creator — session's ``close()`` releases; the creator's
  ``close()`` ends the session outright, matching the protocol).
- Eviction is detected on the evicted client's *next* operation (there is no
  push channel), and output is delivered with the client's polling latency
  (~100 ms) rather than streamed.
- The broker stops draining output ~1 s after process exit even if a
  background grandchild still holds the terminal open, mirroring execd's
  "clean EOF means the stream drained, not that every byte was delivered".
- ``since`` replay offsets index ``output.log`` (the merged stream in PTY
  mode, stdout in pipe mode); a pipe-mode attach always tails stderr live.
- ``rows``/``cols`` are applied at spawn (before the command runs), so unlike
  execd there is no window where the command observes the backend default.
"""

import asyncio
import base64
import json
import shlex
import signal as signal_module
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from nemo_gym.sandbox.providers.base import SandboxPtyError


if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from nemo_gym.sandbox.providers.docker.provider import DockerProvider


PTY_STATE_ROOT = "/tmp/.nemo-gym-pty"
# Not "pty.py": the script's own directory is sys.path[0], so that name would
# shadow the stdlib `pty` module the broker imports.
HELPER_FILENAME = "ngpty.py"

# Helper exit codes (chosen away from shell/docker conventions).
EXIT_EVICTED = 90  # owner token no longer matches: another client took over
EXIT_DEAD = 91  # session state or broker is gone
EXIT_HELD = 92  # attach without takeover while another client holds the session
EXIT_NO_PYTHON3 = 97  # staging: python3 missing from the image

READ_POLL_INTERVAL_S = 0.1
POLL_MAX_BYTES = 262144
READY_DEADLINE_S = 30.0
_WRITE_EXTRA_TIMEOUT_S = 15.0  # helper-side FIFO write deadline; give the CLI at least this long

# The in-container helper program. Kept as a string so coverage and import
# machinery never see it; it targets the *image's* python3 (>= 3.6: no
# walrus, no dataclasses, no X | Y annotations).
HELPER_SOURCE = r'''
"""NeMo Gym docker PTY session helper. Runs inside the sandbox container."""
import base64
import errno
import fcntl
import json
import os
import select
import shutil
import signal
import struct
import sys
import time

EXIT_EVICTED = 90
EXIT_DEAD = 91
EXIT_HELD = 92


def _read_text(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _check_token(d, token):
    owner = _read_text(os.path.join(d, "owner"))
    if owner is None:
        sys.exit(EXIT_DEAD)
    if owner != token:
        sys.exit(EXIT_EVICTED)


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _exit_code(d):
    text = _read_text(os.path.join(d, "exit_code"))
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def _broker_alive(d):
    text = _read_text(os.path.join(d, "broker.pid"))
    if text is None:
        return False
    try:
        os.kill(int(text.strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def op_poll(d, token, out_off, err_off, max_bytes):
    _check_token(d, token)

    def _slice(name, off):
        path = os.path.join(d, name)
        size = _size(path)
        data = b""
        if max_bytes > 0 and size > off:
            with open(path, "rb") as f:
                f.seek(off)
                data = f.read(max_bytes)
        return data, size

    out, out_size = _slice("output.log", out_off)
    err, err_size = _slice("stderr.log", err_off)
    print(json.dumps({
        "out": base64.b64encode(out).decode("ascii"),
        "out_size": out_size,
        "err": base64.b64encode(err).decode("ascii"),
        "err_size": err_size,
        "exit": _exit_code(d),
        "alive": _broker_alive(d),
    }))


def _open_fifo_writer(path):
    try:
        return os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as e:
        if e.errno in (errno.ENXIO, errno.ENOENT):
            sys.exit(EXIT_DEAD)
        raise


def _fifo_write_all(fd, data, deadline_s=10.0):
    end = time.time() + deadline_s
    view = memoryview(data)
    while view:
        try:
            n = os.write(fd, view[:4096])
            view = view[n:]
        except OSError as e:
            if e.errno != errno.EAGAIN:
                sys.exit(EXIT_DEAD)
            if time.time() >= end:
                sys.exit(EXIT_DEAD)
            time.sleep(0.05)


def op_write(d, token, fifo_name):
    _check_token(d, token)
    data = sys.stdin.buffer.read()
    fd = _open_fifo_writer(os.path.join(d, fifo_name))
    try:
        _fifo_write_all(fd, data)
    finally:
        os.close(fd)


def op_attach(d, new_token, takeover):
    owner_path = os.path.join(d, "owner")
    owner = _read_text(owner_path)
    if owner is None or _read_text(os.path.join(d, "meta.json")) is None:
        sys.exit(EXIT_DEAD)
    if owner and not takeover:
        sys.exit(EXIT_HELD)
    _write_atomic(owner_path, new_token)
    meta = json.loads(_read_text(os.path.join(d, "meta.json")))
    print(json.dumps({
        "mode": meta.get("mode"),
        "out_size": _size(os.path.join(d, "output.log")),
        "err_size": _size(os.path.join(d, "stderr.log")),
        "exit": _exit_code(d),
        "alive": _broker_alive(d),
    }))


def op_release(d, token):
    owner_path = os.path.join(d, "owner")
    if _read_text(owner_path) == token:
        _write_atomic(owner_path, "")


def _kill_from_pidfile(d, name, whole_group):
    text = _read_text(os.path.join(d, name))
    if text is None:
        return
    try:
        pid = int(text.strip())
    except ValueError:
        return
    try:
        if whole_group:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def op_end(d):
    _kill_from_pidfile(d, "child.pid", True)
    _kill_from_pidfile(d, "broker.pid", False)
    shutil.rmtree(d, ignore_errors=True)


def _status_to_exit_code(status):
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 128


def _handle_ctl(line, master, child_pid):
    try:
        msg = json.loads(line.decode("utf-8", "replace"))
    except ValueError:
        return
    kind = msg.get("type")
    if kind == "resize" and master is not None:
        try:
            import termios
            winsz = struct.pack("HHHH", int(msg["rows"]), int(msg["cols"]), 0, 0)
            fcntl.ioctl(master, termios.TIOCSWINSZ, winsz)
        except (OSError, KeyError, TypeError, ValueError):
            pass
    elif kind == "signal":
        signum = getattr(signal, str(msg.get("signal", "")), None)
        if signum is None:
            return
        try:
            os.killpg(os.getpgid(child_pid), signum)
        except OSError:
            try:
                os.kill(child_pid, signum)
            except OSError:
                pass


def op_broker(d, shell, rows, cols, mode, command):
    log = open(os.path.join(d, "broker.log"), "ab", 0)
    os.dup2(log.fileno(), 2)
    _write_atomic(os.path.join(d, "broker.pid"), str(os.getpid()))
    for name in ("stdin", "control"):
        path = os.path.join(d, name)
        if not os.path.exists(path):
            os.mkfifo(path)
    argv = [shell, "-c", command] if command is not None else [shell]
    out_log = open(os.path.join(d, "output.log"), "ab", 0)
    use_pty = mode == "pty"
    master = None
    if use_pty:
        import pty
        import termios
        pid, master = pty.fork()
        if pid == 0:
            try:
                os.execvp(argv[0], argv)
            except OSError:
                os._exit(127)
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        read_fds = {master: out_log}
        stdin_target = master
    else:
        import subprocess
        err_log = open(os.path.join(d, "stderr.log"), "ab", 0)
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        pid = proc.pid
        read_fds = {proc.stdout.fileno(): out_log, proc.stderr.fileno(): err_log}
        stdin_target = proc.stdin.fileno()
    _write_atomic(os.path.join(d, "child.pid"), str(pid))
    stdin_fd = os.open(os.path.join(d, "stdin"), os.O_RDWR | os.O_NONBLOCK)
    ctl_fd = os.open(os.path.join(d, "control"), os.O_RDWR | os.O_NONBLOCK)
    _write_atomic(os.path.join(d, "ready"), "1")

    ctl_buf = b""
    exit_code = None
    quiet_rounds = 0
    while True:
        try:
            ready, _, _ = select.select(list(read_fds) + [stdin_fd, ctl_fd], [], [], 0.2)
        except InterruptedError:
            continue
        got_output = False
        for fd in list(read_fds):
            if fd not in ready:
                continue
            try:
                data = os.read(fd, 65536)
            except OSError:
                data = b""  # EIO: pty slave side fully closed
            if data:
                read_fds[fd].write(data)
                got_output = True
            else:
                del read_fds[fd]
        if stdin_fd in ready:
            try:
                data = os.read(stdin_fd, 65536)
            except OSError:
                data = b""
            if data:
                try:
                    view = memoryview(data)
                    while view:
                        view = view[os.write(stdin_target, view) :]
                except OSError:
                    pass  # process side gone; drop input
        if ctl_fd in ready:
            try:
                ctl_buf += os.read(ctl_fd, 65536)
            except OSError:
                pass
            while b"\n" in ctl_buf:
                line, ctl_buf = ctl_buf.split(b"\n", 1)
                if line.strip():
                    _handle_ctl(line, master, pid)
        if exit_code is None:
            try:
                wpid, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                wpid, status = pid, 0
            if wpid == pid:
                exit_code = _status_to_exit_code(status)
        if exit_code is not None:
            quiet_rounds = 0 if got_output else quiet_rounds + 1
            if not read_fds or quiet_rounds >= 5:
                break
    _write_atomic(os.path.join(d, "exit_code"), str(exit_code))


def main():
    op = sys.argv[1]
    if op == "broker":
        d, shell, rows, cols, mode = sys.argv[2:7]
        command = sys.argv[7] if len(sys.argv) > 7 else None
        op_broker(d, shell, int(rows), int(cols), mode, command)
    elif op == "poll":
        d, token, out_off, err_off, max_bytes = sys.argv[2:7]
        op_poll(d, token, int(out_off), int(err_off), int(max_bytes))
    elif op == "write":
        op_write(sys.argv[2], sys.argv[3], "stdin")
    elif op == "ctl":
        op_write(sys.argv[2], sys.argv[3], "control")
    elif op == "attach":
        op_attach(sys.argv[2], sys.argv[3], sys.argv[4] == "1")
    elif op == "release":
        op_release(sys.argv[2], sys.argv[3])
    elif op == "end":
        op_end(sys.argv[2])
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
'''


def _session_dir(session_id: str) -> str:
    return f"{PTY_STATE_ROOT}/{session_id}"


def _helper_path(session_id: str) -> str:
    return f"{_session_dir(session_id)}/{HELPER_FILENAME}"


def staging_script(session_id: str, token: str, mode: str) -> str:
    """Shell script that stages the helper (fed on stdin) and seeds session state."""
    d = _session_dir(session_id)
    meta = json.dumps({"mode": mode})
    return (
        f"command -v python3 >/dev/null 2>&1 || exit {EXIT_NO_PYTHON3}\n"
        f"mkdir -p {d} && cat > {d}/{HELPER_FILENAME} && "
        f"printf %s {shlex.quote(token)} > {d}/owner && "
        f"printf %s {shlex.quote(meta)} > {d}/meta.json && "
        f"chmod -R 777 {d}"
    )


class DockerPtySession:
    """One docker PTY session, driven via short-lived ``docker exec`` calls.

    Created by ``DockerProvider.create_pty`` / ``attach_pty``. All methods are
    safe to call from the creating process or any process that rebuilt the
    handle via ``connect()``.
    """

    def __init__(
        self,
        *,
        provider: "DockerProvider",
        container_name: str,
        session_id: str,
        token: str,
        mode: str,
        owned: bool,
        out_offset: int = 0,
        err_offset: int = 0,
        exit_code: int | None = None,
    ) -> None:
        self._provider = provider
        self._container_name = container_name
        self.session_id = session_id
        self._token = token
        self.mode: str | None = mode
        self._owned = owned
        self._out_offset = out_offset
        self._err_offset = err_offset
        self._exit_code = exit_code
        self._closed = False
        self._dead = False  # evicted, broker died, or exited-and-drained
        # Serialize offset bookkeeping so concurrent reads never return
        # duplicate bytes (each read atomically fetches-from-offset+advances).
        self._read_lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        """True once the session can no longer run commands: after ``close()``,
        after eviction by a takeover, or once the process exited and the
        output was drained."""
        return self._closed or self._dead

    async def _helper(
        self, args: list[str], *, stdin: bytes | None = None, extra_timeout_s: float = 0.0
    ) -> tuple[int, str, str]:
        """Run one helper op inside the container, mapping helper exit codes."""
        provider = self._provider
        argv = [provider._binary, "exec"]
        if stdin is not None:
            argv.append("-i")
        argv += [self._container_name, "python3", _helper_path(self.session_id), *args]
        timeout_s = provider._exec_config.default_timeout_s
        if timeout_s is not None:
            timeout_s += extra_timeout_s
        try:
            code, out, err = await provider._run(argv, timeout_s=timeout_s, stdin=stdin)
        except TimeoutError as e:
            raise SandboxPtyError(f"PTY session {self.session_id} operation timed out: {e}") from e
        if code == EXIT_EVICTED:
            self._dead = True
            raise SandboxPtyError("PTY session was taken over by another client")
        if code == EXIT_DEAD:
            self._dead = True
            raise SandboxPtyError(f"PTY session {self.session_id} no longer exists in the sandbox")
        if code == EXIT_HELD:
            raise SandboxPtyError(
                f"PTY session {self.session_id} already has an attached client (pass takeover=True to evict)"
            )
        return code, out, err

    async def _poll(self, *, max_bytes: int) -> dict[str, Any]:
        code, out, err = await self._helper(
            [
                "poll",
                _session_dir(self.session_id),
                self._token,
                str(self._out_offset),
                str(self._err_offset),
                str(max_bytes),
            ]
        )
        if code != 0:
            self._dead = True
            raise SandboxPtyError(f"PTY session {self.session_id} poll failed (code={code}): {err.strip()}")
        try:
            info = json.loads(out)
            info["out_bytes"] = base64.b64decode(info.pop("out"))
            info["err_bytes"] = base64.b64decode(info.pop("err"))
            return info
        except (ValueError, KeyError, TypeError) as e:
            raise SandboxPtyError(f"PTY session {self.session_id} poll returned malformed data: {e!r}") from e

    def _ensure_writable(self) -> None:
        if self._closed:
            raise SandboxPtyError("PTY session is closed")
        if self._dead:
            raise SandboxPtyError("PTY session has ended")

    async def _read_stream(self, stream: str, timeout_s: float | None) -> bytes:
        if self._closed and self._exit_code is None:
            raise SandboxPtyError("PTY session is closed")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s if timeout_s is not None else None
        while True:
            async with self._read_lock:
                if self._dead and self._exit_code is None:
                    raise SandboxPtyError("PTY session has ended")
                info = await self._poll(max_bytes=POLL_MAX_BYTES)
                if info.get("exit") is not None:
                    self._exit_code = int(info["exit"])
                chunk: bytes = info[f"{stream}_bytes"]
                if chunk:
                    self._advance(stream, len(chunk))
                    return chunk
                if self._exit_code is not None:
                    # Exited and this stream is drained.
                    if self._out_offset >= int(info["out_size"]) and self._err_offset >= int(info["err_size"]):
                        self._dead = True
                    return b""
                if not info.get("alive", False):
                    self._dead = True
                    raise SandboxPtyError("PTY session died without exiting")
            now = loop.time()
            if deadline is not None and now >= deadline:
                raise TimeoutError(f"no PTY {'stderr' if stream == 'err' else 'output'} within {timeout_s:g}s")
            sleep_s = READ_POLL_INTERVAL_S if deadline is None else min(READ_POLL_INTERVAL_S, deadline - now)
            await asyncio.sleep(sleep_s)

    def _advance(self, stream: str, n: int) -> None:
        if stream == "out":
            self._out_offset += n
        else:
            self._err_offset += n

    async def read(self, *, timeout_s: float | None = None) -> bytes:
        """Next output chunk (all terminal output in PTY mode; stdout in pipe
        mode). ``b""`` once the process exited and the stream is drained."""
        return await self._read_stream("out", timeout_s)

    async def read_stderr(self, *, timeout_s: float | None = None) -> bytes:
        """Next stderr chunk (pipe mode only; the PTY-mode stderr stream is
        empty and returns ``b""`` once the process exits)."""
        return await self._read_stream("err", timeout_s)

    def __aiter__(self) -> AsyncIterator[bytes]:
        async def _iterate() -> AsyncIterator[bytes]:
            while chunk := await self.read():
                yield chunk

        return _iterate()

    async def write(self, data: bytes) -> None:
        """Send raw bytes to the terminal's stdin (via the session's FIFO)."""
        self._ensure_writable()
        await self._helper(
            ["write", _session_dir(self.session_id), self._token],
            stdin=data,
            extra_timeout_s=_WRITE_EXTRA_TIMEOUT_S,
        )

    async def _ctl(self, payload: dict[str, Any]) -> None:
        self._ensure_writable()
        await self._helper(
            ["ctl", _session_dir(self.session_id), self._token],
            stdin=json.dumps(payload).encode() + b"\n",
            extra_timeout_s=_WRITE_EXTRA_TIMEOUT_S,
        )

    async def resize(self, rows: int, cols: int) -> None:
        """Resize the terminal (ignored by pipe-mode sessions)."""
        await self._ctl({"type": "resize", "rows": int(rows), "cols": int(cols)})

    async def send_signal(self, signal: str) -> None:
        """Deliver a named signal (e.g. ``"SIGTERM"``) to the session's
        process group."""
        if not signal.startswith("SIG") or not hasattr(signal_module, signal):
            raise ValueError(f"unknown signal name: {signal!r}")
        await self._ctl({"type": "signal", "signal": signal})

    async def wait_exit(self, *, timeout_s: float | None = None) -> int:
        """Block (polling the exit-code file) until the process exits."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s if timeout_s is not None else None
        while True:
            if self._exit_code is not None:
                return self._exit_code
            if self._closed:
                raise SandboxPtyError("PTY session closed before process exit")
            info = await self._poll(max_bytes=0)
            if info.get("exit") is not None:
                self._exit_code = int(info["exit"])
                return self._exit_code
            if not info.get("alive", False):
                self._dead = True
                raise SandboxPtyError("PTY session died without exiting")
            now = loop.time()
            if deadline is not None and now >= deadline:
                raise TimeoutError(f"PTY process did not exit within {timeout_s:g}s")
            sleep_s = READ_POLL_INTERVAL_S if deadline is None else min(READ_POLL_INTERVAL_S, deadline - now)
            await asyncio.sleep(sleep_s)

    async def run_detached(self, command: str, *, poll_interval_s: float = 15.0) -> tuple[bytes, int | None]:
        """Run one command, polling the session's output log every
        ``poll_interval_s`` while it works.

        Docker sessions hold no connection between polls and retain the whole
        output log, so unlike the OpenSandbox backend nothing can be lost
        between polls and no retained-window error exists. Returns
        ``(merged output, exit code or None)``; ``None`` means the marker
        line came back mangled. Callers serialize: one command per session.
        """
        token = f"NGPTY{uuid.uuid4().hex[:12]}"
        needle = f"{token}:".encode()
        # Marker from two literals so the terminal's echo of this line cannot
        # match it; the brace group keeps shell state while putting stdin at
        # EOF (same discipline as _run_in_pty_session in the api module).
        await self.write(
            f"{{ {command}\n}} </dev/null\nprintf '%s%s:%s\\n' '{token[:5]}' '{token[5:]}' \"$?\"\n".encode()
        )
        buffer = bytearray()
        while needle not in buffer:
            try:
                chunk = await self.read(timeout_s=1.0)
            except (TimeoutError, asyncio.TimeoutError):
                await asyncio.sleep(poll_interval_s)
                continue
            if not chunk:
                raise SandboxPtyError("PTY session ended before the command finished")
            buffer.extend(chunk)
        output, _, trailing = bytes(buffer).partition(needle)
        while b"\n" not in trailing:
            # The status digits can straddle the chunk that carried the marker.
            chunk = await self.read(timeout_s=5.0)
            if not chunk:
                break
            trailing += chunk
        exit_text = trailing.split(b"\n", 1)[0].strip()
        stderr = bytearray()
        try:
            # Pipe mode carries stderr separately; fold it in best-effort.
            while chunk := await self.read_stderr(timeout_s=0.05):
                stderr.extend(chunk)
        except (TimeoutError, asyncio.TimeoutError):
            pass
        return bytes(output + stderr), int(exit_text) if exit_text.isdigit() else None

    async def close(self) -> None:
        """Idempotent. The creator's session is ended (process killed, state
        removed); an attached session is merely released and lives on."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._owned:
                await self._helper(["end", _session_dir(self.session_id)])
            else:
                await self._helper(["release", _session_dir(self.session_id), self._token])
        except Exception:
            pass  # already-evicted/gone sessions and stopped containers close cleanly

    async def __aenter__(self) -> "DockerPtySession":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
