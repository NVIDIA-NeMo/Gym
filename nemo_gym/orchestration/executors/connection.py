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
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path


class Connection(ABC):
    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @abstractmethod
    def copy(self, local: Path, remote: Path) -> None: ...

    @abstractmethod
    def run(self, commands: list[str]) -> str: ...

    @abstractmethod
    def write_text(self, remote: Path, content: str, *, private: bool = False) -> None: ...

    def close(self) -> None:
        pass


class LocalConnection(Connection):
    # run_with_input_held keeps the shell's stdin open until it exits (see _checked_holding_input).
    holds_input = True

    def run_with_input_held(self, commands: list[str]) -> str:
        return _checked_holding_input(["bash", "-s"], input="\n".join(commands) + "\n", context="local commands")

    def copy(self, local: Path, remote: Path) -> None:
        if remote.exists():
            shutil.rmtree(remote)
        shutil.copytree(local, remote)

    def run(self, commands: list[str]) -> str:
        # Piped to bash, not shlex.split into argv: callers send compound bash
        # (`out=$(...); rc=$?; ...`), and this has to mean the same thing here
        # as it does over SSH.
        return _checked(["bash", "-s"], input="\n".join(commands), context="local commands")

    def write_text(self, remote: Path, content: str, *, private: bool = False) -> None:
        remote.parent.mkdir(parents=True, exist_ok=True)
        if private:
            remote.touch(mode=0o600)
            remote.chmod(0o600)
        remote.write_text(content, encoding="utf-8")


# Optional bounds on the SSH layer, read when a connection is created. Each is a positive number of seconds (the count
# is a number of probes), and unset, as by default, leaves the matching limit to ssh and the operating system.
SSH_CONNECT_TIMEOUT_ENV = "NEMO_GYM_SSH_CONNECT_TIMEOUT_S"  # ssh -o ConnectTimeout: the TCP connect and the handshake
SSH_SERVER_ALIVE_INTERVAL_ENV = (
    "NEMO_GYM_SSH_SERVER_ALIVE_INTERVAL_S"  # ssh -o ServerAliveInterval: probe a quiet link
)
SSH_SERVER_ALIVE_COUNT_ENV = "NEMO_GYM_SSH_SERVER_ALIVE_COUNT_MAX"  # ssh -o ServerAliveCountMax: unanswered probes
SSH_COMMAND_TIMEOUT_ENV = "NEMO_GYM_SSH_COMMAND_TIMEOUT_S"  # each ssh or rsync command Gym runs, start to finish


def _positive_int_from_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(f"{name} must be a positive whole number, got {raw!r}")
    return value


class SSHConnection(Connection):
    """Single SSH master connection; copy and all commands reuse the same socket.

    The environment variables above bound a connection that hangs: a login host that never answers the connect, a link
    that goes quiet, or a command that never finishes, the remote reader included. A command that runs out of its time
    raises RuntimeError.
    """

    _OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
    _MASTER_TIMEOUT = 10  # seconds to wait for the control socket to appear
    # run_with_input_held keeps the remote shell's stdin open until it exits (see _checked_holding_input).
    holds_input = True

    def __init__(self, hostname: str) -> None:
        self._hostname = hostname
        self._opts = list(self._OPTS)
        for env_name, option in (
            (SSH_CONNECT_TIMEOUT_ENV, "ConnectTimeout"),
            (SSH_SERVER_ALIVE_INTERVAL_ENV, "ServerAliveInterval"),
            (SSH_SERVER_ALIVE_COUNT_ENV, "ServerAliveCountMax"),
        ):
            value = _positive_int_from_env(env_name)
            if value is not None:
                self._opts += ["-o", f"{option}={value}"]
        self._command_timeout_s = _positive_int_from_env(SSH_COMMAND_TIMEOUT_ENV)
        self._socket = Path(tempfile.mktemp(prefix="gym-ssh-", suffix=".sock"))
        self._master: subprocess.Popen | None = None

    def __enter__(self) -> "SSHConnection":
        # Open a persistent master connection. Subsequent ssh/rsync calls reuse
        # the same socket via ControlMaster=no, avoiding per-command handshakes.
        self._master = subprocess.Popen(
            [
                "ssh",
                *self._opts,
                "-o",
                "ControlMaster=yes",
                "-o",
                f"ControlPath={self._socket}",
                "-o",
                "ControlPersist=yes",
                "-N",  # no remote command — just keep the tunnel open
                self._hostname,
            ],
            stderr=subprocess.PIPE,
        )
        try:
            self._wait_for_socket()
        except BaseException:
            # __exit__ does not run for a failed or interrupted __enter__; nothing else would close this master.
            self.close()
            raise
        return self

    def _wait_for_socket(self) -> None:
        deadline = time.monotonic() + self._MASTER_TIMEOUT
        while time.monotonic() < deadline:
            if self._socket.exists():
                return
            if self._master and self._master.poll() is not None:
                stderr = self._master.stderr.read().decode(errors="replace") if self._master.stderr else ""
                raise RuntimeError(
                    f"SSH master to '{self._hostname}' exited with code {self._master.returncode}.\n{stderr.strip()}"
                )
            time.sleep(0.2)
        raise RuntimeError(
            f"SSH master to '{self._hostname}' did not produce a control socket within {self._MASTER_TIMEOUT}s."
        )

    def _ssh_opts(self) -> list[str]:
        return [*self._opts, "-o", "ControlMaster=no", "-o", f"ControlPath={self._socket}"]

    def copy(self, local: Path, remote: Path) -> None:
        _checked(
            ["ssh", *self._ssh_opts(), self._hostname, "mkdir", "-p", str(remote)],
            context=f"mkdir -p {self._hostname}:{remote}",
            timeout_s=self._command_timeout_s,
        )
        _checked(
            [
                "rsync",
                "-az",
                "--delete",
                "-e",
                f"ssh {' '.join(self._ssh_opts())}",
                f"{local}/",
                f"{self._hostname}:{remote}",
            ],
            context=f"rsync to {self._hostname}:{remote}",
            timeout_s=self._command_timeout_s,
        )

    def run(self, commands: list[str]) -> str:
        # Send all commands as a single bash script over one SSH session to avoid
        # repeated connection overhead for multi-benchmark submits.
        script = "\n".join(commands)
        return _checked(
            ["ssh", *self._ssh_opts(), self._hostname, "bash", "-s"],
            input=script,
            context=f"ssh commands on {self._hostname}",
            timeout_s=self._command_timeout_s,
        )

    def run_with_input_held(self, commands: list[str]) -> str:
        return _checked_holding_input(
            ["ssh", *self._ssh_opts(), self._hostname, "bash", "-s"],
            input="\n".join(commands) + "\n",
            context=f"ssh commands on {self._hostname}",
            timeout_s=self._command_timeout_s,
        )

    def write_text(self, remote: Path, content: str, *, private: bool = False) -> None:
        # A quoted heredoc delimiter: the payload reaches the file byte for byte,
        # with no parameter or command substitution applied to it on the way.
        # The heredoc supplies the newline before the delimiter, so a payload
        # that already ends in one (jobs.dumps does) must shed it -- otherwise
        # the remote manifest gains a blank line the local index does not have,
        # and the two stores stop being byte-identical.
        payload = content.removesuffix("\n")
        umask = "umask 077\n" if private else ""
        script = f"{umask}cat > {shlex.quote(str(remote))} <<'GYM_EOF'\n{payload}\nGYM_EOF\n"
        _checked(
            ["ssh", *self._ssh_opts(), self._hostname, "bash", "-s"],
            input=script,
            context=f"writing {remote} on {self._hostname}",
            timeout_s=self._command_timeout_s,
        )

    def close(self) -> None:
        try:
            subprocess.run(
                ["ssh", *self._ssh_opts(), "-O", "exit", self._hostname],
                capture_output=True,
                timeout=self._command_timeout_s,
            )
        except subprocess.TimeoutExpired:
            pass  # The master is stopped below.
        self._socket.unlink(missing_ok=True)
        if self._master:
            # The master detaches into the background once it is up; a foreground ssh still running was stopped
            # before that, and would detach later with nothing left to close it.
            if self._master.poll() is None:
                self._master.terminate()
            try:
                self._master.wait(timeout=self._command_timeout_s)
            except subprocess.TimeoutExpired:
                self._master.kill()
                self._master.wait()


def _run_stopping_group_on_timeout(
    cmd: list[str], *, input: str | None, timeout_s: float
) -> subprocess.CompletedProcess[str]:
    """``subprocess.run(capture_output=True, text=True, timeout=timeout_s)``, but an expiry kills the command's whole
    process group, so what it started (an ssh ProxyCommand, an rsync helper) does not outlive the error."""
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            try:
                process.communicate(timeout=5)  # Reap the child and collect what the pipes still hold.
            except subprocess.TimeoutExpired:
                pass  # A process outside the group holds the pipes; the caller's error must not wait for it.
            raise
        except BaseException:
            # An interruption stops the group too: it is in its own session, so the terminal's SIGINT never reached it.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.wait()
            raise
    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)


def _checked(cmd: list[str], *, input: str | None = None, context: str = "", timeout_s: float | None = None) -> str:
    try:
        if timeout_s is None:
            result = subprocess.run(cmd, input=input, text=True, capture_output=True)
        else:
            result = _run_stopping_group_on_timeout(cmd, input=input, timeout_s=timeout_s)
    except subprocess.TimeoutExpired as exc:
        label = f" ({context})" if context else ""
        raise RuntimeError(f"Command timed out after {timeout_s} s{label}:\n  {' '.join(cmd)}") from exc
    if result.returncode != 0:
        label = f" ({context})" if context else ""
        raise RuntimeError(
            f"Command failed{label} with exit code {result.returncode}:\n  {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    return result.stdout


# A line the login-host reader's wrapper prints on stderr while the reader runs; forwarded to this process's stderr.
READER_PROGRESS_PREFIX = "gym-reader progress:"


def _read_stderr_forwarding_progress(stream, output: dict[str, str]) -> None:
    """Keep all of stderr for the error message, and pass each progress line on as it arrives."""
    lines = []
    for line in stream:
        lines.append(line)
        if line.startswith(READER_PROGRESS_PREFIX):
            sys.stderr.write(line)
            sys.stderr.flush()
    output["stderr"] = "".join(lines)


def _reap_in_background(process: subprocess.Popen, threads: list[threading.Thread]) -> None:
    """Reap a stopped command and close its pipes once the threads using them are done, without delaying the caller.

    Closing a pipe while a thread still reads or writes it would block on that thread, which is the overrun the
    deadline exists to prevent, so the close waits for each thread (they end because the command was killed)."""

    def reap() -> None:
        for thread in threads:
            thread.join()
        process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                pass

    threading.Thread(target=reap, daemon=True).start()


def _checked_holding_input(cmd: list[str], *, input: str, context: str = "", timeout_s: float | None = None) -> str:
    """Like ``_checked``, but stdin stays open until the command exits, so its EOF on the far end means that this
    process or its connection is gone. The commands must end the remote shell themselves (``exit``).

    Progress lines on the command's stderr (``READER_PROGRESS_PREFIX``) reach this process's stderr while it runs;
    stdout is returned whole, as before. A command still running after `timeout_s` seconds is killed, which closes its
    stdin and ends the remote shell as any other interruption does, and raises RuntimeError; None sets no limit."""
    deadline = None if timeout_s is None else time.monotonic() + timeout_s

    def remaining_s() -> float | None:
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    def timed_out() -> RuntimeError:
        label = f" ({context})" if context else ""
        return RuntimeError(f"Command timed out after {timeout_s} s{label}:\n  {' '.join(cmd)}")

    # With a limit, the command leads its own process group, so that expiry also stops what it started (an SSH
    # ProxyCommand keeps the output pipes open after ssh itself is gone).
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=timeout_s is not None,
    )

    def stop() -> None:
        try:
            if timeout_s is None:
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()

    output: dict[str, str] = {}

    def write_input() -> None:
        # A command that does not read its stdin blocks this write; the caller's deadline still applies.
        try:
            process.stdin.write(input)
            process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass  # The command ended or was stopped; its exit status and stderr report it.

    threads = [
        threading.Thread(target=write_input, daemon=True),
        threading.Thread(target=lambda: output.__setitem__("stdout", process.stdout.read()), daemon=True),
        threading.Thread(target=_read_stderr_forwarding_progress, args=(process.stderr, output), daemon=True),
    ]
    try:
        for thread in threads:
            thread.start()
        # The writer ends once the input is delivered; the readers end when the command and its descendants close
        # their output.
        for thread in threads:
            thread.join(remaining_s())
        if any(thread.is_alive() for thread in threads):
            raise timed_out()
        returncode = process.wait(timeout=remaining_s())
    except subprocess.TimeoutExpired as exc:
        stop()
        _reap_in_background(process, threads)
        raise timed_out() from exc
    except BaseException:
        stop()
        _reap_in_background(process, threads)
        raise
    # A finished command's pipes are closed here. A stopped one is closed by _reap_in_background.
    for stream in (process.stdin, process.stdout, process.stderr):
        stream.close()
    if returncode != 0:
        label = f" ({context})" if context else ""
        raise RuntimeError(
            f"Command failed{label} with exit code {returncode}:\n  {' '.join(cmd)}\n{output.get('stderr', '').strip()}"
        )
    return output.get("stdout", "")


def get_connection(hostname: str | None) -> Connection:
    if hostname is None or hostname == socket.gethostname():
        return LocalConnection()
    return SSHConnection(hostname)
