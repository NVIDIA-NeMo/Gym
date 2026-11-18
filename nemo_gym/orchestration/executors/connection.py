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

import shlex
import shutil
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


class SSHConnection(Connection):
    """Single SSH master connection; copy and all commands reuse the same socket."""

    _OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
    _MASTER_TIMEOUT = 10  # seconds to wait for the control socket to appear
    # run_with_input_held keeps the remote shell's stdin open until it exits (see _checked_holding_input).
    holds_input = True

    def __init__(self, hostname: str) -> None:
        self._hostname = hostname
        self._socket = Path(tempfile.mktemp(prefix="gym-ssh-", suffix=".sock"))
        self._master: subprocess.Popen | None = None

    def __enter__(self) -> "SSHConnection":
        # Open a persistent master connection. Subsequent ssh/rsync calls reuse
        # the same socket via ControlMaster=no, avoiding per-command handshakes.
        self._master = subprocess.Popen(
            [
                "ssh",
                *self._OPTS,
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
        return [*self._OPTS, "-o", "ControlMaster=no", "-o", f"ControlPath={self._socket}"]

    def copy(self, local: Path, remote: Path) -> None:
        _checked(
            ["ssh", *self._ssh_opts(), self._hostname, "mkdir", "-p", str(remote)],
            context=f"mkdir -p {self._hostname}:{remote}",
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
        )

    def run(self, commands: list[str]) -> str:
        # Send all commands as a single bash script over one SSH session to avoid
        # repeated connection overhead for multi-benchmark submits.
        script = "\n".join(commands)
        return _checked(
            ["ssh", *self._ssh_opts(), self._hostname, "bash", "-s"],
            input=script,
            context=f"ssh commands on {self._hostname}",
        )

    def run_with_input_held(self, commands: list[str]) -> str:
        return _checked_holding_input(
            ["ssh", *self._ssh_opts(), self._hostname, "bash", "-s"],
            input="\n".join(commands) + "\n",
            context=f"ssh commands on {self._hostname}",
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
        )

    def close(self) -> None:
        subprocess.run(
            ["ssh", *self._ssh_opts(), "-O", "exit", self._hostname],
            capture_output=True,
        )
        self._socket.unlink(missing_ok=True)
        if self._master:
            # The master detaches into the background once it is up; a foreground ssh still running was stopped
            # before that, and would detach later with nothing left to close it.
            if self._master.poll() is None:
                self._master.terminate()
            self._master.wait()


def _checked(cmd: list[str], *, input: str | None = None, context: str = "") -> str:
    result = subprocess.run(cmd, input=input, text=True, capture_output=True)
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


def _checked_holding_input(cmd: list[str], *, input: str, context: str = "") -> str:
    """Like ``_checked``, but stdin stays open until the command exits, so its EOF on the far end means that this
    process or its connection is gone. The commands must end the remote shell themselves (``exit``).

    Progress lines on the command's stderr (``READER_PROGRESS_PREFIX``) reach this process's stderr while it runs;
    stdout is returned whole, as before."""
    with subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        output: dict[str, str] = {}
        readers = [
            threading.Thread(target=lambda: output.__setitem__("stdout", process.stdout.read()), daemon=True),
            threading.Thread(target=_read_stderr_forwarding_progress, args=(process.stderr, output), daemon=True),
        ]
        for reader in readers:
            reader.start()
        try:
            process.stdin.write(input)
            process.stdin.flush()
            for reader in readers:
                reader.join()
            returncode = process.wait()
        except BaseException:
            process.kill()
            raise
        finally:
            process.stdin.close()
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
