import shlex
import shutil
import socket
import subprocess
import tempfile
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

    def close(self) -> None:
        pass


class LocalConnection(Connection):
    def copy(self, local: Path, remote: Path) -> None:
        if remote.exists():
            shutil.rmtree(remote)
        shutil.copytree(local, remote)

    def run(self, commands: list[str]) -> str:
        return "\n".join(_checked(shlex.split(cmd)) for cmd in commands)


class SSHConnection(Connection):
    """Single SSH master connection; copy and all commands reuse the same socket."""

    _OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
    _MASTER_TIMEOUT = 10  # seconds to wait for the control socket to appear

    def __init__(self, hostname: str) -> None:
        self._hostname = hostname
        self._socket = Path(tempfile.mktemp(prefix="gym-ssh-", suffix=".sock"))
        self._master: subprocess.Popen | None = None

    def __enter__(self) -> "SSHConnection":
        self._master = subprocess.Popen(
            [
                "ssh",
                *self._OPTS,
                "-o", "ControlMaster=yes",
                "-o", f"ControlPath={self._socket}",
                "-o", "ControlPersist=yes",
                "-N",
                self._hostname,
            ],
            stderr=subprocess.PIPE,
        )
        self._wait_for_socket()
        return self

    def _wait_for_socket(self) -> None:
        deadline = time.monotonic() + self._MASTER_TIMEOUT
        while time.monotonic() < deadline:
            if self._socket.exists():
                return
            if self._master and self._master.poll() is not None:
                stderr = (self._master.stderr.read().decode(errors="replace") if self._master.stderr else "")
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
                "rsync", "-az", "--delete",
                "-e", f"ssh {' '.join(self._ssh_opts())}",
                f"{local}/",
                f"{self._hostname}:{remote}",
            ],
            context=f"rsync to {self._hostname}:{remote}",
        )

    def run(self, commands: list[str]) -> str:
        script = "\n".join(commands)
        return _checked(
            ["ssh", *self._ssh_opts(), self._hostname, "bash", "-s"],
            input=script,
            context=f"ssh commands on {self._hostname}",
        )

    def close(self) -> None:
        subprocess.run(
            ["ssh", *self._ssh_opts(), "-O", "exit", self._hostname],
            capture_output=True,
        )
        self._socket.unlink(missing_ok=True)
        if self._master:
            self._master.wait()


def _checked(cmd: list[str], *, input: str | None = None, context: str = "") -> str:
    result = subprocess.run(cmd, input=input, text=True, capture_output=True)
    if result.returncode != 0:
        label = f" ({context})" if context else ""
        raise RuntimeError(
            f"Command failed{label} with exit code {result.returncode}:\n"
            f"  {' '.join(cmd)}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout


def get_connection(hostname: str | None) -> Connection:
    if hostname is None or hostname == socket.gethostname():
        return LocalConnection()
    return SSHConnection(hostname)
