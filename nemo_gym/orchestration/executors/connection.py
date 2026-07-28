import shlex
import shutil
import socket
import subprocess
import tempfile
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
    def run(self, commands: list[str]) -> None: ...

    def close(self) -> None:
        pass


class LocalConnection(Connection):
    def copy(self, local: Path, remote: Path) -> None:
        if remote.exists():
            shutil.rmtree(remote)
        shutil.copytree(local, remote)

    def run(self, commands: list[str]) -> None:
        for cmd in commands:
            subprocess.run(shlex.split(cmd), check=True)


class SSHConnection(Connection):
    """Single SSH master connection; copy and all commands reuse the same socket."""

    _OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]

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
            ]
        )
        return self

    def _ssh_opts(self) -> list[str]:
        return [*self._OPTS, "-o", "ControlMaster=no", "-o", f"ControlPath={self._socket}"]

    def copy(self, local: Path, remote: Path) -> None:
        subprocess.run(
            [
                "rsync", "-az", "--delete",
                "-e", f"ssh {' '.join(self._ssh_opts())}",
                f"{local}/",
                f"{self._hostname}:{remote}",
            ],
            check=True,
        )

    def run(self, commands: list[str]) -> None:
        script = "\n".join(commands)
        subprocess.run(
            ["ssh", *self._ssh_opts(), self._hostname, "bash", "-s"],
            input=script,
            text=True,
            check=True,
        )

    def close(self) -> None:
        subprocess.run(
            ["ssh", *self._ssh_opts(), "-O", "exit", self._hostname],
            capture_output=True,
        )
        self._socket.unlink(missing_ok=True)
        if self._master:
            self._master.wait()


def get_connection(hostname: str | None) -> Connection:
    if hostname is None or hostname == socket.gethostname():
        return LocalConnection()
    return SSHConnection(hostname)
