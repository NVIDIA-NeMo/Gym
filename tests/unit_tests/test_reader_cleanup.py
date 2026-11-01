# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest

import nemo_gym.orchestration.executors.connection as connection
import nemo_gym.orchestration.executors.slurm as slurm
from nemo_gym.cli.main import _raise_keyboard_interrupt_on_sigterm
from nemo_gym.orchestration.executors.connection import LocalConnection, SSHConnection


SLEEPER = [sys.executable, "-c", "import time; time.sleep(300)"]


def test_the_reader_token_is_the_callers_value_or_one_uuid_per_process(monkeypatch) -> None:
    monkeypatch.setattr(slurm, "_reader_token", None)
    monkeypatch.setenv("GYM_READER_TOKEN", "caller-token")
    assert slurm.reader_token() == "caller-token"

    monkeypatch.setattr(slurm, "_reader_token", None)
    monkeypatch.delenv("GYM_READER_TOKEN")
    generated = slurm.reader_token()
    assert len(generated) == 32 and slurm.reader_token() == generated


@pytest.mark.skipif(shutil.which("timeout") is None, reason="coreutils timeout not installed")
def test_the_reader_runs_with_its_token_in_its_environment(tmp_path: Path) -> None:
    command = slurm._reader_command(
        "printenv GYM_READER_TOKEN", tmp_path, "{}", bound_s=300, factor=1, token="token-in-env"
    )

    output = LocalConnection().run(command)

    assert "token-in-env" in output.splitlines()
    assert "__GYM_READER_RC:0" in output


def test_a_sigterm_during_the_reader_stops_it_by_token_and_unwinds() -> None:
    calls: list[list[str]] = []

    class FakeConnection:
        def run(self, commands: list[str]) -> str:
            calls.append(commands)
            if len(calls) == 1:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
                raise AssertionError("SIGTERM did not interrupt the reader")
            return "__GYM_READER_KILL:term=1 kill=0\n"

    previous = signal.getsignal(signal.SIGTERM)
    _raise_keyboard_interrupt_on_sigterm()
    try:
        with pytest.raises(slurm.RemoteReaderInterrupted, match="interrupted by SIGTERM") as raised:
            slurm._run_reader(FakeConnection(), ["the reader"], "token-on-sigterm")
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert calls == [["the reader"], slurm._kill_reader_command("token-on-sigterm")]
    assert "GYM_READER_TOKEN=token-on-sigterm" in str(raised.value)
    assert "term=1 kill=0" in str(raised.value)


def test_stopping_by_token_kills_only_the_processes_that_carry_it() -> None:
    token = f"token-{uuid.uuid4().hex}"
    in_environment = subprocess.Popen(SLEEPER, env={**os.environ, "GYM_READER_TOKEN": token})
    in_arguments = subprocess.Popen([*SLEEPER, token])
    another = subprocess.Popen(SLEEPER, env={**os.environ, "GYM_READER_TOKEN": "another-token"})
    try:
        report = LocalConnection().run(slurm._kill_reader_command(token))

        assert in_environment.wait(timeout=15) == -signal.SIGTERM
        assert in_arguments.wait(timeout=15) == -signal.SIGTERM
        assert another.poll() is None
        assert "__GYM_READER_KILL:term=2 kill=0" in report
    finally:
        for process in (in_environment, in_arguments, another):
            process.kill()
            process.wait()


class _FakeSubprocess:
    """Stands in for ``subprocess`` in the connection module: records every ssh call, starts nothing."""

    PIPE = subprocess.PIPE

    def __init__(self, *, master_running: bool) -> None:
        self.calls: list[list[str]] = []
        self.master_running = master_running
        self.terminated = False

    def Popen(self, command: list[str], **kwargs) -> "_FakeSubprocess":
        self.calls.append(command)
        return self

    def poll(self) -> int | None:
        return None if self.master_running and not self.terminated else 0

    def terminate(self) -> None:
        self.terminated = True

    def wait(self) -> int:
        return 0

    def run(self, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")


def _closed(calls: list[list[str]], socket: Path) -> bool:
    return any(call[-3:-1] == ["-O", "exit"] and f"ControlPath={socket}" in call for call in calls)


def test_an_interrupted_reader_closes_the_ssh_master_and_its_socket_stays_under_tmpdir(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    fake = _FakeSubprocess(master_running=False)
    monkeypatch.setattr(connection, "subprocess", fake)
    monkeypatch.setattr(SSHConnection, "_wait_for_socket", lambda self: None)

    with pytest.raises(slurm.RemoteReaderInterrupted):
        with SSHConnection("login-host") as conn:
            raise slurm.RemoteReaderInterrupted("remote Store reader interrupted by SIGTERM")

    assert conn._socket.parent == tmp_path
    assert _closed(fake.calls, conn._socket)


def test_a_signal_while_the_master_starts_closes_it_and_stops_the_foreground_ssh(monkeypatch) -> None:
    fake = _FakeSubprocess(master_running=True)
    monkeypatch.setattr(connection, "subprocess", fake)

    def interrupted(self) -> None:
        raise KeyboardInterrupt("SIGTERM")

    monkeypatch.setattr(SSHConnection, "_wait_for_socket", interrupted)
    conn = SSHConnection("login-host")

    with pytest.raises(KeyboardInterrupt):
        with conn:
            raise AssertionError("the with body must not run")

    assert _closed(fake.calls, conn._socket)
    assert fake.terminated


def _token_processes(token: str) -> list[int]:
    """This user's processes that carry the token as GYM_READER_TOKEN in their environment."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            environ = (proc / "environ").read_bytes()
        except OSError:
            continue
        if f"GYM_READER_TOKEN={token}".encode() in environ.split(b"\0"):
            found.append(int(proc.name))
    return found


def _wait_until(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return bool(predicate())


needs_setsid = pytest.mark.skipif(
    shutil.which("setsid") is None or shutil.which("timeout") is None, reason="util-linux setsid or timeout missing"
)


@needs_setsid
def test_the_watched_reader_reports_as_before_over_a_held_input(tmp_path: Path) -> None:
    command = slurm._reader_command(
        "printenv GYM_READER_TOKEN", tmp_path, "{}", bound_s=300, factor=1, token="watched-token", watchdog=True
    )
    result: dict[str, str] = {}
    runner = threading.Thread(target=lambda: result.update(output=LocalConnection().run_with_input_held(command)))
    runner.start()
    runner.join(timeout=30)

    assert not runner.is_alive(), "the reader with its watcher did not end the shell"
    assert "watched-token" in result["output"].splitlines()
    assert "__GYM_READER_RC:0" in result["output"]


@needs_setsid
def test_the_watcher_stops_a_sleeping_reader_when_the_shell_input_closes(tmp_path: Path) -> None:
    token = f"token-{uuid.uuid4().hex}"
    command = slurm._reader_command(
        f"{sys.executable} -c 'import time; time.sleep(300)'",
        tmp_path,
        "{}",
        bound_s=300,
        factor=1,
        token=token,
        watchdog=True,
    )
    shell = subprocess.Popen(
        ["bash", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        shell.stdin.write("\n".join(command) + "\n")
        shell.stdin.flush()
        assert _wait_until(lambda: _token_processes(token), 15), "the reader never started"

        shell.stdin.close()  # The client, or its SSH connection, is gone.
        output, _ = shell.communicate(timeout=30)

        assert _wait_until(lambda: not _token_processes(token), 15), "the reader outlived the closed input"
        assert "__GYM_READER_RC:" in output
    finally:
        shell.kill()
        for pid in _token_processes(token):
            os.kill(pid, signal.SIGKILL)


def test_a_connection_that_holds_its_input_runs_the_reader_with_it_held() -> None:
    calls: list[str] = []

    class HeldConnection:
        holds_input = True

        def run(self, commands: list[str]) -> str:
            calls.append("run")
            return ""

        def run_with_input_held(self, commands: list[str]) -> str:
            calls.append("held")
            return "__GYM_READER_RC:0\n"

    assert slurm._run_reader(HeldConnection(), ["the reader"], "token") == "__GYM_READER_RC:0\n"
    assert calls == ["held"]
