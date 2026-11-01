# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

import nemo_gym.orchestration.executors.slurm as slurm
from nemo_gym.cli.main import _raise_keyboard_interrupt_on_sigterm
from nemo_gym.orchestration.executors.connection import LocalConnection


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
