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

import shutil
import subprocess
import time
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from nemo_gym.orchestration.executors import connection as connection_module
from nemo_gym.orchestration.executors.connection import (
    SSH_COMMAND_TIMEOUT_ENV,
    SSH_CONNECT_TIMEOUT_ENV,
    SSH_SERVER_ALIVE_COUNT_ENV,
    SSH_SERVER_ALIVE_INTERVAL_ENV,
    LocalConnection,
    SSHConnection,
)
from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord


def test_local_connection_runs_a_compound_bash_command(tmp_path):
    # Task 2's sbatch command is bash, not a simple argv. shlex.split would
    # mangle it, so local submits depend on this going through a shell.
    marker = tmp_path / "ran"
    LocalConnection().run([f'out=hello; rc=$?; echo "$out:$rc" > {marker}'])

    assert marker.read_text().strip() == "hello:0"


def test_local_connection_runs_every_command_in_one_shell():
    # Both connections must agree: a failing command does not abort the rest.
    output = LocalConnection().run(["echo first", "false", "echo third"])

    assert "first" in output and "third" in output


def test_local_connection_writes_the_file(tmp_path):
    target = tmp_path / "nested" / "gym-job.json"

    LocalConnection().write_text(target, '{"a": 1}\n')

    assert target.read_text() == '{"a": 1}\n'


def test_local_connection_writes_a_private_file_owner_only(tmp_path):
    target = tmp_path / "resolved-config.yaml"

    LocalConnection().write_text(target, "secret: x\n", private=True)

    assert oct(target.stat().st_mode & 0o777) == oct(0o600)


def _script_for(monkeypatch: MonkeyPatch, remote: Path, content: str, **kwargs) -> str:
    """The bash `SSHConnection.write_text` would send, without opening a socket."""
    captured = {}

    def fake_checked(cmd, *, input=None, context="", timeout_s=None):
        captured["cmd"] = cmd
        captured["input"] = input
        return ""

    monkeypatch.setattr(connection_module, "_checked", fake_checked)
    SSHConnection("login-01").write_text(remote, content, **kwargs)

    assert captured["cmd"][-2:] == ["bash", "-s"]
    return captured["input"]


def test_ssh_connection_writes_the_same_bytes_the_local_index_holds(monkeypatch: MonkeyPatch, tmp_path):
    # Run the generated script through a real bash and compare the file it
    # produces, byte for byte, against `dumps`. Asserting on substrings of the
    # command instead is what let a stray trailing newline ship: the remote
    # manifest and the local index have to be the same bytes, and only the
    # file the shell actually writes can show that.
    #
    # The destination has a space in it, so a build that dropped `shlex.quote`
    # fails here rather than passing on a path that never needed quoting. In
    # production the run directory is `<job.output_path>/gym-job-...`, and
    # `job.output_path` comes from a config file.
    record = SubmissionRecord(
        gym_job_id="gym-job-20260909T100203Z-abc123",
        gym_version="0.6.0",
        submitted_at="2026-09-09T10:02:03Z",
        run_dir="/jobs/gym-job-20260909T100203Z-abc123",
        cluster="hsg",
        executor="slurm",
        submitted_by="wprazuch",
        hostname="login-01",
        benchmarks=[
            BenchmarkJob(
                benchmark="gsm8k",
                job_dir="/jobs/gym-job-20260909T100203Z-abc123/gsm8k",
                job_id="12345",
            )
        ],
    )
    target = tmp_path / "run dir" / "gym-job.json"
    target.parent.mkdir()

    script = _script_for(monkeypatch, target, record.dumps())
    subprocess.run(["bash", "-s"], input=script, text=True, check=True)

    assert target.read_bytes() == record.dumps().encode()
    # And the store this is supposed to match, written the other way.
    local = tmp_path / "local.json"
    LocalConnection().write_text(local, record.dumps())
    assert target.read_bytes() == local.read_bytes()


def test_ssh_connection_quotes_the_heredoc_delimiter(monkeypatch: MonkeyPatch, tmp_path):
    # A quoted delimiter stops the shell expanding anything inside the payload:
    # a manifest carrying `$HOME` or a backtick must land as written.
    target = tmp_path / "gym-job.json"
    content = '{"note": "$HOME and `id` and ${PATH}"}\n'

    script = _script_for(monkeypatch, target, content)
    assert "<<'GYM_EOF'" in script
    subprocess.run(["bash", "-s"], input=script, text=True, check=True)

    assert target.read_text() == content


def test_ssh_connection_writes_a_private_file_owner_only(monkeypatch: MonkeyPatch, tmp_path):
    target = tmp_path / "resolved-config.yaml"

    script = _script_for(monkeypatch, target, "secret: x\n", private=True)
    subprocess.run(["bash", "-s"], input=script, text=True, check=True)

    assert oct(target.stat().st_mode & 0o777) == oct(0o600)
    assert target.read_text() == "secret: x\n"


_SSH_BOUND_ENV = (
    SSH_CONNECT_TIMEOUT_ENV,
    SSH_SERVER_ALIVE_INTERVAL_ENV,
    SSH_SERVER_ALIVE_COUNT_ENV,
    SSH_COMMAND_TIMEOUT_ENV,
)


@pytest.fixture
def no_ssh_bounds(monkeypatch: MonkeyPatch) -> None:
    for name in _SSH_BOUND_ENV:
        monkeypatch.delenv(name, raising=False)


def test_ssh_is_not_bounded_unless_the_environment_asks_for_it(no_ssh_bounds):
    connection = SSHConnection("login-01")

    options = " ".join(connection._ssh_opts())
    assert "ConnectTimeout" not in options and "ServerAlive" not in options
    assert connection._command_timeout_s is None


def test_ssh_options_follow_the_environment(no_ssh_bounds, monkeypatch: MonkeyPatch):
    monkeypatch.setenv(SSH_CONNECT_TIMEOUT_ENV, "20")
    monkeypatch.setenv(SSH_SERVER_ALIVE_INTERVAL_ENV, "15")
    monkeypatch.setenv(SSH_SERVER_ALIVE_COUNT_ENV, "4")
    monkeypatch.setenv(SSH_COMMAND_TIMEOUT_ENV, "600")

    connection = SSHConnection("login-01")

    options = connection._ssh_opts()
    for option in ("ConnectTimeout=20", "ServerAliveInterval=15", "ServerAliveCountMax=4"):
        assert ["-o", option] == options[options.index(option) - 1 : options.index(option) + 1]
    assert connection._command_timeout_s == 600


@pytest.mark.skipif(shutil.which("ssh") is None, reason="needs the ssh client")
def test_ssh_accepts_the_options_it_is_given(no_ssh_bounds, monkeypatch: MonkeyPatch):
    monkeypatch.setenv(SSH_CONNECT_TIMEOUT_ENV, "20")
    monkeypatch.setenv(SSH_SERVER_ALIVE_INTERVAL_ENV, "15")
    monkeypatch.setenv(SSH_SERVER_ALIVE_COUNT_ENV, "4")
    options = SSHConnection("login-01")._ssh_opts()

    resolved = subprocess.run(["ssh", "-G", *options, "login-01"], capture_output=True, text=True, check=True).stdout

    assert "connecttimeout 20" in resolved
    assert "serveraliveinterval 15" in resolved
    assert "serveralivecountmax 4" in resolved


@pytest.mark.parametrize("env_name", _SSH_BOUND_ENV)
@pytest.mark.parametrize("bad_value", ["0", "-5", "soon", "1.5"])
def test_a_bad_ssh_bound_is_an_error_not_silently_ignored(
    no_ssh_bounds, monkeypatch: MonkeyPatch, env_name, bad_value
):
    monkeypatch.setenv(env_name, bad_value)

    with pytest.raises(ValueError, match=env_name):
        SSHConnection("login-01")


def test_the_command_time_limit_reaches_every_ssh_command(no_ssh_bounds, monkeypatch: MonkeyPatch, tmp_path):
    monkeypatch.setenv(SSH_COMMAND_TIMEOUT_ENV, "45")
    seen = []

    def fake_checked(cmd, *, input=None, context="", timeout_s=None):
        seen.append(timeout_s)
        return ""

    monkeypatch.setattr(connection_module, "_checked", fake_checked)
    connection = SSHConnection("login-01")

    connection.write_text(tmp_path / "f", "x")
    connection.run(["true"])
    connection.copy(tmp_path, tmp_path / "dest")

    assert seen == [45, 45, 45, 45]  # write_text, run, then copy's mkdir and rsync


def test_a_command_that_outlasts_its_time_limit_raises(no_ssh_bounds):
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"timed out after 1 s \(slow thing\)"):
        connection_module._checked(["sleep", "30"], context="slow thing", timeout_s=1)

    assert time.monotonic() - started < 10


def test_a_command_without_a_time_limit_is_not_cut_short(no_ssh_bounds):
    assert connection_module._checked(["bash", "-c", "sleep 1; echo done"]) == "done\n"


def test_the_command_time_limit_reaches_the_held_input_path_of_the_remote_reader(
    no_ssh_bounds, monkeypatch: MonkeyPatch
):
    monkeypatch.setenv(SSH_COMMAND_TIMEOUT_ENV, "45")
    seen = []

    def fake_holding_input(cmd, *, input, context="", timeout_s=None):
        seen.append(timeout_s)
        return ""

    monkeypatch.setattr(connection_module, "_checked_holding_input", fake_holding_input)

    SSHConnection("login-01").run_with_input_held(["true"])

    assert seen == [45]


def test_a_held_command_that_outlasts_its_time_limit_is_killed_and_raises(no_ssh_bounds):
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"timed out after 1 s \(slow reader\)"):
        # `exec` replaces the shell, so killing it ends the command and closes its pipes, as killing ssh does.
        connection_module._checked_holding_input(
            ["bash", "-s"], input="exec sleep 30\n", context="slow reader", timeout_s=1
        )

    assert time.monotonic() - started < 10


def test_a_held_command_inside_its_time_limit_keeps_its_stdin_progress_and_output(
    no_ssh_bounds, capsys: pytest.CaptureFixture
):
    commands = f"echo '{connection_module.READER_PROGRESS_PREFIX} halfway' >&2\necho result\nexit\n"

    out = connection_module._checked_holding_input(["bash", "-s"], input=commands, timeout_s=30)

    assert out == "result\n"
    assert f"{connection_module.READER_PROGRESS_PREFIX} halfway" in capsys.readouterr().err


def test_a_held_command_without_a_time_limit_is_not_cut_short(no_ssh_bounds):
    out = connection_module._checked_holding_input(["bash", "-s"], input="sleep 1\necho done\nexit\n")

    assert out == "done\n"


def test_a_failed_held_command_still_reports_its_exit_code_and_stderr(no_ssh_bounds):
    with pytest.raises(RuntimeError, match=r"(?s)exit code 3.*boom"):
        connection_module._checked_holding_input(
            ["bash", "-s"], input="echo boom >&2\nexit 3\n", context="failing", timeout_s=30
        )


def test_a_held_command_that_never_reads_its_input_ends_at_its_time_limit(no_ssh_bounds):
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"timed out after 0.5 s \(stalled\)"):
        # The input is larger than a pipe buffer, so delivering it blocks until the command reads or is stopped.
        connection_module._checked_holding_input(
            ["bash", "-c", "exec sleep 30"], input="x" * 1_000_000, context="stalled", timeout_s=0.5
        )

    assert time.monotonic() - started < 10


def test_a_held_command_whose_descendant_keeps_its_output_open_ends_at_its_time_limit(no_ssh_bounds):
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"timed out after 0.5 s \(proxy\)"):
        # The shell exits at once; the background process keeps stdout and stderr open, as an SSH ProxyCommand does.
        connection_module._checked_holding_input(
            ["bash", "-s"], input="sleep 30 &\nexit\n", context="proxy", timeout_s=0.5
        )

    assert time.monotonic() - started < 10
