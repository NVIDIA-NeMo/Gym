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

import subprocess
from pathlib import Path

from pytest import MonkeyPatch

from nemo_gym.orchestration.executors import connection as connection_module
from nemo_gym.orchestration.executors.connection import LocalConnection, SSHConnection
from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord, dumps


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


def _script_for(monkeypatch: MonkeyPatch, remote: Path, content: str) -> str:
    """The bash `SSHConnection.write_text` would send, without opening a socket."""
    captured = {}

    def fake_checked(cmd, *, input=None, context=""):
        captured["cmd"] = cmd
        captured["input"] = input
        return ""

    monkeypatch.setattr(connection_module, "_checked", fake_checked)
    SSHConnection("login-01").write_text(remote, content)

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

    script = _script_for(monkeypatch, target, dumps(record))
    subprocess.run(["bash", "-s"], input=script, text=True, check=True)

    assert target.read_bytes() == dumps(record).encode()
    # And the store this is supposed to match, written the other way.
    local = tmp_path / "local.json"
    LocalConnection().write_text(local, dumps(record))
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
