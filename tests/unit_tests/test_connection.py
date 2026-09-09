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

from pathlib import Path

from pytest import MonkeyPatch

from nemo_gym.orchestration.executors import connection as connection_module
from nemo_gym.orchestration.executors.connection import LocalConnection, SSHConnection


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


def test_ssh_connection_sends_a_quoted_heredoc(monkeypatch: MonkeyPatch):
    captured = {}

    def fake_checked(cmd, *, input=None, context=""):
        captured["cmd"] = cmd
        captured["input"] = input
        return ""

    monkeypatch.setattr(connection_module, "_checked", fake_checked)
    conn = SSHConnection("login-01")

    conn.write_text(Path("/jobs/run/gym-job.json"), '{"a": 1}\n')

    assert captured["cmd"][-2:] == ["bash", "-s"]
    # A quoted delimiter stops the shell expanding anything inside the payload.
    assert "<<'GYM_EOF'" in captured["input"]
    assert "/jobs/run/gym-job.json" in captured["input"]
    assert '{"a": 1}' in captured["input"]
    assert captured["input"].rstrip().endswith("GYM_EOF")
