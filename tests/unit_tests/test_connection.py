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

import socket
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.connection import LocalConnection, SSHConnection, get_connection
from nemo_gym.orchestration.executors.slurm import SlurmExecutor


# ---------------------------------------------------------------------------
# get_connection routing
# ---------------------------------------------------------------------------


def test_get_connection_none_returns_local():
    assert isinstance(get_connection(None), LocalConnection)


def test_get_connection_own_hostname_returns_local():
    assert isinstance(get_connection(socket.gethostname()), LocalConnection)


def test_get_connection_remote_hostname_returns_ssh():
    assert isinstance(get_connection("remote-login-node.example.com"), SSHConnection)


# ---------------------------------------------------------------------------
# SlurmExecutor on the login node makes no SSH calls
# ---------------------------------------------------------------------------


@pytest.fixture
def login_node_config(tmp_path):
    """SubmitConfig with hostname=None (already on the login node)."""
    return SubmitConfig.model_validate(
        {
            "services": {"vllm_model": {"type": "vllm", "container": "vllm:latest", "model": "org/model"}},
            "compute": {"cluster": {"type": "slurm", "account": "my-account"}},  # hostname omitted → None
            "driver": {
                "container": "python:3.12",
                "policy_model": "vllm_model",
                "benchmarks": {"gsm8k": {}},
            },
            "job": {"output_path": str(tmp_path / "gym-jobs")},
        }
    )


def test_slurm_executor_login_node_no_ssh(login_node_config):
    """When hostname is None (login node), SlurmExecutor must not open any SSH connection."""
    ssh_calls = []

    def fake_subprocess_run(cmd, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "ssh":
            ssh_calls.append(cmd)
        result = MagicMock(spec=subprocess.CompletedProcess)
        result.returncode = 0
        result.stdout = "Submitted batch job 99999\n"
        result.stderr = ""
        return result

    with patch("nemo_gym.orchestration.executors.connection.subprocess.run", side_effect=fake_subprocess_run):
        SlurmExecutor().run(login_node_config)

    assert ssh_calls == [], f"Unexpected SSH subprocess calls on login node: {ssh_calls}"


def test_slurm_executor_login_node_uses_local_connection(login_node_config):
    """SlurmExecutor must resolve to LocalConnection (not SSHConnection) when hostname is None."""
    connections_created = []
    real_get_connection = get_connection

    def spy_get_connection(hostname):
        conn = real_get_connection(hostname)
        connections_created.append(conn)
        return conn

    def fake_subprocess_run(cmd, **kwargs):
        result = MagicMock(spec=subprocess.CompletedProcess)
        result.returncode = 0
        result.stdout = "Submitted batch job 99999\n"
        result.stderr = ""
        return result

    with (
        patch("nemo_gym.orchestration.executors.slurm.get_connection", side_effect=spy_get_connection),
        patch("nemo_gym.orchestration.executors.connection.subprocess.run", side_effect=fake_subprocess_run),
    ):
        SlurmExecutor().run(login_node_config)

    assert len(connections_created) == 1
    assert isinstance(connections_created[0], LocalConnection), (
        f"Expected LocalConnection for login node, got {type(connections_created[0]).__name__}"
    )
