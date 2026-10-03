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
import base64
import json
from types import SimpleNamespace

from mimoagent.environments.datasets import DATASET_REGISTRY

from nemo_gym.sandbox import SandboxExecResult
from resources_servers.mimo_rl_oss.sandbox_env import GymSandboxEnvironment
from resources_servers.mimo_rl_oss.terminal_bench import TerminalBenchEnvironment


class FakeEnv:
    """Records commands and serves a canned reward file, like a box after test.sh ran."""

    def __init__(self, reward_file: str) -> None:
        self.config = SimpleNamespace(cwd="/app")
        self.commands: list[str] = []
        self.files: dict[str, str] = {}
        self.reward_file = reward_file

    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> dict:
        self.commands.append(command)
        if command.startswith("cat /logs/verifier/reward.txt"):
            return {"output": self.reward_file, "returncode": 0, "reason": "ok"}
        return {"output": "", "returncode": 0, "reason": "ok"}

    def copy_to(self, src: str, dest: str, **_) -> None:
        self.files[dest] = open(src).read()


def _instance() -> dict:
    tests = {"test.sh": base64.b64encode(b"echo hi").decode()}
    return {"instance_id": "t", "cwd": "/app", "tests_files": json.dumps(tests), "verifier_timeout_sec": 60}


def test_terminal_bench_registered() -> None:
    assert DATASET_REGISTRY["terminal_bench"] is TerminalBenchEnvironment


def test_terminal_bench_uploads_tests_and_reads_reward() -> None:
    env = FakeEnv("1\n")
    reward, _, extra = TerminalBenchEnvironment(env, _instance())._do_calculate_reward()
    assert reward == 1.0
    assert env.files == {"/tests/test.sh": "echo hi"}
    assert "sh /tests/test.sh" in env.commands


def test_terminal_bench_missing_reward_is_zero() -> None:
    reward, _, extra = TerminalBenchEnvironment(FakeEnv(""), _instance())._do_calculate_reward()
    assert reward == 0.0
    assert "no reward file" in extra["error"]


def test_execute_strips_opensandbox_exit_marker() -> None:
    env = GymSandboxEnvironment.__new__(GymSandboxEnvironment)
    env.config = SimpleNamespace(cwd="/", timeout=10, env={})
    env.handle = object()
    result = SandboxExecResult(stdout="out\n", stderr="err\nCommandExecError: 1", return_code=1)
    env._run = lambda name, factory: result
    assert env.execute("false") == {"output": "out\nerr", "returncode": 1, "reason": "ok"}
