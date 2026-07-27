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

from typing import Any

from responses_api_agents.mini_swe_agent_2.sandbox_environment import (
    MiniSWESandboxEnvironment,
    MiniSWESandboxEnvironmentConfig,
    Submitted,
)


def test_check_finished_raises_submitted_for_submit_sentinel() -> None:
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)

    try:
        env._check_finished(
            {
                "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch contents\n",
                "returncode": 0,
                "exception_info": "",
            }
        )
    except Submitted as error:
        assert error.messages == (
            {
                "role": "exit",
                "content": "patch contents\n",
                "extra": {"exit_status": "Submitted", "submission": "patch contents\n"},
            },
        )
    else:
        raise AssertionError("Expected Submitted")


def test_check_finished_ignores_nonzero_submit_sentinel() -> None:
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)

    env._check_finished(
        {
            "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch contents\n",
            "returncode": 1,
            "exception_info": "",
        }
    )


def _env_with(activate_conda: bool, conda_env):
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)
    env.config = MiniSWESandboxEnvironmentConfig.__new__(MiniSWESandboxEnvironmentConfig)
    env.config.activate_conda = activate_conda
    env.config.conda_env = conda_env
    return env


def test_command_passthrough_when_conda_disabled() -> None:
    env = _env_with(activate_conda=False, conda_env="testbed")
    assert env._command("git apply patch.diff") == "git apply patch.diff"


def test_command_resolves_conda_without_relying_on_path() -> None:
    # Non-login ECS exec shell: conda isn't on PATH, so source conda.sh from known roots
    # via a grouped loop (not an `&&` chain that aborts on a missing root). cwd is passed
    # to exec(cwd=...), so the command itself must not `cd`.
    env = _env_with(activate_conda=True, conda_env="testbed")
    wrapped = env._command("git apply patch.diff")

    assert "for __base in" in wrapped
    assert "/opt/miniconda3" in wrapped  # one of the search roots
    assert "conda activate testbed && git apply patch.diff" in wrapped
    assert wrapped.count("&&") >= 2
    assert not wrapped.startswith("cd ")  # cwd handled by exec(cwd=...), not via cd


def test_execute_passes_configured_cwd_to_exec() -> None:
    class FakeSandbox:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def exec(self, command: str, **kwargs: Any):
            self.calls.append({"command": command, **kwargs})
            return type("Result", (), {"stdout": "ok", "stderr": None, "return_code": 0})()

    fake_sandbox = FakeSandbox()
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)
    env.config = MiniSWESandboxEnvironmentConfig(
        image="image:tag",
        provider={"fake": {}},
        cwd="/default",
        activate_conda=False,
    )
    env._sandbox = fake_sandbox

    assert env.execute("pwd", cwd="/repo") == {"output": "ok", "returncode": 0, "exception_info": ""}
    assert fake_sandbox.calls[-1]["command"] == "pwd"
    assert fake_sandbox.calls[-1]["cwd"] == "/repo"

    env.config.activate_conda = True
    env.config.conda_env = "testbed"
    env.execute("python -V", cwd="/repo")
    cmd = fake_sandbox.calls[-1]["command"]
    assert "for __base in" in cmd and "conda activate testbed && python -V" in cmd
    assert "cd /repo" not in cmd
    assert fake_sandbox.calls[-1]["cwd"] == "/repo"
