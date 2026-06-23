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
"""Tests for the mini SWE sandbox environment: submit-sentinel detection and command execution."""

from typing import Any

from responses_api_agents.mini_swe_agent_2.sandbox_environment import (
    MiniSWESandboxEnvironment,
    MiniSWESandboxEnvironmentConfig,
    Submitted,
)


def test_check_finished_raises_submitted_for_submit_sentinel() -> None:
    """Verify a zero-return-code submit sentinel raises Submitted carrying the patch as the exit message."""
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
    """Verify a submit sentinel with a nonzero return code does not raise Submitted."""
    env = MiniSWESandboxEnvironment.__new__(MiniSWESandboxEnvironment)

    env._check_finished(
        {
            "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch contents\n",
            "returncode": 1,
            "exception_info": "",
        }
    )


def test_execute_passes_configured_cwd_without_conda_cd() -> None:
    """Verify execute forwards the requested cwd and prepends conda activation without a cd into cwd."""

    class FakeSandbox:
        """Stand-in sandbox that records exec calls and returns a successful result."""

        def __init__(self) -> None:
            """Initialize the recorded-call list."""
            self.calls: list[dict[str, Any]] = []

        def exec(self, command: str, **kwargs: Any):
            """Record the command and keyword arguments and return a successful fake result.

            Args:
                command (str): Command string to record.
                **kwargs (Any): Additional exec options (e.g. cwd) to record.

            Returns:
                Result: An object exposing stdout, stderr, and a zero return_code.
            """
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
    assert fake_sandbox.calls[-1]["command"] == (
        "source $(conda info --base)/etc/profile.d/conda.sh && conda activate testbed && python -V"
    )
    assert "cd /repo" not in fake_sandbox.calls[-1]["command"]
    assert fake_sandbox.calls[-1]["cwd"] == "/repo"
