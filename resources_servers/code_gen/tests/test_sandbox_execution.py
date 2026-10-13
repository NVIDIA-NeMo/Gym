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
"""Tests for grading code_gen candidates in a provider sandbox."""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from sandbox_execution import (
    CodeGenSandboxConfig,
    SandboxVerificationError,
    check_correctness_in_sandbox,
    exec_timeout_s,
)

from nemo_gym.failure_kinds import PROVIDER_UNAVAILABLE, VERIFIER_ERROR
from nemo_gym.sandbox import SandboxCreateError, SandboxExecResult, SandboxHandle


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SAMPLE = {"input_output": json.dumps({"inputs": ["3\n", "5\n"], "outputs": ["9\n", "25\n"], "fn_name": None})}
_SQUARE = "n=int(input())\nprint(n*n)"


class _FakeProvider:
    """Records what the sandbox path asks of a provider and replays canned outcomes."""

    name = "fake"

    def __init__(
        self,
        *,
        create_error: Exception | None = None,
        exec_result: SandboxExecResult | None = None,
        result_json: str = '{"result": [true, true], "metadata": null}',
    ) -> None:
        self._create_error = create_error
        self._exec_result = exec_result or SandboxExecResult(stdout="", stderr="", return_code=0)
        self._result_json = result_json
        self.specs = []
        self.uploads = []
        self.execs = []
        self.closed = 0

    async def create(self, spec):
        if self._create_error is not None:
            raise self._create_error
        self.specs.append(spec)
        return SandboxHandle(sandbox_id="fake-1", provider_name=self.name, raw={})

    async def upload_file(self, handle, source_path, target_path):
        self.uploads.append(target_path)

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.execs.append((command, timeout_s))
        if command.startswith("tail "):
            return SandboxExecResult(stdout="Traceback: boom\n", stderr="", return_code=0)
        return self._exec_result

    async def download_file(self, handle, source_path, target_path):
        Path(target_path).write_text(self._result_json)

    async def close(self, handle):
        self.closed += 1

    async def aclose(self):
        pass


async def _grade(provider, config=None, generation=_SQUARE, **kwargs):
    return await check_correctness_in_sandbox(
        _SAMPLE,
        generation,
        3,
        False,
        num_tests=2,
        provider=provider,
        config=config or CodeGenSandboxConfig(image="code-gen-verifier:test", resources={"cpu": 1}),
        metadata={"benchmark": "code-gen"},
        **kwargs,
    )


class TestSandboxLifecycle:
    async def test_request_spec_and_deadline(self) -> None:
        provider = _FakeProvider()
        result, metadata = await _grade(provider)

        assert (result, metadata) == ([True, True], None)
        (spec,) = provider.specs
        assert spec.image == "code-gen-verifier:test"
        assert spec.resources.cpu == 1
        assert spec.env["OMP_NUM_THREADS"] == "1"
        assert spec.metadata == {"benchmark": "code-gen"}
        assert set(provider.uploads) == {
            "/tmp/code_gen_verify/sandbox_runner.py",
            "/tmp/code_gen_verify/lcb_integration/__init__.py",
            "/tmp/code_gen_verify/lcb_integration/correctness.py",
            "/tmp/code_gen_verify/lcb_integration/testing_util.py",
            "/tmp/code_gen_verify/request.json",
        }
        # check_correctness allows (timeout + 1) * num_tests + 5 seconds; the runner gets that plus overhead.
        (command, timeout_s), *_ = provider.execs
        assert "sandbox_runner.py request.json result.json" in command
        assert timeout_s == exec_timeout_s(3, 2) == (3 + 1) * 2 + 5 + 60
        assert spec.ttl_s > spec.ready_timeout_s + timeout_s
        assert provider.closed == 1

    async def test_create_failure_is_provider_unavailable(self) -> None:
        provider = _FakeProvider(create_error=SandboxCreateError("no capacity"))
        with pytest.raises(SandboxVerificationError) as info:
            await _grade(provider)
        assert info.value.failure_kind == PROVIDER_UNAVAILABLE
        assert "no capacity" in str(info.value)

    async def test_timeout_reported_only_by_exit_code_is_verifier_error(self) -> None:
        # ECS-style: exit code 124 without error_type.
        provider = _FakeProvider(
            exec_result=SandboxExecResult(stdout="", stderr="timed out after 73s", return_code=124)
        )
        with pytest.raises(SandboxVerificationError) as info:
            await _grade(provider)
        assert info.value.failure_kind == VERIFIER_ERROR
        assert "exit=124" in str(info.value) and "Traceback: boom" in str(info.value)
        assert provider.closed == 1

    async def test_malformed_result_is_verifier_error(self) -> None:
        provider = _FakeProvider(result_json='{"result": "not a list"}')
        with pytest.raises(SandboxVerificationError) as info:
            await _grade(provider)
        assert info.value.failure_kind == VERIFIER_ERROR
        assert provider.closed == 1

    def test_unknown_resource_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown sandbox resource keys: memory"):
            CodeGenSandboxConfig(image="x", resources={"memory": 4096})


class TestLocalProvider:
    """The real runner and harness through the unisolated `local` provider."""

    async def _grade_locally(self, tmp_path: Path, generation: str, python: str = sys.executable):
        return await _grade(
            {"local": {}},
            config=CodeGenSandboxConfig(python=python),
            generation=generation,
            work_dir=str(tmp_path / "work"),
        )

    async def test_passing_candidate(self, tmp_path: Path) -> None:
        result, metadata = await self._grade_locally(tmp_path, _SQUARE)
        assert result == [True, True]
        assert "execution time" in metadata

    async def test_candidate_failures_are_results(self, tmp_path: Path) -> None:
        wrong, wrong_metadata = await self._grade_locally(tmp_path, "n=int(input())\nprint(n+n)")
        assert wrong == [-2]
        assert wrong_metadata["error_message"] == "Wrong answer at output_line_idx=0: 6 != 9"

        timed_out, _ = await self._grade_locally(tmp_path, "while True:\n    pass")
        assert timed_out == [-3]

    async def test_runner_failure_is_verifier_error_with_log(self, tmp_path: Path) -> None:
        with pytest.raises(SandboxVerificationError) as info:
            await self._grade_locally(tmp_path, _SQUARE, python=str(tmp_path / "missing-python"))
        assert info.value.failure_kind == VERIFIER_ERROR
        assert "runner.log tail" in str(info.value)


_DOCKER_IMAGE = os.environ.get("CODE_GEN_SANDBOX_IMAGE")


@pytest.mark.skipif(
    shutil.which("docker") is None or not _DOCKER_IMAGE,
    reason="needs docker and CODE_GEN_SANDBOX_IMAGE built from sandbox_image/",
)
class TestDockerProvider:
    async def test_memory_limit_fails_the_test_not_the_sandbox(self) -> None:
        docker_config = yaml.safe_load(
            (_REPO_ROOT / "nemo_gym/sandbox/providers/docker/configs/docker.yaml").read_text()
        )["sandbox"]["docker"]
        # The harness's 4 GiB RLIMIT_DATA must trip before the container's memory limit does.
        result, metadata = await _grade(
            {"docker": docker_config},
            config=CodeGenSandboxConfig(image=_DOCKER_IMAGE, resources={"cpu": 1, "memory_mib": 6144}),
            generation="x = bytearray(8 * 1024**3)\nprint(1)",
        )
        assert result == [-4]
        assert metadata["error"] == "MemoryError()"
