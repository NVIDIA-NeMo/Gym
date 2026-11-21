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
import resource
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from sandbox_execution import check_correctness_in_sandbox, exec_timeout_s, task_deadline_s

from nemo_gym.core import VERIFIER_ERROR
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.verification import SandboxVerificationConfig, SandboxVerificationError


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SAMPLE = {"input_output": json.dumps({"inputs": ["3\n", "5\n"], "outputs": ["9\n", "25\n"], "fn_name": None})}
_SQUARE = "n=int(input())\nprint(n*n)"


class _FakeProvider:
    """Records what the sandbox path asks of a provider and returns a canned result file."""

    name = "fake"

    def __init__(self, *, result_json: str = '{"result": [true, true], "metadata": null}') -> None:
        self._result_json = result_json
        self.specs = []
        self.uploads = []
        self.execs = []
        self.closed = 0

    async def create(self, spec):
        self.specs.append(spec)
        return SandboxHandle(sandbox_id="fake-1", provider_name=self.name, raw={})

    async def upload_file(self, handle, source_path, target_path):
        self.uploads.append(target_path)

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.execs.append((command, timeout_s))
        return SandboxExecResult(stdout="", stderr="", return_code=0)

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
        config=config or SandboxVerificationConfig(image="code-gen-verifier:test", resources={"cpu": 1}),
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
        # Each verification gets its own directory under the default work dir.
        call_dir = provider.uploads[0].removeprefix("/tmp/code_gen_verify/").split("/")[0]
        assert set(provider.uploads) == {
            f"/tmp/code_gen_verify/{call_dir}/sandbox_runner.py",
            f"/tmp/code_gen_verify/{call_dir}/lcb_integration/__init__.py",
            f"/tmp/code_gen_verify/{call_dir}/lcb_integration/correctness.py",
            f"/tmp/code_gen_verify/{call_dir}/lcb_integration/testing_util.py",
            f"/tmp/code_gen_verify/{call_dir}/request.json",
        }
        # check_correctness allows (timeout + 1) * num_tests + 5 seconds; the runner gets that plus overhead.
        (command, timeout_s), *_ = provider.execs
        assert "sandbox_runner.py request.json result.json" in command
        assert timeout_s == exec_timeout_s(3, 2) == (3 + 1) * 2 + 5 + 60
        assert spec.ttl_s > spec.ready_timeout_s + timeout_s
        assert provider.closed == 1

    async def test_task_limits_reach_the_runner(self) -> None:
        provider = _FakeProvider()
        config = SandboxVerificationConfig(image="code-gen-verifier:test", resources={"cpu": 1, "memory_mib": 1024})
        await _grade(provider, config=config, memory_limit_mb=512, deadline_s=task_deadline_s(3, 2))

        (spec,) = provider.specs
        request = json.loads(next(text for path, text in spec.files.items() if path.endswith("/request.json")))
        assert (request["memory_limit_mb"], request["deadline_s"]) == (512, 3 + (3 + 1) * 2 + 5)
        # The sandbox grows to hold the task's limit plus the runner and harness.
        assert spec.resources.memory_mib == 512 + 2048
        (_, timeout_s), *_ = provider.execs
        assert timeout_s == request["deadline_s"] + 60

    async def test_malformed_result_is_verifier_error(self) -> None:
        provider = _FakeProvider(result_json='{"result": "not a list"}')
        with pytest.raises(SandboxVerificationError) as info:
            await _grade(provider)
        assert info.value.failure_kind == VERIFIER_ERROR
        assert provider.closed == 1


class TestLocalProvider:
    """The real runner and harness through the unisolated `local` provider."""

    async def _grade_locally(self, tmp_path: Path, generation: str, python: str = sys.executable):
        return await _grade(
            {"local": {}},
            config=SandboxVerificationConfig(python=python),
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

    async def test_fractional_time_limits(self, tmp_path: Path) -> None:
        slow = "import time\ntime.sleep(0.7)\nn=int(input())\nprint(n*n)"
        for timeout, expected in [(0.5, [-3]), (1.5, [True, True])]:
            result, _ = await check_correctness_in_sandbox(
                _SAMPLE,
                slow,
                timeout,
                False,
                num_tests=2,
                provider={"local": {}},
                config=SandboxVerificationConfig(python=sys.executable),
                metadata={},
                work_dir=str(tmp_path / "work"),
            )
            assert result == expected

    @pytest.mark.parametrize(
        "deadline_s, expected",
        # The task deadline leaves 8 s beyond the 2 s of work; a 1 s deadline stops it, where check_correctness's
        # own (2 + 1) * 1 + 5 = 8 s would not.
        [(task_deadline_s(2, 1), [True]), (1.0, [-1])],
    )
    async def test_a_task_deadline_covers_compilation_and_every_test(
        self, tmp_path: Path, deadline_s: float, expected: list
    ) -> None:
        # Module-level code runs during compilation; it and the one call each take half the 2 s limit.
        sample = {"input_output": json.dumps({"inputs": ["3"], "outputs": ["9"], "fn_name": "square"})}
        generation = "import time\ntime.sleep(1)\n\n\ndef square(n):\n    time.sleep(1)\n    return n * n"
        result, _ = await check_correctness_in_sandbox(
            sample,
            generation,
            2,
            False,
            num_tests=1,
            provider={"local": {}},
            config=SandboxVerificationConfig(python=sys.executable),
            metadata={},
            work_dir=str(tmp_path / "work"),
            deadline_s=deadline_s,
        )
        assert result == expected

    async def test_candidate_cannot_read_expected_outputs_from_the_request(self, tmp_path: Path) -> None:
        # The outputs have no relation to the inputs, so only a candidate that reads them can pass.
        sample = {"input_output": json.dumps({"inputs": ["1\n", "2\n"], "outputs": ["kiwi\n", "plum\n"]})}
        generation = (
            "import json\n"
            "io_ = json.loads(json.load(open('request.json'))['sample']['input_output'])\n"
            "print(dict(zip(io_['inputs'], io_['outputs']))[input() + '\\n'])"
        )
        result, metadata = await check_correctness_in_sandbox(
            sample,
            generation,
            3,
            False,
            num_tests=2,
            provider={"local": {}},
            config=SandboxVerificationConfig(python=sys.executable),
            metadata={},
            work_dir=str(tmp_path / "work"),
        )
        assert result == [-4], metadata
        assert "FileNotFoundError" in metadata["error"]

    @pytest.mark.skipif(not Path("/proc/self/limits").exists(), reason="reads /proc/self/limits")
    @pytest.mark.skipif(
        resource.getrlimit(resource.RLIMIT_CORE)[1] == 0,
        reason="the hard RLIMIT_CORE is 0, so the runner cannot set it to one byte",
    )
    async def test_candidates_cannot_dump_core(self, tmp_path: Path) -> None:
        # A one-byte RLIMIT_CORE keeps piped core dumps from stalling a crashing candidate past its time limit.
        # The harness blocks `import resource`, so the candidate reads its limit from /proc.
        generation = (
            "core = [l.split()[4:6] for l in open('/proc/self/limits') if l.startswith('Max core file size')][0]\n"
            "n=int(input())\n"
            "print(n*n if core == ['1', '1'] else 0)"
        )
        result, _ = await self._grade_locally(tmp_path, generation)
        assert result == [True, True]

    async def test_runner_failure_is_verifier_error_with_log(self, tmp_path: Path) -> None:
        with pytest.raises(SandboxVerificationError) as info:
            await self._grade_locally(tmp_path, _SQUARE, python=str(tmp_path / "missing-python"))
        assert info.value.failure_kind == VERIFIER_ERROR
        assert "runner.log tail" in str(info.value)


_DOCKER_IMAGE = os.environ.get("CODE_GEN_SANDBOX_IMAGE")


@pytest.mark.skipif(
    shutil.which("docker") is None or not _DOCKER_IMAGE,
    reason="needs docker and CODE_GEN_SANDBOX_IMAGE built from sandbox/",
)
class TestDockerProvider:
    async def test_memory_limit_fails_the_test_not_the_sandbox(self) -> None:
        docker_config = yaml.safe_load(
            (_REPO_ROOT / "nemo_gym/sandbox/providers/docker/configs/docker.yaml").read_text()
        )["sandbox"]["docker"]
        # The harness's 4 GiB RLIMIT_DATA must trip before the container's memory limit does.
        result, metadata = await _grade(
            {"docker": docker_config},
            config=SandboxVerificationConfig(image=_DOCKER_IMAGE, resources={"cpu": 1, "memory_mib": 6144}),
            generation="x = bytearray(8 * 1024**3)\nprint(1)",
        )
        assert result == [-4]
        assert metadata["error"] == "MemoryError()"

    async def test_a_task_memory_limit_is_an_allowance_above_the_baseline(self) -> None:
        docker_config = yaml.safe_load(
            (_REPO_ROOT / "nemo_gym/sandbox/providers/docker/configs/docker.yaml").read_text()
        )["sandbox"]["docker"]
        allocate = "import numpy as np\nx = np.ones({} * 1024**2 // 8)\nn=int(input())\nprint(n*n)"
        for megabytes, expected in [(192, [True, True]), (320, [-4])]:
            result, metadata = await _grade(
                {"docker": docker_config},
                config=SandboxVerificationConfig(image=_DOCKER_IMAGE, resources={"cpu": 1, "memory_mib": 2048}),
                generation=allocate.format(megabytes),
                memory_limit_mb=256,
            )
            assert result == expected, metadata
        assert "MemoryError" in metadata["error"]
        memory = metadata["memory"]
        assert memory["memory_policy"] == "address-space-above-baseline-v1"
        assert memory["effective_address_space_limit_bytes"] - memory["baseline_address_space_bytes"] == 256 * 1024**2
