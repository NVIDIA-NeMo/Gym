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
"""Tests for grading competitive-coding submissions in a provider sandbox."""

import asyncio
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from nemo_gym.core import VERIFIER_ERROR
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.verification import RUNNER_FILE, SandboxVerificationError
from resources_servers.competitive_coding_challenges.ccc_eval import _load_metadata_file
from resources_servers.competitive_coding_challenges.sandbox_evaluator import (
    _HARNESS_FILES,
    _RUNNER,
    CCCSandboxConfig,
    evaluate_in_sandbox,
    problem_file,
)
from resources_servers.competitive_coding_challenges.split_problem_metadata import split_problem_metadata


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CCC_DIR = Path(__file__).resolve().parents[1]
_EVALUATOR_CONFIG = {"test_batch_size": 2, "time_scale": 1.0, "run_all_tests": False}
_RESULT = {"test_case_results": {"01-a": {"score": 5.0, "outputs": [{"test_name": "t1", "score": 1.0}]}}}


def _toy_metadata(compile_script: str, run_script: str, task_type: str = "Batch") -> dict:
    """One problem, two subtasks; tests carry their expected output for run.sh to compare."""
    return {
        "competition_id": "comp",
        "metadata": {
            "toy": {
                "name": "toy",
                "compile": compile_script,
                "run": run_script,
                "grader_files": [["graders/grader_config.json", json.dumps({"task_type": task_type})]],
                "subtasks": {
                    "01-a": {"aggregation": "min", "score": 5.0, "score_precision": 0, "test_names": ["t1", "t2"]},
                    "02-b": {"aggregation": "min", "score": 7.0, "score_precision": 0, "test_names": ["t3"]},
                },
                "all_tests": {
                    "t1": {"input": "2 3\n", "output": "5\n"},
                    "t2": {"input": "10 20\n", "output": "30\n"},
                    "t3": {"input": "1 1\n", "output": "2\n"},
                },
            }
        },
    }


def _entry(code: str, subtask: str = "01-a") -> dict:
    return {
        "competition_id": "comp",
        "name": "toy",
        "problem_id": "toy",
        "subtask": subtask,
        "generation": f"Here you go.\n```cpp\n{code}\n```",
    }


class _FakeProvider:
    """Records what the sandbox path asks of a provider and returns a canned result file."""

    name = "fake"

    def __init__(self, *, result_json=json.dumps(_RESULT)) -> None:
        self._result_json = result_json
        self.specs, self.uploads, self.execs, self.closed = [], [], [], 0

    async def create(self, spec):
        self.specs.append(spec)
        return SandboxHandle(sandbox_id="fake-1", provider_name=self.name, raw={})

    async def upload_file(self, handle, source_path, target_path):
        self.uploads.append((target_path, Path(source_path).read_text()))

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.execs.append((command, timeout_s))
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def download_file(self, handle, source_path, target_path):
        Path(target_path).write_text(self._result_json)

    async def close(self, handle):
        self.closed += 1

    async def aclose(self):
        pass


async def _grade_with(provider, config=None, entry=None):
    return await evaluate_in_sandbox(
        entry or _entry("int main() {}"),
        evaluator_config=_EVALUATOR_CONFIG,
        provider=provider,
        config=config or CCCSandboxConfig(image="ccc-verifier:test", resources={"cpu": 4}),
        metadata={"benchmark": "ioi"},
    )


class TestSandboxLifecycle:
    async def test_request_spec_and_problem_file(self) -> None:
        provider = _FakeProvider()
        result = await _grade_with(provider)

        assert result == _RESULT
        (spec,) = provider.specs
        assert spec.image == "ccc-verifier:test"
        assert spec.env["OMP_NUM_THREADS"] == "4"
        uploads = dict(provider.uploads)
        # Each verification gets its own directory under the default work dir.
        call_dir = next(iter(uploads)).removeprefix("/tmp/ccc_verify/").split("/")[0]
        assert set(uploads) == {
            f"/tmp/ccc_verify/{call_dir}/ccc_eval.py",
            f"/tmp/ccc_verify/{call_dir}/sandbox_runner.py",
            f"/tmp/ccc_verify/{call_dir}/request.json",
        }
        request = json.loads(uploads[f"/tmp/ccc_verify/{call_dir}/request.json"])
        # Only identifiers travel with the request; test data stays in problem_data_dir.
        assert request["problem_file"] == "/data/ccc/comp/toy.jsonl"
        assert request["config"] == _EVALUATOR_CONFIG
        (command, timeout_s), *_ = provider.execs
        assert "sandbox_runner.py request.json result.json" in command
        assert timeout_s == 3600
        assert spec.ttl_s > spec.ready_timeout_s + timeout_s
        assert provider.closed == 1

    async def test_result_without_subtask_scores_is_verifier_error(self) -> None:
        provider = _FakeProvider(result_json='{"test_case_results": {"01-a": {"outputs": []}}}')
        with pytest.raises(SandboxVerificationError) as info:
            await _grade_with(provider)
        assert info.value.failure_kind == VERIFIER_ERROR
        assert provider.closed == 1


def test_problem_file_sanitizes_identifiers() -> None:
    assert problem_file("/data", "ioi 26", "a/b") == "/data/ioi_26/a_b.jsonl"
    assert problem_file("/data", None, "toy") == "/data/_/toy.jsonl"


def test_split_problem_metadata_round_trips(tmp_path: Path) -> None:
    test_file = tmp_path / "metadata.json"
    test_file.write_text(json.dumps(_toy_metadata("true", "echo 1")))

    assert split_problem_metadata(str(test_file), tmp_path / "problems") == 1
    by_competition, _ = _load_metadata_file(problem_file(str(tmp_path / "problems"), "comp", "toy"))
    original, _ = _load_metadata_file(str(test_file))
    assert by_competition == original


def test_split_problem_metadata_streams_like_the_evaluator_load(tmp_path: Path) -> None:
    # Several lines, two competitions, and one problem repeated: the later line must win, as it
    # does when the evaluator loads the whole file.
    first = _toy_metadata("true", "echo 1")
    other = {"competition_id": "other", "metadata": {"toy2": _toy_metadata("true", "echo 2")["metadata"]["toy"]}}
    replaced = _toy_metadata("true", "echo 3")
    test_file = tmp_path / "metadata.jsonl"
    test_file.write_text("\n".join(json.dumps(entry) for entry in (first, other, replaced)) + "\n\n")

    assert split_problem_metadata(str(test_file), tmp_path / "problems") == 2
    original, _ = _load_metadata_file(str(test_file))
    for competition_id, problem_id in (("comp", "toy"), ("other", "toy2")):
        split, _ = _load_metadata_file(problem_file(str(tmp_path / "problems"), competition_id, problem_id))
        assert split == {competition_id: {problem_id: original[competition_id][problem_id]}}
    assert original["comp"]["toy"]["run"] == "echo 3"


def test_uploaded_harness_imports_with_the_standard_library_only(tmp_path: Path) -> None:
    # The image has only Python's standard library, and the runner imports the uploaded ccc_eval as a top-level module.
    shutil.copy(_RUNNER, tmp_path / RUNNER_FILE)
    for name in _HARNESS_FILES:
        shutil.copy(_CCC_DIR / name, tmp_path / name)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import sandbox_runner",
            tmp_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


class TestProcessSandbox:
    """The executor that replaces the NeMo Skills sandbox inside the verification sandbox."""

    @pytest.fixture(scope="class")
    def process_sandbox(self):
        sys.path.insert(0, str(_CCC_DIR))
        try:
            spec = importlib.util.spec_from_file_location("ccc_sandbox_runner", _RUNNER)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(_CCC_DIR))
        return module.ProcessSandbox()

    def _run(self, sandbox, command: str, **kwargs) -> dict:
        return asyncio.run(sandbox.execute_code(command, language="shell", **kwargs))[0]

    def test_status_follows_exit_code(self, process_sandbox) -> None:
        assert self._run(process_sandbox, "echo ok", timeout=10) == {
            "process_status": "completed",
            "stdout": "ok\n",
            "stderr": "",
        }
        assert self._run(process_sandbox, "echo bad >&2; exit 3", timeout=10)["process_status"] == "error"

    def test_timeout_kills_the_process_group(self, process_sandbox, tmp_path: Path) -> None:
        marker = tmp_path / "survived"
        result = self._run(process_sandbox, f"(sleep 2; touch {marker}) & sleep 30", timeout=0.5)
        assert result == {
            "process_status": "timeout",
            "stdout": "",
            "stderr": "Execution timed out after 0.5 seconds\n",
        }
        # The background child was in bash's process group, so it died before touching the marker.
        time.sleep(2.5)
        assert not marker.exists()

    @pytest.mark.skipif(shutil.which("timeout") is None, reason="needs GNU timeout")
    def test_timeout_does_not_wait_for_processes_outside_the_group(self, process_sandbox) -> None:
        # run.sh's `timeout` moves the submission into its own process group, which killpg does not reach.
        start = time.monotonic()
        result = self._run(process_sandbox, "timeout 3 sleep 3", timeout=0.5)
        assert result["process_status"] == "timeout"
        assert time.monotonic() - start < 2.5

    def test_output_is_cut_like_the_skills_sandbox(self, process_sandbox) -> None:
        result = self._run(process_sandbox, "printf '%05000d' 0", timeout=10, max_output_characters=10)
        assert result["stdout"] == "0" * 10 + "<output cut>"

    def test_applies_the_skills_memory_rlimits(self, process_sandbox) -> None:
        result = self._run(process_sandbox, "ulimit -v; ulimit -d", timeout=10)
        assert result["stdout"].split() == [str(50 * 1024 * 1024)] * 2


# compile.sh/run.sh in the shape the IOI scripts use, without needing a compiler on the test host:
# the "binary" is the submitted source, and a test passes if it contains the expected sum.
_TOY_COMPILE = "#!/bin/bash\ncp graders/toy.cpp compiled_solution\n"
_TOY_RUN = (
    "#!/bin/bash\n"
    'expected=$(tr -d "\\n" < correct_output.txt)\n'
    'if grep -q "answer=$expected" compiled_solution; then echo 1; else echo 0; fi\n'
)


class TestLocalProvider:
    """The real runner and unchanged evaluator through the unisolated `local` provider."""

    async def _grade_locally(
        self, tmp_path: Path, code: str, subtask: str = "01-a", run_script: str = _TOY_RUN
    ) -> dict:
        test_file = tmp_path / "metadata.json"
        test_file.write_text(json.dumps(_toy_metadata(_TOY_COMPILE, run_script)))
        split_problem_metadata(str(test_file), tmp_path / "problems")
        config = CCCSandboxConfig(python=sys.executable, problem_data_dir=str(tmp_path / "problems"))
        return await evaluate_in_sandbox(
            _entry(code, subtask),
            evaluator_config=_EVALUATOR_CONFIG,
            provider={"local": {}},
            config=config,
            metadata={},
            work_dir=str(tmp_path / "work"),
        )

    async def test_passing_submission_scores_the_target_subtask(self, tmp_path: Path) -> None:
        result = await self._grade_locally(tmp_path, "// answer=5 answer=30")
        subtask = result["test_case_results"]["01-a"]
        assert subtask["score"] == 5.0
        assert sorted(output["test_name"] for output in subtask["outputs"]) == ["t1", "t2"]
        # Only the target subtask's tests run.
        assert result["test_case_results"]["02-b"]["outputs"] == []

    async def test_first_failure_stops_a_min_subtask(self, tmp_path: Path) -> None:
        result = await self._grade_locally(tmp_path, "// answer=5")
        subtask = result["test_case_results"]["01-a"]
        assert subtask["score"] == 0.0
        assert {output["test_name"]: output["score"] for output in subtask["outputs"]}["t2"] == 0.0

    @pytest.mark.skipif(not Path("/proc/self/limits").exists(), reason="reads /proc/self/limits")
    @pytest.mark.skipif(
        resource.getrlimit(resource.RLIMIT_CORE)[1] == 0,
        reason="the hard RLIMIT_CORE is 0, so the runner cannot set it to one byte",
    )
    async def test_submissions_cannot_dump_core(self, tmp_path: Path) -> None:
        # A one-byte RLIMIT_CORE keeps piped core dumps from stalling a crashing submission past its time limit.
        run_script = "#!/bin/bash\n[ \"$(awk '/^Max core file size/ {print $5, $6}' /proc/self/limits)\" = '1 1' ] && echo 1 || echo 0\n"
        result = await self._grade_locally(tmp_path, "// any", run_script=run_script)
        assert result["test_case_results"]["01-a"]["score"] == 5.0

    async def test_missing_problem_data_is_verifier_error(self, tmp_path: Path) -> None:
        config = CCCSandboxConfig(python=sys.executable, problem_data_dir=str(tmp_path / "absent"))
        with pytest.raises(SandboxVerificationError) as info:
            await evaluate_in_sandbox(
                _entry("// answer=5"),
                evaluator_config=_EVALUATOR_CONFIG,
                provider={"local": {}},
                config=config,
                metadata={},
                work_dir=str(tmp_path / "work"),
            )
        assert info.value.failure_kind == VERIFIER_ERROR
        assert "does not exist" in str(info.value)


_DOCKER_IMAGE = os.environ.get("CCC_SANDBOX_IMAGE")
# A real compile and run with the image's g++, under a wall-clock limit like the IOI run.sh.
_GPP_COMPILE = "#!/bin/bash\ng++ -std=gnu++20 -O2 -pipe -o graders/toy graders/toy.cpp\n"
_GPP_RUN = (
    "#!/bin/bash\n"
    "timeout --preserve-status 1 ./graders/toy < input.txt > out.txt\n"
    "if [ $? -ne 0 ]; then echo 0; exit 0; fi\n"
    "diff -bq out.txt correct_output.txt > /dev/null && echo 1 || echo 0\n"
)


@pytest.mark.skipif(
    shutil.which("docker") is None or not _DOCKER_IMAGE,
    reason="needs docker and CCC_SANDBOX_IMAGE built from sandbox/",
)
class TestDockerProvider:
    async def _grade(self, tmp_path: Path, code: str) -> dict:
        test_file = tmp_path / "metadata.json"
        test_file.write_text(json.dumps(_toy_metadata(_GPP_COMPILE, _GPP_RUN)))
        split_problem_metadata(str(test_file), tmp_path / "problems")
        docker_config = yaml.safe_load(
            (_REPO_ROOT / "nemo_gym/sandbox/providers/docker/configs/docker.yaml").read_text()
        )["sandbox"]["docker"]
        config = CCCSandboxConfig(
            image=_DOCKER_IMAGE,
            resources={"cpu": 2, "memory_mib": 4096},
            provider_options={"volumes": [f"{tmp_path / 'problems'}:/data/ccc:ro"]},
        )
        return await evaluate_in_sandbox(
            _entry(code),
            evaluator_config=_EVALUATOR_CONFIG,
            provider={"docker": docker_config},
            config=config,
            metadata={},
        )

    async def test_compiles_and_runs_with_the_image_toolchain(self, tmp_path: Path) -> None:
        result = await self._grade(tmp_path, "int main() { long long a, b; cin >> a >> b; cout << a + b << endl; }")
        assert result["test_case_results"]["01-a"]["score"] == 5.0

    async def test_wall_clock_limit_fails_the_test(self, tmp_path: Path) -> None:
        result = await self._grade(tmp_path, "int main() { while (true) {} }")
        assert result["test_case_results"]["01-a"]["score"] == 0.0
