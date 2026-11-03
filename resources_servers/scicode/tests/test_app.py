# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import hashlib
import subprocess
import sys
import tempfile
import venv
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import resources_servers.scicode.app as app
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from resources_servers.scicode.app import (
    ScicodeResourcesServer,
    ScicodeResourcesServerConfig,
    ScicodeVerifyRequest,
)
from resources_servers.scicode.scicode_integration.runner import (
    DEFAULT_OUTPUT_TAIL_CHARS,
    build_test_program,
    exception_class_from_stderr,
    preflight_grading_interpreter,
    run_substep,
    sanitize_test,
)


def _server(test_data_fpath=None, **config_overrides):
    config = ScicodeResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        test_data_fpath=test_data_fpath,
        **config_overrides,
    )
    # Most server tests isolate grading behavior from external scientific dependencies.
    with patch.object(app, "preflight_grading_interpreter", return_value=None):
        server = ScicodeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._resolve_grading_interpreters()
    return server


def _response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="d",
        object="response",
        output=[
            {
                "id": "m",
                "content": [{"annotations": [], "text": "", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _request(solutions, n_steps=2):
    sub_steps = [{"step_number": f"1.{i + 1}", "test_cases": ["assert True"]} for i in range(n_steps)]
    return ScicodeVerifyRequest(
        responses_create_params={"input": []},
        response=_response(),
        problem_id="1",
        sub_steps=sub_steps,
        solutions=solutions,
    )


@contextmanager
def _mock_substep(passed: bool):
    """Stub sub-step execution so verify() runs without executing code or reading test_data.h5."""
    with patch.object(app, "run_substep", lambda *a, **k: {"passed": passed, "error": ""}):
        yield


def _details(passed: bool, **overrides) -> dict:
    """A runner result with the execution details run_substep reports (see runner.run_substep)."""
    result = {
        "passed": passed,
        "error": "" if passed else "Traceback (most recent call last):\n  ...\nValueError: boom\n",
        "returncode": 0 if passed else 1,
        "timed_out": False,
        "duration_s": 0.01,
        "exception_class": None if passed else "ValueError",
        "stdout_tail": "ok\n" if passed else "",
        "stderr_tail": "" if passed else "Traceback (most recent call last):\n  ...\nValueError: boom\n",
    }
    result.update(overrides)
    return result


@contextmanager
def _mock_substep_details(results_by_solution: dict, calls: list = None):
    """Stub run_substep with full result dicts chosen by the solution code the program starts with;
    `calls` (if given) collects (program, kwargs) per invocation."""

    def fake(program, timeout_secs, python_executable=None, **kwargs):
        if calls is not None:
            calls.append((program, {"timeout_secs": timeout_secs, **kwargs}))
        solution = program.split("\n", 1)[0]
        return results_by_solution[solution]

    with patch.object(app, "run_substep", fake):
        yield


# ----------------------------
# runner helpers
# ----------------------------
def test_sanitize_strips_scicode_imports():
    src = "from scicode.compare.cmp import cmp_tuple_or_list\nimport scicode\nassert f(1) == target"
    assert sanitize_test(src) == "assert f(1) == target"


def test_build_test_program_injects_path_and_targets():
    program = build_test_program("def f(x):\n    return x", "/data/test_data.h5", "1.1", ["assert f(1) == target"])
    assert 'H5PY_FILE = "/data/test_data.h5"' in program
    assert "process_hdf5_to_tuple('1.1', 1)" in program
    assert "target = targets[0]" in program
    assert "def cmp_tuple_or_list" in program


def test_run_substep_pass():
    assert run_substep("assert 1 == 1", timeout_secs=10.0)["passed"] is True


def test_run_substep_fail_returns_stderr():
    result = run_substep("raise ValueError('boom')", timeout_secs=10.0)
    assert result["passed"] is False
    assert "ValueError" in result["error"]


def test_run_substep_timeout():
    result = run_substep("import time\ntime.sleep(5)", timeout_secs=0.5)
    # the three legacy keys, plus the execution details the record carries (no exit code on a timeout)
    assert (result["passed"], result["error"], result["infrastructure_error"]) == (False, "timeout", False)
    assert result["timed_out"] is True and result["returncode"] is None and result["exception_class"] is None
    assert result["duration_s"] >= 0.5


def test_run_substep_accepts_explicit_python_interpreter():
    assert run_substep("assert 1 == 1", timeout_secs=10.0, python_executable=sys.executable)["passed"] is True


@pytest.mark.parametrize(
    "message",
    [
        "Error importing numpy",
        "ModuleNotFoundError: No module named 'numpy'",
        "OpenBLAS blas_thread_init: Resource temporarily unavailable",
    ],
)
def test_generated_error_text_is_not_an_infrastructure_failure(message):
    result = run_substep(f"raise RuntimeError({message!r})", timeout_secs=10.0)
    assert result["passed"] is False
    assert message in result["error"]
    assert result["infrastructure_error"] is False


def test_run_substep_reports_interpreter_launch_error():
    result = run_substep("assert True", timeout_secs=10.0, python_executable="/does/not/exist/python")
    assert result["passed"] is False
    assert result["error"]
    assert result["infrastructure_error"] is True


def test_run_substep_limits_numerical_library_threads():
    with (
        patch.dict(
            "resources_servers.scicode.scicode_integration.runner.os.environ",
            {"PYTHONPATH": "/resource/server", "VIRTUAL_ENV": "/resource/server/.venv"},
        ),
        patch(
            "resources_servers.scicode.scicode_integration.runner.subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b""),
        ) as mocked_run,
    ):
        result = run_substep("assert True", timeout_secs=10.0)

    assert result["passed"] is True
    child_env = mocked_run.call_args.kwargs["env"]
    assert child_env["OPENBLAS_NUM_THREADS"] == "1"
    assert child_env["OMP_NUM_THREADS"] == "1"
    assert child_env["MKL_NUM_THREADS"] == "1"
    assert "PYTHONPATH" not in child_env
    assert "VIRTUAL_ENV" not in child_env


def test_preflight_reports_native_library_initialization_failure():
    stderr = b"OpenBLAS blas_thread_init: pthread_create failed: Resource temporarily unavailable"
    with patch(
        "resources_servers.scicode.scicode_integration.runner.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=stderr),
    ):
        error = preflight_grading_interpreter(sys.executable)

    assert "OpenBLAS blas_thread_init" in error


def test_preflight_reports_missing_required_dependency():
    stderr = b"ModuleNotFoundError: No module named 'numpy'"
    with patch(
        "resources_servers.scicode.scicode_integration.runner.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=stderr),
    ):
        error = preflight_grading_interpreter(sys.executable)

    assert "No module named 'numpy'" in error


def test_preflight_detects_genuinely_missing_numpy(tmp_path):
    empty_env = tmp_path / "empty-grading-env"
    venv.EnvBuilder(with_pip=False).create(empty_env)
    error = preflight_grading_interpreter(str(empty_env / "bin/python"))
    assert "No module named 'numpy'" in error


def test_preflight_reports_missing_interpreter():
    assert preflight_grading_interpreter("/does/not/exist/python")


def test_preflight_timeout_is_an_infrastructure_failure():
    with patch(
        "resources_servers.scicode.scicode_integration.runner.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=[sys.executable], timeout=30),
    ):
        assert "timed out" in preflight_grading_interpreter(sys.executable)


@pytest.mark.parametrize("timeout_secs", [30.0, 1800.0])
def test_preflight_uses_only_trusted_imports_and_sanitized_environment(timeout_secs):
    with (
        patch.dict(
            "resources_servers.scicode.scicode_integration.runner.os.environ",
            {"PYTHONPATH": "/resource/server", "VIRTUAL_ENV": "/resource/server/.venv"},
        ),
        patch(
            "resources_servers.scicode.scicode_integration.runner.subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b""),
        ) as mocked_run,
    ):
        assert preflight_grading_interpreter(sys.executable, timeout_secs=timeout_secs) is None
    command = mocked_run.call_args.args[0]
    assert command[:2] == [sys.executable, "-c"]
    assert "import numpy" in command[2]
    assert "import h5py" in command[2]
    assert "matplotlib" not in command[2]  # Optional solution libraries may differ across OR environments.
    assert mocked_run.call_args.kwargs["timeout"] == timeout_secs
    child_env = mocked_run.call_args.kwargs["env"]
    assert child_env["OPENBLAS_NUM_THREADS"] == "1"
    assert "PYTHONPATH" not in child_env
    assert "VIRTUAL_ENV" not in child_env


def test_run_substep_reports_execution_details_on_pass():
    result = run_substep("print('hello from test')\nassert 1 == 1", timeout_secs=10.0)
    assert result["passed"] is True
    assert result["error"] == ""
    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert result["duration_s"] >= 0.0
    assert "hello from test" in result["stdout_tail"]
    assert result["stderr_tail"] == ""
    assert result["exception_class"] is None


def test_run_substep_reports_execution_details_on_failure():
    result = run_substep("raise ValueError('boom')", timeout_secs=10.0)
    assert result["passed"] is False
    assert result["returncode"] == 1
    assert result["timed_out"] is False
    assert "ValueError" in result["stderr_tail"]
    assert "ValueError" in result["error"]  # the legacy key keeps main's bounded stderr
    assert result["exception_class"] == "ValueError"


def test_run_substep_output_tails_are_bounded():
    program = "import sys\nprint('x' * 5000)\nprint('y' * 5000, file=sys.stderr)\nraise SystemExit(3)"
    result = run_substep(program, timeout_secs=10.0, output_tail_chars=100)
    assert result["returncode"] == 3
    assert len(result["stdout_tail"]) == 100
    assert len(result["stderr_tail"]) == 100
    assert result["exception_class"] is None  # no traceback, just a non-zero exit


@pytest.mark.parametrize(
    "stderr, expected",
    [
        ("Traceback (most recent call last):\n  File x\nValueError: boom\n", "ValueError"),
        (
            "Traceback (most recent call last):\n  File x\nnumpy.linalg.LinAlgError: Singular matrix\n",
            "numpy.linalg.LinAlgError",
        ),
        ("Traceback (most recent call last):\n  File x\nAssertionError\n", "AssertionError"),
        ('  File "<string>", line 1\n    def f(:\n          ^\nSyntaxError: invalid syntax\n', "SyntaxError"),
        ("", None),
        ("timeout", None),
        ("some warning text without a traceback\n", None),
        ("ValueError: a message\nwith a second line\n", None),
    ],
)
def test_exception_class_from_stderr(stderr, expected):
    assert exception_class_from_stderr(stderr) == expected


def test_run_substep_zero_tail_retains_nothing():
    """output_tail_chars=0 means "keep no stream text": both tails empty on a plain failure and on a timeout
    (text[-0:] would be the whole string)."""
    program = "import sys\nprint('out')\nprint('err', file=sys.stderr)\nsys.exit(3)"
    result = run_substep(program, timeout_secs=10.0, output_tail_chars=0)
    assert result["passed"] is False and result["returncode"] == 3
    assert result["stdout_tail"] == "" and result["stderr_tail"] == ""
    slow = "import sys, time\nprint('out', flush=True)\nprint('err', file=sys.stderr, flush=True)\ntime.sleep(5)"
    result = run_substep(slow, timeout_secs=0.5, output_tail_chars=0)
    assert result["timed_out"] is True and result["returncode"] is None and result["error"] == "timeout"
    assert result["stdout_tail"] == "" and result["stderr_tail"] == ""


@pytest.mark.parametrize(
    "tail_chars, expected_len", [(DEFAULT_OUTPUT_TAIL_CHARS, DEFAULT_OUTPUT_TAIL_CHARS), (5, 5), (0, 0)]
)
def test_run_substep_classifies_the_exception_from_the_full_stderr(tail_chars, expected_len):
    """The exception class must not depend on the retention bound: a message longer than the tail still
    yields its class, while the stored tail stays exactly as long as configured."""
    result = run_substep("raise ValueError('x' * 3000)", timeout_secs=10.0, output_tail_chars=tail_chars)
    assert result["passed"] is False and result["returncode"] == 1
    assert result["exception_class"] == "ValueError"
    assert len(result["stderr_tail"]) == expected_len


def test_run_substep_null_byte_fails_instead_of_raising():
    result = run_substep("x = 1\n\0\nassert x == 1", timeout_secs=10.0)
    assert result["passed"] is False
    assert "null bytes" in result["error"]


def test_run_substep_program_larger_than_arg_max():
    program = f"payload = {'a' * 4_000_000!r}\nassert len(payload) == 4_000_000"
    assert run_substep(program, timeout_secs=30.0)["passed"] is True


# ----------------------------
# server
# ----------------------------
class TestApp:
    def test_preflight_failure_stops_server_before_model_code(self):
        config = ScicodeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            grading_interpreters=[{"name": "broken", "python_executable": sys.executable}],
        )
        with (
            patch.object(app, "preflight_grading_interpreter", return_value="missing numpy"),
            patch.object(app, "run_substep") as generated_program,
            pytest.raises(RuntimeError, match="preflight failed for 'broken'"),
        ):
            ScicodeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        generated_program.assert_not_called()

    def test_preflight_runs_once_per_resolved_interpreter(self):
        config = ScicodeResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="", timeout_secs=1800.0)
        with patch.object(app, "preflight_grading_interpreter", return_value=None) as checked:
            server = ScicodeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
            server._resolve_grading_interpreters()
            server._resolve_grading_interpreters()
        checked.assert_called_once_with(sys.executable, timeout_secs=1800.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("second_passes", [False, True])
    async def test_verify_generated_error_marker_does_not_abort_or_grading(self, tmp_path, second_passes):
        targets = tmp_path / "targets.h5"
        targets.write_bytes(b"fixture")
        server = _server(
            test_data_fpath=str(targets),
            grading_interpreters=[
                {"name": "2024", "python_executable": sys.executable},
                {"name": "2025", "python_executable": sys.executable},
            ],
            required_grading_interpreters=2,
        )
        # Use real subprocess results for a model-controlled traceback and a healthy program.
        failed = run_substep('raise RuntimeError("Error importing numpy")', timeout_secs=10.0)
        second = run_substep("assert True", timeout_secs=10.0) if second_passes else failed
        with patch.object(app, "run_substep", side_effect=[failed, second]) as mocked_run:
            result = await server.verify(_request(solutions={"1.1": "model code"}, n_steps=1))
        assert mocked_run.call_count == 2
        assert result.reward == float(second_passes)
        assert result.step_environment_results == [{"2024": False, "2025": second_passes}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("has_solution", [False, True])
    async def test_verify_preserves_step_usage(self, has_solution):
        request_json = _request(solutions={"1.1": "a"} if has_solution else None).model_dump()
        request_json["token_usage_version"] = 1
        request_json["step_usage"] = [
            {
                "step_number": "1.1",
                "status": "generated" if has_solution else "skipped",
                "usage": None,
            }
        ]
        request = ScicodeVerifyRequest.model_validate(request_json)
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep(passed=True):
            result = (await _server(h5.name).verify(request)).model_dump()
        assert result["token_usage_version"] == 1
        assert result["step_usage"] == request_json["step_usage"]
        assert result["reward"] == float(has_solution)

    def test_sanity(self):
        _server()

    def test_config_defaults(self):
        config = _server().config
        assert config.num_processes == 20
        assert config.timeout_secs == 30.0
        assert config.test_data_fpath is None
        assert config.test_data_md5 is None
        assert config.grading_interpreters == []
        assert config.required_grading_interpreters == 1

    @pytest.mark.asyncio
    async def test_verify_no_solutions_returns_zero(self):
        result = await _server().verify(_request(solutions=None))
        assert result.reward == 0.0
        assert result.num_steps_total == 0
        assert result.num_steps_passed == 0
        assert result.problem_accuracy is False

    @pytest.mark.asyncio
    async def test_verify_excludes_steps_absent_from_solutions(self):
        # Step 1.2 has no solution entry (prefilled) -> excluded from the denominator entirely.
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep(passed=True):
            result = await _server(h5.name).verify(_request(solutions={"1.1": "a"}, n_steps=2))
        assert result.num_steps_total == 1
        assert result.num_steps_passed == 1
        assert result.reward == 1.0
        assert result.problem_accuracy is True

    @pytest.mark.asyncio
    async def test_verify_unconfigured_test_data_raises(self):
        with pytest.raises(RuntimeError, match="not configured"):
            await _server().verify(_request(solutions={"1.1": "x = 1", "1.2": "y = 2"}))

    @pytest.mark.asyncio
    async def test_verify_missing_test_data_raises(self):
        server = _server(test_data_fpath="/nonexistent/test_data.h5")
        with pytest.raises(RuntimeError, match="not found"):
            await server.verify(_request(solutions={"1.1": "x = 1", "1.2": "y = 2"}))

    def test_server_rejects_test_data_checksum_mismatch_at_startup(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5:
            h5.write(b"not-the-verified-release")
            h5.flush()
            with pytest.raises(RuntimeError, match="checksum mismatch"):
                _server(test_data_fpath=h5.name, test_data_md5="0" * 32)

    @pytest.mark.asyncio
    async def test_server_caches_valid_test_data_checksum(self, tmp_path):
        targets = tmp_path / "targets.h5"
        targets.write_bytes(b"verified-targets")
        expected_md5 = hashlib.md5(targets.read_bytes()).hexdigest()  # noqa: S324 - fixture integrity
        server = _server(test_data_fpath=str(targets), test_data_md5=expected_md5)

        with _mock_substep(passed=True):
            result = await server.verify(_request(solutions={"1.1": "x = 1"}, n_steps=1))

        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_verify_relative_test_data_resolved_under_gym_root(self):
        from nemo_gym import PARENT_DIR

        server = _server(test_data_fpath="nonexistent/test_data.h5")
        with pytest.raises(RuntimeError, match=str(PARENT_DIR)):
            await server.verify(_request(solutions={"1.1": "x = 1", "1.2": "y = 2"}))

    @pytest.mark.asyncio
    async def test_verify_all_pass(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep(passed=True):
            result = await _server(h5.name).verify(_request(solutions={"1.1": "a", "1.2": "b"}))
        assert result.reward == 1.0
        assert result.step_results == [True, True]
        assert result.num_steps_passed == 2
        assert result.problem_accuracy is True
        assert result.subtask_accuracy == 1.0
        assert result.problem_id == "1"  # preserved into the rollout output

    @pytest.mark.asyncio
    async def test_verify_all_fail(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep(passed=False):
            result = await _server(h5.name).verify(_request(solutions={"1.1": "a", "1.2": "b"}))
        assert result.reward == 0.0
        assert result.num_steps_passed == 0
        assert result.problem_accuracy is False
        assert result.subtask_accuracy == 0.0

    @pytest.mark.asyncio
    async def test_verify_out_of_context_step_fails(self):
        # First sub-step runs (mocked pass); second is the out-of-context sentinel -> fails unrun.
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep(passed=True):
            result = await _server(h5.name).verify(_request(solutions={"1.1": "a", "1.2": "_ran_out_of_context_"}))
        assert result.step_results == [True, False]
        assert result.num_steps_passed == 1
        assert result.reward == 0.0
        assert result.subtask_accuracy == 0.5  # per-rollout sub-step pass fraction

    @pytest.mark.asyncio
    async def test_verify_uses_multi_environment_or(self, tmp_path):
        old_python = tmp_path / "python-2024"
        new_python = tmp_path / "python-2025"
        for executable in (old_python, new_python):
            executable.write_text("#!/bin/sh\n")
            executable.chmod(0o755)

        server = _server(
            test_data_fpath=str(tmp_path / "targets.h5"),
            grading_interpreters=[
                {"name": "2024", "python_executable": str(old_python)},
                {"name": "2025", "python_executable": str(new_python)},
            ],
            required_grading_interpreters=2,
        )
        (tmp_path / "targets.h5").write_bytes(b"fixture")

        with patch.object(
            app,
            "run_substep",
            side_effect=[{"passed": False, "error": "old"}, {"passed": True, "error": ""}],
        ) as mocked_run:
            result = await server.verify(_request(solutions={"1.1": "x = 1"}, n_steps=1))

        assert mocked_run.call_count == 2
        assert result.reward == 1.0
        assert result.step_results == [True]
        assert result.scored_step_ids == ["1.1"]
        assert result.step_environment_results == [{"2024": False, "2025": True}]
        assert result.step_environment_errors == [{"2024": "old"}]

    @pytest.mark.asyncio
    async def test_verify_raises_on_grading_infrastructure_error(self, tmp_path):
        python_path = tmp_path / "python"
        python_path.write_text("#!/bin/sh\n")
        python_path.chmod(0o755)
        targets = tmp_path / "targets.h5"
        targets.write_bytes(b"fixture")
        server = _server(
            test_data_fpath=str(targets),
            grading_interpreters=[{"name": "2024", "python_executable": str(python_path)}],
        )

        failure = {
            "passed": False,
            "error": "OpenBLAS blas_thread_init: Resource temporarily unavailable",
            "infrastructure_error": True,
        }
        with (
            patch.object(app, "run_substep", return_value=failure),
            pytest.raises(RuntimeError, match="grading infrastructure failure"),
        ):
            await server.verify(_request(solutions={"1.1": "x = 1"}, n_steps=1))

    @pytest.mark.asyncio
    async def test_verify_multi_environment_short_circuits_after_pass(self, tmp_path):
        python_path = tmp_path / "python"
        python_path.write_text("#!/bin/sh\n")
        python_path.chmod(0o755)
        targets = tmp_path / "targets.h5"
        targets.write_bytes(b"fixture")
        server = _server(
            test_data_fpath=str(targets),
            grading_interpreters=[
                {"name": "2024", "python_executable": str(python_path)},
                {"name": "2025", "python_executable": str(python_path)},
            ],
            required_grading_interpreters=2,
        )

        with patch.object(app, "run_substep", return_value={"passed": True, "error": ""}) as mocked_run:
            result = await server.verify(_request(solutions={"1.1": "x = 1"}, n_steps=1))

        assert mocked_run.call_count == 1
        assert result.step_environment_results == [{"2024": True}]

    def test_server_requires_configured_interpreters_at_startup(self, tmp_path):
        targets = tmp_path / "targets.h5"
        targets.write_bytes(b"fixture")

        with pytest.raises(RuntimeError, match="at least 2 grading interpreters"):
            _server(test_data_fpath=str(targets), required_grading_interpreters=2)

    def test_server_rejects_duplicate_interpreter_names(self):
        with pytest.raises(RuntimeError, match="Duplicate SciCode grading interpreter name"):
            _server(
                grading_interpreters=[
                    {"name": "same", "python_executable": sys.executable},
                    {"name": "same", "python_executable": sys.executable},
                ],
                required_grading_interpreters=2,
            )

    def test_server_resolves_named_and_relative_interpreters(self, monkeypatch, tmp_path):
        relative_python = tmp_path / "env" / "bin" / "python"
        relative_python.parent.mkdir(parents=True)
        relative_python.write_text("#!/bin/sh\n")
        relative_python.chmod(0o755)
        monkeypatch.setattr(app, "PARENT_DIR", tmp_path)
        monkeypatch.setattr(app.shutil, "which", lambda name: sys.executable if name == "python-on-path" else None)

        server = _server(
            grading_interpreters=[
                {"name": "relative", "python_executable": "env/bin/python"},
                {"name": "path", "python_executable": "python-on-path"},
            ],
            required_grading_interpreters=2,
        )

        assert server._resolve_grading_interpreters() == [
            ("relative", str(relative_python.resolve())),
            ("path", sys.executable),
        ]

    def test_server_preserves_virtualenv_python_symlink(self, tmp_path):
        base_python = tmp_path / "python3.12"
        base_python.write_text("#!/bin/sh\n")
        base_python.chmod(0o755)
        venv_python = tmp_path / "grading-env" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.symlink_to(base_python)

        server = _server(grading_interpreters=[{"name": "frozen", "python_executable": str(venv_python)}])

        assert server._resolve_grading_interpreters() == [("frozen", str(venv_python.absolute()))]

    def test_server_rejects_non_executable_interpreter(self, tmp_path):
        not_executable = tmp_path / "python"
        not_executable.write_text("#!/bin/sh\n")

        with pytest.raises(RuntimeError, match="is not executable"):
            _server(grading_interpreters=[{"name": "broken", "python_executable": str(not_executable)}])

    @pytest.mark.asyncio
    async def test_verify_without_the_flag_has_no_step_outputs(self):
        results = {"a": _details(True), "b": _details(False)}
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep_details(results):
            result = (await _server(h5.name).verify(_request(solutions={"1.1": "a", "1.2": "b"}))).model_dump()
        assert result["step_results"] == [True, False]
        assert result["step_outputs"] is None

    @pytest.mark.asyncio
    async def test_verify_with_the_flag_returns_step_outputs_aligned_with_step_results(self):
        results = {"a": _details(True), "b": _details(False)}
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep_details(results):
            server = _server(h5.name, record_step_outputs=True)
            result = await server.verify(_request(solutions={"1.1": "a", "1.2": "b"}))
        assert result.step_results == [True, False]
        assert [o["step_id"] for o in result.step_outputs] == ["1.1", "1.2"]
        assert [o["step_number"] for o in result.step_outputs] == ["1.1", "1.2"]
        assert [o["passed"] for o in result.step_outputs] == [True, False]
        ok, bad = result.step_outputs
        assert ok["environment"] == "current" and bad["environment"] == "current"
        assert ok["status"] == "executed" and ok["returncode"] == 0 and ok["stdout_tail"] == "ok\n"
        assert bad["status"] == "executed" and bad["returncode"] == 1 and bad["timed_out"] is False
        assert bad["exception_class"] == "ValueError" and "ValueError" in bad["stderr_tail"]
        assert bad["duration_s"] == 0.01
        # the summary numbers are what they were without the flag
        assert result.num_steps_passed == 1 and result.num_steps_total == 2 and result.reward == 0.0

    @pytest.mark.asyncio
    async def test_verify_with_the_flag_marks_out_of_context_steps_without_running_them(self):
        calls = []
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep_details({"a": _details(True)}, calls):
            server = _server(h5.name, record_step_outputs=True)
            result = await server.verify(_request(solutions={"1.1": "a", "1.2": "_ran_out_of_context_"}))
        assert len(calls) == 1  # the sentinel step never reaches the runner
        assert result.step_results == [True, False]
        assert result.step_outputs[1] == {
            "step_id": "1.2",
            "step_number": "1.2",
            "environment": None,
            "status": "out_of_context",
            "passed": False,
            "returncode": None,
            "timed_out": False,
            "duration_s": None,
            "exception_class": None,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    @pytest.mark.asyncio
    async def test_verify_passes_the_configured_tail_length_to_the_runner(self):
        calls = []
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep_details({"a": _details(True)}, calls):
            server = _server(h5.name, record_step_outputs=True, output_tail_chars=50)
            await server.verify(_request(solutions={"1.1": "a"}, n_steps=1))
        assert calls[0][1] == {"timeout_secs": 30.0, "output_tail_chars": 50}

    @pytest.mark.asyncio
    async def test_verify_with_the_flag_tolerates_a_runner_result_without_details(self):
        # A stub (or an older runner) that returns only passed/error still yields a well-formed record.
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep(passed=False):
            server = _server(h5.name, record_step_outputs=True)
            result = await server.verify(_request(solutions={"1.1": "a"}, n_steps=1))
        assert result.step_outputs == [
            {
                "step_id": "1.1",
                "step_number": "1.1",
                "environment": "current",
                "status": "executed",
                "passed": False,
                "returncode": None,
                "timed_out": False,
                "duration_s": None,
                "exception_class": None,
                "stdout_tail": "",
                "stderr_tail": "",
            }
        ]

    @pytest.mark.asyncio
    async def test_verify_with_the_flag_and_no_solutions_returns_an_empty_list(self):
        result = await _server(record_step_outputs=True).verify(_request(solutions=None))
        assert result.step_outputs == []
        assert result.num_steps_total == 0

    def test_config_step_code_default(self):
        assert _server().config.record_step_code is False

    @pytest.mark.asyncio
    async def test_verify_with_step_code_records_the_program_each_sub_step_ran_against(self):
        results = {"a": _details(True), "b": _details(False)}
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep_details(results):
            server = _server(h5.name, record_step_outputs=True, record_step_code=True)
            result = await server.verify(_request(solutions={"1.1": "a", "1.2": "b"}))
        assert [o["code"] for o in result.step_outputs] == ["a", "b"]
        assert [o["passed"] for o in result.step_outputs] == [True, False]

    @pytest.mark.asyncio
    async def test_verify_with_step_code_leaves_sentinel_steps_without_code(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep_details({"a": _details(True)}):
            server = _server(h5.name, record_step_outputs=True, record_step_code=True)
            result = await server.verify(_request(solutions={"1.1": "a", "1.2": "_ran_out_of_context_"}))
        assert result.step_outputs[0]["code"] == "a"
        assert result.step_outputs[1]["status"] == "out_of_context"
        assert result.step_outputs[1]["code"] is None

    @pytest.mark.asyncio
    async def test_verify_without_step_code_records_carry_no_code_key(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep_details({"a": _details(True)}):
            server = _server(h5.name, record_step_outputs=True)
            result = await server.verify(_request(solutions={"1.1": "a"}, n_steps=1))
        assert "code" not in result.step_outputs[0]

    @pytest.mark.asyncio
    async def test_step_code_alone_does_not_turn_on_step_outputs(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_substep(passed=True):
            server = _server(h5.name, record_step_code=True)
            result = await server.verify(_request(solutions={"1.1": "a"}, n_steps=1))
        assert result.step_outputs is None

    def test_config_step_record_defaults(self):
        config = _server().config
        assert config.record_step_outputs is False
        assert config.output_tail_chars == 2000
        assert config.record_step_code is False

    def test_config_rejects_a_negative_output_tail(self):
        """A negative tail would keep the whole stream (the runner's unbounded branch); the public config
        accepts 0 (keep nothing) and positive bounds only."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            _server(output_tail_chars=-1)
        assert _server(output_tail_chars=0).config.output_tail_chars == 0

    @pytest.mark.asyncio
    async def test_verify_record_names_the_environment_that_decided_the_step(self, tmp_path):
        """Under the OR rule the record describes the passing environment, else the last one tried; the
        per-environment pass bits stay where SciCode-Verified put them."""
        results = {
            "a": {"py_a": _details(False), "py_b": _details(True)},
            "b": {"py_a": _details(False), "py_b": _details(False)},
        }

        def fake(program, timeout_secs, python_executable=None, **kwargs):
            solution = program.split("\n", 1)[0]
            name = "py_a" if python_executable.endswith("py_a") else "py_b"
            return results[solution][name]

        (tmp_path / "py_a").write_text("#!/bin/sh\n")
        (tmp_path / "py_b").write_text("#!/bin/sh\n")
        for name in ("py_a", "py_b"):
            (tmp_path / name).chmod(0o755)
        interpreters = [
            {"name": "py_a", "python_executable": str(tmp_path / "py_a")},
            {"name": "py_b", "python_executable": str(tmp_path / "py_b")},
        ]
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, patch.object(app, "run_substep", fake):
            server = _server(h5.name, record_step_outputs=True, grading_interpreters=interpreters)
            result = await server.verify(_request(solutions={"1.1": "a", "1.2": "b"}))
        assert result.step_results == [True, False]
        assert result.step_environment_results == [{"py_a": False, "py_b": True}, {"py_a": False, "py_b": False}]
        assert [o["environment"] for o in result.step_outputs] == ["py_b", "py_b"]
        assert [o["passed"] for o in result.step_outputs] == [True, False]
