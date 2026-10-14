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
import tempfile
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import resources_servers.scicode.app as app
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.scicode.app import (
    ScicodeResourcesServer,
    ScicodeResourcesServerConfig,
    ScicodeVerifyRequest,
)
from resources_servers.scicode.scicode_integration.runner import (
    build_test_program,
    exception_class_from_stderr,
    run_substep,
    sanitize_test,
)


def _server(test_data_fpath=None, **config_overrides):
    config = ScicodeResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="", test_data_fpath=test_data_fpath, **config_overrides
    )
    return ScicodeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


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

    def fake(program, timeout_secs, **kwargs):
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
    assert result["passed"] is False
    assert result["error"] == "timeout"
    assert result["timed_out"] is True
    assert result["returncode"] is None
    assert result["exception_class"] is None
    assert result["duration_s"] >= 0.5


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
    assert result["error"] == result["stderr_tail"]  # the legacy key keeps the stderr tail
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
        assert config.record_step_outputs is False
        assert config.output_tail_chars == 2000

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
