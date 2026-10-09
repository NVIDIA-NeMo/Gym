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
from resources_servers.scicode.scicode_integration.runner import build_test_program, run_substep, sanitize_test


def _server(test_data_fpath=None, **config_overrides):
    config = ScicodeResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        test_data_fpath=test_data_fpath,
        **config_overrides,
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
    assert result == {"passed": False, "error": "timeout", "infrastructure_error": False}


def test_run_substep_accepts_explicit_python_interpreter():
    assert run_substep("assert 1 == 1", timeout_secs=10.0, python_executable=sys.executable)["passed"] is True


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


def test_run_substep_classifies_openblas_thread_failure_as_infrastructure_error():
    stderr = b"OpenBLAS blas_thread_init: pthread_create failed: Resource temporarily unavailable"
    with patch(
        "resources_servers.scicode.scicode_integration.runner.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=stderr),
    ):
        result = run_substep("assert True", timeout_secs=10.0)

    assert result["passed"] is False
    assert result["infrastructure_error"] is True


def test_run_substep_classifies_missing_required_dependency_as_infrastructure_error():
    stderr = b"ModuleNotFoundError: No module named 'numpy'"
    with patch(
        "resources_servers.scicode.scicode_integration.runner.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=stderr),
    ):
        result = run_substep("import numpy", timeout_secs=10.0)

    assert result["passed"] is False
    assert result["infrastructure_error"] is True


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
