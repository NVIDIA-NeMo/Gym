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
import sys
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import resources_servers.scicode.app as app
import resources_servers.scicode.scicode_integration.runner as runner
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.scicode.app import (
    ScicodeResourcesServer,
    ScicodeResourcesServerConfig,
    ScicodeVerifyRequest,
)
from resources_servers.scicode.scicode_integration.runner import build_test_program, run_substep, sanitize_test


def _server(test_data_fpath=None):
    config = ScicodeResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="", test_data_fpath=test_data_fpath
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
def _mock_ray(passed: bool):
    """Stub the Ray worker so verify() runs without launching Ray or reading test_data.h5."""

    class _Future:
        pass

    with (
        patch.object(app.run_substep_remote, "remote", lambda *a, **k: _Future()),
        patch.object(app.ray, "get", lambda f: {"passed": passed, "error": ""}),
    ):
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
    assert run_substep("assert 1 == 1", timeout_secs=10.0, interpreter=sys.executable)["passed"] is True


def test_run_substep_fail_returns_stderr():
    result = run_substep("raise ValueError('boom')", timeout_secs=10.0, interpreter=sys.executable)
    assert result["passed"] is False
    assert "ValueError" in result["error"]


def test_run_substep_timeout():
    result = run_substep("import time\ntime.sleep(5)", timeout_secs=0.5, interpreter=sys.executable)
    assert result == {"passed": False, "error": "timeout"}


# FEP-1136: sub-steps must run under the resources-server's interpreter (its venv has h5py /
# scipy<1.14 / sympy), NOT the Ray worker's ambient python. These guard against regressing to
# the silent uniform-0% failure where the program ran in a venv missing the SciCode deps.
def test_run_substep_invokes_provided_interpreter(monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner.os.path, "isfile", lambda p: True)
    run_substep("pass", timeout_secs=5.0, interpreter="/opt/scicode/.venv/bin/python")
    # The program is executed by the *given* interpreter, not the worker's sys.executable.
    assert seen["argv"][:2] == ["/opt/scicode/.venv/bin/python", "-c"]


def test_run_substep_missing_interpreter_returns_clear_error():
    result = run_substep("assert 1 == 1", timeout_secs=5.0, interpreter="/no/such/python")
    assert result["passed"] is False
    assert "interpreter not found" in result["error"]


def test_run_substep_uses_given_interpreter_not_ambient(tmp_path):
    # A distinct (symlinked) path proves run_substep honors the supplied interpreter rather than
    # silently using sys.executable.
    link = tmp_path / "scicode-python"
    link.symlink_to(sys.executable)
    result = run_substep(
        "import sys; assert sys.executable == %r" % str(link), timeout_secs=10.0, interpreter=str(link)
    )
    assert result["passed"] is True


# ----------------------------
# server
# ----------------------------
class TestApp:
    def test_sanity(self):
        _server()

    def test_config_defaults(self):
        config = _server().config
        assert config.num_processes == 20
        assert config.timeout_secs == 30.0
        assert config.test_data_fpath is None

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
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_ray(passed=True):
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
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_ray(passed=True):
            result = await _server(h5.name).verify(_request(solutions={"1.1": "a", "1.2": "b"}))
        assert result.reward == 1.0
        assert result.step_results == [True, True]
        assert result.num_steps_passed == 2
        assert result.problem_accuracy is True
        assert result.problem_id == "1"  # preserved into the rollout output

    @pytest.mark.asyncio
    async def test_verify_all_fail(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_ray(passed=False):
            result = await _server(h5.name).verify(_request(solutions={"1.1": "a", "1.2": "b"}))
        assert result.reward == 0.0
        assert result.num_steps_passed == 0
        assert result.problem_accuracy is False

    @pytest.mark.asyncio
    async def test_verify_out_of_context_step_fails(self):
        # First sub-step runs (mocked pass); second is the out-of-context sentinel -> fails unrun.
        with tempfile.NamedTemporaryFile(suffix=".h5") as h5, _mock_ray(passed=True):
            result = await _server(h5.name).verify(_request(solutions={"1.1": "a", "1.2": "_ran_out_of_context_"}))
        assert result.step_results == [True, False]
        assert result.num_steps_passed == 1
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_verify_dispatches_with_server_interpreter(self):
        # FEP-1136 regression: verify() must dispatch run_substep with the resources-server's
        # OWN interpreter (its venv has h5py/scipy<1.14), not the Ray worker's ambient python.
        captured = {}

        class _Future:
            pass

        def fake_remote(program, timeout, interpreter):
            captured["interpreter"] = interpreter
            return _Future()

        with (
            tempfile.NamedTemporaryFile(suffix=".h5") as h5,
            patch.object(app.run_substep_remote, "remote", fake_remote),
            patch.object(app.ray, "get", lambda f: {"passed": True, "error": ""}),
        ):
            server = _server(h5.name)
            await server.verify(_request(solutions={"1.1": "a"}, n_steps=1))
        assert captured["interpreter"] == sys.executable
