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
"""Python evaluation scripts print per-test results, and a run that could not start its tests has no verdict."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
from swebench.harness.grading import get_logs_eval
from swebench.harness.run_evaluation import make_test_spec

from nemo_gym.client import ServerClient
from nemo_gym.sandbox import SandboxExecResult
from resources_servers.swebench.app import (
    DockerContainer,
    SWEBenchInstanceRequest,
    SwebenchResourcesServer,
    SwebenchResourcesServerConfig,
)
from resources_servers.swebench.swebench_patches import (
    PYTEST_RESULT_SUMMARY,
    run_instance,
    warning_blocking_required_tests,
)


def patch_for(path: str) -> str:
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"


def row(
    repo: str, instance_id: str, version: str, path: str, fail_to_pass: list[str], pass_to_pass: list[str]
) -> dict:
    return {
        "repo": repo,
        "instance_id": instance_id,
        "base_commit": "0" * 40,
        "patch": "",
        "test_patch": patch_for(path),
        "problem_statement": "problem",
        "hints_text": "",
        "created_at": "2019-01-01T00:00:00Z",
        "version": version,
        "FAIL_TO_PASS": json.dumps(fail_to_pass),
        "PASS_TO_PASS": json.dumps(pass_to_pass),
        "environment_setup_commit": "0" * 40,
        "difficulty": "easy",
        "subset": "verified",
        "split": "test",
    }


ASTROPY_HEADER = "astropy/io/fits/tests/test_header.py"
ASTROPY_F2P = f"{ASTROPY_HEADER}::test_shallow_copy"
ASTROPY_P2P = f"{ASTROPY_HEADER}::TestHeaderFunctions::test_card_constructor_default_args"
ASTROPY_ROW = row("astropy/astropy", "astropy__astropy-8707", "3.1", ASTROPY_HEADER, [ASTROPY_F2P], [ASTROPY_P2P])
SPHINX_TEST = "tests/test_ext_autodoc_automodule.py"
SPHINX_F2P = f"{SPHINX_TEST}::test_empty_all"
SPHINX_ROW = row("sphinx-doc/sphinx", "sphinx-doc__sphinx-8595", "3.5", SPHINX_TEST, [SPHINX_F2P], [])

# What pytest prints when astropy turns the DeprecationWarning of a nose-style `setup` into an error: the first tests
# pass, the others error at setup.
NOSE_WARNING_OUTPUT = f"""
==================================== ERRORS ====================================
______ ERROR at setup of TestHeaderFunctions.test_card_constructor_default_args ______
E               pytest.PytestRemovedIn8Warning: Support for nose tests is deprecated and will be removed in a future release.
E               astropy/io/fits/tests/test_header.py::X::test_y is using nose-specific method: setup(self)

/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/_pytest/python.py:898: PytestRemovedIn8Warning
=========================== short test summary info ============================
PASSED {ASTROPY_F2P}
ERROR {ASTROPY_P2P}
======================== 1 passed, 1 error in 16.34s ========================
"""
# What pytest prints when the collection of the test file fails on a DeprecationWarning that became an error.
COLLECTION_WARNING_OUTPUT = f"""
==================================== ERRORS ====================================
_____________ ERROR collecting {ASTROPY_HEADER} _____________
astropy/utils/introspection.py:92: in minversion
    have_version = LooseVersion(have_version)
/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/setuptools/_distutils/version.py:50: in __init__
    warnings.warn(
E   DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
=========================== short test summary info ============================
ERROR {ASTROPY_HEADER} - DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
"""
# The same kind of error, raised by a function of the repo that a model patch changed (a short traceback, then a long one).
REPO_WARNING_OUTPUT = f"""
_____ ERROR at setup of TestHeaderFunctions.test_card_constructor_default_args _____
astropy/io/fits/header.py:210: in update
    warnings.warn("use set instead", DeprecationWarning)
E   DeprecationWarning: use set instead
PASSED {ASTROPY_F2P}
ERROR {ASTROPY_P2P}
"""
REPO_LONG_WARNING_OUTPUT = f"""
_____ ERROR at setup of TestHeaderFunctions.test_card_constructor_default_args _____
E   DeprecationWarning: use set instead

/testbed/astropy/io/fits/header.py:210: DeprecationWarning
PASSED {ASTROPY_F2P}
ERROR {ASTROPY_P2P}
"""
TOX_OUTPUT = f"""
=========================== short test summary info ============================
PASSED {SPHINX_F2P}
============================== 1 passed in 0.50s ===============================
"""


def server() -> SwebenchResourcesServer:
    return SwebenchResourcesServer(
        config=SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            sandbox_provider="test",
            sandbox_config={},
            is_verifying_golden_patch=True,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def spec_for(data: dict) -> Any:
    return server()._make_test_spec(SWEBenchInstanceRequest.model_validate(data))


def log_file(tmp_path: Path, test_output: str) -> str:
    path = tmp_path / "test_output.txt"
    path.write_text(f"{START_TEST_OUTPUT}\n{test_output}\n{END_TEST_OUTPUT}\n")
    return str(path)


class TestPytestResultSummary:
    @pytest.mark.parametrize("data", [SPHINX_ROW, ASTROPY_ROW], ids=["sphinx", "astropy"])
    def test_python_scripts_print_per_test_results_just_before_the_tests_run(self, data: dict) -> None:
        script = spec_for(data).eval_script_list

        start = script.index(f": '{START_TEST_OUTPUT}'")
        assert script[start - 1] == PYTEST_RESULT_SUMMARY
        assert script.count(PYTEST_RESULT_SUMMARY) == 1
        # Only this line is added to what SWE-bench builds.
        assert [line for line in script if line != PYTEST_RESULT_SUMMARY] == make_test_spec(
            data, namespace="swebench"
        ).eval_script_list

    def test_the_line_comes_after_the_install_and_keeps_options_the_environment_set(self) -> None:
        script = spec_for(SPHINX_ROW).eval_script_list

        assert script.index(PYTEST_RESULT_SUMMARY) > script.index("python -m pip install -e .[test]")
        assert PYTEST_RESULT_SUMMARY == 'export PYTEST_ADDOPTS="-rA ${PYTEST_ADDOPTS:-}"'

    def test_the_sphinx_tox_run_prints_results_the_log_parser_reads(self, tmp_path: Path) -> None:
        # tox passes PYTEST_ADDOPTS to the pytest in its commands line; the parser then finds the passed test.
        test_spec = spec_for(SPHINX_ROW)

        status_map, found = get_logs_eval(test_spec, log_file(tmp_path, TOX_OUTPUT))

        assert found and status_map == {SPHINX_F2P: "PASSED"}
        assert any(line.startswith("tox ") for line in test_spec.eval_script_list[-3:])


class TestTestsBlockedByWarning:
    def blocked(self, tmp_path: Path, output: str) -> str | None:
        test_spec = spec_for(ASTROPY_ROW)
        status_map, _ = get_logs_eval(test_spec, log_file(tmp_path, output))
        return warning_blocking_required_tests(test_spec, output, status_map)

    def test_required_tests_that_error_on_a_warning_are_blocked(self, tmp_path: Path) -> None:
        reason = self.blocked(tmp_path, NOSE_WARNING_OUTPUT)

        assert reason is not None and "a warning was raised as an error" in reason
        assert "site-packages/_pytest/python.py: pytest.PytestRemovedIn8Warning: Support for nose tests" in reason

    def test_a_collection_error_on_a_warning_is_blocked(self, tmp_path: Path) -> None:
        reason = self.blocked(tmp_path, COLLECTION_WARNING_OUTPUT)

        assert (
            reason is not None and "setuptools/_distutils/version.py: DeprecationWarning: distutils Version" in reason
        )

    @pytest.mark.parametrize("output", [REPO_WARNING_OUTPUT, REPO_LONG_WARNING_OUTPUT], ids=["short", "long"])
    def test_a_warning_raised_from_a_file_of_the_repo_keeps_the_normal_grading(
        self, tmp_path: Path, output: str
    ) -> None:
        assert self.blocked(tmp_path, output) is None

    def test_one_warning_from_the_repo_among_warnings_from_outside_keeps_the_normal_grading(
        self, tmp_path: Path
    ) -> None:
        assert self.blocked(tmp_path, NOSE_WARNING_OUTPUT + REPO_WARNING_OUTPUT) is None

    def test_a_warning_whose_origin_the_report_does_not_show_keeps_the_normal_grading(self, tmp_path: Path) -> None:
        output = f"E   DeprecationWarning: no frame shown\nERROR {ASTROPY_P2P}\n"

        assert self.blocked(tmp_path, output) is None

    def test_a_failed_required_test_is_the_models_result(self, tmp_path: Path) -> None:
        output = NOSE_WARNING_OUTPUT + f"FAILED {ASTROPY_F2P} - assert 1 == 2\n"

        assert self.blocked(tmp_path, output) is None

    def test_an_error_without_a_warning_is_the_models_result(self, tmp_path: Path) -> None:
        output = (
            f"E   ImportError: cannot import name 'x'\nERROR {ASTROPY_HEADER} - ImportError: cannot import name 'x'\n"
        )

        assert self.blocked(tmp_path, output) is None

    def test_a_warning_shown_beside_passing_tests_is_ignored(self, tmp_path: Path) -> None:
        output = (
            f"E   DeprecationWarning: shown by a test that expects it\nPASSED {ASTROPY_F2P}\nPASSED {ASTROPY_P2P}\n"
        )

        assert self.blocked(tmp_path, output) is None


class _EvalSandbox:
    """A verification sandbox whose evaluation script prints the tests' output on stdout and bash's trace, test-output
    markers included, on stderr, which a sandbox returns separately."""

    def __init__(self, test_output: str) -> None:
        self.test_output = test_output

    async def exec(self, command: str, **kwargs: Any) -> SandboxExecResult:
        if command != "/bin/bash /eval.sh":
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        trace = f"+ : '{START_TEST_OUTPUT}'\n+ pytest -rA\n+ : '{END_TEST_OUTPUT}'\n"
        return SandboxExecResult(stdout=self.test_output, stderr=trace, return_code=0)

    async def upload(self, local_path: Path, remote_path: str) -> None:
        pass

    async def stop(self) -> None:
        pass


async def grade(monkeypatch: MonkeyPatch, tmp_path: Path, test_output: str) -> dict:
    monkeypatch.chdir(tmp_path)  # run_instance writes its logs under the working directory
    test_spec = spec_for(ASTROPY_ROW)
    container = DockerContainer(id="run-id", instance_id=test_spec.instance_id)
    container._inner_container = _EvalSandbox(test_output)
    return await run_instance(
        test_spec=test_spec,
        pred={"instance_id": test_spec.instance_id, "model_patch": "diff --git a/a.py b/a.py\n"},
        rm_image=False,
        force_rebuild=False,
        client=container,
        run_id="run-id",
    )


class TestRunBlockedByWarning:
    async def test_a_run_that_could_not_start_its_tests_has_no_verdict(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        result = await grade(monkeypatch, tmp_path, COLLECTION_WARNING_OUTPUT)

        assert result["completed"] is False and result["resolved"] is False
        assert "a warning was raised as an error" in result["error"]

    async def test_a_passing_run_is_resolved(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        result = await grade(monkeypatch, tmp_path, f"PASSED {ASTROPY_F2P}\nPASSED {ASTROPY_P2P}\n")

        assert result["completed"] is True and result["resolved"] is True and result["error"] is None

    async def test_a_failing_run_is_the_models_zero(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        result = await grade(monkeypatch, tmp_path, f"FAILED {ASTROPY_F2P} - assert 1 == 2\nPASSED {ASTROPY_P2P}\n")

        assert result["completed"] is True and result["resolved"] is False and result["error"] is None
