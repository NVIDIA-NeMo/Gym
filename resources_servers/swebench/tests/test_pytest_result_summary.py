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
"""Python evaluation scripts print per-test results."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
from swebench.harness.grading import get_logs_eval
from swebench.harness.run_evaluation import make_test_spec

from nemo_gym.client import ServerClient
from resources_servers.swebench.app import (
    SWEBenchInstanceRequest,
    SwebenchResourcesServer,
    SwebenchResourcesServerConfig,
)
from resources_servers.swebench.swebench_patches import PYTEST_RESULT_SUMMARY


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
