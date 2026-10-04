# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R2E-Gym grading: test-name matching, colored and verbose pytest output, and R2E-Gym's own reward rule."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from responses_api_agents.anyswe_agent.app import (
    AnySweAgent,
    AnySweInstanceConfig,
    _r2e_expected_resolved,
    _r2e_resolved,
)
from responses_api_agents.anyswe_agent.tests.test_app import _config


class TestR2EGrading:
    def test_r2e_resolved_matches_pytest_node_ids_to_unittest_names(self) -> None:
        log = (
            "PASSED r2e_tests/test_core.py::TestMaskedArray::test_basicattributes\n"
            "PASSED r2e_tests/test_core.py::TestMaskedArray::test_basic0d\n"
            "FAILED r2e_tests/test_core.py::TestOther::test_x - AssertionError\n"
        )
        instance = {
            "FAIL_TO_PASS": ["TestMaskedArray.test_basicattributes", "TestMaskedArray.test_basic0d"],
            "PASS_TO_PASS": [],
        }
        assert _r2e_resolved(instance, log) is True
        assert _r2e_resolved({"FAIL_TO_PASS": ["TestOther.test_x"]}, log) is False
        assert _r2e_resolved({"FAIL_TO_PASS": ["TestMaskedArray.test_missing"]}, log) is False
        assert _r2e_resolved({"FAIL_TO_PASS": []}, log) is False

    def test_r2e_resolved_ignores_ansi_escapes(self) -> None:
        # Some R2E rows list required tests in bold ("\x1b[1mtest_x\x1b[0m"), and R2E images run pytest with color.
        instance = {"FAIL_TO_PASS": ["\x1b[1mtest_sanity\x1b[0m"]}
        assert _r2e_resolved(instance, "PASSED tests/test_file.py::test_sanity\n") is True
        colored = (
            "\x1b[32mPASSED\x1b[0m r2e_tests/test_1.py::\x1b[1mtest_getdata\x1b[0m\n"
            "\x1b[31mFAILED\x1b[0m r2e_tests/test_1.py::\x1b[1mtest_broken\x1b[0m - ValueError\n"
        )
        assert _r2e_resolved({"FAIL_TO_PASS": ["test_getdata"]}, colored) is True
        assert _r2e_resolved({"FAIL_TO_PASS": ["test_getdata", "test_broken"]}, colored) is False

    def test_r2e_resolved_reads_verbose_lines_and_counts_xfail_as_passed(self) -> None:
        log = (
            "tests/test_schema.py::test_repeated PASSED [ 10%]\n"
            "tests/test_schema.py::test_attributes PASSED                  [ 20%]\n"
            "tests/test_schema.py::test_known_bug XFAIL [ 30%]\n"
        )
        instance = {
            "FAIL_TO_PASS": ["tests/test_schema.py::test_repeated"],
            "PASS_TO_PASS": ["tests/test_schema.py::test_attributes", "tests/test_schema.py::test_known_bug"],
        }
        assert _r2e_resolved(instance, log) is True
        assert _r2e_resolved(instance, log.replace("test_attributes PASSED", "test_attributes FAILED")) is False

    def test_r2e_expected_output_is_an_exact_status_match(self) -> None:
        log = (
            "=== short test summary info ===\n"
            "PASSED r2e_tests/test_1.py::TestA::test_fixed\n"
            "FAILED r2e_tests/test_1.py::TestA::test_known_broken - AssertionError: x\n"
        )
        expected = json.dumps({"TestA.test_fixed": "PASSED", "TestA.test_known_broken": "FAILED"})
        assert _r2e_expected_resolved(expected, log) is True
        # An extra test ran, a status differs, or no summary was printed.
        assert _r2e_expected_resolved(json.dumps({"TestA.test_fixed": "PASSED"}), log) is False
        assert (
            _r2e_expected_resolved(
                json.dumps({"TestA.test_fixed": "FAILED", "TestA.test_known_broken": "FAILED"}), log
            )
            is False
        )
        assert _r2e_expected_resolved(expected, "no summary here") is False
        bold = json.dumps({"\x1b[1mTestA.test_fixed\x1b[0m": "PASSED", "TestA.test_known_broken": "FAILED"})
        assert _r2e_expected_resolved(bold, log) is True

    @pytest.mark.parametrize(
        ("lists", "resolved"),
        [
            # No lists: R2E-Gym's expected_output_json rule, FAILED entries included.
            ({"FAIL_TO_PASS": [], "PASS_TO_PASS": []}, True),
            # Explicit lists still take precedence.
            ({"FAIL_TO_PASS": ["TestA.test_known_broken"]}, False),
        ],
    )
    def test_r2e_grading_uses_expected_output_when_lists_are_empty(
        self, tmp_path: Path, lists: dict, resolved: bool
    ) -> None:
        log = (
            "=== short test summary info ===\n"
            "PASSED r2e_tests/test_1.py::TestA::test_fixed\n"
            "FAILED r2e_tests/test_1.py::TestA::test_known_broken - AssertionError: x\n"
        )

        class _FakeSandbox:
            def __init__(self, provider, spec) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def upload(self, local: Path, remote: str) -> None:
                pass

            async def exec(self, command: str, **kwargs):
                return SimpleNamespace(return_code=0, stdout=log, stderr="", error_type=None)

        instance = {
            **lists,
            "expected_output_json": json.dumps({"TestA.test_fixed": "PASSED", "TestA.test_known_broken": "FAILED"}),
        }
        params = AnySweInstanceConfig(
            **_config().model_dump(),
            run_session_id="s",
            base_results_dir=tmp_path,
            model_server_url="http://policy:8000",
            resolved_sandbox_provider={"opensandbox": {}},
            sandbox_default_metadata={},
            problem_info={"instance_id": "pkg__repo-1", "instance_dict": json.dumps(instance)},
            body={"input": "fix it", "model": "policy"},
            persistent_dir=tmp_path,
            metrics_fpath=tmp_path / "metrics.json",
            container="registry.example.com/r2e:pkg__repo-1",
        )

        with patch("responses_api_agents.anyswe_agent.app.AsyncSandbox", _FakeSandbox):
            result = asyncio.run(AnySweAgent.__new__(AnySweAgent)._grade_r2e_patch(params, "diff\n"))

        assert result == (resolved, None)
        assert (tmp_path / "eval_output.txt").read_text() == log
