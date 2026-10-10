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
"""Tests for the SWE-Next resources server's grading and script construction.

Grading mirrors upstream TIGER-AI-Lab/SWE-Next's own ``DockerRuntime._calculate_reward_r2e``
(see verification.py's module docstring for why this is NOT a FAIL_TO_PASS/PASS_TO_PASS check):
parse the "short test summary info" pytest ``-rA`` prints, and require every test
``expected_output_json`` marked PASSED to still be PASSED. This file tests that parser and
grader, plus the stable contract app.py depends on (build_eval_script, verification_files,
run_verification, response shape, multi-worker entrypoint).

Two classes here pin bugs that were actually hit building the sibling SWE resources servers
(scale_swe, swe_rebench): a verify response missing the echoed request fields, and a
multi-worker entrypoint with no module-level `app`. Both looked fine locally and both only
failed once real traffic hit them, so the same tests are written here from the start.
"""

import json
import subprocess
from types import SimpleNamespace

import pytest

from resources_servers.swe_next.verification import (
    TEST_COMMAND,
    TEST_OUTPUT_BEGIN,
    TEST_OUTPUT_END,
    VerificationInputs,
    _slice,
    build_eval_script,
    drop_patch_sections,
    grade,
    parse_pytest_short_summary,
    patch_section_path,
    run_verification,
    verification_files,
)


def _inputs(**overrides) -> VerificationInputs:
    base = dict(
        instance_id="23andMe__Yamale-113e5b0d3993ed291d92eff92893fadcaa58f413",
        workdir="/testbed",
        patch="diff --git a/a.py b/a.py\n",
        test_patch="diff --git a/r2e_tests/test_a.py b/r2e_tests/test_a.py\n",
        expected_output_json=json.dumps({"test_semver": "PASSED"}),
    )
    base.update(overrides)
    return VerificationInputs(**base)


class TestPatchSectionPath:
    def test_extracts_the_b_path_from_a_simple_diff(self) -> None:
        section = "diff --git a/src/foo.py b/src/foo.py\n--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-x\n+y\n"
        assert patch_section_path(section) == "src/foo.py"


class TestDropPatchSections:
    def test_drops_sections_matching_given_paths(self) -> None:
        patch = (
            "diff --git a/keep.py b/keep.py\n--- a/keep.py\n+++ b/keep.py\n@@ -1 +1 @@\n-x\n+y\n"
            "diff --git a/drop.py b/drop.py\n--- /dev/null\n+++ b/drop.py\n@@ -0,0 +1 @@\n+z\n"
        )
        result = drop_patch_sections(patch, {"drop.py"})
        assert "keep.py" in result
        assert "drop.py" not in result


class TestBuildEvalScript:
    def test_applies_patch_then_test_patch(self) -> None:
        script = build_eval_script(_inputs())
        patch_idx = script.index("nemo_gym_patch.diff")
        test_patch_idx = script.index("nemo_gym_test_patch.diff")
        assert patch_idx < test_patch_idx

    def test_omits_test_patch_apply_when_empty(self) -> None:
        script = build_eval_script(_inputs(test_patch=""))
        assert "nemo_gym_test_patch.diff" not in script

    def test_uses_the_fixed_r2e_test_command(self) -> None:
        script = build_eval_script(_inputs())
        assert "r2e_tests" in script
        assert ".venv/bin/python" in script
        assert "-rA" in script

    def test_markers_bracket_the_test_command(self) -> None:
        script = build_eval_script(_inputs())
        assert TEST_OUTPUT_BEGIN in script
        assert TEST_OUTPUT_END in script
        assert script.index(TEST_OUTPUT_BEGIN) < script.index(TEST_COMMAND) < script.index(TEST_OUTPUT_END)

    def test_slice_extracts_between_markers(self) -> None:
        log = (
            f"noise before\n{TEST_OUTPUT_BEGIN}\nPASSED r2e_tests/test_a.py::test_one\n{TEST_OUTPUT_END}\nnoise after"
        )
        sliced = _slice(log, TEST_OUTPUT_BEGIN, TEST_OUTPUT_END)
        assert "noise before" not in sliced
        assert "noise after" not in sliced
        assert "PASSED r2e_tests/test_a.py::test_one" in sliced


class TestVerificationFiles:
    def test_includes_both_patches_when_present(self) -> None:
        files = verification_files(_inputs())
        assert "/tmp/nemo_gym_patch.diff" in files
        assert "/tmp/nemo_gym_test_patch.diff" in files
        assert "/tmp/nemo_gym_eval.sh" in files

    def test_omits_blank_patch(self) -> None:
        files = verification_files(_inputs(patch="   "))
        assert "/tmp/nemo_gym_patch.diff" not in files


class TestParsePytestShortSummary:
    def test_reads_the_short_summary_section(self) -> None:
        log = (
            "============ test session starts ============\n"
            "r2e_tests/test_1.py .F.                                      [100%]\n"
            "================== FAILURES ==================\n"
            "irrelevant traceback noise mentioning PASSED and FAILED\n"
            "============ short test summary info =========\n"
            "PASSED r2e_tests/test_1.py::test_one\n"
            "FAILED r2e_tests/test_1.py::test_two - AssertionError: boom\n"
        )
        statuses = parse_pytest_short_summary(log)
        assert statuses == {"test_one": "PASSED", "test_two": "FAILED"}

    def test_drops_only_the_file_path_segment_keeping_class_and_brackets(self) -> None:
        """Real R2E-Gym reward parsing drops ONLY the first "::" segment (the file path) -- a
        class::method id, or a parametrize bracket that itself contains "::" as data (an
        f-string format spec, seen in a real davidhalter/parso row), must survive intact."""
        log = (
            "short test summary info\n"
            "PASSED r2e_tests/test_x.py::TestFoo::test_bar\n"
            'PASSED r2e_tests/test_x.py::test_valid[f"{1::>4}"]\n'
        )
        statuses = parse_pytest_short_summary(log)
        assert statuses == {"TestFoo::test_bar": "PASSED", 'test_valid[f"{1::>4}"]': "PASSED"}

    def test_falls_back_to_scanning_the_whole_log_without_a_summary_section(self) -> None:
        log = "PASSED r2e_tests/test_1.py::test_one\nFAILED r2e_tests/test_1.py::test_two\n"
        assert parse_pytest_short_summary(log) == {"test_one": "PASSED", "test_two": "FAILED"}

    def test_strips_ansi_and_carriage_returns_before_parsing(self) -> None:
        """A real raw-dataset row (elastic/rally) has color escapes baked directly into the
        surrounding text; upstream's own log_output goes through the identical strip."""
        log = "short test summary info\n\x1b[32mPASSED\x1b[0m r2e_tests/test_x.py::test_one\r\n"
        assert parse_pytest_short_summary(log) == {"test_one": "PASSED"}


class TestGrade:
    def test_resolved_when_every_expected_pass_is_observed_passing(self) -> None:
        expected = {"a": "PASSED", "b": "PASSED"}
        parsed = {"a": "PASSED", "b": "PASSED"}
        assert grade(expected, parsed)["resolved"] is True

    def test_missing_expected_pass_test_fails_the_row(self) -> None:
        expected = {"a": "PASSED", "b": "PASSED"}
        parsed = {"a": "PASSED"}
        report = grade(expected, parsed)
        assert report["resolved"] is False
        assert "b" in report["missing"]

    def test_expected_pass_observed_failed_fails_the_row(self) -> None:
        expected = {"a": "PASSED"}
        parsed = {"a": "FAILED"}
        report = grade(expected, parsed)
        assert report["resolved"] is False
        assert "a" in report["mismatched"]

    def test_expected_failed_test_is_unconstrained(self) -> None:
        """The reference reward is lenient: a test expected FAILED/ERROR may come back anything
        (an agent is allowed to fix more than the golden patch did) without penalty."""
        expected = {"a": "PASSED", "b": "FAILED"}
        parsed = {"a": "PASSED", "b": "PASSED"}
        assert grade(expected, parsed)["resolved"] is True
        parsed_still_failing = {"a": "PASSED", "b": "FAILED"}
        assert grade(expected, parsed_still_failing)["resolved"] is True

    def test_expected_pass_test_absent_from_parsed_counts_as_missing(self) -> None:
        expected = {"a": "PASSED"}
        assert grade(expected, {})["resolved"] is False

    def test_ansi_color_codes_baked_into_expected_output_json_keys_are_stripped(self) -> None:
        """A real raw-dataset row (elastic/rally) bakes terminal color codes into the KEYS of its
        own expected_output_json, not just FAIL_TO_PASS/PASS_TO_PASS -- an upstream data artifact
        that must be stripped on both sides or a clean fresh-run key can never match it."""
        expected = {"\x1b[1mTestFoo::test_bar\x1b[0m": "PASSED"}
        parsed = {"TestFoo::test_bar": "PASSED"}
        assert grade(expected, parsed)["resolved"] is True


class _FakeSandbox:
    def __init__(self, stdout: str, return_code: int = 0) -> None:
        self._result = SimpleNamespace(stdout=stdout, stderr="", return_code=return_code)
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_s=None):
        self.commands.append(command)
        return self._result


class TestRunVerification:
    @pytest.mark.asyncio
    async def test_grades_a_passing_run_via_the_stdout_summary(self) -> None:
        log = f"{TEST_OUTPUT_BEGIN}\nshort test summary info\nPASSED r2e_tests/test_x.py::test_semver\n{TEST_OUTPUT_END}\n"
        sandbox = _FakeSandbox(log)
        result = await run_verification(sandbox=sandbox, inputs=_inputs())
        assert result.completed and result.resolved
        assert sandbox.commands == ["bash /tmp/nemo_gym_eval.sh"]

    @pytest.mark.asyncio
    async def test_missing_workdir_is_incomplete_not_a_zero(self) -> None:
        result = await run_verification(sandbox=_FakeSandbox("", return_code=97), inputs=_inputs())
        assert result.completed is False
        assert result.resolved is False
        assert "not present in the image" in result.error

    @pytest.mark.asyncio
    async def test_empty_expected_output_json_is_incomplete_not_resolved(self) -> None:
        """An empty expected map would trivially satisfy the reference's own "every expected-PASS
        test passes" loop (vacuously true) -- that's a data-pipeline bug on our side (this dataset
        row should never lack the field), not a real resolved instance, so it must not silently
        report resolved=True."""
        log = f"{TEST_OUTPUT_BEGIN}\n{TEST_OUTPUT_END}\n"
        result = await run_verification(sandbox=_FakeSandbox(log), inputs=_inputs(expected_output_json="{}"))
        assert result.completed is False
        assert result.resolved is False
        assert "empty expected_output_json" in result.error


class TestVerifyResponseShape:
    """BaseVerifyResponse extends BaseVerifyRequest, so the request fields are REQUIRED on the
    way out. Building the response from scratch drops them and the resulting ValidationError
    surfaces to the caller as a bare JSON string -- which is how a fully successful verification
    can still fail the whole sweep. (Hit for real in resources_servers/swe_rebench/app.py.)"""

    @staticmethod
    def _body() -> dict:
        return {
            "instance_id": "23andMe__Yamale-113e5b0d3993ed291d92eff92893fadcaa58f413",
            "workdir": "/testbed",
            "image_ref": "lllqaq/23andme_yamale-final:113e5b0d3993ed291d92eff92893fadcaa58f413",
            "language": "python",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
            "expected_output_json": "{}",
            "responses_create_params": {"input": []},
            "response": {
                "output": [],
                "id": "",
                "created_at": 0,
                "model": "",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        }

    def test_spreading_the_request_body_satisfies_the_response_model(self) -> None:
        from resources_servers.swe_next.app import SWENextVerifyResponse

        response = SWENextVerifyResponse.model_validate(
            self._body()
            | {
                "reward": 1.0,
                "evaluation_completed": True,
                "resolved": True,
                "patch_applied": True,
                "test_results": {"resolved": True},
                "test_output": "",
                "error": None,
                "eval_sandbox_start_time_taken": 0.1,
                "patch_verification_time_taken": 0.2,
            }
        )
        assert response.instance_id == "23andMe__Yamale-113e5b0d3993ed291d92eff92893fadcaa58f413"

    def test_building_from_scratch_without_the_request_fields_fails(self) -> None:
        from resources_servers.swe_next.app import SWENextVerifyResponse

        with pytest.raises(Exception):
            SWENextVerifyResponse.model_validate(
                {
                    "reward": 1.0,
                    "evaluation_completed": True,
                    "resolved": True,
                    "patch_applied": True,
                    "instance_id": "x",
                    "language": "python",
                    "test_results": None,
                    "test_output": "",
                    "error": None,
                    "eval_sandbox_start_time_taken": 0.1,
                    "patch_verification_time_taken": 0.2,
                }
            )


class TestAntiCheating:
    """seed_session must scrub the sandbox's git history before an agent gets control of it, or
    it can just `git log --all` / `git show <future-commit>` and read the golden fix instead of
    solving the task. See resources_servers/swebench/anti_cheat.py."""

    @staticmethod
    def _source() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / "app.py").read_text()

    def test_seed_session_calls_the_shared_anti_cheat_helper(self) -> None:
        source = self._source()
        assert "from resources_servers.swebench.anti_cheat import apply_anti_cheat_setup" in source
        assert "apply_anti_cheat_setup(" in source

    def test_config_enables_it_by_default(self) -> None:
        from resources_servers.swe_next.app import SWENextResourcesServerConfig

        assert SWENextResourcesServerConfig.model_fields["apply_anti_cheating"].default is True

    def test_seed_session_inits_a_git_repo_before_scrubbing(self) -> None:
        source = self._source()
        assert "_init_git_repo(" in source
        init_idx = source.index("await self._init_git_repo(")
        anti_cheat_idx = source.index("apply_anti_cheat_setup(sandbox")
        assert init_idx < anti_cheat_idx

    def test_agent_facing_config_turns_it_on(self) -> None:
        from pathlib import Path

        import yaml

        config = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "swe_next.yaml").read_text())
        assert config["swe_next_resources_server"]["resources_servers"]["swe_next"]["apply_anti_cheating"] is True


class _ShellSandbox:
    async def exec(self, command: str, timeout_s=None):
        proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
        return SimpleNamespace(stdout=proc.stdout, stderr=proc.stderr, return_code=proc.returncode)


class TestHidesGradingArtifacts:
    @pytest.mark.asyncio
    async def test_deletes_every_grading_artifact_through_the_testbed_symlink(self, tmp_path) -> None:
        from resources_servers.swe_next.app import hide_grading_artifacts

        workspace = tmp_path / "workspace"
        (workspace / "r2e_tests").mkdir(parents=True)
        (workspace / "r2e_tests" / "test_1.py").write_text("def test_one():\n    assert True\n")
        for name in (
            "parsed_commit.json",
            "modified_files.json",
            "modified_entities.json",
            "syn_issue.json",
            "expected_test_output.json",
            "execution_result.json",
        ):
            (workspace / name).write_text("{}")
        (workspace / "run_tests.sh").write_text(".venv/bin/python -W ignore -m pytest -rA r2e_tests\n")
        (workspace / "install.sh").write_text("pip install -e .\n")
        (workspace / "setup.py").write_text("")
        testbed = tmp_path / "testbed"
        testbed.symlink_to(workspace)

        await hide_grading_artifacts(_ShellSandbox(), str(testbed))

        assert sorted(path.name for path in workspace.iterdir()) == ["install.sh", "setup.py"]

    @pytest.mark.asyncio
    async def test_keeps_a_repo_owned_run_tests_sh(self, tmp_path) -> None:
        from resources_servers.swe_next.app import hide_grading_artifacts

        (tmp_path / "run_tests.sh").write_text("pytest tests/\n")

        await hide_grading_artifacts(_ShellSandbox(), str(tmp_path))

        assert (tmp_path / "run_tests.sh").read_text() == "pytest tests/\n"

    def test_seed_session_hides_them_before_the_git_snapshot(self) -> None:
        source = TestAntiCheating._source()
        assert source.index("await hide_grading_artifacts(") < source.index("await self._init_git_repo(")


class TestMultiWorkerEntrypoint:
    """num_workers > 1 makes uvicorn re-import this entrypoint by path in each forked child.
    Without a module-level `app`, every child exits and uvicorn stops the parent, so the server
    never binds -- the symptom is a flood of connection errors from clients talking to a dead
    port, not an ImportError. (Hit for real in resources_servers/swe_rebench/app.py.)"""

    @staticmethod
    def _source() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / "app.py").read_text()

    def test_exposes_a_module_level_app_for_forked_workers(self) -> None:
        source = self._source()
        assert "is_nemo_gym_fastapi_entrypoint(__file__)" in source
        assert "app = SWENextResourcesServer.run_webserver()" in source

    def test_the_config_that_needs_it_still_sets_num_workers(self) -> None:
        """Pins the pair: if num_workers is configured, the entrypoint branch must exist."""
        from pathlib import Path

        import yaml

        config = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "swe_next.yaml").read_text())
        for name, block in config.items():
            workers = block["resources_servers"]["swe_next"].get("num_workers")
            if workers and workers > 1:
                assert "is_nemo_gym_fastapi_entrypoint(__file__)" in self._source(), (
                    f"{name} sets num_workers={workers} but the entrypoint has no module-level app"
                )
