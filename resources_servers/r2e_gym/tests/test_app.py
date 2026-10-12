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
"""Tests for the R2E-Gym resources server's grading, script construction and data shaping.

These cover the parts that decide whether a task counts as resolved and what reaches the agent. The
sandbox lifecycle itself is exercised by the golden-patch run, which needs a live OpenSandbox endpoint.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from resources_servers.r2e_gym.apply_golden_patch import EMPTY_RESPONSE, build_payload
from resources_servers.r2e_gym.r2e_patch import (
    file_diff_patch,
    golden_patch,
    image_name,
    instance_id_for,
    is_test_file,
    issue_text,
)
from resources_servers.r2e_gym.verification import (
    AGENT_HIDDEN_PATHS,
    EXIT_NO_REPO,
    EXIT_NO_TESTS,
    PATCH_FAILED,
    TEST_OUTPUT_BEGIN,
    TEST_OUTPUT_END,
    VerificationInputs,
    build_eval_script,
    drop_hidden_test_sections,
    drop_patch_sections,
    grade,
    hide_from_agent_command,
    is_hidden_test_path,
    parse_expected,
    parse_log_pytest,
    run_verification,
    slice_test_output,
    verification_files,
)


SERVER_DIR = Path(__file__).resolve().parent.parent

PARSED_COMMIT = {
    "file_diffs": [
        {
            "header": {"file": {"path": "Orange/widgets/settings.py"}},
            "index_line": {"old_commit_hash": "8be8bf0ae", "new_commit_hash": "75ebe4129", "mode": "100644"},
            "minus_file": {"path": "a/Orange/widgets/settings.py"},
            "plus_file": {"path": "b/Orange/widgets/settings.py"},
            "hunks": [
                {
                    "descriptor": {
                        "old_range": {"start": 49, "length": 7},
                        "new_range": {"start": 49, "length": 8},
                        "section": "log = logging.getLogger(__name__)",
                    },
                    "line_group": {
                        "all_lines": [
                            {"type": "context", "content": "__all__ = ["},
                            {
                                "type": "deleted",
                                "content": '           "ClassValuesContextHandler", "widget_settings_dir"]',
                            },
                            {"type": "added", "content": '           "ClassValuesContextHandler",'},
                            {"type": "added", "content": '           "widget_settings_dir"]'},
                        ]
                    },
                }
            ],
        },
        {
            "header": {"file": {"path": "Orange/widgets/tests/test_context_handler.py"}},
            "minus_file": {"path": "a/Orange/widgets/tests/test_context_handler.py"},
            "plus_file": {"path": "b/Orange/widgets/tests/test_context_handler.py"},
            "hunks": [
                {
                    "descriptor": {
                        "old_range": {"start": 1, "length": None},
                        "new_range": {"start": 1, "length": 2},
                        "section": "",
                    },
                    "line_group": {
                        "all_lines": [
                            {"type": "added", "content": "import x"},
                            {"type": "note", "content": "No newline at end of file"},
                        ]
                    },
                }
            ],
        },
        {"header": {"file": {"path": "doc/development/source/tutorial-settings.rst"}}, "hunks": []},
    ]
}


def _inputs(**overrides) -> VerificationInputs:
    base = dict(
        instance_id="orange3__2d9617bd0cb1f0ba61771258410ab8fae8e7e24d",
        repo_name="orange3",
        patch="diff --git a/Orange/widgets/settings.py b/Orange/widgets/settings.py\n",
        expected={"TestContextHandler.test_close_context": "PASSED", "TestContextHandler.test_fast_save": "PASSED"},
    )
    base.update(overrides)
    return VerificationInputs(**base)


class TestGoldenPatch:
    def test_rebuilds_the_unified_diff_like_r2e_gym(self) -> None:
        patch = file_diff_patch(PARSED_COMMIT["file_diffs"][0])
        assert patch == (
            "diff --git a/Orange/widgets/settings.py b/Orange/widgets/settings.py\n"
            "index 8be8bf0ae..75ebe4129 100644\n"
            "--- a/Orange/widgets/settings.py\n"
            "+++ b/Orange/widgets/settings.py\n"
            "@@ -49,7 +49,8 @@ log = logging.getLogger(__name__)\n"
            " __all__ = [\n"
            '-           "ClassValuesContextHandler", "widget_settings_dir"]\n'
            '+           "ClassValuesContextHandler",\n'
            '+           "widget_settings_dir"]\n'
        )

    def test_ranges_without_a_length_and_note_lines(self) -> None:
        patch = file_diff_patch(PARSED_COMMIT["file_diffs"][1])
        assert "@@ -1 +1,2 @@\n+import x\n\\ No newline at end of file\n" in patch

    def test_default_keeps_python_files_only_test_and_non_test(self) -> None:
        patch = golden_patch(PARSED_COMMIT)
        assert "Orange/widgets/settings.py" in patch
        assert "Orange/widgets/tests/test_context_handler.py" in patch
        assert "tutorial-settings.rst" not in patch

    def test_can_drop_test_files(self) -> None:
        patch = golden_patch(PARSED_COMMIT, test_file=False)
        assert "test_context_handler.py" not in patch and "settings.py" in patch

    def test_empty_commit_yields_an_empty_patch(self) -> None:
        assert golden_patch({"file_diffs": []}) == ""

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("Orange/widgets/tests/test_context_handler.py", True),
            ("pkg/test/x.py", True),
            ("foo_test.py", True),
            ("pkg/test_foo.py", True),
            ("Orange/widgets/settings.py", False),
            ("pkg/testing_utils.py", False),
        ],
    )
    def test_is_test_file_matches_r2e_gyms_rule(self, path, expected) -> None:
        assert is_test_file(path) is expected


class TestRowShaping:
    def test_issue_text_is_the_tagged_region(self) -> None:
        assert issue_text("preamble [ISSUE]\n**Title:** x\n\n**Description:** y\n[/ISSUE] trailer") == (
            "**Title:** x\n\n**Description:** y"
        )

    def test_issue_text_falls_back_to_the_whole_statement(self) -> None:
        assert issue_text("  no tags here ") == "no tags here"

    def test_instance_id_and_image_name(self) -> None:
        assert instance_id_for("orange3", "2d96") == "orange3__2d96"
        assert image_name("namanjain12/orange3_final:2d96") == "docker.io/namanjain12/orange3_final:2d96"
        assert image_name("docker.io/namanjain12/orange3_final:2d96") == "docker.io/namanjain12/orange3_final:2d96"

    def test_build_row_drops_the_heavy_and_leaky_columns(self) -> None:
        from resources_servers.r2e_gym.prepare_r2e_gym import build_row

        row = build_row(
            {
                "repo_name": "orange3",
                "docker_image": "namanjain12/orange3_final:2d96",
                "commit_hash": "2d96",
                "problem_statement": "[ISSUE]\nTitle\n[/ISSUE]",
                "expected_output_json": '{"T.t": "PASSED"}',
                "modified_files": ["Orange/widgets/settings.py"],
                "num_non_test_files": 1,
                "num_non_test_func_methods": 1,
                "num_non_test_lines": 25,
                "parsed_commit_content": json.dumps(PARSED_COMMIT),
                "execution_result_content": "huge",
                "modified_entity_summaries": [],
                "relevant_files": [],
                "prompt": "issue-writing instruction",
            }
        )
        assert row["instance_id"] == "orange3__2d96" and row["image_name"].startswith("docker.io/")
        assert row["patch"] == golden_patch(PARSED_COMMIT)
        assert row["responses_create_params"]["input"] == [{"role": "user", "content": "Title"}]
        for dropped in ("parsed_commit_content", "execution_result_content", "modified_entity_summaries", "prompt"):
            assert dropped not in row
        assert row["agent_ref"]["name"] == "r2e_gym_opencode_sandboxed_agent"


class TestParseLogPytest:
    def test_reads_the_summary_block_with_r2e_gyms_naming(self) -> None:
        log = (
            "r2e_tests/test_1.py ..F.\n"
            "=========================== short test summary info ============================\n"
            "PASSED r2e_tests/test_1.py::TestContextHandler::test_close_context\n"
            "FAILED r2e_tests/test_1.py::TestContextHandler::test_fast_save - AssertionError: boom\n"
            "ERROR r2e_tests/test_1.py::test_module_level - ImportError\n"
            "========================= 1 failed, 1 passed in 0.3s ==========================\n"
        )
        assert parse_log_pytest(log) == {
            "TestContextHandler.test_close_context": "PASSED",
            "TestContextHandler.test_fast_save": "FAILED",
            "test_module_level": "ERROR",
        }

    def test_no_summary_block_means_nothing_observed(self) -> None:
        assert parse_log_pytest("collected 0 items\n") == {}

    def test_strips_ansi_colour_codes(self) -> None:
        log = "short test summary info\n\x1b[32mPASSED\x1b[0m r2e_tests/test_1.py::test_a\n"
        assert parse_log_pytest(log) == {"test_a": "PASSED"}


class TestGrade:
    def test_exact_match_resolves(self) -> None:
        expected = {"T.a": "PASSED", "T.b": "FAILED"}
        assert grade({"T.a": "PASSED", "T.b": "FAILED"}, expected)["resolved"] is True

    def test_a_status_change_does_not_resolve(self) -> None:
        report = grade({"T.a": "PASSED", "T.b": "PASSED"}, {"T.a": "PASSED", "T.b": "FAILED"})
        assert report["resolved"] is False
        assert report["mismatched"] == {"T.b": {"expected": "FAILED", "observed": "PASSED"}}

    def test_a_missing_test_does_not_resolve(self) -> None:
        report = grade({"T.a": "PASSED"}, {"T.a": "PASSED", "T.b": "PASSED"})
        assert report["resolved"] is False and report["missing"] == ["T.b"]

    def test_an_extra_test_does_not_resolve(self) -> None:
        report = grade({"T.a": "PASSED", "T.c": "PASSED"}, {"T.a": "PASSED"})
        assert report["resolved"] is False and report["unexpected"] == ["T.c"]

    def test_reason_suffixes_are_ignored_on_both_sides(self) -> None:
        assert grade({"T.a - oops": "FAILED"}, {"T.a": "FAILED"})["resolved"] is True

    def test_nothing_expected_never_resolves(self) -> None:
        assert grade({}, {})["resolved"] is False

    def test_parse_expected_accepts_the_hub_string_or_a_dict(self) -> None:
        assert parse_expected('{"T.a": "PASSED"}') == {"T.a": "PASSED"}
        assert parse_expected({"T.a": "PASSED"}) == {"T.a": "PASSED"}
        assert parse_expected("") == {} and parse_expected(None) == {}


class TestBuildEvalScript:
    def test_refuses_an_image_without_the_hidden_tests(self) -> None:
        script = build_eval_script(_inputs())
        assert f"cd /testbed || exit {EXIT_NO_REPO}" in script
        assert f"[ -d /r2e_tests ] && [ -f /testbed/run_tests.sh ] || exit {EXIT_NO_TESTS}" in script

    def test_applies_the_patch_like_r2e_gym_excluding_untracked_files(self) -> None:
        script = build_eval_script(_inputs())
        assert "git ls-files --others --exclude-standard" in script
        assert "git apply --whitespace=fix $EXCLUDES /tmp/nemo_gym_patch.diff" in script
        assert f"echo {PATCH_FAILED}" in script

    def test_stages_the_runner_and_tests_then_runs_them_between_markers(self) -> None:
        script = build_eval_script(_inputs())
        order = [
            script.index("git apply --whitespace=fix"),
            script.index("mv /testbed/run_tests.sh /root/run_tests.sh"),
            script.index("mv /r2e_tests /root/r2e_tests"),
            script.index("ln -s /root/r2e_tests /testbed/r2e_tests"),
            script.index(TEST_OUTPUT_BEGIN),
            script.index("bash /root/run_tests.sh"),
            script.index(TEST_OUTPUT_END),
        ]
        assert order == sorted(order)

    def test_omits_the_patch_step_when_there_is_no_patch(self) -> None:
        assert "nemo_gym_patch.diff" not in build_eval_script(_inputs(patch=""))

    def test_verification_files_ship_the_script_and_the_patch_only_when_present(self) -> None:
        assert set(verification_files(_inputs())) == {"/tmp/nemo_gym_eval.sh", "/tmp/nemo_gym_patch.diff"}
        assert set(verification_files(_inputs(patch=""))) == {"/tmp/nemo_gym_eval.sh"}


class TestHiddenTestProtection:
    """The R2E analogue of SWE-bench's test-file reset: nothing a model writes may reach the hidden
    tests' path or the runner, whether through its patch or through the repo checkout."""

    MODEL_PATCH = (
        "diff --git a/Orange/x.py b/Orange/x.py\n--- a/Orange/x.py\n+++ b/Orange/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/r2e_tests/test_1.py b/r2e_tests/test_1.py\nnew file mode 100644\n--- /dev/null\n"
        "+++ b/r2e_tests/test_1.py\n@@ -0,0 +1 @@\n+def test_x(): pass\n"
        "diff --git a/run_tests.sh b/run_tests.sh\n--- a/run_tests.sh\n+++ b/run_tests.sh\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/expected_test_output.json b/expected_test_output.json\nnew file mode 100644\n--- /dev/null\n"
        "+++ b/expected_test_output.json\n@@ -0,0 +1 @@\n+{}\n"
    )

    def test_sections_under_r2e_tests_and_the_runner_are_dropped(self) -> None:
        kept = drop_hidden_test_sections(self.MODEL_PATCH)
        assert [line for line in kept.splitlines() if line.startswith("diff --git")] == [
            "diff --git a/Orange/x.py b/Orange/x.py"
        ]

    def test_ordinary_patches_pass_through_untouched(self) -> None:
        patch = "diff --git a/Orange/x.py b/Orange/x.py\n--- a/Orange/x.py\n+++ b/Orange/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        assert drop_hidden_test_sections(patch) == patch and drop_hidden_test_sections("") == ""

    @pytest.mark.parametrize(
        "path, hidden",
        [("r2e_tests/test_1.py", True), ("r2e_tests", True), ("run_tests.sh", True), ("parsed_commit.json", True),
         ("Orange/tests/test_x.py", False), ("my_r2e_tests/x.py", False), (None, False)],
    )  # fmt: skip
    def test_is_hidden_test_path(self, path, hidden) -> None:
        assert is_hidden_test_path(path) is hidden

    def test_drop_patch_sections_removes_named_files_only(self) -> None:
        kept = drop_patch_sections(self.MODEL_PATCH, ["run_tests.sh"])
        assert "run_tests.sh" not in kept and "Orange/x.py" in kept and "r2e_tests/test_1.py" in kept

    def test_eval_script_clears_a_repo_side_r2e_tests_before_linking_the_real_one(self) -> None:
        script = build_eval_script(_inputs())
        assert (
            script.index("mv /r2e_tests /root/r2e_tests")
            < script.index("rm -rf /testbed/r2e_tests")
            < script.index("ln -s /root/r2e_tests /testbed/r2e_tests")
        )

    def test_the_server_applies_the_drop_to_every_candidate_patch(self) -> None:
        source = (SERVER_DIR / "app.py").read_text()
        inputs = source[source.index("def _inputs") : source.index("async def _create_sandbox")]
        assert "patch=drop_hidden_test_sections(patch)" in inputs
        assert "drop_sections=drop_patch_sections" in source, (
            "pristine untracked files must be stripped from captured patches"
        )


class _FakeSandbox:
    def __init__(self, stdout: str, return_code: int = 0, stderr: str = "") -> None:
        self.result = SimpleNamespace(stdout=stdout, stderr=stderr, return_code=return_code)

    async def exec(self, command: str, timeout_s=None, cwd=None):
        return self.result


class TestRunVerification:
    @pytest.mark.asyncio
    async def test_grades_a_matching_run(self) -> None:
        log = (
            f"{TEST_OUTPUT_BEGIN}\nshort test summary info\n"
            "PASSED r2e_tests/test_1.py::TestContextHandler::test_close_context\n"
            "PASSED r2e_tests/test_1.py::TestContextHandler::test_fast_save\n"
            f"{TEST_OUTPUT_END}\n"
        )
        result = await run_verification(_FakeSandbox(log), _inputs())
        assert result.completed and result.resolved and result.patch_applied

    @pytest.mark.asyncio
    async def test_a_patch_that_does_not_apply_is_a_completed_zero(self) -> None:
        result = await run_verification(_FakeSandbox(f"error: patch failed\n{PATCH_FAILED}\n"), _inputs())
        assert result.completed is True and result.resolved is False and result.patch_applied is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", [EXIT_NO_REPO, EXIT_NO_TESTS])
    async def test_a_wrong_image_is_incomplete_not_a_zero(self, code) -> None:
        result = await run_verification(_FakeSandbox("", return_code=code), _inputs())
        assert result.completed is False and result.test_results is None

    @pytest.mark.asyncio
    async def test_only_the_marked_region_reaches_the_parser(self) -> None:
        log = (
            "short test summary info\nPASSED r2e_tests/test_1.py::TestContextHandler::test_fast_save\n"  # before marker
            f"{TEST_OUTPUT_BEGIN}\nshort test summary info\n"
            "PASSED r2e_tests/test_1.py::TestContextHandler::test_close_context\n"
            f"{TEST_OUTPUT_END}\n"
        )
        result = await run_verification(_FakeSandbox(log), _inputs())
        assert result.resolved is False and result.test_results["missing"] == ["TestContextHandler.test_fast_save"]

    def test_slice_falls_back_to_the_whole_log(self) -> None:
        assert slice_test_output("plain") == "plain"


class TestVerifyResponseShape:
    @staticmethod
    def _body() -> dict:
        return {
            "instance_id": "orange3__2d96",
            "repo_name": "orange3",
            "image_name": "docker.io/namanjain12/orange3_final:2d96",
            "expected_output_json": '{"T.a": "PASSED"}',
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
        from resources_servers.r2e_gym.app import R2EGymVerifyResponse

        response = R2EGymVerifyResponse.model_validate(
            self._body()
            | {
                "reward": 1.0,
                "evaluation_completed": True,
                "resolved": True,
                "patch_applied": True,
                "language": "python",
                "test_results": {"resolved": True},
                "test_output": "",
                "error": None,
                "eval_sandbox_start_time_taken": 1.0,
                "patch_verification_time_taken": 2.0,
            }
        )
        assert response.reward == 1.0 and response.resolved
        assert response.repo_name == "orange3", "row identity must survive into sweep outputs"

    def test_omitting_the_request_fields_is_rejected(self) -> None:
        import pydantic

        from resources_servers.r2e_gym.app import R2EGymVerifyResponse

        with pytest.raises(pydantic.ValidationError, match="responses_create_params|response"):
            R2EGymVerifyResponse.model_validate(
                {
                    "reward": 1.0,
                    "evaluation_completed": True,
                    "resolved": True,
                    "patch_applied": True,
                    "instance_id": "i",
                    "language": "python",
                    "test_results": None,
                    "test_output": "",
                    "error": None,
                    "eval_sandbox_start_time_taken": 0.0,
                    "patch_verification_time_taken": 0.0,
                }
            )


class TestAgentSandbox:
    """What the agent's sandbox is and is not given."""

    @staticmethod
    def _source() -> str:
        return (SERVER_DIR / "app.py").read_text()

    def test_hidden_tests_runner_and_sidecars_are_removed(self) -> None:
        command = hide_from_agent_command()
        assert command.startswith("rm -rf ")
        for path in (
            "/r2e_tests",
            "/testbed/run_tests.sh",
            "/testbed/expected_test_output.json",
            "/testbed/parsed_commit.json",
        ):
            assert path in AGENT_HIDDEN_PATHS and path in command

    def test_seed_session_hides_tests_before_scrubbing_history_and_handing_over(self) -> None:
        source = self._source()
        seed = source[source.index("async def seed_session") : source.index("def _response")]
        assert (
            seed.index("_hide_tests_from_agent(")
            < seed.index("apply_anti_cheat_setup(")
            < seed.index("self._session_id_to_sandbox[session_id] = sandbox")
        )
        assert "_create_sandbox(body)" in seed and "files=" not in seed, "the agent sandbox gets no row files"

    def test_a_failed_scrub_refuses_the_rollout(self) -> None:
        source = self._source()
        hide = source[
            source.index("async def _hide_tests_from_agent") : source.index("async def _pristine_untracked_files")
        ]
        assert "raise RuntimeError" in hide

    def test_seed_session_calls_the_shared_anti_cheat_helper(self) -> None:
        source = self._source()
        assert "from resources_servers.swebench.anti_cheat import apply_anti_cheat_setup" in source

    def test_config_enables_anti_cheating_by_default_and_in_the_agent_config(self) -> None:
        from resources_servers.r2e_gym.app import R2EGymResourcesServerConfig

        assert R2EGymResourcesServerConfig.model_fields["apply_anti_cheating"].default is True
        config = yaml.safe_load((SERVER_DIR / "configs" / "r2e_gym.yaml").read_text())
        assert config["r2e_gym_resources_server"]["resources_servers"]["r2e_gym"]["apply_anti_cheating"] is True

    def test_the_app_wires_cpu_cap_env_into_the_spec(self) -> None:
        source = self._source()
        assert "cpu_cap_env(sandbox_resources.cpu)" in source and "env=env," in source


class TestMultiWorkerEntrypoint:
    def test_exposes_a_module_level_app_for_forked_workers(self) -> None:
        source = (SERVER_DIR / "app.py").read_text()
        assert "is_nemo_gym_fastapi_entrypoint(__file__)" in source
        assert "app = R2EGymResourcesServer.run_webserver()" in source

    def test_the_config_that_needs_it_sets_num_workers(self) -> None:
        config = yaml.safe_load((SERVER_DIR / "configs" / "r2e_gym.yaml").read_text())
        golden = config["r2e_gym_golden_patch_resources_server"]["resources_servers"]["r2e_gym"]
        assert golden["num_workers"] > 1 and golden["is_verifying_golden_patch"] is True


class TestGoldenPatchAggregation:
    @staticmethod
    def _obs(completed: bool, resolved: bool) -> dict:
        return {"evaluation_completed": completed, "resolved": resolved}

    def test_buckets(self) -> None:
        from resources_servers.r2e_gym.aggregate_golden_patch import classify

        assert classify([self._obs(True, True)] * 3, 3) == "supported"
        assert classify([self._obs(True, True), self._obs(True, False), self._obs(True, True)], 3) == "flaky"
        assert classify([self._obs(True, False)] * 3, 3) == "broken"
        assert classify([self._obs(True, True), self._obs(True, True), self._obs(False, False)], 3) == "inconclusive"


def test_build_payload_empty_patch_control_blanks_only_the_patch():
    example = {"instance_id": "x__1", "patch": "diff --git a/f.py b/f.py\n", "repo_name": "x"}
    golden = build_payload(example)
    assert golden["patch"] == example["patch"]
    assert golden["response"] == EMPTY_RESPONSE
    assert golden["responses_create_params"] == {"input": []}
    control = build_payload(example, empty_patch=True)
    assert control["patch"] == ""
    assert control["instance_id"] == "x__1"
    assert example["patch"] != ""  # the input row is left untouched
