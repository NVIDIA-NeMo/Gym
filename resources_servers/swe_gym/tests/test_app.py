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
"""Tests for the SWE-Gym resources server's grading, script construction and data shaping.

These cover the parts that decide whether a task counts as resolved and what reaches the agent. The
sandbox lifecycle itself is exercised by the golden-patch run, which needs a live OpenSandbox endpoint.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from resources_servers.swe_gym.apply_golden_patch import EMPTY_RESPONSE, build_payload
from resources_servers.swe_gym.swebench_specs import (
    SPECS,
    normalize_test_id,
    parse_log_pytest,
    pytest_command,
    spec_for,
    touched_test_files,
)
from resources_servers.swe_gym.verification import (
    EXIT_NO_CONDA,
    EXIT_NO_REPO,
    PATCH_FAILED,
    TEST_OUTPUT_BEGIN,
    TEST_OUTPUT_END,
    TEST_PATCH_FAILED,
    VerificationInputs,
    as_list,
    build_eval_script,
    drop_test_patch_files,
    grade,
    run_verification,
    slice_test_output,
    verification_files,
)


SERVER_DIR = Path(__file__).resolve().parent.parent


def _inputs(**overrides) -> VerificationInputs:
    base = dict(
        instance_id="getmoto__moto-7365",
        repo="getmoto/moto",
        version="5.0",
        base_commit="7f6c9cb1deafb280fe7fcc7551c38e397f11a706",  # pragma: allowlist secret
        patch="diff --git a/moto/dynamodb/models/dynamo_type.py b/moto/dynamodb/models/dynamo_type.py\n",
        test_patch="diff --git a/tests/test_dynamodb/test_update.py b/tests/test_dynamodb/test_update.py\n",
        fail_to_pass=["tests/test_dynamodb/test_update.py::test_update_item_add_float"],
        pass_to_pass=["tests/test_dynamodb/test_update.py::test_update_different_map_elements"],
    )
    base.update(overrides)
    return VerificationInputs(**base)


class TestSpecs:
    def test_every_swe_gym_repo_has_a_spec(self) -> None:
        repos = {
            "getmoto/moto", "python/mypy", "conan-io/conan", "iterative/dvc", "dask/dask", "pydantic/pydantic",
            "pandas-dev/pandas", "facebookresearch/hydra", "bokeh/bokeh", "Project-MONAI/MONAI", "modin-project/modin",
        }  # fmt: skip
        assert repos <= set(SPECS)
        assert sum(len(v) for v in SPECS.values()) == 200

    def test_every_spec_runs_pytest(self) -> None:
        for repo, versions in SPECS.items():
            for version, spec in versions.items():
                assert spec["test_cmd"].startswith("pytest"), (repo, version)

    def test_unknown_version_is_a_named_error_not_a_fallback(self) -> None:
        with pytest.raises(KeyError, match="getmoto/moto version '0.1'"):
            spec_for("getmoto/moto", "0.1")

    def test_directives_skip_data_files_touched_by_the_test_patch(self) -> None:
        patch = (
            "diff --git a/tests/test_a.py b/tests/test_a.py\n"
            "diff --git a/tests/data/fixture.json b/tests/data/fixture.json\n"
            "diff --git a/tests/data/table.csv b/tests/data/table.csv\n"
        )
        assert touched_test_files("getmoto/moto", patch) == ["tests/test_a.py"]

    def test_mypy_selects_cases_from_the_test_patch(self) -> None:
        patch = "+[case testNarrowingUnion]\n+[case testNarrowingNone]\n"
        assert (
            pytest_command("python/mypy", "1.4", patch)
            == 'pytest -n0 -rA -k "testNarrowingUnion or testNarrowingNone"'
        )

    def test_mypy_without_cases_runs_the_files_without_the_dangling_k(self) -> None:
        """The harness recipe would emit `-k mypy/test/x.py`, which pytest rejects; seen on 4 SWE-Gym rows."""
        patch = "diff --git a/mypy/test/test_find_sources.py b/mypy/test/test_find_sources.py\n"
        assert pytest_command("python/mypy", "1.4", patch) == "pytest -n0 -rA mypy/test/test_find_sources.py"


class TestParseLogPytest:
    def test_reads_every_status_and_strips_failure_reasons(self) -> None:
        log = (
            "PASSED tests/a.py::test_one\n"
            "FAILED tests/a.py::test_two - AssertionError: 1 != 2\n"
            "ERROR tests/a.py::test_three - ImportError\n"
            "XFAIL tests/a.py::test_four\n"
            "SKIPPED [1] tests/a.py:3: no reason\n"
            "=== 1 failed, 1 passed in 0.1s ===\n"
        )
        statuses = parse_log_pytest(log)
        assert statuses["tests/a.py::test_one"] == "PASSED"
        assert statuses["tests/a.py::test_two"] == "FAILED"
        assert statuses["tests/a.py::test_three"] == "ERROR"
        assert statuses["tests/a.py::test_four"] == "XFAIL"
        assert "=== 1 failed, 1 passed in 0.1s ===" not in statuses

    def test_ignores_lines_that_only_name_a_status(self) -> None:
        assert parse_log_pytest("PASSED\nFAILED\n") == {}

    def test_reads_status_last_verbose_lines_as_pydantic_prints_them(self) -> None:
        """pydantic's pytest-pretty plugin replaces the -rA summary, so the `-vv` progress lines
        (`name STATUS`) are the only per-test record; SWE-bench's v2 parser reads these too."""
        log = (
            "tests/test_aliases.py::test_alias_generator PASSED\n"
            "tests/test_create_model.py::test_x[bool-field_info0] PASSED\n"
            "tests/test_json_schema.py::test_secrets[a b] XFAIL\n"
            "tests/test_json_schema.py::test_y XPASS (working on V2)\n"
            "==================================== PASSES ====================================\n"
            "Results (3.10s):\n       225 passed\n"
        )
        statuses = parse_log_pytest(log)
        assert statuses == {
            "tests/test_aliases.py::test_alias_generator": "PASSED",
            "tests/test_create_model.py::test_x[bool-field_info0]": "PASSED",
            "tests/test_json_schema.py::test_secrets[a": "XFAIL",  # truncated at the space, like the dataset ids
        }

    def test_verbose_ids_are_truncated_at_whitespace_like_the_dataset(self) -> None:
        """SWE-Gym's test lists came from a whitespace-splitting parser, so `t[a: b]` is stored as `t[a:`."""
        statuses = parse_log_pytest("tests/a.py::t[border-left: 2pt solid red-attrs10] PASSED\n")
        assert statuses == {"tests/a.py::t[border-left:": "PASSED"}

    def test_strips_ansi_colour_and_ignores_prose_ending_in_a_status_word(self) -> None:
        assert parse_log_pytest("\x1b[32mtests/c.py::v PASSED\x1b[0m\nall tests have PASSED\n") == {
            "tests/c.py::v": "PASSED"
        }


class TestNormalizeTestId:
    @pytest.mark.parametrize(
        "escaped, decoded",
        [
            (r"t[/the-key-un\xeecode/test]", "t[/the-key-unîcode/test]"),
            (r"t[https://example.verm\xf6gensberatung]", "t[https://example.vermögensberatung]"),
            (r"t[\u73e0\u5b9d]", "t[珠宝]"),
            (r"t[a\nb]", "t[a\nb]"),
            ("t[already-decoded-ñ]", "t[already-decoded-ñ]"),
            ("t[plain]", "t[plain]"),
        ],
    )
    def test_decodes_pytests_escaped_parameter_ids(self, escaped, decoded) -> None:
        assert normalize_test_id(escaped) == decoded


class TestGrade:
    def test_resolves_only_when_every_required_test_passes(self) -> None:
        statuses = {"t::a": "PASSED", "t::b": "PASSED"}
        assert grade(statuses, ["t::a"], ["t::b"])["resolved"] is True

    def test_xfail_counts_as_passing_like_the_swebench_harness(self) -> None:
        assert grade({"t::a": "XFAIL"}, ["t::a"], [])["resolved"] is True

    def test_a_missing_test_counts_as_not_passing(self) -> None:
        report = grade({"t::a": "PASSED"}, ["t::a"], ["t::missing"])
        assert report["resolved"] is False
        assert report["PASS_TO_PASS"]["failure"] == ["t::missing"]

    def test_skipped_and_error_are_not_passed(self) -> None:
        assert grade({"t::a": "SKIPPED"}, ["t::a"], [])["resolved"] is False
        assert grade({"t::a": "ERROR"}, ["t::a"], [])["resolved"] is False

    def test_escaped_observed_ids_match_the_datasets_decoded_ones(self) -> None:
        statuses = {r"tests/s3.py::t[/the-key-un\xeecode/test]": "PASSED"}
        assert grade(statuses, ["tests/s3.py::t[/the-key-unîcode/test]"], [])["resolved"] is True

    def test_reports_which_tests_failed(self) -> None:
        report = grade({"t::a": "FAILED", "t::b": "PASSED"}, ["t::a", "t::b"], [])
        assert report["FAIL_TO_PASS"] == {"success": ["t::b"], "failure": ["t::a"]}
        assert report["tests_observed"] == 2


class TestAsList:
    @pytest.mark.parametrize(
        "value, expected",
        [(["a", "b"], ["a", "b"]), ('["a", "b"]', ["a", "b"]), ("", []), (None, []), ("not json", ["not json"])],
    )
    def test_accepts_lists_and_their_json_string_form(self, value, expected) -> None:
        assert as_list(value) == expected


class TestBuildEvalScript:
    def test_activates_the_testbed_env_and_fails_loudly_without_it(self) -> None:
        script = build_eval_script(_inputs())
        assert "source /opt/miniconda3/bin/activate" in script
        assert f"conda activate testbed >/dev/null 2>&1 || exit {EXIT_NO_CONDA}" in script
        assert f"cd /testbed || exit {EXIT_NO_REPO}" in script

    def test_applies_the_patch_strictly_with_the_harness_fallback(self) -> None:
        script = build_eval_script(_inputs())
        assert "git apply -v /tmp/nemo_gym_patch.diff" in script
        assert "patch --batch --fuzz=5 -p1 -i /tmp/nemo_gym_patch.diff" in script
        assert f"echo {PATCH_FAILED}" in script

    def test_follows_the_swebench_recipe_in_order(self) -> None:
        script = build_eval_script(_inputs())
        reset = "git checkout 7f6c9cb1deafb280fe7fcc7551c38e397f11a706 tests/test_dynamodb/test_update.py"
        order = [
            script.index("make init"),  # moto 5.0's install step, re-run as the harness does
            script.index(reset),
            script.index("git apply -v /tmp/nemo_gym_test_patch.diff"),
            script.index(TEST_OUTPUT_BEGIN),
            script.index("pytest -n0 -rA tests/test_dynamodb/test_update.py"),
            script.index(TEST_OUTPUT_END),
            script.rindex(reset),
        ]
        assert order == sorted(order), "install, reset, test patch, markers, tests, reset must stay in that order"

    def test_flags_a_test_patch_that_fails_to_apply(self) -> None:
        assert TEST_PATCH_FAILED in build_eval_script(_inputs())

    def test_omits_the_patch_step_when_there_is_no_patch(self) -> None:
        script = build_eval_script(_inputs(patch=""))
        assert "nemo_gym_patch.diff" not in script

    def test_includes_eval_commands_when_the_spec_has_them(self) -> None:
        script = build_eval_script(_inputs(repo="conan-io/conan", version="2.0"))
        assert "export PYTHONPATH=${PYTHONPATH:-}:$(pwd)" in script

    def test_propagates_the_test_exit_code(self) -> None:
        script = build_eval_script(_inputs())
        assert script.rstrip().endswith("exit $__test_exit")


class TestVerificationFiles:
    def test_ships_the_script_and_only_the_patches_that_exist(self) -> None:
        files = verification_files(_inputs(test_patch=""))
        assert set(files) == {"/tmp/nemo_gym_eval.sh", "/tmp/nemo_gym_patch.diff"}
        files = verification_files(_inputs())
        assert "/tmp/nemo_gym_test_patch.diff" in files


class TestDropTestPatchFiles:
    def test_drops_the_model_sections_for_files_the_test_patch_touches(self) -> None:
        model = (
            "diff --git a/moto/x.py b/moto/x.py\n--- a/moto/x.py\n+++ b/moto/x.py\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1 +1 @@\n-c\n+d\n"
        )
        test_patch = "diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"
        kept = drop_test_patch_files(model, test_patch)
        assert "moto/x.py" in kept and "tests/test_x.py" not in kept

    def test_keeps_the_patch_when_there_is_no_test_patch(self) -> None:
        assert drop_test_patch_files("diff --git a/x b/x\n", "") == "diff --git a/x b/x\n"


class _FakeSandbox:
    def __init__(self, stdout: str, return_code: int = 0, stderr: str = "") -> None:
        self.result = SimpleNamespace(stdout=stdout, stderr=stderr, return_code=return_code)
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_s=None, cwd=None):
        self.commands.append(command)
        return self.result


class TestRunVerification:
    @pytest.mark.asyncio
    async def test_grades_a_passing_run(self) -> None:
        log = (
            f"{TEST_OUTPUT_BEGIN}\nPASSED tests/test_dynamodb/test_update.py::test_update_item_add_float\n"
            f"PASSED tests/test_dynamodb/test_update.py::test_update_different_map_elements\n{TEST_OUTPUT_END}\n"
        )
        result = await run_verification(_FakeSandbox(log), _inputs())
        assert result.completed and result.resolved and result.patch_applied

    @pytest.mark.asyncio
    async def test_a_patch_that_does_not_apply_is_a_completed_zero(self) -> None:
        result = await run_verification(_FakeSandbox(f"error: patch failed\n{PATCH_FAILED}\n"), _inputs())
        assert result.completed is True and result.resolved is False and result.patch_applied is False
        assert result.error == "patch does not apply"

    @pytest.mark.asyncio
    async def test_a_test_patch_that_does_not_apply_is_incomplete(self) -> None:
        log = f"{TEST_PATCH_FAILED}\n{TEST_OUTPUT_BEGIN}\n{TEST_OUTPUT_END}\n"
        result = await run_verification(_FakeSandbox(log), _inputs())
        assert result.completed is False and result.test_patch_failed is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", [EXIT_NO_REPO, EXIT_NO_CONDA])
    async def test_a_wrong_image_is_incomplete_not_a_zero(self, code) -> None:
        result = await run_verification(_FakeSandbox("", return_code=code), _inputs())
        assert result.completed is False and result.test_results is None

    @pytest.mark.asyncio
    async def test_only_the_marked_region_reaches_the_parser(self) -> None:
        log = (
            "PASSED tests/test_dynamodb/test_update.py::test_update_different_map_elements\n"  # install noise
            f"{TEST_OUTPUT_BEGIN}\nPASSED tests/test_dynamodb/test_update.py::test_update_item_add_float\n{TEST_OUTPUT_END}\n"
        )
        result = await run_verification(_FakeSandbox(log), _inputs())
        assert result.resolved is False
        assert result.test_results["PASS_TO_PASS"]["failure"] == [
            "tests/test_dynamodb/test_update.py::test_update_different_map_elements"
        ]


class TestSliceTestOutput:
    def test_keeps_only_the_graded_region(self) -> None:
        assert slice_test_output(f"a{TEST_OUTPUT_BEGIN}b{TEST_OUTPUT_END}c") == "b"

    def test_falls_back_to_the_whole_log_when_markers_are_absent(self) -> None:
        assert slice_test_output("plain") == "plain"


class TestPrepare:
    def test_image_name_uses_swe_gyms_separator(self) -> None:
        from resources_servers.swe_gym.prepare_swe_gym import image_name

        assert image_name("getmoto__moto-7365") == "docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest"
        # Docker repository names are lower case; the Hub only knows the lower-cased MONAI images.
        assert (
            image_name("Project-MONAI__MONAI-3715")
            == "docker.io/xingyaoww/sweb.eval.x86_64.project-monai_s_monai-3715:latest"
        )

    def test_rows_drop_hints_and_prompt_only_the_issue(self) -> None:
        from resources_servers.swe_gym.prepare_swe_gym import build_row

        row = build_row(
            {
                "instance_id": "getmoto__moto-7365",
                "repo": "getmoto/moto",
                "version": 5.0,
                "base_commit": "abc",
                "patch": "fix",
                "test_patch": "tests",
                "problem_statement": "issue text",
                "hints_text": "the fix is to change X",
                "created_at": "2024",
                "FAIL_TO_PASS": '["t::a"]',
                "PASS_TO_PASS": ["t::b"],
            }
        )
        assert "hints_text" not in row and "created_at" not in row
        assert row["version"] == "5.0" and row["FAIL_TO_PASS"] == ["t::a"] and row["PASS_TO_PASS"] == ["t::b"]
        assert row["responses_create_params"]["input"] == [{"role": "user", "content": "issue text"}]
        assert row["agent_ref"]["name"] == "swe_gym_opencode_sandboxed_agent"


class TestVerifyResponseShape:
    """BaseVerifyResponse extends BaseVerifyRequest, so the request fields are REQUIRED on the way out."""

    @staticmethod
    def _body() -> dict:
        return {
            "instance_id": "getmoto__moto-7365",
            "repo": "getmoto/moto",
            "version": "5.0",
            "base_commit": "abc",
            "image_name": "docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest",
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
        from resources_servers.swe_gym.app import SWEGymVerifyResponse

        response = SWEGymVerifyResponse.model_validate(
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
        assert response.repo == "getmoto/moto" and response.version == "5.0", (
            "row identity must survive into sweep outputs"
        )

    def test_omitting_the_request_fields_is_rejected(self) -> None:
        import pydantic

        from resources_servers.swe_gym.app import SWEGymVerifyResponse

        with pytest.raises(pydantic.ValidationError, match="responses_create_params|response"):
            SWEGymVerifyResponse.model_validate(
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

    def test_request_accepts_test_lists_as_json_strings(self) -> None:
        from resources_servers.swe_gym.app import SWEGymInstanceRequest

        body = SWEGymInstanceRequest.model_validate(self._body() | {"FAIL_TO_PASS": '["t::a"]'})
        assert as_list(body.FAIL_TO_PASS) == ["t::a"]


class TestAgentSandbox:
    """What the agent's sandbox is and is not given."""

    @staticmethod
    def _source() -> str:
        return (SERVER_DIR / "app.py").read_text()

    def test_seed_session_calls_the_shared_anti_cheat_helper(self) -> None:
        source = self._source()
        assert "from resources_servers.swebench.anti_cheat import apply_anti_cheat_setup" in source
        assert "apply_anti_cheat_setup(" in source

    def test_config_enables_anti_cheating_by_default(self) -> None:
        from resources_servers.swe_gym.app import SWEGymResourcesServerConfig

        assert SWEGymResourcesServerConfig.model_fields["apply_anti_cheating"].default is True

    def test_agent_facing_config_turns_it_on(self) -> None:
        config = yaml.safe_load((SERVER_DIR / "configs" / "swe_gym.yaml").read_text())
        assert config["swe_gym_resources_server"]["resources_servers"]["swe_gym"]["apply_anti_cheating"] is True

    def test_the_agent_sandbox_gets_no_row_files(self) -> None:
        """seed_session creates the sandbox with no `files`; only verify seeds eval files."""
        source = self._source()
        seed = source[source.index("async def seed_session") : source.index("def _response")]
        assert "_create_sandbox(body)" in seed and "files=" not in seed

    def test_the_testbed_env_is_first_on_path(self) -> None:
        from resources_servers.swe_gym.app import TESTBED_ENV

        assert TESTBED_ENV["PATH"].startswith("/opt/miniconda3/envs/testbed/bin:")
        assert TESTBED_ENV["CONDA_DEFAULT_ENV"] == "testbed"

    def test_the_app_wires_cpu_cap_env_into_the_spec(self) -> None:
        source = self._source()
        assert "cpu_cap_env(sandbox_resources.cpu)" in source and "env=env," in source


class TestInSandboxCancellation:
    """A CancelledError raised from inside the sandbox client (a long run tripping an inner timeout) must
    become an incomplete verdict, while a cancellation of the request itself still propagates."""

    def test_verify_distinguishes_the_two(self) -> None:
        source = (SERVER_DIR / "app.py").read_text()
        verify = source[source.index("async def verify") :]
        assert "except asyncio.CancelledError:" in verify
        assert "asyncio.current_task().cancelling()" in verify and "raise" in verify
        assert "Verification cancelled inside the sandbox client" in verify


class TestMultiWorkerEntrypoint:
    def test_exposes_a_module_level_app_for_forked_workers(self) -> None:
        source = (SERVER_DIR / "app.py").read_text()
        assert "is_nemo_gym_fastapi_entrypoint(__file__)" in source
        assert "app = SWEGymResourcesServer.run_webserver()" in source

    def test_the_config_that_needs_it_sets_num_workers(self) -> None:
        config = yaml.safe_load((SERVER_DIR / "configs" / "swe_gym.yaml").read_text())
        golden = config["swe_gym_golden_patch_resources_server"]["resources_servers"]["swe_gym"]
        assert golden["num_workers"] > 1 and golden["is_verifying_golden_patch"] is True


class TestGoldenPatchAggregation:
    @staticmethod
    def _obs(completed: bool, resolved: bool) -> dict:
        return {"evaluation_completed": completed, "resolved": resolved}

    def test_buckets(self) -> None:
        from resources_servers.swe_gym.aggregate_golden_patch import classify

        assert classify([self._obs(True, True)] * 3, 3) == "supported"
        assert classify([self._obs(True, True), self._obs(True, False), self._obs(True, True)], 3) == "flaky"
        assert classify([self._obs(True, False)] * 3, 3) == "broken"
        assert classify([self._obs(True, True), self._obs(True, True), self._obs(False, False)], 3) == "inconclusive"
        assert classify([self._obs(True, True)] * 2, 3) == "inconclusive"


def test_build_payload_empty_patch_control_blanks_only_the_patch():
    example = {"instance_id": "x__1", "patch": "diff --git a/f.py b/f.py\n", "repo": "x"}
    golden = build_payload(example)
    assert golden["patch"] == example["patch"]
    assert golden["response"] == EMPTY_RESPONSE
    assert golden["responses_create_params"] == {"input": []}
    control = build_payload(example, empty_patch=True)
    assert control["patch"] == ""
    assert control["instance_id"] == "x__1"
    assert example["patch"] != ""  # the input row is left untouched
