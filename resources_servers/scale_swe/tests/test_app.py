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
"""Tests for the Scale-SWE resources server's grading and script construction.

These cover the parts that decide whether a task counts as resolved. The sandbox lifecycle
itself is exercised by the golden-patch run, which needs a live OpenSandbox endpoint.

Several tests here pin bugs that were actually hit building resources_servers/swe_rebench (the
sibling server): a verify response missing the echoed request fields, and a multi-worker
entrypoint with no module-level `app`. Both looked fine locally and both only failed once real
traffic hit them. Writing the same tests here from the start is cheaper than re-discovering the
same two bugs on a second dataset.
"""

from types import SimpleNamespace

import pytest

from resources_servers.scale_swe.verification import (
    F2P_SCRIPT_NAME,
    TEST_OUTPUT_BEGIN,
    TEST_OUTPUT_END,
    VerificationInputs,
    as_id_list,
    build_eval_script,
    extract_statuses,
    grade,
    run_verification,
    slice_test_output,
    unique_test_files,
    verification_files,
)


def _inputs(**overrides) -> VerificationInputs:
    base = dict(
        instance_id="auth0_auth0-python_pr671",
        workdir="/workspace/auth0-python",
        patch="diff --git a/a b/a\n",
        pre_commands="git checkout deadbeef -f && git reset --hard HEAD",
        f2p_patch="",
        f2p_script="import pytest\ndef test_thing(): assert True\n",
        fail_to_pass=[f"{F2P_SCRIPT_NAME}::test_thing"],
        pass_to_pass=["tests/test_base.py::TestBase::test_get"],
    )
    base.update(overrides)
    return VerificationInputs(**base)


class TestAsIdList:
    def test_decodes_the_json_encoded_string_the_dataset_ships(self) -> None:
        # FAIL_TO_PASS / PASS_TO_PASS arrive as JSON-encoded strings in this dataset, not lists.
        assert as_id_list('["a::b", "c::d"]') == ["a::b", "c::d"]

    def test_accepts_an_already_decoded_list(self) -> None:
        assert as_id_list(["a::b"]) == ["a::b"]

    @pytest.mark.parametrize("value", [None, "", "[]"])
    def test_empty_inputs_yield_no_ids(self, value) -> None:
        assert as_id_list(value) == []

    def test_malformed_json_falls_back_to_a_single_literal_id_rather_than_crashing(self) -> None:
        # A row-level data error must degrade to "one id that will not match anything" (and
        # so masks/fails safely) rather than raise and take the whole verification down with it.
        assert as_id_list("not json") == ["not json"]


class TestTestFilesFor:
    def test_extracts_unique_files_preserving_first_seen_order(self) -> None:
        ids = ["b/test.py::test_1", "a/test.py::test_2", "b/test.py::test_3"]
        assert unique_test_files(ids) == ["b/test.py", "a/test.py"]

    def test_a_row_with_dozens_of_node_ids_still_yields_one_entry_per_file(self) -> None:
        # A real row can carry up to ~69 files across hundreds of node ids; pytest is pointed
        # at files, not the full id list, specifically to avoid a huge command line.
        ids = [f"pkg/test_{i}.py::test_case_{j}" for i in range(5) for j in range(20)]
        assert len(unique_test_files(ids)) == 5


class TestExtractStatuses:
    """extract_statuses matches against the row's OWN known ids rather than parsing an id out
    of free text -- see the docstring on the function for why a whitespace-delimited regex is
    wrong for this dataset. These tests are built from a real failure: one row in a 200-row
    validation pass had 954/954 tests pass, but the naive parser truncated every parametrized
    id at its first space and scored the row as a near-total regression."""

    def test_reads_passed_and_failed_for_plain_ids(self) -> None:
        log = "PASSED test_a.py::test_1\nFAILED test_a.py::test_2 - AssertionError\n"
        statuses = extract_statuses(log, ["test_a.py::test_1", "test_a.py::test_2"])
        assert statuses["test_a.py::test_1"] == "PASSED"
        assert statuses["test_a.py::test_2"] == "FAILED"

    def test_only_reports_the_ids_it_was_asked_about(self) -> None:
        log = "PASSED test_a.py::test_1\nPASSED test_a.py::test_unrelated\n"
        assert extract_statuses(log, ["test_a.py::test_1"]) == {"test_a.py::test_1": "PASSED"}

    def test_handles_a_parametrized_id_containing_a_literal_space(self) -> None:
        # The actual bug: a whitespace-delimited regex truncates this at "[No".
        node_id = "tests/test_parser.py::test_parse_address[No address here-None]"
        log = f"PASSED {node_id}\n"
        assert extract_statuses(log, [node_id]) == {node_id: "PASSED"}

    def test_handles_a_failed_id_whose_own_text_contains_the_reason_delimiter(self) -> None:
        # A real dataset id: "...[2590 Elm Road NE - Warren, OH 44483-expected1]" contains a
        # literal " - " inside its brackets, so splitting a FAILED line on " - " to find where
        # the id ends and the reason begins is ambiguous in general. Matching against the known
        # id directly sidesteps that.
        node_id = "tests/test_parser.py::test_parse_address[2590 Elm Road NE - Warren, OH 44483-expected1]"
        log = f"FAILED {node_id} - AssertionError: mismatch\n"
        assert extract_statuses(log, [node_id]) == {node_id: "FAILED"}

    def test_a_short_id_is_not_mistaken_for_a_prefix_of_a_longer_unrelated_one(self) -> None:
        # "test_foo" must not match inside a line reporting "test_foo_extra", and trying the
        # longer candidate id first is what prevents it.
        log = "PASSED test_a.py::test_foo_extra\n"
        assert extract_statuses(log, ["test_a.py::test_foo"]) == {}

    def test_longer_ids_are_preferred_when_both_are_targets(self) -> None:
        log = "PASSED test_a.py::test_foo_extra\n"
        statuses = extract_statuses(log, ["test_a.py::test_foo", "test_a.py::test_foo_extra"])
        assert statuses == {"test_a.py::test_foo_extra": "PASSED"}

    def test_later_lines_win_when_a_test_id_appears_twice(self) -> None:
        # pytest prints progress dots/lines first and the short summary last; the summary is
        # authoritative.
        log = "FAILED test_a.py::test_1\nsome retry output\nPASSED test_a.py::test_1\n"
        assert extract_statuses(log, ["test_a.py::test_1"]) == {"test_a.py::test_1": "PASSED"}

    def test_unrecognised_output_yields_no_statuses(self) -> None:
        assert extract_statuses("collecting ... \ninternal error\n", ["test_a.py::test_1"]) == {}

    def test_an_id_that_never_appears_is_absent_not_a_crash(self) -> None:
        assert extract_statuses("PASSED test_a.py::test_1\n", ["test_a.py::test_missing"]) == {}


class TestSliceTestOutput:
    def test_keeps_only_the_graded_region(self) -> None:
        log = f"pip installing\nFAILED bogus\n{TEST_OUTPUT_BEGIN}\nPASSED real\n{TEST_OUTPUT_END}\ncleanup"
        sliced = slice_test_output(log)
        assert "PASSED real" in sliced
        assert "FAILED bogus" not in sliced
        assert "cleanup" not in sliced

    def test_falls_back_to_the_whole_log_when_markers_are_absent(self) -> None:
        assert slice_test_output("PASSED a") == "PASSED a"


class TestGrade:
    def test_resolves_only_when_every_required_test_passes(self) -> None:
        report = grade({"f2p": "PASSED", "p2p": "PASSED"}, ["f2p"], ["p2p"])
        assert report["resolved"] is True

    def test_a_failing_fail_to_pass_does_not_resolve(self) -> None:
        report = grade({"f2p": "FAILED", "p2p": "PASSED"}, ["f2p"], ["p2p"])
        assert report["resolved"] is False
        assert report["FAIL_TO_PASS"]["failure"] == ["f2p"]

    def test_a_regressed_pass_to_pass_does_not_resolve(self) -> None:
        report = grade({"f2p": "PASSED", "p2p": "FAILED"}, ["f2p"], ["p2p"])
        assert report["resolved"] is False

    def test_a_missing_test_counts_as_not_passing(self) -> None:
        """A test id that never appears in the output (a collection error, e.g.) must not read
        as success or an empty pytest run scores as a resolved instance."""
        report = grade({}, ["f2p"], ["p2p"])
        assert report["resolved"] is False
        assert report["tests_observed"] == 0

    def test_skipped_is_not_passed(self) -> None:
        assert grade({"f2p": "SKIPPED"}, ["f2p"], [])["resolved"] is False


class TestBuildEvalScript:
    def test_runs_pre_commands_before_anything_else(self) -> None:
        # pre_commands performs the checkout of parent_commit and scrubs the fix commit out of
        # git history; applying a patch before it would be immediately discarded.
        script = build_eval_script(_inputs())
        assert script.index("git checkout deadbeef") < script.index("nemo_gym_patch.diff")

    def test_signals_a_missing_workdir(self) -> None:
        script = build_eval_script(_inputs())
        assert "exit 97" in script
        assert any(line.startswith("cd ") and "/workspace/auth0-python" in line for line in script.splitlines())

    def test_omits_the_patch_step_when_there_is_no_patch(self) -> None:
        script = build_eval_script(_inputs(patch=""))
        assert "nemo_gym_patch.diff" not in script

    def test_installs_f2p_script_from_tmp_not_by_writing_into_the_repo_first(self) -> None:
        # f2p_script cannot be written into the repo before pre_commands: `git clean -fd` (part
        # of every row's pre_commands) would delete it. It is uploaded to /tmp and copied in
        # afterwards instead.
        script = build_eval_script(_inputs())
        assert f"cp /tmp/nemo_gym_f2p_script.py {F2P_SCRIPT_NAME}" in script
        copy_at = script.index("cp /tmp/nemo_gym_f2p_script.py")
        pre_at = script.index("git checkout deadbeef")
        assert pre_at < copy_at

    def test_omits_the_f2p_script_step_when_the_row_has_none(self) -> None:
        script = build_eval_script(_inputs(f2p_script=""))
        assert "nemo_gym_f2p_script.py" not in script

    def test_applies_f2p_patch_when_present(self) -> None:
        script = build_eval_script(_inputs(f2p_patch="diff --git a/t b/t\n"))
        assert "nemo_gym_f2p.diff" in script

    def test_targets_only_the_files_named_by_fail_to_pass_and_pass_to_pass(self) -> None:
        script = build_eval_script(_inputs(fail_to_pass=["a/test_x.py::t1"], pass_to_pass=["b/test_y.py::t2"]))
        assert "a/test_x.py" in script
        assert "b/test_y.py" in script

    def test_brackets_the_test_command_with_markers(self) -> None:
        script = build_eval_script(_inputs())
        assert script.index(TEST_OUTPUT_BEGIN) < script.index("pytest") < script.index(TEST_OUTPUT_END)

    def test_propagates_the_test_exit_code_not_the_setup_ones(self) -> None:
        script = build_eval_script(_inputs())
        assert "__test_exit=$?" in script and "exit $__test_exit" in script
        # pre_commands and the patch steps must not abort the script (`set +e`) or a row whose
        # history-scrub emits a nonzero exit never reaches the tests at all.
        assert "set +e" in script


class TestVerificationFiles:
    def test_ships_the_script_and_only_the_pieces_that_exist(self) -> None:
        files = verification_files(_inputs(patch="", f2p_patch=""))
        assert "/tmp/nemo_gym_eval.sh" in files
        assert "/tmp/nemo_gym_patch.diff" not in files
        assert "/tmp/nemo_gym_f2p.diff" not in files
        assert "/tmp/nemo_gym_f2p_script.py" in files

    def test_ships_the_f2p_patch_when_the_row_has_one(self) -> None:
        files = verification_files(_inputs(f2p_patch="diff --git a/t b/t\n"))
        assert "/tmp/nemo_gym_f2p.diff" in files


class _FakeSandbox:
    def __init__(self, stdout: str, return_code: int = 0) -> None:
        self._result = SimpleNamespace(stdout=stdout, stderr="", return_code=return_code)
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_s=None):
        self.commands.append(command)
        return self._result


class TestRunVerification:
    @pytest.mark.asyncio
    async def test_grades_a_passing_run(self) -> None:
        log = f"{TEST_OUTPUT_BEGIN}\nPASSED {F2P_SCRIPT_NAME}::test_thing\n{TEST_OUTPUT_END}"
        sandbox = _FakeSandbox(log)
        # Isolate the thing under test (marker slicing + grading of one passing id) from the
        # unrelated PASS_TO_PASS id the shared fixture carries by default.
        result = await run_verification(sandbox=sandbox, inputs=_inputs(pass_to_pass=[]))
        assert result.completed and result.resolved
        assert sandbox.commands == ["bash /tmp/nemo_gym_eval.sh"]

    @pytest.mark.asyncio
    async def test_missing_workdir_is_incomplete_not_a_zero(self) -> None:
        result = await run_verification(sandbox=_FakeSandbox("", return_code=97), inputs=_inputs())
        assert result.completed is False
        assert result.resolved is False
        assert "not present in the image" in result.error

    @pytest.mark.asyncio
    async def test_only_the_marked_region_reaches_the_parser(self) -> None:
        log = (
            f"pip: FAILED to fetch\n{TEST_OUTPUT_BEGIN}\n"
            f"PASSED {F2P_SCRIPT_NAME}::test_thing\n{TEST_OUTPUT_END}\ntrailing"
        )
        result = await run_verification(sandbox=_FakeSandbox(log), inputs=_inputs(pass_to_pass=[]))
        assert result.resolved is True  # would be masked by the install-time "FAILED" if unsliced


class TestVerifyResponseShape:
    """BaseVerifyResponse extends BaseVerifyRequest, so the request fields are REQUIRED on the
    way out. Building the response from scratch drops them and the resulting ValidationError
    surfaces to the caller as a bare JSON string -- which is how a fully successful verification
    can still fail the whole sweep. (Hit for real in resources_servers/swe_rebench/app.py.)"""

    @staticmethod
    def _body() -> dict:
        return {
            "instance_id": "auth0_auth0-python_pr671",
            "workdir": "/workspace/auth0-python",
            "image_url": "aweaiteam/scaleswe:auth0_auth0-python_pr671",
            "language": "python",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
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
        from resources_servers.scale_swe.app import ScaleSWEVerifyResponse

        response = ScaleSWEVerifyResponse.model_validate(
            self._body()
            | {
                "reward": 1.0,
                "evaluation_completed": True,
                "resolved": True,
                "patch_applied": True,
                "test_results": {"resolved": True},
                "test_output": "",
                "error": None,
                "eval_sandbox_start_time_taken": 1.0,
                "patch_verification_time_taken": 2.0,
            }
        )
        assert response.reward == 1.0 and response.resolved

    def test_omitting_the_request_fields_is_rejected(self) -> None:
        import pydantic

        from resources_servers.scale_swe.app import ScaleSWEVerifyResponse

        with pytest.raises(pydantic.ValidationError, match="responses_create_params|response"):
            ScaleSWEVerifyResponse.model_validate(
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
        assert "app = ScaleSWEResourcesServer.run_webserver()" in source

    def test_the_config_that_needs_it_still_sets_num_workers(self) -> None:
        """Pins the pair: if num_workers is configured, the entrypoint branch must exist."""
        from pathlib import Path

        import yaml

        config = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "scale_swe.yaml").read_text())
        for name, block in config.items():
            workers = block["resources_servers"]["scale_swe"].get("num_workers")
            if workers and workers > 1:
                assert "is_nemo_gym_fastapi_entrypoint(__file__)" in self._source(), (
                    f"{name} sets num_workers={workers} but the entrypoint has no module-level app"
                )
