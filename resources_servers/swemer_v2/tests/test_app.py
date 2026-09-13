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
"""Tests for the Swemer-v2 resources server's grading and script construction.

Two classes here pin bugs that were actually hit building the sibling SWE resources servers
(scale_swe, swe_rebench): a verify response missing the echoed request fields, and a
multi-worker entrypoint with no module-level `app`. Both looked fine locally and both only
failed once real traffic hit them, so the same tests are written here from the start.
"""

import json

from resources_servers.swemer_v2.verification import (
    SUPPORTED_FRAMEWORKS,
    TEST_OUTPUT_BEGIN,
    TEST_OUTPUT_END,
    VerificationInputs,
    build_eval_script,
    grade,
    inject_output_flags,
    slice_test_output,
    verification_files,
)


def _inputs(**overrides) -> VerificationInputs:
    base = dict(
        instance_id="1024pix-pix-11380-agentic-v2",
        workdir="/workspace/repo",
        patch="diff --git a/a b/a\n",
        test_patch="diff --git a/b/test_a.py b/b/test_a.py\n",
        test_framework="pytest",
        test_command="pytest tests/test_x.py -v",
        fail_to_pass=["tests.test_x::test_one"],
        pass_to_pass=["tests.test_x::test_two"],
    )
    base.update(overrides)
    return VerificationInputs(**base)


class TestInjectOutputFlags:
    def test_pytest_appends_rA_when_absent(self) -> None:
        assert inject_output_flags("pytest", "pytest tests/test_x.py -v").endswith(" -rA")

    def test_pytest_does_not_duplicate_rA(self) -> None:
        cmd = "pytest tests/test_x.py -rA"
        assert inject_output_flags("pytest", cmd) == cmd

    def test_go_inserts_json_after_go_test(self) -> None:
        out = inject_output_flags("go", "cd /workspace/repo && go test ./... -count=1")
        assert "go test -json ./... -count=1" in out

    def test_go_does_not_duplicate_json(self) -> None:
        cmd = "go test -json ./... -count=1"
        assert inject_output_flags("go", cmd) == cmd

    def test_jest_appends_json(self) -> None:
        assert inject_output_flags("jest", "npx jest foo.test.ts").endswith(" --json")

    def test_mocha_appends_reporter_json_even_if_another_reporter_present(self) -> None:
        out = inject_output_flags("mocha", "mocha --reporter tap test.js")
        # Last --reporter flag wins on mocha's CLI, so appending overrides the existing one.
        assert out.endswith(" --reporter json")

    def test_vitest_appends_reporter_json(self) -> None:
        assert inject_output_flags("vitest", "vitest run foo.test.ts").endswith(" --reporter=json")

    def test_unsupported_framework_raises(self) -> None:
        import pytest as _pytest

        with _pytest.raises(ValueError):
            inject_output_flags("maven", "mvn test")


class TestBuildEvalScript:
    def test_applies_patch_then_test_patch(self) -> None:
        script = build_eval_script(_inputs())
        patch_idx = script.index("nemo_gym_patch.diff")
        test_patch_idx = script.index("nemo_gym_test_patch.diff")
        assert patch_idx < test_patch_idx

    def test_omits_test_patch_apply_when_empty(self) -> None:
        script = build_eval_script(_inputs(test_patch=""))
        assert "nemo_gym_test_patch.diff" not in script

    def test_markers_bracket_the_test_command(self) -> None:
        script = build_eval_script(_inputs(test_command="pytest tests/test_x.py -v"))
        assert TEST_OUTPUT_BEGIN in script
        assert TEST_OUTPUT_END in script
        assert "pytest tests/test_x.py -v -rA" in script

    def test_slice_test_output_extracts_between_markers(self) -> None:
        log = f"noise before\n{TEST_OUTPUT_BEGIN}\nPASSED tests/test_x.py::test_one\n{TEST_OUTPUT_END}\nnoise after"
        sliced = slice_test_output(log)
        assert "noise before" not in sliced
        assert "noise after" not in sliced
        assert "PASSED tests/test_x.py::test_one" in sliced


class TestVerificationFiles:
    def test_includes_both_patches_when_present(self) -> None:
        files = verification_files(_inputs())
        assert "/tmp/nemo_gym_patch.diff" in files
        assert "/tmp/nemo_gym_test_patch.diff" in files
        assert "/tmp/nemo_gym_eval.sh" in files

    def test_omits_blank_patch(self) -> None:
        files = verification_files(_inputs(patch="   "))
        assert "/tmp/nemo_gym_patch.diff" not in files


class TestGrade:
    def test_resolved_requires_every_target_id_passing(self) -> None:
        statuses = {"a": "PASSED", "b": "PASSED"}
        report = grade(statuses, fail_to_pass=["a"], pass_to_pass=["b"])
        assert report["resolved"] is True

    def test_missing_id_counts_as_failed_not_passed(self) -> None:
        statuses = {"a": "PASSED"}
        report = grade(statuses, fail_to_pass=["a"], pass_to_pass=["b"])
        assert report["resolved"] is False
        assert "b" in report["PASS_TO_PASS"]["failure"]

    def test_failed_status_counts_as_failed(self) -> None:
        statuses = {"a": "FAILED"}
        report = grade(statuses, fail_to_pass=["a"], pass_to_pass=[])
        assert report["resolved"] is False


class TestSupportedFrameworks:
    def test_covers_exactly_phase_one(self) -> None:
        assert SUPPORTED_FRAMEWORKS == frozenset({"pytest", "go", "jest", "mocha", "vitest"})


class TestVerifyResponseShape:
    """BaseVerifyResponse extends BaseVerifyRequest, so the request fields are REQUIRED on the
    way out. Building the response from scratch drops them and the resulting ValidationError
    surfaces to the caller as a bare JSON string -- which is how a fully successful verification
    can still fail the whole sweep. (Hit for real in resources_servers/swe_rebench/app.py.)"""

    @staticmethod
    def _body() -> dict:
        return {
            "instance_id": "1024pix-pix-11380-agentic-v2",
            "delivery": "delivery_08_06",
            "workdir": "/workspace/repo",
            "image_ref": "942195279341.dkr.ecr.us-east-2.amazonaws.com/ext-nvidia-agentic-v2:delivery_08_06__1024pix-pix-11380-agentic-v2",
            "language": "javascript",
            "test_framework": "mocha",
            "test_command": "mocha test.js",
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
        from resources_servers.swemer_v2.app import SwemerV2VerifyResponse

        response = SwemerV2VerifyResponse.model_validate(
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
        assert response.instance_id == "1024pix-pix-11380-agentic-v2"
        assert response.test_framework == "mocha"

    def test_building_from_scratch_without_the_request_fields_fails(self) -> None:
        import pytest

        from resources_servers.swemer_v2.app import SwemerV2VerifyResponse

        with pytest.raises(Exception):
            SwemerV2VerifyResponse.model_validate(
                {
                    "reward": 1.0,
                    "evaluation_completed": True,
                    "resolved": True,
                    "patch_applied": True,
                    "instance_id": "x",
                    "language": "javascript",
                    "test_framework": "mocha",
                    "test_results": None,
                    "test_output": "",
                    "error": None,
                    "eval_sandbox_start_time_taken": 0.1,
                    "patch_verification_time_taken": 0.2,
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
        assert "app = SwemerV2ResourcesServer.run_webserver()" in source

    def test_the_config_that_needs_it_still_sets_num_workers(self) -> None:
        """Pins the pair: if num_workers is configured, the entrypoint branch must exist."""
        from pathlib import Path

        import yaml

        config = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "swemer_v2.yaml").read_text())
        for name, block in config.items():
            workers = block["resources_servers"]["swemer_v2"].get("num_workers")
            if workers and workers > 1:
                assert "is_nemo_gym_fastapi_entrypoint(__file__)" in self._source(), (
                    f"{name} sets num_workers={workers} but the entrypoint has no module-level app"
                )


class TestFrameworkParsersAgainstRealCapturedOutput:
    """Each fixture is real sandbox output: golden patch applied, `inject_output_flags`
    command run for real inside a live OpenSandbox pod against the task's actual image (see
    the module docstring / README for how these were captured). A golden patch is expected to
    resolve, so `grade()` reporting `resolved: True` here is an end-to-end check that both the
    flag injection and the parser agree with what really happened -- not just that the parser
    doesn't crash.
    """

    @staticmethod
    def _fixtures_dir():
        from pathlib import Path

        return Path(__file__).parent / "fixtures"

    @classmethod
    def _manifest(cls) -> dict:
        import json

        return json.loads((cls._fixtures_dir() / "manifest.json").read_text())

    @classmethod
    def _output(cls, framework: str) -> str:
        return (cls._fixtures_dir() / f"{framework}_output.txt").read_text()

    def _assert_resolved(self, framework: str) -> None:
        from resources_servers.swemer_v2.verification import grade, parse_statuses

        entry = self._manifest()[framework]
        output = self._output(framework)
        statuses = parse_statuses(framework, output, entry["fail_to_pass"] + entry["pass_to_pass"])
        report = grade(statuses, entry["fail_to_pass"], entry["pass_to_pass"])
        assert report["resolved"], (
            f"{framework}: expected the golden-patch capture to resolve; "
            f"failed F2P={report['FAIL_TO_PASS']['failure']} failed P2P={report['PASS_TO_PASS']['failure']}"
        )

    def test_pytest_golden_capture_resolves(self) -> None:
        self._assert_resolved("pytest")

    def test_go_golden_capture_resolves(self) -> None:
        self._assert_resolved("go")

    def test_jest_golden_capture_resolves(self) -> None:
        self._assert_resolved("jest")

    def test_mocha_golden_capture_resolves(self) -> None:
        self._assert_resolved("mocha")

    def test_vitest_golden_capture_resolves(self) -> None:
        self._assert_resolved("vitest")

    def test_pytest_parser_matches_class_less_test(self) -> None:
        from resources_servers.swemer_v2.verification import _parse_pytest

        log = "PASSED tests/test_engine.py::test_x[query_args0]\n"
        assert _parse_pytest(log, ["tests.test_engine::test_x[query_args0]"]) == {
            "tests.test_engine::test_x[query_args0]": "PASSED"
        }

    def test_pytest_parser_matches_class_qualified_test(self) -> None:
        """Real bug: a naive dots-to-slashes transform of the dataset's id cannot tell a
        module-path dot from a class-name dot (`tests.test_requirements.RequirementTreeTests`
        could be a 3-level package path or `module.ClassName`), and silently scored every
        class-based row as unresolved -- caught by a live golden-patch pilot, not by any
        synthetic fixture, since the one real capture used to build this parser had no
        class-based tests in it.
        """
        from resources_servers.swemer_v2.verification import _parse_pytest

        log = "PASSED tests/test_requirements.py::RequirementTreeTests::test_changed_child\n"
        target_id = "tests.test_requirements.RequirementTreeTests::test_changed_child"
        assert _parse_pytest(log, [target_id]) == {target_id: "PASSED"}

    def test_pytest_parser_class_qualified_failure_with_trailing_reason(self) -> None:
        from resources_servers.swemer_v2.verification import _parse_pytest

        log = "FAILED tests/test_x.py::TestFoo::test_bar - AssertionError: boom\n"
        target_id = "tests.test_x.TestFoo::test_bar"
        assert _parse_pytest(log, [target_id]) == {target_id: "FAILED"}

    def test_go_parser_ignores_run_and_output_events(self) -> None:
        from resources_servers.swemer_v2.verification import _parse_go

        log = "\n".join(
            [
                '{"Action":"run","Package":"pkg","Test":"TestFoo"}',
                '{"Action":"output","Package":"pkg","Test":"TestFoo","Output":"=== RUN\\n"}',
                '{"Action":"pass","Package":"pkg","Test":"TestFoo","Elapsed":0}',
            ]
        )
        assert _parse_go(log, ["pkg::TestFoo"]) == {"pkg::TestFoo": "PASSED"}

    def test_jest_like_parser_ignores_trailing_npm_notice(self) -> None:
        from resources_servers.swemer_v2.verification import _parse_jest_like

        blob = '{"testResults":[{"name":"f.js","assertionResults":[{"fullName":"a","status":"passed"}]}]}'
        noisy = blob + "\nnpm notice\nnpm notice New major version of npm available!\n"
        assert _parse_jest_like(noisy, ["f.js::a"]) == {"f.js::a": "PASSED"}

    def test_mocha_parser_handles_both_id_shapes(self) -> None:
        from resources_servers.swemer_v2.verification import _parse_mocha

        blob = json.dumps({"passes": [{"fullTitle": "Suite does a thing"}], "failures": [], "pending": []})
        assert _parse_mocha(blob, ["::Suite does a thing"]) == {"::Suite does a thing": "PASSED"}
        assert _parse_mocha(blob, ["Suite does a thing"]) == {"Suite does a thing": "PASSED"}

    def test_jest_like_parser_reports_failed_status(self) -> None:
        from resources_servers.swemer_v2.verification import _parse_jest_like

        blob = json.dumps(
            {
                "testResults": [
                    {
                        "name": "f.js",
                        "assertionResults": [
                            {"fullName": "a", "status": "passed"},
                            {"fullName": "b", "status": "failed"},
                        ],
                    }
                ]
            }
        )
        assert _parse_jest_like(blob, ["f.js::a", "f.js::b"]) == {"f.js::a": "PASSED", "f.js::b": "FAILED"}

    def test_mocha_parser_reports_failed_from_failures_array(self) -> None:
        from resources_servers.swemer_v2.verification import _parse_mocha

        blob = json.dumps(
            {
                "passes": [{"fullTitle": "Suite passes"}],
                "failures": [{"fullTitle": "Suite fails"}],
                "pending": [],
            }
        )
        result = _parse_mocha(blob, ["Suite passes", "Suite fails"])
        assert result == {"Suite passes": "PASSED", "Suite fails": "FAILED"}

    def test_go_parser_reports_failed_action(self) -> None:
        from resources_servers.swemer_v2.verification import _parse_go

        log = '{"Action":"fail","Package":"pkg","Test":"TestFoo","Elapsed":0}'
        assert _parse_go(log, ["pkg::TestFoo"]) == {"pkg::TestFoo": "FAILED"}

    def test_pytest_parser_reports_failed_status(self) -> None:
        from resources_servers.swemer_v2.verification import _parse_pytest

        log = "FAILED tests/test_x.py::test_bad\nPASSED tests/test_x.py::test_good\n"
        result = _parse_pytest(log, ["tests.test_x::test_bad", "tests.test_x::test_good"])
        assert result == {"tests.test_x::test_bad": "FAILED", "tests.test_x::test_good": "PASSED"}

    def test_pytest_parser_falls_back_to_plain_v_output(self) -> None:
        """Real bug: pytest 3.3.2 (pinned in one task's image) predates the `-rA` short-summary
        `A` category (added in pytest 3.6), so the flag is a silent no-op and the summary
        section this parser originally relied on never appears -- 198/198 real passes read as
        0 observed. Caught by the same live pilot run as the class-qualification bug above.
        """
        from resources_servers.swemer_v2.verification import _parse_pytest

        log = (
            "tests/test_client_request.py::test_close[pyloop] PASSED                  [ 94%]\n"
            "tests/test_x.py::TestFoo::test_bar FAILED                                 [ 95%]\n"
        )
        result = _parse_pytest(
            log, ["tests.test_client_request::test_close[pyloop]", "tests.test_x.TestFoo::test_bar"]
        )
        assert result == {
            "tests.test_client_request::test_close[pyloop]": "PASSED",
            "tests.test_x.TestFoo::test_bar": "FAILED",
        }

    def test_pytest_parser_prefers_rA_summary_when_both_shapes_present(self) -> None:
        """The `-rA` summary is authoritative (pytest prints it last, after the `-v` inline
        run) -- a later line for the same id overwrites an earlier one, so this is really just
        "process lines in order," but pinned explicitly since it's the reason overwriting is
        safe rather than a bug.
        """
        from resources_servers.swemer_v2.verification import _parse_pytest

        log = (
            "tests/test_x.py::test_flaky PASSED                                       [100%]\n"
            "========================== short test summary info ===========================\n"
            "FAILED tests/test_x.py::test_flaky - reran and failed on retry\n"
        )
        assert _parse_pytest(log, ["tests.test_x::test_flaky"]) == {"tests.test_x::test_flaky": "FAILED"}

    def test_mocha_parser_handles_real_file_path_prefix(self) -> None:
        """Real bug: two of three ids shapes were handled (empty `::title`, bare `title`), but
        a real file-path prefix (`/workspace/repo/.../Swap.js::title`) fell through untouched
        and never matched anything, silently scoring a fully-resolved row as broken. Caught by
        the same live pilot run as the pytest bugs above.
        """
        from resources_servers.swemer_v2.verification import _parse_mocha

        blob = json.dumps({"passes": [{"fullTitle": "Swap Unit setProtocolFee"}], "failures": [], "pending": []})
        target_id = "/workspace/repo/source/swap/test/Swap.js::Swap Unit setProtocolFee"
        assert _parse_mocha(blob, [target_id]) == {target_id: "PASSED"}


class TestInjectOutputFlagsSkipsOnlyBareTrailingExit:
    """Real bug, and a real over-correction of it, both caught on live full-set runs:

    1. Appending a flag to a command whose last statement is a bare `exit $VAR` corrupts it --
       confirmed on a "pytest" row that was a bash loop ending in `exit $EXIT`; ` -rA` made it
       `exit $EXIT -rA`, invalid for the `exit` builtin, breaking execution outright.
    2. The first fix for (1) skipped injection whenever the framework's own invocation word
       (`pytest`, `jest`, ...) was missing from the command, on the theory that meant a wrapper
       script. That cost 44 jest/mocha/vitest rows their correct resolution on a live rerun:
       most wrapper commands tolerate the appended flag fine, so "no tool word" was a bad proxy
       for "will corrupt". Only the trailing-`exit` shape actually does.
    """

    def test_bare_trailing_exit_var_is_left_untouched(self) -> None:
        cmd = 'for f in a.sh b.sh; do bash "$f" || EXIT=1; done; exit $EXIT'
        assert inject_output_flags("pytest", cmd) == cmd

    def test_bare_trailing_exit_status_is_left_untouched(self) -> None:
        cmd = "some_command; exit $?"
        assert inject_output_flags("jest", cmd) == cmd

    def test_bare_trailing_exit_literal_is_left_untouched(self) -> None:
        cmd = "some_command; exit 1"
        assert inject_output_flags("mocha", cmd) == cmd

    def test_wrapper_command_without_the_tool_word_still_gets_the_flag(self) -> None:
        """The over-corrected case: no `jest` in the command, but nothing about its shape (no
        trailing bare exit) suggests appending would break it, so it should still get injected
        -- this is exactly the shape of the 44 rows the old "skip if tool word missing" check
        cost a correct resolution on a live run.
        """
        assert inject_output_flags("jest", "/workspace/run_karma.sh").endswith(" --json")
        assert inject_output_flags("mocha", "python3 -c \"print('OK')\"").endswith(" --reporter json")
        assert inject_output_flags("vitest", "./custom_test_runner.sh --strict").endswith(" --reporter=json")

    def test_go_wrapper_command_without_go_test_is_untouched_by_the_replace(self) -> None:
        # go's injection is a targeted string replace of "go test", which is already a safe
        # no-op when that substring is absent -- nothing to special-case here.
        cmd = "./run_custom_go_checks.sh"
        assert inject_output_flags("go", cmd) == cmd

    def test_real_command_still_gets_the_flag(self) -> None:
        assert inject_output_flags("pytest", "pytest tests/test_x.py -v").endswith(" -rA")


class TestInjectOutputFlagsTrailingSubshellAndMultiInvocation:
    """Two more real bugs caught on a full-set rerun, both in commands that chain more than one
    statement rather than running the framework's CLI as a single trailing step.
    """

    def test_command_ending_in_subshell_close_paren_is_left_untouched(self) -> None:
        """Real bug: `getsentry-sentry-javascript-11564-agentic-v2`'s command chains several
        `(cd pkg && npx jest ...)` subshells with `;`. Appending ` --json` after the final `)`
        is a bash syntax error (`syntax error near unexpected token '--json'`) regardless of what
        came before it -- the exact same class of corruption as a trailing bare `exit`, just a
        different shape.
        """
        cmd = "cd /workspace/repo; (cd packages/opentelemetry && yarn build); (cd pkg && npx jest suite)"
        assert inject_output_flags("jest", cmd) == cmd

    def test_multiple_go_test_invocations_all_get_json(self) -> None:
        """Real bug: `erigontech-erigon-14994-agentic-v2`'s command runs `go test -v` twice,
        `;`-separated (one per package). `str.replace(..., count=1)` only jsonified the first
        invocation, so the second package's tests -- which is where this row's own FAIL_TO_PASS
        ids actually lived -- printed plain text `_parse_go` could never match, silently grading
        a fully-passing golden patch as fully unresolved.
        """
        cmd = "cd a && go test -v ./... -count=1; cd b && go test -v ./... -count=1"
        out = inject_output_flags("go", cmd)
        assert out.count("go test -json") == 2


class TestInjectOutputFlagsCatsOutputFileBack:
    """Real bug: several vitest/jest rows already request `--outputFile=<path>`, writing the
    JSON report to a file instead of stdout -- the only channel this harness captures. 5 of 8
    vitest rows using `--outputFile=` in one full-set run had no JSON anywhere in captured output
    for exactly this reason, despite the run itself passing.
    """

    def test_appends_cat_of_the_output_file(self) -> None:
        cmd = "vitest run foo.test.ts --reporter=json --outputFile=/workspace/test-results/output.json"
        out = inject_output_flags("vitest", cmd)
        assert out.endswith("; cat /workspace/test-results/output.json 2>/dev/null || true")

    def test_no_output_file_means_no_cat_suffix(self) -> None:
        out = inject_output_flags("vitest", "vitest run foo.test.ts")
        assert "cat" not in out

    def test_cat_suffix_is_still_appended_after_a_trailing_subshell_guard(self) -> None:
        # `; cat ...` is a safe statement separator after a closing `)`, unlike a bare flag, so
        # the two fixes compose rather than conflict.
        cmd = "(cd pkg && npx jest --outputFile=/tmp/out.json suite)"
        out = inject_output_flags("jest", cmd)
        assert out == cmd + "; cat /tmp/out.json 2>/dev/null || true"


class TestPytestSuffixMatchingIsSymmetric:
    """Real bug caught on a live full-set run, affecting rows in BOTH directions -- an earlier
    version of `_match_pytest_rest` only handled the dataset id being SHORTER than the real path
    (a missing leading package directory), not LONGER.
    """

    def test_dataset_id_longer_than_real_path_still_matches(self) -> None:
        """`buildbot-buildbot-8461-agentic-v2`'s command does `cd .../master && pytest ...`, so
        the real node id (`buildbot/test/unit/worker/test_protocols_msgpack.py::TestConnection::
        test_x`) never contains the `master` directory pytest was invoked from, but the dataset's
        own id does (`master.buildbot.test.unit.worker.test_protocols_msgpack.TestConnection::
        test_x`) -- a dataset id LONGER than any real path could ever confirm from output alone.
        """
        from resources_servers.swemer_v2.verification import _parse_pytest

        log = "PASSED buildbot/test/unit/worker/test_protocols_msgpack.py::TestConnection::test_x\n"
        target_id = "master.buildbot.test.unit.worker.test_protocols_msgpack.TestConnection::test_x"
        assert _parse_pytest(log, [target_id]) == {target_id: "PASSED"}


class TestPytestErrorNeverDowngradesPassed:
    def test_teardown_error_does_not_downgrade_a_passed_test(self) -> None:
        """Real bug, all 5 `getsentry-sentry-*` rows in one full-set run: every target id printed
        as both `PASSED <id>` (the test's own body ran correctly) and, later in the same `-rA`
        summary, `ERROR <id>` from an unrelated teardown-phase fixture. Plain last-line-wins
        graded every one of these fully-passing golden patches as fully unresolved.
        """
        from resources_servers.swemer_v2.verification import _parse_pytest

        log = (
            "PASSED tests/x/test_y.py::TestFoo::test_bar\n"
            "ERROR tests/x/test_y.py::TestFoo::test_bar\n"
            "========================= 1 passed, 1 errors in 1.00s ==========================\n"
        )
        target_id = "tests.x.test_y.TestFoo::test_bar"
        assert _parse_pytest(log, [target_id]) == {target_id: "PASSED"}

    def test_a_genuine_rerun_failure_still_overwrites_an_earlier_pass(self) -> None:
        """Unlike ERROR, a later FAILED line is a real verdict on the test itself (e.g. a rerun
        plugin re-executing a flaky test), so it must still win -- see
        `test_pytest_parser_prefers_rA_summary_when_both_shapes_present` for the base case this
        must not regress.
        """
        from resources_servers.swemer_v2.verification import _parse_pytest

        log = "PASSED tests/test_x.py::test_flaky\nFAILED tests/test_x.py::test_flaky\n"
        target_id = "tests.test_x::test_flaky"
        assert _parse_pytest(log, [target_id]) == {target_id: "FAILED"}


class TestMochaSpecReporterTextFallback:
    """Real bug affecting roughly a third of mocha rows in one full-set run: `test_command` wraps
    mocha inside a project's own runner script (ts-node, jake, npm), so the appended
    `--reporter json` lands on the wrapper's argv, never mocha's, and the run falls back to
    mocha's default `spec` text reporter. 29 of 49 non-karma mocha rows with no JSON output in
    one full-set run were exactly this: a real, fully-passing run with unambiguous checkmarks.
    """

    def test_passing_checkmarks_are_extracted_when_no_json_present(self) -> None:
        from resources_servers.swemer_v2.verification import _parse_mocha

        blob = (
            "\n  VectorValue\n"
            "    ✔ fromJSON reconstructs the value from toJSON\n"
            "    ✔ fromJSON parameter order does not matter (267ms)\n"
            "\n  2 passing (525ms)\n"
        )
        result = _parse_mocha(blob, ["fromJSON reconstructs the value from toJSON", "some other test"])
        assert result == {"fromJSON reconstructs the value from toJSON": "PASSED"}

    def test_json_reporter_output_is_still_preferred_when_present(self) -> None:
        """The fallback must never fire when real JSON output exists -- guards against silently
        losing the more-precise `fullTitle`-based match to a looser text scan.
        """
        from resources_servers.swemer_v2.verification import _parse_mocha

        blob = json.dumps({"passes": [{"fullTitle": "Suite does a thing"}], "failures": [], "pending": []})
        assert _parse_mocha(blob, ["Suite does a thing"]) == {"Suite does a thing": "PASSED"}

    def test_karma_output_with_no_checkmarks_still_raises(self) -> None:
        # Karma output has no mocha checkmarks and no JSON -- correctly surfaces as unparseable
        # rather than silently grading every target id as failed with no error at all.
        import pytest as _pytest

        from resources_servers.swemer_v2.verification import _parse_mocha

        blob = "Karma v6.3.16 server started at http://localhost:9876/\n"
        with _pytest.raises(ValueError):
            _parse_mocha(blob, ["some test"])
