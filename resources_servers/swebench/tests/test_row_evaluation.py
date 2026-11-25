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
"""A row that carries its own image, evaluation script, log parser and evaluation type is evaluated with them."""

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

from nemo_gym.client import ServerClient
from nemo_gym.sandbox import SandboxExecResult
from resources_servers.swebench.app import (
    DockerContainer,
    SwebenchResourcesServer,
    SwebenchResourcesServerConfig,
    SWEBenchVerifyRequest,
)
from resources_servers.swebench.row_evaluation import (
    END_TEST_OUTPUT,
    START_TEST_OUTPUT,
    RowTestSpec,
    get_eval_tests_report,
    get_logs_eval,
    get_row_eval_report,
    insert_test_asset_restore,
    karma_runs_completed,
    make_row_test_spec,
)
from resources_servers.swebench.swebench_patches import (
    EVAL_SCRIPT_COMMAND,
    OPENLAYERS_CHROME_WRAPPER,
    XVFB_RUN_WRAPPER,
    run_instance,
)


# A row as the SWE-bench Multimodal test split ships it: the evaluation script resets the test files, applies the
# test patch, runs the tests between the markers, and resets the test files again.
EVAL_SCRIPT = "\n".join(
    [
        "#!/bin/bash",
        "set -uxo pipefail",
        "cd /testbed",
        "git checkout c83fccf packages/react/src/a.test.js",
        "git apply -v - <<'EOF_114329324912'",
        "diff --git a/packages/react/src/a.test.js b/packages/react/src/a.test.js",
        "EOF_114329324912",
        f": '{START_TEST_OUTPUT}'",
        "yarn jest packages/react/src/a.test.js",
        f": '{END_TEST_OUTPUT}'",
        "git checkout c83fccf packages/react/src/a.test.js",
    ]
)
F2P = "packages/react/src/a.test.js > renders the label"
P2P = "packages/react/src/a.test.js > renders"
ROW = {
    "instance_id": "carbon-design-system__carbon-7619",
    "repo": "carbon-design-system/carbon",
    "base_commit": "c83fccf",
    "version": "10.20",
    "patch": "diff --git a/packages/react/src/a.js b/packages/react/src/a.js\n",
    "test_patch": "diff --git a/packages/react/src/a.test.js b/packages/react/src/a.test.js\n",
    "problem_statement": "The label is missing",
    "hints_text": "",
    "created_at": "2021-01-01T00:00:00Z",
    "FAIL_TO_PASS": json.dumps([F2P]),
    "PASS_TO_PASS": json.dumps([P2P]),
    "environment_setup_commit": "",
    "difficulty": "",
    "subset": "multimodal",
    "split": "test",
    "image": "swebench/sweb.eval.x86_64.carbon-design-system_1776_carbon-7619:latest",
    "eval_script": EVAL_SCRIPT,
    "log_parser": "parse_log_carbon",
    "eval_type": "pass_and_fail",
    "responses_create_params": {"input": []},
    "response": {
        "id": "resp",
        "created_at": 0.0,
        "model": "m",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    },
}
PNG = b"\x89PNG\r\n\x1a\n" + b"reference"


def spec_for(row: dict = ROW, **overrides: Any) -> RowTestSpec:
    row = row | overrides
    return make_row_test_spec(
        instance_id=row["instance_id"],
        repo=row["repo"],
        version=row["version"],
        base_commit=row["base_commit"],
        image=row["image"],
        eval_script=row["eval_script"],
        log_parser=row["log_parser"],
        eval_type=row["eval_type"],
        fail_to_pass=row["FAIL_TO_PASS"],
        pass_to_pass=row["PASS_TO_PASS"],
    )


def make_server(**config: Any) -> SwebenchResourcesServer:
    return SwebenchResourcesServer(
        config=SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            sandbox_provider="test",
            sandbox_config=dict(),
            is_verifying_golden_patch=True,
            **config,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def run_output(tests_printed: str, *, exit_code: int = 0) -> str:
    """What the row's script prints with its streams joined: bash's trace with the markers, and the tests."""
    return (
        f"+ : '{START_TEST_OUTPUT}'\n+ yarn jest packages/react/src/a.test.js\n{tests_printed}\n"
        f"+ SWEBENCH_TEST_EXIT_CODE={exit_code}\n+ : '{END_TEST_OUTPUT}'\n>>>>> Test Exit Code: {exit_code}\n"
    )


class TestRowTestSpec:
    def test_the_spec_runs_the_rows_image_and_script_and_records_the_test_exit_code(self) -> None:
        spec = spec_for()
        assert spec.instance_image_key == ROW["image"] and spec.is_remote_image
        assert spec.FAIL_TO_PASS == [F2P] and spec.PASS_TO_PASS == [P2P]
        lines = spec.eval_script.splitlines()
        # The shebang and shell options appear once, and the exit status is taken right after the test command.
        assert lines[:2] == ["#!/bin/bash", "set -uxo pipefail"] and lines.count("set -uxo pipefail") == 1
        end = lines.index(f": '{END_TEST_OUTPUT}'")
        assert lines[end - 2 : end + 2] == [
            "yarn jest packages/react/src/a.test.js",
            "SWEBENCH_TEST_EXIT_CODE=$?",
            f": '{END_TEST_OUTPUT}'",
            'echo ">>>>> Test Exit Code: $SWEBENCH_TEST_EXIT_CODE"',
        ]

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"eval_script": None, "log_parser": ""}, "no eval_script, log_parser"),
            ({"log_parser": "parse_log_pytest"}, "names log parser 'parse_log_pytest'"),
            ({"eval_type": "pass_only"}, "has eval_type 'pass_only'"),
        ],
    )
    def test_a_row_that_does_not_define_its_evaluation_is_rejected(self, overrides: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            spec_for(**overrides)

    def test_test_assets_are_copied_in_after_the_test_patch_and_before_the_tests(self) -> None:
        spec = spec_for()
        insert_test_asset_restore(
            spec, [("/tmp/staged-0", "packages/react/src/__snapshots__/a b.png"), ("/tmp/staged-1", "x/y.png")]
        )
        start = spec.eval_script_list.index(f": '{START_TEST_OUTPUT}'")
        assert spec.eval_script_list[start - 2 : start] == [
            "mkdir -p $(dirname 'packages/react/src/__snapshots__/a b.png') && cp /tmp/staged-0 "
            "'packages/react/src/__snapshots__/a b.png'",
            "mkdir -p $(dirname x/y.png) && cp /tmp/staged-1 x/y.png",
        ]
        assert spec.eval_script_list[start - 3] == "EOF_114329324912"


class TestRowGrading:
    def _report(self, tmp_path: Path, output: str, *, model_patch: str | None = "diff", **overrides: Any) -> dict:
        log = tmp_path / "test_output.txt"
        log.write_text(output)
        spec = spec_for(**overrides)
        prediction = {"instance_id": spec.instance_id, "model_patch": model_patch}
        return get_row_eval_report(spec, prediction, str(log), include_tests_status=True)[spec.instance_id]

    def test_pass_and_fail_needs_every_required_test_to_pass(self, tmp_path: Path) -> None:
        passing = run_output(f"PASS {F2P} (12ms)\nPASS {P2P} (3ms)")
        assert self._report(tmp_path, passing)["resolved"] is True
        regression = run_output(f"PASS {F2P} (12ms)\nFAIL {P2P} (3ms)", exit_code=1)
        report = self._report(tmp_path, regression)
        assert report["resolved"] is False and report["tests_status"]["PASS_TO_PASS"]["failure"] == [P2P]
        # Under pass_and_fail a required test missing from the log has not passed.
        assert self._report(tmp_path, run_output(f"PASS {P2P} (3ms)"))["resolved"] is False

    def test_fail_only_reads_a_missing_test_as_passing_only_from_a_suite_that_ran(self, tmp_path: Path) -> None:
        fail_only = {"eval_type": "fail_only", "log_parser": "parse_log_openlayers"}
        ran = run_output("Chrome Headless 120.0.0.0 (Linux x86_64): Executed 980 of 980 SUCCESS (12 secs)")
        assert self._report(tmp_path, ran, **fail_only)["resolved"] is True
        never_ran = run_output("ERROR [launcher]: Cannot start ChromeHeadless\nExecuted 0 of 0 ERROR")
        report = self._report(tmp_path, never_ran, **fail_only)
        assert report["resolved"] is False and report["patch_successfully_applied"] is False
        failed = run_output(f"Chrome Headless 120.0.0.0 (Linux x86_64) {F2P} FAILED\nExecuted 980 of 980 (1 FAILED)")
        assert self._report(tmp_path, failed, **fail_only)["resolved"] is False

    def test_a_log_that_disagrees_with_the_test_exit_status_has_no_result(self, tmp_path: Path) -> None:
        # A patch can print its own PASS lines; the tests exiting non-zero with no failure in the log is not a pass.
        forged = run_output(f"PASS {F2P} (12ms)\nPASS {P2P} (3ms)", exit_code=1)
        assert self._report(tmp_path, forged)["resolved"] is False

    def test_a_skipped_required_test_is_not_a_resolution_but_a_skipped_kept_test_is_no_regression(self) -> None:
        report = get_eval_tests_report({F2P: "SKIPPED", P2P: "SKIPPED"}, [F2P], [P2P], eval_type="pass_and_fail")
        assert report["FAIL_TO_PASS"]["failure"] == [F2P] and report["PASS_TO_PASS"]["success"] == [P2P]

    def test_the_rows_parser_reads_the_region_between_the_markers(self) -> None:
        status_map, found = get_logs_eval(spec_for(), "PASS before the markers\n" + run_output(f"PASS {F2P} (12ms)"))
        assert found and status_map == {F2P: "PASSED"}

    def test_output_without_both_markers_or_without_a_patch_is_unresolved(self, tmp_path: Path) -> None:
        assert self._report(tmp_path, f"PASS {F2P} (12ms)\nPASS {P2P} (3ms)")["resolved"] is False
        report = self._report(tmp_path, run_output(f"PASS {F2P} (12ms)\nPASS {P2P} (3ms)"), model_patch=None)
        assert report["patch_is_None"] is True and report["resolved"] is False


class TestKarmaRunsCompleted:
    @pytest.mark.parametrize(
        ("output", "completed"),
        [
            # Summaries as bpmn-js and openlayers print them, ANSI escapes included.
            ("\x1b[1A\x1b[2KHeadlessChrome 151.0.0 (Linux 0.0.0): Executed 1339 of 1346 (skipped 7) SUCCESS", True),
            (
                "Executed 3 of 1338\nExecuted 1331 of 1338 (7 FAILED) (skipped 7) (1 min 1.577 secs / 12.085 secs)",
                True,
            ),
            ("Chrome 120.0.0.0 (Linux x86_64): Executed 980 of 980 SUCCESS (12 secs)", True),
            ("Chrome 151.0.0 (Linux 0.0.0): Executed 1723 of 3052 (2 FAILED) (5.611 secs / 2.785 secs)", False),
            ("Executed 0 of 0 ERROR", False),
            ("18812 passing\n15 failing", True),
        ],
    )
    def test_a_karma_run_is_complete_when_its_summary_accounts_for_every_planned_test(
        self, output: str, completed: bool
    ) -> None:
        assert karma_runs_completed(output) is completed


class TestServer:
    def test_the_row_source_builds_the_spec_from_the_row_with_its_test_assets(self) -> None:
        asset = {
            "path": "x/y.png",
            "url": "https://example.com/y.png",
            "content_base64": base64.b64encode(PNG).decode(),
        }
        request = SWEBenchVerifyRequest.model_validate(ROW | {"image_assets": {"test_patch": [asset]}})
        spec = make_server(evaluation_spec_source="row")._make_test_spec(request)
        assert isinstance(spec, RowTestSpec) and spec.instance_image_key == ROW["image"]
        assert "mkdir -p $(dirname x/y.png) && cp /tmp/nemo-gym-swebench-test-asset-0 x/y.png" in spec.eval_script_list

    def test_the_harness_source_ignores_the_rows_evaluation_fields(self, monkeypatch: MonkeyPatch) -> None:
        received: dict = {}

        def make_test_spec(instance: dict, **kwargs: Any) -> MagicMock:
            received.update(instance)
            return MagicMock(eval_script_list=[], repo=instance["repo"], version=instance["version"])

        monkeypatch.setattr("resources_servers.swebench.app.make_test_spec", make_test_spec)
        make_server()._make_test_spec(SWEBenchVerifyRequest.model_validate(ROW))
        assert received["instance_id"] == ROW["instance_id"]
        assert not {"image", "eval_script", "log_parser", "eval_type"} & received.keys()

    def test_openlayers_gets_the_software_webgl_wrapper_before_its_tests_and_other_repos_do_not(self) -> None:
        server = make_server(evaluation_spec_source="row")
        openlayers = server._make_test_spec(
            SWEBenchVerifyRequest.model_validate(ROW | {"repo": "openlayers/openlayers", "eval_type": "fail_only"})
        )
        start = openlayers.eval_script_list.index(f": '{START_TEST_OUTPUT}'")
        assert openlayers.eval_script_list[start - 1] == OPENLAYERS_CHROME_WRAPPER
        assert openlayers.eval_script_list.count(OPENLAYERS_CHROME_WRAPPER) == 1
        carbon = server._make_test_spec(SWEBenchVerifyRequest.model_validate(ROW))
        assert OPENLAYERS_CHROME_WRAPPER not in carbon.eval_script_list

    def test_a_script_that_runs_xvfb_waits_for_the_previous_x_server(self) -> None:
        server = make_server(evaluation_spec_source="row")
        xvfb = 'xvfb-run --server-args="-screen 0 1280x1024x24 -ac :99" su chromeuser -c "npm run test select"'
        script = EVAL_SCRIPT.replace("yarn jest packages/react/src/a.test.js", f"{xvfb}\n{xvfb}")
        spec = server._make_test_spec(SWEBenchVerifyRequest.model_validate(ROW | {"eval_script": script}))
        start = spec.eval_script_list.index(f": '{START_TEST_OUTPUT}'")
        assert spec.eval_script_list[start - 1] == XVFB_RUN_WRAPPER
        plain = server._make_test_spec(SWEBenchVerifyRequest.model_validate(ROW))
        assert XVFB_RUN_WRAPPER not in plain.eval_script_list

    def test_the_row_source_rejects_a_row_without_its_evaluation(self) -> None:
        row = {key: value for key, value in ROW.items() if key != "eval_script"}
        with pytest.raises(ValueError, match="no eval_script"):
            make_server(evaluation_spec_source="row")._make_test_spec(SWEBenchVerifyRequest.model_validate(row))


class _FakeEvalSandbox:
    """A verification sandbox in which every command succeeds and the evaluation command prints ``output``."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.eval_script = ""

    async def exec(self, command: str, **kwargs: Any) -> SandboxExecResult:
        if command != EVAL_SCRIPT_COMMAND:
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        return SandboxExecResult(stdout=self.output, stderr="", return_code=0)

    async def upload(self, local_path: Path, remote_path: str) -> None:
        if remote_path == "/eval.sh":
            self.eval_script = Path(local_path).read_bytes().decode()

    async def stop(self) -> None:
        pass


class TestRunInstance:
    async def _run(self, monkeypatch: MonkeyPatch, tmp_path: Path, output: str, **overrides: Any) -> tuple[dict, str]:
        monkeypatch.chdir(tmp_path)  # run_instance writes its logs under the working directory
        request = SWEBenchVerifyRequest.model_validate(ROW | overrides)
        test_spec = make_server(evaluation_spec_source="row")._make_test_spec(request)
        sandbox = _FakeEvalSandbox(output)
        container = DockerContainer(id="run-id", instance_id=test_spec.instance_id)
        container._inner_container = sandbox
        result = await run_instance(
            test_spec=test_spec,
            pred={"instance_id": test_spec.instance_id, "model_patch": ROW["patch"]},
            rm_image=False,
            force_rebuild=False,
            client=container,
            run_id="run-id",
        )
        return result, sandbox.eval_script

    async def test_the_rows_script_runs_and_its_parser_grades_the_output(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        result, script = await self._run(monkeypatch, tmp_path, run_output(f"PASS {F2P} (12ms)\nPASS {P2P} (3ms)"))
        assert result["completed"] is True and result["resolved"] is True and result["error"] is None
        assert "yarn jest packages/react/src/a.test.js\nSWEBENCH_TEST_EXIT_CODE=$?" in script

    async def test_a_failing_required_test_is_a_completed_unresolved_evaluation(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        result, _ = await self._run(monkeypatch, tmp_path, run_output(f"FAIL {F2P} (12ms)\nPASS {P2P}", exit_code=1))
        assert result["completed"] is True and result["resolved"] is False

    async def test_a_fail_only_row_is_not_held_to_the_harness_suite_checks(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        # The harness's failure-only completion checks know only Chart.js, marked and p5.js; the row grader's own
        # evidence that the suite ran decides for the test split's fail_only repositories.
        output = run_output("Chrome Headless 120.0.0.0 (Linux x86_64): Executed 980 of 980 SUCCESS (12 secs)")
        result, _ = await self._run(
            monkeypatch,
            tmp_path,
            output,
            repo="openlayers/openlayers",
            log_parser="parse_log_openlayers",
            eval_type="fail_only",
        )
        assert result["completed"] is True and result["resolved"] is True

    async def test_a_test_patch_with_crlf_line_endings_reaches_the_sandbox_unchanged(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        # The test patch of a file with CRLF line endings applies only with its carriage returns intact.
        crlf_hunk = "-import {\r\n+import { inject,\r\n"
        script = EVAL_SCRIPT.replace("EOF_114329324912\n:", f"{crlf_hunk}EOF_114329324912\n:")
        assert crlf_hunk in script
        _, uploaded = await self._run(
            monkeypatch, tmp_path, run_output(f"PASS {F2P} (12ms)\nPASS {P2P} (3ms)"), eval_script=script
        )
        assert crlf_hunk in uploaded

    async def test_a_fail_only_karma_run_that_stopped_part_way_has_no_verdict(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        # Missing tests count as passing under fail_only, so a run that stopped before them must not resolve.
        output = run_output("Chrome 151.0.0 (Linux 0.0.0): Executed 1723 of 3052 (2 FAILED) (5.611 secs / 2.785 secs)")
        result, _ = await self._run(
            monkeypatch,
            tmp_path,
            output,
            repo="openlayers/openlayers",
            log_parser="parse_log_openlayers",
            eval_type="fail_only",
        )
        assert result["completed"] is False and result["resolved"] is False
        assert "did not execute every planned test" in result["error"]
