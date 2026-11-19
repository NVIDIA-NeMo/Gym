# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grading, script construction and seeding behavior of swe_runner_report."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client.server_client import ServerClient
from nemo_gym.sandbox.providers.base import SandboxExecResult
from resources_servers.swe_runner_report import verification as v


F2P = ["src/a.test.ts::suite fixes the bug"]
P2P = ["src/a.test.ts::suite keeps working"]
WORKDIR = "/app/repo"


def _inputs(**overrides) -> v.VerificationInputs:
    base = dict(
        instance_id="org__repo-1",
        workdir=WORKDIR,
        patch="",
        test_patch="diff --git a/src/a.test.ts b/src/a.test.ts\n--- a/src/a.test.ts\n+++ b/src/a.test.ts\n@@ -1 +1 @@\n-a\n+b\n",
        test_argv=["node", "vitest.mjs", "run", "--outputFile.json={results}", "{specs}"],
        results_format="vitest",
        spec_files=["src/a.test.ts"],
        fail_to_pass=F2P,
        pass_to_pass=P2P,
        source_prefixes=["src/"],
        source_suffixes=[".ts"],
    )
    return v.VerificationInputs(**(base | overrides))


def _vitest_report(fix: str, keep: str = "passed") -> dict:
    return {
        "testResults": [
            {
                "name": f"{WORKDIR}/src/a.test.ts",
                "assertionResults": [
                    {"fullName": "suite fixes the bug", "status": fix},
                    {"fullName": "suite keeps working", "status": keep},
                ],
            }
        ]
    }


def _script_output(returncode: int, report: object | None) -> str:
    body = json.dumps(report) if report is not None else ""
    return (
        f"{v.TEST_OUTPUT_BEGIN}\nrunner log\n{v.TEST_OUTPUT_END}\n{v.TEST_EXIT}{returncode}\n"
        f"{v.REPORT_BEGIN}\n{body}\n\n{v.REPORT_END}\n"
    )


class TestRunnerStatuses:
    def test_vitest_ids_are_workdir_relative(self) -> None:
        assert v.runner_statuses(_vitest_report("passed"), "vitest", WORKDIR) == {F2P[0]: "passed", P2P[0]: "passed"}

    def test_mocha_report(self) -> None:
        report = {
            "passes": [{"file": f"{WORKDIR}/test/x.js", "fullTitle": "x works"}],
            "failures": [{"file": f"{WORKDIR}/test/x.js", "fullTitle": "x breaks"}],
            "pending": [],
        }
        assert v.runner_statuses(report, "mocha", WORKDIR) == {
            "test/x.js::x works": "passed",
            "test/x.js::x breaks": "failed",
        }

    @pytest.mark.parametrize("report", [[], {"testResults": "x"}, {"testResults": [{"assertionResults": [1]}]}])
    def test_malformed_vitest_report_is_no_report(self, report) -> None:
        assert v.runner_statuses(report, "vitest", WORKDIR) is None


class TestGrade:
    def test_resolves_only_with_every_required_test_passing_and_a_clean_exit(self) -> None:
        statuses = v.runner_statuses(_vitest_report("passed"), "vitest", WORKDIR)
        assert v.grade(statuses, F2P, P2P, returncode=0)["resolved"] is True
        # Candidate code runs in the test process: a passing report from a failed or hung run never resolves.
        assert v.grade(statuses, F2P, P2P, returncode=1)["classification"] == "nonzero_exit"
        assert v.grade(statuses, F2P, P2P, returncode=124)["classification"] == "tests_timed_out"

    def test_failing_or_missing_required_tests(self) -> None:
        failed = v.grade(v.runner_statuses(_vitest_report("failed"), "vitest", WORKDIR), F2P, P2P, returncode=1)
        assert failed["resolved"] is False and failed["FAIL_TO_PASS"]["counts"]["failure"] == 1
        missing = v.grade({}, F2P, P2P, returncode=0)
        assert missing["classification"] == "unresolved" and missing["PASS_TO_PASS"]["counts"]["missing"] == 1
        assert v.grade(None, F2P, P2P, returncode=0)["classification"] == "no_report"


class TestReplayedPatch:
    def test_only_editable_source_sections_are_replayed(self) -> None:
        patch = (
            "diff --git a/src/a.ts b/src/a.ts\n--- a/src/a.ts\n+++ b/src/a.ts\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/src/a.test.ts b/src/a.test.ts\n--- a/src/a.test.ts\n+++ b/src/a.test.ts\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/package.json b/package.json\n--- a/package.json\n+++ b/package.json\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/src/b.ts b/lib/b.ts\nsimilarity index 100%\nrename from src/b.ts\nrename to lib/b.ts\n"
        )
        kept, dropped = _inputs(patch=patch).replayed_patch()
        assert v.patch_paths(kept) == {"src/a.ts"}
        assert sorted(dropped) == ["lib/b.ts", "package.json", "src/a.test.ts"]

    def test_verification_files_carry_only_the_replayed_patch(self) -> None:
        files = v.verification_files(_inputs(patch="diff --git a/README.md b/README.md\n+++ b/README.md\n"))
        assert "/tmp/nemo_gym_patch.diff" not in files and "nemo_gym_patch.diff" not in files["/tmp/nemo_gym_eval.sh"]


class TestEvalScript:
    def test_order_test_patch_then_setup_then_tests(self) -> None:
        script = v.build_eval_script(_inputs(setup_commands=[["npm", "run", "gen"]], setup_timeout_s=60))
        assert script.index("nemo_gym_test_patch.diff") < script.index("npm run gen") < script.index("vitest.mjs")
        assert "timeout -k 10 60 npm run gen" in script

    def test_placeholders_expand(self) -> None:
        assert v.expand_test_argv(["x", "--out={results}", "{specs}"], ["a.ts", "b.ts"]) == [
            "x",
            f"--out={v.RESULTS_PATH}",
            "a.ts",
            "b.ts",
        ]


class TestParseEvalOutput:
    def test_resolved_and_unresolved_runs_are_completed(self) -> None:
        ok = v.parse_eval_output(_script_output(0, _vitest_report("passed")), _inputs())
        assert ok.completed and ok.resolved and ok.test_output.strip() == "runner log"
        bad = v.parse_eval_output(_script_output(1, _vitest_report("failed")), _inputs())
        assert bad.completed and not bad.resolved

    def test_unapplied_test_patch_is_inconclusive(self) -> None:
        result = v.parse_eval_output(v.TEST_PATCH_FAILED, _inputs())
        assert not result.completed and result.test_patch_failed

    def test_setup_broken_by_the_candidate_is_a_completed_zero(self) -> None:
        result = v.parse_eval_output(v.SETUP_FAILED, _inputs())
        assert result.completed and not result.resolved

    def test_a_script_that_never_reached_the_tests_is_inconclusive(self) -> None:
        assert v.parse_eval_output("killed", _inputs()).completed is False

    async def test_run_verification_writes_the_log(self, tmp_path: Path) -> None:
        sandbox = MagicMock()
        sandbox.exec = AsyncMock(
            return_value=SandboxExecResult(
                return_code=0, stdout=_script_output(0, _vitest_report("passed")), stderr=""
            )
        )
        result = await v.run_verification(sandbox, _inputs(), log_dir=tmp_path)
        assert result.resolved and (tmp_path / "test_output.log").exists()


class TestSeeding:
    async def test_setup_runs_before_the_baseline_commit(self) -> None:
        from resources_servers.swe_runner_report import app

        calls: list[str] = []
        sandbox = MagicMock()

        async def exec_(command: str, **_: object) -> SandboxExecResult:
            calls.append(command)
            return SandboxExecResult(return_code=1 if "rev-parse --git-dir" in command else 0, stdout="", stderr="")

        sandbox.exec = exec_
        server = app.SweRunnerReportResourcesServer(
            config=app.SweRunnerReportResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="swe_runner_report",
                sandbox_provider="docker",
                sandbox_config={},
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        row = json.loads((Path(app.__file__).parent / "data/example.jsonl").read_text().splitlines()[0])
        body = app.SweRunnerReportInstanceRequest.model_validate(row | {"setup_commands": [["npm", "run", "gen"]]})
        token = app._SEEDING.set(body)
        try:
            await server._ensure_git_repo(sandbox, body.workdir)
        finally:
            app._SEEDING.reset(token)
        setup = next(i for i, c in enumerate(calls) if "npm run gen" in c)
        commit = next(i for i, c in enumerate(calls) if "git commit" in c)
        assert setup < commit
