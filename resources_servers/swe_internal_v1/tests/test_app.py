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
"""Tests for the swe_internal_v1 resources server: eval-script construction, the files staged
into the verification sandbox, grading of parsing_script.py's output, response shape, anti-cheat
wiring and the multi-worker entrypoint."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from resources_servers.swe_internal_v1.verification import (
    OUTPUT_FILE,
    PARSING_SCRIPT_FILE,
    PATCH_APPLIED_MARK,
    PATCH_FILE,
    RESULT_FILE_BEGIN,
    RESULT_FILE_END,
    RUN_SCRIPT_FILE,
    TEST_OUTPUT_BEGIN,
    TEST_OUTPUT_END,
    TEST_PATCH_FAILED,
    TEST_PATCH_FILE,
    WORKDIR_MISSING_EXIT,
    VerificationInputs,
    build_eval_script,
    drop_test_patch_files,
    grade,
    parse_verification_output,
    run_verification,
    verification_files,
)


SERVER_DIR = Path(__file__).resolve().parent.parent


def _inputs(**overrides) -> VerificationInputs:
    base = dict(
        instance_id="instance_org__repo-abc",
        workdir="/app",
        base_commit="40b9faa17dcb6b111db365a6f7a3b3b0ddfaecb7",  # pragma: allowlist secret  (a git sha, not a credential)
        patch="diff --git a/src/a.py b/src/a.py\n",
        run_script="#!/bin/bash\necho run $@\n",
        parsing_script="print('parse')\n",
        test_files=["tests/test_a.py", "/app/tests/test_a.py"],
        test_patch="diff --git a/tests/test_a.py b/tests/test_a.py\n",
        test_patch_checkout_cmd="git checkout f2f1b7 -- tests/test_a.py",
        env_exports=['export PYTEST_ADDOPTS="--tb=short"', "UV_HTTP_TIMEOUT=60"],
        fail_to_pass=["tests/test_a.py | test_new"],
        pass_to_pass=["tests/test_a.py | test_old"],
    )
    base.update(overrides)
    return VerificationInputs(**base)


def _results(**statuses: str) -> str:
    return json.dumps({"tests": [{"name": name, "status": status} for name, status in statuses.items()]})


def _output(applied: str = "1", changed: str = "1", body: str = "ran\n", results: str | None = None) -> str:
    if results is None:
        results = _results(**{"tests/test_a.py | test_new": "PASSED", "tests/test_a.py | test_old": "PASSED"})
    return (
        f"{PATCH_APPLIED_MARK} {applied} {changed}\n{TEST_OUTPUT_BEGIN}\n{body}{TEST_OUTPUT_END}\n"
        f"{RESULT_FILE_BEGIN}\n{results}\n{RESULT_FILE_END}\n"
    )


class TestBuildEvalScript:
    def test_follows_the_vendor_harness_order(self) -> None:
        script = build_eval_script(_inputs())
        assert f"cd /app || exit {WORKDIR_MISSING_EXIT}" in script
        i_env = script.index('export PYTEST_ADDOPTS="--tb=short"')
        i_reset = script.index("git reset -q --hard 40b9faa17dcb6b111db365a6f7a3b3b0ddfaecb7")
        i_apply = script.index(f"git apply --ignore-space-change --ignore-whitespace --reject -v {PATCH_FILE}")
        i_tests = script.index("git checkout f2f1b7 -- tests/test_a.py")
        i_run = script.index(f"bash {RUN_SCRIPT_FILE} tests/test_a.py,/app/tests/test_a.py")
        i_parse = script.index(PARSING_SCRIPT_FILE)
        assert i_env < i_reset < i_apply < i_tests < i_run < i_parse
        assert "export UV_HTTP_TIMEOUT=60" in script  # bare KEY=VALUE lines get the export prefix

    def test_hidden_tests_fall_back_to_the_test_patch(self) -> None:
        script = build_eval_script(_inputs())
        assert "git checkout f2f1b7 -- tests/test_a.py ) >/tmp/nemo_gym_test_checkout.log 2>&1 || git apply" in script
        assert TEST_PATCH_FILE in script and TEST_PATCH_FAILED in script

    def test_without_checkout_only_the_test_patch_is_applied(self) -> None:
        script = build_eval_script(_inputs(test_patch_checkout_cmd=""))
        assert "git checkout" not in script.split(PATCH_APPLIED_MARK)[1]
        assert TEST_PATCH_FILE in script

    def test_empty_patch_skips_the_apply_and_still_counts_as_applied(self) -> None:
        script = build_eval_script(_inputs(patch="  \n"))
        assert PATCH_FILE not in script
        assert "patch_applied=1" in script and "tree_changed=1" in script

    def test_parser_runs_on_stdout_stderr_and_writes_the_result_file(self) -> None:
        script = build_eval_script(_inputs())
        assert f"{PARSING_SCRIPT_FILE} /tmp/nemo_gym_stdout.log /tmp/nemo_gym_stderr.log {OUTPUT_FILE}" in script
        assert "command -v python3 || command -v python" in script
        assert script.index(TEST_OUTPUT_END) < script.index(RESULT_FILE_BEGIN) < script.index(f"cat {OUTPUT_FILE}")

    def test_quotes_an_unusual_workdir(self) -> None:
        assert "cd '/work dir/repo'" in build_eval_script(_inputs(workdir="/work dir/repo"))


class TestVerificationFiles:
    def test_stages_scripts_and_patches(self) -> None:
        files = verification_files(_inputs())
        assert set(files) == {
            "/tmp/nemo_gym_eval.sh",
            RUN_SCRIPT_FILE,
            PARSING_SCRIPT_FILE,
            PATCH_FILE,
            TEST_PATCH_FILE,
        }
        assert files[RUN_SCRIPT_FILE].startswith("#!/bin/bash") and files[PARSING_SCRIPT_FILE] == "print('parse')\n"

    def test_omits_blank_patch_and_blank_test_patch(self) -> None:
        files = verification_files(_inputs(patch="   ", test_patch=""))
        assert PATCH_FILE not in files and TEST_PATCH_FILE not in files


class TestGrade:
    def test_requires_every_f2p_and_p2p_passed(self) -> None:
        statuses = {"a | x": "PASSED", "a | y": "PASSED", "a | z": "FAILED"}
        assert grade(statuses, ["a | x"], ["a | y"])["resolved"] is True
        report = grade(statuses, ["a | x"], ["a | z"])
        assert report["resolved"] is False and report["PASS_TO_PASS"]["failure"] == ["a | z"]

    def test_absent_test_is_not_passing_and_empty_requirements_never_resolve(self) -> None:
        assert grade({"a | x": "PASSED"}, ["a | missing"], [])["resolved"] is False
        assert grade({"a | x": "PASSED"}, [], [])["resolved"] is False
        assert grade({}, ["a | x"], [])["tests_observed"] == 0


class TestParseVerificationOutput:
    def test_all_required_passed_resolves(self) -> None:
        result = parse_verification_output(_output(), 0, _inputs())
        assert result.completed and result.resolved and result.patch_applied and result.error is None
        assert result.test_results["tests_observed"] == 2 and result.test_results["test_exit_code"] == 0

    def test_failed_required_test_is_a_completed_failure(self) -> None:
        results = _results(**{"tests/test_a.py | test_new": "FAILED", "tests/test_a.py | test_old": "PASSED"})
        result = parse_verification_output(_output(results=results), 1, _inputs())
        assert result.completed and not result.resolved
        assert result.test_results["FAIL_TO_PASS"]["failure"] == ["tests/test_a.py | test_new"]

    def test_green_suite_with_an_unapplied_patch_is_not_resolved(self) -> None:
        result = parse_verification_output(_output(applied="0", changed="0"), 0, _inputs())
        assert result.completed and not result.resolved and not result.patch_applied
        assert result.error == "candidate patch did not apply"

    def test_partial_apply_keeps_the_verdict(self) -> None:
        result = parse_verification_output(_output(applied="0", changed="1"), 0, _inputs())
        assert result.completed and result.resolved and result.patch_applied
        assert result.error == "candidate patch applied only partially"
        assert result.test_results["patch_partially_applied"] is True

    def test_missing_result_file_is_incomplete(self) -> None:
        result = parse_verification_output(_output(results=""), 1, _inputs())
        assert not result.completed and not result.resolved
        assert "no result file" in result.error

    def test_hidden_tests_not_installed_is_incomplete(self) -> None:
        result = parse_verification_output(_output(body=f"{TEST_PATCH_FAILED}\nerror: patch failed\n"), 1, _inputs())
        assert not result.completed and result.test_patch_failed
        assert "did not install" in result.error

    def test_missing_workdir_exit_code(self) -> None:
        result = parse_verification_output("", WORKDIR_MISSING_EXIT, _inputs())
        assert not result.completed and "not present in the image" in result.error

    def test_output_without_markers_is_incomplete(self) -> None:
        result = parse_verification_output("bash: something exploded\n", 2, _inputs())
        assert not result.completed and result.test_results is None
        assert result.error == "eval script did not reach the patch step"

    def test_golden_mode_with_no_patch_text_can_still_resolve(self) -> None:
        # Empty patch (the dataset row has none): the script marks applied=1/changed=1 itself.
        result = parse_verification_output(_output(), 0, _inputs(patch=""))
        assert result.resolved


class _FakeSandbox:
    def __init__(self, stdout: str, return_code: int = 0) -> None:
        self._result = SimpleNamespace(stdout=stdout, stderr="", return_code=return_code)
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_s=None):
        self.commands.append(command)
        return self._result


class TestRunVerification:
    @pytest.mark.asyncio
    async def test_runs_the_eval_script_and_persists_the_log(self, tmp_path) -> None:
        sandbox = _FakeSandbox(_output())
        result = await run_verification(sandbox, _inputs(), timeout_s=10, log_dir=tmp_path / "logs" / "x")
        assert sandbox.commands == ["bash /tmp/nemo_gym_eval.sh"]
        assert result.resolved
        assert (tmp_path / "logs" / "x" / "test_output.log").read_text().startswith(PATCH_APPLIED_MARK)


class TestDropTestPatchFiles:
    def test_model_edits_to_hidden_test_files_are_dropped(self) -> None:
        patch = (
            "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/tests/t.py b/tests/t.py\n--- a/tests/t.py\n+++ b/tests/t.py\n@@ -1 +1 @@\n-x\n+mine\n"
        )
        test_patch = (
            "diff --git a/tests/t.py b/tests/t.py\n--- a/tests/t.py\n+++ b/tests/t.py\n@@ -1 +1 @@\n-x\n+hidden\n"
        )
        kept = drop_test_patch_files(patch, test_patch)
        assert "a/src/a.py" in kept and "tests/t.py" not in kept


class TestVerifyResponseShape:
    """BaseVerifyResponse extends BaseVerifyRequest, so the request fields are REQUIRED on the way
    out; spreading the request body is what satisfies them."""

    @staticmethod
    def _body() -> dict:
        return {
            "instance_id": "instance_org__repo-abc",
            "delivery": "internal-v1",
            "workdir": "/app",
            "image_ref": "942195279341.dkr.ecr.us-east-2.amazonaws.com/sweap-pro:instance_org__repo-abc",
            "language": "python",
            "run_script": "#!/bin/bash\n",
            "parsing_script": "",
            "FAIL_TO_PASS": ["a | x"],
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
        from resources_servers.swe_internal_v1.app import SweInternalV1VerifyResponse

        response = SweInternalV1VerifyResponse.model_validate(
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
                "test_framework": "",
            }
        )
        assert response.instance_id == "instance_org__repo-abc"
        assert response.patch_source == "none" and response.model_patch is None

    def test_building_from_scratch_without_the_request_fields_fails(self) -> None:
        from resources_servers.swe_internal_v1.app import SweInternalV1VerifyResponse

        with pytest.raises(Exception):
            SweInternalV1VerifyResponse.model_validate({"reward": 1.0, "evaluation_completed": True, "resolved": True})

    def test_instance_request_requires_the_scripts(self) -> None:
        from resources_servers.swe_internal_v1.app import SweInternalV1InstanceRequest

        body = self._body()
        for key in ("run_script", "parsing_script", "image_ref"):
            with pytest.raises(Exception):
                SweInternalV1InstanceRequest.model_validate({k: v for k, v in body.items() if k != key})

    def test_task_data_declares_every_prepared_key(self) -> None:
        from resources_servers.swe_internal_v1.task_data import TaskData

        declared = set(TaskData.model_fields)
        for key in (
            "run_script",
            "parsing_script",
            "test_files",
            "test_patch_checkout_cmd",
            "env_exports",
            "solution_commit",
            "nydus_ref",
            "original_issue_url",
            "issue_categories",
            "evaluation_time",
        ):
            assert key in declared


class TestServerWiring:
    @staticmethod
    def _source() -> str:
        return (SERVER_DIR / "app.py").read_text()

    def test_seed_session_restores_blobs_scrubs_then_seeds_a_committer_identity(self) -> None:
        source = self._source()
        assert source.index("await self._ensure_git_repo(") < source.index("await self._restore_missing_blobs(")
        assert source.index("await self._restore_missing_blobs(") < source.index("apply_anti_cheat_setup(sandbox")
        assert source.index("apply_anti_cheat_setup(sandbox") < source.index("await prepare_git_for_commits(")

    def test_config_defaults(self) -> None:
        from resources_servers.swe_internal_v1.app import SweInternalV1ResourcesServerConfig

        fields = SweInternalV1ResourcesServerConfig.model_fields
        assert fields["apply_anti_cheating"].default is True
        assert fields["patch_capture_mode"].default == "worktree"
        assert fields["evaluation_timeout"].default == 1800

    def test_yaml_enables_anti_cheat_and_names_both_servers(self) -> None:
        config = yaml.safe_load((SERVER_DIR / "configs" / "swe_internal_v1.yaml").read_text())
        base = config["swe_internal_v1_resources_server"]["resources_servers"]["swe_internal_v1"]
        golden = config["swe_internal_v1_golden_patch_resources_server"]["resources_servers"]["swe_internal_v1"]
        assert base["apply_anti_cheating"] is True and base["is_verifying_golden_patch"] is False
        assert golden["is_verifying_golden_patch"] is True
        assert base["evaluation_timeout"] == golden["evaluation_timeout"] == 1800

    def test_exposes_a_module_level_app_for_forked_workers(self) -> None:
        source = self._source()
        assert "is_nemo_gym_fastapi_entrypoint(__file__)" in source
        assert "app = SweInternalV1ResourcesServer.run_webserver()" in source
