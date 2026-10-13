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
"""Tests for the swemer_oml resources server: eval-script construction, the files staged into the
verification sandbox, verdict parsing of the package grader's output, response shape, anti-cheat
wiring and the multi-worker entrypoint."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from resources_servers.swemer_oml.verification import (
    NO_REWARD_TEXT,
    PATCH_APPLIED_MARK,
    REWARD_BEGIN,
    REWARD_END,
    TEST_OUTPUT_BEGIN,
    TEST_OUTPUT_END,
    TEST_PATCH_FAILED_TEXT,
    TESTS_DIR,
    WORKDIR_MISSING_EXIT,
    VerificationInputs,
    build_eval_script,
    drop_test_patch_files,
    parse_verification_output,
    run_verification,
    verification_files,
)


SERVER_DIR = Path(__file__).resolve().parent.parent


def _inputs(**overrides) -> VerificationInputs:
    base = dict(
        instance_id="0xd4d-iced-65",
        workdir="/workspace/repo",
        patch="diff --git a/a b/a\n",
        test_sh="#!/bin/bash\necho hi\n",
        config_json='{"FAIL_TO_PASS": ["T::a"], "PASS_TO_PASS": [], "framework": "dotnet"}',
        grade_py="print('grader')\n",
        test_patch="diff --git a/b/test_a.py b/b/test_a.py\n",
        test_framework="dotnet",
        fail_to_pass=["T::a"],
        pass_to_pass=[],
    )
    base.update(overrides)
    return VerificationInputs(**base)


def _output(applied: str = "1", reward: str = "1", body: str = "tests ran\n", changed: str = "1") -> str:
    return (
        f"{PATCH_APPLIED_MARK} {applied} {changed}\n{TEST_OUTPUT_BEGIN}\n{body}{TEST_OUTPUT_END}\n"
        f"{REWARD_BEGIN}\n{reward}\n{REWARD_END}\n"
    )


class TestBuildEvalScript:
    def test_applies_patch_before_running_the_package_grader(self) -> None:
        script = build_eval_script(_inputs())
        assert script.index("apply_patch /workspace/repo /tmp/nemo_gym_patch.diff") < script.index(
            f"bash {TESTS_DIR}/test.sh"
        )
        assert f"cd /workspace/repo || exit {WORKDIR_MISSING_EXIT}" in script
        assert "cat /logs/verifier/reward.txt" in script
        assert script.index("__before=$(git -C /workspace/repo status --porcelain") < script.index(
            "apply_patch /workspace/repo"
        )
        assert '[ "$__before" != "$__after" ] && tree_changed=1' in script

    def test_uses_the_packages_own_apply_chain(self) -> None:
        script = build_eval_script(_inputs())
        assert 'patch -p1 -d "$repo" $primary -i "$pf"' in script
        assert "git apply --3way --ignore-whitespace" in script
        assert 'patch -p0 -d "$repo" --fuzz=5' in script

    def test_golden_mode_passes_the_packages_fuzz_primary(self) -> None:
        assert "apply_patch /workspace/repo /tmp/nemo_gym_patch.diff --fuzz=5;" in build_eval_script(
            _inputs(apply_primary="--fuzz=5")
        )
        assert "apply_patch /workspace/repo /tmp/nemo_gym_patch.diff '';" in build_eval_script(_inputs())

    def test_empty_patch_skips_the_apply_and_still_counts_as_applied(self) -> None:
        script = build_eval_script(_inputs(patch="  \n"))
        assert "apply_patch /workspace/repo" not in script
        assert "patch_applied=1" in script

    def test_markers_bracket_test_sh_and_the_reward(self) -> None:
        script = build_eval_script(_inputs())
        for marker in (PATCH_APPLIED_MARK, TEST_OUTPUT_BEGIN, TEST_OUTPUT_END, REWARD_BEGIN, REWARD_END):
            assert marker in script
        assert script.index(TEST_OUTPUT_END) < script.index(REWARD_BEGIN)

    def test_quotes_an_unusual_workdir(self) -> None:
        script = build_eval_script(_inputs(workdir="/work dir/repo"))
        assert "cd '/work dir/repo'" in script


class TestVerificationFiles:
    def test_stages_the_tests_folder_where_test_sh_expects_it(self) -> None:
        files = verification_files(_inputs())
        assert {
            "/tmp/nemo_gym_eval.sh",
            f"{TESTS_DIR}/test.sh",
            f"{TESTS_DIR}/config.json",
            f"{TESTS_DIR}/grade.py",
            f"{TESTS_DIR}/test.patch",
            "/tmp/nemo_gym_patch.diff",
        } <= set(files)
        assert "/root/.m2/settings.xml" in files  # Maven Central mirror, as in swemer_v1
        assert files[f"{TESTS_DIR}/grade.py"] == "print('grader')\n"

    def test_omits_blank_patch_and_blank_test_patch(self) -> None:
        files = verification_files(_inputs(patch="   ", test_patch=""))
        assert "/tmp/nemo_gym_patch.diff" not in files
        assert f"{TESTS_DIR}/test.patch" not in files


class TestParseVerificationOutput:
    def test_reward_one_with_applied_patch_resolves(self) -> None:
        result = parse_verification_output(_output(), 0, _inputs())
        assert result.completed and result.resolved and result.patch_applied
        assert result.error is None
        assert result.test_results["reward"] == 1 and result.test_results["framework"] == "dotnet"

    def test_reward_zero_is_a_completed_failure(self) -> None:
        result = parse_verification_output(_output(reward="0"), 1, _inputs())
        assert result.completed and not result.resolved
        assert result.test_results["test_exit_code"] == 1

    def test_reward_one_but_patch_not_applied_is_not_resolved(self) -> None:
        # A patch that changed nothing cannot claim the credit for a green suite.
        result = parse_verification_output(_output(applied="0", reward="1", changed="0"), 0, _inputs())
        assert result.completed and not result.resolved and not result.patch_applied
        assert result.error == "candidate patch did not apply"

    def test_partial_apply_keeps_the_graders_verdict(self) -> None:
        # `patch` applies the code hunks and fails on a drifted lockfile: the chain exits non-zero
        # but the tree changed, and the package grader's 1 stands.
        result = parse_verification_output(_output(applied="0", reward="1", changed="1"), 0, _inputs())
        assert result.completed and result.resolved and result.patch_applied
        assert result.error == "candidate patch applied only partially"
        assert result.test_results["patch_partially_applied"] is True
        assert parse_verification_output(_output(applied="0", reward="0", changed="1"), 1, _inputs()).resolved is False

    def test_missing_reward_file_is_incomplete(self) -> None:
        result = parse_verification_output(_output(reward=""), 1, _inputs())
        assert not result.completed and not result.resolved
        assert "no reward" in result.error

    def test_hidden_test_patch_failure_is_incomplete(self) -> None:
        result = parse_verification_output(
            _output(reward="0", body=f"{TEST_PATCH_FAILED_TEXT} -- hidden tests not installed\n"), 1, _inputs()
        )
        assert not result.completed and result.test_patch_failed
        assert result.error == "held-out test patch did not apply"

    def test_grader_fail_closed_is_incomplete(self) -> None:
        result = parse_verification_output(
            _output(reward="0", body=f"{NO_REWARD_TEXT}; failing closed\n"), 1, _inputs()
        )
        assert not result.completed and "no reward" in result.error

    def test_missing_workdir_exit_code(self) -> None:
        result = parse_verification_output("", WORKDIR_MISSING_EXIT, _inputs())
        assert not result.completed and "not present in the image" in result.error

    def test_output_without_markers_is_incomplete(self) -> None:
        result = parse_verification_output("bash: something exploded\n", 2, _inputs())
        assert not result.completed and result.test_results is None
        assert result.error == "eval script did not reach the patch step"


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
            "diff --git a/src/a.rs b/src/a.rs\n--- a/src/a.rs\n+++ b/src/a.rs\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/tests/t.rs b/tests/t.rs\n--- a/tests/t.rs\n+++ b/tests/t.rs\n@@ -1 +1 @@\n-x\n+mine\n"
        )
        test_patch = (
            "diff --git a/tests/t.rs b/tests/t.rs\n--- a/tests/t.rs\n+++ b/tests/t.rs\n@@ -1 +1 @@\n-x\n+hidden\n"
        )
        kept = drop_test_patch_files(patch, test_patch)
        assert "a/src/a.rs" in kept and "tests/t.rs" not in kept


class TestVerifyResponseShape:
    """BaseVerifyResponse extends BaseVerifyRequest, so the request fields are REQUIRED on the way
    out; spreading the request body is what satisfies them (see swemer_v2's tests for the history)."""

    @staticmethod
    def _body() -> dict:
        return {
            "instance_id": "0xd4d-iced-65",
            "delivery": "tasks",
            "workdir": "/workspace/repo",
            "image_ref": "942195279341.dkr.ecr.us-east-2.amazonaws.com/ext-oml-swe-bench:0xd4d-iced-65",
            "language": "rust",
            "test_sh": "#!/bin/bash\n",
            "config_json": "{}",
            "grade_py": "",
            "test_framework": "dotnet",
            "FAIL_TO_PASS": ["T::a"],
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
        from resources_servers.swemer_oml.app import SwemerOmlVerifyResponse

        response = SwemerOmlVerifyResponse.model_validate(
            self._body()
            | {
                "reward": 1.0,
                "evaluation_completed": True,
                "resolved": True,
                "patch_applied": True,
                "test_results": {"reward": 1},
                "test_output": "",
                "error": None,
                "eval_sandbox_start_time_taken": 0.1,
                "patch_verification_time_taken": 0.2,
            }
        )
        assert response.instance_id == "0xd4d-iced-65"
        assert response.patch_source == "none" and response.model_patch is None

    def test_building_from_scratch_without_the_request_fields_fails(self) -> None:
        from resources_servers.swemer_oml.app import SwemerOmlVerifyResponse

        with pytest.raises(Exception):
            SwemerOmlVerifyResponse.model_validate({"reward": 1.0, "evaluation_completed": True, "resolved": True})

    def test_instance_request_requires_the_grader_files(self) -> None:
        from resources_servers.swemer_oml.app import SwemerOmlInstanceRequest

        body = self._body()
        for key in ("test_sh", "config_json", "grade_py"):
            with pytest.raises(Exception):
                SwemerOmlInstanceRequest.model_validate({k: v for k, v in body.items() if k != key})


class TestServerWiring:
    @staticmethod
    def _source() -> str:
        return (SERVER_DIR / "app.py").read_text()

    def test_seed_session_scrubs_then_seeds_a_committer_identity(self) -> None:
        source = self._source()
        assert source.index("await self._ensure_git_repo(") < source.index("apply_anti_cheat_setup(sandbox")
        # Blob restore must run before the scrub: its `git reset --hard` deletes files it cannot re-read.
        assert source.index("await self._restore_missing_blobs(") < source.index("apply_anti_cheat_setup(sandbox")
        assert source.index("apply_anti_cheat_setup(sandbox") < source.index("await prepare_git_for_commits(")

    def test_config_defaults(self) -> None:
        from resources_servers.swemer_oml.app import SwemerOmlResourcesServerConfig

        fields = SwemerOmlResourcesServerConfig.model_fields
        assert fields["apply_anti_cheating"].default is True
        assert fields["patch_capture_mode"].default == "worktree"
        assert fields["evaluation_timeout"].default == 3000

    def test_yaml_enables_anti_cheat_and_names_both_servers(self) -> None:
        config = yaml.safe_load((SERVER_DIR / "configs" / "swemer_oml.yaml").read_text())
        base = config["swemer_oml_resources_server"]["resources_servers"]["swemer_oml"]
        golden = config["swemer_oml_golden_patch_resources_server"]["resources_servers"]["swemer_oml"]
        assert base["apply_anti_cheating"] is True and base["is_verifying_golden_patch"] is False
        assert golden["is_verifying_golden_patch"] is True
        assert base["evaluation_timeout"] == golden["evaluation_timeout"] == 3000

    def test_exposes_a_module_level_app_for_forked_workers(self) -> None:
        source = self._source()
        assert "is_nemo_gym_fastapi_entrypoint(__file__)" in source
        assert "app = SwemerOmlResourcesServer.run_webserver()" in source


class TestRestoreMissingBlobs:
    class _Sandbox:
        def __init__(self, probe_rc: int) -> None:
            self.probe_rc = probe_rc
            self.commands: list[str] = []

        async def exec(self, command: str, timeout_s=None, **kwargs):
            self.commands.append(command)
            rc = self.probe_rc if "cat-file -e" in command else 0
            return SimpleNamespace(return_code=rc, stdout="", stderr="")

    @pytest.mark.asyncio
    async def test_intact_repo_is_left_alone(self) -> None:
        from resources_servers.swemer_oml.app import SwemerOmlResourcesServer

        sb = self._Sandbox(probe_rc=0)
        await SwemerOmlResourcesServer._restore_missing_blobs(None, sb, "/workspace/repo")
        assert len(sb.commands) == 1 and "cat-file -e" in sb.commands[0]

    @pytest.mark.asyncio
    async def test_blob_stripped_repo_is_re_added_and_committed(self) -> None:
        from resources_servers.swemer_oml.app import SwemerOmlResourcesServer

        sb = self._Sandbox(probe_rc=3)
        await SwemerOmlResourcesServer._restore_missing_blobs(None, sb, "/workspace/repo")
        assert len(sb.commands) == 2
        cmd = sb.commands[1]
        assert "git ls-files -z --stage" in cmd and "git hash-object -w --" in cmd
        assert "100644|100755" in cmd  # gitlinks (160000) and symlinks (120000) are skipped
        assert "git add -u" in cmd and "git add -A" not in cmd and "commit -q" in cmd
