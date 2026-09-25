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

Grading itself (output-flag injection, framework parsing) is delegated to
``responses_api_agents.swe_agents.swe_bench_ext``, which owns its own test coverage; this file
tests the stable contract swemer_v2's app.py depends on (build_eval_script, verification_files,
grade, response shape, multi-worker entrypoint), not swe_bench_ext's internals.

Two classes here pin bugs that were actually hit building the sibling SWE resources servers
(scale_swe, swe_rebench): a verify response missing the echoed request fields, and a
multi-worker entrypoint with no module-level `app`. Both looked fine locally and both only
failed once real traffic hit them, so the same tests are written here from the start.
"""

from types import SimpleNamespace

import pytest

from resources_servers.swemer_v2.verification import (
    RESULT_FILE_BEGIN,
    RESULT_FILE_END,
    SUPPORTED_FRAMEWORKS,
    TEST_OUTPUT_BEGIN,
    TEST_OUTPUT_END,
    TEST_PATCH_FAILED,
    VerificationInputs,
    _slice,
    build_eval_script,
    grade,
    run_verification,
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


class TestBuildEvalScript:
    def test_applies_patch_then_test_patch(self) -> None:
        script = build_eval_script(_inputs())
        patch_idx = script.index("nemo_gym_patch.diff")
        test_patch_idx = script.index("nemo_gym_test_patch.diff")
        assert patch_idx < test_patch_idx

    def test_omits_test_patch_apply_when_empty(self) -> None:
        script = build_eval_script(_inputs(test_patch=""))
        assert "nemo_gym_test_patch.diff" not in script

    def test_flags_a_test_patch_that_fails_to_apply(self) -> None:
        script = build_eval_script(_inputs())
        assert f"grep -q '^error: ' /tmp/nemo_gym_test_patch.log && echo {TEST_PATCH_FAILED}" in script

    def test_markers_bracket_the_test_command(self) -> None:
        script = build_eval_script(_inputs(test_command="pytest tests/test_x.py -v"))
        assert TEST_OUTPUT_BEGIN in script
        assert TEST_OUTPUT_END in script
        assert "pytest tests/test_x.py -v --junitxml=" in script

    def test_result_file_markers_present_for_frameworks_that_write_one(self) -> None:
        # pytest's swe_bench_ext config writes JUnit XML to a file rather than stdout.
        script = build_eval_script(_inputs(test_framework="pytest"))
        assert RESULT_FILE_BEGIN in script
        assert RESULT_FILE_END in script
        assert "cat" in script

    def test_go_has_no_result_file_to_cat(self) -> None:
        # go test -json writes to stdout directly; there is nothing to cat back. swe_bench_ext
        # appends the flag at the end of the command rather than inserting it after "go test".
        script = build_eval_script(_inputs(test_framework="go", test_command="go test ./... -count=1"))
        assert "go test ./... -count=1 -json" in script

    def test_slice_extracts_between_markers(self) -> None:
        log = f"noise before\n{TEST_OUTPUT_BEGIN}\nPASSED tests/test_x.py::test_one\n{TEST_OUTPUT_END}\nnoise after"
        sliced = _slice(log, TEST_OUTPUT_BEGIN, TEST_OUTPUT_END)
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


class _FakeSandbox:
    def __init__(self, stdout: str) -> None:
        self._result = SimpleNamespace(stdout=stdout, stderr="", return_code=0)

    async def exec(self, command: str, timeout_s=None):
        return self._result


class TestRunVerification:
    @pytest.mark.asyncio
    async def test_a_failed_test_patch_scores_zero(self) -> None:
        result = await run_verification(
            sandbox=_FakeSandbox(
                f"{TEST_PATCH_FAILED}\n"
                + (
                    f"{TEST_OUTPUT_BEGIN}\n"
                    '{"Action": "pass", "Test": "TestOne", "Package": "example.com/pkg"}\n'
                    f"{TEST_OUTPUT_END}\n{RESULT_FILE_BEGIN}\n{RESULT_FILE_END}\n"
                )
            ),
            inputs=_inputs(
                test_framework="go",
                test_command="go test ./...",
                fail_to_pass=["example.com/pkg::TestOne"],
                pass_to_pass=[],
            ),
        )
        assert result.completed is True
        assert result.resolved is False
        assert result.test_patch_failed is True


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

    def test_normalized_id_matches_dataset_dotted_convention(self) -> None:
        """The dataset's ids are dotted (e.g. tests.pkg.mod::test_x); swe_bench_ext's parsers key
        their output by real node ids (tests/pkg/mod.py::test_x). Exact match misses this, so the
        normalize_test_id fallback in grade() is what makes the two sides agree -- this is the
        core behavior the swap to swe_bench_ext depends on.
        """
        statuses = {"tests/pkg/test_mod.py::TestFoo::test_x": "PASSED"}
        report = grade(
            statuses, fail_to_pass=["tests.pkg.test_mod.TestFoo::test_x"], pass_to_pass=[], test_framework="pytest"
        )
        assert report["resolved"] is True

    def test_exact_match_still_preferred_over_normalized(self) -> None:
        """Normalization must be a fallback, not the primary path -- an id present in BOTH its
        exact form and only distinguishable after normalization should resolve via the exact
        match, matching whatever status the exact key carries even if a same-normalized-form key
        also exists with a different status.
        """
        statuses = {"a::b": "PASSED", "a.b": "FAILED"}
        report = grade(statuses, fail_to_pass=["a::b"], pass_to_pass=[], test_framework="")
        assert report["resolved"] is True


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


class TestAntiCheating:
    """seed_session must scrub the sandbox's git history before an agent gets control of it. See
    resources_servers/swebench/anti_cheat.py."""

    @staticmethod
    def _source() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / "app.py").read_text()

    def test_seed_session_calls_the_shared_anti_cheat_helper(self) -> None:
        source = self._source()
        assert "from resources_servers.swebench.anti_cheat import apply_anti_cheat_setup" in source
        assert "apply_anti_cheat_setup(" in source

    def test_config_enables_it_by_default(self) -> None:
        from resources_servers.swemer_v2.app import SwemerV2ResourcesServerConfig

        assert SwemerV2ResourcesServerConfig.model_fields["apply_anti_cheating"].default is True

    def test_seed_session_ensures_a_git_repo_before_scrubbing(self) -> None:
        source = self._source()
        assert "_ensure_git_repo(" in source
        ensure_idx = source.index("await self._ensure_git_repo(")
        anti_cheat_idx = source.index("apply_anti_cheat_setup(sandbox")
        assert ensure_idx < anti_cheat_idx

    def test_agent_facing_config_turns_it_on(self) -> None:
        from pathlib import Path

        import yaml

        config = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "swemer_v2.yaml").read_text())
        assert config["swemer_v2_resources_server"]["resources_servers"]["swemer_v2"]["apply_anti_cheating"] is True


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
