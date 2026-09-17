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
"""Tests for the DeNovoSWE resources server's grading and script construction.

Grading itself (the per-file pytest collect-then-intersect-then-run sweep) is delegated to
``_denovoswe_eval.py``, a verbatim local copy of the already-validated in-container evaluator from
the SIF-based swe_agents harness; this file tests the stable contract app.py depends on
(seed_prep_script, build_eval_script, verification_files, run_verification's report.json parsing,
response shape, multi-worker entrypoint) -- and the golden/non-golden branching that is unique to
this server (no other SWE resources server here skips patch application on the golden path).

Two classes here pin bugs that were actually hit building the sibling SWE resources servers
(scale_swe, swe_rebench): a verify response missing the echoed request fields, and a
multi-worker entrypoint with no module-level `app`. Both looked fine locally and both only
failed once real traffic hit them, so the same tests are written here from the start.
"""

from types import SimpleNamespace

import pytest

from resources_servers.denovo_swe.verification import (
    CLEAN_SH_PATH,
    DOCUMENT_PATH,
    EVAL_PY_PATH,
    META_PATH,
    REPORT_BEGIN,
    REPORT_END,
    TEST_BINARY_PATH,
    TEST_PATCH_PATH,
    VerificationInputs,
    _slice,
    build_eval_script,
    drop_patch_sections,
    patch_section_path,
    run_verification,
    seed_prep_script,
    verification_files,
)


def _inputs(**overrides) -> VerificationInputs:
    base = dict(
        instance_id="langchain-ai_langgraph-supervisor-py_pr184",
        workdir="/workspace/langgraph-supervisor-py",
        base_commit="1bf5acae5966bec5b174a903be038c1d271dccd7",
        patch="diff --git a/a.py b/a.py\n",
        test_patch="diff --git a/tests/test_a.py b/tests/test_a.py\n",
        document="# langgraph-supervisor-py\n\nA spec the agent must implement.\n",
        pypi_name="langgraph-supervisor",
        passed_ptp=["tests/test_a.py::test_one"],
        failed_ptp=[],
    )
    base.update(overrides)
    return VerificationInputs(**base)


class TestPatchSectionPath:
    def test_extracts_the_b_path_from_a_simple_diff(self) -> None:
        section = "diff --git a/src/foo.py b/src/foo.py\n--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-x\n+y\n"
        assert patch_section_path(section) == "src/foo.py"

    def test_a_new_file_uses_the_b_path(self) -> None:
        section = "diff --git a/src/new.py b/src/new.py\n--- /dev/null\n+++ b/src/new.py\n@@ -0,0 +1 @@\n+x\n"
        assert patch_section_path(section) == "src/new.py"


class TestDropPatchSections:
    def test_drops_sections_matching_given_paths(self) -> None:
        patch = (
            "diff --git a/keep.py b/keep.py\n--- a/keep.py\n+++ b/keep.py\n@@ -1 +1 @@\n-x\n+y\n"
            "diff --git a/drop.py b/drop.py\n--- /dev/null\n+++ b/drop.py\n@@ -0,0 +1 @@\n+z\n"
        )
        result = drop_patch_sections(patch, {"drop.py"})
        assert "keep.py" in result
        assert "drop.py" not in result


class TestSeedPrepScript:
    def test_wipes_before_reinjecting_the_document(self) -> None:
        script = seed_prep_script("/workspace/pkg")
        clean_idx = script.index(CLEAN_SH_PATH)
        readme_idx = script.index("README.md")
        assert clean_idx < readme_idx

    def test_targets_the_given_workdir(self) -> None:
        script = seed_prep_script("/workspace/pkg")
        assert "/workspace/pkg" in script


class TestBuildEvalScript:
    def test_golden_path_skips_the_wipe_and_patch_apply(self) -> None:
        script = build_eval_script(_inputs(), is_golden=True)
        assert CLEAN_SH_PATH not in script
        assert "nemo_gym_patch.diff" not in script

    def test_non_golden_path_wipes_and_reinjects_the_document(self) -> None:
        script = build_eval_script(_inputs(), is_golden=False)
        assert CLEAN_SH_PATH in script
        assert "README.md" in script

    def test_non_golden_path_applies_the_patch_when_present(self) -> None:
        script = build_eval_script(_inputs(patch="diff --git a/a.py b/a.py\n"), is_golden=False)
        assert "nemo_gym_patch.diff" in script

    def test_non_golden_path_omits_patch_apply_when_patch_is_empty(self) -> None:
        script = build_eval_script(_inputs(patch=""), is_golden=False)
        assert "nemo_gym_patch.diff" not in script

    def test_golden_path_never_applies_a_patch_even_if_one_is_present(self) -> None:
        # is_golden gates on the flag, not patch emptiness -- an agent that crashed/timed out
        # produces an empty patch too, and keying on emptiness would conflate the two.
        script = build_eval_script(_inputs(patch="diff --git a/a.py b/a.py\n"), is_golden=True)
        assert "nemo_gym_patch.diff" not in script

    def test_checks_out_the_base_commit(self) -> None:
        script = build_eval_script(_inputs(), is_golden=True)
        assert "1bf5acae5966bec5b174a903be038c1d271dccd7" in script

    def test_deletes_pre_existing_test_files(self) -> None:
        script = build_eval_script(_inputs(), is_golden=True)
        assert "-iname tests" in script
        assert "conftest.py" in script

    def test_applies_test_patch(self) -> None:
        script = build_eval_script(_inputs(), is_golden=True)
        assert TEST_PATCH_PATH in script

    def test_reinstalls_the_package_by_pypi_name(self) -> None:
        script = build_eval_script(_inputs(pypi_name="langgraph-supervisor"), is_golden=True)
        assert "langgraph-supervisor" in script
        assert "pip install -e ." in script

    def test_runs_the_eval_script_and_brackets_the_report(self) -> None:
        script = build_eval_script(_inputs(), is_golden=True)
        assert EVAL_PY_PATH in script
        assert REPORT_BEGIN in script
        assert REPORT_END in script

    def test_slice_extracts_between_markers(self) -> None:
        log = f'noise before\n{REPORT_BEGIN}\n{{"reward": "1"}}\n{REPORT_END}\nnoise after'
        sliced = _slice(log, REPORT_BEGIN, REPORT_END)
        assert "noise before" not in sliced
        assert "noise after" not in sliced
        assert '"reward": "1"' in sliced


class TestVerificationFiles:
    def test_always_ships_the_clean_and_eval_scripts_and_meta(self) -> None:
        files = verification_files(_inputs(), is_golden=True)
        assert CLEAN_SH_PATH in files
        assert EVAL_PY_PATH in files
        assert META_PATH in files
        assert DOCUMENT_PATH in files
        assert TEST_PATCH_PATH in files

    def test_ships_the_binary_archive_only_when_present(self) -> None:
        files = verification_files(_inputs(test_binary_archive_b64=""), is_golden=True)
        assert files[TEST_BINARY_PATH] == ""

    def test_golden_never_ships_a_patch_file(self) -> None:
        files = verification_files(_inputs(patch="diff --git a/a.py b/a.py\n"), is_golden=True)
        assert "/tmp/nemo_gym_patch.diff" not in files

    def test_non_golden_ships_the_patch_file_when_present(self) -> None:
        files = verification_files(_inputs(patch="diff --git a/a.py b/a.py\n"), is_golden=False)
        assert "/tmp/nemo_gym_patch.diff" in files

    def test_meta_carries_passed_ptp(self) -> None:
        import json

        files = verification_files(_inputs(passed_ptp=["a::b", "a::c"]), is_golden=True)
        meta = json.loads(files[META_PATH])
        assert meta["passed_ptp"] == ["a::b", "a::c"]


class _FakeSandbox:
    def __init__(self, stdout: str, return_code: int = 0) -> None:
        self._result = SimpleNamespace(stdout=stdout, stderr="", return_code=return_code)
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_s=None):
        self.commands.append(command)
        return self._result


class TestRunVerification:
    @pytest.mark.asyncio
    async def test_resolved_iff_reward_is_the_string_one(self) -> None:
        report = '{"_test_completed": true, "reward": "1", "passed": 1, "total_expected": 1}'
        log = f"{REPORT_BEGIN}\n{report}\n{REPORT_END}\n"
        result = await run_verification(sandbox=_FakeSandbox(log), inputs=_inputs(), is_golden=True)
        assert result.completed and result.resolved

    @pytest.mark.asyncio
    async def test_reward_zero_is_not_resolved(self) -> None:
        report = '{"_test_completed": true, "reward": "0", "passed": 0, "total_expected": 1}'
        log = f"{REPORT_BEGIN}\n{report}\n{REPORT_END}\n"
        result = await run_verification(sandbox=_FakeSandbox(log), inputs=_inputs(), is_golden=True)
        assert result.completed and not result.resolved

    @pytest.mark.asyncio
    async def test_missing_workdir_is_incomplete_not_a_zero(self) -> None:
        result = await run_verification(sandbox=_FakeSandbox("", return_code=97), inputs=_inputs(), is_golden=True)
        assert result.completed is False
        assert result.resolved is False
        assert "not present in the image" in result.error

    @pytest.mark.asyncio
    async def test_no_report_produced_is_incomplete(self) -> None:
        result = await run_verification(sandbox=_FakeSandbox("nothing bracketed"), inputs=_inputs(), is_golden=True)
        assert result.completed is False
        assert result.resolved is False

    @pytest.mark.asyncio
    async def test_malformed_report_is_incomplete_not_a_crash(self) -> None:
        log = f"{REPORT_BEGIN}\nnot valid json\n{REPORT_END}\n"
        result = await run_verification(sandbox=_FakeSandbox(log), inputs=_inputs(), is_golden=True)
        assert result.completed is False
        assert result.resolved is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_carries_the_per_file_report_through_as_test_results(self) -> None:
        report = '{"_test_completed": true, "reward": "1", "per_file": {"tests/test_a.py": {"passed": 1}}}'
        log = f"{REPORT_BEGIN}\n{report}\n{REPORT_END}\n"
        result = await run_verification(sandbox=_FakeSandbox(log), inputs=_inputs(), is_golden=True)
        assert result.test_results["per_file"]["tests/test_a.py"]["passed"] == 1


class TestVerifyResponseShape:
    """BaseVerifyResponse extends BaseVerifyRequest, so the request fields are REQUIRED on the
    way out. Building the response from scratch drops them and the resulting ValidationError
    surfaces to the caller as a bare JSON string -- which is how a fully successful verification
    can still fail the whole sweep. (Hit for real in resources_servers/swe_rebench/app.py.)"""

    @staticmethod
    def _body() -> dict:
        return {
            "instance_id": "langchain-ai_langgraph-supervisor-py_pr184",
            "workdir": "/workspace/langgraph-supervisor-py",
            "image_ref": "docker.io/aweaiteam/denovoswe:langchain-ai_langgraph-supervisor-py_pr184",
            "base_commit": "1bf5acae5966bec5b174a903be038c1d271dccd7",
            "language": "python",
            "passed_ptp": [],
            "failed_ptp": [],
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
        from resources_servers.denovo_swe.app import DeNovoSWEVerifyResponse

        response = DeNovoSWEVerifyResponse.model_validate(
            self._body()
            | {
                "reward": 1.0,
                "evaluation_completed": True,
                "resolved": True,
                "patch_applied": True,
                "test_results": {"reward": "1"},
                "test_output": "",
                "error": None,
                "eval_sandbox_start_time_taken": 0.1,
                "patch_verification_time_taken": 0.2,
            }
        )
        assert response.instance_id == "langchain-ai_langgraph-supervisor-py_pr184"

    def test_building_from_scratch_without_the_request_fields_fails(self) -> None:
        from resources_servers.denovo_swe.app import DeNovoSWEVerifyResponse

        with pytest.raises(Exception):
            DeNovoSWEVerifyResponse.model_validate(
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
        assert "app = DeNovoSWEResourcesServer.run_webserver()" in source

    def test_the_config_that_needs_it_still_sets_num_workers(self) -> None:
        """Pins the pair: if num_workers is configured, the entrypoint branch must exist."""
        from pathlib import Path

        import yaml

        config = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "denovo_swe.yaml").read_text())
        for name, block in config.items():
            workers = block["resources_servers"]["denovo_swe"].get("num_workers")
            if workers and workers > 1:
                assert "is_nemo_gym_fastapi_entrypoint(__file__)" in self._source(), (
                    f"{name} sets num_workers={workers} but the entrypoint has no module-level app"
                )
