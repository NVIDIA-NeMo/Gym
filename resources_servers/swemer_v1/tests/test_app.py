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
"""Tests for the Swemer-v1 resources server's grading and script construction.

Grading itself (output-flag injection, framework parsing) is delegated to
``responses_api_agents.swe_agents.swe_bench_ext``, which owns its own test coverage; this file
tests the stable contract swemer_v1's app.py depends on (build_eval_script, mirror_files,
verification_files, grade, run_verification, response shape, multi-worker entrypoint), plus the
patch-section helpers that are unique to v1 (``app._extract_model_patch`` uses them to turn an
agent's own sandbox diff into a clean candidate patch).

Two classes here pin bugs that were actually hit building the sibling SWE resources servers
(scale_swe, swe_rebench): a verify response missing the echoed request fields, and a
multi-worker entrypoint with no module-level `app`. Both looked fine locally and both only
failed once real traffic hit them, so the same tests are written here from the start.
"""

from types import SimpleNamespace

import pytest

from resources_servers.swemer_v1.verification import (
    GRADLE_INIT_PATH,
    MAVEN_SETTINGS_PATH,
    RESULT_FILE_BEGIN,
    RESULT_FILE_END,
    SUPPORTED_FRAMEWORKS,
    TEST_OUTPUT_BEGIN,
    TEST_OUTPUT_END,
    VerificationInputs,
    _slice,
    build_eval_script,
    drop_patch_sections,
    grade,
    mirror_files,
    patch_section_path,
    run_verification,
    verification_files,
)


def _inputs(**overrides) -> VerificationInputs:
    base = dict(
        instance_id="delivery_01__example-repo-issue-1",
        workdir="/workspace/repo",
        patch="diff --git a/a.py b/a.py\n",
        test_patch="diff --git a/tests/test_a.py b/tests/test_a.py\n",
        test_framework="pytest",
        test_command="pytest tests/test_a.py -v",
        fail_to_pass=["tests/test_a.py::test_one"],
        pass_to_pass=["tests/test_a.py::test_two"],
    )
    base.update(overrides)
    return VerificationInputs(**base)


class TestPatchSectionPath:
    def test_extracts_the_b_path_from_a_simple_diff(self) -> None:
        section = "diff --git a/src/foo.py b/src/foo.py\n--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-x\n+y\n"
        assert patch_section_path(section) == "src/foo.py"

    def test_a_deleted_file_falls_back_to_the_a_path(self) -> None:
        section = "diff --git a/src/gone.py b/src/gone.py\n--- a/src/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
        assert patch_section_path(section) == "src/gone.py"

    def test_a_new_file_uses_the_b_path(self) -> None:
        section = "diff --git a/src/new.py b/src/new.py\n--- /dev/null\n+++ b/src/new.py\n@@ -0,0 +1 @@\n+x\n"
        assert patch_section_path(section) == "src/new.py"

    def test_falls_back_to_the_header_line_when_there_are_no_hunk_markers(self) -> None:
        # e.g. a binary-file diff, which has no ---/+++ lines at all.
        section = "diff --git a/img.png b/img.png\nBinary files a/img.png and b/img.png differ\n"
        assert patch_section_path(section) == "img.png"

    def test_empty_section_yields_no_path(self) -> None:
        assert patch_section_path("") is None


class TestDropPatchSections:
    _PATCH = (
        "diff --git a/keep.py b/keep.py\n--- a/keep.py\n+++ b/keep.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/drop.py b/drop.py\n--- /dev/null\n+++ b/drop.py\n@@ -0,0 +1 @@\n+z\n"
    )

    def test_drops_sections_matching_given_paths(self) -> None:
        result = drop_patch_sections(self._PATCH, {"drop.py"})
        assert "keep.py" in result
        assert "drop.py" not in result

    def test_keeps_everything_when_no_paths_are_given(self) -> None:
        assert drop_patch_sections(self._PATCH, set()) == self._PATCH

    def test_empty_patch_stays_empty(self) -> None:
        assert drop_patch_sections("", {"drop.py"}) == ""


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
        script = build_eval_script(_inputs(test_command="pytest tests/test_a.py -v"))
        assert TEST_OUTPUT_BEGIN in script
        assert TEST_OUTPUT_END in script
        assert "pytest tests/test_a.py -v --junitxml=" in script

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

    def test_junit_uses_a_find_command_for_surefire_reports(self) -> None:
        # JUnit/Maven results land under */target/surefire-reports/TEST-*.xml, one file per test
        # class, not a single fixed path -- so the result-file step is a `find`, not a plain `cat`.
        script = build_eval_script(_inputs(test_framework="junit", test_command="mvn test"))
        assert "find /workspace/repo -path '*/target/surefire-reports' -name 'TEST-*.xml'" in script

    def test_slice_extracts_between_markers(self) -> None:
        log = f"noise before\n{TEST_OUTPUT_BEGIN}\nPASSED tests/test_a.py::test_one\n{TEST_OUTPUT_END}\nnoise after"
        sliced = _slice(log, TEST_OUTPUT_BEGIN, TEST_OUTPUT_END)
        assert "noise before" not in sliced
        assert "noise after" not in sliced
        assert "PASSED tests/test_a.py::test_one" in sliced


class TestMirrorFiles:
    """The Maven/Gradle mirror fix is local to swemer_v1 (not
    responses_api_agents/swe_agents/maven_mirror/) because init.gradle here has a
    swemer_v1-specific Gradle<6.8 compatibility fix; these tests pin that the local copy, not the
    shared one, is what ships, and that the compatibility fix is actually present in it."""

    def test_ships_to_the_paths_maven_and_gradle_actually_read(self) -> None:
        files = mirror_files()
        assert MAVEN_SETTINGS_PATH in files
        assert GRADLE_INIT_PATH in files

    def test_maven_settings_point_at_the_google_mirror(self) -> None:
        files = mirror_files()
        assert "maven-central.storage-download.googleapis.com" in files[MAVEN_SETTINGS_PATH]
        assert "<mirrorOf>central</mirrorOf>" in files[MAVEN_SETTINGS_PATH]

    def test_gradle_init_wraps_beforesettings_for_older_gradle(self) -> None:
        # gradle.beforeSettings is a Gradle 6.8+ API; calling it unguarded throws
        # MissingMethodException on older Gradle and fails the whole build outright.
        init_gradle = mirror_files()[GRADLE_INIT_PATH]
        before_settings_idx = init_gradle.index("gradle.beforeSettings")
        try_idx = init_gradle.rindex("try {", 0, before_settings_idx)
        catch_idx = init_gradle.index("catch (Throwable ignored)", before_settings_idx)
        assert try_idx < before_settings_idx < catch_idx


class TestVerificationFiles:
    def test_includes_both_patches_and_the_mirror_files(self) -> None:
        files = verification_files(_inputs())
        assert "/tmp/nemo_gym_patch.diff" in files
        assert "/tmp/nemo_gym_test_patch.diff" in files
        assert "/tmp/nemo_gym_eval.sh" in files
        assert MAVEN_SETTINGS_PATH in files
        assert GRADLE_INIT_PATH in files

    def test_omits_blank_patch(self) -> None:
        files = verification_files(_inputs(patch="   "))
        assert "/tmp/nemo_gym_patch.diff" not in files

    def test_ships_the_mirror_even_for_a_non_jvm_framework(self) -> None:
        # Harmless for non-JVM rows, which never read these files -- but shipping them
        # unconditionally means the agent's own sandbox (seed_session) doesn't need to know the
        # row's framework ahead of time either.
        files = verification_files(_inputs(test_framework="go", test_command="go test ./..."))
        assert MAVEN_SETTINGS_PATH in files
        assert GRADLE_INIT_PATH in files


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

    def test_cargo_nextest_counter_prefix_matches_via_normalize_fallback(self) -> None:
        """cargo-nextest's raw output prefixes each id with its PARALLEL completion order, e.g.
        "( 4/10) mod::test" -- not a stable per-test identity, so it can differ between the run
        that recorded FAIL_TO_PASS and any later run of the exact same test. Confirmed for real:
        several cargo rows showed "N tests run: N passed" while still grading as unresolved
        before this fallback existed."""
        statuses = {"( 4/10) mod::test": "PASSED"}
        report = grade(statuses, fail_to_pass=["( 7/10) mod::test"], pass_to_pass=[], test_framework="cargo-nextest")
        assert report["resolved"] is True

    def test_exact_match_still_preferred_over_normalized(self) -> None:
        """Normalization must be a fallback, not the primary path -- an id present in BOTH its
        exact form and only distinguishable after normalization should resolve via the exact
        match, matching whatever status the exact key carries even if a same-normalized-form key
        also exists with a different status."""
        statuses = {"a::b": "PASSED", "a.b": "FAILED"}
        report = grade(statuses, fail_to_pass=["a::b"], pass_to_pass=[], test_framework="")
        assert report["resolved"] is True


class TestSupportedFrameworks:
    def test_covers_swe_bench_ext_minus_bazel_and_jasmine(self) -> None:
        assert SUPPORTED_FRAMEWORKS == frozenset(
            {
                "pytest",
                "unittest",
                "junit",
                "maven",
                "gtest",
                "cargo-nextest",
                "cargo",
                "go",
                "jest",
                "vitest",
                "mocha",
                "bun",
                "ctest",
                "cppunit",
                "bespoke_libgeos",
                "xctest",
                "testing",
                "busted",
                "luaunit",
                "telescope",
                "lust",
                "minitest",
                "tap",
                "tape",
                "hardhat",
            }
        )


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
        # go has no result file, so this exercises the plain stdout path end to end through the
        # real swe_bench_ext go-json parser.
        log = (
            f"{TEST_OUTPUT_BEGIN}\n"
            '{"Action": "pass", "Test": "TestOne", "Package": "example.com/pkg"}\n'
            f"{TEST_OUTPUT_END}\n"
            f"{RESULT_FILE_BEGIN}\n{RESULT_FILE_END}\n"
        )
        sandbox = _FakeSandbox(log)
        inputs = _inputs(
            test_framework="go",
            test_command="go test ./...",
            fail_to_pass=["example.com/pkg::TestOne"],
            pass_to_pass=[],
        )
        result = await run_verification(sandbox=sandbox, inputs=inputs)
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
            '{"Action": "pass", "Test": "TestOne", "Package": "pkg"}\n'
            f"{TEST_OUTPUT_END}\ntrailing\n{RESULT_FILE_BEGIN}\n{RESULT_FILE_END}\n"
        )
        inputs = _inputs(
            test_framework="go", test_command="go test ./...", fail_to_pass=["pkg::TestOne"], pass_to_pass=[]
        )
        result = await run_verification(sandbox=_FakeSandbox(log), inputs=inputs)
        assert result.resolved is True  # would be masked by the install-time "FAILED" if unsliced

    @pytest.mark.asyncio
    async def test_result_file_is_preferred_over_stdout_when_both_are_present(self) -> None:
        # A minimal JUnit XML result file reporting a pass, alongside stdout noise that would
        # not parse as JUnit XML at all -- proving the result file, not stdout, drives the verdict.
        result_file = (
            '<?xml version="1.0"?><testsuite><testcase classname="tests.test_a" name="test_one"/></testsuite>'
        )
        log = (
            f"{TEST_OUTPUT_BEGIN}\nsome unrelated pytest console noise\n{TEST_OUTPUT_END}\n"
            f"{RESULT_FILE_BEGIN}\n{result_file}\n{RESULT_FILE_END}\n"
        )
        inputs = _inputs(fail_to_pass=["tests.test_a::test_one"], pass_to_pass=[])
        result = await run_verification(sandbox=_FakeSandbox(log), inputs=inputs)
        assert result.resolved is True

    @pytest.mark.asyncio
    async def test_falls_back_to_stdout_when_the_result_file_is_unparseable(self) -> None:
        # The result file has content but it isn't valid JUnit XML (e.g. the framework never got
        # far enough to write a real one) -- parsing it yields no statuses, so run_verification
        # retries against stdout rather than treating an empty parse as "nothing passed".
        result_file = "not xml at all, and no recognizable error text either"
        stdout = '<?xml version="1.0"?><testsuite><testcase classname="tests.test_a" name="test_one"/></testsuite>'
        log = (
            f"{TEST_OUTPUT_BEGIN}\n{stdout}\n{TEST_OUTPUT_END}\n"
            f"{RESULT_FILE_BEGIN}\n{result_file}\n{RESULT_FILE_END}\n"
        )
        inputs = _inputs(fail_to_pass=["tests.test_a::test_one"], pass_to_pass=[])
        result = await run_verification(sandbox=_FakeSandbox(log), inputs=inputs)
        assert result.completed and result.resolved


class TestVerifyResponseShape:
    """BaseVerifyResponse extends BaseVerifyRequest, so the request fields are REQUIRED on the
    way out. Building the response from scratch drops them and the resulting ValidationError
    surfaces to the caller as a bare JSON string -- which is how a fully successful verification
    can still fail the whole sweep. (Hit for real in resources_servers/swe_rebench/app.py.)"""

    @staticmethod
    def _body() -> dict:
        return {
            "instance_id": "delivery_01__example-repo-issue-1",
            "workdir": "/workspace/repo",
            "image_ref": "942195279341.dkr.ecr.us-east-2.amazonaws.com/ext-nvidia-agentic-v1:delivery_01__example-repo-issue-1",
            "language": "python",
            "test_framework": "pytest",
            "test_command": "pytest tests/test_a.py -v",
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
        from resources_servers.swemer_v1.app import SwemerV1VerifyResponse

        response = SwemerV1VerifyResponse.model_validate(
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
        assert response.instance_id == "delivery_01__example-repo-issue-1"
        assert response.test_framework == "pytest"

    def test_building_from_scratch_without_the_request_fields_fails(self) -> None:
        from resources_servers.swemer_v1.app import SwemerV1VerifyResponse

        with pytest.raises(Exception):
            SwemerV1VerifyResponse.model_validate(
                {
                    "reward": 1.0,
                    "evaluation_completed": True,
                    "resolved": True,
                    "patch_applied": True,
                    "instance_id": "x",
                    "language": "python",
                    "test_framework": "pytest",
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
        from resources_servers.swemer_v1.app import SwemerV1ResourcesServerConfig

        assert SwemerV1ResourcesServerConfig.model_fields["apply_anti_cheating"].default is True

    def test_seed_session_ensures_a_git_repo_before_scrubbing(self) -> None:
        source = self._source()
        assert "_ensure_git_repo(" in source
        ensure_idx = source.index("await self._ensure_git_repo(")
        anti_cheat_idx = source.index("apply_anti_cheat_setup(sandbox")
        assert ensure_idx < anti_cheat_idx

    def test_agent_facing_config_turns_it_on(self) -> None:
        from pathlib import Path

        import yaml

        config = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "swemer_v1.yaml").read_text())
        assert config["swemer_v1_resources_server"]["resources_servers"]["swemer_v1"]["apply_anti_cheating"] is True


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
        assert "app = SwemerV1ResourcesServer.run_webserver()" in source

    def test_the_config_that_needs_it_still_sets_num_workers(self) -> None:
        """Pins the pair: if num_workers is configured, the entrypoint branch must exist."""
        from pathlib import Path

        import yaml

        config = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "swemer_v1.yaml").read_text())
        for name, block in config.items():
            workers = block["resources_servers"]["swemer_v1"].get("num_workers")
            if workers and workers > 1:
                assert "is_nemo_gym_fastapi_entrypoint(__file__)" in self._source(), (
                    f"{name} sets num_workers={workers} but the entrypoint has no module-level app"
                )
