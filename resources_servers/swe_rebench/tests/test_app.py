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
"""Tests for the SWE-rebench-V2 resources server's grading and script construction.

These cover the parts that decide whether a task counts as resolved. The sandbox lifecycle
itself is exercised by the golden-patch run, which needs a live OpenSandbox endpoint.
"""

from types import SimpleNamespace

import pytest

from resources_servers.swe_rebench.verification import (
    TEST_OUTPUT_BEGIN,
    TEST_OUTPUT_END,
    VerificationInputs,
    as_command_list,
    build_eval_script,
    grade,
    normalize_test_name,
    repo_directory,
    run_verification,
    slice_test_output,
    verification_files,
)


def _inputs(**overrides) -> VerificationInputs:
    base = dict(
        instance_id="elastic__synthetics-316",
        repo="elastic/synthetics",
        base_commit="f52f0bf",
        patch="diff --git a/a b/a\n",
        test_patch="diff --git a/t b/t\n",
        install=["npm ci --quiet"],
        test_cmd=["npm run test:unit"],
        log_parser="parse_log_js_4",
        fail_to_pass=["run journey - failed on beforeAll"],
        pass_to_pass=["log to specified fd"],
    )
    base.update(overrides)
    return VerificationInputs(**base)


class TestRepoDirectory:
    def test_uses_the_repo_name_not_the_owner(self) -> None:
        # The images check out at /<repo-name>; using the owner would cd into a missing path
        # and every task would fail identically.
        assert repo_directory("elastic/synthetics") == "/synthetics"

    def test_tolerates_a_bare_repo_name(self) -> None:
        assert repo_directory("synthetics") == "/synthetics"


class TestCommandList:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("npm test", ["npm test"]),
            (["a", "b"], ["a", "b"]),
            (None, []),
            ("", []),
        ],
    )
    def test_accepts_both_dataset_spellings(self, value, expected) -> None:
        """install/test_cmd are a string in some rows and a list in others."""
        assert as_command_list(value) == expected


class TestNormalizeTestName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("suite > case [123 ms]", "suite > case"),
            ("suite > case (1.5 s)", "suite > case"),
            ("test_thing in 2.0 sec", "test_thing"),
            ("plain name", "plain name"),
        ],
    )
    def test_strips_per_run_timings(self, raw, expected) -> None:
        """The expectation and the observed name otherwise differ only by a duration."""
        assert normalize_test_name(raw) == expected


class TestSliceTestOutput:
    def test_keeps_only_the_graded_region(self) -> None:
        log = f"installing deps\nFAILED bogus\n{TEST_OUTPUT_BEGIN}\nPASSED real\n{TEST_OUTPUT_END}\ncleanup"
        sliced = slice_test_output(log)
        assert "PASSED real" in sliced
        # Install noise must not reach the parser: several parsers would read that FAILED line
        # as a test result and grade a healthy run as broken.
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
        """The important one: a broken test command reports nothing, and 'absent' must never
        read as success or every such row scores as resolved."""
        report = grade({}, ["f2p"], ["p2p"])
        assert report["resolved"] is False
        assert report["FAIL_TO_PASS"]["failure"] == ["f2p"]
        assert report["tests_observed"] == 0

    def test_timing_suffixes_do_not_break_matching(self) -> None:
        report = grade({"case [12 ms]": "PASSED"}, ["case"], [])
        assert report["resolved"] is True

    def test_skipped_is_not_passed(self) -> None:
        assert grade({"f2p": "SKIPPED"}, ["f2p"], [])["resolved"] is False


class TestBuildEvalScript:
    def test_runs_in_the_repo_directory_and_signals_a_missing_one(self) -> None:
        script = build_eval_script(_inputs())
        # shlex.quote leaves a plain path bare, so assert the behaviour rather than the quoting.
        assert "exit 97" in script
        assert any(line.startswith("cd ") and "/synthetics" in line for line in script.splitlines())

    def test_applies_both_patches_before_testing(self) -> None:
        script = build_eval_script(_inputs())
        patch_at = script.index("nemo_gym_patch.diff")
        test_patch_at = script.index("nemo_gym_test_patch.diff")
        test_at = script.index("npm run test:unit")
        assert patch_at < test_at and test_patch_at < test_at

    def test_omits_the_patch_step_when_there_is_no_patch(self) -> None:
        """An empty model patch must not produce a `git apply` of a nonexistent file."""
        script = build_eval_script(_inputs(patch=""))
        assert "nemo_gym_patch.diff" not in script
        assert "nemo_gym_test_patch.diff" in script

    def test_brackets_the_test_command_with_markers(self) -> None:
        script = build_eval_script(_inputs())
        assert script.index(TEST_OUTPUT_BEGIN) < script.index("npm run test:unit") < script.index(TEST_OUTPUT_END)

    def test_propagates_the_test_exit_code_not_the_install_one(self) -> None:
        script = build_eval_script(_inputs())
        assert "__test_exit=$?" in script and "exit $__test_exit" in script


class TestVerificationFiles:
    def test_ships_the_script_and_only_the_patches_that_exist(self) -> None:
        files = verification_files(_inputs(patch=""))
        assert "/tmp/nemo_gym_eval.sh" in files
        assert "/tmp/nemo_gym_patch.diff" not in files
        assert "/tmp/nemo_gym_test_patch.diff" in files


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
        log = f"{TEST_OUTPUT_BEGIN}\nirrelevant\n{TEST_OUTPUT_END}"
        sandbox = _FakeSandbox(log)
        result = await run_verification(
            sandbox=sandbox,
            inputs=_inputs(fail_to_pass=["a"], pass_to_pass=[]),
            parser=lambda _log: {"a": "PASSED"},
        )
        assert result.completed and result.resolved
        assert sandbox.commands == ["bash /tmp/nemo_gym_eval.sh"]

    @pytest.mark.asyncio
    async def test_missing_repo_directory_is_incomplete_not_a_zero(self) -> None:
        """Exit 97 means the image lacks the checkout — an infrastructure fault. Reporting it
        as a completed, unresolved run would make a broken image look like a hard task."""
        result = await run_verification(
            sandbox=_FakeSandbox("", return_code=97),
            inputs=_inputs(),
            parser=lambda _log: {},
        )
        assert result.completed is False
        assert result.resolved is False
        assert "not present in the image" in result.error

    @pytest.mark.asyncio
    async def test_a_parser_crash_is_reported_rather_than_scored_zero(self) -> None:
        def boom(_log: str) -> dict[str, str]:
            raise ValueError("bad log")

        result = await run_verification(sandbox=_FakeSandbox("x"), inputs=_inputs(), parser=boom)
        assert result.completed is False
        assert "raised" in result.error

    @pytest.mark.asyncio
    async def test_only_the_marked_region_reaches_the_parser(self) -> None:
        seen: list[str] = []

        def capture(log: str) -> dict[str, str]:
            seen.append(log)
            return {}

        log = f"install FAILED noise\n{TEST_OUTPUT_BEGIN}\nreal output\n{TEST_OUTPUT_END}\ntrailing"
        await run_verification(sandbox=_FakeSandbox(log), inputs=_inputs(), parser=capture)
        assert "real output" in seen[0]
        assert "install FAILED noise" not in seen[0]


class TestVerifyResponseShape:
    """BaseVerifyResponse extends BaseVerifyRequest, so the request fields are REQUIRED on the
    way out. Building the response from scratch drops them, and the resulting ValidationError
    surfaces to the caller as a bare JSON string rather than a result object — which is how a
    fully successful verification (reward 1.0, tests parsed) still failed the whole sweep."""

    @staticmethod
    def _body() -> dict:
        return {
            "instance_id": "elastic__synthetics-316",
            "repo": "elastic/synthetics",
            "base_commit": "f52f0bf",
            "image_name": "docker.io/swerebenchv2/elastic-synthetics:316-f52f0bf",
            "language": "ts",
            "install_config": {"test_cmd": "npm test", "log_parser": "parse_log_js_4"},
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
        from resources_servers.swe_rebench.app import SWERebenchVerifyResponse

        response = SWERebenchVerifyResponse.model_validate(
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
        """Pins the actual failure: without the echoed request fields, validation fails."""
        import pydantic

        from resources_servers.swe_rebench.app import SWERebenchVerifyResponse

        with pytest.raises(pydantic.ValidationError, match="responses_create_params|response"):
            SWERebenchVerifyResponse.model_validate(
                {
                    "reward": 1.0,
                    "evaluation_completed": True,
                    "resolved": True,
                    "patch_applied": True,
                    "instance_id": "i",
                    "language": "ts",
                    "test_results": None,
                    "test_output": "",
                    "error": None,
                    "eval_sandbox_start_time_taken": 0.0,
                    "patch_verification_time_taken": 0.0,
                }
            )


class TestSandboxCpuCap:
    """Build tools size their worker pools from the HOST core count, not the cgroup quota, so
    without these caps a 4-CPU sandbox on a 96-core node runs ~96 compilers and CFS-throttles.
    This set is the compile-heaviest in the repo, so it is the one that most needs them."""

    def test_cpu_cap_env_is_derived_from_the_cpu_limit(self) -> None:
        from nemo_gym.sandbox.utils import cpu_cap_env

        env = cpu_cap_env(4)
        assert env["OMP_NUM_THREADS"] == "4"
        assert env["GOMAXPROCS"] == "4"
        assert env["CARGO_BUILD_JOBS"] == "4"

    def test_the_server_applies_them_and_explicit_env_still_wins(self) -> None:
        from nemo_gym.sandbox.utils import cpu_cap_env

        # Mirrors the precedence in _create_sandbox: derived caps first, explicit env over them.
        derived = cpu_cap_env(4)
        explicit = {"GOMAXPROCS": "1", "MY_VAR": "x"}
        merged = derived | explicit
        assert merged["GOMAXPROCS"] == "1", "an explicit override must not be clobbered"
        assert merged["OMP_NUM_THREADS"] == "4"
        assert merged["MY_VAR"] == "x"

    def test_the_app_wires_cpu_cap_env_into_the_spec(self) -> None:
        """Source pin: the call is easy to drop in a refactor and its absence is invisible —
        tasks still pass, just far slower, which reads as 'these languages are slow'."""
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent / "app.py").read_text()
        assert "cpu_cap_env(sandbox_resources.cpu)" in source
        assert "env=env," in source, "the derived env must reach SandboxSpec"


class TestGoldenPatchAggregation:
    """The 3x repeat only means something if 'failed once' and 'had no verdict once' are kept
    apart: the first is a flaky test, the second is an image pull or provider fault that says
    nothing about the row."""

    @staticmethod
    def _obs(completed: bool, resolved: bool) -> dict:
        return {"evaluation_completed": completed, "resolved": resolved}

    def test_resolved_in_every_pass_is_supported(self) -> None:
        from resources_servers.swe_rebench.aggregate_golden_patch import classify

        assert classify([self._obs(True, True)] * 3, 3) == "supported"

    def test_resolved_in_some_passes_is_flaky(self) -> None:
        from resources_servers.swe_rebench.aggregate_golden_patch import classify

        observations = [self._obs(True, True), self._obs(True, False), self._obs(True, True)]
        assert classify(observations, 3) == "flaky"

    def test_never_resolved_is_broken_not_flaky(self) -> None:
        from resources_servers.swe_rebench.aggregate_golden_patch import classify

        assert classify([self._obs(True, False)] * 3, 3) == "broken"

    def test_a_missing_verdict_is_inconclusive_not_broken(self) -> None:
        """The important one: an infrastructure fault must not delete a row that resolved
        cleanly every time it was actually measured."""
        from resources_servers.swe_rebench.aggregate_golden_patch import classify

        observations = [self._obs(True, True), self._obs(True, True), self._obs(False, False)]
        assert classify(observations, 3) == "inconclusive"

    def test_a_row_missing_a_pass_entirely_is_inconclusive(self) -> None:
        from resources_servers.swe_rebench.aggregate_golden_patch import classify

        assert classify([self._obs(True, True), self._obs(True, True)], 3) == "inconclusive"


class TestMultiWorkerEntrypoint:
    """num_workers > 1 makes uvicorn re-import this entrypoint by path in each forked child.

    Without a module-level `app` every child exits and uvicorn stops the parent, so the server
    never binds. The symptom is not an ImportError but a flood of connection errors from clients
    talking to a dead port -- which is exactly how it presented: 715k ClientOSError and zero rows.
    """

    @staticmethod
    def _source() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / "app.py").read_text()

    def test_exposes_a_module_level_app_for_forked_workers(self) -> None:
        source = self._source()
        assert "is_nemo_gym_fastapi_entrypoint(__file__)" in source
        assert "app = SWERebenchResourcesServer.run_webserver()" in source

    def test_the_config_that_needs_it_still_sets_num_workers(self) -> None:
        """Pins the pair: if num_workers is configured, the entrypoint branch must exist."""
        from pathlib import Path

        import yaml

        config = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "swe_rebench.yaml").read_text())
        for name, block in config.items():
            workers = block["resources_servers"]["swe_rebench"].get("num_workers")
            if workers and workers > 1:
                assert "is_nemo_gym_fastapi_entrypoint(__file__)" in self._source(), (
                    f"{name} sets num_workers={workers} but the entrypoint has no module-level app"
                )


class TestFailureDiagnosis:
    """A resource failure and a broken row both end as resolved=False with a parsed log, so the
    two are only separable by evidence in the output. Getting this wrong in either direction is
    costly: miss it and good rows get marked broken; over-claim it and real failures get excused
    by raising limits that were never the problem."""

    def test_detects_an_oom_kill(self) -> None:
        from resources_servers.swe_rebench.diagnose_failures import classify_output

        assert classify_output("java.lang.OutOfMemoryError: Java heap space") == "oom"
        assert classify_output("fatal error: runtime: out of memory") == "oom"

    def test_detects_a_full_disk(self) -> None:
        from resources_servers.swe_rebench.diagnose_failures import classify_output

        assert classify_output("write error: No space left on device") == "disk_full"

    def test_disk_is_reported_ahead_of_a_bare_kill(self) -> None:
        """Ordering matters: a disk-full run often also prints 'Killed', and the actionable
        cause is the disk, not the signal."""
        from resources_servers.swe_rebench.diagnose_failures import classify_output

        assert classify_output("No space left on device\nKilled") == "disk_full"

    def test_an_ordinary_test_failure_is_not_a_resource_problem(self) -> None:
        from resources_servers.swe_rebench.diagnose_failures import classify_output

        assert classify_output("FAILED tests/test_x.py::test_y - AssertionError: 1 != 2") is None

    def test_network_faults_are_separated_from_row_failures(self) -> None:
        """Sandboxes are network restricted, so a dependency fetch escaping the image is an
        environment problem, not evidence that the golden patch is wrong."""
        from resources_servers.swe_rebench.diagnose_failures import classify_output

        assert classify_output("npm ERR! could not resolve host registry.npmjs.org") == "network"


class TestMavenMirror:
    """Every JVM failure in the 200-row pass was `repo.maven.apache.org ... Network is
    unreachable`, while crates.io, proxy.golang.org, npmjs and packagist all fetched fine from
    the same sandboxes. So egress works and Maven Central specifically does not; the fix is to
    point Maven and Gradle at the Google-hosted read-through mirror."""

    def test_mirror_configs_are_shipped_to_every_sandbox(self) -> None:
        from resources_servers.swe_rebench.verification import (
            GRADLE_INIT_PATH,
            MAVEN_SETTINGS_PATH,
            verification_files,
        )

        files = verification_files(_inputs())
        assert MAVEN_SETTINGS_PATH in files, "Maven reads ~/.m2/settings.xml by default"
        assert GRADLE_INIT_PATH in files, "Gradle auto-loads $GRADLE_USER_HOME/init.d/*.gradle"
        assert "maven-central.storage-download.googleapis.com" in files[MAVEN_SETTINGS_PATH]

    def test_gradle_home_is_set_or_the_init_script_is_never_read(self) -> None:
        """Shipping init.gradle is useless on its own: Gradle only scans init.d under
        GRADLE_USER_HOME, so without this the file sits there unread."""
        from resources_servers.swe_rebench.app import JVM_MIRROR_ENV
        from resources_servers.swe_rebench.verification import GRADLE_INIT_PATH

        assert JVM_MIRROR_ENV["GRADLE_USER_HOME"] == "/root/.gradle"
        assert GRADLE_INIT_PATH.startswith(JVM_MIRROR_ENV["GRADLE_USER_HOME"] + "/init.d/")

    def test_the_mirror_only_redirects_central(self) -> None:
        """mirrorOf must stay `central`, not `*`: a project's own snapshot or internal repos
        have to keep resolving from their real URLs or unrelated rows start failing."""
        from resources_servers.swe_rebench.verification import MAVEN_SETTINGS_PATH, verification_files

        settings = verification_files(_inputs())[MAVEN_SETTINGS_PATH]
        assert "<mirrorOf>central</mirrorOf>" in settings
        assert "<mirrorOf>*</mirrorOf>" not in settings

    def test_explicit_config_env_still_wins(self) -> None:
        from resources_servers.swe_rebench.app import JVM_MIRROR_ENV

        merged = JVM_MIRROR_ENV | {"GRADLE_USER_HOME": "/custom"}
        assert merged["GRADLE_USER_HOME"] == "/custom"


class TestFailureDiagnosis:
    """A resource failure and a broken row both end as resolved=False with a parsed log, so the
    two are only separable by evidence in the output. Getting this wrong in either direction is
    costly: miss it and good rows get marked broken; over-claim it and real failures get excused
    by raising limits that were never the problem."""

    def test_detects_an_oom_kill(self) -> None:
        from resources_servers.swe_rebench.diagnose_failures import classify_output

        assert classify_output("java.lang.OutOfMemoryError: Java heap space") == "oom"
        assert classify_output("fatal error: runtime: out of memory") == "oom"

    def test_detects_a_full_disk(self) -> None:
        from resources_servers.swe_rebench.diagnose_failures import classify_output

        assert classify_output("write error: No space left on device") == "disk_full"

    def test_disk_is_reported_ahead_of_a_bare_kill(self) -> None:
        """Ordering matters: a disk-full run often also prints 'Killed', and the actionable
        cause is the disk, not the signal."""
        from resources_servers.swe_rebench.diagnose_failures import classify_output

        assert classify_output("No space left on device\nKilled") == "disk_full"

    def test_an_ordinary_test_failure_is_not_a_resource_problem(self) -> None:
        from resources_servers.swe_rebench.diagnose_failures import classify_output

        assert classify_output("FAILED tests/test_x.py::test_y - AssertionError: 1 != 2") is None

    def test_network_faults_are_separated_from_row_failures(self) -> None:
        """Sandboxes are network restricted, so a dependency fetch escaping the image is an
        environment problem, not evidence that the golden patch is wrong."""
        from resources_servers.swe_rebench.diagnose_failures import classify_output

        assert classify_output("npm ERR! could not resolve host registry.npmjs.org") == "network"


class TestJvmNetworkWorkaround:
    """The sandbox has working egress -- Go, npm and composer fetch fine in it. What fails is a
    JVM on a dual-stack network preferring an IPv6 address with no route, so Maven and Gradle
    dependency resolution dies with 'Network is unreachable'. Measured on a 200-row pass: 12 of
    13 network failures were JVM-family and every one showed that error."""

    def test_the_flag_targets_the_build_tools(self) -> None:
        from resources_servers.swe_rebench.app import JVM_IPV4_ENV

        assert set(JVM_IPV4_ENV) == {"MAVEN_OPTS", "GRADLE_OPTS", "SBT_OPTS"}
        for value in JVM_IPV4_ENV.values():
            assert "preferIPv6Addresses=false" in value

    def test_it_avoids_java_options_which_pollutes_the_graded_log(self) -> None:
        """_JAVA_OPTIONS would work but makes every JVM print 'Picked up _JAVA_OPTIONS: ...' to
        stderr, which lands inside the marked test region and is fed to line-oriented parsers."""
        from resources_servers.swe_rebench.app import JVM_IPV4_ENV

        assert "_JAVA_OPTIONS" not in JVM_IPV4_ENV

    def test_explicit_config_env_still_wins(self) -> None:
        from resources_servers.swe_rebench.app import JVM_IPV4_ENV

        merged = JVM_IPV4_ENV | {"MAVEN_OPTS": "-Xmx8g"}
        assert merged["MAVEN_OPTS"] == "-Xmx8g"
        assert "preferIPv6Addresses" in merged["GRADLE_OPTS"]
