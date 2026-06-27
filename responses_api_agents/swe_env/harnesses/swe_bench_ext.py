# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""swe-bench-ext harness: flat, host-graded reference family.

Applies the model patch (and test patch) against the repository checkout, runs
the framework test command, and grades host-side with the parser
(:func:`responses_api_agents.swe_env.parsing.parse_and_check_tests`).

Grading delegates the full per-framework logic to ``parse_and_check_tests``:
junit-xml parsing, test-id normalization, the fuzzy matcher, the framework
dispatch, the ``::build``/``::compile`` synthetic-PASS injection, and
build-failed-package propagation.

``resolved`` is taken from the parser's verdict (all FAIL_TO_PASS passed AND all
PASS_TO_PASS passed). It does not depend on ``patch_applied``: the model and test
patches are applied best-effort and grading is on the tests only.
``patch_applied`` is still recorded for information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo_gym.sandbox import SandboxResources, SandboxSpec
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask, SweTaskHarness
from responses_api_agents.swe_env.parsing import (
    get_framework_config,
    get_test_command_with_output,
    parse_and_check_tests,
)


if TYPE_CHECKING:
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment


# Default checkout locations probed (in order) when locating the repo, mirroring main's
# ``cd /testbed 2>/dev/null || cd /workspace/repo 2>/dev/null || cd /app 2>/dev/null`` ladder
# in SweBenchExtDatasetProcessor's eval script.
_REPO_WORKDIR_LADDER = ("/testbed", "/workspace/repo", "/app", "/root/repo")


# Output markers the parser (parse_and_check_tests) extracts content between.
_TEST_OUTPUT_START = "<<<SWE_BENCH_EXT_TEST_OUTPUT_START>>>"
_TEST_OUTPUT_END = "<<<SWE_BENCH_EXT_TEST_OUTPUT_END>>>"
_RESULT_FILE_START = "<<<SWE_BENCH_EXT_RESULT_FILE_START>>>"
_RESULT_FILE_END = "<<<SWE_BENCH_EXT_RESULT_FILE_END>>>"


class SweBenchExtHarness(SweTaskHarness):
    """Flat, host-graded harness for the swe-bench-ext task family.

    Runs the task's framework test command inside a single sandbox and grades the
    captured output on the host. Works on any exec-capable sandbox provider.
    """

    name = "swe-bench-ext"
    grade_strategy = "flat-host-grade"

    def build_spec(self, task: SweTask) -> SandboxSpec:
        """Build the sandbox specification for a task.

        Args:
            task: The SWE task describing the image, working directory, and
                per-task metadata (timeouts, resources, provider options).

        Returns:
            SandboxSpec: The sandbox spec used to launch the task's container.
        """
        return SandboxSpec(
            image=task.image,
            workdir=task.repo_workdir,
            ttl_s=task.metadata.get("ttl_s", 1800),
            ready_timeout_s=task.metadata.get("ready_timeout_s", 600),
            env={"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_PAGER": "cat"},
            metadata={
                "instance_id": task.instance_id[:63],
                "benchmark": task.benchmark,
                "harness": self.name,
            },
            resources=SandboxResources.from_mapping(task.metadata.get("resources", {})),
            provider_options=task.metadata.get("provider_options", {}),
        )

    def supports_provider(self, provider_name: str) -> bool:
        """Report whether this harness supports a sandbox provider.

        Being flat and host-graded, it works on any exec-capable provider.

        Args:
            provider_name: The name of the sandbox provider.

        Returns:
            bool: Always ``True``.
        """
        return True

    async def _resolve_repo_workdir(self, env: "AsyncSweEnvironment", task: SweTask) -> str:
        """Locate the repository checkout, mirroring main's ``cd`` fallback ladder.

        Main's ``SweBenchExtDatasetProcessor`` eval script runs
        ``cd /testbed 2>/dev/null || cd /workspace/repo 2>/dev/null || cd /app 2>/dev/null``
        so a repo that is not at ``/testbed`` is still found. This reproduces that
        host-side: a row-provided ``repo_workdir`` that differs from the default and holds a
        ``.git`` checkout wins; otherwise the ladder (``/testbed``, ``/workspace/repo``,
        ``/app``, ``/root/repo``) is probed for a ``.git`` directory. If nothing matches the
        task's ``repo_workdir`` is returned unchanged (preserving prior behavior).

        Args:
            env: The async environment used to probe the sandbox.
            task: The SWE task whose ``repo_workdir`` is the preferred/default location.

        Returns:
            str: The resolved repository working directory inside the sandbox.
        """
        # Prefer an explicit, non-default row workdir holding a checkout.
        candidates: list[str] = []
        if task.repo_workdir and task.repo_workdir != "/testbed":
            candidates.append(task.repo_workdir)
        candidates.extend(d for d in _REPO_WORKDIR_LADDER if d not in candidates)
        for candidate in candidates:
            probe = await env.execute(f'test -d "{candidate}/.git"', cwd="/")
            if probe["returncode"] == 0:
                return candidate
        return task.repo_workdir

    async def reset_repo(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Reset the located checkout to ``base_commit`` for hermetic grading.

        Resolves the repo workdir via the same ladder main uses (so a non-``/testbed``
        checkout is found), then defers to the base ``git reset --hard`` behavior.

        Args:
            env: The started environment to reset.
            task: The task whose ``base_commit`` is restored.
        """
        if task.base_commit:
            workdir = await self._resolve_repo_workdir(env, task)
            await env.execute(f"git reset --hard {task.base_commit}", cwd=workdir)

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        """Apply patches, run the test command, and capture the evaluation output.

        Applies the model patch (and test patch) best-effort, then runs the
        framework test command wrapped between output markers so the parser can
        extract the structured result file or marked stdout.

        Args:
            env: The async environment used to execute commands in the sandbox.
            task: The SWE task providing the patches, test command, and framework.

        Returns:
            EvalArtifacts: The captured test output, return code, whether the
                model patch applied, and the execution error type if any.
        """
        # Resolve the checkout via main's cd ladder so a non-/testbed repo is found.
        workdir = await self._resolve_repo_workdir(env, task)
        patch_applied = True
        # Best-effort apply: a bad apply never fails the run (grading is on the
        # tests only); we still record whether the model patch applied for info.
        apply_flags = "--reject --recount --ignore-space-change --ignore-whitespace"
        if task.model_patch:
            applied = await env.execute(
                f"git apply {apply_flags} /root/patch.diff",
                cwd=workdir,
            )
            patch_applied = applied["returncode"] == 0
        if task.test_patch:
            await env.execute(
                f"git apply {apply_flags} /root/test_patch.diff",
                cwd=workdir,
            )
        # Wrap the command's output: add structured-output flags (--junitxml/--json)
        # via get_test_command_with_output, run it between the markers, and dump the
        # framework result file so parse_and_check_tests receives junit-xml (preferred)
        # or the marked stdout.
        #
        # The framework is passed through verbatim. An empty framework must NOT be
        # coerced to "pytest": for a non-pytest instance whose framework is absent, the
        # parser's auto-detect path is what grades correctly, and the default framework
        # config adds no flags and no result file. grade() reuses this SAME value via
        # _resolve_framework so the two stay in lockstep.
        framework = self._resolve_framework(task)
        # Use the row's test command verbatim, with NO default runner. Main's
        # SweBenchExtDatasetProcessor uses ``inst.get("test_command", "")`` (empty when
        # absent): a command-less row runs no runner and grades unresolved. Injecting a
        # default ``python -m pytest`` here would diverge from main by fabricating results.
        base_command = task.test_command
        test_cmd = get_test_command_with_output(base_command, framework)
        result_file = (get_framework_config(framework, base_command) or {}).get("result_file")
        result = await env.execute(self._wrap_eval_command(test_cmd, result_file), cwd=workdir, is_eval=True)
        return EvalArtifacts(
            test_output=result["output"],
            return_code=result["returncode"],
            patch_applied=patch_applied,
            raw={"error_type": result.get("error_type")},
        )

    @staticmethod
    def _resolve_framework(task: SweTask) -> str:
        """Return the framework value used by both ``run_eval`` and ``grade``.

        Returns the task's framework verbatim. An empty or unknown value is
        intentionally passed through unchanged: coercing it to ``"pytest"`` would
        mis-dispatch the parser for non-pytest instances that ship no framework.
        Centralizing this guarantees ``run_eval`` (which selects the
        structured-output flag and result file) and ``grade`` (which parses the
        output) agree on the framework.

        Args:
            task: The SWE task whose framework value is returned.

        Returns:
            str: The task's test framework name (possibly empty).
        """
        return task.test_framework

    @staticmethod
    def _wrap_eval_command(test_cmd: str, result_file: str | None) -> str:
        """Wrap the eval command in the output markers and a result-file dump.

        The parser prefers the junit/json result file (emitted between the
        RESULT_FILE markers) and falls back to the marked stdout. The ``mkdir -p``
        ensures ``/workspace/test-results`` exists first, since some frameworks
        (e.g. junit/gradle, xctest) write their result file there.

        Args:
            test_cmd: The test command to run inside the markers.
            result_file: Path or glob of the framework result file to dump, or
                ``None`` when the framework produces no result file.

        Returns:
            str: A shell script that runs the test command and emits the marked
                output and result-file blocks.
        """
        mkdir_block = "mkdir -p /workspace/test-results\n"
        if result_file and "*" in result_file:
            result_block = (
                f'echo "{_RESULT_FILE_START}"\n'
                f"for f in {result_file}; do\n"
                f'    if [ -f "$f" ]; then echo "=== FILE: $f ==="; cat "$f"; echo ""; fi\n'
                f"done 2>/dev/null || true\n"
                f'echo "{_RESULT_FILE_END}"\n'
            )
        elif result_file:
            result_block = (
                f'echo "{_RESULT_FILE_START}"\n'
                f'if [ -f "{result_file}" ]; then cat "{result_file}"; fi\n'
                f'echo "{_RESULT_FILE_END}"\n'
            )
        else:
            result_block = ""
        return f'{mkdir_block}echo "{_TEST_OUTPUT_START}"\n{test_cmd}\n{result_block}echo "{_TEST_OUTPUT_END}"\n'

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        """Grade captured evaluation artifacts into a report.

        Infrastructure failures are masked via ``error_kind`` and never scored as
        unresolved. Otherwise the test output is handed to ``parse_and_check_tests``
        and ``resolved`` is taken from the parser's verdict.

        Args:
            task: The SWE task providing the expected test sets and framework.
            artifacts: The captured test output, return code, and error type.

        Returns:
            SweEvalReport: The grading report, including ``resolved``,
                ``patch_applied``, ``patch_exists``, and the parsed test status (or
                ``error_kind`` on infrastructure failure).
        """
        # Infra failure: mask via error_kind (never scored as "unresolved").
        if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=artifacts.raw["error_type"],
            )
        # Delegate to the parser, passing the framework verbatim via the SAME
        # _resolve_framework value run_eval used. An empty/unknown framework falls
        # through to the parser's auto-detect path; coercing it to "pytest" here would
        # mis-grade non-pytest instances.
        test_framework = self._resolve_framework(task)
        result = parse_and_check_tests(
            test_output=artifacts.test_output,
            test_framework=test_framework,
            fail_to_pass=task.fail_to_pass,
            pass_to_pass=task.pass_to_pass,
            instance_id=task.instance_id,
        )
        # resolved is the parser's verdict (all F2P passed AND all P2P passed); it
        # does NOT gate on patch_applied (grading is on tests only).
        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=bool(result["resolved"]),
            patch_applied=artifacts.patch_applied,
            patch_exists=bool(task.model_patch),
            tests_status=result,
        )
