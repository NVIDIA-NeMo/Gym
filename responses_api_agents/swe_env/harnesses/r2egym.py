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

"""r2e-gym harness — host-side (flat) graded.

Runs the instance's eval script in the sandbox and parses the log host-side via the shared
flat-eval path, so it runs on any exec-capable provider.

NOTE: the apptainer-only nested ``run_local_evaluation`` path (which produced r2e-gym's own
``report.json`` in-container) was removed when PR #1694 took ownership of the apptainer
provider. Re-wiring r2e-gym's nested grading + ``.sif``/mounts onto #1694's provider is tracked
for a follow-up PR (see APPTAINER_PR3_TRACKER.md); until then r2e-gym grades flat (it needs an
``eval_script`` in task metadata, else the flat grader records an **unmasked** ``resolved=False``
— reward 0, kept in the gradient — not an eval-error mask; see
``test_r2egym.py::test_run_eval_missing_eval_script_is_unmasked_unresolved``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo_gym.sandbox import SandboxResources, SandboxSpec
from responses_api_agents.swe_env.harness import (
    EvalArtifacts,
    SweEvalReport,
    SweTask,
    SweTaskHarness,
    _ensure_trailing_newline,
    compute_resolved,
)
from responses_api_agents.swe_env.harnesses import flat_eval


if TYPE_CHECKING:
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment


class R2EGymHarness(SweTaskHarness):
    """Harness for the r2e-gym family of SWE tasks (host-side / flat graded)."""

    name = "r2e-gym"
    grade_strategy = "flat-host-grade"

    def build_spec(self, task: SweTask) -> SandboxSpec:
        """Build the sandbox spec for an r2e-gym task.

        Args:
            task: The SWE task whose metadata, image, and workdir describe the sandbox.

        Returns:
            SandboxSpec: The populated sandbox spec (image, workdir, TTL, env, metadata,
            resources, and any provider options carried on the task).
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
            provider_options=dict(task.metadata.get("provider_options", {})),
        )

    async def materialize(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Write the bare ``/root/patch.diff`` the eval script applies.

        Args:
            env: The active SWE environment used to write files into the sandbox.
            task: The SWE task supplying the model patch (newline-normalized).
        """
        if task.model_patch:
            await env.write_text("/root/patch.diff", _ensure_trailing_newline(task.model_patch))

    async def reset_repo(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Reset the repository checkout (no-op for r2e-gym).

        Args:
            env: The active SWE environment (unused).
            task: The SWE task (unused).
        """
        return None

    def hide_eval_tests_commands(self) -> list[str]:
        """Build shell commands that strip the held-out eval tests from the agent's checkout.

        ``/r2e_tests`` holds the evaluation tests the agent must not see; ``run_tests.sh``
        launches them. ``run_tests.sh`` is deleted only when it references ``r2e_tests``
        (substring guard). The agent adapter runs these after ``materialize``.

        Returns:
            list[str]: One shell command per checkout root (``""``, ``/root``, ``/testbed``).
        """
        commands: list[str] = []
        for root_dir in ["", "/root", "/testbed"]:
            commands.append(
                f"rm -rf {root_dir}/r2e_tests && "
                f"if grep -qs r2e_tests {root_dir}/run_tests.sh; then rm -rf {root_dir}/run_tests.sh; fi"
            )
        return commands

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        """Run the instance's eval script in-sandbox and grade the log host-side.

        Args:
            env: The active SWE environment used to execute commands in the sandbox.
            task: The SWE task whose ``metadata['eval_script']`` is run.

        Returns:
            EvalArtifacts: The captured test output, return code, patch existence, and flat
            markers (masked as ``eval_error`` when no eval script is present).
        """
        return await flat_eval.flat_run_eval(env, task)

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        """Grade an r2e-gym task from its evaluation artifacts (host-side, flat).

        Unlike the SWE-bench flat grader, this path does NOT gate ``resolved`` on the
        SWE-bench ``>>>>> Start/End Test Output`` marker pair: r2e-gym's ``run_tests.sh``
        does not emit those swebench sentinels, so requiring them would mask every r2e-gym
        sample as unresolved. Per-test status lines are parsed from the whole log and the
        node-ids are matched directly against the required ``fail_to_pass`` / ``pass_to_pass``
        sets (R2E-Gym uses pytest node-ids verbatim). Only genuine infra failures
        (sandbox/timeout) are masked.

        Args:
            task: The SWE task being graded.
            artifacts: The evaluation artifacts produced by ``run_eval``.

        Returns:
            SweEvalReport: The resolved/unresolved verdict with patch state and any error kind.
        """
        if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=artifacts.raw["error_type"],
            )
        # Parse per-test status lines from the whole log (no swebench-marker gate). An
        # unbuildable / empty log yields an empty status map -> no required test passes ->
        # unmasked unresolved, and compute_resolved still returns False for an empty
        # required set (the edge validated by main).
        status_map = flat_eval._parse_pytest_status_lines(artifacts.test_output)
        passed = flat_eval.passed_tests(status_map)
        # Thread the full status_map so compute_resolved mirrors swebench's
        # get_eval_tests_report semantics: neutral-status required tests (SKIPPED/XPASS)
        # are excluded rather than treated as failures.
        resolved = compute_resolved(
            fail_to_pass=task.fail_to_pass,
            pass_to_pass=task.pass_to_pass,
            passed=passed,
            status_map=status_map,
        )
        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_applied=bool(status_map),
            patch_exists=bool(task.model_patch),
            tests_status={"passed": passed, "all": status_map},
        )
