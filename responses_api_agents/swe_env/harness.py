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

"""Task model and harness contract for the SWE environment library.

The harness contract is intentionally split across a trust boundary:

* ``build_spec`` / ``supports_provider`` / ``materialize`` are **provisioning**
  methods imported and called by *agents* (and the verifier).
* ``reset_repo`` / ``run_eval`` / ``grade`` are **grading** methods used
  **only** by the grader (``verify_task``). A test asserts agent adapters never
  reference them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nemo_gym.sandbox import SandboxSpec


if TYPE_CHECKING:
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment


@dataclass
class SweTask:
    """A single SWE task to provision and/or verify.

    Holds the instance metadata needed to launch a sandbox, materialize patches,
    run the evaluation, and grade the result.
    """

    instance_id: str
    image: str | None = None
    base_commit: str | None = None
    repo_workdir: str = "/testbed"
    test_command: str = ""
    test_framework: str = ""
    model_patch: str = ""
    test_patch: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    benchmark: str = "swe-bench-ext"
    split: str = "test"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalArtifacts:
    """Raw evaluation output retrieved from the sandbox, before grading."""

    test_output: str = ""
    return_code: int = 0
    patch_applied: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweEvalReport:
    """Graded result of a single task. ``error_kind`` masks a sample.

    ``error_kind`` is ``None`` for a clean grade. A non-``None`` value (e.g.
    ``"sandbox"`` / ``"eval_error"``) marks an infra failure: the sample is
    masked via this flag and ``reward_from_report`` returns ``0.0`` — **never**
    ``None`` (the wire ``reward`` field is a non-nullable ``float``).
    """

    instance_id: str
    resolved: bool = False
    patch_applied: bool = False
    patch_exists: bool = False
    error_kind: str | None = None
    tests_status: dict[str, Any] = field(default_factory=dict)


class SweTaskHarness(ABC):
    """Per-family provisioning + (server-private) grading recipe."""

    #: registry key, e.g. ``"swe-bench-ext"``.
    name: str = ""
    #: ``"flat-host-grade"`` (parse host-side) or ``"nested-harness"`` (in-container grader).
    grade_strategy: str = "flat-host-grade"

    # --- provisioning (agent-facing + verifier) ------------------------------

    @abstractmethod
    def build_spec(self, task: SweTask) -> SandboxSpec:
        """Build the sandbox spec for a task.

        Args:
            task (SweTask): The task to provision a sandbox for.

        Returns:
            SandboxSpec: The spec describing image, workdir, env, ttl, and
                provider options for the task.
        """

    def supports_provider(self, provider_name: str) -> bool:
        """Report whether this harness can run on the named provider.

        Nested-Docker families override this to reject exec-only providers.

        Args:
            provider_name (str): The name of the sandbox provider.

        Returns:
            bool: ``True`` if the provider is supported.
        """
        return True

    def with_flat_eval(self) -> "SweTaskHarness":
        """Return a variant that grades host-side (flat) on any exec-capable provider.

        Flat-graded families already grade host-side and return themselves. Nested-Docker
        families override this to return a flat-constructed copy, lifting the apptainer-only
        gate so they can grade on docker/opensandbox.

        Returns:
            SweTaskHarness: A harness whose grading runs host-side.
        """
        return self

    async def materialize(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Upload the model patch and test patch into the started sandbox.

        Args:
            env (AsyncSweEnvironment): The started environment to write into.
            task (SweTask): The task whose patches are uploaded.
        """
        if task.model_patch:
            await env.write_text("/root/patch.diff", _ensure_trailing_newline(task.model_patch))
        if task.test_patch:
            await env.write_text("/root/test_patch.diff", _ensure_trailing_newline(task.test_patch))

    # --- server-private grading (verifier only) ------------------------------

    async def reset_repo(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Reset the in-sandbox checkout to ``base_commit`` for hermetic grading.

        Uses only ``git reset --hard``, never ``git clean -fdx``: verification
        runs in a fresh sandbox (no agent edits to scrub), and a clean would
        delete the image's prebuilt artifacts (compiled C extensions, installed
        environment) and break the tests.

        Args:
            env (AsyncSweEnvironment): The started environment to reset.
            task (SweTask): The task whose ``base_commit`` and ``repo_workdir``
                are used.
        """
        if task.base_commit:
            await env.execute(f"git reset --hard {task.base_commit}", cwd=task.repo_workdir)

    @abstractmethod
    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        """Apply the patches and run the evaluation, returning raw artifacts.

        Args:
            env (AsyncSweEnvironment): The started environment to evaluate in.
            task (SweTask): The task being evaluated.

        Returns:
            EvalArtifacts: The raw evaluation output retrieved from the sandbox.
        """

    @abstractmethod
    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        """Parse raw artifacts host-side into a graded report.

        Args:
            task (SweTask): The task that was evaluated.
            artifacts (EvalArtifacts): The raw evaluation output to parse.

        Returns:
            SweEvalReport: The graded result for the task.
        """


def _ensure_trailing_newline(text: str) -> str:
    """Return the text with a single trailing newline.

    Args:
        text (str): The input text.

    Returns:
        str: The text unchanged if it already ends in a newline, otherwise the
            text with a newline appended.
    """
    return text if text.endswith("\n") else text + "\n"
