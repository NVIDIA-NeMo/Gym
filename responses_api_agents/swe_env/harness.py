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

This module also holds the name->harness registry
(``register_harness``/``get_harness``/``list_harnesses``) and the pure grading
helpers (``compute_resolved``/``reward_from_report``), merged here so the harness
contract, its dispatch, and its scoring live in one place.
"""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nemo_gym.sandbox import SandboxSpec


if TYPE_CHECKING:
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment


class GraderDependencyError(RuntimeError):
    """A required grading dependency is unavailable for a task the harness must grade exactly.

    Raised by a harness when it cannot grade an instance faithfully (e.g. ``swebench`` is
    missing for a SWE-bench instance) and degrading to a generic parser would silently skew
    the result. ``verify_task`` propagates this rather than swallowing it into an unmasked
    reward-0, so a misconfigured grader fails loudly instead of quietly degrading scores.
    """


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

        The base harness accepts every provider; flat host-graded families work on any
        exec-capable provider.

        Args:
            provider_name (str): The name of the sandbox provider.

        Returns:
            bool: ``True`` if the provider is supported.
        """
        return True

    def with_flat_eval(self) -> "SweTaskHarness":
        """Return a variant that grades host-side (flat) on any exec-capable provider.

        All families already grade host-side, so the base implementation returns ``self``.

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
            await env.execute(f"git reset --hard {shlex.quote(task.base_commit)}", cwd=task.repo_workdir)

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


# --- name->harness registry ----------------------------

_HARNESSES: dict[str, SweTaskHarness] = {}


def register_harness(harness: SweTaskHarness, *, override: bool = False) -> None:
    """Register a harness under its ``name``.

    Args:
        harness (SweTaskHarness): The harness to register. Its ``name`` must be
            non-empty.
        override (bool): If ``True``, replace an existing harness with the same
            name instead of raising.

    Raises:
        ValueError: If the harness name is empty, or a harness with the same name
            is already registered and ``override`` is ``False``.
    """
    if not harness.name:
        raise ValueError("Harness must define a non-empty 'name'")
    if not override and harness.name in _HARNESSES:
        raise ValueError(f"Harness {harness.name!r} is already registered")
    _HARNESSES[harness.name] = harness


# HuggingFace dataset names don't match registry keys; map by substring (most-specific first)
# so callers can pass a raw ``dataset_name`` (e.g. "princeton-nlp/SWE-bench_Verified").
_HF_NAME_ALIASES: list[tuple[str, str]] = [
    ("SWE-bench_Multilingual", "swe-bench-multilingual"),
    ("R2E-Gym", "r2e-gym"),
    ("SWE-rebench", "swe-rebench"),
    ("SWE-bench", "swe-bench"),
]


def _ensure_registered() -> None:
    """Lazily register the built-in harnesses if the registry is empty.

    Importing ``responses_api_agents.swe_env.harnesses`` registers all families, but a fresh
    process (e.g. a Ray worker running the decoupled agent) may call ``get_harness`` before that
    import has run. Registering on demand keeps lookups robust regardless of import order.
    """
    if _HARNESSES:
        return
    from responses_api_agents.swe_env.harnesses import register_builtin_harnesses

    register_builtin_harnesses()


def get_harness(name: str) -> SweTaskHarness:
    """Look up a harness by registry key, or by HuggingFace dataset-name substring.

    Built-in harnesses are registered on first use (robust to import order). An exact key match
    wins; otherwise a HuggingFace ``dataset_name`` substring is resolved to its key (e.g.
    ``"princeton-nlp/SWE-bench_Verified"`` -> ``"swe-bench"``).

    Args:
        name (str): The registry key, or a HuggingFace dataset name.

    Returns:
        SweTaskHarness: The registered harness.

    Raises:
        KeyError: If no harness matches ``name``.
    """
    _ensure_registered()
    if name in _HARNESSES:
        return _HARNESSES[name]
    for needle, key in _HF_NAME_ALIASES:
        if needle in name and key in _HARNESSES:
            return _HARNESSES[key]
    available = ", ".join(sorted(_HARNESSES)) or "(none)"
    raise KeyError(f"Unknown SWE harness {name!r}. Registered: {available}")


def list_harnesses() -> list[str]:
    """List the names of all registered harnesses.

    Returns:
        list[str]: The registered harness names, sorted alphabetically.
    """
    return sorted(_HARNESSES)


# --- pure grading helpers -------------------------------


def compute_resolved(
    *,
    fail_to_pass: Iterable[str],
    pass_to_pass: Iterable[str],
    passed: Iterable[str],
    eval_type: str = "pass_and_fail",
    status_map: dict[str, str] | None = None,
) -> bool:
    """Apply the SWE-bench resolution rule.

    Two eval types are supported, mirroring swebench's per-repo selection
    (``swebench.harness.grading.get_eval_report`` /
    ``get_eval_tests_report`` + ``get_resolution_status``):

    * ``"pass_and_fail"`` (default): mirrors swebench's ``check_pass_and_fail``
      classification combined with the ratio-based ``get_resolution_status``. When a
      ``status_map`` is supplied, each required test is a **success** when present and
      PASSED/XFAIL (``test_passed``), a **failure** when absent or FAILED/ERROR
      (``test_failed``), and **neutral** (excluded from both counts) for any other
      status (e.g. SKIPPED/XPASS). A task is resolved only when there are zero
      failures across FAIL_TO_PASS and PASS_TO_PASS (each ratio ``== 1``; an
      all-neutral category with total ``0`` counts as ``1``). Without a
      ``status_map`` it falls back to plain ``passed``-set membership.
    * ``"fail_only"``: used for the JS multilingual repos in swebench's
      ``FAIL_ONLY_REPOS`` (chartjs/Chart.js, processing/p5.js, markedjs/marked). A
      required test counts as success **unless** it is present in ``status_map``
      **and** its status is ``FAILED``. This mirrors swebench's ``check_fail_only``.

    Args:
        fail_to_pass (Iterable[str]): Tests that must transition from failing to
            passing.
        pass_to_pass (Iterable[str]): Tests that must remain passing.
        passed (Iterable[str]): The tests that actually passed.
        eval_type (str): ``"pass_and_fail"`` or ``"fail_only"`` (selected by the
            caller from ``test_spec.repo``).
        status_map (dict[str, str] | None): Full per-test status map. Required for
            the ``"fail_only"`` rule (to detect a present-and-FAILED required test)
            and used by ``"pass_and_fail"`` to exclude neutral-status required tests
            exactly as swebench does.

    Returns:
        bool: ``True`` if all required tests passed under the selected rule,
            ``False`` if there are no required tests or any required test did not
            pass.
    """
    required = list(fail_to_pass) + list(pass_to_pass)
    if not required:
        return False
    if eval_type == "fail_only":
        sm = status_map or {}
        # Mirror swebench's check_fail_only: a required test is a failure only when
        # present in the status map AND explicitly FAILED; anything else is success.
        return all(not (test in sm and sm[test] == "FAILED") for test in required)
    if status_map is not None:
        # Mirror swebench's check_pass_and_fail + get_resolution_status: a required
        # test is a failure only when it is absent or its status is FAILED/ERROR;
        # PASSED/XFAIL are successes and any other status (SKIPPED/XPASS) is neutral
        # (excluded). Resolution requires zero failures in BOTH categories.
        return all(not (test not in status_map or status_map[test] in ("FAILED", "ERROR")) for test in required)
    passed_set = set(passed)
    return all(test in passed_set for test in required)


def reward_from_report(report: SweEvalReport) -> float:
    """Map a graded report to a reward.

    An infra or eval failure (``error_kind`` set) yields ``0.0`` and is masked
    via the flag downstream; the result is always a ``float`` and never ``None``.

    Args:
        report (SweEvalReport): The graded result to convert.

    Returns:
        float: ``1.0`` if the task resolved with no error, otherwise ``0.0``.
    """
    if report.error_kind is not None:
        return 0.0
    return 1.0 if report.resolved else 0.0
