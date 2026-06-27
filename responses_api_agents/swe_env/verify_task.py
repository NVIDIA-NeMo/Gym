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

"""Verification orchestrator for the SWE environment.

Grades an agent's patch inline as a library call (used by anyswe and the
``swe_env`` self-driving harness) — there is no separate verifier server. Runs a
fresh-only sequence via ``acquire_sandbox`` (always-teardown), bounded by a
per-call eval timeout. A genuine wall-clock eval timeout is masked as a typed
``error_kind`` (reward 0.0); other non-timeout eval-stage failures are reported
unmasked as ``resolved=False`` (reward 0.0, kept in the gradient) to mirror
main's app.py, rather than raising.

Every eval spec is stamped with a ``ttl_s`` so TTL-honoring backends (such as
opensandbox) self-expire orphaned sandboxes.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping
from typing import Any

# Importing this package registers the swe_env harnesses; the docker/apptainer
# providers are built into nemo_gym.sandbox and resolve lazily (no import needed).
import responses_api_agents.swe_env.harnesses  # noqa: F401
from nemo_gym.sandbox import SandboxCreateError, SandboxProvider
from responses_api_agents.swe_env.harness import (
    GraderDependencyError,
    SweEvalReport,
    SweTask,
    get_harness,
    reward_from_report,
)
from responses_api_agents.swe_env.sandbox import acquire_sandbox


#: Slack added to the eval timeout when stamping a sandbox TTL (covers spin-up +
#: teardown so a TTL-honoring backend does not expire a still-running eval).
_TTL_SLACK_S = 600.0


class ProviderCapabilityError(RuntimeError):
    """Raised when a task's harness does not support the configured provider."""


def _provider_name(provider: Mapping[str, Any] | SandboxProvider) -> str:
    """Return the provider's name.

    Args:
        provider: Either a single-key provider mapping or a ``SandboxProvider``
            instance.

    Returns:
        str: The provider name, or ``"?"`` if it cannot be determined.
    """
    if isinstance(provider, Mapping):
        return next(iter(provider), "?")
    return getattr(provider, "name", "?")


async def verify_task(
    provider: Mapping[str, Any] | SandboxProvider,
    task: SweTask,
    *,
    run_golden: bool = False,
    eval_timeout_s: float | None = None,
) -> SweEvalReport:
    """Grade a task's patch in a fresh sandbox and return a report.

    Selects the harness for the task's benchmark, optionally substitutes the
    golden patch, then resets the repo, materializes the patch, runs the eval,
    and grades the artifacts. An empty patch short-circuits without spinning up
    a sandbox. A genuine wall-clock eval timeout is returned as a report carrying
    ``error_kind="eval_timeout"``; other non-timeout eval-stage failures are
    returned unmasked (``resolved=False``, ``error_kind=None``) to mirror main,
    rather than raised.

    Args:
        provider: Single-key provider mapping or ``SandboxProvider`` selecting
            the sandbox backend.
        task: The task whose patch is graded.
        run_golden: When True, grade the task's golden patch instead of the
            model patch.
        eval_timeout_s: Optional override for the per-call eval timeout in
            seconds; falls back to the task metadata or a default.

    Returns:
        SweEvalReport: The grading outcome. ``error_kind="eval_timeout"`` on a genuine
            wall-clock eval timeout and ``error_kind="sandbox"`` on a sandbox-create /
            image-pull failure (both masked); other non-timeout eval-stage failures are
            reported unmasked (``resolved=False``, ``error_kind=None``).

    Raises:
        ProviderCapabilityError: If the task's harness does not support the provider.
        GraderDependencyError: If a required grading dependency is unavailable for an
            instance the harness must grade exactly (propagated, not swallowed).
    """
    harness = get_harness(task.benchmark)
    if task.metadata.get("flat_eval"):
        # Grade host-side (flat) so nested families (swe-bench / r2e-gym) can be graded on
        # exec-only providers like docker; a no-op for already-flat families.
        harness = harness.with_flat_eval()

    if run_golden:
        task = dataclasses.replace(task, model_patch=task.metadata.get("golden_patch", ""))

    # Empty/falsy-patch fast path: skip eval spin-up entirely.
    if not (task.model_patch or "").strip():
        return SweEvalReport(instance_id=task.instance_id, patch_exists=False, resolved=False)

    provider_name = _provider_name(provider)
    if not harness.supports_provider(provider_name):
        raise ProviderCapabilityError(
            f"Harness {harness.name!r} does not support provider {provider_name!r} "
            f"(grade_strategy={harness.grade_strategy})"
        )

    spec = harness.build_spec(task)
    timeout = eval_timeout_s if eval_timeout_s is not None else float(task.metadata.get("eval_timeout_s", 1800))
    # Stamp a TTL so backends that honor it (opensandbox) self-expire an eval sandbox
    # orphaned by a hard crash. docker ignores ttl_s; its finally-teardown covers it.
    if spec.ttl_s is None:
        spec = dataclasses.replace(spec, ttl_s=timeout + _TTL_SLACK_S)

    try:
        async with acquire_sandbox(provider, spec, instance_id=task.instance_id) as env:

            async def _sequence() -> SweEvalReport:
                await harness.reset_repo(env, task)
                await harness.materialize(env, task)
                artifacts = await harness.run_eval(env, task)
                return harness.grade(task, artifacts)

            return await asyncio.wait_for(_sequence(), timeout=timeout)
    except GraderDependencyError:
        # A required grader dependency is missing (e.g. swebench for a SWE-bench instance).
        # Propagate rather than degrading to an unmasked reward-0 so the misconfiguration is
        # loud (a crash in the standalone path; every sample masked in the anyswe path) instead
        # of silently skewing the resolve rate.
        raise
    except (asyncio.TimeoutError, TimeoutError):
        # Genuine wall-clock eval timeout: mask via error_kind. This mirrors main's
        # app.py, which sets eval_timed_out (-> mask_sample) only when the final eval
        # elapsed time reaches the configured tests timeout.
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            error_kind="eval_timeout",
            tests_status={"timeout_s": timeout},
        )
    except SandboxCreateError as exc:
        # Infra failure provisioning the eval sandbox (e.g. a docker image-pull failure on the
        # pull-on-demand path). Mask it as a typed error_kind so an infra hiccup doesn't depress
        # the resolve rate or leak into the training signal — consistent with the harnesses'
        # sandbox/eval_error masking — rather than degrading to an unmasked reward-0.
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            error_kind="sandbox",
            tests_status={"sandbox_create_error": repr(exc)},
        )
    except Exception as exc:  # non-timeout eval-stage failure -> unmasked reward 0
        # A non-timeout eval-stage crash is NOT masked: main's app.py catches any eval
        # exception, returns no report file (resolved=False) and leaves eval_timed_out
        # False, so the sample stays in the gradient at reward 0. Returning
        # error_kind=None here keeps mask_sample aligned with main rather than masking
        # the infra crash (which main does not do).
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            resolved=False,
            error_kind=None,
            tests_status={"exception": repr(exc)},
        )


def report_to_reward(report: SweEvalReport) -> float:
    """Convert an eval report into a scalar reward.

    Args:
        report: The grading outcome to score.

    Returns:
        float: The reward derived from the report.
    """
    return reward_from_report(report)
