# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional file-level rollout health evidence, separate from suite statistics."""

from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.comparison.suite.schema import SuiteHealthSummary


def check_rollout_file_health(path: Path, *, workers: int | None = None) -> SuiteHealthSummary:
    """Check this exact JSONL and its own sibling inventory and failures, without changing source artifacts.

    This checks all agents in the file. It does not resolve merged-run shards, reuse
    saved quality reports or failure limits, or read separately configured inventories.
    Reports produced by the public health checker are discarded with the temporary directory.
    """
    from nemo_gym.rollout_health import run_health_checks

    try:
        with TemporaryDirectory(prefix="gym-suite-health-") as directory:
            result = run_health_checks(path, output_dir=Path(directory), workers=workers)
    except (OSError, ValueError) as error:
        return SuiteHealthSummary(unavailable_reason=" ".join(str(error).split()))

    run = result.summary["run"]
    artifacts = run["artifacts"]
    return SuiteHealthSummary(
        record_count=artifacts["records"],
        **run["verdicts"],
        issues={check: count for check, count in run["issues"].items() if count > 0},
        run_healthy=artifacts["healthy"],
        run_unhealthy_reasons=artifacts["unhealthy_reasons"],
    )


def rollout_health_warning(summary: SuiteHealthSummary) -> str | None:
    """Return at most one line for a source with unhealthy, missing, or unobserved evidence."""
    prefix = "Rollout file health: "
    if summary.unavailable_reason is not None:
        return (
            prefix
            + f"unavailable ({summary.unavailable_reason}); check that the rollout file and sidecars are readable."
        )
    if summary.record_count == 0:
        return prefix + "the rollout file is empty; no trajectory health could be checked."
    if not (summary.unhealthy or summary.unobserved or summary.issues or summary.run_healthy is False):
        return None

    parts = [
        f"{summary.healthy} healthy, {summary.unhealthy} unhealthy, "
        f"{summary.unobserved} unobserved of {summary.record_count} records"
        + (" (unobserved means insufficient evidence)" if summary.unobserved else "")
    ]
    if summary.issues:
        issues = sorted(summary.issues.items(), key=lambda item: (-item[1], item[0]))
        preview = ", ".join(f"{check}={count}" for check, count in issues[:3])
        if len(issues) > 3:
            preview += f", +{len(issues) - 3} more check types"
        parts.append(f"checks: {preview}")
    if summary.run_unhealthy_reasons:
        parts.append("run: " + ", ".join(summary.run_unhealthy_reasons))
    return prefix + "; ".join(parts) + "."
