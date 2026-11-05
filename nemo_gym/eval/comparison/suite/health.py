# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional file-level rollout health evidence, separate from suite statistics."""

import logging
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from nemo_gym.config.config_types import ConfigError
from nemo_gym.eval.comparison.suite.schema import SuiteHealthSummary


logger = logging.getLogger(__name__)
_HEALTH_PROGRESS_INTERVAL_SECONDS = 30


def _log_health_progress(path: Path, started: float) -> Callable[[str, int, int | None], None]:
    """Log phase changes and throttle record updates without introducing worker-side logging."""
    last_phase = None
    last_logged = started

    def report_progress(phase: str, completed: int, total: int | None) -> None:
        nonlocal last_phase, last_logged
        now = monotonic()
        if phase != last_phase or now - last_logged >= _HEALTH_PROGRESS_INTERVAL_SECONDS:
            count = f"{completed} rows" if total is None else f"{completed}/{total} records"
            logger.info(
                "Rollout file health progress: %s: %s (%s; %.0f seconds elapsed)",
                path,
                phase,
                count,
                now - started,
            )
            last_phase = phase
            last_logged = now

    return report_progress


def check_rollout_file_health(
    path: Path, *, workers: int | None = None, output_dir: Path | None = None
) -> SuiteHealthSummary:
    """Check this exact JSONL and its own sibling inventory and failures, without changing source artifacts.

    This checks all agents in the file, using selected completed attempts for managed
    histories and every saved record for loose files. It does not resolve merged-run
    shards, reuse saved quality reports or failure limits, or read separately configured inventories.
    Both failure thresholds are zero, flagging any unresolved failure when the expected
    rollout count is known from its run history or materialized-input sibling.
    ``output_dir`` retains the public checker's reports. If omitted, the reports are
    discarded with a temporary directory. Phase changes and periodic record progress
    are logged while the checker processes the file.
    """
    started = monotonic()
    logger.info("Checking rollout file health: %s", path)
    from nemo_gym.rollouts.health.types import RunHealthLimits
    from nemo_gym.rollouts.rollout_health import run_health_checks

    directory_context = (
        nullcontext(output_dir) if output_dir is not None else TemporaryDirectory(prefix="gym-suite-health-")
    )
    with directory_context as directory:
        try:
            result = run_health_checks(
                path,
                output_dir=Path(directory),
                workers=workers,
                limits=RunHealthLimits(max_not_final_failure_fraction=0, max_final_failure_fraction=0),
                limits_source="override",
                progress=_log_health_progress(path, started),
            )
        except (OSError, ValueError, TypeError, ConfigError) as error:
            reason = " ".join(str(error).split())
            logger.warning(
                "Rollout file health unavailable: %s (%.1f seconds elapsed): %s", path, monotonic() - started, reason
            )
            return SuiteHealthSummary(unavailable_reason=reason)

    run = result.summary["run"]
    artifacts = run["artifacts"]
    summary = SuiteHealthSummary(
        record_count=artifacts["records"],
        **run["verdicts"],
        failed=artifacts["failed"],
        issues={check: count for check, count in run["issues"].items() if count > 0},
        run_healthy=artifacts["healthy"],
        run_unhealthy_reasons=artifacts["unhealthy_reasons"],
        summary_path=result.summary_path.resolve() if output_dir is not None else None,
        verdicts_path=result.verdicts_path.resolve() if output_dir is not None else None,
    )

    logger.info(
        "Completed rollout file health: %s (%.1f seconds elapsed): %d healthy, %d unhealthy, "
        "%d unobserved of %d saved records; failed (no saved result): %s",
        path,
        monotonic() - started,
        summary.healthy,
        summary.unhealthy,
        summary.unobserved,
        summary.record_count,
        summary.failed.total if summary.failed is not None else "unknown",
    )
    return summary


def rollout_health_warning(summary: SuiteHealthSummary) -> str | None:
    """Return at most one line for a source with unhealthy, missing, or unobserved evidence."""
    prefix = "Rollout file health: "
    if summary.unavailable_reason is not None:
        return (
            prefix
            + f"unavailable ({summary.unavailable_reason}); check that the rollout file and sidecars are readable."
        )
    failed = summary.failed if summary.failed is not None and summary.failed.total > 0 else None
    if summary.record_count != 0 and not (
        summary.unhealthy or summary.unobserved or summary.issues or summary.run_healthy is False or failed
    ):
        return None

    if summary.record_count == 0:
        parts = ["the rollout file is empty; no trajectory health could be checked"]
    else:
        parts = [
            f"{summary.healthy} healthy, {summary.unhealthy} unhealthy, "
            f"{summary.unobserved} unobserved of {summary.record_count} records"
            + (" (unobserved means insufficient evidence)" if summary.unobserved else "")
        ]
    if failed is not None:
        details = f"{failed.final} final, {failed.not_final} retryable"
        if failed.by_class:
            details += "; " + ", ".join(f"{name}={count}" for name, count in sorted(failed.by_class.items()))
        noun = "rollout" if failed.total == 1 else "rollouts"
        parts.append(f"{failed.total} failed {noun} without saved results ({details})")
    if summary.issues:
        issues = sorted(summary.issues.items(), key=lambda item: (-item[1], item[0]))
        preview = ", ".join(f"{check}={count}" for check, count in issues[:3])
        if len(issues) > 3:
            preview += f", +{len(issues) - 3} more check types"
        parts.append(f"checks: {preview}")
    if summary.run_unhealthy_reasons:
        parts.append("run: " + ", ".join(summary.run_unhealthy_reasons))
    return prefix + "; ".join(parts) + "."
