# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scenario expectations over the public gym eval health-check engine."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from nemo_gym.health.types import CheckInput, CheckScope, Verdict
from nemo_gym.rollout_health import CHECK_REGISTRY, run_health_checks

from .checks import HealthCheck
from .results import Results


def inspect_health(
    rollout_paths: Sequence[Path],
    *,
    output: Path,
    expectations: Mapping[str, Verdict],
    steps: bool = True,
) -> list[dict]:
    """Run health once on saved artifacts and compare each rollout's check results.

    Registry entries default to healthy; scenarios explicitly declare exceptions.
    Task-level checks need repeated tasks and are outside this scenario contract.
    Missing records fail even when an unobserved health result was expected.
    """
    specs = [spec for spec in CHECK_REGISTRY if spec.evaluation_scope == CheckScope.ROLLOUT]
    unknown = expectations.keys() - {spec.id for spec in specs}
    if unknown:
        raise ValueError(f"unknown rollout health expectations: {', '.join(sorted(unknown))}")
    if any(value not in ("healthy", "unhealthy", "unobserved") for value in expectations.values()):
        raise ValueError("health expectations must be healthy, unhealthy or unobserved")
    report = run_health_checks(rollout_paths, output_dir=output, workers=1) if rollout_paths else None
    coverage = report.summary["run"]["artifacts"]["coverage"] if report else {}
    results = Results()
    for spec in specs:
        expected = expectations.get(spec.id, "healthy")
        # A terminal rejection has no model-driven decision; respect the same
        # explicit step scope used by the artifact checks, never infer it from data.
        applies = steps or CheckInput.AGENT_TURNS not in spec.reads
        for index, digest in enumerate((report.rollouts if report else []) or [None]):
            actual: Verdict | None = None
            if digest is not None and spec.id in coverage and not coverage[spec.id]["ignored"]:
                actual = (
                    "unobserved"
                    if spec.id in digest.unobserved
                    else "unhealthy"
                    if any(finding.check == spec.id for finding in digest.findings)
                    else "healthy"
                )
            results.run(
                HealthCheck(
                    id=f"health.{spec.id}",
                    tier="P0",
                    evidence=(),
                    location=f"{output / 'rollout_verdicts.jsonl'}:{index + 1}",
                    reason=f"expected {expected}; observed {actual or 'no health result'}",
                    expected=expected,
                    actual=actual,
                    applies=applies,
                    available=actual is not None,
                )
            )
    return results.dump()
