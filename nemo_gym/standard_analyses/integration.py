# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-health analyses over persisted evidence, independent of rollout upload."""

import asyncio
import logging
from collections import Counter
from pathlib import Path

import orjson

from nemo_gym.exporters import export_standard_analyses
from nemo_gym.global_config import rollout_run_key
from nemo_gym.health.types import HealthCheckResult
from nemo_gym.path_utils import aggregate_metrics_path_for
from nemo_gym.standard_analyses.export import write_report
from nemo_gym.standard_analyses.inputs import _digest, read_records
from nemo_gym.standard_analyses.models import Report
from nemo_gym.standard_analyses.suite import analyze


logger = logging.getLogger(__name__)


def _reports(
    output: Path, paths: list[Path], health_result: HealthCheckResult | None
) -> tuple[Path | None, list[tuple[Report, str]]]:
    groups = read_records(paths, run_id=output.stem, health_result=health_result)
    # Observability is opt-in. A default run must not publish empty diagnostic tables.
    if not any(r.calls for records in groups.values() for r in records):
        return None, []
    aggregate_path = aggregate_metrics_path_for(output)
    aggregates = orjson.loads(aggregate_path.read_bytes()) if aggregate_path.exists() else []
    aggregates_by_run = {rollout_run_key(entry): entry.get("agent_metrics", {}) for entry in aggregates}
    cohorts_per_run = Counter(key for records in groups.values() for key in {r.run_key for r in records})
    directory = output.with_name(output.stem + "_analyses")
    directory.mkdir(parents=True, exist_ok=True)
    manifest, reports = [], []
    for cohort, records in sorted(groups.items()):
        cohort_id = _digest(cohort)
        namespace = "efficiency" if len(groups) == 1 else f"efficiency/cohorts/{cohort_id}"
        run_keys = {r.run_key for r in records}
        run_key = next(iter(run_keys)) if len(run_keys) == 1 else None
        metrics = aggregates_by_run.get(run_key) if run_key is not None and cohorts_per_run[run_key] == 1 else None
        report = analyze(records, aggregate_metrics=metrics)
        write_report(report, directory / cohort_id)
        reports.append((report, namespace))
        manifest.append({"cohort": report.cohort, "directory": cohort_id, "namespace": namespace})
    (directory / "manifest.json").write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    return directory, reports


async def analyze_collection(
    output: Path, *, paths: list[Path] | None = None, health_result: HealthCheckResult | None = None
) -> Path | None:
    """Write cohort reports and export on the caller's thread, after health checks.

    Shard paths must match the health check's input order; no merged JSONL is required.
    Analysis failures are isolated from the collection's results and completion status.
    """
    try:
        directory, reports = await asyncio.to_thread(_reports, output, paths or [output], health_result)
        for report, namespace in reports:
            export_standard_analyses(report, namespace=namespace)
        return directory
    except Exception:
        logger.exception("Standard analyses failed; collected rollout artifacts are still available.")
        return None
