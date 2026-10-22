# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-collection analyses over persisted evidence, independent of rollout upload."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from nemo_gym.exporters import get_exporters
from nemo_gym.standard_analyses.export import write_report
from nemo_gym.standard_analyses.inputs import from_gym
from nemo_gym.standard_analyses.models import Rollout
from nemo_gym.standard_analyses.suite import analyze


def analyze_collection(output: Path) -> Path:
    """Write per-cohort reports beside the rollouts and export to active backends.

    Parse one raw row at a time and retain only the compact analysis records.
    Distinct agent/model/benchmark cohorts never share repeat baselines or metrics.
    """
    groups: dict[tuple, list[Rollout]] = defaultdict(list)
    with output.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                trajectory = row.get("ng_trajectory") or {}
                row["model"] = row.get("model") or "unknown"
                row["agent_ref"] = row.get("agent_ref") or row.get("_ng_agent_name") or "unknown"
                task = row.get("task_id")
                taskset = task.get("taskset") if isinstance(task, dict) else None
                benchmark = str(
                    taskset
                    or row.get("task_source")
                    or row.get("_ng_environment_server")
                    or trajectory.get("environment_id")
                    or "collection"
                )
                sampling = {
                    k: v
                    for k, v in (row.get("responses_create_params") or {}).items()
                    if k not in ("input", "metadata", "instructions", "tools")
                }
                config_id = hashlib.sha256(json.dumps(sampling, sort_keys=True).encode()).hexdigest()[:12]
                record = from_gym(row, run_id=output.stem, benchmark=benchmark, config_id=config_id)
                groups[record.cohort].append(record)
    directory = output.with_name(output.stem + "_analyses")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cohort, records in sorted(groups.items()):
        cohort_id = hashlib.sha256(json.dumps(cohort).encode()).hexdigest()[:12]
        namespace = "efficiency" if len(groups) == 1 else f"efficiency/cohorts/{cohort_id}"
        report = analyze(records)
        write_report(report, directory / cohort_id)
        for exporter in get_exporters():
            exporter.export_standard_analyses(report, namespace=namespace)
        manifest.append({"cohort": report.cohort, "directory": cohort_id, "namespace": namespace})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory
