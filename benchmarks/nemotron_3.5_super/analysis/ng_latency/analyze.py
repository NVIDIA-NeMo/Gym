# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline analysis of a run directory: derived metrics per trial, run summary, HTML timelines."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import orjson

from ng_latency.build import _identity
from ng_latency.check import _load_timelines
from ng_latency.derive import DEFINITIONS, FLAGS, derive_trial
from ng_latency.paths import METRICS_DIR_NAME, locate, metrics_dir
from ng_latency.spans import TIMELINE_SCHEMA_VERSION
from ng_latency.summarize import summarize
from ng_latency.viz import render_index_html, render_trial_html


TIMELINE_NAME = "rollouts_timeline.jsonl"
_BUILD_HINT = "run `python -m nemo_gym.latency.timeline build` there first"


def _replica_dirs(run_dir: Path, rollouts_name: str) -> list[Path]:
    """Directories that hold a run: every directory with ``rollouts_name`` plus every directory with a
    timeline (at its root or under ``metrics/``)."""
    dirs = {p.parent for p in run_dir.rglob(rollouts_name)}
    for timeline in run_dir.rglob(TIMELINE_NAME):
        dirs.add(timeline.parent.parent if timeline.parent.name == METRICS_DIR_NAME else timeline.parent)
    return sorted(dirs)


def discover_timelines(
    run_dir: Path, *, rollouts_name: str = "rollouts.jsonl"
) -> list[tuple[Path, Path, Optional[Path]]]:
    """``(replica_dir, timeline_path, rollouts_path)`` per run under ``run_dir`` (or ``run_dir`` itself)."""
    run_dir = Path(run_dir)
    own = locate(run_dir, TIMELINE_NAME)
    if own.exists():
        rollouts = run_dir / rollouts_name
        return [(run_dir, own, rollouts if rollouts.exists() else None)]
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} is not a directory")
    found: list[tuple[Path, Path, Optional[Path]]] = []
    for replica_dir in _replica_dirs(run_dir, rollouts_name):
        timeline = locate(replica_dir, TIMELINE_NAME)
        if not timeline.exists():
            raise FileNotFoundError(f"{replica_dir} has {rollouts_name} but no {TIMELINE_NAME}; {_BUILD_HINT}")
        rollouts = replica_dir / rollouts_name
        found.append((replica_dir, timeline, rollouts if rollouts.exists() else None))
    if not found:
        raise FileNotFoundError(f"no {TIMELINE_NAME} under {run_dir}; {_BUILD_HINT}")
    return found


def _rows_by_rollout_id(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if path is None:
        return rows
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                row = orjson.loads(line)
                rows[_identity(row)["rollout_id"]] = row
    return rows


def timeline_filename(replica: str, rollout_id: str) -> str:
    def safe(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "_", text)

    return f"{safe(replica)}__{safe(rollout_id)}.html"


def run_analysis(run_dir: Path, *, timelines: bool = True, rollouts_name: str = "rollouts.jsonl") -> dict[str, Any]:
    run_dir = Path(run_dir)
    found = discover_timelines(run_dir, rollouts_name=rollouts_name)
    trials: list[dict[str, Any]] = []
    records: dict[tuple[str, str], dict[str, Any]] = {}
    inputs: list[str] = []
    coverage: list[dict[str, Any]] = []
    for replica_dir, timeline_path, rollouts_path in found:
        replica = "." if replica_dir == run_dir else str(replica_dir.relative_to(run_dir))
        inputs.append(str(timeline_path.relative_to(run_dir)))
        rows = _rows_by_rollout_id(rollouts_path)
        timelines_here = _load_timelines(timeline_path)
        coverage.append(
            {
                "input": inputs[-1],
                "rows": len(rows),
                "records": len(timelines_here),
                "rows_without_record": len(set(rows) - set(timelines_here)),
                "records_without_row": len(set(timelines_here) - set(rows)),
            }
        )
        for rollout_id, record in sorted(timelines_here.items()):
            row = rows.get(rollout_id)
            trial = derive_trial(record, row, replica=replica)
            if row is None:
                trial["gaps"].append({"code": "row_missing"})
            trials.append(trial)
            records[(replica, rollout_id)] = record
    trials.sort(key=lambda t: (str(t["replica"]), str(t["rollout_id"])))
    totals = {
        key: sum(c[key] for c in coverage) for key in ("rows", "records", "rows_without_record", "records_without_row")
    }
    summary = {**summarize(trials), "coverage": totals}
    header = {
        "kind": "header",
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "flags": FLAGS,
        "definitions": DEFINITIONS,
        "inputs": inputs,
        "coverage": coverage,
    }
    outputs: list[Path] = []
    out_dir = metrics_dir(run_dir)
    metrics_path = out_dir / "derived_metrics.jsonl"
    with metrics_path.open("wb") as handle:
        handle.write(orjson.dumps(header, option=orjson.OPT_SORT_KEYS) + b"\n")
        for trial in trials:
            handle.write(orjson.dumps(trial, option=orjson.OPT_SORT_KEYS) + b"\n")
    outputs.append(metrics_path)
    summary_path = out_dir / "derived_summary.json"
    summary_path.write_text(
        json.dumps({**summary, "inputs": inputs}, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    outputs.append(summary_path)
    if timelines:
        folder = out_dir / "timelines"
        folder.mkdir(exist_ok=True)
        links: dict[str, str] = {}
        for trial in trials:
            name = timeline_filename(str(trial["replica"]), str(trial["rollout_id"]))
            links[f"{trial['replica']}/{trial['rollout_id']}"] = name
            page = render_trial_html(records[(trial["replica"], trial["rollout_id"])], trial)
            (folder / name).write_text(page, encoding="utf-8")
            outputs.append(folder / name)
        (folder / "index.html").write_text(render_index_html(trials, links), encoding="utf-8")
        outputs.append(folder / "index.html")
    return {"summary": summary, "trials": len(trials), "outputs": [str(p) for p in outputs]}
