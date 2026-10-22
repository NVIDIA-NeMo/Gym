# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact local results and optional W&B export to an existing run handle."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from wandb import Run

from .models import Report


def write_report(report: Report, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report.to_dict(), indent=2, allow_nan=False)
    (output / "results.json").write_text(serialized + "\n")
    from .preview import write_preview

    write_preview(report, output / "report.html")


def log_wandb(report: Report, run: Run, *, namespace: str = "efficiency") -> None:
    """Uses an already-open W&B run. Does not read credentials or open a new run."""
    import wandb

    run.config.update(
        {
            (
                "standard_analyses_metadata"
                if namespace == "efficiency"
                else f"standard_analyses_metadata/{namespace}"
            ): {
                "schema_version": report.schema_version,
                "analysis_version": report.analysis_version,
                "cohort": report.cohort,
            }
        }
    )
    scalars = {key.replace("efficiency/", namespace + "/", 1): value for key, value in report.scalars().items()}
    run.summary.update(scalars)
    # ScalarChart consumes history metric names, so publish the computed scalars
    # in the same history step as the tables as well as in run.summary.
    statuses, payload = [], dict(scalars)
    by_name = {r.name: r for r in report.results}
    for result in report.results:
        statuses.append([result.name, result.status, result.reason, "\n".join(result.notes)])
        for name, table in result.tables.items():
            key = f"{namespace}/{result.name}/{name}"
            payload[key] = wandb.Table(columns=table.columns, data=table.rows)
            run.summary[f"{key}_total_rows"] = table.total_rows
            run.summary[f"{key}_truncated"] = table.truncated
    payload[f"{namespace}/availability"] = wandb.Table(
        columns=["analysis", "status", "reason", "notes"],
        data=statuses,
    )
    if "long_calls" in by_name:
        rows = by_name["long_calls"].tables["detail"].rows
        for column, label in ((6, "call_share"), (7, "time_share"), (8, "token_share")):
            data = [[f"{row[0]} >{row[1]}", row[column]] for row in rows if row[column] is not None]
            if data:
                table = wandb.Table(columns=["threshold", "fraction"], data=data)
                payload[f"{namespace}/long_calls/{label}"] = wandb.plot.bar(
                    table,
                    "threshold",
                    "fraction",
                    title=f"Long-call {label.replace('_', ' ')}",
                )
    if "repeat_excess" in by_name:
        rows = by_name["repeat_excess"].tables["detail"].rows
        data = [[f"{r[1]} / repeat {r[2]}", r[7]] for r in rows if r[0] == "all_scored"][:100]
        if data:
            table = wandb.Table(columns=["task_repeat", "excess_model_s"], data=data)
            payload[f"{namespace}/repeat_excess/chart"] = wandb.plot.bar(
                table,
                "task_repeat",
                "excess_model_s",
                title="Observed excess model seconds (up to 100 repeats)",
            )
    if "token_accumulation" in by_name:
        series = defaultdict(list)
        for row in by_name["token_accumulation"].tables["detail"].rows:
            if row[8] is not None:
                series[f"{row[0]} / repeat {row[1]}"].append((row[2], row[8]))
        keys = sorted(series)[:24]
        if keys:
            payload[f"{namespace}/token_accumulation/chart"] = wandb.plot.line_series(
                xs=[[point[0] for point in series[key]] for key in keys],
                ys=[[point[1] for point in series[key]] for key in keys],
                keys=keys,
                title="Cumulative output tokens (up to 24 trajectories)",
                xname="capture call index",
            )
    run.log(payload)
