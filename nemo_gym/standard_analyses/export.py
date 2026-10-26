# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact local results and optional W&B export to an existing run handle."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import orjson


if TYPE_CHECKING:
    from wandb import Run

from .models import Report


def write_report(report: Report, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_bytes(orjson.dumps(report.to_dict(), option=orjson.OPT_INDENT_2))


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
    scalars = report.scalars(namespace=namespace)
    run.summary.update(scalars)
    # ScalarChart consumes history metric names, so publish the computed scalars
    # in the same history step as the tables as well as in run.summary.
    statuses, payload = [], dict(scalars)
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
    run.log(payload)
