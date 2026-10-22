# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One reusable project saved view; no analysis computation in panel definitions."""

from __future__ import annotations

from .suite import ANALYSES


def chart_specifications():
    """Views over existing exported data; no recomputation or run mutation."""

    def table(title, name, preset, **fields):
        return {"title": title, "table": name, "preset": preset, "fields": fields}

    def bars(title, *metrics):
        return {"title": title, "metrics": list(metrics)}

    return {
        "long_calls": [
            table(
                f"Long-call {label} share",
                f"long_calls/{key}_share_table",
                "bar",
                label="threshold",
                value="fraction",
            )
            for key, label in (("call", "count"), ("time", "model-time"), ("token", "token"))
        ],
        "repeat_excess": [
            table(
                "Observed excess model seconds (up to 100 repeats)",
                "repeat_excess/chart_table",
                "bar",
                label="task_repeat",
                value="excess_model_s",
            ),
        ],
        "effort_outliers": [
            table(
                "Rollout effort: output tokens vs model seconds",
                "effort_outliers/detail",
                "scatter",
                x="output_tokens",
                y="model_s",
            ),
        ],
        "token_accumulation": [
            table(
                "Cumulative output tokens (up to 24 trajectories)",
                "token_accumulation/chart_table",
                "lineseries",
                step="step",
                lineKey="lineKey",
                lineVal="lineVal",
            ),
        ],
        "wasted_turns": [
            bars(
                "Potential redundant actions and failed actions (separate counts)",
                "wasted_turns/flagged_actions",
                "wasted_turns/failed_actions",
            ),
        ],
        "turn_classification": [
            bars(
                "Tool-action phases (heuristic counts)",
                *[
                    f"turn_classification/{phase}_count"
                    for phase in ("exploration", "implementation", "verification", "unknown")
                ],
            ),
        ],
        "compaction_impact": [
            table(
                "Mean reasoning tokens per call: before vs after compaction",
                "compaction_impact/detail",
                "scatter",
                x="mean_reasoning_before",
                y="mean_reasoning_after",
            ),
        ],
        "pass_at_k": [
            table(
                "Estimated pass@k by task",
                "pass_at_k/detail",
                "line",
                x="k",
                y="pass_at_k",
                stroke="task_id",
            ),
        ],
        "system_flags": [
            bars(
                "Observed errors and slow calls (may overlap)",
                "system_flags/error_count",
                "system_flags/slow_call_count",
            ),
        ],
        "runtime_tails": [
            table(
                "Median rollout wall time by task (seconds)",
                "runtime_tails/detail",
                "bar",
                label="task_id",
                value="median_wall_s",
            ),
        ],
        "time_distribution": [
            bars(
                "Summed model and tool seconds (may overlap)",
                "time_distribution/model_s",
                "time_distribution/tool_s",
            ),
            table(
                "Per-rollout model vs tool seconds",
                "time_distribution/detail",
                "scatter",
                x="model_s",
                y="tool_s",
            ),
        ],
    }


def build_chart(spec, *, namespace: str = "efficiency"):
    import wandb_workspaces.reports.v2 as wr

    if "metrics" in spec:
        return wr.BarPlot(
            title=spec["title"],
            metrics=[namespace + "/" + metric for metric in spec["metrics"]],
            orientation="v",
        )
    panel = wr.CustomChart.from_table(
        table_name=namespace + "/" + spec["table"],
        chart_fields=spec["fields"],
        chart_strings={
            "title": spec["title"],
            **({"xname": "capture call index"} if spec["preset"] == "lineseries" else {}),
        },
    )
    panel.chart_name = f"wandb/{spec['preset']}/v0"
    return panel


def panel_specifications():
    return {
        "schema_version": "1.0",
        "charts": chart_specifications(),
        "name": "Standard analyses",
        "summary_metrics": [
            ("Long output >1k: model-time share", "long_calls/output_gt_1000_time_fraction"),
            ("Long output >10k: model-time share", "long_calls/output_gt_10000_time_fraction"),
            (
                "Observed excess / all comparable repeats",
                "repeat_excess/all_scored_excess_fraction",
            ),
            ("Observed excess / successful repeats", "repeat_excess/successful_excess_fraction"),
            ("pass@1", "pass_at_k/pass_at_1"),
            ("pass@4", "pass_at_k/pass_at_4"),
        ],
        "detail_tables": [f"efficiency/{fn.__name__}/detail" for fn in ANALYSES],
        "availability_table": "efficiency/availability",
    }


def build_workspace(entity: str, project: str, *, namespace: str = "efficiency"):
    """Build locally; .save() is an explicit external write performed by the CLI."""
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    spec = panel_specifications()
    sections = [
        ws.Section(
            name="Standard analyses: overview",
            is_open=True,
            panels=[
                wr.ScalarChart(title=title, metric=namespace + "/" + metric)
                for title, metric in spec["summary_metrics"]
            ],
        )
    ]
    sections.append(
        ws.Section(
            name="Standard analyses: coverage",
            is_open=True,
            panels=[
                wr.WeavePanelSummaryTable(table_name=namespace + "/availability"),
            ],
        )
    )
    for fn in ANALYSES:
        sections.append(
            ws.Section(
                name="Standard analyses: " + fn.__name__,
                is_open=True,
                panels=[
                    *[build_chart(chart, namespace=namespace) for chart in spec["charts"][fn.__name__]],
                    wr.WeavePanelSummaryTable(table_name=f"{namespace}/{fn.__name__}/detail"),
                ],
            )
        )
    return ws.Workspace(entity=entity, project=project, name=spec["name"], sections=sections)
