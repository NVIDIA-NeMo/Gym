# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_AGENT_NAME = "osprey_full_doc_benchmark_agent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize an Osprey full_doc NeMo Gym rollout or aggregate-metrics artifact."
    )
    parser.add_argument(
        "--aggregate-json",
        type=Path,
        default=None,
        help="Optional path to the Gym-written *_aggregate_metrics.json file.",
    )
    parser.add_argument(
        "--rollouts-jsonl",
        type=Path,
        default=None,
        help="Optional path to the rollout JSONL file written by ng_collect_rollouts.",
    )
    parser.add_argument(
        "--expected-input-jsonl",
        type=Path,
        default=None,
        help="Optional prepared benchmark JSONL for completion reporting.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional output path for the compact JSON summary.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=None,
        help="Optional output path for the Markdown report.",
    )
    parser.add_argument(
        "--agent-name",
        default=DEFAULT_AGENT_NAME,
        help="Agent name to summarize from aggregate metrics payloads.",
    )
    args = parser.parse_args()
    if args.aggregate_json is None and args.rollouts_jsonl is None:
        parser.error("at least one of --aggregate-json or --rollouts-jsonl is required")
    return args


def default_output_path(base_path: Path, suffix: str) -> Path:
    stem = base_path.name.removesuffix("_aggregate_metrics.json")
    if stem == base_path.name:
        stem = base_path.name.removesuffix(".jsonl")
    if stem == base_path.name:
        stem = base_path.stem
    return base_path.with_name(f"{stem}_{suffix}")


def load_agent_entry(aggregate_json: Path, agent_name: str) -> dict[str, Any]:
    payload = json.loads(aggregate_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list in {aggregate_json}")

    for entry in payload:
        if entry.get("agent_ref", {}).get("name") == agent_name:
            return entry

    available = sorted({entry.get("agent_ref", {}).get("name") for entry in payload if isinstance(entry, dict)})
    raise ValueError(f"agent {agent_name!r} not found in {aggregate_json}; available agents: {available}")


def build_aggregate_summary(
    aggregate_json: Path,
    entry: dict[str, Any],
    agent_name: str,
) -> dict[str, Any]:
    key_metrics = entry.get("key_metrics", {})
    agent_metrics = entry.get("agent_metrics", {})
    return {
        "mode": "aggregate_metrics",
        "agent_name": agent_name,
        "aggregate_json": str(aggregate_json),
        "num_tasks": len(entry.get("group_level_metrics", [])),
        "headline_metrics": {
            key: value
            for key, value in {
                "mean/reward": key_metrics.get("mean/reward", agent_metrics.get("mean/reward")),
                "mean/is_correct": key_metrics.get("mean/is_correct", agent_metrics.get("mean/is_correct")),
                "mean/input_tokens": key_metrics.get("mean/input_tokens", agent_metrics.get("mean/input_tokens")),
                "mean/output_tokens": key_metrics.get("mean/output_tokens", agent_metrics.get("mean/output_tokens")),
                "mean/total_tokens": key_metrics.get("mean/total_tokens", agent_metrics.get("mean/total_tokens")),
            }.items()
            if isinstance(value, (int, float))
        },
        "key_metrics": key_metrics,
        "agent_metrics": agent_metrics,
    }


def build_rollout_summary(
    rollouts_jsonl: Path,
    expected_input_jsonl: Path | None,
    agent_name: str,
    aggregate_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = 0
    correct = 0
    reward_sum = 0.0
    wrong_type_counts: Counter[str] = Counter()
    line_item_total_counts: Counter[str] = Counter()
    line_item_error_counts: Counter[str] = Counter()
    doc_total_counts: Counter[str] = Counter()
    doc_error_counts: Counter[str] = Counter()
    completed_task_indexes: list[int] = []

    with rollouts_jsonl.open("rt", encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            total += 1

            reward = row.get("reward")
            if isinstance(reward, (int, float)):
                reward_sum += float(reward)

            is_correct = bool(row.get("is_correct"))
            if is_correct:
                correct += 1
                wrong_type = "Correct"
            else:
                wrong_type = row.get("wrong_prediction_type") or "Unknown"

            wrong_type_counts[wrong_type] += 1

            line_item_name = row.get("line_item_name") or "<missing>"
            doc_name = row.get("doc_name") or "<missing>"
            line_item_total_counts[line_item_name] += 1
            doc_total_counts[doc_name] += 1
            if not is_correct:
                line_item_error_counts[line_item_name] += 1
                doc_error_counts[doc_name] += 1

            task_index = row.get("_ng_task_index")
            if isinstance(task_index, int):
                completed_task_indexes.append(task_index)

    headline_metrics: dict[str, Any] = {
        "mean/reward": reward_sum / total if total else 0.0,
        "exact_match_accuracy": correct / total if total else 0.0,
        "rollout_count/correct": correct,
        "rollout_count/incorrect": total - correct,
    }
    if aggregate_summary is not None:
        for key, value in aggregate_summary.get("headline_metrics", {}).items():
            if key not in headline_metrics and isinstance(value, (int, float)):
                headline_metrics[key] = value
    for label, count in sorted(wrong_type_counts.items()):
        key_label = label.lower().replace(" ", "_").replace("/", "_")
        headline_metrics[f"rollout_count/{key_label}"] = count
        headline_metrics[f"rollout_pct/{key_label}"] = (100.0 * count / total) if total else 0.0

    summary: dict[str, Any] = {
        "mode": "rollouts",
        "agent_name": agent_name,
        "rollouts_jsonl": str(rollouts_jsonl),
        "num_completed_rollouts": total,
        "correct_count": correct,
        "incorrect_count": total - correct,
        "headline_metrics": headline_metrics,
        "wrong_prediction_type_counts": dict(sorted(wrong_type_counts.items())),
        "top_error_line_items": [
            {"line_item_name": name, "errors": count, "total": line_item_total_counts[name]}
            for name, count in line_item_error_counts.most_common(10)
        ],
        "top_error_docs": [
            {"doc_name": name, "errors": count, "total": doc_total_counts[name]}
            for name, count in doc_error_counts.most_common(10)
        ],
    }
    if aggregate_summary is not None:
        summary["aggregate_json"] = aggregate_summary["aggregate_json"]

    if expected_input_jsonl is not None:
        expected_rows: list[dict[str, Any]] = []
        with expected_input_jsonl.open("rt", encoding="utf-8") as fin:
            for idx, line in enumerate(fin):
                row = json.loads(line)
                expected_rows.append(
                    {
                        "index": idx,
                        "id": row.get("id", f"row-{idx}"),
                        "doc_name": row.get("doc_name"),
                        "line_item_name": row.get("line_item_name"),
                    }
                )
        completed_idx_set = set(completed_task_indexes)
        missing_rows = [row for row in expected_rows if row["index"] not in completed_idx_set]
        summary["expected_input_jsonl"] = str(expected_input_jsonl)
        summary["expected_rollouts"] = len(expected_rows)
        summary["missing_rollouts"] = len(missing_rows)
        summary["completion_rate"] = (total / len(expected_rows)) if expected_rows else 0.0
        summary["missing_rows_preview"] = missing_rows[:20]

    return summary


def write_report(report_md: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Osprey Full Doc Eval Summary",
        "",
        f"- mode: `{summary['mode']}`",
        f"- agent: `{summary['agent_name']}`",
    ]
    if summary["mode"] == "aggregate_metrics":
        lines.extend(
            [
                f"- aggregate metrics: `{summary['aggregate_json']}`",
                f"- tasks: `{summary['num_tasks']}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- rollouts: `{summary['rollouts_jsonl']}`",
                f"- completed rollouts: `{summary['num_completed_rollouts']}`",
            ]
        )
        if "aggregate_json" in summary:
            lines.append(f"- aggregate metrics: `{summary['aggregate_json']}`")
        if "expected_rollouts" in summary:
            lines.extend(
                [
                    f"- expected rollouts: `{summary['expected_rollouts']}`",
                    f"- missing rollouts: `{summary['missing_rollouts']}`",
                    f"- completion rate: `{summary['completion_rate']:.6f}`",
                ]
            )

    lines.extend(
        [
            "",
            "## Headline Metrics",
            "",
        ]
    )
    for key in sorted(summary["headline_metrics"]):
        value = summary["headline_metrics"][key]
        if isinstance(value, float):
            lines.append(f"- `{key}`: `{value:.6f}`")
        else:
            lines.append(f"- `{key}`: `{value}`")

    if "wrong_prediction_type_counts" in summary:
        lines.extend(
            [
                "",
                "## Wrong Prediction Types",
                "",
            ]
        )
        for key, value in summary["wrong_prediction_type_counts"].items():
            lines.append(f"- `{key}`: `{value}`")

    if summary.get("top_error_line_items"):
        lines.extend(
            [
                "",
                "## Top Error Line Items",
                "",
            ]
        )
        for row in summary["top_error_line_items"]:
            lines.append(f"- `{row['line_item_name']}`: `{row['errors']}` errors out of `{row['total']}`")

    if summary.get("top_error_docs"):
        lines.extend(
            [
                "",
                "## Top Error Docs",
                "",
            ]
        )
        for row in summary["top_error_docs"]:
            lines.append(f"- `{row['doc_name']}`: `{row['errors']}` errors out of `{row['total']}`")

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    aggregate_summary: dict[str, Any] | None = None
    default_base = args.rollouts_jsonl or args.aggregate_json
    if default_base is None:
        raise AssertionError("parse_args should ensure at least one input is provided")

    if args.aggregate_json is not None:
        aggregate_json = args.aggregate_json.resolve()
        aggregate_entry = load_agent_entry(aggregate_json=aggregate_json, agent_name=args.agent_name)
        aggregate_summary = build_aggregate_summary(
            aggregate_json=aggregate_json,
            entry=aggregate_entry,
            agent_name=args.agent_name,
        )

    if args.rollouts_jsonl is not None:
        rollouts_jsonl = args.rollouts_jsonl.resolve()
        expected_input_jsonl = args.expected_input_jsonl.resolve() if args.expected_input_jsonl is not None else None
        summary = build_rollout_summary(
            rollouts_jsonl=rollouts_jsonl,
            expected_input_jsonl=expected_input_jsonl,
            agent_name=args.agent_name,
            aggregate_summary=aggregate_summary,
        )
        default_base = rollouts_jsonl
    else:
        summary = aggregate_summary
        if summary is None:
            raise AssertionError("aggregate summary should exist when rollouts are absent")

    summary_json = (args.summary_json or default_output_path(default_base.resolve(), "summary.json")).resolve()
    report_md = (args.report_md or default_output_path(default_base.resolve(), "report.md")).resolve()

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(report_md=report_md, summary=summary)

    print(f"Wrote {summary_json}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
