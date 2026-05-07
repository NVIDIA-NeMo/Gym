#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Self-contained CVDP report module for NeMo-Gym.
# Combines build_collateral(), combine_reports(), and re-exports Report.
# No dependency on the CVDP benchmark repo.

import datetime
import json
import os
from pathlib import Path
from typing import List

from cvdp_lib.cvdp_report_lib import Report, auto_generate_text_report  # noqa: F401


# ---------------------------------------------------------------------------
# build_collateral  (recovered from git: 28e6dd1c)
# ---------------------------------------------------------------------------

_VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def _parse_task_id(task_id: str):
    """
    'cvdp_copilot_foo_bar_0042' -> (base='cvdp_copilot_foo_bar', num=42)
    Matches the per-task directory layout used by cvdp_benchmark.
    """
    parts = task_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return task_id, 0


def _prompt_to_markdown(responses_create_params: dict) -> str:
    """
    Convert responses_create_params.input messages into a markdown prompt file,
    matching cvdp_benchmark's prompts/{id}.md format.
    """
    lines = []
    for msg in (responses_create_params or {}).get("input") or []:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role:
            lines.append(f"## {role.capitalize()}\n")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def build_collateral(rollouts_path: str, output_dir: Path) -> dict:
    """
    Parse rollouts.jsonl and write all per-task collateral files.

    Handles num_repeats>1: each repeat gets its own report file
    ({task_base}/reports/{num}_{repeat}.txt) and all repeats are
    aggregated into the tests list in raw_result.

    Returns the raw_result dict (with log paths filled in) ready for
    CVDP's report.Report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_result = {}
    prompt_response = {}
    run_log_lines = ["=== NeMo-Gym CVDP Rollout Log ===\n"]
    repeat_counts = {}
    skipped = 0

    with open(rollouts_path) as f:
        for line in f:
            if not line.strip():
                continue
            rollout = json.loads(line)

            task_id = rollout.get("task_id")
            if not task_id:
                continue

            category = rollout.get("category")
            difficulty = rollout.get("difficulty")

            if not category:
                print(f"Warning: skipping {task_id} -- missing category")
                skipped += 1
                continue
            if difficulty not in _VALID_DIFFICULTIES:
                print(f"Warning: skipping {task_id} -- invalid difficulty {difficulty!r}")
                skipped += 1
                continue

            exit_code = rollout.get("container_exit_code")
            exec_time = rollout.get("execution_time") or 0.0
            extracted_rtl = rollout.get("extracted_rtl") or {}
            rcp = rollout.get("responses_create_params") or {}
            container_services = rollout.get("container_services")

            result = 0 if exit_code == 0 else 1
            task_base, task_num = _parse_task_id(task_id)
            repeat_idx = repeat_counts.get(task_id, 0)
            repeat_counts[task_id] = repeat_idx + 1

            # Per-task prompt: {task_base}/prompts/{num}.md  (write once)
            if repeat_idx == 0:
                prompt_dir = output_dir / task_base / "prompts"
                prompt_dir.mkdir(parents=True, exist_ok=True)
                (prompt_dir / f"{task_num}.md").write_text(_prompt_to_markdown(rcp), encoding="utf-8")

            # Per-service report files + test entries
            report_dir = output_dir / task_base / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)

            if container_services:
                test_entries = []
                for svc in container_services:
                    svc_name = svc.get("service", "unknown")
                    svc_exit = svc.get("exit_code", 1)
                    svc_stderr = svc.get("stderr") or ""
                    report_file = report_dir / f"{task_num}_{svc_name}_{repeat_idx}.txt"
                    report_file.write_text(svc_stderr, encoding="utf-8")
                    test_entries.append(
                        {
                            "result": 0 if svc_exit == 0 else 1,
                            "execution": exec_time / len(container_services),
                            "log": str(report_file),
                        }
                    )
            else:
                stderr = rollout.get("container_stderr") or ""
                report_file = report_dir / f"{task_num}_{repeat_idx}.txt"
                report_file.write_text(stderr, encoding="utf-8")
                test_entries = [{"result": result, "execution": exec_time, "log": str(report_file)}]

            # raw_result entry
            rollout_errors = sum(e["result"] for e in test_entries)
            if task_id not in raw_result:
                raw_result[task_id] = {
                    "category": category,
                    "difficulty": difficulty,
                    "tests": test_entries,
                    "errors": rollout_errors,
                }
            else:
                raw_result[task_id]["tests"].extend(test_entries)
                raw_result[task_id]["errors"] += rollout_errors

            # prompt_response entry
            if task_id not in prompt_response or result == 0:
                prompt_response[task_id] = {
                    "input": {},
                    "output": extracted_rtl,
                    "obj": result == 0,
                }

            # run.log line
            status = "PASS" if result == 0 else "FAIL"
            run_log_lines.append(
                f"{task_id}  repeat={repeat_idx}  [{status}]  exit={exit_code}  time={exec_time:.2f}s"
            )

    # Write top-level collateral
    (output_dir / "prompt_response.jsonl").write_text(json.dumps(prompt_response, indent=2) + "\n", encoding="utf-8")
    (output_dir / "run.log").write_text("\n".join(run_log_lines) + "\n", encoding="utf-8")

    if skipped:
        print(f"Skipped {skipped} rollouts (missing category or difficulty)")

    return raw_result


# ---------------------------------------------------------------------------
# combine_reports  (copied from cvdp_benchmark/run_samples.py)
# ---------------------------------------------------------------------------


def extract_problem_id_from_test_id(test_id: str) -> str:
    """Extract problem ID from test ID, handling dots in problem IDs."""
    if "." not in test_id:
        return test_id
    return test_id.rsplit(".", 1)[0]


def combine_reports(sample_prefixes: List[str], output_prefix: str, n_samples: int, k_threshold: int) -> None:
    """Combine multiple report.json files into a composite report."""
    os.makedirs(output_prefix, exist_ok=True)

    composite_report = {
        "metadata": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": n_samples,
            "k_threshold": k_threshold,
            "sample_prefixes": sample_prefixes,
            "composite": True,
        },
        "samples": [],
    }

    for i, prefix in enumerate(sample_prefixes):
        report_path = os.path.join(prefix, "report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                try:
                    report = json.load(f)
                    report["sample_index"] = i

                    if "test_details" in report:
                        for test_list_key in ("passing_tests", "failing_tests"):
                            if test_list_key in report["test_details"]:
                                for test in report["test_details"][test_list_key]:
                                    if "test_id" in test and "difficulty" not in test:
                                        test_id = test["test_id"]
                                        problem_id = extract_problem_id_from_test_id(test_id)
                                        for category, cat_data in report.items():
                                            if category in ("metadata", "test_details", "sample_index"):
                                                continue
                                            for difficulty in ("easy", "medium", "hard"):
                                                if difficulty in cat_data and "problems" in cat_data[difficulty]:
                                                    if any(
                                                        p.get("id") == problem_id
                                                        for p in cat_data[difficulty]["problems"]
                                                    ):
                                                        test["difficulty"] = difficulty
                                                        break

                    composite_report["samples"].append(report)

                    if i == 0 and "metadata" in report:
                        for key, value in report["metadata"].items():
                            if key not in composite_report["metadata"]:
                                composite_report["metadata"][key] = value
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse report at {report_path}")
        else:
            print(f"Warning: Report not found at {report_path}")

    if not composite_report["samples"]:
        print("No valid samples found. Cannot create composite report.")
        return

    problem_ids = set()
    for sample in composite_report["samples"]:
        for category, cat_data in sample.items():
            if category in ("metadata", "sample_index", "test_details"):
                continue
            logs = cat_data.get("logs", [])
            for log in logs:
                if "id" in log:
                    problem_ids.add(log["id"])

    print(f"Found {len(problem_ids)} unique problems across {len(composite_report['samples'])} samples")

    for i, sample in enumerate(composite_report["samples"]):
        total_passed = 0
        total_problems = 0
        for category, cat_data in sample.items():
            if category in ("metadata", "sample_index", "test_details"):
                continue
            for difficulty in ("easy", "medium", "hard"):
                if difficulty in cat_data:
                    total_passed += cat_data[difficulty].get("Passed Problems", 0)
                    total_problems += cat_data[difficulty].get("Total Problems", 0)
        if total_problems > 0:
            pass_rate = (total_passed / total_problems) * 100
            print(f"Sample {i + 1}: {total_passed}/{total_problems} problems passed ({pass_rate:.2f}%)")

    output_path = os.path.join(output_prefix, "composite_report.json")
    with open(output_path, "w") as f:
        json.dump(composite_report, f, indent=2)

    print(f"\nComposite report written to {output_path}")
    auto_generate_text_report(output_path)
    print(f"Use run_reporter.py to analyze pass@{k_threshold} metrics")
