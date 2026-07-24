# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the full synthetic tool-use Gym datasets.

Writes each dataset to a temporary JSONL/report path and atomically replaces the
final paths only after that dataset finishes.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resources_servers.synthetic_tool_use_simulation.scripts.build_synthetic_tool_use_dataset import (
    DEFAULT_POLICY_TEMPERATURE,
    build_sample_dataset,
    default_source_dirs_from_env,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class BuildJob:
    key: str
    dataset_name: str
    source_indexes: tuple[int, ...]
    source_names: tuple[str, ...]
    parallel_tool_calls: bool


JOBS: tuple[BuildJob, ...] = (
    BuildJob(
        key="simple",
        dataset_name="260625_nemotron_synthetic_tool_use_conversational_simple",
        source_indexes=(0,),
        source_names=("260625_nemotron_synthetic_tool_use_conversational_simple",),
        parallel_tool_calls=False,
    ),
    BuildJob(
        key="proactive",
        dataset_name="260625_nemotron_synthetic_tool_use_conversational_proactive",
        source_indexes=(1,),
        source_names=("260625_nemotron_synthetic_tool_use_conversational_proactive",),
        parallel_tool_calls=False,
    ),
    BuildJob(
        key="combined",
        dataset_name="260625_nemotron_synthetic_tool_use_conversational_combined",
        source_indexes=(0, 1),
        source_names=(
            "260625_nemotron_synthetic_tool_use_conversational_simple",
            "260625_nemotron_synthetic_tool_use_conversational_proactive",
        ),
        parallel_tool_calls=False,
    ),
    BuildJob(
        key="simple_parallel",
        dataset_name="260625_nemotron_synthetic_tool_use_conversational_simple_parallel_tool_calls",
        source_indexes=(0,),
        source_names=("260625_nemotron_synthetic_tool_use_conversational_simple",),
        parallel_tool_calls=True,
    ),
    BuildJob(
        key="proactive_parallel",
        dataset_name="260625_nemotron_synthetic_tool_use_conversational_proactive_parallel_tool_calls",
        source_indexes=(1,),
        source_names=("260625_nemotron_synthetic_tool_use_conversational_proactive",),
        parallel_tool_calls=True,
    ),
    BuildJob(
        key="combined_parallel",
        dataset_name="260625_nemotron_synthetic_tool_use_conversational_combined_parallel_tool_calls",
        source_indexes=(0, 1),
        source_names=(
            "260625_nemotron_synthetic_tool_use_conversational_simple",
            "260625_nemotron_synthetic_tool_use_conversational_proactive",
        ),
        parallel_tool_calls=True,
    ),
)


def build_job(
    job: BuildJob,
    data_dir: Path,
    skip_existing: bool,
    policy_temperature: float = DEFAULT_POLICY_TEMPERATURE,
) -> dict[str, Any]:
    final_output = data_dir / f"{job.dataset_name}.jsonl"
    final_report = data_dir / f"{job.dataset_name}.report.json"
    if skip_existing and final_output.is_file() and final_report.is_file():
        return {
            "job": job.key,
            "dataset_name": job.dataset_name,
            "skipped": True,
            "output_path": str(final_output),
            "report_path": str(final_report),
        }

    tmp_output = data_dir / f"{job.dataset_name}.jsonl.tmp"
    tmp_report = data_dir / f"{job.dataset_name}.report.json.tmp"
    default_source_dirs = default_source_dirs_from_env()
    source_dirs = [default_source_dirs[index] for index in job.source_indexes]
    report = build_sample_dataset(
        source_dirs=source_dirs,
        output_path=tmp_output,
        report_path=tmp_report,
        max_rows=None,
        dataset_name=job.dataset_name,
        source_names=list(job.source_names),
        max_rows_per_domain=None,
        scan_domains_per_source=None,
        parallel_tool_calls=job.parallel_tool_calls,
        policy_temperature=policy_temperature,
    )
    tmp_output.replace(final_output)
    tmp_report.replace(final_report)
    return {
        "job": job.key,
        "dataset_name": job.dataset_name,
        "skipped": False,
        "rows_written": report["rows_written"],
        "parallel_tool_calls": job.parallel_tool_calls,
        "output_path": str(final_output),
        "report_path": str(final_report),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=["all"],
        choices=["all"] + [job.key for job in JOBS],
        help="Datasets to build. Use all, or a subset such as simple proactive simple_parallel.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of dataset builds to run concurrently. Use 2 for moderate parallelism.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose final JSONL and report already exist.",
    )
    parser.add_argument("--policy-temperature", type=float, default=DEFAULT_POLICY_TEMPERATURE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(JOBS) if "all" in args.jobs else [job for job in JOBS if job.key in set(args.jobs)]
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.workers == 1:
        for job in selected:
            print(f"START {job.key} parallel={job.parallel_tool_calls}", flush=True)
            print(
                json.dumps(build_job(job, args.data_dir, args.skip_existing, args.policy_temperature), indent=2),
                flush=True,
            )
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(build_job, job, args.data_dir, args.skip_existing, args.policy_temperature): job
            for job in selected
        }
        for future in as_completed(futures):
            job = futures[future]
            print(f"DONE {job.key}", flush=True)
            print(json.dumps(future.result(), indent=2), flush=True)


if __name__ == "__main__":
    main()
