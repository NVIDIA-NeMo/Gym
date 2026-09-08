# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the full conversational tool-use Gym datasets.

Writes each dataset to a temporary JSONL/report path and atomically replaces the
final paths only after that dataset finishes.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resources_servers.conversational_tool_use_simulation.scripts.build_conversational_tool_use_dataset import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_AGENT_NAME,
    DatasetMetadataConfig,
    GenerationProfile,
    build_sample_dataset,
    dataset_file_stats,
    default_source_dirs_from_env,
    source_tree_fingerprint,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class BuildJob:
    key: str
    dataset_name: str
    source_indexes: tuple[int, ...]
    source_names: tuple[str, ...]
    source_profiles: tuple[GenerationProfile, ...]
    parallel_tool_calls: bool


JOBS: tuple[BuildJob, ...] = (
    BuildJob(
        key="general",
        dataset_name="conversational_tool_use_general",
        source_indexes=(0,),
        source_names=("conversational_tool_use_general",),
        source_profiles=("general",),
        parallel_tool_calls=False,
    ),
    BuildJob(
        key="proactive",
        dataset_name="conversational_tool_use_proactive",
        source_indexes=(1,),
        source_names=("conversational_tool_use_proactive",),
        source_profiles=("proactive",),
        parallel_tool_calls=False,
    ),
    BuildJob(
        key="combined",
        dataset_name="conversational_tool_use_combined",
        source_indexes=(0, 1),
        source_names=(
            "conversational_tool_use_general",
            "conversational_tool_use_proactive",
        ),
        source_profiles=("general", "proactive"),
        parallel_tool_calls=False,
    ),
    BuildJob(
        key="general_parallel",
        dataset_name="conversational_tool_use_general_parallel_tool_calls",
        source_indexes=(0,),
        source_names=("conversational_tool_use_general",),
        source_profiles=("general",),
        parallel_tool_calls=True,
    ),
    BuildJob(
        key="proactive_parallel",
        dataset_name="conversational_tool_use_proactive_parallel_tool_calls",
        source_indexes=(1,),
        source_names=("conversational_tool_use_proactive",),
        source_profiles=("proactive",),
        parallel_tool_calls=True,
    ),
    BuildJob(
        key="combined_parallel",
        dataset_name="conversational_tool_use_combined_parallel_tool_calls",
        source_indexes=(0, 1),
        source_names=(
            "conversational_tool_use_general",
            "conversational_tool_use_proactive",
        ),
        source_profiles=("general", "proactive"),
        parallel_tool_calls=True,
    ),
)


def build_job(
    job: BuildJob,
    data_dir: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    final_output = data_dir / f"{job.dataset_name}.jsonl"
    final_report = data_dir / f"{job.dataset_name}.report.json"
    if skip_existing and _existing_output_is_current(job, final_output, final_report):
        return {
            "job": job.key,
            "dataset_name": job.dataset_name,
            "skipped": True,
            "output_path": str(final_output),
            "report_path": str(final_report),
        }

    source_dirs = default_source_dirs_from_env(job.source_indexes)
    report = build_sample_dataset(
        source_dirs=source_dirs,
        output_path=final_output,
        report_path=final_report,
        max_rows=None,
        dataset_name=job.dataset_name,
        source_names=list(job.source_names),
        source_profiles=list(job.source_profiles),
        max_rows_per_domain=None,
        scan_domains_per_source=None,
        parallel_tool_calls=job.parallel_tool_calls,
    )
    return {
        "job": job.key,
        "dataset_name": job.dataset_name,
        "skipped": False,
        "rows_written": report["rows_written"],
        "parallel_tool_calls": job.parallel_tool_calls,
        "output_path": str(final_output),
        "report_path": str(final_report),
    }


def _existing_output_is_current(
    job: BuildJob,
    output_path: Path,
    report_path: Path,
) -> bool:
    if not output_path.is_file() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        output_stats = dataset_file_stats(output_path, validate_rows=True)
        source_dirs = default_source_dirs_from_env(job.source_indexes)
        source_fingerprints = [source_tree_fingerprint(source_dir) for source_dir in source_dirs]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    expected_metadata = DatasetMetadataConfig(dataset_name=job.dataset_name).to_dict()
    return (
        report.get("dataset_schema_version") == DATASET_SCHEMA_VERSION
        and report.get("metadata") == expected_metadata
        and report.get("source_names") == list(job.source_names)
        and report.get("source_profiles") == list(job.source_profiles)
        and report.get("source_fingerprints") == source_fingerprints
        and report.get("parallel_tool_calls") is job.parallel_tool_calls
        and report.get("max_rows") is None
        and report.get("max_rows_per_domain") is None
        and report.get("scan_domains_per_source") is None
        and report.get("agent_name") == DEFAULT_AGENT_NAME
        and report.get("rows_written") == output_stats.rows
        and report.get("output_size_bytes") == output_stats.size_bytes
        and report.get("output_sha256") == output_stats.sha256
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=["all"],
        choices=["all"] + [job.key for job in JOBS],
        help="Datasets to build. Use all, or a subset such as general proactive general_parallel.",
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
        help="Skip jobs whose JSONL and report match the current build contract.",
    )
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
                json.dumps(build_job(job, args.data_dir, args.skip_existing), indent=2),
                flush=True,
            )
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build_job, job, args.data_dir, args.skip_existing): job for job in selected}
        for future in as_completed(futures):
            job = futures[future]
            print(f"DONE {job.key}", flush=True)
            print(json.dumps(future.result(), indent=2), flush=True)


if __name__ == "__main__":
    main()
