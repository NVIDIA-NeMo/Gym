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


"""What `gym job` prints: text for a person to read.

Every function takes data and returns a string, and imports nothing that talks to a scheduler, so what a
command prints can be checked without a cluster. The commands that call them are in `job.py`.
"""

import yaml

from nemo_gym.orchestration.jobs import (
    FailureCategory,
    JobInfo,
    JobState,
    JobStatus,
    SubmissionRecord,
    SubmissionStatus,
)


def render_list(records: list[SubmissionRecord]) -> str:
    if not records:
        return "No submissions in this machine's job index."
    rows = [
        (
            record.gym_job_id,
            _time(record.submitted_at),
            record.cluster,
            record.executor,
            ", ".join(job.benchmark for job in record.benchmarks),
        )
        for record in records
    ]
    return _table(("ID", "SUBMITTED", "CLUSTER", "EXECUTOR", "BENCHMARKS"), rows)


def render_status(results: list[SubmissionStatus], *, note: str | None = None) -> str:
    if not results:
        return "No submissions to show."
    blocks = [render_status_block(result) for result in results]
    if note:
        blocks.append(note)
    return "\n\n".join(blocks)


def render_cancel(results: list[JobStatus]) -> str:
    return "\n".join(f"{result.benchmark}: {_cancel_outcome(result)}" for result in results)


def render_info(described: JobInfo, *, show_config: bool) -> str:
    record = described.record
    lines = [
        f"Submission  {record.gym_job_id}",
        f"Executor    {record.executor}, {record.cluster}",
        f"Submitted   {_time(record.submitted_at)} by {record.submitted_by}",
        f"Gym         {record.gym_version}" + (f" (commit {record.gym_commit[:9]})" if record.gym_commit else ""),
        f"Run dir     {record.run_dir}",
    ]
    for benchmark in described.benchmarks:
        newest = benchmark.job_ids[-1] if benchmark.job_ids else None
        lines += ["", benchmark.benchmark, f"  Job dir   {benchmark.job_dir}"]
        lines.append(f"  Jobs      {', '.join(benchmark.job_ids) if benchmark.job_ids else 'none recorded'}")
        if benchmark.log_sources:
            rows = [
                (source.name, source.instance or "-", source.location.replace("{job_id}", newest or "{job_id}"))
                for source in benchmark.log_sources
            ]
            table = _table(("SOURCE", "INSTANCE", "LOCATION"), rows).splitlines()
            lines.append(f"  Logs      {table[0]}")
            lines += [f"            {row}" for row in table[1:]]
        else:
            lines.append("  Logs      no log sources recorded (submitted before log sources were tracked)")
    lines.append("")
    if described.config is None:
        lines.append(f"Config      not available: {described.config_unavailable}")
    elif show_config:
        lines += ["Config", yaml.safe_dump(described.config, sort_keys=False).rstrip()]
    else:
        lines.append("Config      not shown; use --config to print the resolved config")
    return "\n".join(lines)


def render_status_block(result: SubmissionStatus) -> str:
    record = result.record
    counts = ", ".join(f"{count} {state.value}" for state, count in result.counts.items())
    lines = [f"{record.gym_job_id}  {record.cluster}  {result.state.value}" + (f" ({counts})" if counts else "")]
    if not result.jobs:
        return lines[0] + "\n  (no benchmarks)"

    recorded = {job.benchmark: job.job_id for job in record.benchmarks}
    rows = []
    for job in result.jobs:
        job_id = job.job_id or "-"
        if recorded.get(job.benchmark) and job.job_id and recorded[job.benchmark] != job.job_id:
            job_id = f"{job.job_id} (record: {recorded[job.benchmark]})"
        rows.append((job.benchmark, job.state.value, job_id, job.detail or "-"))
    table = _table(("BENCHMARK", "STATE", "JOB ID", "DETAIL"), rows).splitlines()
    lines.append(f"  {table[0]}")
    indent = " " * (2 + max(len("BENCHMARK"), *(len(row[0]) for row in rows)) + 2)
    for job, row in zip(result.jobs, table[1:], strict=True):
        lines.append(f"  {row}")
        if job.failure is not None:
            lines.append(f"{indent}{job.failure.message}")
            if job.failure.logs:
                lines.append(f"{indent}logs: {job.failure.logs}")
    return "\n".join(lines)


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in (header, *rows)
    )


def _time(submitted_at: str) -> str:
    return submitted_at.replace("T", " ")


def could_not_stop(result: JobStatus) -> bool:
    return result.failure is not None and result.failure.category is FailureCategory.CANCELLED


def _cancel_outcome(result: JobStatus) -> str:
    if could_not_stop(result):
        return f"could not be stopped: {result.failure.message}"
    if result.state is JobState.CANCELLED:
        return "cancelled"
    if result.state in (JobState.QUEUED, JobState.RUNNING):
        return f"cancel requested; still {result.state.value} (the scheduler can take a moment to report CANCELLED)"
    if result.state is JobState.UNKNOWN:
        return "cancel requested, but its state could not be read afterwards"
    return f"already finished ({result.state.value}); nothing was cancelled"
