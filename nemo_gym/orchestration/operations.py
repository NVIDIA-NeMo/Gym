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

"""What you do with a submission: status, info, cancel and logs.

This is where the policy lives -- guards, ordering, partial-failure handling, grouping and the mapping
to `JobState` -- so that an executor only answers questions about its own scheduler. Each function
takes an executor typed as the capability it needs (see `executors/capabilities.py`) and calls only
that capability's methods, so nothing here knows which executor it is talking to. This module imports
no executor: `submit.py`, which knows which executors exist, resolves one from a record and calls in.
"""

import logging
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from nemo_gym.orchestration.executors.capabilities import Inspectable, LogReadable, Terminable
from nemo_gym.orchestration.jobs import (
    CHAIN_INCOMPLETE_FILE,
    EVAL_EXIT_FILE,
    RESOLVED_CONFIG_NAME,
    BenchmarkInfo,
    BenchmarkJob,
    Failure,
    FailureCategory,
    JobInfo,
    JobKey,
    JobRef,
    JobState,
    JobStatus,
    LogSource,
    ObservedStatus,
    SubmissionRecord,
    SubmissionStatus,
)


logger = logging.getLogger(__name__)

# At most this many connections are asked at once; a status over many hosts would otherwise open them all.
_MAX_CONCURRENT_CONNECTIONS = 8

# A job in one of these states has finished, or the executor could not tell, so what the job wrote decides.
_DECIDED_BY_FILES = {JobState.FAILED, JobState.SUCCEEDED, JobState.UNKNOWN}

_NOT_REPORTED = ObservedStatus(state=JobState.UNKNOWN, detail="the executor did not report this job")


def status(
    executor: Inspectable, records: list[SubmissionRecord], benchmarks: list[str] | None = None
) -> list[SubmissionStatus]:
    """The live status of `records`, one `SubmissionStatus` each, in the order given.

    `benchmarks` limits every submission to those benchmarks; a submission that lacks one is an error,
    raised before anything is asked, listing each offender with the benchmarks it does have. Records are
    grouped by `connection_key` and `query_states` is called once per group, so the cost grows with the
    number of connections, not submissions; the groups run concurrently. A group whose connection fails
    yields `UNKNOWN` for its jobs and does not hide the others. The executor's `state` is reconciled with
    the files the job wrote, and `CANCELLED` is never overridden by a file. Read-only: it never writes the
    record.
    """
    selected = _select_benchmarks(records, benchmarks)
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(executor.connection_key(record), []).append(index)

    def ask(indices: list[int]) -> dict[JobKey, ObservedStatus]:
        refs = [JobRef(records[index], job) for index in indices for job in selected[index]]
        if not refs:
            return {}
        try:
            return executor.query_states(refs)
        except Exception as error:  # deliberate: one unreachable scheduler must not hide the others
            logger.warning("Could not query the scheduler for %d job(s): %s", len(refs), error)
            reason = f"{type(error).__name__}: {error}"
            return {ref.key: ObservedStatus(state=JobState.UNKNOWN, detail=reason) for ref in refs}

    observed: dict[JobKey, ObservedStatus] = {}
    if groups:
        with ThreadPoolExecutor(max_workers=min(len(groups), _MAX_CONCURRENT_CONNECTIONS)) as pool:
            for answered in pool.map(ask, groups.values()):
                observed.update(answered)

    results = []
    for record, jobs in zip(records, selected, strict=True):
        statuses = [
            _reconcile(record, job, observed.get((record.gym_job_id, job.benchmark), _NOT_REPORTED)) for job in jobs
        ]
        states = [job_status.state for job_status in statuses]
        counts = Counter(states)
        results.append(
            SubmissionStatus(
                record=record,
                jobs=statuses,
                state=_roll_up(states),
                counts={state: counts[state] for state in JobState if counts[state]},
            )
        )
    return results


def info(executor: Inspectable, record: SubmissionRecord, benchmark: str | None = None) -> JobInfo:
    """Describe `record`, or only `benchmark` of it: its directories, the job ids observed, its log
    sources and, when `read_run_files` can read it, the resolved config. Read-only.

    The observed job ids are best effort: when the scheduler cannot be reached the recorded ids are shown.
    """
    jobs = [record.benchmark_job(benchmark)] if benchmark is not None else list(record.benchmarks)
    try:
        observed = executor.query_states([JobRef(record, job) for job in jobs]) if jobs else {}
    except Exception as error:  # best effort: the description is useful without the live job ids
        logger.warning("Could not ask the scheduler for the job ids of %s: %s", record.gym_job_id, error)
        observed = {}

    described = []
    for job in jobs:
        seen = observed.get((record.gym_job_id, job.benchmark))
        job_ids = seen.job_ids if seen and seen.job_ids else ([job.job_id] if job.job_id else [])
        described.append(
            BenchmarkInfo(benchmark=job.benchmark, job_dir=job.job_dir, job_ids=job_ids, log_sources=job.log_sources)
        )
    config, unavailable = _read_config(executor, record)
    return JobInfo(record=record, benchmarks=described, config=config, config_unavailable=unavailable)


def cancel(executor: Terminable, record: SubmissionRecord, benchmark: str) -> JobStatus:
    """Stop `benchmark`'s job and return its status as observed afterwards.

    Re-queries the scheduler first, so it acts on the jobs that exist now and not on a stale record, and
    leaves a job that is already terminal alone. Calls `terminate` and writes nothing to the record. A
    refusal from the scheduler (permission denied, say) is raised for the caller to report against this
    benchmark.
    """
    raise NotImplementedError


def cancel_submission(executor: Terminable, record: SubmissionRecord, *, confirm: str) -> list[JobStatus]:
    """Stop every live benchmark of `record`, and return each one's status as observed afterwards.

    `confirm` has no default and must equal `record.gym_job_id`, or `ValueError` is raised: cancelling a
    whole submission by accident costs more than typing its id. A benchmark that cannot be stopped is
    reported on its own and the others proceed.
    """
    raise NotImplementedError


def logs(
    executor: LogReadable,
    record: SubmissionRecord,
    benchmark: str,
    *,
    source: str = "driver",
    instance: str | None = None,
    tail: int | None = None,
    follow: bool = False,
) -> Iterator[str]:
    """Yield the lines of one of `benchmark`'s logs.

    `source` and `instance` are resolved against `log_sources`, so `stream_logs` receives one `LogSource`
    and executors never validate names. A name that matches nothing, or a source with several instances
    and none chosen, raises `ValueError` listing what exists. The current job is found with
    `query_states`, and its id fills the location's `{job_id}`.
    """
    raise NotImplementedError


def log_sources(record: SubmissionRecord, benchmark: str) -> list[LogSource]:
    """The logs `benchmark`'s job was set up to write, as planned at submit time.

    Read from the record, so it needs no executor and costs no remote call. A record written before
    sources were planned has none. Raises `ValueError` naming the known benchmarks for one the record lacks.
    """
    return list(record.benchmark_job(benchmark).log_sources)


def _select_benchmarks(records: list[SubmissionRecord], benchmarks: list[str] | None) -> list[list[BenchmarkJob]]:
    """The jobs to ask about, per record; a record that lacks a named benchmark is an error for the whole call."""
    if benchmarks is None:
        return [list(record.benchmarks) for record in records]
    problems = []
    for record in records:
        have = [job.benchmark for job in record.benchmarks]
        absent = [name for name in benchmarks if name not in have]
        if absent:
            names = ", ".join(map(repr, absent))
            problems.append(f"{record.gym_job_id} has no benchmark {names} (it has: {', '.join(have)})")
    if problems:
        raise ValueError("; ".join(problems) + ". Nothing was queried.")
    wanted = set(benchmarks)
    return [[job for job in record.benchmarks if job.benchmark in wanted] for record in records]


def _reconcile(record: SubmissionRecord, job: BenchmarkJob, observed: ObservedStatus) -> JobStatus:
    """The status shown for `job`: the executor's state, corrected by what the job wrote.

    Active states stand. For a job that finished, or that the executor could not place, a chain that
    stopped early makes it `FAILED`, then the exit code of the run decides. `CANCELLED` is never
    overridden: a file left by an earlier attempt would otherwise mask a cancel.
    """
    state, exit_code, failure = observed.state, observed.exit_code, None
    logs_hint = f"gym job logs {record.gym_job_id} {job.benchmark} --tail 100"

    if state in _DECIDED_BY_FILES:
        incomplete = observed.files.get(CHAIN_INCOMPLETE_FILE)
        exit_text = observed.files.get(EVAL_EXIT_FILE)
        run_status = _as_int(exit_text) if exit_text is not None else None
        if incomplete is not None:
            state = JobState.FAILED
            failure = Failure(
                category=FailureCategory.INCOMPLETE,
                message=incomplete.strip() or "The chain of jobs stopped before the run finished.",
                exit_code=exit_code,
                source="job directory",
                logs=logs_hint,
            )
        elif run_status is not None:
            exit_code = run_status
            if run_status == 0:
                state = JobState.SUCCEEDED
            else:
                state = JobState.FAILED
                failure = Failure(
                    category=FailureCategory.RUN,
                    message=f"The run exited with status {run_status}.",
                    exit_code=run_status,
                    source="job directory",
                    logs=logs_hint,
                )

    if state is JobState.FAILED and failure is None:
        if job.job_id is None and job.error and not observed.job_id:
            failure = Failure(category=FailureCategory.ENQUEUE, message=job.error, source="record")
        else:
            failure = Failure(
                category=FailureCategory.SCHEDULER,
                message=observed.reason or observed.detail or "The job failed.",
                exit_code=exit_code,
                source="scheduler",
                logs=logs_hint,
            )

    return JobStatus(
        benchmark=job.benchmark,
        state=state,
        job_id=observed.job_id or job.job_id,
        detail=observed.detail,
        exit_code=exit_code,
        failure=failure,
    )


def _roll_up(states: list[JobState]) -> JobState:
    """One state for a submission: running wins, then queued, failed, cancelled, and only all-succeeded succeeds."""
    for state in (JobState.RUNNING, JobState.QUEUED, JobState.FAILED, JobState.CANCELLED):
        if state in states:
            return state
    if states and all(state is JobState.SUCCEEDED for state in states):
        return JobState.SUCCEEDED
    return JobState.UNKNOWN


def _as_int(text: str) -> int | None:
    try:
        return int(text.strip())
    except ValueError:
        return None


def _read_config(executor: Inspectable, record: SubmissionRecord) -> tuple[dict[str, object] | None, str | None]:
    """The resolved config of `record` and None, or None and why it is not available."""
    path = Path(record.run_dir) / RESOLVED_CONFIG_NAME
    try:
        text = executor.read_run_files(record, [path]).get(path)
    except Exception as error:  # best effort, like the rest of the description
        return None, f"the run directory could not be read: {error}"
    if text is None:
        return None, "the run directory is not readable by this executor"
    try:
        config = yaml.safe_load(text)
    except yaml.YAMLError as error:
        return None, f"{RESOLVED_CONFIG_NAME} is not valid YAML: {error}"
    if not isinstance(config, dict):
        return None, f"{RESOLVED_CONFIG_NAME} does not hold a mapping"
    return config, None
