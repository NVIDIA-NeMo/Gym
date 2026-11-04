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

import inspect
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from nemo_gym.orchestration import operations
from nemo_gym.orchestration.executors.base import BaseExecutor, UnsupportedOperationError
from nemo_gym.orchestration.executors.capabilities import (
    Inspectable,
    LogReadable,
    Resubmittable,
    Terminable,
    require,
)
from nemo_gym.orchestration.jobs import (
    CHAIN_INCOMPLETE_FILE,
    EVAL_EXIT_FILE,
    BenchmarkJob,
    FailureCategory,
    JobKey,
    JobState,
    LogSource,
    ObservedStatus,
    SubmissionRecord,
)


def _record(*, log_sources: list[LogSource] | None = None) -> SubmissionRecord:
    return SubmissionRecord(
        gym_job_id="gym-job-20260909T100203Z-abc123",
        gym_version="0.6.0",
        submitted_at="2026-09-09T10:02:03Z",
        run_dir="/jobs/gym-job-20260909T100203Z-abc123",
        cluster="cluster",
        executor="slurm",
        submitted_by="someone",
        benchmarks=[
            BenchmarkJob(
                benchmark="gsm8k",
                job_id="12345",
                job_dir="/jobs/gym-job-20260909T100203Z-abc123/gsm8k",
                log_sources=log_sources or [],
            )
        ],
    )


class _OnlyRun(BaseExecutor):
    name = "only-run"

    def run(self, config, *, dry_run=False):
        return None


class _Inspector(_OnlyRun, Inspectable):
    def connection_key(self, record):
        return "host"

    def query_states(self, refs):
        return {}


# --- Capabilities --------------------------------------------------------------------------------------------


def test_a_new_executor_has_to_implement_only_run():
    assert BaseExecutor.__abstractmethods__ == {"run"}

    executor = _OnlyRun()

    assert not isinstance(executor, (Resubmittable, Inspectable, Terminable, LogReadable))


@pytest.mark.parametrize(
    ("capability", "implemented", "missing"),
    [
        (Inspectable, {"connection_key": lambda self, record: "host"}, "query_states"),
        (
            Terminable,
            {"connection_key": lambda self, record: "host", "query_states": lambda self, refs: {}},
            "terminate",
        ),
        (
            LogReadable,
            {"connection_key": lambda self, record: "host", "query_states": lambda self, refs: {}},
            "stream_logs",
        ),
        (Resubmittable, {}, "resubmit"),
    ],
)
def test_a_capability_cannot_be_half_implemented(capability, implemented, missing):
    half = type("Half", (_OnlyRun, capability), implemented)

    with pytest.raises(TypeError, match=missing):
        half()


def test_the_capabilities_that_read_state_build_on_inspectable():
    assert issubclass(Terminable, Inspectable)
    assert issubclass(LogReadable, Inspectable)
    assert not issubclass(Resubmittable, Inspectable)


def test_an_inspectable_executor_reads_no_run_files_and_reports_no_progress_unless_it_says_otherwise():
    executor, record = _Inspector(), _record()

    assert executor.read_run_files(record, [Path("a"), Path("b")]) == {Path("a"): None, Path("b"): None}
    assert executor.read_progress(record, record.benchmarks[0]) is None


def test_require_hands_back_an_executor_that_has_the_capability():
    executor = _Inspector()

    assert require(executor, Inspectable, "status") is executor


def test_require_names_the_executor_and_the_operation_it_cannot_do():
    with pytest.raises(UnsupportedOperationError, match="only-run executor does not support cancel"):
        require(_OnlyRun(), Terminable, "cancel")


def test_require_falls_back_to_the_class_name_for_an_executor_without_a_name():
    with pytest.raises(UnsupportedOperationError, match="object executor does not support status"):
        require(object(), Inspectable, "status")


# --- Operations ----------------------------------------------------------------------------------------------


def _sources() -> list[LogSource]:
    return [
        LogSource(name="driver", kind="driver", location="logs/driver-{job_id}.log"),
        LogSource(name="vllm", kind="service", location="logs/vllm-{job_id}.log"),
    ]


def test_log_sources_come_from_the_record_and_are_a_copy():
    record = _record(log_sources=_sources())

    sources = operations.log_sources(record, "gsm8k")

    assert [(s.name, s.kind) for s in sources] == [("driver", "driver"), ("vllm", "service")]
    sources.clear()
    assert len(record.benchmarks[0].log_sources) == 2


def test_a_record_without_log_sources_has_none_to_list():
    assert operations.log_sources(_record(), "gsm8k") == []


def test_log_sources_of_an_unknown_benchmark_name_the_known_ones():
    with pytest.raises(ValueError, match="has no benchmark 'nope'; it has: gsm8k"):
        operations.log_sources(_record(), "nope")


def test_cancelling_a_whole_submission_needs_its_id_typed_as_a_keyword():
    confirm = inspect.signature(operations.cancel_submission).parameters["confirm"]

    assert confirm.kind is inspect.Parameter.KEYWORD_ONLY
    assert confirm.default is inspect.Parameter.empty


def test_operations_know_no_executor():
    """`operations.py` must work for any executor, so importing it must not load Slurm or Kubernetes.

    Checked in a SUBPROCESS on purpose: this test module imports executors itself, so asserting against the
    already-loaded sys.modules here would pass no matter what operations.py does.
    """
    # `script_templates` is a shared helper that `api.py` already imports; the rest are the interfaces.
    probe = (
        "import sys;"
        "import nemo_gym.orchestration.operations;"
        "known = {'base', 'capabilities', 'script_templates'};"
        "loaded = {m.split('.')[-1] for m in sys.modules if m.startswith('nemo_gym.orchestration.executors.')};"
        "print(','.join(sorted(loaded - known)))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "", f"operations.py pulled in executor modules: {result.stdout.strip()}"


# --- status ----------------------------------------------------------------------------------------------------


def _submission(
    gym_job_id: str, *, hostname: str | None = None, benchmarks=("gsm8k",), error=None
) -> SubmissionRecord:
    return SubmissionRecord(
        gym_job_id=gym_job_id,
        gym_version="0",
        submitted_at="2026-01-01T00:00:00Z",
        run_dir=f"/jobs/{gym_job_id}",
        cluster="cluster",
        executor="slurm",
        hostname=hostname,
        submitted_by="me",
        benchmarks=[
            BenchmarkJob(
                benchmark=name,
                job_dir=f"/jobs/{gym_job_id}/{name}",
                job_id=None if error else f"{index}00",
                error=error,
            )
            for index, name in enumerate(benchmarks, start=1)
        ],
    )


class _Scheduler(Inspectable):
    """Answers from a table, and keeps how it was asked."""

    def __init__(self, answers=None, *, unreachable=(), run_files=None):
        self.answers = answers or {}
        self.unreachable = unreachable
        self.run_files = run_files
        self.calls: list[list[JobKey]] = []

    def connection_key(self, record):
        return record.hostname or "local"

    def query_states(self, refs):
        self.calls.append([ref.key for ref in refs])
        host = refs[0].record.hostname
        if host in self.unreachable:
            raise ConnectionError(f"cannot reach {host}")
        default = ObservedStatus(state=JobState.SUCCEEDED)
        return {ref.key: self.answers.get(ref.key, default) for ref in refs}

    def read_run_files(self, record, paths):
        if isinstance(self.run_files, Exception):
            raise self.run_files
        return {path: self.run_files for path in paths}


def test_status_asks_once_per_connection_and_not_once_per_record():
    records = [
        _submission("a", hostname="login-1"),
        _submission("b", hostname="login-2"),
        _submission("c", hostname="login-1"),
        _submission("d", hostname="login-2"),
        _submission("e", hostname="login-1"),
    ]
    scheduler = _Scheduler()

    operations.status(scheduler, records)

    assert len(scheduler.calls) == 2
    asked = {tuple(sorted({gym_job_id for gym_job_id, _ in call})) for call in scheduler.calls}
    assert asked == {("a", "c", "e"), ("b", "d")}


def test_status_keeps_the_order_it_was_given():
    records = [_submission("c", hostname="h2"), _submission("a", hostname="h1"), _submission("b", hostname="h2")]

    results = operations.status(_Scheduler(), records)

    assert [result.record.gym_job_id for result in results] == ["c", "a", "b"]


def test_connections_are_asked_at_the_same_time():
    both_asked = threading.Barrier(2, timeout=5)

    class _Rendezvous(_Scheduler):
        def query_states(self, refs):
            both_asked.wait()  # only passes if the other connection is being asked at this moment
            return super().query_states(refs)

    results = operations.status(_Rendezvous(), [_submission("a", hostname="h1"), _submission("b", hostname="h2")])

    assert [result.state for result in results] == [JobState.SUCCEEDED, JobState.SUCCEEDED]


def test_an_unreachable_connection_is_unknown_and_hides_nothing_else():
    records = [_submission("up", hostname="h1"), _submission("down", hostname="h2")]

    up, down = operations.status(_Scheduler(unreachable=("h2",)), records)

    assert up.state is JobState.SUCCEEDED
    assert down.state is JobState.UNKNOWN
    assert "ConnectionError: cannot reach h2" in down.jobs[0].detail


def test_a_job_the_executor_did_not_report_is_unknown():
    class _Forgetful(_Scheduler):
        def query_states(self, refs):
            return {}

    (result,) = operations.status(_Forgetful(), [_submission("a")])

    assert result.jobs[0].state is JobState.UNKNOWN
    assert "did not report" in result.jobs[0].detail


def test_status_of_nothing_asks_nothing():
    scheduler = _Scheduler()

    assert operations.status(scheduler, []) == []
    assert scheduler.calls == []


def test_a_benchmark_a_submission_lacks_fails_the_whole_call_before_anything_is_asked():
    records = [_submission("has", benchmarks=("gpqa", "ruler")), _submission("lacks", benchmarks=("ruler",))]
    scheduler = _Scheduler()

    with pytest.raises(ValueError, match=r"lacks has no benchmark 'gpqa' \(it has: ruler\)") as error:
        operations.status(scheduler, records, ["gpqa"])

    assert "Nothing was queried" in str(error.value)
    assert scheduler.calls == []


def test_only_the_named_benchmarks_are_asked_about_and_rolled_up():
    records = [_submission("a", benchmarks=("gpqa", "ruler", "ifbench"))]
    scheduler = _Scheduler({("a", "ruler"): ObservedStatus(state=JobState.RUNNING)})

    (result,) = operations.status(scheduler, records, ["gpqa", "ifbench"])

    assert scheduler.calls == [[("a", "gpqa"), ("a", "ifbench")]]
    assert [job.benchmark for job in result.jobs] == ["gpqa", "ifbench"]
    assert result.state is JobState.SUCCEEDED
    assert result.counts == {JobState.SUCCEEDED: 2}


@pytest.mark.parametrize(
    ("states", "rolled_up"),
    [
        ([JobState.SUCCEEDED, JobState.RUNNING, JobState.FAILED], JobState.RUNNING),
        ([JobState.SUCCEEDED, JobState.QUEUED, JobState.FAILED], JobState.QUEUED),
        ([JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED], JobState.FAILED),
        ([JobState.SUCCEEDED, JobState.CANCELLED], JobState.CANCELLED),
        ([JobState.SUCCEEDED, JobState.SUCCEEDED], JobState.SUCCEEDED),
        ([JobState.SUCCEEDED, JobState.UNKNOWN], JobState.UNKNOWN),
    ],
)
def test_a_submission_rolls_its_benchmarks_up_to_one_state_and_counts_them(states, rolled_up):
    names = tuple(f"b{index}" for index in range(len(states)))
    answers = {("a", name): ObservedStatus(state=state) for name, state in zip(names, states, strict=True)}

    (result,) = operations.status(_Scheduler(answers), [_submission("a", benchmarks=names)])

    assert result.state is rolled_up
    assert sum(result.counts.values()) == len(states)
    assert all(count == states.count(state) for state, count in result.counts.items())


def _status_of(observed: ObservedStatus, record: SubmissionRecord | None = None):
    record = record or _submission("a")
    (result,) = operations.status(_Scheduler({("a", "gsm8k"): observed}), [record])
    return result.jobs[0]


def test_a_running_job_is_not_corrected_by_files():
    job = _status_of(ObservedStatus(state=JobState.RUNNING, files={EVAL_EXIT_FILE: "1\n"}))

    assert job.state is JobState.RUNNING
    assert job.failure is None


def test_a_chain_that_stopped_early_is_a_failure_with_the_reason_it_wrote():
    observed = ObservedStatus(
        state=JobState.FAILED,
        detail="FAILED",
        files={CHAIN_INCOMPLETE_FILE: "Chain stopped at the walltime cap; resubmit to continue.\n"},
    )

    job = _status_of(observed)

    assert job.state is JobState.FAILED
    assert job.failure.category is FailureCategory.INCOMPLETE
    assert job.failure.message == "Chain stopped at the walltime cap; resubmit to continue."
    assert job.failure.logs == "gym job logs a gsm8k --tail 100"


def test_the_exit_code_of_the_run_overrules_the_scheduler_and_a_stopped_chain_overrules_the_exit_code():
    wrong_exit = _status_of(ObservedStatus(state=JobState.SUCCEEDED, files={EVAL_EXIT_FILE: "1\n"}))
    both = _status_of(
        ObservedStatus(state=JobState.SUCCEEDED, files={EVAL_EXIT_FILE: "0", CHAIN_INCOMPLETE_FILE: "stopped"})
    )

    assert (wrong_exit.state, wrong_exit.exit_code, wrong_exit.failure.category) == (
        JobState.FAILED,
        1,
        FailureCategory.RUN,
    )
    assert (both.state, both.failure.category) == (JobState.FAILED, FailureCategory.INCOMPLETE)


def test_a_file_decides_the_outcome_when_the_scheduler_no_longer_remembers_the_job():
    job = _status_of(ObservedStatus(state=JobState.UNKNOWN, files={EVAL_EXIT_FILE: "0\n"}))

    assert job.state is JobState.SUCCEEDED
    assert job.exit_code == 0


@pytest.mark.parametrize("files", [{EVAL_EXIT_FILE: "0"}, {CHAIN_INCOMPLETE_FILE: "stopped"}])
def test_a_cancelled_job_is_never_overridden_by_a_file_left_by_an_earlier_attempt(files):
    assert _status_of(ObservedStatus(state=JobState.CANCELLED, files=files)).state is JobState.CANCELLED


def test_an_exit_file_that_is_not_a_number_is_ignored():
    job = _status_of(ObservedStatus(state=JobState.SUCCEEDED, files={EVAL_EXIT_FILE: "not a number"}))

    assert job.state is JobState.SUCCEEDED


def test_a_failure_the_scheduler_reports_carries_its_reason_and_where_it_came_from():
    job = _status_of(
        ObservedStatus(state=JobState.FAILED, detail="TIMEOUT", reason="The Slurm job ended as TIMEOUT.", exit_code=0)
    )

    assert job.failure.category is FailureCategory.SCHEDULER
    assert (job.failure.message, job.failure.source) == ("The Slurm job ended as TIMEOUT.", "scheduler")


def test_a_benchmark_that_was_never_queued_is_an_enqueue_failure_with_the_recorded_error():
    record = _submission("a", error="sbatch: error: bad account")

    job = _status_of(ObservedStatus(state=JobState.FAILED, detail="sbatch: error: bad account"), record)

    assert job.failure.category is FailureCategory.ENQUEUE
    assert (job.failure.message, job.failure.source) == ("sbatch: error: bad account", "record")


def test_the_job_id_shown_is_the_one_observed_and_falls_back_to_the_recorded_one():
    resumed = _status_of(ObservedStatus(state=JobState.RUNNING, job_id="777"))
    unseen = _status_of(ObservedStatus(state=JobState.UNKNOWN))

    assert resumed.job_id == "777"
    assert unseen.job_id == "100"


# --- info ------------------------------------------------------------------------------------------------------


def test_info_lists_the_observed_job_ids_and_the_planned_log_sources():
    record = _record(log_sources=_sources())
    answers = {(record.gym_job_id, "gsm8k"): ObservedStatus(state=JobState.RUNNING, job_ids=["12345", "12400"])}

    described = operations.info(_Scheduler(answers), record)

    (benchmark,) = described.benchmarks
    assert benchmark.job_ids == ["12345", "12400"]
    assert [source.name for source in benchmark.log_sources] == ["driver", "vllm"]
    assert benchmark.job_dir == record.benchmarks[0].job_dir


def test_info_shows_the_recorded_job_id_when_the_scheduler_cannot_be_reached():
    record = _record()

    described = operations.info(_Scheduler(unreachable=(None,)), record)

    assert described.benchmarks[0].job_ids == ["12345"]


def test_info_of_one_benchmark_names_the_known_ones_when_it_is_wrong():
    with pytest.raises(ValueError, match="has no benchmark 'nope'; it has: gsm8k"):
        operations.info(_Scheduler(), _record(), "nope")


def test_info_reads_the_resolved_config_through_the_executor():
    described = operations.info(_Scheduler(run_files="driver:\n  container: gym:latest\n"), _record())

    assert described.config == {"driver": {"container": "gym:latest"}}
    assert described.config_unavailable is None


@pytest.mark.parametrize(
    ("run_files", "reason"),
    [
        (None, "not readable by this executor"),
        ("a: [unclosed", "is not valid YAML"),
        ("- just\n- a list\n", "does not hold a mapping"),
        (OSError("permission denied"), "permission denied"),
    ],
)
def test_info_says_why_the_config_is_not_available(run_files, reason):
    described = operations.info(_Scheduler(run_files=run_files), _record())

    assert described.config is None
    assert reason in described.config_unavailable
