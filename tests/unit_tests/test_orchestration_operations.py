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
from nemo_gym.orchestration.jobs import BenchmarkJob, LogSource, SubmissionRecord


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
