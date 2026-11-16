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
"""A Store reader that could not run decides nothing; one that ran and refused the artifacts still stops."""

import shlex
import sys

import pytest

from nemo_gym.orchestration import recovery
from nemo_gym.orchestration.api import ResumeConfig
from nemo_gym.orchestration.executors import slurm
from nemo_gym.orchestration.executors.connection import Connection, LocalConnection
from nemo_gym.orchestration.recovery import select_managed_evidence
from tests.unit_tests.orchestration.test_allocation_cut_recovery import _cut_run
from tests.unit_tests.orchestration.test_eval_resubmit import _handle, _resumable_run
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_managed_recovery_selection import collect, prepare_managed
from tests.unit_tests.orchestration.test_remote_managed_recovery import config
from tests.unit_tests.orchestration.test_remote_reader_transport import _collected, _install, _job
from tests.unit_tests.orchestration.test_resume_script import _run_prologue


CRASH = "remote Store reader unavailable: exited 1 after 3 s (the 900 s bound); stderr: error: install failed"
REFUSAL = "remote Store reader changed collection evidence"


# The source: what the reader reports says whether it ran.


def test_a_reader_whose_install_crashes_is_unavailable(tmp_path, monkeypatch):
    job = _job(tmp_path)
    _install(monkeypatch, "echo 'error: install failed' >&2; exit 1")
    result = slurm._remote_managed_evidence(LocalConnection(), _collected(job), job, config())
    assert (result.managed_selection, result.managed_reader_unavailable) == (None, True)


def test_a_reader_that_ran_and_refused_the_artifacts_is_not_unavailable(tmp_path, monkeypatch):
    job = _job(tmp_path)
    collected = _collected(job)
    changed = "__GYM_MANAGED:" + collected.model_copy(update={"resubmit_offset": 5}).model_dump_json()
    _install(monkeypatch, f"cat <&3 >/dev/null; echo {shlex.quote(changed)}")
    result = slurm._remote_managed_evidence(LocalConnection(), collected, job, config())
    assert REFUSAL in result.managed_reader_error and result.managed_reader_unavailable is False


def test_a_local_reader_that_cannot_import_the_store_is_unavailable(tmp_path, monkeypatch):
    job = tmp_path / "job"
    prepare_managed(job, "missing", n=2)
    monkeypatch.setitem(sys.modules, "nemo_gym.rollouts.rollout_store", None)
    result = select_managed_evidence(collect(job), job)
    assert (result.managed_selection, result.managed_reader_unavailable) == (None, True)


# The inspection and the recovery.


class _Remote(Connection):
    """The exact serialized commands run locally, behind the SSH capability boundary."""

    def run(self, commands):
        return LocalConnection().run(commands)

    def copy(self, local, remote):
        raise AssertionError("read only")

    def write_text(self, remote, content, *, private=False):
        raise AssertionError("no dispatch")


def _remote_reader(monkeypatch, *, unavailable: bool, error: str) -> None:
    monkeypatch.setattr(slurm, "get_connection", lambda hostname: _Remote())
    monkeypatch.setattr(
        slurm,
        "_remote_managed_evidence",
        lambda conn, collected, job_dir, config: collected.model_copy(
            update={
                "managed_selection": None,
                "managed_reader_error": error,
                "managed_reader_unavailable": unavailable,
            }
        ),
    )


def _managed_run(tmp_path):
    job, _, _ = _resumable_run(tmp_path)
    prepare_managed(job, "missing", n=2)
    return job


def test_an_unavailable_reader_decides_nothing_and_names_the_inspection_to_run_again(tmp_path, slurm_bin, monkeypatch):
    from nemo_gym.orchestration.submit import inspect_recovery, recover

    job = _managed_run(tmp_path)
    _remote_reader(monkeypatch, unavailable=True, error=CRASH)
    inspect = "gym eval resubmit {} {} --recover --inspect".format(*_handle(job))
    for out in (inspect_recovery(*_handle(job)), recover(*_handle(job))):
        assert (out.decision.action.value, out.ok, out.dispatched) == ("unknown", False, None)
        assert out.decision.scheduler_live is False
        reason = out.decision.reason
        assert "could not run" in reason and "nothing was decided" in reason and CRASH in reason
        assert "recovery stops" not in reason and reason.endswith(inspect)
    assert not (slurm_bin / "env").exists()


def test_an_unavailable_reader_while_a_chain_job_runs_is_live(tmp_path, slurm_bin, monkeypatch):
    from nemo_gym.orchestration.submit import inspect_recovery

    job = _managed_run(tmp_path)
    (slurm_bin / "accounting").write_text("100|FAILED|7200|0\n101|RUNNING|60|0\n")
    _remote_reader(monkeypatch, unavailable=True, error=CRASH)
    out = inspect_recovery(*_handle(job))
    assert (out.decision.action.value, out.decision.scheduler_live) == ("noop", True)
    assert out.decision.reason == "continuation 101 (RUNNING) live; nothing to decide"
    assert out.gate_task["status"] == "RUNNING"


def test_a_reader_that_ran_and_refused_still_stops(tmp_path, slurm_bin, monkeypatch):
    from nemo_gym.orchestration.submit import inspect_recovery

    job = _managed_run(tmp_path)
    _remote_reader(monkeypatch, unavailable=False, error=REFUSAL)
    out = inspect_recovery(*_handle(job))
    assert (out.decision.action.value, out.ok) == ("stop", False)
    assert out.decision.reason.endswith(
        "an authoritative Store reader is unavailable or could not verify these artifacts; recovery stops: " + REFUSAL
    )


# The chain's own continuation.


def _admission(job, monkeypatch, capsys, *, unavailable: bool, error: str) -> tuple[str, str]:
    monkeypatch.setattr(
        recovery,
        "select_managed_evidence",
        lambda collected, job_dir: collected.model_copy(
            update={
                "managed_selection": None,
                "managed_reader_error": error,
                "managed_reader_unavailable": unavailable,
            }
        ),
    )
    verdict = recovery.continuation_admission(str(job), 2, None, "0:15", "100", "TIMEOUT")
    return verdict, capsys.readouterr().out


def test_an_unavailable_reader_in_the_chain_ends_it_with_an_explicit_unknown(tmp_path, monkeypatch, capsys):
    job = tmp_path / "job"
    _cut_run(job)
    verdict, printed = _admission(job, monkeypatch, capsys, unavailable=True, error="No module named 'x'")
    assert verdict == "unknown"
    assert "the Store reader could not run, so nothing was decided: No module named 'x'" in printed
    assert "gym eval resubmit --recover --inspect" in printed


def test_a_reader_that_refused_in_the_chain_still_stops(tmp_path, monkeypatch, capsys):
    job = tmp_path / "job"
    _cut_run(job)
    verdict, printed = _admission(job, monkeypatch, capsys, unavailable=False, error="superseded journal")
    assert verdict == "stop"
    assert "the Store could not verify the saved artifacts: superseded journal" in printed


@pytest.mark.parametrize(
    "verdict,reason,marker",
    [
        ("unknown", "the Store reader could not run", "Saved recovery evidence was not read; the Store reader could"),
        ("stop", "the Store could not verify", "Saved recovery evidence has no eligible retries; the Store could"),
    ],
)
def test_the_chain_records_an_unread_reader_apart_from_a_stop(tmp_path, verdict, reason, marker):
    _cut_run(tmp_path)
    admission = tmp_path / "admission.sh"
    admission.write_text(
        f"#!/bin/bash\necho '{recovery.CONTINUATION_REASON_PREFIX}{reason}'\necho '__GYM_CONTINUATION:{verdict}'\n"
    )
    admission.chmod(0o755)
    result = _run_prologue(
        tmp_path, ResumeConfig(max_retries=2), "TIMEOUT", exit_code="0:15", admission_command=str(admission)
    )
    assert "REACHED_WORK" not in result.stdout and not (tmp_path / "sbatch_calls").exists()
    assert (tmp_path / ".gym_chain_incomplete").read_text().startswith(marker)
