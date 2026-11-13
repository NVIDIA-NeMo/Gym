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
"""The remote Store reader's transport: a size-aware bound, one retry, and the real cause in the decision."""

import shlex

import pytest

from nemo_gym.orchestration.executors import slurm
from nemo_gym.orchestration.executors.connection import Connection, LocalConnection
from nemo_gym.orchestration.recovery import CollectedEvidence
from tests.unit_tests.orchestration.test_remote_managed_recovery import config


def _collected(job_dir) -> CollectedEvidence:
    stat = job_dir.stat()
    return CollectedEvidence(
        materialized=[],
        results=[],
        failures=[],
        managed_store=True,
        source_job_identity=(str(job_dir.resolve()), stat.st_dev, stat.st_ino),
    )


def _job(tmp_path, *, rollout_bytes: int = 0):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    with open(artifacts / "rollouts.jsonl", "wb") as handle:
        handle.truncate(rollout_bytes)  # sparse: the size counts, no disk is used
    (artifacts / "rollouts_failures.jsonl").write_bytes(b"")
    return tmp_path


def _install(monkeypatch, script: str) -> None:
    """Replace only the bootstrap: the outer transport runs as written, in a real local shell."""
    monkeypatch.setattr(slurm, "render_driver_entrypoint", lambda **_: "bash -c " + shlex.quote(script))


def test_a_reader_that_runs_out_of_memory_names_its_exit_code_and_stderr(tmp_path, monkeypatch):
    job = _job(tmp_path)
    _install(monkeypatch, "echo 'Traceback (most recent call last):' >&2; echo MemoryError >&2; exit 1")
    result = slurm._remote_managed_evidence(LocalConnection(), _collected(job), job, config())
    assert result.managed_selection is None
    assert result.managed_reader_error.startswith("remote Store reader unavailable: exited 1 after ")
    assert "stderr: Traceback (most recent call last): MemoryError" in result.managed_reader_error


def test_the_bound_scales_with_the_saved_rollout_bytes(tmp_path, monkeypatch):
    monkeypatch.delenv("NEMO_GYM_REMOTE_READER_TIMEOUT_S", raising=False)
    job = _job(tmp_path, rollout_bytes=5 * 2**30)
    _install(monkeypatch, "exit 3")
    result = slurm._remote_managed_evidence(LocalConnection(), _collected(job), job, config())
    # 900 s floor plus 100 s per GB of rollout files.
    assert "(the 1400 s bound for 5.0 GB of rollout files)" in result.managed_reader_error


def test_an_explicit_bound_overrides_the_size(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_REMOTE_READER_TIMEOUT_S", "2400")
    job = _job(tmp_path, rollout_bytes=5 * 2**30)
    _install(monkeypatch, "exit 3")
    result = slurm._remote_managed_evidence(LocalConnection(), _collected(job), job, config())
    assert "(the 2400 s bound for 5.0 GB of rollout files)" in result.managed_reader_error


def test_a_reader_killed_before_its_bound_is_named_with_its_exit_code_and_not_retried(tmp_path, monkeypatch):
    monkeypatch.delenv("NEMO_GYM_REMOTE_READER_TIMEOUT_S", raising=False)
    job = _job(tmp_path)
    marker = tmp_path / "attempts"
    _install(monkeypatch, f"echo x >> {shlex.quote(str(marker))}; echo 'reading rows' >&2; kill -9 $$")
    result = slurm._remote_managed_evidence(LocalConnection(), _collected(job), job, config())
    assert marker.read_text().count("x") == 1
    assert result.managed_reader_error.startswith(
        "remote Store reader unavailable: killed (SIGKILL; OOM or external kill) after "
    )
    assert result.managed_reader_error.endswith(
        "below the 900 s bound for 0.0 GB of rollout files, exit code 137; stderr: reading rows"
    )


@pytest.mark.parametrize(
    "rc,elapsed,cause,retry",
    [
        ("124", 3, "timed out after 3 s (the 2 s bound), exit code 124", True),
        ("137", 12, "timed out after 12 s (the 2 s bound), exit code 137", True),
        ("137", 1, "killed (SIGKILL; OOM or external kill) after 1 s, below the 2 s bound, exit code 137", False),
        ("2", 1, "exited 2 after 1 s (the 2 s bound)", False),
    ],
    ids=["timeout", "killed-after-the-grace-period", "killed-before-the-bound", "error"],
)
def test_only_a_reached_deadline_is_a_timeout_and_every_cause_keeps_the_exit_code_and_stderr(
    rc, elapsed, cause, retry
):
    output = f"__GYM_READER_BOUND:2\n__GYM_READER_RC:{rc}\n__GYM_READER_ERR:last line\n"
    assert slurm._reader_failure(output, elapsed) == (
        f"remote Store reader unavailable: {cause}; stderr: last line",
        retry,
    )


def test_the_stderr_tail_survives_carriage_returns_from_progress_output():
    err = "Updating files:  83%\rUpdating files: 100%\rerror: Required uv version `>=0.13.4` does not match"
    output = f"__GYM_READER_BOUND:2\n__GYM_READER_RC:2\n__GYM_READER_ERR:{err}\n"
    message, _ = slurm._reader_failure(output, 1.0)
    assert message.endswith("does not match")
    assert "Required uv version" in message


def test_a_timeout_is_retried_once_with_twice_the_bound(tmp_path, monkeypatch):
    monkeypatch.delenv("NEMO_GYM_REMOTE_READER_TIMEOUT_S", raising=False)
    monkeypatch.setattr(slurm, "_READER_FLOOR_S", 1)
    job = _job(tmp_path)
    marker = tmp_path / "attempts"
    _install(monkeypatch, f"echo x >> {shlex.quote(str(marker))}; sleep 30")
    result = slurm._remote_managed_evidence(LocalConnection(), _collected(job), job, config())
    assert marker.read_text().count("x") == 2
    assert "timed out after" in result.managed_reader_error
    assert "(the 2 s bound for 0.0 GB of rollout files)" in result.managed_reader_error
    assert result.managed_reader_error.endswith("(after one retry with twice the bound)")


def test_a_reader_that_answers_on_the_retry_is_used(tmp_path, monkeypatch):
    monkeypatch.delenv("NEMO_GYM_REMOTE_READER_TIMEOUT_S", raising=False)
    monkeypatch.setattr(slurm, "_READER_FLOOR_S", 1)
    job = _job(tmp_path)
    collected = _collected(job)
    marker = tmp_path / "attempts"
    projection = "__GYM_MANAGED:" + collected.model_dump_json()
    script = (
        f"if [ -e {shlex.quote(str(marker))} ]; then cat <&3 >/dev/null; echo {shlex.quote(projection)}; "
        f"else touch {shlex.quote(str(marker))}; sleep 30; fi"
    )
    _install(monkeypatch, script)
    result = slurm._remote_managed_evidence(LocalConnection(), collected, job, config())
    assert result.managed_reader_error is None


def test_a_field_the_pinned_reader_does_not_know_is_not_a_change_but_a_known_one_is(tmp_path, monkeypatch):
    """A Store reader at an older Gym drops the fields its model lacks; only the fields it reports are compared."""
    job = _job(tmp_path)
    collected = _collected(job).model_copy(update={"followup_claims": {"100": "101"}})
    older = "__GYM_MANAGED:" + collected.model_dump_json(exclude={"followup_claims"})
    _install(monkeypatch, f"cat <&3 >/dev/null; echo {shlex.quote(older)}")
    kept = slurm._remote_managed_evidence(LocalConnection(), collected, job, config())
    assert (kept.managed_reader_error, kept.followup_claims) == (None, {"100": "101"})

    changed = "__GYM_MANAGED:" + collected.model_copy(update={"resubmit_offset": 5}).model_dump_json()
    _install(monkeypatch, f"cat <&3 >/dev/null; echo {shlex.quote(changed)}")
    refused = slurm._remote_managed_evidence(LocalConnection(), collected, job, config())
    assert "remote Store reader changed collection evidence" in refused.managed_reader_error


class _Hop(Connection):
    """A login-host hop whose SSH fails ``failures`` times with exit code 255, then runs the steps locally."""

    def __init__(self, failures: int):
        self.failures, self.calls = failures, 0

    def run(self, commands):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(
                "Command failed (ssh commands on login) with exit code 255:\n  ssh login bash -s\n"
                "kex_exchange_identification: read: Connection reset by peer"
            )
        return LocalConnection().run(commands)

    def copy(self, local, remote):
        raise NotImplementedError

    def write_text(self, remote, content, *, private=False):
        raise NotImplementedError


@pytest.mark.parametrize("failures,answered", [(1, True), (2, False)])
def test_a_failed_ssh_hop_is_retried_once_after_a_backoff(tmp_path, monkeypatch, failures, answered):
    monkeypatch.setattr(slurm.time, "sleep", lambda _: None)
    job = _job(tmp_path)
    collected = _collected(job)
    _install(monkeypatch, "cat <&3 >/dev/null; echo " + shlex.quote("__GYM_MANAGED:" + collected.model_dump_json()))
    hop = _Hop(failures)
    result = slurm._remote_managed_evidence(hop, collected, job, config())
    assert hop.calls == 2
    if answered:
        assert result.managed_reader_error is None
    else:
        assert result.managed_reader_error.startswith(
            "remote Store reader unavailable: RuntimeError: SSH to the login host failed twice (exit code 255):"
        )
        assert result.managed_reader_error.endswith("Connection reset by peer")


def test_the_reader_reports_nothing_on_success(tmp_path, monkeypatch):
    job = _job(tmp_path)
    collected = _collected(job)
    _install(monkeypatch, "cat <&3 >/dev/null; echo " + shlex.quote("__GYM_MANAGED:" + collected.model_dump_json()))
    result = slurm._remote_managed_evidence(LocalConnection(), collected, job, config())
    assert result.managed_reader_error is None
