# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote transport boundary checks; actual Store selection uses persisted fixtures."""

import pytest

from nemo_gym.orchestration.api import ResumeConfig, SubmitConfig
from nemo_gym.orchestration.executors.slurm import _remote_managed_evidence
from nemo_gym.orchestration.recovery import select_managed_evidence
from tests.unit_tests.test_managed_recovery_selection import collect, decide, prepare_managed, saved_bytes
from tests.unit_tests.test_rollout_recovery import runner_config as runner_config


class ProjectionConnection:
    """Explicit transport stub: production selector runs on real files, bootstrap is separately tested."""

    def __init__(self, output):
        self.output = output
        self.commands = None

    def run(self, commands):
        self.commands = commands
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def config(ref="a" * 40):
    return SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"cluster": {"type": "slurm", "account": "test"}},
            "driver": {
                "gym_install": {"repo": "https://example.test/gym.git", "ref": ref},
                "benchmarks": {"task": {"resumable": True}},
            },
            "job": {"output_path": "/tmp"},
        }
    )


@pytest.mark.parametrize(
    "case,expected",
    [
        ("healthy", "noop"),
        ("wrong", "noop"),
        ("final_zero", "noop"),
        ("infra", "resubmit"),
        ("judge", "reverify"),
        ("prior_reservation", "stop"),
        ("unknown_kill", "stop"),
        ("exhausted", "stop"),
        ("duplicate_history", "noop"),
    ],
)
def test_remote_projection_retains_actual_store_selection(tmp_path, case, expected):
    prepare_managed(tmp_path, case)
    before = saved_bytes(tmp_path)
    raw = collect(tmp_path)
    selected = select_managed_evidence(raw, tmp_path)
    # This tests the response validation boundary, not a real SSH/bootstrap operation.
    conn = ProjectionConnection("installer output\n__GYM_MANAGED:" + selected.model_dump_json())
    result = _remote_managed_evidence(conn, raw, tmp_path, config())
    assert decide(result).action.value == expected
    assert result.managed_selection == selected.managed_selection
    assert result.source_job_identity == raw.source_job_identity
    assert saved_bytes(tmp_path) == before
    assert "synthetic saved answer" not in "\n".join(conn.commands)
    command = "\n".join(conn.commands)
    assert "timeout --kill-after=10s 300s" in command
    assert "prepare" not in command
    assert "a" * 40 in command
    assert "rm -rf" in command


@pytest.mark.parametrize(
    "mutation", ["snapshot", "budget", "score", "identity", "malformed", "duplicate", "bootstrap"]
)
def test_remote_projection_cannot_replace_collection_evidence(tmp_path, mutation):
    prepare_managed(tmp_path, "infra")
    raw = collect(tmp_path)
    selected = select_managed_evidence(raw, tmp_path)
    if mutation == "snapshot":
        selected.source_digests = {}
    elif mutation == "budget":
        selected.infra_retries_used = 0
    elif mutation == "score":
        selected.score = 1
    elif mutation == "identity":
        selected.source_job_identity = ("/different", 1, 2)
    output = "__GYM_MANAGED:" + selected.model_dump_json()
    if mutation == "malformed":
        output = "__GYM_MANAGED:{}"
    elif mutation == "duplicate":
        output += "\n" + output
    elif mutation == "bootstrap":
        output = RuntimeError("runtime unavailable")
    conn = ProjectionConnection(output)
    result = _remote_managed_evidence(conn, raw, tmp_path, config())
    assert result.managed_selection is None
    assert decide(result).action.value == "stop"
    assert result.source_digests == raw.source_digests


def test_unpinned_remote_runtime_is_not_installed(tmp_path):
    prepare_managed(tmp_path, "healthy")
    conn = ProjectionConnection(AssertionError("must not connect"))
    result = _remote_managed_evidence(conn, collect(tmp_path), tmp_path, config("main"))
    assert result.managed_selection is None
    assert conn.commands is None


def test_collector_binds_the_exact_directory_not_only_identical_contents(tmp_path):
    prepare_managed(tmp_path, "healthy")
    raw = collect(tmp_path)
    stat = tmp_path.stat()
    assert raw.source_job_identity == (str(tmp_path.resolve()), stat.st_dev, stat.st_ino)


def test_unbound_remote_directory_is_not_installed(tmp_path):
    prepare_managed(tmp_path, "healthy")
    raw = collect(tmp_path)
    raw.source_job_identity = None
    conn = ProjectionConnection(AssertionError("must not connect"))
    result = _remote_managed_evidence(conn, raw, tmp_path, config())
    assert result.managed_selection is None
    assert conn.commands is None


@pytest.mark.parametrize("changed_at", ["remote_selection", "collection"])
def test_remote_executor_refuses_changed_saved_budget(tmp_path, monkeypatch, changed_at):
    from nemo_gym.orchestration.executors import slurm
    from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord

    prepare_managed(tmp_path, "infra")
    raw = collect(tmp_path)
    selected = select_managed_evidence(raw, tmp_path)

    class Transport:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def run(self, commands):
            assert len(commands) == 1 and "python3 -c" in commands[0]
            return raw.model_dump_json()

    saved = config()
    changed = saved.model_copy(deep=True)
    changed.driver.benchmarks["task"].resumable = ResumeConfig(max_retries=0)
    # Reads: before collection, right after collection, after remote selection.
    reads = iter([saved, changed] if changed_at == "collection" else [saved, saved, changed])
    monkeypatch.setattr(slurm, "get_connection", lambda _: Transport())
    monkeypatch.setattr(slurm, "_remote_managed_evidence", lambda *_: selected)
    monkeypatch.setattr(slurm.SlurmExecutor, "_saved_config", staticmethod(lambda *_: next(reads)))
    record = SubmissionRecord(
        gym_job_id="synthetic",
        gym_version="test",
        submitted_at="2026-01-01T00:00:00Z",
        run_dir=str(tmp_path),
        cluster="test",
        executor="slurm",
        submitted_by="test",
        hostname="remote.test",
        benchmarks=[BenchmarkJob(benchmark="task", job_dir=str(tmp_path))],
    )
    if changed_at == "collection":
        with pytest.raises(RuntimeError, match="changed while collecting recovery evidence"):
            slurm.SlurmExecutor().recovery_evidence(record, "task")
        return
    decision, evidence = slurm.SlurmExecutor().recovery_evidence(record, "task")
    assert decision.action.value == "stop"
    assert evidence.managed_selection is None
    assert "config changed" in evidence.managed_reader_error


def test_large_projection_streams_outside_installer_argv(tmp_path, monkeypatch):
    import shlex

    from nemo_gym.orchestration.executors import slurm
    from nemo_gym.orchestration.executors.connection import LocalConnection

    prepare_managed(tmp_path, "healthy")
    raw = collect(tmp_path)
    raw.results[0]["synthetic_projection_padding"] = "x" * (2 * 1024 * 1024)

    # Only replace bootstrap and Store selection here: execute the actual outer transport in a real shell.
    # Full Store/installed-source behavior is exercised separately; this isolates the OS argv boundary.
    def installed_stub(**kwargs):
        assert kwargs["command"].endswith(" <&3")
        code = "import json,sys; d=json.load(sys.stdin); print('__GYM_MANAGED:'+json.dumps(d))"
        return "bash -c " + shlex.quote("python3 -c " + shlex.quote(code) + " <&3")

    monkeypatch.setattr(slurm, "render_driver_entrypoint", installed_stub)
    result = _remote_managed_evidence(LocalConnection(), raw, tmp_path, config())
    assert result.managed_reader_error is None
    assert result.results == raw.results
    assert result.source_job_identity == raw.source_job_identity
