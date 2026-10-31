# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual read-only Store selection through the stdlib projection and local executor; no real scheduler."""

import hashlib
import json
import subprocess

import pytest

from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    decide_from_collected,
    select_managed_evidence,
)
from nemo_gym.rollout_recovery import RunManifest
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_eval_resubmit import _handle, _provider_failures, _resumable_run, _sbatch_env
from tests.unit_tests.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.test_rollout_recovery import runner_config as runner_config


def saved_bytes(job):
    return {str(p.relative_to(job)): hashlib.sha256(p.read_bytes()).hexdigest() for p in job.rglob("*") if p.is_file()}


def prepare_managed(job, case, n=1):
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(n)]
    source = job / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        inputs = json.loads(json.dumps(rows))
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    output = artifacts / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for index in range(n):
            row = dict(rows[index])
            selected_case = case if index == 0 else "healthy"
            if selected_case == "missing":
                continue
            if selected_case in ("infra", "exhausted", "reset"):
                attempts = 3 if selected_case == "exhausted" else 2
                for attempt in range(attempts):
                    row = dict(rows[index])
                    store.allocate_attempt(row)
                    failure = json.loads(_provider_failures(index, attempt + 1, 3).splitlines()[-1])
                    store.record_outcome(row | failure)
                continue
            store.record_dispatch(row)
            if selected_case in ("healthy", "wrong", "prior_reservation", "duplicate_history"):
                store.record_outcome(row | {"reward": 0.0 if selected_case == "wrong" else 1.0})
                if selected_case in ("prior_reservation", "duplicate_history"):
                    store.allocate_attempt(row)
                    if selected_case == "duplicate_history":
                        store.record_outcome(row | {"reward": 0.0})
            elif selected_case in ("judge", "judge_then_reservation"):
                store.record_outcome(
                    row
                    | {
                        "_ng_failure_class": "judge_failed",
                        "failure_kind": "judge_failed",
                        "response": {
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [{"type": "output_text", "text": "synthetic saved answer"}],
                                }
                            ],
                        },
                    }
                )
                if selected_case == "judge_then_reservation":
                    store.allocate_attempt(row)
            elif selected_case == "final_zero":
                store.record_outcome(
                    row
                    | {
                        "_ng_failure_class": "agent_run_error",
                        "failure_kind": "model_context_limit",
                        "_ng_failure_terminal": True,
                        "_ng_failure_counts_as_zero": True,
                    }
                )
            elif selected_case == "unknown_kill":
                store.record_outcome(row | {"_ng_failure_class": "kill_shaped", "_ng_omitted_rollout": True})
            else:
                raise AssertionError(selected_case)
    if case == "reset":
        with RolloutStore.start_or_resume(output, prepare, resume=True, resubmit=True):
            pass
    return output


def collect(job):
    result = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
    )
    return CollectedEvidence.model_validate_json(result.stdout)


def decide(c):
    return decide_from_collected(c, max_infra_retries=3, max_walltime_s=None, live_job=False)


@pytest.mark.parametrize(
    "case,action,state,attempts",
    [
        ("healthy", "noop", "scored", 0),
        ("wrong", "noop", "scored", 0),
        ("final_zero", "noop", "model_final", 0),
        ("judge", "reverify", "judge_recoverable", 1),
        ("prior_reservation", "stop", "unknown", 0),
        ("judge_then_reservation", "stop", "unknown", 1),
        ("infra", "resubmit", "retryable", 2),
        ("exhausted", "stop", "exhausted", 3),
        ("reset", "resubmit", "retryable", 0),
        ("unknown_kill", "stop", "unknown", 0),
        ("missing", "resubmit", "missing", 0),
        ("duplicate_history", "noop", "scored", 0),
    ],
)
def test_authoritative_store_selects_saved_outcomes_and_budgets(tmp_path, case, action, state, attempts):
    prepare_managed(tmp_path, case)
    before = saved_bytes(tmp_path)
    raw = collect(tmp_path)
    assert decide(raw).action.value == "stop"  # unsupported remote-reader bridge remains explicit
    selected = select_managed_evidence(raw, tmp_path)
    result = decide(selected)
    assert selected.managed_reader_error is None
    assert (result.action.value, result.expected, result.counts) == (action, 1, {state: 1})
    assert result.evidence[0].attempts == attempts
    assert result.evidence[0].max_attempts == 3
    assert not result.duplicate_results
    assert saved_bytes(tmp_path) == before
    assert "synthetic saved answer" not in selected.model_dump_json()


@pytest.mark.parametrize(
    "case,action",
    [
        ("healthy", "noop"),
        ("wrong", "noop"),
        ("final_zero", "noop"),
        ("judge", "stop"),
        ("prior_reservation", "stop"),
        ("infra", "resubmit"),
        ("exhausted", "stop"),
        ("unknown_kill", "stop"),
    ],
)
def test_actual_local_slurm_executor_keeps_budget_and_never_regenerates_saved_answer(
    tmp_path, slurm_bin, case, action
):
    from nemo_gym.orchestration.submit import recover

    job, _, _ = _resumable_run(tmp_path)
    prepare_managed(job, case, n=2)
    before = saved_bytes(job / "artifacts")
    result = recover(*_handle(job))
    assert result.decision.action.value == action
    assert (result.dispatched == "resubmit") is (action == "resubmit")
    if case == "judge":
        assert not result.ok and result.unsupported
        assert "no existing CPU-only judge execution profile" in result.unsupported
        assert result.decision.counts == {"judge_recoverable": 1, "scored": 1}
        assert result.dispatched is None
    assert saved_bytes(job / "artifacts") == before
    assert (job / ".gym_infra_retries").read_text() == "2\n"
    if action == "resubmit":
        assert _sbatch_env(slurm_bin)["GYM_RESUBMIT_KEEP_BUDGET"] == "1"
        assert (job / ".gym_accumulated_walltime").read_text() == "7260\n"
    else:
        assert not (slurm_bin / "env").exists()
        assert (job / ".gym_accumulated_walltime").read_text() == "7200\n"


@pytest.mark.parametrize(
    "name",
    [
        "rollouts.jsonl",
        "rollouts_failures.jsonl",
        "rollouts_materialized_inputs.jsonl",
        "rollouts_resolved_config.yaml",
        "rollouts_aggregate_metrics.json",
        "quality_summary.json",
        "rollouts_manifest.json",
        "rollouts_attempts.jsonl",
        "rollouts_resubmit_offset",
        "../.gym_infra_retries",
        "../.gym_accumulated_walltime",
        "../.gym_job_chain",
    ],
)
def test_changed_history_between_collector_and_reader_stops(tmp_path, name):
    prepare_managed(tmp_path, "healthy")
    raw = collect(tmp_path)
    path = tmp_path / "artifacts" / name
    path.write_bytes((path.read_bytes() if path.exists() else b"") + b"\n")
    after = select_managed_evidence(raw, tmp_path)
    assert after.managed_selection is None and decide(after).action.value == "stop"


def test_manifest_changed_during_store_read_stops(tmp_path, monkeypatch):
    prepare_managed(tmp_path, "healthy")
    raw = collect(tmp_path)
    original = RolloutStore.read

    def changing(*args, **kwargs):
        store = original(*args, **kwargs)
        path = tmp_path / "artifacts/rollouts_manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return store

    monkeypatch.setattr(RolloutStore, "read", changing)
    after = select_managed_evidence(raw, tmp_path)
    assert after.managed_selection is None and decide(after).action.value == "stop"


def test_missing_saved_attempt_cap_is_not_replaced_by_process_defaults(tmp_path):
    prepare_managed(tmp_path, "infra")
    path = tmp_path / "artifacts/rollouts_manifest.json"
    payload = json.loads(path.read_text())
    payload["max_rollout_attempts"] = None
    path.write_text(json.dumps(payload))
    selected = select_managed_evidence(collect(tmp_path), tmp_path)
    assert selected.managed_selection is None
    assert "attempt cap" in selected.managed_reader_error
    assert decide(selected).action.value == "stop"


def test_superseded_journal_is_not_parsed_by_the_consumer(tmp_path):
    prepare_managed(tmp_path, "healthy")
    (tmp_path / "artifacts/rollouts_attempts.jsonl").write_text("{}\n")
    selected = select_managed_evidence(collect(tmp_path), tmp_path)
    assert selected.managed_selection is None
    assert "superseded" in selected.managed_reader_error
    assert decide(selected).action.value == "stop"


def test_remote_connection_preserves_unsupported_reader_stop(tmp_path, slurm_bin, monkeypatch):
    from nemo_gym.orchestration.executors import slurm
    from nemo_gym.orchestration.executors.connection import Connection, LocalConnection
    from nemo_gym.orchestration.submit import recover

    job, _, _ = _resumable_run(tmp_path)
    prepare_managed(job, "healthy", n=2)

    class RemoteTransport(Connection):
        # Execute the exact serialized commands locally, but expose the SSH capability boundary.
        def run(self, commands):
            return LocalConnection().run(commands)

        def copy(self, local, remote):
            raise AssertionError("read only")

        def write_text(self, remote, content, *, private=False):
            raise AssertionError("no dispatch")

    monkeypatch.setattr(slurm, "get_connection", lambda hostname: RemoteTransport())
    result = recover(*_handle(job))
    assert result.decision.action.value == "stop" and result.dispatched is None
    assert "authoritative Store reader" in result.decision.reason
    assert not (slurm_bin / "env").exists()


@pytest.mark.parametrize("other", ["missing", "different-run"])
def test_reader_cannot_bind_collected_evidence_to_another_local_path(tmp_path, other):
    job = tmp_path / "job"
    prepare_managed(job, "healthy")
    raw = collect(job)
    different = tmp_path / other
    if other == "different-run":
        prepare_managed(different, "wrong")
    selected = select_managed_evidence(raw, different)
    assert selected.managed_selection is None
    assert decide(selected).action.value == "stop"


def test_managed_selection_never_materializes_a_whole_rollout_file(tmp_path, monkeypatch):
    prepare_managed(tmp_path, "healthy")
    raw = collect(tmp_path)
    original = type(tmp_path).read_bytes

    def bounded(path):
        if path.name in {"rollouts.jsonl", "rollouts_failures.jsonl", "rollouts_materialized_inputs.jsonl"}:
            raise AssertionError("whole rollout artifact read is not bounded")
        return original(path)

    monkeypatch.setattr(type(tmp_path), "read_bytes", bounded)
    selected = select_managed_evidence(raw, tmp_path)
    assert selected.managed_reader_error is None
    assert decide(selected).action.value == "noop"


@pytest.mark.parametrize("reader", ["managed", "collector"])
def test_snapshot_hash_memory_is_bounded_on_large_sparse_file(tmp_path, reader):
    import tracemalloc

    from nemo_gym.orchestration.recovery import _managed_source_digest

    path = tmp_path / "large.jsonl"
    size = 256 << 20
    with path.open("wb") as stream:
        stream.truncate(size)
    digest = _managed_source_digest if reader == "managed" else collector_hash()
    tracemalloc.start()
    result = digest(path)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    expected = hashlib.sha256()
    block = bytes(1 << 20)
    for _ in range(size // len(block)):
        expected.update(block)
    assert result == expected.hexdigest()
    assert peak < 4 << 20


def collector_hash():
    # Exercise the exact stdlib function serialized to the host, without starting collection.
    import ast
    import os

    namespace = {"os": os, "hashlib": hashlib}
    node = next(
        node for node in ast.parse(COLLECT_SCRIPT).body if isinstance(node, ast.FunctionDef) and node.name == "sha256"
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), "collector-hash", "exec"), namespace)
    return namespace["sha256"]


@pytest.mark.parametrize("reader", ["managed", "collector"])
@pytest.mark.parametrize("mutation", ["append", "replace", "rewrite"])
def test_hashing_rejects_mid_read_file_mutation(tmp_path, monkeypatch, reader, mutation):
    import builtins

    from nemo_gym.orchestration.recovery import _managed_source_digest

    path = tmp_path / "source.jsonl"
    path.write_bytes(b"a" * (2 << 20))
    original = builtins.open
    mutated = False

    class MutatingRead:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.source.close()

        def fileno(self):
            return self.source.fileno()

        def read(self, size):
            nonlocal mutated
            block = self.source.read(size)
            if block and not mutated:
                mutated = True
                if mutation == "replace":
                    replacement = tmp_path / "replacement"
                    with original(replacement, "wb") as stream:
                        stream.write(b"a" * (2 << 20))
                    replacement.replace(path)
                else:
                    with original(path, "ab" if mutation == "append" else "r+b") as stream:
                        stream.write(b"b")
            return block

    def instrumented_open(target, mode="r", *args, **kwargs):
        source = original(target, mode, *args, **kwargs)
        return MutatingRead(source) if str(target) == str(path) and mode == "rb" else source

    def instrumented_path_open(target, mode="r", *args, **kwargs):
        return instrumented_open(target, mode, *args, **kwargs)

    if reader == "managed":
        monkeypatch.setattr(type(path), "open", instrumented_path_open)
        digest = _managed_source_digest
    else:
        monkeypatch.setattr(builtins, "open", instrumented_open)
        digest = collector_hash()
    with pytest.raises(RuntimeError, match="changed while hashing"):
        digest(path)
    assert mutated


async def test_actual_collection_native_outcomes_reach_managed_recovery(runner_config, monkeypatch, tmp_path):
    from tests.unit_tests.test_rollout_recovery import (
        test_typed_episode_outcomes_settle_stop_or_retry_through_collection_and_resume as produce,
    )

    runner_config.output_jsonl_fpath = str(tmp_path / "artifacts/rollouts.jsonl")
    await produce(runner_config, monkeypatch)
    before = saved_bytes(tmp_path)
    selected = select_managed_evidence(collect(tmp_path), tmp_path)
    result = decide(selected)
    assert selected.managed_reader_error is None
    assert result.action.value == "stop"
    assert result.expected == 7 and not result.duplicate_results
    assert saved_bytes(tmp_path) == before
    states = {item.identity[0]: item.state.value for item in result.evidence}
    proof = {
        "states": states,
        "attempts": {item.identity[0]: item.attempts for item in result.evidence},
        "selected_results": len(selected.results),
        "selected_failures": len(selected.failures),
        "no_source_mutation": True,
        "action": result.action.value,
    }
    (tmp_path / "consumer-proof.json").write_text(json.dumps(proof, indent=2))
    assert states == {
        0: "scored",
        1: "model_final",
        2: "exhausted",
        3: "judge_recoverable",
        4: "unknown",
        5: "scored",
        6: "unknown",
    }
    assert result.evidence[2].attempts == 3


@pytest.mark.parametrize(
    "kind,selected_kind,expected",
    [
        (None, None, "model_final"),
        ("provider_unavailable", None, "unknown"),
        (None, "provider_unavailable", "unknown"),
    ],
)
def test_managed_canonical_final_zero_does_not_override_explicit_cause(tmp_path, kind, selected_kind, expected):
    prepare_managed(tmp_path, "final_zero")
    path = tmp_path / "artifacts/rollouts_failures.jsonl"
    row = json.loads(path.read_text())
    row["_ng_failure_class"] = "model_policy_response_timeout"
    row["failure_kind"] = kind
    if selected_kind:
        row["_ng_failure_id"] = "selected"
        row["_ng_failure_events"] = {
            "selected": {
                "event_id": "selected",
                "failure_kind": selected_kind,
                "boundary": "model",
                "operation": "model.request",
            }
        }
    path.write_text(json.dumps(row) + "\n")
    selected = select_managed_evidence(collect(tmp_path), tmp_path)
    assert selected.managed_reader_error is None
    assert decide(selected).evidence[0].state.value == expected


def test_managed_terminal_infrastructure_is_not_a_model_zero(tmp_path):
    prepare_managed(tmp_path, "final_zero")
    path = tmp_path / "artifacts/rollouts_failures.jsonl"
    row = json.loads(path.read_text())
    row["_ng_failure_class"] = "provider_unavailable"
    row["failure_kind"] = None
    path.write_text(json.dumps(row) + "\n")
    selected = select_managed_evidence(collect(tmp_path), tmp_path)
    assert decide(selected).action.value == "stop"
    assert decide(selected).evidence[0].state.value == "unknown"


@pytest.mark.parametrize("age", ["fresh", "aged"])
@pytest.mark.parametrize("reader", ["managed", "collector"])
def test_a_same_tick_in_place_rewrite_is_refused_when_stat_cannot_see_it(tmp_path, monkeypatch, reader, age):
    # Coarse filesystem timestamps: freeze every stat result, so only the bytes show the rewrite.
    import builtins
    import os

    from nemo_gym.orchestration.recovery import _managed_source_digest

    path = tmp_path / "source.jsonl"
    path.write_bytes(b"a" * (2 << 20))
    if age == "aged":
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    frozen = os.stat(path)
    monkeypatch.setattr(os, "fstat", lambda _fd: frozen)
    monkeypatch.setattr(os, "stat", lambda *_args, **_kwargs: frozen)
    digest = _managed_source_digest if reader == "managed" else collector_hash()
    original_read = None

    def rewrite_once(read):
        def wrapped(size):
            nonlocal original_read
            block = read(size)
            if block and original_read is None:
                original_read = block
                with builtins.open(path, "r+b") as stream:
                    stream.write(b"b")
            return block

        return wrapped

    opener = builtins.open

    def instrumented(target, mode="r", *args, **kwargs):
        source = opener(target, mode, *args, **kwargs)
        if str(target) == str(path) and mode == "rb" and original_read is None:
            source.read = rewrite_once(source.read)
        return source

    monkeypatch.setattr(builtins, "open", instrumented)
    monkeypatch.setattr(type(path), "open", lambda target, mode="r", *a, **k: instrumented(target, mode, *a, **k))
    with pytest.raises(RuntimeError, match="changed while hashing"):
        digest(path)


@pytest.mark.parametrize("reader", ["managed", "collector"])
def test_an_unchanged_file_gets_the_same_digest_from_both_readers(tmp_path, reader):
    from nemo_gym.orchestration.recovery import _managed_source_digest

    path = tmp_path / "source.jsonl"
    path.write_bytes(b"a" * (2 << 20) + b"tail")
    digest = _managed_source_digest if reader == "managed" else collector_hash()
    assert digest(path) == hashlib.sha256(b"a" * (2 << 20) + b"tail").hexdigest()
    assert digest(tmp_path / "absent.jsonl") is None
