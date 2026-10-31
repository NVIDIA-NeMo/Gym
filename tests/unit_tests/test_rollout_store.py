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

"""Exercise artifact ownership and interruption at persistence boundaries."""

import gc
import tracemalloc
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import orjson
import pytest

from nemo_gym.config.config_types import ConfigError
from nemo_gym.rollout_records import (
    RolloutRecord,
    journal_path_for,
    materialized_path_for,
    read_records,
)
from nemo_gym.rollout_recovery import RunManifest, atomic_write_json, manifest_path_for
from nemo_gym.rollout_store import RolloutStore
from nemo_gym.utils.path_utils import failures_path_for


@pytest.fixture
def prepared_run(tmp_path):
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(2)]
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    prepare = Mock(
        side_effect=lambda: (
            rows,
            RunManifest.create(source, rows, {}, {"agent": {"responses_api_agents": {"impl": {}}}}),
        )
    )
    return tmp_path / "rollouts.jsonl", prepare


def snapshot(output):
    return {path.name: path.read_bytes() for path in output.parent.iterdir() if path.is_file()}


def test_invalid_outcomes_are_rejected_before_appending_bytes(prepared_run):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row, undispatched = store.pending(3)
        store.record_dispatch(row)
        result = row | {"reward": 0.0, "response": {}}
        store.record_outcome(result)
        before = snapshot(output)
        for invalid in (result | {"reward": 1.0}, undispatched | {"reward": 0.0, "response": {}}):
            with pytest.raises(ConfigError, match="Conflicting outcomes|no reserved attempt"):
                store.record_outcome(invalid)
            assert snapshot(output) == before
    assert RolloutStore.read(output).selected("success") == [result]


def test_partial_open_failure_closes_files_and_allows_retry(prepared_run, monkeypatch):
    output, prepare = prepared_run
    store = RolloutStore.start_or_resume(output, prepare, resume=False)
    original_open = Path.open
    opened = []

    def failing_open(path, *args, **kwargs):
        if args and args[0] == "ab":
            if path == failures_path_for(output):
                raise OSError("cannot open sidecar")
            file = original_open(path, *args, **kwargs)
            opened.append(file)
            return file
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", failing_open)
        with pytest.raises(OSError, match="cannot open sidecar"):
            with store:
                pytest.fail("The store must not open partially")
    assert len(opened) == 1 and all(file.closed for file in opened)
    with store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"reward": 0.0, "response": {}})
    assert RolloutStore.read(output).coverage()["successful"] == 1


def test_store_requires_write_context_and_offline_reads_never_open_writers(prepared_run):
    output, prepare = prepared_run
    store = RolloutStore.start_or_resume(output, prepare, resume=False)
    row = store.pending(3)[0]
    with pytest.raises(RuntimeError, match="open store context"):
        store.record_dispatch(row)
    with store:
        with pytest.raises(RuntimeError, match="already open"):
            with store:
                pytest.fail("Cannot open the same store twice")
        store.record_dispatch(row)
    before = snapshot(output)
    reader = RolloutStore.read(output)
    with pytest.raises(RuntimeError, match="read-only"):
        with reader:
            pytest.fail("Cannot write through an offline reader")
    assert snapshot(output) == before
    with pytest.raises(RuntimeError, match="open store context"):
        store.record_outcome(row | {"reward": 0.0, "response": {}})


def test_recovery_and_offline_selection_use_the_newest_dispatch(prepared_run):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        original = store.pending(3)[0]
        store.record_dispatch(original)
        store.record_dispatch(original | {"_ng_attempt_index": 1})
        store.record_outcome(original | {"reward": 1.0, "response": {}})
        assert store.selected("success") == []
    before = snapshot(output)
    offline = RolloutStore.read(output)
    assert offline.selected("success") == []
    assert offline.coverage()["selection_policy"] == "latest_allocated"
    assert snapshot(output) == before
    with RolloutStore.start_or_resume(output, prepare, resume=True) as resumed:
        assert resumed.coverage() == offline.coverage()
        assert resumed.pending(3) == offline.pending(3)
        assert resumed.pending(3)[0]["_ng_attempt_index"] == 2
        retry = resumed.pending(3)[0]
        resumed.record_dispatch(retry)
        resumed.record_outcome(retry | {"reward": 0.0, "response": {}})
    assert [row["reward"] for row in RolloutStore.read(output).selected("success")] == [0.0]
    assert len(list(read_records(output))) == 2


@pytest.mark.parametrize("reopen", [False, True])
def test_large_history_retains_offsets_instead_of_trajectories(prepared_run, reopen, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "80")
    output, prepare = prepared_run
    store = RolloutStore.start_or_resume(output, prepare, resume=False)
    blob = "x" * (256 * 1024)

    def write_history():
        with store:
            rows = store.pending(80)
            for attempt in range(64):
                row = rows[attempt % 2] | {"_ng_attempt_index": attempt}
                store.record_dispatch(row)
                payload = row | {"response": {"output_text": blob + str(attempt)}, "reward": 0.0}
                if attempt % 2:
                    payload["_ng_failure_class"] = "provider_unavailable"
                store.record_outcome(payload)

    if reopen:
        write_history()
        del store
    gc.collect()
    tracemalloc.start()
    try:
        if reopen:
            store = RolloutStore.read(output)
        else:
            write_history()
        gc.collect()
        retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Sixteen MiB of saved responses must not become sixteen MiB of retained state.
    # Allow room for parsing/serialization of one record and small metadata.
    artifact_bytes = output.stat().st_size + failures_path_for(output).stat().st_size
    assert retained < artifact_bytes / 4
    assert peak < artifact_bytes / 2

    def unexpected_read(*args):
        pytest.fail("Coverage and retry selection must use metadata without rereading trajectories")

    with monkeypatch.context() as patch:
        patch.setattr(RolloutRecord, "_read", unexpected_read)
        assert store.coverage()["successful"] == 1
        assert store.coverage()["failed"] == 1
        assert store.pending(80)[0]["_ng_attempt_index"] == 32
        locations = store.selected_records("success")
        assert len(locations) == 1

    record = next(iter(locations.values()))
    assert record.path == output
    assert record.offset > 0
    assert record.read()["_ng_attempt_index"] == 31
    assert store.selected("success")[0]["response"]["output_text"] == blob + "62"


def test_indexed_legacy_attempts_preserve_migration_without_rewriting(prepared_run):
    output, prepare = prepared_run
    rows, _ = prepare()
    materialized_path_for(output).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    first = rows[0] | {"response": {"output_text": "snow: 雪"}, "reward": 0.0, "_ng_attempt_index": 0}
    second = first | {"reward": 1.0}
    raw_first = orjson.dumps(first) + b"\n"
    raw_second = orjson.dumps(second)  # Complete JSON without a final newline remains readable.
    original = b"\n" + raw_first + b"  \n" + raw_second
    output.write_bytes(original)

    store = RolloutStore.read(output)
    record = next(iter(store.selected_records("success").values()))
    assert (record.path, record.offset, record.length, record.line_number) == (
        output,
        len(b"\n" + raw_first + b"  \n"),
        len(raw_second),
        4,
    )
    assert record.read() == second | {"_ng_attempt_index": 1}
    assert store.selected("success") == [second | {"_ng_attempt_index": 1}]
    assert output.read_bytes() == original


def test_legacy_import_preserves_artifacts_and_does_not_require_original_source(prepared_run):
    output, prepare = prepared_run
    rows, _ = prepare()
    materialized_path_for(output).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    old_result = rows[0] | {"reward": 0.0, "response": {}}
    output.write_bytes(orjson.dumps(old_result) + b"\n")
    prepare.reset_mock()
    prepare.side_effect = AssertionError("Original dataset no longer exists")
    original = output.read_bytes()
    before = snapshot(output)
    assert RolloutStore.read(output).coverage()["successful"] == 1
    assert snapshot(output) == before
    with pytest.warns(UserWarning, match="allow_unsafe_resume"):
        store = RolloutStore.start_or_resume(output, prepare, resume=True, allow_unsafe=True)
    prepare.assert_not_called()
    with store:
        assert [row["_ng_task_index"] for row in store.pending(3)] == [1]
        assert not store.coverage()["identity_verified"]
        row = store.pending(3)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"reward": 1.0, "response": {}})
    assert output.read_bytes().startswith(original)
    assert RolloutStore.read(output).coverage()["complete"]
    assert RunManifest.model_validate_json(manifest_path_for(output).read_bytes()).legacy_import


def test_legacy_file_without_inventory_has_unknown_coverage(tmp_path):
    output = tmp_path / "legacy.jsonl"
    output.write_bytes(b'{"reward":0.0,"response":{}}\n')
    assert RolloutStore.read(output) is None


def test_prepared_but_never_opened_run_can_resume(prepared_run):
    output, prepare = prepared_run
    abandoned = RolloutStore.start_or_resume(output, prepare, resume=False)
    with RolloutStore.start_or_resume(output, prepare, resume=True) as resumed:
        assert resumed.manifest.run_id == abandoned.manifest.run_id
        assert resumed.pending(3) == abandoned.pending(3)
        assert resumed.coverage()["attempts"] == 0
        for row in resumed.pending(3):
            resumed.record_dispatch(row)
            resumed.record_outcome(row | {"reward": 0.0, "response": {}})
    assert RolloutStore.read(output).coverage()["complete"]


@pytest.mark.parametrize("allow_unsafe", [False, True])
@pytest.mark.parametrize("artifact", ["materialized", "output", "failures"])
def test_incomplete_initial_cache_restarts_from_current_inputs(prepared_run, capsys, artifact, allow_unsafe):
    output, prepare = prepared_run
    paths = {"materialized": materialized_path_for(output), "output": output, "failures": failures_path_for(output)}
    # Preparation can restart only before any result or failure has been saved.
    paths[artifact].write_bytes(b'{"old_partial_run":true}\n' if artifact == "materialized" else b"")
    with RolloutStore.start_or_resume(output, prepare, resume=True, allow_unsafe=allow_unsafe) as store:
        assert store.coverage()["attempts"] == 0
        assert len(store.pending(3)) == 2
        assert list(read_records(output)) == []
        assert list(read_records(failures_path_for(output))) == []
    prepare.assert_called_once()
    assert "Skipping resume_from_cache" in capsys.readouterr().out


def test_interruption_after_materialization_can_restart(prepared_run, monkeypatch):
    output, prepare = prepared_run
    touch = Path.touch

    def interrupted_touch(path, *args, **kwargs):
        if path == output:
            raise OSError("interrupted before output creation")
        return touch(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "touch", interrupted_touch)
        with pytest.raises(OSError, match="interrupted before output"):
            RolloutStore.start_or_resume(output, prepare, resume=False)
    assert materialized_path_for(output).exists()
    assert not manifest_path_for(output).exists()
    assert not journal_path_for(output).exists()
    with RolloutStore.start_or_resume(output, prepare, resume=True) as store:
        assert len(store.pending(3)) == 2
        assert store.coverage()["attempts"] == 0


@pytest.mark.parametrize("metadata", [manifest_path_for])
@pytest.mark.parametrize("allow_unsafe", [False, True])
def test_incomplete_journaled_cache_does_not_fall_back_to_fresh(prepared_run, metadata, allow_unsafe):
    output, prepare = prepared_run
    metadata(output).write_bytes(b"{}\n")
    before = snapshot(output)
    with pytest.raises(ConfigError, match="missing"):
        RolloutStore.start_or_resume(output, prepare, resume=True, allow_unsafe=allow_unsafe)
    prepare.assert_not_called()
    assert snapshot(output) == before


@pytest.mark.parametrize("remove_manifest", [False, True])
def test_unsafe_import_rejects_foreign_run_mixtures(prepared_run, remove_manifest):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for row in store.pending(3):
            store.record_dispatch(row)
            store.record_outcome(row | {"reward": 0.0, "response": {}})
    records = list(read_records(output))
    records[1]["_ng_run_id"] = "foreign"
    output.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in records))
    journal_path_for(output).unlink(missing_ok=True)
    if remove_manifest:
        manifest_path_for(output).unlink()
    before = snapshot(output)
    with pytest.raises(ConfigError, match="different run"):
        RolloutStore.start_or_resume(output, prepare, resume=True, allow_unsafe=True)
    assert snapshot(output) == before


def test_legacy_conflicting_attempts_and_failure_in_main_remain_readable(prepared_run):
    output, prepare = prepared_run
    rows, _ = prepare()
    materialized_path_for(output).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    failure = rows[0] | {"_ng_attempt_index": 0, "_ng_failure_class": "judge_failed"}
    result = rows[0] | {"_ng_attempt_index": 0, "reward": 1.0, "response": {}}
    output.write_bytes(orjson.dumps(failure) + b"\n" + orjson.dumps(result) + b"\n")
    before = snapshot(output)
    store = RolloutStore.read(output)
    assert store.selected("success") == [result | {"_ng_attempt_index": 1}]
    assert store.coverage()["attempts"] == 2
    assert snapshot(output) == before


def test_legacy_omitted_rollout_rows_keep_the_latest_execution(prepared_run):
    """Rows record_omitted_rollouts wrote before the run had a manifest.

    Task 0 failed in the judge and its retry was interrupted: the interruption is the latest execution and spends
    no retry. Task 1 failed in the judge and its retry was never started: that row is skipped, so the failure stands.
    """
    output, prepare = prepared_run
    rows, _ = prepare()
    materialized_path_for(output).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    output.write_bytes(b"")
    omitted = {"_ng_no_persist": True, "_ng_omitted_rollout": True}
    sidecar = [
        rows[0] | {"_ng_failure_class": "judge_failed"},
        rows[0] | omitted | {"_ng_failure_class": "kill_shaped"},
        rows[1] | {"_ng_failure_class": "judge_failed"},
        rows[1] | omitted | {"_ng_failure_class": "cancelled", "_ng_dispatch_drained": True},
    ]
    failures_path_for(output).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in sidecar))

    store = RolloutStore.read(output)

    assert [(row["_ng_task_index"], row["_ng_failure_class"]) for row in store.failures()] == [
        (0, "kill_shaped"),
        (1, "judge_failed"),
    ]
    assert [store.attempt_count(row) for row in rows] == [1, 1]
    coverage = store.coverage()
    assert (coverage["attempts"], coverage["counted_failures"]) == (3, 2)


def test_atomic_metadata_preserves_sharing_permissions(tmp_path):
    ordinary = tmp_path / "ordinary.json"
    ordinary.write_text("{}")
    output = tmp_path / "metadata.json"
    atomic_write_json(output, {"generation": 1})
    assert output.stat().st_mode & 0o777 == ordinary.stat().st_mode & 0o777
    output.chmod(0o640)
    atomic_write_json(output, {"generation": 2})
    assert output.stat().st_mode & 0o777 == 0o640
    assert orjson.loads(output.read_bytes()) == {"generation": 2}


def test_failed_atomic_publication_keeps_prior_metadata(tmp_path, monkeypatch):
    output = tmp_path / "metadata.json"
    atomic_write_json(output, {"generation": 1})
    before = output.read_bytes()

    def interrupted_replace(*args):
        raise OSError("interrupted publication")

    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(OSError, match="interrupted publication"):
        atomic_write_json(output, {"generation": 2})
    assert output.read_bytes() == before
    assert list(tmp_path.iterdir()) == [output]


def test_reverification_overwrite_preserves_existing_run(prepared_run):
    from nemo_gym.rollout_reverification import _prepare_output_fpaths

    output, prepare = prepared_run
    RolloutStore.start_or_resume(output, prepare, resume=False)
    before = snapshot(output)
    with pytest.raises(ConfigError, match="Cannot overwrite a manifest-backed run"):
        _prepare_output_fpaths("", str(output), False, True, False)
    assert snapshot(output) == before


@pytest.mark.parametrize("missing", ["output", "materialized"])
def test_resume_rejects_missing_artifacts_before_changing_saved_work(prepared_run, missing):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"reward": 0.0, "response": {}})
    paths = {"output": output, "materialized": materialized_path_for(output), "journal": journal_path_for(output)}
    paths[missing].unlink()
    before = snapshot(output)
    with pytest.raises(ConfigError, match="missing|without attempt history"):
        RolloutStore.start_or_resume(output, prepare, resume=True)
    assert snapshot(output) == before


def test_missing_current_source_requires_visible_identity_override(prepared_run):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        result = row | {"reward": 0.0, "response": {}}
        store.record_outcome(result)
    before = snapshot(output)
    prepare.side_effect = FileNotFoundError("Original source was removed")
    with pytest.raises(FileNotFoundError, match="Original source was removed"):
        RolloutStore.start_or_resume(output, prepare, resume=True)
    assert snapshot(output) == before
    with pytest.warns(UserWarning, match="allow_unsafe_resume"):
        resumed = RolloutStore.start_or_resume(output, prepare, resume=True, allow_unsafe=True)
    assert resumed.selected("success") == [result]
    assert not resumed.coverage()["identity_verified"]
    assert resumed.manifest.run_id == store.manifest.run_id
    saved = RunManifest.model_validate_json(manifest_path_for(output).read_bytes())
    assert saved.identity_overridden


@pytest.mark.parametrize("corruption", ["foreign", "conflict", "invalid_event"])
def test_unsafe_rebuild_does_not_hide_other_corruption(prepared_run, corruption):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for row in store.pending(3):
            store.record_dispatch(row)
            store.record_outcome(row | {"reward": 1.0, "response": {}})
    records = list(read_records(output))
    if corruption == "invalid_event":
        journal_path_for(output).write_bytes(b"{}\n")
    if corruption == "foreign":
        records[1]["_ng_run_id"] = "other-run"
    elif corruption == "conflict":
        records.append(records[0] | {"reward": 0.0})
    output.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in records))
    before = snapshot(output)
    with pytest.raises(ConfigError):
        RolloutStore.start_or_resume(output, prepare, resume=True, allow_unsafe=True)
    assert snapshot(output) == before


@pytest.mark.parametrize("cap, expected_exhausted", [(1, 1), (2, 0)])
def test_exhaustion_counts_failures_but_not_interruptions_terminal_or_completed(
    tmp_path, monkeypatch, cap, expected_exhausted
):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", str(cap))
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(6)]
    source = tmp_path / "source.jsonl"
    source.write_text("source")
    output = tmp_path / "out.jsonl"
    with RolloutStore.start_or_resume(
        output,
        lambda: (rows, RunManifest.create(source, rows, {}, {"agent": {"responses_api_agents": {"impl": {}}}})),
        resume=False,
    ) as store:
        dispatched = store.pending(cap)
        for row in dispatched[:5]:
            store.record_dispatch(row)
            store.record_dispatch(row)  # Repeated bookkeeping is not another attempt.
        store.record_outcome(dispatched[0] | {"reward": 0.0, "response": {}})
        store.record_outcome(dispatched[1] | {"_ng_failure_class": "agent_run_error"})
        store.record_outcome(dispatched[2] | {"_ng_failure_class": "timeout_exceeded", "_ng_failure_terminal": True})
        store.record_omission(dispatched[3], "intentional")
        # Row 4 has an unknown dispatched outcome; row 5 was never dispatched.
    reopened = RolloutStore.read(output)
    report = reopened.coverage()
    assert report["failure_budget_exhausted"] == expected_exhausted
    assert (report["failed"], report["missing"], report["not_started"], report["attempts"]) == (2, 1, 1, 5)
    from nemo_gym.rollout_records import coverage_path_for

    assert orjson.loads(coverage_path_for(output).read_bytes())["failure_budget_exhausted"] == expected_exhausted


@pytest.mark.parametrize("alias", [False, True])
async def test_journal_reverification_waits_for_followup_without_changing_files(prepared_run, monkeypatch, alias):
    from nemo_gym.rollout_reverification import RolloutReverificationConfig, RolloutReverificationHelper

    output, prepare = prepared_run
    RolloutStore.start_or_resume(output, prepare, resume=False)
    if alias:
        shortcut = output.with_name("shortcut.jsonl")
        shortcut.symlink_to(output)
        output = shortcut
    before = snapshot(output)
    config = RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(materialized_path_for(output)),
        rollouts_jsonl_fpath=str(output),
        output_jsonl_fpath=str(output),
        judge_failed_only=True,
        append=True,
    )
    with pytest.raises(ConfigError, match="Manifest-backed reverification is a follow-up"):
        await RolloutReverificationHelper().run_from_config(config)
    assert snapshot(output) == before
    # The final prefixed path must be checked too, including resume/append.
    from nemo_gym.rollout_reverification import _prepare_output_fpaths

    for append, resume in [(True, False), (False, True)]:
        with pytest.raises(ConfigError, match="Manifest-backed reverification is a follow-up"):
            _prepare_output_fpaths("", str(output), resume, False, append)
    assert snapshot(output) == before


@pytest.mark.parametrize("alias", [False, True])
def test_journal_health_waits_for_followup_without_reporting_stale_results(prepared_run, alias):
    from nemo_gym.rollout_health import run_health_checks

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"reward": 1.0, "response": {}})
        store.allocate_attempt(row)  # Interrupted newer attempt supersedes the saved answer.
    if alias:
        shortcut = output.with_name("shortcut.jsonl")
        shortcut.symlink_to(output)
        output = shortcut
    before = snapshot(output)
    with pytest.raises(ConfigError, match="histories with superseded outcomes"):
        run_health_checks(output, workers=1)
    assert snapshot(output) == before


@pytest.mark.parametrize("mismatch", ["run_id", "rollout_id", "attempt", "terminal", "failure_kind"])
def test_nested_failure_must_agree_with_saved_envelope(prepared_run, mismatch):
    from nemo_gym.core.episode import EpisodeId
    from nemo_gym.rollout_collection import _failure_compatibility_row
    from nemo_gym.rollout_outcomes import RolloutFailureRecord
    from nemo_gym.rollout_records import logical_rollout_id
    from nemo_gym.server_runtime.episode_types import EpisodeFailure

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        failure = RolloutFailureRecord(
            episode_id=EpisodeId(rollout_id=logical_rollout_id(row)),
            run_id=store.manifest.run_id,
            source="environment",
            delivery="delivered",
            failure=EpisodeFailure(failure_reason="Judge unavailable", terminal=False, failure_kind="judge_failed"),
        )
        result = row | _failure_compatibility_row(failure)
        nested = result["_ng_failure_record"]
        if mismatch == "run_id":
            nested["run_id"] = "foreign"
        elif mismatch in ("rollout_id", "attempt"):
            nested["episode_id"][mismatch] = "foreign" if mismatch == "rollout_id" else 7
        else:
            nested["failure"][mismatch] = True if mismatch == "terminal" else "agent_run_error"
        before = snapshot(output)
        with pytest.raises(ConfigError, match="inconsistent"):
            store.record_outcome(result)
        assert snapshot(output) == before
    failures_path_for(output).write_bytes(orjson.dumps(result) + b"\n")
    before = snapshot(output)
    with pytest.raises(ConfigError, match="inconsistent"):
        RolloutStore.read(output)
    assert snapshot(output) == before


@pytest.mark.parametrize("input_order", ["alias", "alias-first", "real-first"])
async def test_source_alias_uses_the_same_selected_history(prepared_run, monkeypatch, input_order):
    from unittest.mock import AsyncMock

    from nemo_gym.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper, RolloutCollectionHelper
    from nemo_gym.rollout_records import coverage_path_for

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"reward": 1.0, "response": {}})
        store.record_dispatch(row | {"_ng_attempt_index": 1})
    alias = output.with_name("shortcut.jsonl")
    alias.symlink_to(output)
    paths = {"alias": [alias], "alias-first": [alias, output], "real-first": [output, alias]}[input_order]
    before = snapshot(output)
    assert RolloutStore.read(alias).selected("success") == []
    aggregate = AsyncMock(return_value=None)
    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", aggregate)
    destination = output.with_name("aggregate.jsonl")
    await RolloutAggregationHelper().run_from_config(
        RolloutAggregationConfig(
            input_glob=",".join(map(str, paths)),
            output_jsonl_fpath=str(destination),
            disable_health_check=True,
        )
    )
    assert aggregate.call_args.args[0] == []
    report = orjson.loads(coverage_path_for(destination).read_bytes())
    assert report["successful"] == 0 and report["not_started"] == report["missing"] == 1 and len(report["shards"]) == 1
    assert all((output.parent / name).read_bytes() == data for name, data in before.items())


@pytest.mark.parametrize("retry_timeouts", [False, True])
def test_offline_coverage_declares_and_applies_timeout_retry_policy(prepared_run, monkeypatch, retry_timeouts):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "1")
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(1, retry_terminal_timeouts=retry_timeouts)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"_ng_failure_class": "timeout_exceeded", "_ng_failure_terminal": True})
        online = store.coverage()
    offline = RolloutStore.read(output).coverage()
    assert offline == online
    assert offline["retry_terminal_timeouts"] is retry_timeouts
    assert offline["failure_budget_exhausted"] == int(retry_timeouts)


@pytest.mark.parametrize("previous", ["success", "failure", "none"])
@pytest.mark.parametrize("other", ["failure", "unscored", "omitted"])
async def test_missing_zero_uses_latest_attempt_without_changing_recovery(prepared_run, monkeypatch, previous, other):
    from nemo_gym.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper, RolloutCollectionHelper
    from nemo_gym.rollout_records import coverage_path_for

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row, second = store.pending(3)
        store.record_dispatch(row)
        if previous != "none":
            payload = {"reward": 1.0} if previous == "success" else {"_ng_failure_class": "agent_run_error"}
            store.record_outcome(row | payload)
            # The retry was dispatched but never saved an outcome. The older
            # success/failure must not hide the new unknown attempt.
            store.record_dispatch(row | {"_ng_attempt_index": 1})
        store.record_dispatch(second)
        other_payload = {
            "failure": {"_ng_failure_class": "agent_run_error"},
            "unscored": {"execute_only": True, "response": {}},
            "omitted": {"_ng_failure_class": "skipped", "_ng_failure_terminal": True},
        }[other]
        store.record_outcome(second | other_payload)
    before = snapshot(output)
    metrics = AsyncMock(return_value=None)
    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", metrics)
    merged = output.parent / "aggregate" / "merged.jsonl"
    await RolloutAggregationHelper().run_from_config(
        RolloutAggregationConfig(
            input_glob=str(output),
            output_jsonl_fpath=str(merged),
            count_missing_rollouts_as_zero=True,
            disable_health_check=True,
        )
    )
    supplied = metrics.await_args.args[0]
    zeros = [row for row in supplied if "reward" in row]
    assert [(row["_ng_task_index"], row["reward"]) for row in zeros] == [(0, 0.0)]
    assert len(supplied) == (2 if other == "unscored" else 1)
    report = orjson.loads(coverage_path_for(merged).read_bytes())
    assert (report["expected"], report["missing"], report["imputed"], report["scored"]) == (2, 1, 1, 1)
    assert report["failures_counted_as_zero"] == 0
    assert not report["complete"]
    assert snapshot(output) == before
    assert 0 in [row["_ng_task_index"] for row in RolloutStore.read(output).pending(3)]


async def test_missing_zero_keeps_separate_journal_run_identities(prepared_run, monkeypatch):
    from nemo_gym.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper, RolloutCollectionHelper
    from nemo_gym.rollout_records import coverage_path_for

    output, prepare = prepared_run
    other = output.with_name("second.jsonl")
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"reward": 1.0})
    with RolloutStore.start_or_resume(other, prepare, resume=False):
        pass
    metrics = AsyncMock(return_value=None)
    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", metrics)
    merged = output.parent / "merged.jsonl"
    await RolloutAggregationHelper().run_from_config(
        RolloutAggregationConfig(
            input_glob=f"{output},{other}",
            output_jsonl_fpath=str(merged),
            count_missing_rollouts_as_zero=True,
            disable_health_check=True,
        )
    )
    supplied = metrics.await_args.args[0]
    assert len(supplied) == 4
    assert sum(row["reward"] for row in supplied) == 1.0
    report = orjson.loads(coverage_path_for(merged).read_bytes())
    assert (report["expected"], report["successful"], report["imputed"], report["not_started"]) == (4, 1, 3, 3)
    assert not report["complete"]


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("append_before_manifest", [False, True])
def test_live_reader_does_not_consume_outcomes_newer_than_its_manifest(
    prepared_run, monkeypatch, failure, append_before_manifest
):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as writer:
        first, second = writer.pending(3)
        writer.record_dispatch(first)
        writer.record_outcome(first | {"reward": 1.0})
        read_bytes = Path.read_bytes
        appended = False

        def concurrent_read(path):
            nonlocal appended
            if path != manifest_path_for(output) or appended:
                return read_bytes(path)
            appended = True
            old_manifest = read_bytes(path)
            writer.record_dispatch(second)
            writer.record_outcome(second | ({"_ng_failure_class": "judge_failed"} if failure else {"reward": 0.0}))
            return read_bytes(path) if append_before_manifest else old_manifest

        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", concurrent_read)
            reader = RolloutStore.read(output)
        assert appended
        assert reader.selected("success") == [first | {"reward": 1.0}]
        assert reader.failures() == []
        assert reader.coverage()["missing" if append_before_manifest else "not_started"] == 1
    current = RolloutStore.read(output)
    assert current.coverage()["successful"] == (1 if failure else 2)
    assert current.coverage()["failed"] == int(failure)


def test_live_snapshot_does_not_relax_reservation_validation(prepared_run):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as writer:
        row = writer.pending(3)[0]
        with output.open("ab") as file:
            file.write(orjson.dumps(row | {"reward": 1.0}) + b"\n")
        with pytest.raises(ConfigError, match="no reserved attempt"):
            RolloutStore.read(output)


def test_health_reads_success_file_after_failed_attempt_is_retried(prepared_run):
    from nemo_gym.rollout_health import run_health_checks
    from nemo_gym.rollout_store import raw_outcomes_are_selected

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as writer:
        row = writer.pending(3)[0]
        writer.record_dispatch(row)
        writer.record_outcome(row | {"_ng_failure_class": "judge_failed"})
        writer.allocate_attempt(row)
        writer.record_outcome(row | {"reward": 1.0, "response": {}})
    assert not raw_outcomes_are_selected(output)  # Reverification still considers both files.
    assert run_health_checks(output, workers=1).summary["run"]["artifacts"]["records"] == 1


@pytest.mark.parametrize("superseded", [False, True])
@pytest.mark.parametrize("answer_key", ["response", "_ng_failure_partial_response"])
async def test_judge_only_reads_selected_source_into_separate_output(
    prepared_run, monkeypatch, superseded, answer_key
):
    import nemo_gym.rollout_reverification as reverify

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        first, second = store.pending(3)
        store.record_dispatches([first, second])
        store.record_outcome(first | {"reward": 0.25, "response": {"answer": "already scored"}})
        failed = second | {"_ng_failure_class": "judge_failed", answer_key: {"answer": "saved answer"}}
        store.record_outcome(failed)
        if superseded:
            # Construct an explicitly superseding historical verification attempt; normal resume stays blocked.
            assert second not in store.pending(3)
            retry = dict(second)
            store.allocate_attempt(retry)
            store.record_dispatch(retry)
            store.record_outcome(retry | {"_ng_failure_class": "judge_failed", "response": {"answer": "new answer"}})
    before = snapshot(output)
    destination = output.with_name("rejudged.jsonl")
    verified = []

    async def verify(row):
        verified.append(row)
        return row, {"reward": 1.0, "response": row["response"]}

    monkeypatch.setattr(
        reverify, "_run_verification_payloads", lambda payloads, **kwargs: [verify(row) for row in payloads]
    )
    config = reverify.RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(materialized_path_for(output)),
        rollouts_jsonl_fpath=str(output),
        output_jsonl_fpath=str(destination),
        judge_failed_only=True,
        disable_aggregation=True,
        upload_rollouts=False,
    )
    if superseded:
        with pytest.raises(ConfigError, match="superseded outcomes"):
            await reverify.RolloutReverificationHelper().run_from_config(config)
        assert not destination.exists() and not verified
    else:
        await reverify.RolloutReverificationHelper().run_from_config(config)
        assert [row["response"] for row in verified] == [{"answer": "saved answer"}]
        assert [row["reward"] for row in read_records(destination)] == [0.25, 1.0]
    assert all((output.parent / name).read_bytes() == data for name, data in before.items())


@pytest.mark.parametrize(
    "saved",
    [
        {"_ng_failure_class": "judge_failed", "_ng_failure_response": {"answer": "diagnostic only"}},
        {"_ng_failure_class": "agent_run_error", "_ng_failure_partial_response": {"answer": "unfinished"}},
    ],
)
def test_judge_recovery_does_not_promote_diagnostics_to_saved_answers(saved):
    from nemo_gym.rollout_reverification import _rollout_response

    with pytest.raises(ConfigError, match="has no `response`"):
        _rollout_response(saved)


@pytest.mark.parametrize("stop_after_reset", [False, True])
def test_resubmit_budget_survives_reopen_without_reusing_attempt_ids(prepared_run, stop_after_reset, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "2")
    from nemo_gym.utils.path_utils import resubmit_offset_path_for

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        first, second = store.pending(2)
        store.record_dispatch(first)
        store.record_outcome(first | {"reward": 1.0})
        for attempt in (0, 1):
            row = second | {"_ng_attempt_index": attempt}
            store.record_dispatch(row)
            store.record_outcome(row | {"_ng_failure_class": "agent_run_error"})
    before = snapshot(output)
    assert RolloutStore.read(output).pending(2) == []
    store = RolloutStore.start_or_resume(output, prepare, resume=False, resubmit=True)
    assert all(
        path.read_bytes() == before[path.name]
        for path in (output, failures_path_for(output), manifest_path_for(output))
    )
    if stop_after_reset:
        store = RolloutStore.start_or_resume(output, prepare, resume=True)
    for attempt in (2, 3):
        with store:
            (row,) = store.pending(2)
            assert row["_ng_task_index"] == 1
            assert row["_ng_attempt_index"] == attempt
            store.record_dispatch(row)
            store.record_outcome(row | {"_ng_failure_class": "agent_run_error"})
        store = RolloutStore.start_or_resume(output, prepare, resume=True)
        offline = RolloutStore.read(output)
        assert offline.coverage() == store.coverage()
        assert offline.coverage()["counted_failures"] == attempt - 1
    assert store.pending(2) == []
    assert resubmit_offset_path_for(output).read_text() == "2\n"
    assert len(list(read_records(failures_path_for(output)))) == 4
    assert len(store.selected("success")) == 1
    assert store.coverage()["attempts"] == 5


@pytest.mark.parametrize("tail", [b"", b"\n\n", b'{"interrupted":'])
def test_resubmit_repaired_tail_does_not_exempt_new_failures(prepared_run, tail, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "1")
    from nemo_gym.utils.path_utils import resubmit_offset_path_for

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for row in store.pending(1):
            store.record_dispatch(row)
            store.record_outcome(
                row | {"_ng_failure_class": "agent_run_error", "_ng_failure_terminal": row["_ng_task_index"] == 0}
            )
    with failures_path_for(output).open("ab") as file:
        file.write(tail)
    with RolloutStore.start_or_resume(output, prepare, resume=True, resubmit=True) as store:
        (row,) = store.pending(1)
        assert row["_ng_task_index"] == 1
        assert row["_ng_attempt_index"] == 1
        assert int(resubmit_offset_path_for(output).read_text()) == (4 if tail == b"\n\n" else 2)
        store.record_dispatch(row)
        store.record_outcome(row | {"_ng_failure_class": "agent_run_error"})
    offline = RolloutStore.read(output)
    assert offline.pending(1) == []
    assert offline.coverage()["counted_failures"] == 1
    assert offline.selected("failure")[0]["_ng_failure_terminal"] is True


def test_resubmit_identity_refusal_does_not_reset_budget(prepared_run, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "1")
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(1)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"_ng_failure_class": "agent_run_error"})
    source = output.parent / "source.jsonl"
    source.write_bytes(source.read_bytes() + b"\n")
    before = snapshot(output)
    with pytest.raises(ConfigError, match="incompatible source dataset"):
        RolloutStore.start_or_resume(output, prepare, resume=True, resubmit=True)
    assert snapshot(output) == before


def test_resubmit_offset_is_part_of_alias_ownership(prepared_run):
    from nemo_gym.utils.path_utils import resubmit_offset_path_for

    output, prepare = prepared_run
    RolloutStore.start_or_resume(output, prepare, resume=False)
    alias = output.with_name("alias.jsonl")
    alias.symlink_to(output)
    resubmit_offset_path_for(alias).write_text("0\n")
    before = snapshot(output)
    with pytest.raises(ConfigError, match="beside the rollout alias"):
        RolloutStore.start_or_resume(alias, prepare, resume=True, resubmit=True)
    assert snapshot(output) == before


def test_resubmit_legacy_main_file_failures_match_offline_budget(prepared_run, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "1")
    output, prepare = prepared_run
    rows, _ = prepare()
    materialized_path_for(output).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    output.write_bytes(
        orjson.dumps(rows[0] | {"_ng_failure_class": "agent_run_error"})
        + b"\n"
        + orjson.dumps(rows[1] | {"reward": 1.0})
        + b"\n"
    )
    with RolloutStore.start_or_resume(output, prepare, resume=True, allow_unsafe=True, resubmit=True) as store:
        assert store.coverage()["counted_failures"] == 0
        assert RolloutStore.read(output).coverage() == store.coverage()
        (row,) = store.pending(1)
        assert row["_ng_task_index"] == 0 and row["_ng_attempt_index"] == 1
        store.record_dispatch(row)
        store.record_outcome(row | {"_ng_failure_class": "agent_run_error"})
    offline = RolloutStore.read(output)
    assert offline.coverage()["counted_failures"] == 1
    assert offline.pending(1) == []


@pytest.mark.parametrize("value", ["-1", "broken"])
def test_resubmit_invalid_offset_is_actionable_and_preserves_history(prepared_run, value):
    from nemo_gym.utils.path_utils import resubmit_offset_path_for

    output, prepare = prepared_run
    RolloutStore.start_or_resume(output, prepare, resume=False)
    resubmit_offset_path_for(output).write_text(value)
    before = snapshot(output)
    with pytest.raises(ConfigError, match="resubmit budget offset"):
        RolloutStore.start_or_resume(output, prepare, resume=True)
    assert snapshot(output) == before


def test_resubmit_offset_helper_keeps_existing_public_contract(tmp_path):
    from nemo_gym.rollout_collection import load_resubmit_offset, record_resubmit_offset

    output = tmp_path / "new" / "run.jsonl"
    assert load_resubmit_offset(output) == 0
    assert record_resubmit_offset(output) == 0
    failures_path_for(output).write_bytes(b'{"failure": 1}\n\n')
    assert record_resubmit_offset(output) == 2
    assert load_resubmit_offset(output) == 2


def test_saved_retry_cap_drives_resume_exporters_and_health(prepared_run, monkeypatch):
    from nemo_gym.health.types import RunHealthLimits
    from nemo_gym.rollout_collection import _failure_export_metrics
    from nemo_gym.rollout_health import run_health_checks, write_provisional_summary

    output, prepare = prepared_run
    rows, _ = prepare()
    rows.extend({"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(2, 5))
    (output.parent / "source.jsonl").write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "1")
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        success, failure, omitted, interrupted, _ = store.pending()
        store.record_dispatches([success, failure, interrupted])
        store.record_outcome(success | {"reward": 1.0})
        store.record_outcome(failure | {"_ng_failure_class": "agent_run_error"})
        store.record_omission(omitted, "Intentionally excluded")
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "99")
    resumed = RolloutStore.start_or_resume(output, prepare, resume=True)
    reader = RolloutStore.read(output)
    assert reader.manifest.max_rollout_attempts == 1
    assert resumed.coverage() == reader.coverage()
    assert [row["_ng_task_index"] for row in resumed.pending()] == [3, 4]
    coverage = reader.coverage()
    assert (coverage["failed_final"], coverage["failed_not_final"], coverage["failure_budget_exhausted"]) == (1, 0, 1)
    assert (coverage["missing"], coverage["not_started"], coverage["intentionally_omitted"]) == (1, 1, 1)
    metrics = _failure_export_metrics([], lambda row: None, set(), histories=[reader])
    assert metrics["retries/max_attempts"] == metrics["retries/at_cap"] == 1
    assert metrics["coverage/failed/agent_run_error"] == 1
    assert metrics["coverage/failed/skipped"] == 0
    limits = RunHealthLimits(max_final_failure_fraction=0.3)
    full = run_health_checks(output, workers=1, limits=limits).summary["run"]["artifacts"]
    assert full["failed"] == {"total": 1, "final": 1, "not_final": 0, "by_class": {"agent_run_error": 1}}
    assert (full["missing"], full["not_started"], full["omitted"]) == (1, 1, 1)
    assert "final failures" not in " ".join(full["unhealthy_reasons"])
    monkeypatch.setattr(RolloutStore, "read", lambda *a, **kw: pytest.fail("SIGTERM must not reload trajectories"))
    summary = write_provisional_summary(output, {(0, 0)}, limits=limits, run_coverage=coverage)
    provisional = orjson.loads(summary.read_bytes())["run"]["artifacts"]
    for key in ("expected", "failed", "missing", "not_started", "omitted", "unhealthy_reasons"):
        assert provisional[key] == full[key]


def test_resubmit_and_interruptions_are_free_in_exported_budget(prepared_run, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "2")
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        first, second = store.pending()
        store.record_dispatch(second)
        store.record_outcome(second | {"reward": 1.0})
        for attempt in range(2):
            row = first | {"_ng_attempt_index": attempt}
            store.record_dispatch(row)
            store.record_outcome(row | {"_ng_failure_class": "agent_run_error"})
    assert RolloutStore.read(output).failure_export_metrics()["retries/at_cap"] == 1
    with RolloutStore.start_or_resume(output, prepare, resume=True, resubmit=True) as store:
        (row,) = store.pending()
        store.record_dispatch(row)
        store.record_outcome(
            row | {"_ng_failure_class": "agent_request_failed", "_ng_failure_legacy_class": "kill_shaped"}
        )
        # The interruption names no eligible cause, so automatic recovery stops instead of running it again.
        assert store.pending() == []
    reader = RolloutStore.read(output)
    assert reader.failure_export_metrics()["retries/at_cap"] == reader.coverage()["failure_budget_exhausted"] == 0
    assert reader.failure_export_metrics()["retries/attempts"] == 3
    # The resubmit's fresh budget and the interruption spend nothing; the earlier attempts stay recorded.
    assert reader.coverage()["counted_failures"] == 0
    assert reader.pending() == []


def test_newer_unknown_attempt_supersedes_failure_in_health(prepared_run):
    from nemo_gym.rollout_health import run_health_checks

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        first, second = store.pending()
        store.record_dispatches([first, second])
        store.record_outcome(first | {"_ng_failure_class": "agent_run_error"})
        store.record_outcome(second | {"reward": 1.0})
        store.allocate_attempt(first)
    artifacts = run_health_checks(output, workers=1).summary["run"]["artifacts"]
    assert artifacts["failed"]["total"] == 0
    assert artifacts["missing"] == 1
    assert not artifacts["healthy"]


def test_older_manifest_records_retry_cap_on_first_validated_resume(prepared_run, monkeypatch):
    output, prepare = prepared_run
    RolloutStore.start_or_resume(output, prepare, resume=False)
    saved = orjson.loads(manifest_path_for(output).read_bytes())
    saved.pop("max_rollout_attempts")
    atomic_write_json(manifest_path_for(output), saved)
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "5")
    resumed = RolloutStore.start_or_resume(output, prepare, resume=True)
    assert resumed.manifest.max_rollout_attempts == 5
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "1")
    assert RolloutStore.read(output).coverage()["max_rollout_attempts"] == 5
    assert RolloutStore.start_or_resume(output, prepare, resume=True).coverage()["max_rollout_attempts"] == 5


@pytest.mark.parametrize("retry_timeouts", [False, True])
def test_health_recheck_keeps_the_saved_timeout_retry_policy(prepared_run, retry_timeouts):
    from nemo_gym.health.types import RunHealthLimits
    from nemo_gym.rollout_health import health_check_run_dir, run_health_checks

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        success, timeout = store.pending(retry_terminal_timeouts=retry_timeouts)
        for row in (success, timeout):
            store.record_dispatch(row)
        store.record_outcome(success | {"reward": 1.0, "response": {}})
        store.record_outcome(timeout | {"_ng_failure_class": "timeout_exceeded", "_ng_failure_terminal": True})
        coverage = store.coverage()
    limits = RunHealthLimits(max_final_failure_fraction=0.3)
    online = run_health_checks(output, workers=1, limits=limits, run_coverage=[coverage]).summary["run"]["artifacts"]
    offline = health_check_run_dir(output.parent, workers=1).summary["run"]["artifacts"]
    assert online["failed"] == offline["failed"]
    assert offline["failed"]["final"] == int(not retry_timeouts)
    assert offline["healthy"] == online["healthy"] == retry_timeouts
    assert RolloutStore.read(output).coverage() == coverage


def test_older_manifest_defaults_to_no_timeout_override_until_resumed(prepared_run):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending()[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"_ng_failure_class": "timeout_exceeded", "_ng_failure_terminal": True})
    path = manifest_path_for(output)
    legacy = orjson.loads(path.read_bytes())
    legacy.pop("retry_terminal_timeouts")
    path.write_bytes(orjson.dumps(legacy))
    before = snapshot(output)
    assert RolloutStore.read(output).coverage()["failed_final"] == 1
    assert snapshot(output) == before
    with RolloutStore.start_or_resume(output, prepare, resume=True) as store:
        assert len(store.pending(retry_terminal_timeouts=True)) == 2
    assert RolloutStore.read(output).coverage()["failed_not_final"] == 1
    assert orjson.loads(path.read_bytes())["retry_terminal_timeouts"] is True


@pytest.mark.parametrize("retry_timeouts", [False, True])
def test_a_producer_final_zero_settles_its_rollout_through_save_and_reload(tmp_path, retry_timeouts) -> None:
    from nemo_gym.rollout_recovery import RunManifest
    from tests.unit_tests.test_rollout_collection import TestRolloutAggregationHelper

    rollout = TestRolloutAggregationHelper._rollout
    output = tmp_path / "run" / "rollouts.jsonl"
    output.parent.mkdir()
    inputs = [rollout(task, "agent_a") for task in range(4)]
    source = output.parent / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in inputs))
    manifest = RunManifest.create(source, inputs, {}, {"agent_a": {"responses_api_agents": {"impl": {}}}})
    marker = {"_ng_failure_class": "timeout_exceeded", "_ng_failure_counts_as_zero": True}
    with RolloutStore.start_or_resume(output, lambda: (inputs, manifest), resume=False) as store:
        for row in store.pending(retry_terminal_timeouts=retry_timeouts):
            store.record_dispatch(row)
            task = row["_ng_task_index"]
            if task == 0:
                store.record_outcome(row | {"reward": 1.0, "response": {}})
            elif task == 1:  # settled by its producer: final and marked
                store.record_outcome(row | marker | {"_ng_failure_terminal": True})
            elif task == 2:  # marked but not final: resume may still run it
                store.record_outcome(row | marker | {"_ng_failure_terminal": False})
            else:  # a class the reporting knob may count, without the producer's marker
                store.record_outcome(row | {"_ng_failure_class": "agent_run_error", "_ng_failure_terminal": True})
        live = store.coverage()
    reloaded = RolloutStore.read(output)
    for coverage in (live, reloaded.coverage()):
        assert (coverage["successful"], coverage["final_zeros"], coverage["failed"]) == (1, 1, 3)
        assert coverage["failed_final"] >= 1 and coverage["complete"] is False
    # Neither the saved timeout policy nor an explicit one reopens the settled rollout as another model chance.
    pending = {row["_ng_task_index"] for row in reloaded.pending(retry_terminal_timeouts=retry_timeouts)}
    assert 1 not in pending and 2 in pending
    [settled] = [row for row in reloaded.failures() if row["_ng_task_index"] == 1]
    assert settled["_ng_failure_counts_as_zero"] is True and "reward" not in settled


def test_an_interruption_with_no_recorded_cause_stops_automatic_recovery_and_keeps_its_debits(tmp_path) -> None:
    from nemo_gym.rollout_recovery import RunManifest
    from tests.unit_tests.test_rollout_collection import TestRolloutAggregationHelper

    rollout = TestRolloutAggregationHelper._rollout
    output = tmp_path / "run" / "rollouts.jsonl"
    output.parent.mkdir()
    inputs = [rollout(0, "agent_a"), rollout(1, "agent_a")]
    source = output.parent / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in inputs))
    manifest = RunManifest.create(source, inputs, {}, {"agent_a": {"responses_api_agents": {"impl": {}}}})
    with RolloutStore.start_or_resume(output, lambda: (inputs, manifest), resume=False) as store:
        for row in store.pending():
            store.record_dispatch(row)
            store.record_outcome(row | {"_ng_failure_class": "agent_run_error"})
        [first] = [row for row in store.pending() if row["_ng_task_index"] == 0]
        store.record_dispatch(first)
        store.record_outcome(first | {"_ng_failure_class": "kill_shaped"})
    reloaded = RolloutStore.read(output)
    assert {row["_ng_task_index"] for row in reloaded.pending()} == {1}
    coverage = reloaded.coverage()
    # The earlier counted failure stays debited; the interruption itself spends nothing and grades nothing.
    assert coverage["counted_failures"] == 2 and coverage["successful"] == 0


@pytest.mark.parametrize(
    ("stage", "partial", "retried"),
    [
        ("admission", None, True),  # a known cause before any answer: resume may run it again
        ("verification", None, False),  # the answer exists and waits on verification: never regenerated
        (
            None,
            {"id": "kept", "status": "completed", "output": [{"type": "message", "role": "assistant"}]},
            False,
        ),
    ],
)
def test_a_known_interruption_retries_only_before_an_answer(prepared_run, stage, partial, retried) -> None:
    from nemo_gym.core.episode import EpisodeId
    from nemo_gym.rollout_collection import _failure_compatibility_row
    from nemo_gym.rollout_outcomes import RolloutFailureRecord
    from nemo_gym.rollout_records import logical_rollout_id
    from nemo_gym.server_runtime.episode_types import EpisodeFailure

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        failure = RolloutFailureRecord(
            episode_id=EpisodeId(rollout_id=logical_rollout_id(row), attempt=row.get("_ng_attempt_index", 0)),
            run_id=store.manifest.run_id,
            source="collector",
            delivery="not_sent",
            failure=EpisodeFailure(failure_reason="interrupted", terminal=False, stage=stage),
        )
        result = row | _failure_compatibility_row(failure) | {"_ng_failure_legacy_class": "kill_shaped"}
        if partial is not None:
            result["_ng_failure_partial_response"] = partial
        store.record_outcome(result)
    reloaded = RolloutStore.read(output)
    pending = {logical_rollout_id(r) for r in reloaded.pending()}
    assert (logical_rollout_id(row) in pending) is retried


@pytest.mark.parametrize("saved_answer", [True, False])
def test_a_delivered_verification_failure_with_its_answer_is_not_regenerated(prepared_run, saved_answer) -> None:
    from nemo_gym.core.episode import EpisodeId
    from nemo_gym.rollout_collection import _failure_compatibility_row
    from nemo_gym.rollout_outcomes import RolloutFailureRecord
    from nemo_gym.rollout_records import logical_rollout_id
    from nemo_gym.server_runtime.episode_types import EpisodeFailure

    answer = {
        "id": "retained",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "status": "completed", "content": []}],
    }
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        failure = RolloutFailureRecord(
            episode_id=EpisodeId(rollout_id=logical_rollout_id(row), attempt=row.get("_ng_attempt_index", 0)),
            run_id=store.manifest.run_id,
            source="environment",
            delivery="delivered",
            failure=EpisodeFailure(
                failure_reason="judge unavailable",
                terminal=False,
                failure_kind="judge_failed",
                stage="verification" if saved_answer else "agent",
            ),
        )
        result = row | _failure_compatibility_row(failure, verification_response=answer if saved_answer else None)
        store.record_outcome(result)
    reloaded = RolloutStore.read(output)
    pending = {logical_rollout_id(r) for r in reloaded.pending()}
    # A judge failure never reruns the agent: a missing saved answer is unsupported, not retryable.
    assert logical_rollout_id(row) not in pending
    [saved] = [r for r in reloaded.failures() if logical_rollout_id(r) == logical_rollout_id(row)]
    assert (saved.get("response") == answer) is saved_answer
    assert reloaded.coverage()["counted_failures"] == 1


@pytest.mark.parametrize("answered", [True, False])
def test_a_reservation_without_outcome_never_regenerates_an_earlier_saved_answer(prepared_run, answered) -> None:
    from nemo_gym.rollout_records import logical_rollout_id

    answer = {"id": "kept", "status": "completed", "output": [{"type": "message", "role": "assistant"}]}
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"_ng_failure_class": "judge_failed", "response": answer if answered else {}})
    with RolloutStore.start_or_resume(output, prepare, resume=True) as store:
        store.record_dispatch(row | {"_ng_attempt_index": 1})  # reserved, then the run ended before any outcome
    reloaded = RolloutStore.read(output)
    pending = {logical_rollout_id(r) for r in reloaded.pending()}
    assert logical_rollout_id(row) not in pending
    assert reloaded.coverage()["counted_failures"] == 1


@pytest.mark.parametrize("resubmit", [False, True], ids=["resume", "explicit-budget-reset"])
@pytest.mark.parametrize(
    "case", ["final-zero", "unknown-kill", "infrastructure", "exhausted-infrastructure", "unstarted"]
)
def test_missing_reservation_preserves_prior_replay_gate_and_explicit_budget_reset(
    prepared_run, monkeypatch, case, resubmit
) -> None:
    from nemo_gym.rollout_records import logical_rollout_id

    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "3")
    output, prepare = prepared_run
    prior_failures = 2 if case == "exhausted-infrastructure" else 1
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending()[0]
        if case != "unstarted":
            for attempt in range(prior_failures):
                charged = row | {"_ng_attempt_index": attempt}
                store.record_dispatch(charged)
                store.record_outcome(charged | {"_ng_failure_class": "transport_timeout", "_ng_elapsed_s": 1.25})
            prior = row | {"_ng_attempt_index": prior_failures}
            payload = {"_ng_failure_class": "transport_timeout", "_ng_elapsed_s": 2.5}
            if case == "final-zero":
                payload.update(
                    _ng_failure_class="model_policy_response_timeout",
                    _ng_failure_terminal=True,
                    _ng_failure_counts_as_zero=True,
                )
            elif case == "unknown-kill":
                payload["_ng_failure_class"] = "kill_shaped"
            store.record_dispatch(prior)
            store.record_outcome(prior | payload)
        # Simulate a reservation made by an older process that stopped before writing any outcome.
        reserved_attempt = 0 if case == "unstarted" else prior_failures + 1
        store.record_dispatch(row | {"_ng_attempt_index": reserved_attempt})
    sidecar = output.with_name(output.stem + "_failures.jsonl")
    saved_history = sidecar.read_bytes() if sidecar.exists() else None
    charged_failures = (
        0 if case == "unstarted" else prior_failures + (case in {"infrastructure", "exhausted-infrastructure"})
    )
    assert RolloutStore.read(output).coverage()["counted_failures"] == charged_failures

    with RolloutStore.start_or_resume(output, prepare, resume=True, resubmit=resubmit) as store:
        pending = {logical_rollout_id(item): item for item in store.pending()}
        eligible = case not in {"final-zero", "unknown-kill"} and (resubmit or charged_failures < 3)
        assert (logical_rollout_id(row) in pending) is eligible
        if eligible:
            assert pending[logical_rollout_id(row)]["_ng_attempt_index"] == reserved_attempt + 1
        assert store.coverage()["counted_failures"] == (0 if resubmit else charged_failures)
    reloaded = RolloutStore.read(output)
    assert (logical_rollout_id(row) in {logical_rollout_id(item) for item in reloaded.pending()}) is eligible
    # The latest reservation remains missing; a reset changes the budget, never the retained outcome or elapsed time.
    assert reloaded.coverage()["missing"] == 1
    assert reloaded.coverage()["counted_failures"] == (0 if resubmit else charged_failures)
    assert (sidecar.read_bytes() if sidecar.exists() else None) == saved_history


@pytest.mark.parametrize("new_attempt", [False, True])
def test_current_unrouted_dispatch_gates_admission_without_replacing_history(prepared_run, new_attempt):
    from nemo_gym.omitted_rollouts import omitted_rollout_record

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        original = row | {"_ng_failure_class": "provider_unavailable", "reward": 0.0}
        store.record_outcome(original)
        diagnostic = row
        if new_attempt:
            diagnostic = row | {"_ng_attempt_index": 1}
            store.record_dispatch(diagnostic)
        store.record_unstarted(
            omitted_rollout_record(diagnostic | {"_ng_failure_class": "agent_run_error"})
            | {"_ng_unrouted_run_error": True}
        )
    reader = RolloutStore.read(output)
    assert not any(item["_ng_task_index"] == 0 for item in reader.pending(3))
    assert reader.coverage()["counted_failures"] == 1
    assert list(read_records(failures_path_for(output)))[0] == original
    # A genuinely newer valid outcome supersedes that diagnostic; no historical blocker leaks forward.
    with RolloutStore.start_or_resume(output, prepare, resume=True) as store:
        later = row | {"_ng_attempt_index": 2 if new_attempt else 1}
        store.record_dispatch(later)
        store.record_outcome(later | {"_ng_failure_class": "provider_unavailable", "reward": 0.0})
    reader = RolloutStore.read(output)
    assert any(item["_ng_task_index"] == 0 for item in reader.pending(3))
    assert reader.coverage()["counted_failures"] == 2


@pytest.mark.parametrize("legacy_selector", [False, True])
async def test_offline_dropped_summary_excludes_producer_and_selected_zeros(
    tmp_path, monkeypatch, capsys, legacy_selector
):
    from nemo_gym.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper, RolloutCollectionHelper
    from nemo_gym.rollout_records import coverage_path_for

    output = tmp_path / "rollouts.jsonl"
    inputs = [{"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(3)]
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in inputs))
    manifest = RunManifest.create(source, inputs, {}, {"agent": {"responses_api_agents": {"impl": {}}}})
    with RolloutStore.start_or_resume(output, lambda: (inputs, manifest), resume=False) as store:
        for row in store.pending():
            store.record_dispatch(row)
            payload = (
                {"reward": 1.0, "response": {}}
                if row["_ng_task_index"] == 0
                else {
                    "_ng_failure_class": "model_policy_response_timeout",
                    "_ng_failure_legacy_class": "timeout_exceeded",
                    "_ng_failure_terminal": True,
                    "_ng_failure_counts_as_zero": not legacy_selector,
                }
                if row["_ng_task_index"] == 1
                else {"_ng_failure_class": "judge_failed"}
            )
            store.record_outcome(row | payload)
    before = snapshot(output)
    metrics = AsyncMock(return_value=None)
    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", metrics)
    merged = tmp_path / "aggregate" / "merged.jsonl"
    await RolloutAggregationHelper().run_from_config(
        RolloutAggregationConfig(
            input_glob=str(output),
            output_jsonl_fpath=str(merged),
            count_failure_classes_as_zero=["timeout_exceeded"] if legacy_selector else [],
            disable_health_check=True,
        )
    )
    assert [row["reward"] for row in metrics.await_args.args[0]] == [1.0, 0.0]
    report = orjson.loads(coverage_path_for(merged).read_bytes())
    assert report["failures_counted_as_zero"] == 1
    printed = capsys.readouterr().out
    assert "1 judge_failed" in printed
    assert "1 model_policy_response_timeout" not in printed
    assert snapshot(output) == before


def test_rejudgment_settles_the_saved_judge_failure_without_a_model_attempt(prepared_run):
    output, prepare = prepared_run
    answer = {"object": "response", "id": "resp_saved", "status": "completed", "output": []}
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        failure = row | {"_ng_failure_class": "judge_failed", "_ng_failure_id": "f-1", "response": answer}
        store.record_outcome(failure)
    failures_before = failures_path_for(output).read_bytes()
    # A re-judged result carries no failure fields of its own; the failure stays in the observations it retains.
    result = {k: v for k, v in failure.items() if not k.startswith("_ng_failure")} | {
        "reward": 0.0,
        "_ng_rejudged_failure_id": "f-1",
    }
    with RolloutStore.open_for_rejudgment(output) as store:
        with pytest.raises(ConfigError, match="no matching admitted re-judgment"):
            store.record_rejudgment(failure, result)  # no admission: refused
    with RolloutStore.open_for_rejudgment(output) as store:
        pending = store.admit_rejudgment(failure)
        result = result | {
            "_ng_rejudge_operation_id": pending.operation_id,
            "_ng_rejudge_admitted_at": pending.admitted_at,
        }
        with pytest.raises(ConfigError, match="no matching admitted re-judgment"):
            store.record_rejudgment(failure, result | {"_ng_rejudge_operation_id": "0" * 32})
        for invalid in (
            result | {"response": answer | {"id": "resp_other"}},
            result | {"reward": float("nan")},
            result | {"_ng_run_id": "other-run"},
            result | {"_ng_attempt_index": 1},
        ):
            with pytest.raises(ConfigError):
                store.record_rejudgment(failure, invalid)
        assert store.record_rejudgment(failure, result) is True
        assert store.record_rejudgment(failure, result) is False
        with pytest.raises(ConfigError):
            store.record_rejudgment(failure, result | {"reward": 1.0})
    reloaded = RolloutStore.read(output)
    (selected,) = reloaded.selected("success")
    assert selected["reward"] == 0.0 and selected["response"] == answer
    assert not [key for key in selected if key.startswith("_ng_failure")]  # the old judge cause is not selected
    assert selected["_ng_rejudged_failure_observations"]["_ng_failure_id"] == "f-1"
    assert reloaded.manifest.next_attempt == store.manifest.next_attempt  # no model attempt added
    assert reloaded._state.attempt_counts[next(iter(reloaded._state.expected))] == 1  # the failure's debit stays
    assert failures_path_for(output).read_bytes() == failures_before


@pytest.mark.parametrize(
    "field,value",
    [
        ("operation_id", ""),
        ("predecessor_failure_id", ""),
        ("predecessor_sha256", "wrong"),
        ("admitted_at", True),
        ("admitted_at", -1.0),
        ("admitted_at", float("nan")),
        ("admitted_at", float("inf")),
    ],
)
def test_rejudge_admission_rejects_malformed_facts(field, value):
    from pydantic import ValidationError

    from nemo_gym.rollout_recovery import PendingRejudge

    facts = dict(operation_id="a" * 32, predecessor_failure_id="f", predecessor_sha256="b" * 64, admitted_at=1.0)
    with pytest.raises(ValidationError):
        PendingRejudge.model_validate(facts | {field: value})


@pytest.mark.parametrize("mutation", ["exact", "hash", "predecessor", "time", "foreign"])
def test_rejudge_cold_binding_is_shared_and_read_only(prepared_run, mutation):
    from nemo_gym.rollout_recovery import run_lock

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        failure = row | {
            "_ng_failure_class": "judge_failed",
            "_ng_failure_id": "original",
            "response": {"id": "saved", "object": "response", "status": "completed", "output": []},
        }
        store.record_outcome(failure)
    with run_lock(output):
        with RolloutStore.open_for_rejudgment(output) as store:
            pending = store.admit_rejudgment(failure)
            result = {k: v for k, v in failure.items() if k != "_ng_failure_class"} | {
                "reward": 0.0,
                "_ng_rejudged_failure_id": "original",
                "_ng_rejudge_operation_id": pending.operation_id,
                "_ng_rejudge_admitted_at": pending.admitted_at,
            }
            assert store.record_rejudgment(failure, result)
        identity = "0-0"
        if mutation == "hash":
            pending = pending.model_copy(update={"predecessor_sha256": "0" * 64})
        elif mutation == "predecessor":
            pending = pending.model_copy(update={"predecessor_failure_id": "other"})
        elif mutation == "time":
            pending = pending.model_copy(update={"admitted_at": pending.admitted_at + 1.0})
        elif mutation == "foreign":
            identity = "1-0"
        store.manifest.model_copy(update={"pending_rejudge": {identity: pending}}).write(manifest_path_for(output))
    before = snapshot(output)
    cold = RolloutStore.read(output)
    assert bool(cold.unresolved_rejudgments()) is (mutation != "exact")
    assert snapshot(output) == before
    assert cold.manifest.next_attempt == {"0-0": 1}
    assert cold._state.failure_count("0-0") == 1
