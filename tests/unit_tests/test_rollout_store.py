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

from nemo_gym.config_types import ConfigError
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_records import (
    RolloutRecord,
    journal_path_for,
    materialized_path_for,
    read_records,
)
from nemo_gym.rollout_recovery import RunManifest, atomic_write_json, manifest_path_for
from nemo_gym.rollout_store import RolloutStore


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
                    payload["_ng_failure_class"] = "judge_failed"
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
def test_incomplete_legacy_cache_restarts_from_current_inputs(prepared_run, capsys, artifact, allow_unsafe):
    output, prepare = prepared_run
    paths = {"materialized": materialized_path_for(output), "output": output, "failures": failures_path_for(output)}
    paths[artifact].write_bytes(b'{"old_partial_run":true}\n')
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
    assert report["attempts_exhausted"] == expected_exhausted
    assert (report["failed"], report["unknown"], report["never_dispatched"], report["attempts"]) == (2, 2, 1, 5)
    from nemo_gym.rollout_records import coverage_path_for

    assert orjson.loads(coverage_path_for(output).read_bytes())["attempts_exhausted"] == expected_exhausted


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
    RolloutStore.start_or_resume(output, prepare, resume=False)
    if alias:
        shortcut = output.with_name("shortcut.jsonl")
        shortcut.symlink_to(output)
        output = shortcut
    before = snapshot(output)
    with pytest.raises(ConfigError, match="Manifest-aware health reports are a follow-up"):
        run_health_checks(output, workers=1)
    assert snapshot(output) == before


@pytest.mark.parametrize("artifact", ["output", "failures"])
@pytest.mark.parametrize("complete_corruption", [False, True])
def test_migration_repairs_only_validated_incomplete_tails(prepared_run, artifact, complete_corruption):
    from nemo_gym.rollout_collection import migrate_invalid_judge_main_rows

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"reward": 0.0, "response": {}, "invalid_judge_response": True})
    target = output if artifact == "output" else failures_path_for(output)
    with target.open("ab") as file:
        file.write(b'{"reward":' + (b"\n" if complete_corruption else b""))
    before = snapshot(output)
    if complete_corruption:
        with pytest.raises(ConfigError):
            RolloutStore.start_or_resume(
                output, prepare, resume=True, migrate_outcomes=migrate_invalid_judge_main_rows
            )
        assert snapshot(output) == before
        return
    for _ in range(2):
        with RolloutStore.start_or_resume(
            output, prepare, resume=True, migrate_outcomes=migrate_invalid_judge_main_rows
        ) as recovered:
            assert recovered.coverage()["failed"] == 1
            retry = recovered.pending(3)[0]
            assert retry["_ng_attempt_index"] == 1
            assert recovered.failures()[0]["_ng_failure_class"] == "judge_invalid"
    assert list(read_records(output)) == []
    assert len(list(read_records(failures_path_for(output)))) == 1


@pytest.mark.parametrize("newer_outcome", [None, "failure", "omission"])
def test_cached_deliverable_survives_unknown_attempt_but_not_newer_outcome(prepared_run, newer_outcome):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(4)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"_ng_failure_class": "judge_invalid", "reuse_cached_deliverable": True})
        retry = store.pending(4)[0]
        assert retry["reuse_cached_deliverable"]
        store.record_dispatch(retry)
        if newer_outcome == "failure":
            store.record_outcome(
                {k: v for k, v in retry.items() if k != "reuse_cached_deliverable"}
                | {"_ng_failure_class": "agent_run_error"}
            )
        elif newer_outcome == "omission":
            store.record_omission(retry, "operator omitted")
        store.record_dispatch(row | {"_ng_attempt_index": 2})
    recovered = RolloutStore.start_or_resume(output, prepare, resume=True)
    retry = recovered.pending(4)[0]
    assert retry["_ng_attempt_index"] == 3
    assert bool(retry.get("reuse_cached_deliverable")) == (newer_outcome is None)


@pytest.mark.parametrize("mismatch", ["run_id", "rollout_id", "attempt", "terminal", "failure_kind"])
def test_nested_failure_must_agree_with_saved_envelope(prepared_run, mismatch):
    from nemo_gym.episode_types import EpisodeFailure, EpisodeId
    from nemo_gym.rollout_collection import _failure_compatibility_row
    from nemo_gym.rollout_outcomes import RolloutFailure
    from nemo_gym.rollout_records import logical_rollout_id

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        failure = RolloutFailure(
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
    assert report["successful"] == 0 and report["unknown"] == 2 and len(report["shards"]) == 1
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
    offline = RolloutStore.read(output, retry_terminal_timeouts=retry_timeouts).coverage()
    assert offline == online
    assert offline["retry_terminal_timeouts"] is retry_timeouts
    assert offline["attempts_exhausted"] == int(retry_timeouts)


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
    assert (report["expected"], report["unknown"], report["imputed"], report["scored"]) == (2, 1, 1, 1)
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
    assert (report["expected"], report["successful"], report["imputed"], report["unknown"]) == (4, 1, 3, 3)
    assert not report["complete"]
