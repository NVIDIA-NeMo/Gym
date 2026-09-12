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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import orjson
import pytest

import nemo_gym.rollout_store as persistence
from nemo_gym.config_types import ConfigError
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_journal import RolloutJournal, journal_path_for, materialized_path_for, read_records
from nemo_gym.rollout_recovery import RunManifest, atomic_write_json, manifest_path_for
from nemo_gym.rollout_store import RolloutStore


@pytest.fixture
def prepared_run(tmp_path):
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(2)]
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    prepare = Mock(side_effect=lambda: (rows, RunManifest.create(source, rows, {}, {})))
    return tmp_path / "rollouts.jsonl", prepare


def snapshot(output):
    return {path.name: path.read_bytes() for path in output.parent.iterdir() if path.is_file()}


@pytest.mark.parametrize("interruption", ["outcome_event", "fsync"])
def test_flushed_payload_survives_interrupted_commit(prepared_run, monkeypatch, interruption):
    output, prepare = prepared_run
    store = RolloutStore.start_or_resume(output, prepare, resume=False)
    original_event = RolloutJournal._event

    def event(state, key, status, reason=None):
        if status == "success":
            # The payload must be visible even though its outcome event fails.
            assert list(read_records(output)) == [result]
            raise OSError("interrupted outcome event")
        return original_event(state, key, status, reason)

    def fsync(fd):
        assert list(read_records(output)) == [result]
        assert [row["status"] for row in read_records(journal_path_for(output))] == ["dispatched"]
        raise OSError("interrupted fsync")

    monkeypatch.setattr(RolloutJournal, "_event", event)
    if interruption == "fsync":
        monkeypatch.setattr(persistence, "os", SimpleNamespace(fsync=fsync))
    with pytest.raises(OSError, match="interrupted"):
        with store:
            row = store.pending(3)[0]
            store.record_dispatch(row)
            result = row | {"reward": 0.0, "response": {}}
            store.record_outcome(result, sync=interruption == "fsync")
    before = snapshot(output)
    recovered = RolloutStore.read(output)
    assert recovered.selected("success") == [result]
    assert [row["_ng_task_index"] for row in recovered.pending(3)] == [1]
    assert recovered.coverage()["successful"] == 1
    assert snapshot(output) == before


def test_invalid_outcomes_are_rejected_before_appending_bytes(prepared_run):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row, undispatched = store.pending(3)
        store.record_dispatch(row)
        result = row | {"reward": 0.0, "response": {}}
        store.record_outcome(result)
        before = snapshot(output)
        for invalid in (result | {"reward": 1.0}, undispatched | {"reward": 0.0, "response": {}}):
            with pytest.raises(ConfigError, match="Conflicting outcomes|no dispatch"):
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
    assert len(opened) == 2 and all(file.closed for file in opened)
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
    assert offline.coverage()["selection_policy"] == "latest_dispatched"
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


@pytest.mark.parametrize("missing", [("journal",), ("manifest",), ("manifest", "journal")])
def test_unsafe_recovery_rebuilds_own_artifacts_without_relabeling(prepared_run, missing):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as original:
        row = original.pending(3)[0]
        original.record_dispatch(row)
        result = row | {"reward": 0.0, "response": {}}
        original.record_outcome(result)
    paths = {"journal": journal_path_for(output), "manifest": manifest_path_for(output)}
    for name in missing:
        paths[name].unlink()
    saved_output = output.read_bytes()
    with pytest.raises(ConfigError):
        RolloutStore.start_or_resume(output, prepare, resume=True)
    with pytest.warns(UserWarning, match="allow_unsafe_resume"):
        resumed = RolloutStore.start_or_resume(output, prepare, resume=True, allow_unsafe=True)
    with resumed:
        assert resumed.manifest.run_id == original.manifest.run_id
        assert resumed.selected("success") == [result]
        assert [row["_ng_task_index"] for row in resumed.pending(3)] == [1]
        assert not resumed.coverage()["identity_verified"]
        pending = resumed.pending(3)[0]
        resumed.record_dispatch(pending)
        resumed.record_outcome(pending | {"reward": 1.0, "response": {}})
    assert output.read_bytes().startswith(saved_output)
    assert RolloutStore.read(output).coverage()["complete"]


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
    journal_path_for(output).unlink()
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


def test_reverification_rejects_changed_inputs_before_dispatch(prepared_run):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False):
        pass
    writer = RolloutStore.append_existing(output)
    before = snapshot(output)
    rows, _ = prepare()
    for invalid in (rows[0] | {"_ng_task_index": 999}, rows[0] | {"agent_ref": {"name": "other"}}):
        with pytest.raises(ConfigError, match="Reverification input"):
            writer.for_reverification([invalid | {"response": {}}])
    assert snapshot(output) == before
    journal_path_for(output).unlink()
    with pytest.raises(ConfigError, match="without attempt history"):
        RolloutStore.append_existing(output)


def test_reverification_overwrite_preserves_existing_run(prepared_run):
    from nemo_gym.rollout_reverification import _prepare_output_fpaths

    output, prepare = prepared_run
    RolloutStore.start_or_resume(output, prepare, resume=False)
    before = snapshot(output)
    with pytest.raises(ConfigError, match="Cannot overwrite a journal-backed run"):
        _prepare_output_fpaths("", str(output), False, True, False)
    assert snapshot(output) == before


@pytest.mark.parametrize("missing", ["output", "materialized", "journal"])
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
