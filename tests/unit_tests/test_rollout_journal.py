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

import multiprocessing
import os
import signal
from collections import Counter
from contextlib import contextmanager
from itertools import permutations, product

import orjson
import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_journal import (
    RUN_ID_KEY,
    RolloutJournal,
    journal_path_for,
    materialized_path_for,
    prepare_append,
    read_records,
)
from nemo_gym.rollout_recovery import RunManifest, manifest_path_for


@pytest.fixture
def run(tmp_path):
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(5)]
    output = tmp_path / "rollouts.jsonl"
    materialized_path_for(output).write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    manifest = RunManifest.create(materialized_path_for(output), rows, {}, {})
    manifest.write(manifest_path_for(output))
    output.touch()
    failures_path_for(output).touch()
    return output, manifest, rows


@contextmanager
def writer(run, *, resume=False):
    output, manifest, rows = run
    history = RolloutJournal.load(output, manifest) if resume else RolloutJournal(manifest, rows)
    with journal_path_for(output).open("ab") as file:
        history.file = file
        yield history


def save(run, history, row, *, failure=None, reward=0.0, record_outcome=True):
    output, manifest, _ = run
    result = dict(row, **{RUN_ID_KEY: manifest.run_id})
    if failure is not None:
        result["_ng_failure_class"] = failure
        target = failures_path_for(output)
    else:
        result.update(reward=reward, response={})
        target = output
    with target.open("ab") as file:
        file.write(orjson.dumps(result) + b"\n")
        file.flush()
    if record_outcome:
        history.outcome(result)
    return result


def test_every_expected_rollout_has_one_disposition(run):
    output, manifest, rows = run
    with writer(run) as history:
        for row in rows[:4]:
            history.dispatch(row)
        save(run, history, rows[0], reward=0.0)
        save(run, history, rows[1], failure="agent_request_failed")
        history.omit(rows[2], "No cached deliverable; producer intentionally skipped this task")
        # Row 3 was dispatched and disappeared. Row 4 was never dispatched.
    recovered = RolloutJournal.load(output, manifest)
    coverage = recovered.coverage()
    assert (coverage["expected"], coverage["successful"], coverage["failed"]) == (5, 1, 1)
    assert (coverage["intentionally_omitted"], coverage["unknown"], coverage["never_dispatched"]) == (1, 2, 1)
    assert not coverage["complete"] and not coverage["reconciled"]
    assert [row["_ng_task_index"] for row in recovered.pending(3)] == [1, 3, 4]
    assert recovered.selected("success")[0]["reward"] == 0.0


@pytest.mark.parametrize("dispositions", product(("measured", "masked", "failed", "omitted", "unknown"), repeat=2))
def test_measurement_split_reconciles_without_changing_recovery(run, dispositions):
    output, manifest, rows = run
    with writer(run) as history:
        for row, disposition in zip(rows, dispositions):
            history.dispatch(row)
            if disposition == "omitted":
                history.omit(row, "Intentionally skipped")
            elif disposition == "failed":
                save(run, history, row, failure="agent_run_error")
            elif disposition != "unknown":
                save(run, history, row | {"mask_sample": disposition == "masked"}, reward=0.0)
    recovered = RolloutJournal.load(output, manifest)
    report = recovered.coverage()
    expected = Counter(dispositions)
    assert report["measured"] == expected["measured"]
    assert report["masked"] == expected["masked"]
    assert report["successful"] == report["measured"] + report["masked"]
    assert report["failed"] == expected["failed"]
    assert report["intentionally_omitted"] == expected["omitted"]
    assert report["unknown"] == expected["unknown"] + 3  # Remaining inventory was never dispatched.
    assert sum(report[key] for key in ("measured", "masked", "failed", "intentionally_omitted", "unknown")) == 5
    assert [row["_ng_task_index"] for row in recovered.pending(3)] == [
        i
        for i, disposition in enumerate((*dispositions, "unknown", "unknown", "unknown"))
        if disposition in {"failed", "unknown"}
    ]


def test_fully_masked_run_is_complete_without_unmasked_measurements(run):
    output, manifest, rows = run
    with writer(run) as history:
        for row in rows:
            history.dispatch(row)
            save(run, history, row | {"mask_sample": True})
    recovered = RolloutJournal.load(output, manifest)
    report = recovered.coverage()
    assert (report["expected"], report["successful"], report["measured"], report["masked"]) == (5, 5, 0, 5)
    assert report["complete"] and report["reconciled"]
    assert recovered.pending(3) == []
    assert len(recovered.selected("success")) == 5


@pytest.mark.parametrize("corruption", ["schema", "undispatched", "indices", "artifact", "scalar"])
def test_corrupt_history_or_payload_is_rejected_without_mutation(run, corruption):
    output, manifest, rows = run
    with writer(run) as history:
        history.dispatch(rows[0])
        payload = save(run, history, rows[0])
    journal = journal_path_for(output)
    events = list(read_records(journal))
    if corruption == "schema":
        events[0]["schema_version"] = 99
    elif corruption == "undispatched":
        events = events[1:]
    elif corruption == "indices":
        payload["_ng_rollout_id"] = "0-0"
        payload["_ng_task_index"] = 1
    elif corruption == "artifact":
        payload["_ng_failure_class"] = "judge_failed"
    else:
        payload = []
    journal.write_bytes(b"".join(orjson.dumps(event) + b"\n" for event in events))
    output.write_bytes(orjson.dumps(payload) + b"\n")
    before = (journal.read_bytes(), output.read_bytes())
    with pytest.raises(ConfigError):
        RolloutJournal.load(output, manifest)
    assert (journal.read_bytes(), output.read_bytes()) == before


def test_duplicate_inventory_cannot_conflate_distinct_tasks(run):
    _, manifest, rows = run
    with pytest.raises(ConfigError, match="Duplicate logical rollout"):
        RolloutJournal(manifest, [rows[0], rows[0] | {"question": "Different question"}])


@pytest.mark.parametrize("arrival_order", list(permutations(range(3))))
def test_latest_attempt_wins_independently_of_arrival_order(run, arrival_order):
    output, manifest, rows = run
    attempts = [dict(rows[0], _ng_attempt_index=index) for index in range(3)]
    with writer(run) as history:
        for row in attempts:
            history.dispatch(row)
        prefix = journal_path_for(output).read_bytes()
        for index in arrival_order:
            save(run, history, attempts[index], reward=index / 2)
    assert journal_path_for(output).read_bytes().startswith(prefix)
    assert len(list(read_records(output))) == 3  # Older payloads remain append-only.
    recovered = RolloutJournal.load(output, manifest)
    assert [row["reward"] for row in recovered.selected("success")] == [1.0]
    assert recovered.coverage()["successful"] == 1
    assert all(row["_ng_task_index"] != 0 for row in recovered.pending(3))


def test_new_dispatch_fences_late_success_and_unknown_attempt_is_not_reused(run):
    output, manifest, rows = run
    retry = dict(rows[0], _ng_attempt_index=1)
    with writer(run) as history:
        history.dispatch(rows[0])
        history.dispatch(retry)
        save(run, history, rows[0])
    recovered = RolloutJournal.load(output, manifest)
    assert recovered.selected("success") == []
    assert recovered.pending(3)[0]["_ng_attempt_index"] == 2
    assert all(row["_ng_task_index"] != 0 for row in recovered.pending(2))
    assert recovered.coverage()["unknown"] == 5  # Exhaustion does not invent a failure/reward.


def test_latest_failure_is_not_hidden_by_a_late_older_success(run):
    output, manifest, rows = run
    retry = dict(rows[0], _ng_attempt_index=1)
    with writer(run) as history:
        history.dispatch(rows[0])
        history.dispatch(retry)
        save(run, history, retry, failure="judge_failed")
        save(run, history, rows[0], reward=1.0)
    recovered = RolloutJournal.load(output, manifest)
    assert recovered.selected("success") == []
    assert recovered.selected("failure")[0]["_ng_attempt_index"] == 1


def test_crash_between_payload_flush_and_outcome_event_keeps_result(run):
    output, manifest, rows = run
    with writer(run) as history:
        history.dispatch(rows[0])
        save(run, history, rows[0], record_outcome=False)
    assert [event["status"] for event in read_records(journal_path_for(output))] == ["dispatched"]
    recovered = RolloutJournal.load(output, manifest)
    assert recovered.selected("success")[0]["reward"] == 0.0
    assert all(row["_ng_task_index"] != 0 for row in recovered.pending(3))


@pytest.mark.parametrize("artifact", ["history", "payload"])
def test_incomplete_tail_is_repaired_without_rewriting_prior_records(run, artifact):
    output, manifest, rows = run
    with writer(run) as history:
        history.dispatch(rows[0])
        save(run, history, rows[0])
    path = journal_path_for(output) if artifact == "history" else output
    prefix = path.read_bytes()
    with path.open("ab") as file:
        file.write(b'{"interrupted":')
    with pytest.warns(UserWarning, match="incomplete final"):
        recovered = RolloutJournal.load(output, manifest)
    assert recovered.coverage()["successful"] == 1
    with pytest.warns(UserWarning, match="incomplete final"):
        prepare_append(path)
    assert path.read_bytes() == prefix
    with writer(run, resume=True) as history:
        history.dispatch(rows[1])
        save(run, history, rows[1])
    assert RolloutJournal.load(output, manifest).coverage()["successful"] == 2


def test_complete_unterminated_tail_gets_a_newline(tmp_path):
    path = tmp_path / "records.jsonl"
    original = b'{"a": 1}\n{"a": 2}'
    path.write_bytes(original)
    prepare_append(path)
    assert path.read_bytes() == original + b"\n"
    assert list(read_records(path)) == [{"a": 1}, {"a": 2}]


def test_interior_corruption_is_rejected(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"a": 1}\n{bad}\n{"a": 2}\n')
    with pytest.raises(ConfigError, match="line 2"):
        list(read_records(path))


@pytest.mark.parametrize("artifact", ["history", "payload", "materialized"])
def test_foreign_run_or_changed_inventory_is_rejected(run, artifact):
    output, manifest, rows = run
    with writer(run) as history:
        history.dispatch(rows[0])
        save(run, history, rows[0])
    if artifact == "materialized":
        path = materialized_path_for(output)
        path.write_bytes(path.read_bytes().replace(b'"agent"', b'"other-agent"'))
    else:
        path = journal_path_for(output) if artifact == "history" else output
        path.write_bytes(path.read_bytes().replace(manifest.run_id.encode(), b"another-run"))
    with pytest.raises(ConfigError, match="different run|do not match"):
        RolloutJournal.load(output, manifest)


def test_duplicate_delivery_is_idempotent_but_conflicting_payloads_are_rejected(run):
    output, manifest, rows = run
    with writer(run) as history:
        history.dispatch(rows[0])
        save(run, history, rows[0])
        save(run, history, rows[0])
    assert RolloutJournal.load(output, manifest).coverage()["successful"] == 1
    with writer(run, resume=True) as history:
        with pytest.raises(ConfigError, match="Conflicting outcomes"):
            save(run, history, rows[0], reward=1.0)
    with pytest.raises(ConfigError, match="Conflicting outcomes"):
        RolloutJournal.load(output, manifest)


def test_terminal_skip_is_a_durable_omission(run):
    output, manifest, rows = run
    row = dict(rows[0], _ng_failure_terminal=True)
    with writer(run) as history:
        history.dispatch(row)
        save(run, history, row, failure="skipped")
    recovered = RolloutJournal.load(output, manifest)
    assert recovered.coverage()["intentionally_omitted"] == 1
    assert recovered.coverage()["failed"] == 0
    assert all(row["_ng_task_index"] != 0 for row in recovered.pending(3))


def dispatch_then_die(output, manifest_dict, rows):
    history = RolloutJournal(RunManifest.model_validate(manifest_dict), rows)
    with journal_path_for(output).open("ab") as file:
        history.file = file
        history.dispatch(rows[0])
        os.kill(os.getpid(), signal.SIGKILL)


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="Requires process kill without cleanup")
def test_killed_worker_leaves_durable_unknown_attempt(run):
    output, manifest, rows = run
    process = multiprocessing.get_context("spawn").Process(
        target=dispatch_then_die, args=(output, manifest.model_dump(), rows)
    )
    process.start()
    try:
        process.join(30)
        assert process.exitcode == -signal.SIGKILL
        history = RolloutJournal.load(output, manifest)
        assert history.coverage()["unknown"] == 5
        assert history.coverage()["never_dispatched"] == 4
        assert history.pending(3)[0]["_ng_attempt_index"] == 1
        assert output.read_bytes() == b""
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("count_failures_as_zero", [False, True])
async def test_offline_aggregation_uses_newest_attempt_and_full_inventory(
    run, monkeypatch, masked, count_failures_as_zero
):
    import nemo_gym.rollout_collection as collection
    from nemo_gym.rollout_collection import (
        RolloutAggregationConfig,
        RolloutAggregationHelper,
        RolloutCollectionHelper,
    )
    from nemo_gym.rollout_journal import coverage_path_for

    output, _, rows = run
    retry = dict(rows[0], _ng_attempt_index=1)
    with writer(run) as history:
        history.dispatch(rows[0])
        history.dispatch(retry)
        save(run, history, retry | {"mask_sample": masked}, reward=0.0)
        save(run, history, rows[0] | {"mask_sample": not masked}, reward=1.0)
        history.dispatch(rows[1])
        save(run, history, rows[1], failure="agent_run_error")
        history.omit(rows[2], "No cached deliverable")
        history.dispatch(rows[3])  # Dispatched unknown; row 4 was never dispatched.

    scored = []

    async def aggregate(self, results, rows, path):
        scored.extend(results)
        return None

    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", aggregate)
    exported = []
    monkeypatch.setattr(collection, "get_exporters", lambda: True)
    monkeypatch.setattr(collection, "export_metrics", exported.append)
    merged = output.with_name("merged.jsonl")
    await RolloutAggregationHelper().run_from_config(
        RolloutAggregationConfig(
            input_glob=str(output.with_name("rollouts*.jsonl")),
            output_jsonl_fpath=str(merged),
            disable_health_check=True,
            count_failure_classes_as_zero=["agent_run_error"] if count_failures_as_zero else [],
        )
    )
    assert [row["_ng_task_index"] for row in scored] == ([0, 1] if count_failures_as_zero else [0])
    assert all(row["reward"] == 0.0 for row in scored)
    assert [row["reward"] for row in read_records(merged)] == [0.0]
    report = orjson.loads(coverage_path_for(merged).read_bytes())
    assert (report["expected"], report["successful"], report["unknown"]) == (5, 1, 2)
    assert (report["measured"], report["masked"], report["failed"], report["intentionally_omitted"]) == (
        int(not masked),
        int(masked),
        1,
        1,
    )
    assert report["scored"] == 1 + int(count_failures_as_zero)
    assert report["failures_counted_as_zero"] == int(count_failures_as_zero)
    assert sum(report[key] for key in ("measured", "masked", "failed", "intentionally_omitted", "unknown")) == 5
    assert exported[-1] == {
        "coverage/expected": 5,
        "coverage/scored": report["scored"],
        "coverage/missing": 5 - report["scored"],
        "coverage/known": 1,
        "coverage/measured": int(not masked),
        "coverage/masked": int(masked),
        "coverage/failed": 1,
        "coverage/omitted": 1,
        "coverage/unknown": 2,
    }
    assert report["coverage_known"] and not report["complete"]
    assert len(list(read_records(output))) == 2  # Aggregating does not rewrite history.


@pytest.mark.parametrize("masked", [False, True])
async def test_legacy_aggregation_cannot_claim_complete_without_inventory(tmp_path, monkeypatch, capsys, masked):
    import nemo_gym.rollout_collection as collection
    from nemo_gym.rollout_journal import coverage_path_for

    output = tmp_path / "legacy.jsonl"
    output.write_bytes(
        orjson.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0, "mask_sample": masked}) + b"\n"
    )

    async def aggregate(*args):
        return None

    exported = []
    monkeypatch.setattr(collection.RolloutCollectionHelper, "_call_aggregate_metrics", aggregate)
    monkeypatch.setattr(collection, "get_exporters", lambda: True)
    monkeypatch.setattr(collection, "export_metrics", exported.append)
    merged = tmp_path / "merged.jsonl"
    await collection.RolloutAggregationHelper().run_from_config(
        collection.RolloutAggregationConfig(
            input_glob=str(output), output_jsonl_fpath=str(merged), disable_health_check=True
        )
    )
    report = orjson.loads(coverage_path_for(merged).read_bytes())
    assert report["expected"] is None and report["unknown"] is None
    assert (report["measured"], report["masked"]) == (int(not masked), int(masked))
    assert not report["coverage_known"] and not report["complete"]
    assert exported == [
        {
            "coverage/scored": 1,
            "coverage/known": 0,
            "coverage/measured": int(not masked),
            "coverage/masked": int(masked),
        }
    ]
    assert "scores may be partial" in capsys.readouterr().out
