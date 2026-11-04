# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import nemo_gym.rollouts.rollout_collection as collection
from nemo_gym.rollouts.rollout_collection import RolloutCollectionHelper
from tests.unit_tests.rollouts.test_rollout_collection import FakeResponse, install_fake_server_client
from tests.unit_tests.rollouts.test_rollout_recovery import runner_config  # noqa: F401


@pytest.mark.parametrize("retry_failure", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("alias", [False, True])
async def test_resume_preserves_health_and_isolates_failed_attempt(
    runner_config, monkeypatch, retry_failure, workers, alias
):
    from nemo_gym.rollouts.rollout_health import run_health_checks
    from tests.unit_tests.rollouts.test_rollout_health import _record

    runner_config.disable_aggregation = runner_config.disable_health_check = False
    runner_config.health_check_workers = workers
    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", AsyncMock(return_value=None))
    monkeypatch.setattr(collection, "get_exporters", list)

    async def post(**kwargs):
        row = kwargs["json"]
        first = row.get("_ng_attempt_index", 0) == 0
        result = _record(row["_ng_task_index"], 0, answer="" if row["task"] == 1 and first else "ok", refs=[])
        result["reward"] = 1.0
        if retry_failure and row["task"] == 1 and first:
            result["_ng_failure_class"] = "agent_run_error"
        return FakeResponse(200, result)

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    helper = RolloutCollectionHelper()
    await helper.run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    before = output.read_bytes()
    if alias:
        shortcut = output.with_name("shortcut.jsonl")
        shortcut.symlink_to(output)
        runner_config.output_jsonl_fpath = str(shortcut)
    health_source = Path(runner_config.output_jsonl_fpath)
    original_report = json.loads((output.parent / "quality_summary.json").read_bytes())["run"]
    runner_config.resume_from_cache = True
    await helper.run_from_config(runner_config)
    assert client.post.await_count == 3 + int(retry_failure)
    assert output.read_bytes().startswith(before)
    if not retry_failure:
        assert output.read_bytes() == before
    expected = json.loads((output.parent / "quality_summary.json").read_bytes())["run"]
    assert expected["artifacts"]["records"] == 3
    assert expected["issues"]["rollout_duplicate_identity"] == 0
    assert expected["issues"]["agent_turn_hollow"] == int(not retry_failure)
    if not retry_failure:
        assert expected == original_report
    else:
        failed = run_health_checks(
            collection.failures_path_for(output), workers=1, output_dir=output.parent / "failed"
        )
        assert failed.summary["run"]["issues"]["agent_turn_hollow"] == 1
    for merge in (False, True):
        target = output.parent / f"aggregate-{merge}" / "rollouts.jsonl"
        await collection.RolloutAggregationHelper().run_from_config(
            collection.RolloutAggregationConfig(
                input_glob=str(health_source),
                output_jsonl_fpath=str(target),
                merge_shards=merge,
                health_check_workers=workers,
            )
        )
        aggregated = json.loads((target.parent / "quality_summary.json").read_bytes())["run"]
        assert {k: v for k, v in aggregated.items() if k != "rollouts"} == {
            k: v for k, v in expected.items() if k != "rollouts"
        }
        assert all((target.parent / name).resolve().is_file() for name in aggregated["rollouts"])
    report_dir = output.parent / "standalone"
    standalone = run_health_checks(health_source, workers=workers, output_dir=report_dir).summary["run"]
    assert {k: v for k, v in standalone.items() if k != "rollouts"} == {
        k: v for k, v in expected.items() if k != "rollouts"
    }
    assert [(report_dir / name).resolve() for name in standalone["rollouts"]] == [output.resolve()]


@pytest.mark.parametrize("field", ["_ng_run_id", "_ng_attempt_index", "_ng_task_index"])
def test_health_rejects_same_inode_rewritten_identity(tmp_path, field):
    import orjson

    from nemo_gym.rollouts.rollout_health import _index_jsonl, _read_record
    from nemo_gym.rollouts.rollout_recovery import RunManifest
    from nemo_gym.rollouts.rollout_store import RolloutStore

    output = tmp_path / "out.jsonl"
    source = tmp_path / "source.jsonl"
    rows = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
    source.write_bytes(orjson.dumps(rows[0]) + b"\n")
    manifest = RunManifest.create(source, rows, {}, {})
    with RolloutStore.start_or_resume(output, lambda: (rows, manifest), resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        payload = row | {"reward": 1.0, "response": {}}
        store.record_outcome(payload)
    [indexed] = _index_jsonl([output])
    original_size = output.stat().st_size
    original_inode = output.stat().st_ino
    payload[field] = "x" * len(payload[field]) if field == "_ng_run_id" else 9
    output.write_bytes(orjson.dumps(payload) + b"\n")
    assert output.stat().st_ino == original_inode and output.stat().st_size == original_size
    assert _read_record(indexed) == ({}, "rollout identity changed after indexing")


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("mixed", [False, True])
async def test_review_aggregation_health_uses_the_scored_snapshot(tmp_path, monkeypatch, merge, workers, mixed):
    from nemo_gym.rollouts.rollout_health import run_health_checks
    from nemo_gym.rollouts.rollout_recovery import RunManifest
    from nemo_gym.rollouts.rollout_store import RolloutStore
    from tests.unit_tests.rollouts.test_rollout_health import _record

    output = tmp_path / "source" / "rollouts.jsonl"
    output.parent.mkdir()
    source = tmp_path / "input.jsonl"
    inputs = [{"_ng_task_index": 0, "_ng_rollout_index": 0, "agent_ref": {"name": "my_agent"}}]
    source.write_text(json.dumps(inputs[0]) + "\n")
    manifest = RunManifest.create(source, inputs, {}, {"my_agent": {"responses_api_agents": {"impl": {}}}})
    with RolloutStore.start_or_resume(output, lambda: (inputs, manifest), resume=False) as store:
        row = store.pending()[0]
        store.record_dispatch(row)
        store.record_outcome(row | _record(0, 0) | {"reward": 1.0})

    paths = [output]
    if mixed:
        loose = tmp_path / "loose.jsonl"
        loose.write_text(json.dumps(_record(5, 0) | {"reward": 0.5, "agent_ref": {"name": "my_agent"}}) + "\n")
        paths.append(loose)
    install_fake_server_client(monkeypatch, AsyncMock())

    async def score(results, *args, **kwargs):
        assert len(results) == 1 + int(mixed)
        with RolloutStore.start_or_resume(output, lambda: (inputs, manifest), resume=True) as writer:
            newer = dict(row)
            writer.allocate_attempt(newer)
            writer.record_outcome(newer | {"_ng_failure_class": "judge_failed", "response": {"id": "saved"}})
        return None

    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", AsyncMock(side_effect=score))
    target = tmp_path / "report" / "merged.jsonl"
    await collection.RolloutAggregationHelper().run_from_config(
        collection.RolloutAggregationConfig(
            input_glob=",".join(map(str, paths)),
            output_jsonl_fpath=str(target),
            merge_shards=merge,
            health_check_workers=workers,
        )
    )
    report = json.loads((target.parent / "quality_summary.json").read_text())["run"]
    assert report["artifacts"]["records"] == 1 + int(mixed)
    assert report["artifacts"]["failed"]["total"] == 0
    assert report["stats"]["timeouts"]["failed_without_signal"] == 0
    assert report["issues"]["rollout_duplicate_identity"] == 0
    if merge:
        assert len(target.read_text().splitlines()) == 1 + int(mixed)
    current = run_health_checks(output, workers=workers, output_dir=tmp_path / "standalone").summary["run"]
    assert current["artifacts"]["records"] == 0
    assert current["artifacts"]["failed"]["total"] == 1
    assert current["stats"]["timeouts"]["failed_without_signal"] == 1
    assert RolloutStore.read(output).coverage()["failed"] == 1


def test_run_directory_health_reads_the_stacks_settled_judge_answer(tmp_path):
    from nemo_gym.rollouts.rollout_health import health_check_run_dir
    from tests.unit_tests.eval.test_rejudge_saved_judge_failures import _run, _settle

    output, failures = _run(tmp_path, judge_failures=1)
    _settle(output, failures[0], 0.25)
    before = output.read_bytes()
    report = health_check_run_dir(output.parent, workers=1).summary["run"]
    assert report["artifacts"]["records"] == 3
    assert report["artifacts"]["failed"]["total"] == 0
    assert report["issues"]["rollout_duplicate_identity"] == 0
    assert output.read_bytes() == before


@pytest.mark.parametrize("key_style", ["relative", "alias"])
@pytest.mark.parametrize("managed", [False, True])
def test_health_accepts_relative_and_aliased_history_keys(tmp_path, monkeypatch, key_style, managed):
    from nemo_gym.rollouts.rollout_health import run_health_checks
    from nemo_gym.rollouts.rollout_recovery import RunManifest
    from nemo_gym.rollouts.rollout_store import RolloutStore
    from tests.unit_tests.rollouts.test_rollout_health import _record

    output = tmp_path / "rollouts.jsonl"
    history = None
    if managed:
        inputs = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
        source = tmp_path / "inputs.jsonl"
        source.write_text(json.dumps(inputs[0]) + "\n")
        manifest = RunManifest.create(source, inputs, {}, {})
        with RolloutStore.start_or_resume(output, lambda: (inputs, manifest), resume=False) as history:
            row = history.pending()[0]
            history.record_dispatch(row)
            history.record_outcome(row | _record(0, 0) | {"reward": 1.0})
    else:
        output.write_text(json.dumps(_record(0, 0) | {"reward": 1.0}) + "\n")
    monkeypatch.chdir(tmp_path)
    key = Path(output.name)
    if key_style == "alias":
        key = tmp_path / "alias.jsonl"
        key.symlink_to(output)

    report = run_health_checks(output, workers=1, source_histories={key: history}, output_dir=tmp_path / "health")
    assert report.summary["run"]["artifacts"]["records"] == 1


def test_health_names_missing_or_conflicting_history_paths(tmp_path):
    from nemo_gym.rollouts.rollout_health import run_health_checks

    output = tmp_path / "rollouts.jsonl"
    output.write_text("{}\n")
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(output)
    with pytest.raises(ValueError, match="No source history supplied") as error:
        run_health_checks(output, workers=1, source_histories={})
    assert str(output) in str(error.value)
    with pytest.raises(ValueError, match="Conflicting source histories supplied") as error:
        run_health_checks(output, workers=1, source_histories={output: None, alias: object()})
    assert str(output) in str(error.value)


def test_health_names_selected_record_without_file_identity(tmp_path):
    from unittest.mock import Mock

    from nemo_gym.rollouts.rollout_health import _index_jsonl
    from nemo_gym.rollouts.rollout_records import RolloutRecord
    from nemo_gym.rollouts.rollout_store import RolloutStore

    output = tmp_path / "rollouts.jsonl"
    output.write_text("{}\n")
    record = RolloutRecord(path=output, offset=0, length=3, line_number=37)
    history = Mock(spec=RolloutStore)
    history.selected_records.return_value = {("task", 0): record}
    with pytest.raises(ValueError, match="has no file identity") as error:
        _index_jsonl([output], histories={output: history})
    assert f"{output}:37" in str(error.value)
