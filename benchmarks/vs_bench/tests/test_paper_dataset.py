# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The paper gameplay dataset must keep each competitive hand independently identifiable."""

import json
from collections import Counter

import pytest

from benchmarks.vs_bench_gameplay.prepare import prepare


def test_paper_dataset_is_stable_and_has_balanced_unique_tasks(tmp_path):
    first = prepare(tmp_path / "first.jsonl")
    second = prepare(tmp_path / "second.jsonl")
    assert first.read_bytes() == second.read_bytes()
    rows = [json.loads(line)["verifier_metadata"] for line in first.read_text().splitlines()]
    assert len(rows) == 1228
    assert Counter(row["game"] for row in rows)["kuhn_poker"] == 1200
    assert Counter(row["game"] for row in rows)["breakthrough"] == 20
    competitive = [row["paper_episode"] for row in rows if "paper_episode" in row]
    assert len({(row["game"], row["block"], row["index"]) for row in competitive}) == 1220
    assert all(row["protocol"] == "paper_competitive_v1" for row in rows)
    with pytest.raises(FileExistsError):
        prepare(first)


def test_materialized_fingerprint_covers_seat_and_deal_identity(tmp_path):
    from copy import deepcopy

    from nemo_gym.rollouts.batch_status import (
        AGENT_REF_KEY_NAME,
        ROLLOUT_INDEX_KEY_NAME,
        TASK_INDEX_KEY_NAME,
        TASK_SOURCE_KEY_NAME,
        observe_materialized_rows,
    )

    first = prepare(tmp_path / "paper.jsonl")
    rows = [json.loads(line) for line in first.read_text().splitlines()]
    materialized = [
        row
        | {
            AGENT_REF_KEY_NAME: {"name": "gameplay"},
            TASK_SOURCE_KEY_NAME: "paper-v1",
            TASK_INDEX_KEY_NAME: i,
            ROLLOUT_INDEX_KEY_NAME: 0,
        }
        for i, row in enumerate(rows)
    ]
    fingerprint = observe_materialized_rows(materialized)["gameplay"]
    assert fingerprint.task_count == 1228
    changed = deepcopy(materialized)
    changed[3]["verifier_metadata"]["paper_episode"]["model_seat"] = 1
    assert observe_materialized_rows(changed)["gameplay"].dataset_sha256 != fingerprint.dataset_sha256


def test_actual_rollout_store_resumes_only_unfinished_paper_hands(tmp_path):
    import orjson

    from nemo_gym.rollouts.rollout_recovery import RunManifest
    from nemo_gym.rollouts.rollout_store import RolloutStore

    source = prepare(tmp_path / "paper.jsonl")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows = [
        row | {"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}}
        for i, row in enumerate(rows)
    ]
    # Manifest hashes the same input and identity-bearing metadata used for collection.
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))

    def prepared():
        return rows, RunManifest.create(source, rows, {}, {"agent": {"responses_api_agents": {"impl": {}}}})

    output = tmp_path / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, prepared, resume=False) as store:
        pending = store.pending(3)
        for row in pending[:2]:
            store.record_dispatch(row)
            store.record_outcome(row | {"reward": 1.0, "response": {}})
        store.record_dispatch(pending[2])  # interruption during third task
    with RolloutStore.start_or_resume(output, prepared, resume=True) as resumed:
        pending = resumed.pending(3)
        assert len(pending) == 1226
        assert {row["_ng_task_index"] for row in pending}.isdisjoint({0, 1})
        retry = next(row for row in pending if row["_ng_task_index"] == 2)
        assert retry["_ng_attempt_index"] == 1
        assert len(resumed.selected("success")) == 2
