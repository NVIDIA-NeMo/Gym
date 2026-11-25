# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual group routing, execution identity and dataset-weight preservation."""

from collections import Counter
from copy import deepcopy

import pytest

from nemo_gym.core.harness_routing import HarnessRoutingMetadata, annotate_prompt_groups


def test_interleaved_benchmarks_balance_groups_without_replicating_rows():
    rows = [
        {"task_source": source, "task_input": {"task_data": {"literal": index}}, "prompt": f"p{index}"}
        for index in range(10)
        for source in ("avb", "claw", "video")
    ]
    before = deepcopy(rows)
    sources = {
        source: {"benchmark": source.upper(), "dataset": source + "-data", "env": source + "-env", "task_type": kind}
        for source, kind in [("avb", "agentic"), ("claw", "agentic"), ("video", "non_agentic")]
    }
    prepared = annotate_prompt_groups(rows, sources=sources, routing_seed=7)
    assert len(prepared) == len(rows) and rows == before
    for source in ("avb", "claw"):
        counts = Counter(
            row["harness_routing"]["selected_harness"] for row in prepared if row["task_source"] == source
        )
        assert set(counts) == {"opencode", "codex", "hermes"} and max(counts.values()) - min(counts.values()) == 1
    for original, row in zip(before, prepared, strict=True):
        assert (
            row["prompt"] == original["prompt"]
            and row["task_input"]["task_data"] == original["task_input"]["task_data"]
        )
        assert row["harness_routing"] == row["task_input"]["harness_routing"]
        if row["task_source"] == "video":
            assert row["harness_routing"]["selected_harness"] is None
            assert row["harness_routing"]["executed_harness"] is None


def test_replicas_retries_and_serialized_checkpoint_keep_route():
    route = HarnessRoutingMetadata(
        benchmark="AVB",
        dataset="train",
        env="avb",
        task_type="agentic",
        group_index=4,
        routing_seed=9,
        group_id="prompt-group",
    )
    selected = route.select()
    replicas = [HarnessRoutingMetadata.model_validate_json(selected.model_dump_json()) for _ in range(16)]
    assert {item.select().selected_harness for item in replicas} == {"codex"}
    assert all(item.group_id == "prompt-group" for item in replicas)


def test_actual_bridge_mismatch_is_rejected():
    selected = HarnessRoutingMetadata(
        benchmark="AVB", dataset="train", env="avb", task_type="agentic", group_index=0
    ).select()
    assert selected.with_execution("opencode").executed_harness == "opencode"
    with pytest.raises(ValueError, match="executed bridge"):
        selected.with_execution("hermes")


def test_non_agentic_cannot_claim_external_harness():
    with pytest.raises(ValueError, match="non-agentic"):
        HarnessRoutingMetadata(
            benchmark="tracking", dataset="data", env="tracking", task_type="non_agentic", selected_harness="codex"
        )


def test_loader_shuffle_preserves_prepared_group_routes():
    rows = [
        {
            "task_source": "AVB",
            "task_id": {"taskset": "train", "task_id": str(i)},
            "task_input": {"task_data": {"id": i}},
        }
        for i in range(12)
    ]
    sources = {"AVB": {"benchmark": "AVB", "dataset": "train", "env": "avb", "task_type": "agentic"}}
    prepared = annotate_prompt_groups(rows, sources=sources, routing_seed=42)
    expected = {row["task_id"]["task_id"]: row["harness_routing"] for row in prepared}
    import random

    random.Random(42).shuffle(prepared)
    for row in prepared:
        route = HarnessRoutingMetadata.model_validate(row["harness_routing"])
        assert route.select().model_dump() == expected[row["task_id"]["task_id"]]
