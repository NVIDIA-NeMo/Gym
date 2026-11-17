# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for environments/unified_video_agentic/prepare.py: routing stamps, weights, determinism, example data."""

import json
from collections import Counter
from pathlib import Path

import pytest

from environments.unified_video_agentic import prepare


REPO = Path(__file__).resolve().parents[3]
ENTRY = REPO / "environments/unified_video_agentic"


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _row(i: int, **extra) -> dict:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": f"q{i}"}]},
        "verifier_metadata": {"i": i},
        **extra,
    }


def test_mix_stamps_each_member_agent_and_keeps_every_row(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.jsonl", [_row(i) for i in range(4)])
    b = _write(tmp_path / "b.jsonl", [_row(i, task_source="b_source") for i in range(2)])
    rows, stats = prepare.mix([("agent_a", a, 1.0), ("agent_b", b, 1.0)])
    assert stats["rows"] == 6 and stats["per_agent"] == {"agent_a": 4, "agent_b": 2}
    assert Counter(r["agent_ref"]["name"] for r in rows) == {"agent_a": 4, "agent_b": 2}
    assert all(r["agent_ref"]["type"] == "responses_api_agents" for r in rows)
    assert {r.get("task_source") for r in rows if r["agent_ref"]["name"] == "agent_b"} == {"b_source"}


def test_weights_subsample_and_repeat(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.jsonl", [_row(i) for i in range(10)])
    rows, stats = prepare.mix([("agent_a", a, 0.3)])
    assert stats["per_agent"] == {"agent_a": 3} and len({r["verifier_metadata"]["i"] for r in rows}) == 3
    rows, stats = prepare.mix([("agent_a", a, 1.5)])
    assert stats["per_agent"] == {"agent_a": 15}
    assert Counter(r["verifier_metadata"]["i"] for r in rows).most_common(1)[0][1] == 2


def test_same_seed_same_order(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.jsonl", [_row(i) for i in range(6)])
    b = _write(tmp_path / "b.jsonl", [_row(i) for i in range(6)])
    members = [("agent_a", a, 0.5), ("agent_b", b, 1.0)]
    first, _ = prepare.mix(members, seed=3)
    second, _ = prepare.mix(members, seed=3)
    assert first == second
    other, _ = prepare.mix(members, seed=4)
    assert other != first


def test_rows_routed_elsewhere_or_without_input_are_refused(tmp_path: Path) -> None:
    routed = _write(tmp_path / "routed.jsonl", [_row(0, agent_ref={"name": "someone_else"})])
    with pytest.raises(ValueError, match="already routes"):
        prepare.mix([("agent_a", routed, 1.0)])
    same = _write(tmp_path / "same.jsonl", [_row(0, agent_ref={"name": "agent_a"})])
    rows, _ = prepare.mix([("agent_a", same, 1.0)])
    assert rows[0]["agent_ref"] == {"type": "responses_api_agents", "name": "agent_a"}
    empty = _write(tmp_path / "empty.jsonl", [{"verifier_metadata": {}}])
    with pytest.raises(ValueError, match="responses_create_params.input"):
        prepare.mix([("agent_a", empty, 1.0)])


def test_parse_member_accepts_optional_weight() -> None:
    assert prepare.parse_member("agent_a=/tmp/x.jsonl") == ("agent_a", Path("/tmp/x.jsonl"), 1.0)
    assert prepare.parse_member("agent_a=/tmp/x.jsonl:0.25") == ("agent_a", Path("/tmp/x.jsonl"), 0.25)
    assert prepare.parse_member("agent_a=C:/x.jsonl") == ("agent_a", Path("C:/x.jsonl"), 1.0)


def test_committed_example_is_the_members_examples_routed(tmp_path: Path) -> None:
    committed = [json.loads(line) for line in (ENTRY / "data/example.jsonl").read_text().splitlines() if line.strip()]
    assert len(committed) == 5
    assert Counter(r["agent_ref"]["name"] for r in committed) == {
        "videomme_v1_single_turn": 3,
        "video_caption_simple_agent": 2,
    }
    out = tmp_path / "example.jsonl"
    prepare.main(["--example", "--output", str(out)])
    regenerated = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert regenerated == committed
    for agent, path in prepare.EXAMPLE_MEMBERS.items():
        source = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in committed:
            if row["agent_ref"]["name"] == agent:
                assert {k: v for k, v in row.items() if k != "agent_ref"} in source


def _materialized_row(taskset: str, task: str) -> dict:
    return {
        "task_id": {"taskset": taskset, "task_id": task},
        "task_input": {"responses_create_params": {"input": [{"role": "user", "content": f"do {task}"}]}},
        "task_source": taskset,
    }


def test_taskset_rows_join_unstamped(tmp_path: Path) -> None:
    harbor = tmp_path / "tasks.jsonl"
    harbor.write_text("\n".join(json.dumps(_materialized_row("claweval_train", f"t{i}")) for i in range(4)) + "\n")
    flat = tmp_path / "flat.jsonl"
    flat.write_text(json.dumps({"responses_create_params": {"input": [{"role": "user", "content": "q"}]}}) + "\n")

    rows, stats = prepare.mix([("videomme_v1_single_turn", flat, 1.0), (None, harbor, 0.5)], seed=3)

    assert stats["per_agent"] == {"videomme_v1_single_turn": 1, "taskset:claweval_train": 2}
    assert [r["agent_ref"]["name"] for r in rows if "agent_ref" in r] == ["videomme_v1_single_turn"]
    assert all("agent_ref" not in r and r["task_id"]["taskset"] == "claweval_train" for r in rows if "task_input" in r)
    assert prepare.parse_taskset(str(harbor) + ":0.5") == (None, harbor, 0.5)
    with pytest.raises(ValueError, match="not a materialized task row"):
        prepare.read_materialized_rows(flat)
    stamped = tmp_path / "stamped.jsonl"
    stamped.write_text(json.dumps({**_materialized_row("x", "t"), "agent_ref": {"name": "a"}}) + "\n")
    with pytest.raises(ValueError, match="routes by taskset"):
        prepare.read_materialized_rows(stamped)
