# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The build step: rollouts.jsonl in, one timeline record per rollout out, tool text tokenised."""

from __future__ import annotations

import json
from pathlib import Path

import orjson
from ng_latency.build import build_run_timeline

from tests.unit_tests.latency.conftest import opencode_row, terminus_row


def _write_run(tmp_path: Path, rows: list[dict], failures: list[dict] | None = None) -> Path:
    run_dir = tmp_path / "replica_0"
    run_dir.mkdir()
    (run_dir / "rollouts.jsonl").write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in rows))
    if failures is not None:
        (run_dir / "rollouts_failures.jsonl").write_bytes(b"".join(orjson.dumps(f) + b"\n" for f in failures))
    return run_dir


def test_build_writes_one_record_per_rollout_and_counts_the_failure_sidecar(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        [terminus_row(), opencode_row(), {"_ng_task_index": 3, "_ng_rollout_index": 0, "reward": 0.0}],
        failures=[{"_ng_failure_class": "agent_request_failed"}, {"_ng_failure_class": "agent_request_failed"}],
    )
    summary = build_run_timeline(run_dir, tokenize_url=None)
    lines = [
        orjson.loads(line) for line in (run_dir / "metrics" / "rollouts_timeline.jsonl").read_bytes().splitlines()
    ]
    assert lines[0]["kind"] == "header" and "clock_notes" in lines[0] and lines[0]["schema_version"] == "2.0"
    assert [line["rollout_id"] for line in lines[1:]] == ["4-1", "9-0"]
    assert summary["rollouts"] == 3 and summary["rollouts_with_timeline"] == 2
    assert summary["model_calls"] == 6 and summary["server_coverage"] == 1.0
    assert summary["agent_kinds"] == {"terminus2": 1, "opencode": 1}
    assert summary["flags"] == {"tool_response_boundary_approximate": 1}
    assert summary["gap_histogram"] == {"timeline_unavailable": 1}
    failures = summary["collector_failures"]
    assert failures["present"] and failures["rows"] == 2 and failures["by_class"] == {"agent_request_failed": 2}
    assert failures["path"].endswith("replica_0/rollouts_failures.jsonl")
    assert json.loads((run_dir / "metrics" / "timeline_summary.json").read_text())["rollouts"] == 3


def test_tool_arguments_and_outputs_are_tokenised_and_arguments_credited_to_the_producing_call(tmp_path) -> None:
    run_dir = _write_run(tmp_path, [terminus_row(), opencode_row()])
    seen: list[str] = []

    def counter(texts: list[str]) -> list[int | None]:
        seen.extend(texts)
        return [len(text.split()) if text else 0 for text in texts]

    import ng_latency.build as build_module

    original = build_module.count_tokens
    build_module.count_tokens = lambda url, texts: counter(texts)
    try:
        build_run_timeline(run_dir, tokenize_url="http://engine/tokenize")
    finally:
        build_module.count_tokens = original
    records = [
        orjson.loads(line) for line in (run_dir / "metrics" / "rollouts_timeline.jsonl").read_bytes().splitlines()
    ][1:]
    terminus, opencode = records
    by_name = lambda rec, name: [s for s in rec["spans"] if s["name"] == name]  # noqa: E731
    # Terminus: the command text is the argument; the batch's observation (output item 3) is the output.
    tools = sorted(by_name(terminus, "tool_call"), key=lambda s: s["start_wall_s"])
    assert tools[0]["attrs"]["tokens"]["tool_args"] == 2 and tools[1]["attrs"]["tokens"]["tool_args"] == 2
    assert tools[0]["attrs"]["tokens_source"] == "tokenize_standalone"
    batches = sorted(by_name(terminus, "tool_batch"), key=lambda s: s["start_wall_s"])
    assert batches[0]["attrs"]["tokens"]["tool_output"] == 1  # "x" * 240 is one whitespace token
    servers = sorted(by_name(terminus, "model_call_server"), key=lambda s: s["start_wall_s"])
    assert servers[0]["attrs"]["tokens"]["tool_args"] == 4  # both commands came out of step 1
    assert "tool_args" not in servers[1]["attrs"]["tokens"]
    # OpenCode: arguments from the recorded function_call items, outputs from their function_call_output items.
    oc_tools = {s["ids"]["tool_call_id"]: s for s in by_name(opencode, "tool_call")}
    assert oc_tools["call-bash"]["attrs"]["tokens"] == {"tool_args": 3, "tool_output": 1}
    oc_servers = sorted(by_name(opencode, "model_call_server"), key=lambda s: s["start_wall_s"])
    assert oc_servers[0]["attrs"]["tokens"]["tool_args"] == 3 + 2  # both tools came out of step 1
    assert '{"command": "pytest -q"}' in seen
