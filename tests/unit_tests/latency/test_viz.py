# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The single timelines page: both synthetic rollouts embedded, lanes and placement as designed."""

from __future__ import annotations

import json
import re

import pytest
from ng_latency.derive import derive_trial
from ng_latency.project import project_rollout
from ng_latency.viz import render_timelines_html, timeline_items


@pytest.fixture
def entries(terminus_record, opencode_record):
    out = []
    for row in (terminus_record, opencode_record):
        record = project_rollout(row)
        out.append((record, derive_trial(record, row, replica="replica_0")))
    return out


def test_terminus_items_carry_engine_phases_critical_path_and_gaps(entries):
    record, trial = entries[0]
    page = timeline_items(record, trial)
    lanes = {lane["id"]: lane for lane in page["lanes"]}
    assert set(lanes) >= {"phase", "model", "attempts", "engine", "tool", "harness", "compaction", "gap"}
    by_lane: dict[str, list[dict]] = {}
    for item in page["items"]:
        by_lane.setdefault(item["l"], []).append(item)
    assert len(by_lane["model"]) == 3 and sum(i["cp"] for i in by_lane["model"]) == 3
    attempts = by_lane["attempts"]
    assert sum(i["n"] == "model_call_server" for i in attempts) == 3
    assert [i["st"] for i in attempts if i["n"] == "model_call_attempt" and i["st"] != "ok"] == ["timeout"]
    # the engine lane ends where the server span ends and is split into queue / prefill / decode
    engine = sorted(by_lane["engine"], key=lambda i: i["s"])
    assert [i["n"] for i in engine[:3]] == ["queue", "prefill", "decode"]
    server_ends = sorted(i["e"] for i in by_lane["attempts"] if i["n"] == "model_call_server")
    assert engine[2]["e"] == pytest.approx(server_ends[0], abs=1e-6)
    assert lanes["tool"]["rows"] >= 1 and len(by_lane["tool"]) == 3 + 2  # commands + synthesized batches
    assert sum(i["e"] - i["s"] for i in by_lane["gap"]) == pytest.approx(9.2, abs=1e-3)
    assert page["placement_note"] == "all spans on the node clock"
    assert page["t0"] == 0.0 and page["t1"] == pytest.approx(94.9995, abs=1e-3)
    assert ["flags", "unexplained_gap_over_5pct"] in page["kv"]


def test_opencode_sandbox_spans_are_placed_by_the_probe(entries):
    record, trial = entries[1]
    page = timeline_items(record, trial)
    assert page["placement_note"].startswith("sandbox:") and "sandbox clock probe" in page["placement_note"]
    sub_lanes = [lane for lane in page["lanes"] if lane["id"].startswith("subagent:")]
    assert len(sub_lanes) == 1
    agent = next(i for i in page["items"] if i["n"] == "agent")
    model = [i for i in page["items"] if i["l"] == "model"]
    assert all(agent["s"] - 1e-6 <= i["s"] and i["e"] <= agent["e"] + 1e-6 for i in model)


def test_page_embeds_every_rollout_once_and_is_self_contained(entries):
    html_text = render_timelines_html(entries)
    assert html_text.count("<script") == 2 and "http" not in html_text.split("<script")[0].split("<style")[1]
    payload = re.search(r'<script id="data" type="application/json">(.*?)</script>', html_text, re.S).group(1)
    data = json.loads(payload)
    assert set(data) == {"replica_0/" + str(t["rollout_id"]) for _, t in entries}
    body = html_text.split('<script id="data"')[0]
    assert body.count("<tr>") == 1 + len(entries)  # header + one index row per rollout
    assert "\\u003c" in payload or "<" not in payload
