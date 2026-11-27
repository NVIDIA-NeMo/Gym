# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in operation fallback retains source, bounds, and complete prompt groups."""

import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from resources_servers.deepcad.dataset import lazy_inputs
from resources_servers.deepcad.dataset.corpus_inputs import ReferenceRejected
from resources_servers.deepcad.tests.test_lazy_render_timeout import setup_rows, worker


def examples(tmp_path, candidates, *, allow=True):
    bad, good = setup_rows(tmp_path)
    bad["deepcad_lazy"]["config"]["allow_partial_operation_replacements"] = allow
    bad["deepcad_lazy"]["record"].update(dataset="source-a", reference_operations=["cut", "hole", "loft"])
    with sqlite3.connect(bad["deepcad_lazy"]["config"]["reference_catalog"]) as db:
        db.execute("DELETE FROM assets")
        for name, operations, source, status in candidates:
            candidate = deepcopy(good)
            candidate["deepcad_lazy"]["record"].update(
                id=name,
                asset_id=name,
                task_id=name,
                source_sha256=name,
                reference_step=f"references/{name}.step",
                dataset=source,
                reference_operations=operations,
            )
            db.execute("INSERT INTO assets VALUES (?,?,?)", (name, status, json.dumps(candidate)))
    return bad


def renderer(monkeypatch, rejected=("bad",)):
    calls = []

    def run(argv, **kwargs):
        name = json.loads(Path(argv[-2]).read_text())["record"]["id"]
        calls.append(name)
        if name in rejected:
            Path(argv[-1]).with_suffix(".rejected.json").write_text(json.dumps({"reason": "Invalid CAD solid"}))
            return SimpleNamespace(returncode=0)
        return worker(argv, **kwargs)

    monkeypatch.setattr(lazy_inputs.subprocess, "run", run)
    return calls


def test_strict_default_does_not_relax_operations(tmp_path, monkeypatch):
    bad = examples(tmp_path, [("near", ["cut", "hole"], "source-a", "selected")], allow=False)
    assert lazy_inputs.LazyRenderConfig.model_fields["allow_partial_operation_replacements"].default is False
    calls = renderer(monkeypatch)
    with pytest.raises(ReferenceRejected, match="Invalid CAD solid"):
        lazy_inputs.materialize_one(bad)
    assert calls == ["bad"]


def test_best_overlap_preserves_source_and_records_missing_coverage(tmp_path, monkeypatch, caplog):
    bad = examples(
        tmp_path,
        [
            ("cross-source", ["cut", "hole", "loft"], "source-b", "selected"),
            ("inactive", ["cut", "hole", "loft"], "source-a", "rejected"),
            ("less", ["cut", "cut", "cut"], "source-a", "selected"),
            ("best", ["hole", "loft"], "source-a", "selected"),
        ],
    )
    calls = renderer(monkeypatch)
    result = lazy_inputs.materialize_one(bad)
    assert calls == ["bad", "best"]
    assert result["verifier_metadata"]["task_id"] == "best"
    assert result["verifier_metadata"]["reference_operation_fallback"] == {
        "required_operations": ["cut", "hole", "loft"],
        "retained_operations": ["hole", "loft"],
        "missing_operations": ["cut"],
    }
    assert "source=source-a" in caplog.text and "missing=['cut']" in caplog.text
    assert bad["deepcad_lazy"]["record"]["asset_id"] == "bad"


def test_full_coverage_candidate_wins_before_fallback(tmp_path, monkeypatch):
    bad = examples(
        tmp_path,
        [
            ("near", ["cut", "hole"], "source-a", "selected"),
            ("exact", ["cut", "hole", "loft", "union"], "source-a", "selected"),
        ],
    )
    calls = renderer(monkeypatch)
    result = lazy_inputs.materialize_one(bad)
    assert calls == ["bad", "exact"]
    assert "reference_operation_fallback" not in result["verifier_metadata"]


def test_fallback_is_identical_for_whole_grpo_group(tmp_path, monkeypatch):
    bad = examples(
        tmp_path,
        [
            ("tie-a", ["cut", "hole"], "source-a", "selected"),
            ("tie-b", ["hole", "loft"], "source-a", "selected"),
        ],
    )
    bad["agent_ref"] = {"name": "chosen-agent"}
    calls = renderer(monkeypatch)
    rows = [{**deepcopy(bad), "_rowidx": i} for i in range(16)]
    lazy_inputs.materialize_examples(rows)
    again = lazy_inputs.materialize_one(bad)
    assert len(calls) == 2 and calls[0] == "bad"
    assert all(row["responses_create_params"] == again["responses_create_params"] for row in rows)
    assert all(row["verifier_metadata"] == again["verifier_metadata"] for row in rows)
    assert all(row["agent_ref"] == bad["agent_ref"] for row in rows)
    assert [row["_rowidx"] for row in rows] == list(range(16))


def test_failed_fallback_is_excluded_and_next_best_is_selected(tmp_path, monkeypatch):
    bad = examples(
        tmp_path,
        [
            ("best", ["cut", "hole"], "source-a", "selected"),
            ("next", ["loft"], "source-a", "selected"),
        ],
    )
    calls = renderer(monkeypatch, rejected=("bad", "best"))
    result = lazy_inputs.materialize_one(bad)
    assert calls == ["bad", "best", "next"]
    assert result["verifier_metadata"]["replaced_reference_asset_ids"] == ["bad", "best"]
    assert result["verifier_metadata"]["reference_operation_fallback"]["missing_operations"] == ["cut", "hole"]


def test_partial_replacements_respect_the_existing_bound(tmp_path, monkeypatch):
    bad = examples(
        tmp_path,
        [
            ("best", ["cut", "hole"], "source-a", "selected"),
            ("next", ["loft"], "source-a", "selected"),
        ],
    )
    bad["deepcad_lazy"]["config"]["max_reference_replacements"] = 1
    calls = renderer(monkeypatch, rejected=("bad", "best"))
    with pytest.raises(ReferenceRejected):
        lazy_inputs.materialize_one(bad)
    assert calls == ["bad", "best"]


@pytest.mark.parametrize("source,status", [("source-b", "selected"), ("source-a", "rejected")])
def test_no_same_source_selected_candidate_still_fails(tmp_path, monkeypatch, source, status):
    bad = examples(tmp_path, [("ineligible", ["cut", "hole"], source, status)])
    calls = renderer(monkeypatch)
    with pytest.raises(ReferenceRejected):
        lazy_inputs.materialize_one(bad)
    assert calls == ["bad"]


def test_fallback_setting_preserves_existing_render_cache(tmp_path, monkeypatch):
    _, good = setup_rows(tmp_path)
    calls = renderer(monkeypatch)
    first = lazy_inputs._materialize_one(good)
    good["deepcad_lazy"]["config"]["allow_partial_operation_replacements"] = True
    assert lazy_inputs._materialize_one(good) == first
    assert calls == ["good"]


def test_indexed_catalog_keeps_json_source_guard(tmp_path, monkeypatch):
    bad = examples(
        tmp_path,
        [
            ("cross-source", ["cut", "hole", "loft"], "source-b", "selected"),
            ("best", ["hole", "loft"], "source-a", "selected"),
        ],
    )
    catalog = bad["deepcad_lazy"]["config"]["reference_catalog"]
    with sqlite3.connect(catalog) as db:
        db.execute("ALTER TABLE assets ADD COLUMN dataset TEXT")
        # A stale physical field must not let a different JSON source through.
        db.execute("UPDATE assets SET dataset='source-a'")
        db.execute("CREATE INDEX browse ON assets(status,dataset,id)")
    calls = renderer(monkeypatch)
    real_connect = sqlite3.connect
    statements = []

    def connect(*args, **kwargs):
        db = real_connect(*args, **kwargs)
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(lazy_inputs.sqlite3, "connect", connect)
    result = lazy_inputs.materialize_one(bad)
    assert calls == ["bad", "best"]
    assert result["verifier_metadata"]["reference_operation_fallback"]["missing_operations"] == ["cut"]
    selections = [sql for sql in statements if sql.startswith("SELECT row_json FROM assets")]
    assert selections
    with real_connect(catalog) as db:
        for sql in selections:
            plan = db.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
            assert any("USING INDEX browse (status=? AND dataset=?" in row[-1] for row in plan), plan
