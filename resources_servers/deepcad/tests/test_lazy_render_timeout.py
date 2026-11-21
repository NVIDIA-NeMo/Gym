# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render-timeout recovery without changing GRPO prompt identity."""

import json
import sqlite3
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from resources_servers.deepcad.dataset import lazy_inputs


def setup_rows(tmp_path):
    config = lazy_inputs.LazyRenderConfig(
        dataset_root=str(tmp_path),
        renderer_python="python",
        cache_root=str(tmp_path / "cache"),
        reference_catalog=str(tmp_path / "catalog.sqlite"),
        max_reference_replacements=2,
    )

    def example(name):
        return {
            "task_source": "deepcad_simple_agent",
            "deepcad_lazy": {
                "config": config.model_dump(),
                "record": {
                    "id": name,
                    "asset_id": name,
                    "task_id": name,
                    "source_kind": "cad_corpus",
                    "source_sha256": name,
                    "reference_step": f"references/{name}.step",
                    "difficulty": 5,
                },
            },
        }

    bad, good = example("bad"), example("good")
    with sqlite3.connect(config.reference_catalog) as db:
        db.execute("CREATE TABLE assets (id TEXT PRIMARY KEY,status TEXT,row_json TEXT)")
        db.execute("INSERT INTO assets VALUES (?,?,?)", ("good", "selected", json.dumps(good)))
    return bad, good


def worker(argv, **kwargs):
    record = json.loads(Path(argv[-2]).read_text())["record"]
    if record["id"] == "bad" and not kwargs.get("recover"):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
    row = {
        "verifier_metadata": {
            "task_id": record["task_id"],
            "source_sha256": record["source_sha256"],
            "reference_step": Path(record["reference_step"]).name,
            "difficulty": 5,
            "view_mode": "diagonal",
        },
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "data:image/png;base64,same", "detail": "high"}],
                }
            ]
        },
    }
    Path(argv[-1]).write_text(json.dumps(row))
    return SimpleNamespace(returncode=0)


def test_timeout_replacement_is_cached_and_identical_for_grpo(tmp_path, monkeypatch):
    bad, _ = setup_rows(tmp_path)
    calls = []

    def run(argv, **kwargs):
        calls.append(json.loads(Path(argv[-2]).read_text())["record"]["id"])
        return worker(argv, **kwargs)

    monkeypatch.setattr(lazy_inputs.subprocess, "run", run)
    rows = [{**deepcopy(bad), "_rowidx": i} for i in range(16)]
    lazy_inputs.materialize_examples(rows)
    again = lazy_inputs.materialize_one(bad)
    assert calls == ["bad", "good"]
    assert all(r["responses_create_params"] == again["responses_create_params"] for r in rows)
    assert [r["_rowidx"] for r in rows] == list(range(16))
    assert again["verifier_metadata"]["replaced_reference_asset_ids"] == ["bad"]
    assert again["verifier_metadata"]["reference_replacement_reasons"][0]["type"] == "ReferenceRenderTimeout"
    markers = list((tmp_path / "cache").glob("*.timeout-180.json"))
    assert len(markers) == 1
    evidence = json.loads(markers[0].read_text())
    assert evidence["status"] == "render_timeout_unverified"
    assert "confirmation_count" not in evidence
    assert not list((tmp_path / "cache").glob("*.rejected.json"))
    assert "deepcad_lazy" in bad


def test_timeout_budget_change_allows_retry(tmp_path, monkeypatch):
    bad, _ = setup_rows(tmp_path)
    monkeypatch.setattr(lazy_inputs.subprocess, "run", worker)
    for budget in (180, 181):
        bad["deepcad_lazy"]["config"]["timeout_s"] = budget
        lazy_inputs.materialize_one(bad)
    assert len(list((tmp_path / "cache").glob("*.timeout-*.json"))) == 2


def test_timeout_replacements_are_bounded(tmp_path, monkeypatch):
    bad, _ = setup_rows(tmp_path)

    calls = []

    def timeout(argv, **kwargs):
        calls.append(json.loads(Path(argv[-2]).read_text())["record"]["id"])
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(lazy_inputs.subprocess, "run", timeout)
    bad["deepcad_lazy"]["config"]["max_reference_replacements"] = 1
    with pytest.raises(lazy_inputs.ReferenceRenderTimeout):
        lazy_inputs.materialize_one(bad)
    assert calls == ["bad", "good"]


@pytest.mark.parametrize("kind", ["no_catalog", "deepcad", "outage"])
def test_other_errors_still_propagate(tmp_path, monkeypatch, kind):
    bad, _ = setup_rows(tmp_path)
    if kind == "no_catalog":
        bad["deepcad_lazy"]["config"]["reference_catalog"] = None
    elif kind == "deepcad":
        bad["deepcad_lazy"]["record"]["source_kind"] = "deepcad"

    def fail(*args, **kwargs):
        if kind == "outage":
            raise OSError("renderer unavailable")
        raise subprocess.TimeoutExpired("renderer", 180)

    monkeypatch.setattr(lazy_inputs.subprocess, "run", fail)
    with pytest.raises(OSError if kind == "outage" else subprocess.TimeoutExpired):
        lazy_inputs.materialize_one(bad)
    assert not list((tmp_path / "cache").glob("*.timeout-*.json"))


@pytest.mark.parametrize("exit_code", [-11, 139])
def test_segfault_replacement_logs_quarantine_and_preserves_grpo(tmp_path, monkeypatch, caplog, exit_code):
    bad, _ = setup_rows(tmp_path)
    calls = []

    def run(argv, **kwargs):
        record = json.loads(Path(argv[-2]).read_text())["record"]
        calls.append(record["id"])
        if record["id"] == "bad":
            return SimpleNamespace(returncode=exit_code, stderr="native CAD segfault")
        return worker(argv, **kwargs)

    monkeypatch.setattr(lazy_inputs.subprocess, "run", run)
    rows = [{**deepcopy(bad), "_rowidx": i} for i in range(16)]
    with pytest.raises(RuntimeError, match="Unconfirmed reference failure"):
        lazy_inputs.materialize_one(bad)
    assert not list((tmp_path / "cache").glob("*.rejected.json"))
    assert not list((tmp_path / "cache").glob("*.timeout-*.json"))
    lazy_inputs.materialize_examples(rows)
    again = lazy_inputs.materialize_one(bad)
    assert calls == ["bad", "bad", "good"]
    assert all(r["responses_create_params"] == again["responses_create_params"] for r in rows)
    assert [r["_rowidx"] for r in rows] == list(range(16))
    assert again["verifier_metadata"]["requested_task_id"] == "bad"
    assert again["verifier_metadata"]["replaced_reference_asset_ids"] == ["bad"]
    markers = list((tmp_path / "cache").glob("*.rejected.json"))
    assert len(markers) == 1
    evidence = json.loads(markers[0].read_text())
    assert evidence["return_code"] == exit_code
    assert evidence["asset_id"] == "bad" and evidence["source_sha256"] == "bad"
    assert evidence["status"] == "segfault_quarantined"
    assert evidence["stderr"] == "native CAD segfault"
    assert "Quarantined lazy reference" in caplog.text
    assert "Replacing lazy reference asset=bad" in caplog.text


@pytest.mark.parametrize("exit_code", [1, -9, 137])
def test_render_non_segfault_crashes_are_not_silently_skipped(tmp_path, monkeypatch, exit_code):
    bad, _ = setup_rows(tmp_path)
    monkeypatch.setattr(
        lazy_inputs.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=exit_code, stderr="outage")
    )
    with pytest.raises(RuntimeError, match="input render failed"):
        lazy_inputs.materialize_one(bad)
    assert not list((tmp_path / "cache").glob("*.rejected.json"))


def test_one_time_segfault_retries_original_asset(tmp_path, monkeypatch):
    bad, _ = setup_rows(tmp_path)
    attempts = []

    def run(argv, **kwargs):
        attempts.append(Path(argv[-2]))
        if len(attempts) == 1:
            return SimpleNamespace(returncode=-11, stderr="VTK crashed once")
        return worker(argv, **{**kwargs, "recover": True})

    monkeypatch.setattr(lazy_inputs.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Unconfirmed reference failure"):
        lazy_inputs.materialize_one(bad)
    # A batch failure cannot publish a replacement or mutate the caller.
    assert bad["deepcad_lazy"]["record"]["asset_id"] == "bad"
    result = lazy_inputs.materialize_one(bad)
    assert result["verifier_metadata"]["task_id"] == "bad"
    assert len(attempts) == 2 and attempts[0].parent != attempts[1].parent
    assert lazy_inputs.materialize_one(bad) == result
    assert not list((tmp_path / "cache").glob("*.failure.json"))
    assert not list((tmp_path / "cache").glob("*.rejected.json"))
    assert not list((tmp_path / "cache").glob("*.timeout-*.json"))


def test_non_corpus_crash_preserves_detail_without_asset_id(tmp_path, monkeypatch):
    bad, _ = setup_rows(tmp_path)
    bad["deepcad_lazy"]["record"]["source_kind"] = "deepcad"
    bad["deepcad_lazy"]["record"].pop("asset_id")
    monkeypatch.setattr(
        lazy_inputs.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=-11, stderr="VTK crash")
    )
    with pytest.raises(RuntimeError, match="VTK crash"):
        lazy_inputs.materialize_one(bad)
    assert not list((tmp_path / "cache").glob("*.rejected.json"))


def test_reference_kill_is_confirmed_before_replacement(tmp_path, monkeypatch):
    bad, _ = setup_rows(tmp_path)
    calls = []

    def run(argv, **kwargs):
        record = json.loads(Path(argv[-2]).read_text())["record"]
        calls.append(record["id"])
        if record["id"] == "bad":
            Path(argv[-1]).with_suffix(".rejected.json").write_text(
                json.dumps({"reason": "reference process killed", "confirmation_key": "reference_signal:137"})
            )
            return SimpleNamespace(returncode=0)
        return worker(argv, **kwargs)

    monkeypatch.setattr(lazy_inputs.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Unconfirmed reference failure"):
        lazy_inputs.materialize_one(bad)
    result = lazy_inputs.materialize_one(bad)
    assert result["verifier_metadata"]["replaced_reference_asset_ids"] == ["bad"]
    assert calls == ["bad", "bad", "good"]
    assert json.loads(next((tmp_path / "cache").glob("*.rejected.json")).read_text())["confirmation_count"] == 2


def test_changed_failure_does_not_confirm_previous_crash(tmp_path, monkeypatch):
    bad, _ = setup_rows(tmp_path)
    calls = 0

    def run(argv, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(returncode=-11, stderr="VTK crashed")
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(lazy_inputs.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Unconfirmed reference failure"):
        lazy_inputs.materialize_one(bad)
    # The new timeout reaches replacement without confirming the prior segfault.
    with pytest.raises(lazy_inputs.ReferenceRenderTimeout):
        lazy_inputs.materialize_one(bad)
    assert not list((tmp_path / "cache").glob("*.rejected.json"))
    marker = next((tmp_path / "cache").glob("*.timeout-*.json"))
    assert "confirmation_count" not in json.loads(marker.read_text())


def test_legacy_single_segfault_marker_does_not_block_recovery(tmp_path, monkeypatch):
    bad, _ = setup_rows(tmp_path)
    monkeypatch.setattr(
        lazy_inputs.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=-11, stderr="crash once")
    )
    with pytest.raises(RuntimeError, match="Unconfirmed reference failure"):
        lazy_inputs.materialize_one(bad)
    pending = next((tmp_path / "cache").glob("*.failure.json"))
    key = pending.name.removesuffix(".failure.json")
    marker = tmp_path / "cache" / (key + ".rejected.json")
    marker.write_text(
        json.dumps(
            {
                "reason": "old one-off failure",
                "status": "segfault_quarantined",
            }
        )
    )
    monkeypatch.setattr(lazy_inputs.subprocess, "run", lambda argv, **kw: worker(argv, **{**kw, "recover": True}))
    result = lazy_inputs.materialize_one(bad)
    assert result["verifier_metadata"]["task_id"] == "bad"
    assert not pending.exists()


def test_changed_renderer_does_not_confirm_prior_failure(tmp_path, monkeypatch):
    bad, _ = setup_rows(tmp_path)
    monkeypatch.setattr(
        lazy_inputs.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=-11, stderr="crash")
    )
    for renderer in ("python", "fixed-python"):
        bad["deepcad_lazy"]["config"]["renderer_python"] = renderer
        with pytest.raises(RuntimeError, match="Unconfirmed reference failure"):
            lazy_inputs.materialize_one(bad)
    assert not list((tmp_path / "cache").glob("*.rejected.json"))


@pytest.mark.parametrize("execution_budget", [60, 61])
def test_execution_timeout_replaces_on_first_observation(tmp_path, monkeypatch, execution_budget):
    bad, _ = setup_rows(tmp_path)
    bad["deepcad_lazy"]["config"]["reference_execution_timeout_s"] = execution_budget
    calls = []

    def run(argv, **kwargs):
        record = json.loads(Path(argv[-2]).read_text())["record"]
        calls.append(record["id"])
        if record["id"] == "bad":
            Path(argv[-1]).with_suffix(".rejected.json").write_text(
                json.dumps(
                    {
                        "reason": "Reference program exceeded execution budget",
                        "confirmation_key": "reference_execution_timeout",
                    }
                )
            )
            return SimpleNamespace(returncode=0)
        return worker(argv, **kwargs)

    monkeypatch.setattr(lazy_inputs.subprocess, "run", run)
    rows = [{**deepcopy(bad), "_rowidx": i} for i in range(16)]
    lazy_inputs.materialize_examples(rows)
    again = lazy_inputs.materialize_one(bad)
    assert calls == ["bad", "good"]
    assert all(row["responses_create_params"] == again["responses_create_params"] for row in rows)
    assert [row["_rowidx"] for row in rows] == list(range(16))
    assert again["verifier_metadata"]["replaced_reference_asset_ids"] == ["bad"]
    assert again["verifier_metadata"]["reference_replacement_reasons"][0]["type"] == "ReferenceRenderTimeout"
    marker = next((tmp_path / "cache").glob("*.timeout-180.json"))
    assert json.loads(marker.read_text())["status"] == "render_timeout_unverified"
    assert not list((tmp_path / "cache").glob("*.rejected.json"))
    assert not list((tmp_path / "cache").glob("*.failure.json"))
