# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preparation builds both arms from a checksum-verified snapshot without touching the prompts."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.acmg import prepare
from resources_servers.acmg.task_data import TaskData


SYSTEM = "Classify the variant.\n\nCRITERIA: ...\nCLASSIFICATION: ...\n"
RULE = "# GENE1 rules\n\nPM2_Supporting when absent.\n"
CASE = {
    "case_id": "case_a",
    "rule_document": "GENE1_v1.md",
    "packet": {
        "record_id": "case_a",
        "gene": "GENE1",
        "evidence": [{"id": "E1", "kind": "population", "observation": "Absent."}],
    },
}
KEY = {
    "case_id": "case_a",
    "label": "VUS",
    "label_scoring": "scored",
    "accepted_labels": ["VUS"],
    "required": [{"code": "PM2", "strengths": ["Supporting"]}],
    "acceptable": [{"code": "PM2", "strength": "Supporting"}],
    "combining_rules": {"systems": ["general"], "rules": []},
    "gene": "GENE1",
    "role": "control",
    "skill": "control",
    "consequence": "missense",
    "approved_2026": True,
    "version_status": "matched",
    "confirmation": "pending",
}


def _snapshot(tmp_path: Path, monkeypatch) -> Path:
    files = {
        "data/cases.jsonl": json.dumps(CASE) + "\n",
        "data/answers.jsonl": json.dumps(KEY) + "\n",
        "prompts/system_prompt.md": SYSTEM,
        "rules/GENE1_v1.md": RULE,
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    checksums = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
    (tmp_path / "CHECKSUMS.json").write_text(json.dumps(checksums))
    monkeypatch.setattr(prepare, "CASES", 1)
    monkeypatch.setattr(prepare, "PINNED_SHA256", {k: checksums[k] for k in prepare.PINNED_SHA256})
    return tmp_path


def test_prepare_builds_both_arms_from_the_snapshot(tmp_path, monkeypatch) -> None:
    source = _snapshot(tmp_path, monkeypatch)
    output = tmp_path / "out" / "acmg_benchmark.jsonl"
    assert prepare.prepare(source, output) == output
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["verifier_metadata"]["record_id"] for r in rows] == ["case_a:with_rules", "case_a:without_rules"]
    with_rules, without_rules = rows
    assert with_rules["responses_create_params"]["input"][0]["content"] == SYSTEM.strip() + "\n\n" + RULE.strip()
    assert without_rules["responses_create_params"]["input"][0]["content"] == SYSTEM.strip()
    packet = json.dumps(CASE["packet"], sort_keys=True, separators=(",", ":"))
    assert all(r["responses_create_params"]["input"][1] == {"role": "user", "content": packet} for r in rows)
    for row in rows:
        task = TaskData.model_validate(row["verifier_metadata"])
        assert task.reference.required == {"PM2": ["Supporting"]}
        assert task.reference.acceptable == [("PM2", "Supporting")]
        assert task.reference.combining == {"systems": ["general"], "rules": []}
        assert (task.gene, task.rule_document, task.approved_2026) == ("GENE1", "GENE1_v1.md", True)
    # The packet, not the answer key, is what the model sees.
    assert "VUS" not in without_rules["responses_create_params"]["input"][1]["content"]


def test_prepare_refuses_a_changed_source_file(tmp_path, monkeypatch) -> None:
    source = _snapshot(tmp_path, monkeypatch)
    (source / "rules/GENE1_v1.md").write_text(RULE + "An extra line.\n")
    output = tmp_path / "out" / "acmg_benchmark.jsonl"
    output.parent.mkdir()
    output.write_text("previous\n")
    with pytest.raises(ValueError, match="does not match revision"):
        prepare.prepare(source, output)
    assert output.read_text() == "previous\n"


def test_prepare_refuses_an_unexpected_case_count(tmp_path, monkeypatch) -> None:
    source = _snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(prepare, "CASES", 395)
    with pytest.raises(ValueError, match="395 cases"):
        prepare.prepare(source, tmp_path / "out.jsonl")


def test_prepare_downloads_when_no_snapshot_is_given(tmp_path, monkeypatch) -> None:
    source = _snapshot(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(prepare, "download_snapshot", lambda: calls.append("download") or source)
    output = tmp_path / "out.jsonl"
    assert prepare.prepare(output_path=output) == output
    assert calls == ["download"]
