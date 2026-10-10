# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reporting toolchain tests on a small, legally safe fixture run (shared shape across the three benchmarks)."""

import json
from pathlib import Path

import pytest

from benchmarks.facts_parametric.reporting import model_card
from benchmarks.facts_parametric.reporting.cli import main as cli_main
from benchmarks.facts_parametric.reporting.common import load_run, reconcile
from benchmarks.facts_parametric.reporting.model_card import PROMPT_TEMPLATE
from benchmarks.facts_parametric.reporting.normalize import normalize, receipts_for
from benchmarks.facts_parametric.reporting.package import validate_package, verify_checksums
from benchmarks.facts_parametric.reporting.render import to_html, to_markdown
from benchmarks.facts_parametric.reporting.schema import CalibrationSummary, ModelCardDocument


_FIXTURE = json.loads((Path(__file__).parent / "reporting_fixture.json").read_text(encoding="utf-8"))
STEM = _FIXTURE["stem"]
RUN_INFO = _FIXTURE["run_info"]
FIXTURE_ROLLOUTS = _FIXTURE["rollouts"]
FIXTURE_FAILURES = _FIXTURE["failures"]


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / f"{STEM}.jsonl").open("w", encoding="utf-8") as handle:
        for row in FIXTURE_ROLLOUTS:
            handle.write(json.dumps(row) + "\n")
    with (run_dir / f"{STEM}_failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in FIXTURE_FAILURES:
            handle.write(json.dumps(row) + "\n")
    materialized = {
        row["_ng_task_index"]: {"id": row["id"], "_ng_task_index": row["_ng_task_index"]}
        for row in FIXTURE_ROLLOUTS + FIXTURE_FAILURES
    }
    with (run_dir / f"{STEM}_materialized_inputs.jsonl").open("w", encoding="utf-8") as handle:
        for index in sorted(materialized):
            handle.write(json.dumps(materialized[index]) + "\n")
    (run_dir / f"{STEM}_aggregate_metrics.json").write_text(
        json.dumps([{"agent_ref": {"name": STEM}, "agent_metrics": {"mean/output_tokens": 100.0}}]), encoding="utf-8"
    )
    return run_dir


def _calibration(tmp_path: Path) -> Path:
    cal = tmp_path / "calibration"
    cal.mkdir()
    (cal / "upstream-vs-gym.jsonl").write_text(json.dumps({"case_id": "c1", "agree": True}) + "\n", encoding="utf-8")
    (cal / "summary.json").write_text(
        json.dumps(
            {
                "normalized": {
                    "method": "replay",
                    "cases": 1,
                    "agreement": 1,
                    "disagreements": 0,
                    "notes": ["replay: 1 of 1 cases agree"],
                    "details": {"case_ids": ["c1"]},
                }
            }
        ),
        encoding="utf-8",
    )
    return cal


def test_reconciliation_separates_failure_classes_and_superseded_attempts(tmp_path):
    artifacts = load_run(_write_run(tmp_path), STEM)
    outcomes, scored, unresolved = reconcile(artifacts, task_id_of=lambda row: str(row["id"]))
    assert outcomes.expected_rollouts == len(FIXTURE_ROLLOUTS) + 1
    assert outcomes.scored_rollouts == len(FIXTURE_ROLLOUTS)
    assert outcomes.judge_failed + outcomes.simulation_failed + outcomes.infrastructure_failed == 1
    assert outcomes.replaced_attempts == 1 and outcomes.missing_rollouts == 0
    assert len(unresolved) == 1


def test_normalize_produces_grounded_metrics_and_anchors(tmp_path):
    artifacts = load_run(_write_run(tmp_path), STEM)
    run = normalize(
        artifacts,
        run_info=dict(RUN_INFO) | {"run_id": "fixture", "model": "fixture-model"},
        calibration=CalibrationSummary(
            method="replay", cases=1, agreement=1, disagreements=0, details={"case_ids": ["c1"]}
        ),
    )
    primary = run.primary_metric()
    assert primary.value is not None and primary.denominator
    known = (
        {r.task_id for r in run.rollouts}
        | {r.rollout_id for r in run.rollouts}
        | {rid for r in run.rollouts for rid in r.receipt_ids}
        | {"c1"}
    )
    for fact in run.anchor_facts:
        assert fact.evidence and all(evidence in known for evidence in fact.evidence), fact.id
    assert any(fact.category == "example" for fact in run.anchor_facts)
    assert run.blade.not_applicable and "pass@k" in run.blade.not_applicable
    assert all(receipt["receipt_id"] for row in FIXTURE_ROLLOUTS for receipt in receipts_for(row))


def test_package_build_is_deterministic_validates_and_detects_tampering(tmp_path):
    run_dir = _write_run(tmp_path)
    cal = _calibration(tmp_path)
    info = tmp_path / "run-info.json"
    info.write_text(json.dumps(RUN_INFO), encoding="utf-8")
    args = [
        "build",
        "--run-dir",
        str(run_dir),
        "--stem",
        STEM,
        "--run-id",
        "fixture",
        "--model",
        "fixture-model",
        "--run-info",
        str(info),
        "--calibration-dir",
        str(cal),
        "--generated-at",
        "2026-01-01T00:00:00Z",
    ]
    package_a = tmp_path / "a" / "fixture-package"
    package_b = tmp_path / "b" / "fixture-package"
    cli_main(args + ["--package", str(package_a)])
    cli_main(args + ["--package", str(package_b)])
    checksums_a = (package_a / "checksums.sha256").read_text(encoding="utf-8")
    assert checksums_a == (package_b / "checksums.sha256").read_text(encoding="utf-8")
    assert {"README.md", "manifest", "raw", "calibration", "blade", "prose", "checksums.sha256"} <= {
        p.name for p in package_a.iterdir()
    }
    assert validate_package(package_a) == []
    receipts = (package_a / "raw" / "judge-or-verifier-receipts.jsonl").read_text(encoding="utf-8").splitlines()
    assert receipts and all(json.loads(line)["receipt_id"] for line in receipts)
    metrics = json.loads((package_a / "blade" / "metrics.json").read_text(encoding="utf-8"))
    metrics["primary_metric"]["value"] = 0.123456
    (package_a / "blade" / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    problems = validate_package(package_a)
    assert any("checksum mismatch" in p for p in problems) and any("primary metric" in p for p in problems)
    cli_main(["validate", "--package", str(package_b)])
    with pytest.raises(SystemExit):
        cli_main(["validate", "--package", str(package_a)])
    assert verify_checksums(package_b) == []


class _FakeClient:
    base_url = "https://example.invalid/v1"
    model = "fake-writer"

    def __init__(self, documents):
        self.documents = list(documents)
        self.calls = 0

    def complete(self, messages, *, schema):
        self.calls += 1
        return json.dumps(self.documents.pop(0)), {"model": self.model, "usage": {"total_tokens": 1}}


def _good_document(run):
    example_ids = [fact.id for fact in run.anchor_facts if fact.category == "example"][:4]
    coverage = next(fact.id for fact in run.anchor_facts if fact.category == "coverage")
    outcome = next(fact.id for fact in run.anchor_facts if fact.category == "outcome")
    return {
        "purpose": "The benchmark measures the property described in the protocol.",
        "what_the_run_says": [
            {
                "text": f"{run.outcomes.scored_rollouts} of {run.outcomes.expected_rollouts} expected rollouts were scored.",
                "evidence": [coverage],
            },
            {"text": "The primary metric is reported with its denominator in the table.", "evidence": [outcome]},
        ],
        "examples": [
            {
                "anchor_id": anchor_id,
                "label": "Example",
                "description": "A sanitized description of what the model did.",
            }
            for anchor_id in example_ids
        ],
        "calibration_and_reference": [{"text": "The replay calibration agreed on every case.", "evidence": [outcome]}],
        "limitations": [
            {"text": "Only the public split was scored.", "evidence": [coverage]},
            {"text": "The judge samples, so verdicts carry noise.", "evidence": [outcome]},
        ],
        "interpretation": "The numbers describe the scored rollouts only. They are not a leaderboard submission. The judge protocol follows the published reference implementation.",
    }


def test_model_card_accepts_grounded_prose_and_rejects_invented_numbers(tmp_path):
    artifacts = load_run(_write_run(tmp_path), STEM)
    run = normalize(
        artifacts, run_info=dict(RUN_INFO) | {"run_id": "fixture", "model": "fixture-model"}, calibration=None
    )
    good = _good_document(run)
    bad = json.loads(json.dumps(good))
    bad["interpretation"] = "The model reached 93.7% on this benchmark, which is state-of-the-art."
    bad["limitations"][0]["evidence"] = ["fact-does-not-exist"]
    client = _FakeClient([bad, good])
    document, record = model_card.generate(run, client, max_attempts=2)
    assert client.calls == 2
    assert record["attempts"][0]["problems"] and any("93.7" in p for p in record["attempts"][0]["problems"])
    assert any("unknown evidence" in p for p in record["attempts"][0]["problems"])
    assert any("forbidden" in p for p in record["attempts"][0]["problems"])
    assert isinstance(document, ModelCardDocument)
    markdown = to_markdown(run, document)
    html = to_html(run, document)
    assert "SNORKEL AI x NVIDIA" in markdown and "What the run says" in markdown
    assert "Snorkel AI" in html and run.run["model"] in html and "<table" in html
    failing = _FakeClient([bad, bad])
    with pytest.raises(ValueError, match="failed closed"):
        model_card.generate(run, failing, max_attempts=2)


def test_openai_client_merges_extra_body_and_never_leaks_the_key(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "id": "x",
                    "model": "m",
                    "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                    "usage": {},
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode())
        captured["headers"] = dict(request.header_items())
        return _Response()

    monkeypatch.setattr(model_card.urllib.request, "urlopen", fake_urlopen)
    client = model_card.OpenAICompatibleClient(
        "https://example.invalid/v1/",
        "writer",
        "secret-key",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content, metadata = client.complete([{"role": "user", "content": "hi"}], schema={"type": "object"})
    assert content == "{}" and captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["body"]["response_format"]["type"] == "json_schema"
    assert "secret-key" not in json.dumps(metadata) and "secret-key" not in json.dumps(captured["body"])
    assert captured["headers"]["Authorization"] == "Bearer secret-key"


def test_prompt_template_is_packaged_with_the_writer():
    # The template must live outside any "temp*" directory (gitignored) and carry no underscore in its name
    # (Markdown filename hook), or a fresh checkout cannot run the report writer.
    assert PROMPT_TEMPLATE.is_file(), PROMPT_TEMPLATE
    assert "temp" not in PROMPT_TEMPLATE.parent.name and "_" not in PROMPT_TEMPLATE.name
    text = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    assert "{context_json}" in text
    assert "320 characters" in text
