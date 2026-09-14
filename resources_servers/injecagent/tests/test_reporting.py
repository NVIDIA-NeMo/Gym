# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared reporting-core tests: determinism, validation, BLADE, fail-closed model-card writing."""

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.injecagent.reporting import model_card, package, render
from benchmarks.injecagent.reporting.schema import (
    AnchorFact,
    CalibrationSummary,
    MetricValue,
    ModelCardDocument,
    NormalizedRun,
    OutcomeCounts,
    ReferenceComparison,
    RolloutSummary,
)


def _run() -> NormalizedRun:
    rollouts = [
        RolloutSummary(
            task_id=f"task-{i}",
            rollout_id=f"rollout-{i}",
            receipt_id=f"receipt-task-{i}",
            slices={"group": "a" if i % 2 else "b"},
            outcome_class="policy_scored",
            reward=float(i % 2),
            facts={"label": i % 2},
            prompt_excerpt="p",
            response_excerpt="r",
        )
        for i in range(4)
    ]
    return NormalizedRun(
        benchmark={
            "id": "demo",
            "display_name": "Demo Bench",
            "protocol": "demo protocol, k=1",
            "dataset": {
                "name": "demo.jsonl",
                "count": 4,
                "sha256": "0" * 64,
                "cohort": "4 demo rows",
                "license": "MIT",
            },
            "paper": {"title": "t", "citation": "c", "url": "u", "version": "v1", "sha256": None},
            "upstream": {"repository": "https://example.invalid/repo", "revision": "abc"},
        },
        run={
            "run_id": "run-1",
            "model": "demo/model",
            "endpoint_type": "test",
            "harness": "simple_agent",
            "gym_revision": "deadbeef",
            "sampling": {"temperature": 0.0},
            "generation_limits": {"max_output_tokens": 16},
            "finished_at": "2026-09-14T00:00:00+00:00",
            "verifier": {"summary": "demo verifier"},
            "generation_summary": "temperature 0.0",
            "protocol_box": {"Benchmark": "Demo"},
            "fine_print": "fine print",
        },
        outcomes=OutcomeCounts(
            expected_tasks=4,
            expected_rollouts=4,
            scored_rollouts=4,
            policy_scored=4,
            invalid_model_output=0,
            infrastructure_failure=0,
            judge_missing=0,
            pending_annotation=0,
        ),
        metrics=[
            MetricValue(
                id="rate_primary",
                name="Primary rate",
                value=0.5,
                numerator=2,
                denominator=4,
                unit="rate",
                direction="lower_is_better",
                kind="primary",
                definition="d",
            ),
            MetricValue(
                id="rate_slice/group/a",
                name="Rate · a",
                value=1.0,
                numerator=2,
                denominator=2,
                unit="rate",
                direction="lower_is_better",
                kind="component",
                slice={"group": "a"},
                definition="d",
            ),
            MetricValue(
                id="count_ops",
                name="Ops count",
                value=3,
                unit="count",
                direction="neutral",
                kind="operational",
                definition="d",
            ),
        ],
        rollouts=rollouts,
        anchor_facts=[
            AnchorFact(
                id="fact-1", category="outcome", fact="Primary rate is 50.0% (2/4).", evidence=["receipt-task-1"]
            ),
            AnchorFact(
                id="fact-cal",
                category="calibration",
                fact="Calibration agreed on 4 of 4 cases.",
                evidence=["receipt-task-0"],
            ),
            AnchorFact(
                id="example-1",
                category="example",
                fact="task-1: example behavior",
                evidence=["receipt-task-1", "task-1"],
                excerpt="example excerpt",
            ),
        ],
        calibration=CalibrationSummary(method="m", cases=4, agreement=4, disagreements=0, notes=["n"], details={}),
        limitations=["k=1 only.", "Demo limitation."],
        reference_comparisons=[
            ReferenceComparison(label="ref", value="12.5%", source="paper", comparability="not comparable")
        ],
        reward_is_semantic=True,
        reward_semantics="1.0 = good",
    )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("".join(json.dumps({"task": i}) + "\n" for i in range(4)))
    calibration = tmp_path / "calib"
    calibration.mkdir(exist_ok=True)
    (calibration / "summary.json").write_text("{}")
    (calibration / "upstream-vs-gym.jsonl").write_text("{}\n")
    return {"rollouts": rollouts, "calibration": calibration}


def _build(tmp_path: Path, name: str = "pkg") -> Path:
    inputs = _inputs(tmp_path)
    return package.build_package(
        _run(),
        output_dir=tmp_path / name,
        package_name="demo-model-run-1",
        generated_at="2026-09-14T00:00:00+00:00",
        rollouts=inputs["rollouts"],
        failures=None,
        materialized_inputs=None,
        aggregate_metrics=None,
        calibration_dir=inputs["calibration"],
    )


def test_package_is_deterministic_and_validates(tmp_path):
    first = _build(tmp_path, "one")
    second = _build(tmp_path, "two")
    for path in sorted(first.rglob("*")):
        if path.is_file():
            assert path.read_bytes() == (second / path.relative_to(first)).read_bytes(), path.name
    assert package.validate_package(first) == []
    for relative in (
        "blade/metrics.json",
        "blade/d1-metrics.json",
        "blade/d2-anchor-facts.json",
        "blade/d3-shallow-baseline.md",
        "blade/BLADE-report.md",
        "calibration/summary.json",
        "raw/verifier-or-judge-receipts.jsonl",
        "manifest/run-manifest.json",
        "checksums.sha256",
    ):
        assert (first / relative).is_file(), relative
    d1 = json.loads((first / "blade/d1-metrics.json").read_text())
    assert d1["pass_at_1"] == 0.5 and d1["expected_k"] == 1 and "pass@k for k>1" in d1["not_applicable"][0]
    assert "Primary rate" in (first / "blade/BLADE-report.md").read_text()


def test_validation_detects_tampering(tmp_path):
    built = _build(tmp_path)
    (built / "blade/metrics.json").write_text("{}")
    problems = package.validate_package(built)
    assert any("checksum mismatch" in problem for problem in problems)
    (built / "extra.txt").write_text("x")
    assert any("not covered" in problem for problem in package.validate_package(built))


def test_validation_detects_denominator_drift():
    run = _run()
    run.outcomes.scored_rollouts = 5
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    inputs = _inputs(tmp)
    built = package.build_package(
        run,
        output_dir=tmp / "pkg",
        package_name="p",
        generated_at="t",
        rollouts=inputs["rollouts"],
        failures=None,
        materialized_inputs=None,
        aggregate_metrics=None,
        calibration_dir=None,
    )
    problems = package.validate_package(built)
    assert any("outcome classes" in p or "rollout summaries" in p for p in problems)


class _StubClient:
    base_url = "https://stub.invalid/v1"
    model = "stub-model"

    def __init__(self, *documents: dict[str, Any]):
        self._documents = list(documents)
        self.calls = 0

    def complete(self, messages, *, schema):
        self.calls += 1
        document = self._documents[min(self.calls - 1, len(self._documents) - 1)]
        return json.dumps(document), {"model": self.model, "finish_reason": "stop"}


def _good_document() -> dict[str, Any]:
    return {
        "purpose": "Demo Bench measures a demo property.",
        "what_the_run_says": [
            {"text": "The primary rate was 50.0% over 4 rollouts.", "evidence": ["fact-1"]},
            {"text": "Group a reached 100.0%.", "evidence": ["fact-1"]},
        ],
        "examples": [
            {"anchor_id": "example-1", "label": "Example", "description": "The model showed the example behavior."}
        ],
        "calibration_and_reference": [{"text": "Calibration agreed on 4 of 4 cases.", "evidence": ["fact-cal"]}],
        "limitations": [
            {"text": "k=1 only.", "evidence": ["fact-1"]},
            {"text": "Demo limitation.", "evidence": ["fact-cal"]},
        ],
        "interpretation": "The run shows a 50.0% rate; lower is better.",
    }


def test_model_card_accepts_grounded_document_and_renders(tmp_path):
    built = _build(tmp_path)
    document, record = model_card.generate(_run(), _StubClient(_good_document()))
    assert isinstance(document, ModelCardDocument) and record["attempts"][0]["problems"] == []
    outputs = model_card.write_report(_run(), document, built, generation_record=record)
    assert outputs["md"].read_text().startswith("SNORKEL AI x NVIDIA")
    html = outputs["html"].read_text()
    assert "Snorkel AI</strong> x NVIDIA" in html and "50.0%" in html and "example-1" in html
    assert package.validate_package(built) == []
    assert json.loads(outputs["generation"].read_text())["pdf_rendered"] in (True, False)


def test_model_card_fails_closed_on_invented_numbers_and_unknown_evidence():
    bad_number = _good_document()
    bad_number["interpretation"] = "The rate was 37.2% which is remarkable."
    with pytest.raises(ValueError, match="failed closed"):
        model_card.generate(_run(), _StubClient(bad_number, bad_number))
    bad_evidence = _good_document()
    bad_evidence["what_the_run_says"][0]["evidence"] = ["fact-does-not-exist"]
    with pytest.raises(ValueError, match="unknown evidence"):
        model_card.generate(_run(), _StubClient(bad_evidence, bad_evidence))
    advice = _good_document()
    advice["interpretation"] = "We recommend fine-tuning on these cases."
    with pytest.raises(ValueError, match="forbidden"):
        model_card.generate(_run(), _StubClient(advice, advice))
    wrong_example = _good_document()
    wrong_example["examples"][0]["anchor_id"] = "fact-1"
    with pytest.raises(ValueError, match="not an example anchor"):
        model_card.generate(_run(), _StubClient(wrong_example, wrong_example))


def test_model_card_requires_examples_when_anchors_exist():
    run = _run()
    run.anchor_facts.append(
        AnchorFact(id="example-2", category="example", fact="task-2: second example", evidence=["receipt-task-2"])
    )
    missing = _good_document()  # only one example although two anchors exist
    with pytest.raises(ValueError, match="example anchors are available"):
        model_card.generate(run, _StubClient(missing, missing))


def test_model_card_rejects_truncated_sentences():
    truncated = _good_document()
    truncated["interpretation"] = "The run shows a 50.0% rate; because the"
    with pytest.raises(ValueError, match="complete sentence"):
        model_card.generate(_run(), _StubClient(truncated, truncated))


def test_model_card_retries_once_with_feedback():
    bad = _good_document()
    bad["interpretation"] = "An invented 99.9% figure."
    client = _StubClient(bad, _good_document())
    document, record = model_card.generate(_run(), client)
    assert client.calls == 2 and len(record["attempts"]) == 2 and document.interpretation.startswith("The run shows")


def test_schema_rejects_malformed_documents():
    with pytest.raises(Exception):
        ModelCardDocument.model_validate({"purpose": "x"})
    assert "examples" in ModelCardDocument.model_json_schema()["required"]
    with pytest.raises(Exception):
        ModelCardDocument.model_validate({**_good_document(), "unexpected": 1})


def test_allowed_numbers_cover_metric_formats():
    allowed = model_card.allowed_numbers(_run())
    assert {"50.0", "50", "2", "4", "3", "12.5"} <= allowed
    assert "37.2" not in allowed
    run = _run()
    run.benchmark["dataset"]["cohort"] = "4 rows from the 320-row source"
    assert "320" in model_card.allowed_numbers(run)


def test_duplicate_evidence_ids_and_inline_citations_are_normalized():
    document = ModelCardDocument.model_validate(_good_document())
    document.what_the_run_says[0].evidence = ["fact-1", "fact-1"]
    document.what_the_run_says[0].text = "The primary rate was 50.0% over 4 rollouts. [fact-1] [fact-1]"
    assert model_card.validate_document(document, _run()) == []
    assert document.what_the_run_says[0].evidence == ["fact-1"]
    assert document.what_the_run_says[0].text == "The primary rate was 50.0% over 4 rollouts."


def test_render_only_cli_rerenders_from_existing_json(tmp_path):
    built = _build(tmp_path)
    document, record = model_card.generate(_run(), _StubClient(_good_document()))
    model_card.write_report(_run(), document, built, generation_record=record)
    (built / "prose" / "report.html").unlink()
    model_card.main(
        ["--package", str(built), "--base-url", "http://unused.invalid/v1", "--model", "x", "--render-only"]
    )
    assert (built / "prose" / "report.html").exists() and package.validate_package(built) == []


def test_render_markdown_lists_every_claim():
    document = ModelCardDocument.model_validate(_good_document())
    markdown = render.to_markdown(_run(), document)
    assert "[fact-1]" in markdown and "## Limitations and coverage" in markdown and "fine print" in markdown


def test_openai_client_merges_extra_body_and_never_leaks_the_key(monkeypatch):
    captured = {}

    class _Resp:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode()

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode())
        captured["auth"] = request.get_header("Authorization")
        return _Resp(
            {
                "id": "r",
                "model": "m",
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                "usage": {},
            }
        )

    monkeypatch.setattr(model_card.urllib.request, "urlopen", fake_urlopen)
    client = model_card.OpenAICompatibleClient(
        "https://x/v1",
        "m",
        "sk-secret",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "model": "ignored"},
    )
    content, metadata = client.complete([{"role": "user", "content": "hi"}], schema={"type": "object"})
    assert content == "{}" and captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["body"]["model"] == "m"  # explicit parameters win over extra-body collisions
    assert captured["auth"] == "Bearer sk-secret" and "sk-secret" not in json.dumps(metadata)
    assert metadata["request_parameters"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
