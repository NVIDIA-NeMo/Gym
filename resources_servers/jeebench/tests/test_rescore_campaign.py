# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil

import pytest

from resources_servers.jeebench.audit_rescore import audit
from resources_servers.jeebench.rescore_campaign import digest, rescore
from resources_servers.jeebench.tests.test_app import body


@pytest.fixture
def completed_campaign(tmp_path):
    campaign = tmp_path / "campaign"
    data = campaign / "data"
    data.mkdir(parents=True)
    identities = ["correct", "wrong", "reasoning_only"]
    (data / "hi.jsonl").write_text("".join(json.dumps({"uuid": task}) + "\n" for task in identities))
    (data / "bn.jsonl").write_text(json.dumps({"uuid": "missing"}) + "\n")
    shard = campaign / "model" / "hi"
    shard.mkdir(parents=True)
    rows = []
    for task, gold, text in zip(
        identities,
        ["A", "B", "A"],
        ["The correct option is (A).", "The correct option is (A).", "<think>Answer: A</think>"],
    ):
        row = body(text, gold=gold).model_dump(mode="json")
        row["verifier_metadata"]["task_id"] = task
        row.update(
            extracted_answer="None",
            extraction_method="unparsed",
            extraction_failed=True,
            extraction_source="automatic",
            reward=0,
            exact_match=False,
            candidate_truncated=False,
        )
        rows.append(row)
    original = "".join(json.dumps(row) + "\n" for row in rows)
    (shard / "rollouts.jsonl").write_text(original)
    (shard / "validation.json").write_text(
        json.dumps(
            {"responses": 3, "score_sum": 0, "post_thinking_scoring_audit": "passed", "sampling_audit": "passed"}
        )
    )
    (shard / "DONE").touch()
    # Smoke and unfinished shards are deliberately excluded, even if files exist.
    smoke = campaign / "model" / "smoke"
    smoke.mkdir()
    (smoke / "DONE").touch()
    return campaign, rows, original


def test_rescore_uses_all_completed_rows_without_changing_sources(tmp_path, completed_campaign):
    campaign, rows, original = completed_campaign
    shard = campaign / "model/hi"
    output = tmp_path / "corrected"
    report = rescore(campaign, output)
    corrected = [json.loads(line) for line in (output / "model/hi/rollouts.jsonl").read_text().splitlines()]
    assert [r["extracted_answer"] for r in corrected] == ["A", "A", "None"]
    assert [r["reward"] for r in corrected] == [1, 0, 0]
    assert [r["response"] for r in corrected] == [r["response"] for r in rows]
    assert (shard / "rollouts.jsonl").read_text() == original
    assert report["models"]["model"]["n"] == 3
    assert report["models"]["model"]["score"] == pytest.approx(100 / 3)
    assert report["models"]["model"]["remaining_review"] == 1
    assert report["models"]["model"]["missing_languages"] == ["bn"]
    assert not report["models"]["model"]["complete"]
    assert (output / "COMPLETE").exists()
    assert len(report["sources"]) == 1
    assert len(report["sources"][0]["source_sha256"]) == 64
    with pytest.raises(FileExistsError):
        rescore(campaign, output)
    result = audit(output)
    assert result["status"] == "passed"
    assert result["rows_checked"] == 3
    assert result["shards_checked"] == result["models_checked"] == 1
    for name in ("changes.jsonl", "review_required.jsonl"):
        for entry in (json.loads(line) for line in (output / "model/hi" / name).read_text().splitlines()):
            assert not {"gold", "reward", "score"} & entry.keys()


@pytest.mark.parametrize("corruption", ["response", "score", "aggregate", "hash"])
def test_audit_rejects_changed_responses_scores_and_aggregates(tmp_path, completed_campaign, corruption):
    campaign, _, _ = completed_campaign
    output = tmp_path / "corrected"
    report = rescore(campaign, output)
    corrected_path = output / "model/hi/rollouts.jsonl"
    rows = [json.loads(line) for line in corrected_path.read_text().splitlines()]
    if corruption == "response":
        rows[0]["response"]["model"] = "different-model"
    elif corruption == "score":
        rows[0]["reward"] = 0.5
    elif corruption == "aggregate":
        report["models"]["model"]["score"] = 100
    corrected_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if corruption == "hash":
        corrected_path.write_text(corrected_path.read_text() + "\n")
    # Recomputed digests cannot hide semantic corruption from the independent audit.
    if corruption != "hash":
        report["sources"][0]["corrected_sha256"] = digest(corrected_path)
    (output / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        audit(output)
    assert not (output / "integrity_audit.json").exists()


@pytest.mark.parametrize("failure", ["duplicate", "missing", "masked", "reviewed", "unaudited"])
def test_rescore_rejects_ineligible_shards(tmp_path, completed_campaign, failure):
    campaign, rows, _ = completed_campaign
    shard = campaign / "model/hi"
    if failure == "duplicate":
        rows[1] = rows[0]
    elif failure == "missing":
        rows.pop()
    elif failure == "masked":
        rows[0]["mask_sample"] = True
    elif failure == "reviewed":
        rows[0]["extraction_source"] = "human_review"
    else:
        (shard / "validation.json").write_text("{}")
    (shard / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        rescore(campaign, output)
    assert not (output / "COMPLETE").exists()


def test_monotonic_rescore_requires_audited_baseline(tmp_path, completed_campaign, monkeypatch):
    campaign, _, _ = completed_campaign
    baseline = tmp_path / "baseline"
    rescore(campaign, baseline)
    with pytest.raises(ValueError, match="no passing integrity audit"):
        rescore(campaign, tmp_path / "rejected", baseline_corrected=baseline)
    audit(baseline)
    calls = []

    def extractor(final, question_type, *, question):
        calls.append(final)
        return "None", "unparsed"

    monkeypatch.setattr("resources_servers.jeebench.rescore_campaign.extract_answer", extractor)
    output = tmp_path / "next"
    rescore(campaign, output, baseline_corrected=baseline)
    # Previously extracted answers survive even when a new parser no longer understands them.
    rows = [json.loads(line) for line in (output / "model/hi/rollouts.jsonl").read_text().splitlines()]
    assert [row["extracted_answer"] for row in rows] == ["A", "A", "None"]
    assert calls == [""]  # Only the unresolved reasoning-only response is revisited.
    assert audit(output)["rows_checked"] == 3


def test_baseline_from_different_generations_is_rejected(tmp_path, completed_campaign):
    campaign, rows, _ = completed_campaign
    baseline = tmp_path / "baseline"
    rescore(campaign, baseline)
    audit(baseline)
    current = tmp_path / "current"
    shutil.copytree(campaign, current)
    rows[0]["response"]["output"][0]["content"][0]["text"] = "Final answer: B"
    (current / "model/hi/rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # The old campaign remains valid, but cannot attest a new generation of the same task.
    assert audit(baseline)["status"] == "passed"
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Baseline source does not match"):
        rescore(current, output, baseline_corrected=baseline)
    assert not output.exists()


@pytest.mark.parametrize("artifact", ["source", "corrected", "validation", "code", "aggregate"])
def test_baseline_is_reaudited_before_reuse(tmp_path, completed_campaign, artifact):
    campaign, _, _ = completed_campaign
    baseline = tmp_path / "baseline"
    rescore(campaign, baseline)
    audit(baseline)
    paths = {
        "source": campaign / "model/hi/rollouts.jsonl",
        "corrected": baseline / "model/hi/rollouts.jsonl",
        "validation": campaign / "model/hi/validation.json",
        "code": baseline / "code/extraction.py",
    }
    if artifact == "aggregate":
        path = baseline / "summary.json"
        report = json.loads(path.read_text())
        report["models"]["model"]["score"] = 100
        path.write_text(json.dumps(report))
    else:
        path = paths[artifact]
        path.write_text(path.read_text() + "\n")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        rescore(campaign, output, baseline_corrected=baseline)
    assert not output.exists()


def test_baseline_answers_come_from_verified_rollouts_not_changes(tmp_path, completed_campaign):
    campaign, _, _ = completed_campaign
    baseline = tmp_path / "baseline"
    rescore(campaign, baseline)
    audit(baseline)
    changes = baseline / "model/hi/changes.jsonl"
    changes.write_text(
        json.dumps({"task_id": "correct", "answer": "B", "method": "boxed", "extraction_failed": False}) + "\n"
    )
    output = tmp_path / "next"
    rescore(campaign, output, baseline_corrected=baseline)
    rows = [json.loads(line) for line in (output / "model/hi/rollouts.jsonl").read_text().splitlines()]
    assert [row["extracted_answer"] for row in rows] == ["A", "A", "None"]
    assert audit(output)["rows_checked"] == 3
