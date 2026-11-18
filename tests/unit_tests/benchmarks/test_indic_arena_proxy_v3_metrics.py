# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.indic.arena_proxy_v3.summarize import GROUPS, INDIC_LANGUAGES, SLICES, summarize


def _task(language: str = "hi", index: int = 0) -> dict:
    return {
        "uid": f"{language}:{index}",
        "language": language,
        "group": tuple(GROUPS)[index % len(GROUPS)],
        "num_user_turns": 1 if index % 2 == 0 else 2,
        "question": f"Question {index}",
        "baseline_answer": f"Reference {index}",
        "category": "hard_prompt",
        "messages_sha256": f"conversation-digest-{index}",
        "baseline_model": "gpt-5.4-mini",
    }


def _judged(task: dict, *, verdicts: tuple[str | None, str | None] = ("A>B", "B>A")) -> dict:
    return task | {
        "verdict_gen_base": verdicts[0],
        "verdict_base_gen": verdicts[1],
        "invalid_gen_base": verdicts[0] is None,
        "invalid_base_gen": verdicts[1] is None,
        "reward": 1.0,
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _preparation_report(path: Path, rows: list[dict], *, limit: int | None = None) -> None:
    report = {
        "prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "limit_per_language": limit,
        "coverage": {
            language: {"prepared_rows": count, "source_rows": 400, "eligible_rows": 400, "excluded_untranslated": []}
            for language, count in Counter(row["language"] for row in rows).items()
        },
    }
    (path.parent / "preparation_report.json").write_text(json.dumps(report))


def test_indic_headline_excludes_english_and_preserves_all_breakdowns(tmp_path: Path) -> None:
    inputs = [_task(language, index) for language in ("en", *INDIC_LANGUAGES) for index in range(400)]
    results = [
        _judged(task, verdicts=("B>A", "A>B") if task["language"] == "en" else ("A>B", "B>A")) for task in inputs
    ]
    input_path = _write(tmp_path / "inputs.jsonl", inputs)
    _preparation_report(input_path, inputs)
    metrics = summarize(_write(tmp_path / "rollouts.jsonl", results), inputs=input_path)
    assert metrics["_avg"]["all"] == 100.0
    assert metrics["en"]["all"]["score"] == 0.0
    assert metrics["_coverage"]["headline_available"] is True
    assert metrics["_avg"]["complete"] is True
    assert metrics["hi"]["single_turn"]["n"] == 200
    assert metrics["hi"]["multi_turn"]["n"] == 200
    assert all(metrics["hi"][name]["n"] in (57, 58) for name in GROUPS.values())
    assert all(name in metrics["hi"] for name in SLICES)


def test_strong_verdicts_weight_three_and_bootstrap_is_reproducible(tmp_path: Path) -> None:
    task = _task()
    inputs = _write(tmp_path / "inputs.jsonl", [task])
    rollouts = _write(tmp_path / "rollouts.jsonl", [_judged(task, verdicts=("A>>B", "A>B"))])
    first = summarize(rollouts, inputs=inputs)
    second = summarize(rollouts, inputs=inputs)
    assert first == second
    score = first["hi"]["all"]
    assert score["score"] == pytest.approx(75.0, abs=0.01)
    assert score["ci_lower"] < 0
    assert score["ci_upper"] >= 0
    assert score["invalid_scores"] == 0
    assert first["_avg"]["all"] is None
    assert first["_avg"]["complete"] is False


@pytest.mark.parametrize("container", ["task_input", "task_data", "result"])
def test_task_and_episode_envelopes_keep_metadata(tmp_path: Path, container: str) -> None:
    task = _task()
    inputs = _write(
        tmp_path / "inputs.jsonl",
        [{"task_id": {"task_id": "0", "taskset": "test"}, "task_input": {"task_data": task}}],
    )
    row = {container: _judged(task)}
    metrics = summarize(_write(tmp_path / "rollouts.jsonl", [row]), inputs=inputs)
    assert metrics["hi"]["all"]["score"] == 100.0


@pytest.mark.parametrize(
    "failure",
    [
        {},
        {"evaluation_completed": False},
        {"mask_sample": True},
        {"verification_skipped": True},
        {"_ng_failure_class": "judge_error"},
    ],
)
def test_missing_or_failed_task_suppresses_headline(tmp_path: Path, failure: dict) -> None:
    inputs = [_task(language) for language in INDIC_LANGUAGES]
    results = [_judged(task) for task in inputs[:-1]]
    if failure:
        results.append(_judged(inputs[-1]) | failure)
    metrics = summarize(_write(tmp_path / "rollouts.jsonl", results), inputs=_write(tmp_path / "inputs.jsonl", inputs))
    assert metrics["_avg"]["all"] is None
    assert metrics["_avg"]["complete"] is False
    assert metrics["_coverage"]["complete"] is False
    assert metrics["ur"]["all"]["score"] is None
    assert metrics["ur"]["missing_or_failed_uids"] == ["ur:0"]


def test_failure_sidecar_prevents_stale_success_from_counting(tmp_path: Path) -> None:
    task = _task()
    rollouts = _write(tmp_path / "rollouts.jsonl", [_judged(task)])
    _write(tmp_path / "rollouts_failures.jsonl", [task | {"_ng_failure_class": "judge_error"}])
    metrics = summarize(rollouts, inputs=_write(tmp_path / "inputs.jsonl", [task]))
    assert metrics["hi"]["failed"] == 1
    assert metrics["hi"]["judged"] == 0
    assert metrics["_coverage"]["failure_records"] == 1


@pytest.mark.parametrize("which", ["inputs", "rollouts"])
def test_duplicate_tasks_or_repeats_are_rejected(tmp_path: Path, which: str) -> None:
    task = _task()
    inputs = _write(tmp_path / "inputs.jsonl", [task] * (2 if which == "inputs" else 1))
    rollouts = _write(tmp_path / "rollouts.jsonl", [_judged(task)] * (2 if which == "rollouts" else 1))
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(rollouts, inputs=inputs)


def test_successful_invalid_verdicts_are_counted_without_inventing_losses(tmp_path: Path) -> None:
    task = _task()
    metrics = summarize(
        _write(tmp_path / "rollouts.jsonl", [_judged(task, verdicts=(None, "B>A"))]),
        inputs=_write(tmp_path / "inputs.jsonl", [task]),
    )
    assert metrics["hi"]["all"]["invalid_scores"] == 1
    assert metrics["hi"]["all"]["score"] == 100.0
    assert metrics["hi"]["all"]["verdicts_gen_base"] == {"invalid": 1}


def test_all_invalid_scores_emit_json_null_and_no_headline(tmp_path: Path) -> None:
    inputs = [_task(language) for language in INDIC_LANGUAGES]
    results = [_judged(task, verdicts=(None, None)) for task in inputs]
    metrics = summarize(_write(tmp_path / "rollouts.jsonl", results), inputs=_write(tmp_path / "inputs.jsonl", inputs))
    assert metrics["_coverage"]["complete"] is True
    assert metrics["_coverage"]["headline_available"] is False
    assert metrics["hi"]["all"]["invalid_scores"] == 2
    assert metrics["_avg"]["all"] is None
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"group": "Math / reasoning"}, "disagrees"),
        ({"question": "A different question"}, "disagrees"),
        ({"baseline_answer": "A different reference"}, "disagrees"),
        ({"category": "creative_writing"}, "disagrees"),
        ({"messages_sha256": "different conversation digest"}, "disagrees"),
        ({"uid": "hi:unknown"}, "Unexpected"),
        ({"verdict_gen_base": "A>=B"}, "unsupported"),
        ({"invalid_gen_base": True}, "inconsistent"),
        ({"result": {"language": "en"}}, "Conflicting"),
    ],
)
def test_malformed_or_mismatched_rollouts_are_rejected(tmp_path: Path, change: dict, message: str) -> None:
    task = _task()
    inputs = _write(tmp_path / "inputs.jsonl", [task])
    rollouts = _write(tmp_path / "rollouts.jsonl", [_judged(task) | change])
    with pytest.raises(ValueError, match=message):
        summarize(rollouts, inputs=inputs)


def test_subset_is_explicit_and_does_not_claim_twelve_language_mean(tmp_path: Path) -> None:
    inputs = [_task("en"), _task("hi")]
    metrics = summarize(
        _write(tmp_path / "rollouts.jsonl", [_judged(task) for task in inputs]),
        inputs=_write(tmp_path / "inputs.jsonl", inputs),
        languages=["en"],
    )
    assert metrics["_coverage"]["complete"] is True
    assert metrics["_coverage"]["languages"] == ["en"]
    assert "hi" not in metrics
    assert metrics["_avg"]["all"] is None


def test_smoke_inputs_across_twelve_languages_have_no_full_headline(tmp_path: Path) -> None:
    inputs = [_task(language) for language in INDIC_LANGUAGES]
    input_path = _write(tmp_path / "inputs.jsonl", inputs)
    _preparation_report(input_path, inputs, limit=1)
    metrics = summarize(_write(tmp_path / "rollouts.jsonl", [_judged(task) for task in inputs]), inputs=input_path)
    assert metrics["_coverage"]["population_scope"] == "subset"
    assert metrics["_coverage"]["complete"] is True
    assert metrics["hi"]["all"]["score"] == 100.0
    assert metrics["_avg"]["complete"] is False
    assert metrics["_avg"]["all"] is None


def test_preparation_report_must_match_input_bytes(tmp_path: Path) -> None:
    inputs = [_task()]
    input_path = _write(tmp_path / "inputs.jsonl", inputs)
    _preparation_report(input_path, inputs, limit=1)
    _write(input_path, inputs + [_task(index=1)])
    with pytest.raises(ValueError, match="do not match preparation_report"):
        summarize(_write(tmp_path / "rollouts.jsonl", [_judged(inputs[0])]), inputs=input_path)


def test_unidentified_failure_cannot_be_ignored(tmp_path: Path) -> None:
    task = _task()
    rollouts = _write(tmp_path / "rollouts.jsonl", [_judged(task)])
    _write(tmp_path / "rollouts_failures.jsonl", [{"_ng_failure_class": "transport_error"}])
    with pytest.raises(ValueError, match="failure sidecar row lacks a known uid"):
        summarize(rollouts, inputs=_write(tmp_path / "inputs.jsonl", [task]))
