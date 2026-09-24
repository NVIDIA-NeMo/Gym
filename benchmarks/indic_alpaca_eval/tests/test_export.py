# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from benchmarks.indic_alpaca_eval.export import export_outputs


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


@pytest.fixture
def case(tmp_path: Path) -> tuple[dict, list[dict], list[dict]]:
    tasks = [
        {"uid": f"{lang}:id", "language": lang, "instruction": text, "baseline_answer": f"reference-{lang}"}
        for lang, text in [("en", "Explain gravity."), ("hi", "गुरुत्वाकर्षण\u2028समझाइए।")]
    ]
    rows = [
        {
            "uid": t["uid"],
            "language": t["language"],
            "question": t["instruction"],
            "response": {
                "status": "completed",
                "output": [
                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "PRIVATE REASONING"}]},
                    {"type": "message", "content": [{"type": "output_text", "text": "उत्तर"}]},
                ],
            },
        }
        for t in tasks
    ]
    kwargs = {
        "dataset_file": str(tmp_path / "dataset.jsonl"),
        "rollouts_file": str(tmp_path / "rollouts.jsonl"),
        "language": "hi",
        "generator": "candidate",
        "output_file": str(tmp_path / "model_outputs.json"),
    }
    _write(Path(kwargs["dataset_file"]), tasks)
    _write(Path(kwargs["rollouts_file"]), rows)
    return kwargs, tasks, rows


def test_export_uses_selected_language_final_answer_and_matching_reference(case: tuple, tmp_path: Path) -> None:
    kwargs, tasks, _ = case
    references = tmp_path / "reference_outputs.json"
    output = export_outputs(**kwargs, reference_output_file=str(references), reference_generator="baseline")
    assert json.loads(output.read_text()) == [
        {"instruction": tasks[1]["instruction"], "output": "उत्तर", "generator": "candidate"}
    ]
    assert json.loads(references.read_text()) == [
        {"instruction": tasks[1]["instruction"], "output": "reference-hi", "generator": "baseline"}
    ]
    assert "PRIVATE REASONING" not in output.read_text()


def test_keeps_empty_and_truncated_answers(case: tuple) -> None:
    kwargs, _, rows = case
    rows[1]["response"]["output"] = rows[1]["response"]["output"][:1]
    rows[1]["response"]["status"] = "incomplete"
    _write(Path(kwargs["rollouts_file"]), rows)
    assert json.loads(export_outputs(**kwargs).read_text())[0]["output"] == ""


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unexpected", "mismatch"])
def test_bad_rollout_coverage_never_overwrites_existing_export(case: tuple, problem: str) -> None:
    kwargs, _, rows = case
    output = Path(kwargs["output_file"])
    output.write_text("existing export")
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(rows[1])
    elif problem == "unexpected":
        rows[1]["uid"] = "hi:other"
    else:
        rows[1]["question"] = "Different instruction"
    _write(Path(kwargs["rollouts_file"]), rows)
    with pytest.raises(ValueError):
        export_outputs(**kwargs)
    assert output.read_text() == "existing export"


@pytest.mark.parametrize(
    "change",
    [
        {"_ng_failure_class": "judge_failed"},
        {"failure_kind": "upstream_error"},
        {"mask_sample": True},
        {"response": None},
        {"response": {"output": [], "status": "failed"}},
        {"response": {"output": [{"type": "message", "content": [{"type": "output_text", "text": None}]}]}},
    ],
)
def test_rejects_failed_or_malformed_responses(case: tuple, change: dict) -> None:
    kwargs, _, rows = case
    rows[1].update(change)
    _write(Path(kwargs["rollouts_file"]), rows)
    with pytest.raises(ValueError):
        export_outputs(**kwargs)


@pytest.mark.parametrize("problem", ["missing_reference", "duplicate_id", "duplicate_instruction", "unknown_language"])
def test_rejects_bad_prepared_data(case: tuple, tmp_path: Path, problem: str) -> None:
    kwargs, tasks, _ = case
    if problem == "missing_reference":
        tasks[1]["baseline_answer"] = ""
        kwargs.update(reference_output_file=str(tmp_path / "refs.json"), reference_generator="ref")
    elif problem == "duplicate_id":
        tasks.append(dict(tasks[1]))
    elif problem == "duplicate_instruction":
        tasks.append({**tasks[1], "uid": "hi:other"})
    else:
        kwargs["language"] = "zz"
    _write(Path(kwargs["dataset_file"]), tasks)
    with pytest.raises(ValueError):
        export_outputs(**kwargs)
    assert not Path(kwargs["output_file"]).exists()


@pytest.mark.parametrize(
    "change",
    [
        {"generator": "  "},
        {"reference_output_file": "refs.json"},
        {"reference_generator": "ref"},
        {"reference_output_file": "refs.json", "reference_generator": ""},
    ],
)
def test_validates_generator_arguments(case: tuple, change: dict) -> None:
    kwargs, _, _ = case
    with pytest.raises(ValueError):
        export_outputs(**(kwargs | change))


def test_rejects_output_overwriting_input(case: tuple) -> None:
    kwargs, _, _ = case
    kwargs["output_file"] = kwargs["dataset_file"]
    with pytest.raises(ValueError, match="distinct"):
        export_outputs(**kwargs)
