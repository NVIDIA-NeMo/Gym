# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.indic_alpaca_eval import prepare as module


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    rows = {
        language: [
            {
                "row_id": "helpful_base-0000",
                "row_index": 0,
                "instruction": instruction,
                "language": name,
                "language_code": language,
                "judge_pass_stage": "first_judge_pass",
                "human_evaluation_pending": True,
                "human_review_reason": "random_sample",
            }
        ]
        for language, name, instruction in [
            ("en", "English", "Explain gravity."),
            ("hi", "Hindi", "गुरुत्वाकर्षण समझाइए।"),
        ]
    }
    monkeypatch.setattr(module, "OUTPUT_FPATH", tmp_path / "data" / "prepared.jsonl")
    monkeypatch.setattr(module.datasets, "get_dataset_config_names", Mock(return_value=list(rows)))
    monkeypatch.setattr(module.datasets, "load_dataset", Mock(side_effect=lambda _, name, **kwargs: rows[name]))
    return rows


def _read(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def _references(tmp_path: Path, rows: list[dict]) -> str:
    path = tmp_path / "references.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return str(path)


def test_preserves_instructions_metadata_and_language_identity(source: dict) -> None:
    output = module.prepare()
    rows = _read(output)
    assert [r["uid"] for r in rows] == ["en:helpful_base-0000", "hi:helpful_base-0000"]
    for row in rows:
        original = source[row["language"]][0]
        assert row["instruction"] == row["question"] == original["instruction"]
        assert row["language_name"] == original["language"]
        assert row["subset_for_metrics"] == row["language_code"] == row["language"]
        assert row["baseline_answer"] == ""
        for name in ("row_id", "row_index", "judge_pass_stage", "human_evaluation_pending", "human_review_reason"):
            assert row[name] == original[name]
    module.datasets.get_dataset_config_names.assert_called_once_with(module.HF_DATASET, revision=module.HF_REVISION)
    module.datasets.load_dataset.assert_any_call(
        module.HF_DATASET, name="hi", split="eval", revision=module.HF_REVISION
    )


def test_selects_languages_once(source: dict) -> None:
    assert [r["language"] for r in _read(module.prepare(languages=["hi", "hi"]))] == ["hi"]


@pytest.mark.parametrize("languages,match", [([], "at least one"), (["zz"], "Unsupported languages")])
def test_rejects_bad_selection(source: dict, languages: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        module.prepare(languages=languages)


def test_reads_local_parquet_without_hub(source: dict, tmp_path: Path) -> None:
    root = tmp_path / "local"
    for language, rows in source.items():
        path = root / "data" / language / "eval.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
    source["hi"][0]["instruction"] += "\u2028अधिक बताएं।"
    pq.write_table(pa.Table.from_pylist(source["hi"]), root / "data/hi/eval.parquet")
    rows = _read(module.prepare(dataset_dir=str(root)))
    assert [r["language"] for r in rows] == ["en", "hi"]
    assert rows[1]["question"] == source["hi"][0]["instruction"]
    module.datasets.get_dataset_config_names.assert_not_called()
    module.datasets.load_dataset.assert_not_called()


def test_local_missing_layout(source: dict, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="eval.parquet"):
        module.prepare(dataset_dir=str(tmp_path))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("language_code", "ta", "Language mismatch"),
        ("instruction", "", "Empty instruction"),
        ("instruction", None, "Empty instruction"),
        ("row_id", "", "Empty row_id"),
        ("row_id", 1, "Empty row_id"),
    ],
)
def test_bad_source_rows_preserve_existing_data(source: dict, field: str, value: object, match: str) -> None:
    output = module.prepare()
    previous = output.read_bytes()
    source["hi"][0][field] = value
    with pytest.raises(ValueError, match=match):
        module.prepare()
    assert output.read_bytes() == previous


def test_duplicate_source_ids(source: dict) -> None:
    source["hi"].append(dict(source["hi"][0]))
    with pytest.raises(ValueError, match="Duplicate instruction ID"):
        module.prepare()


def test_empty_language_split(source: dict) -> None:
    source["hi"].clear()
    with pytest.raises(ValueError, match="No eval rows"):
        module.prepare()


def test_reference_join_is_language_specific(source: dict, tmp_path: Path) -> None:
    reference = _references(
        tmp_path,
        [
            {"language": lang, "row_id": "helpful_base-0000", "generation": answer}
            for lang, answer in [("hi", "उत्तर"), ("en", "Answer")]
        ],
    )
    rows = _read(module.prepare(reference_file=reference))
    assert [(r["language"], r["baseline_answer"]) for r in rows] == [("en", "Answer"), ("hi", "उत्तर")]


@pytest.mark.parametrize("answer", ["", "   ", None, 42])
def test_rejects_empty_reference(source: dict, tmp_path: Path, answer: object) -> None:
    reference = _references(tmp_path, [{"language": "hi", "row_id": "helpful_base-0000", "generation": answer}])
    with pytest.raises(ValueError, match="Empty reference"):
        module.prepare(languages=["hi"], reference_file=reference)


def test_duplicate_reference(source: dict, tmp_path: Path) -> None:
    row = {"language": "hi", "row_id": "helpful_base-0000", "generation": "उत्तर"}
    with pytest.raises(ValueError, match="Duplicate reference"):
        module.prepare(languages=["hi"], reference_file=_references(tmp_path, [row, row]))


def test_missing_reference_does_not_replace_output(source: dict, tmp_path: Path) -> None:
    output = module.prepare()
    previous = output.read_bytes()
    reference = _references(tmp_path, [{"language": "en", "row_id": "helpful_base-0000", "generation": "Answer"}])
    with pytest.raises(ValueError, match="Missing reference.*hi"):
        module.prepare(reference_file=reference)
    assert output.read_bytes() == previous
