# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from datasets import Dataset

from benchmarks.indic_arena_hard import prepare as module


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list[dict]]:
    rows = {
        language: [
            {
                "row_id": "shared-id",
                "row_index": 0,
                "prompt": prompt,
                "language": name,
                "language_code": language,
                "judge_pass_stage": "failed_after_correction_review_needed",
                "human_evaluation_pending": True,
                "human_review_reason": "failed_final_judge",
            }
        ]
        for language, name, prompt in [("en", "English", "Explain gravity."), ("hi", "Hindi", "गुरुत्वाकर्षण समझाइए।")]
    }
    monkeypatch.setattr(module, "OUTPUT_FPATH", tmp_path / "data" / "prepared.jsonl")
    monkeypatch.setattr(module.datasets, "get_dataset_config_names", Mock(return_value=list(rows)))
    monkeypatch.setattr(
        module.datasets, "load_dataset", Mock(side_effect=lambda _, name, **kw: Dataset.from_list(rows[name]))
    )
    return rows


def _baseline_file(tmp_path: Path, rows: list[dict]) -> str:
    path = tmp_path / "baselines.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def test_prepare_retains_translations_and_review_metadata(source: dict) -> None:
    with pytest.warns(UserWarning, match="No baseline file supplied"):
        output = module.prepare()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["uid"] for row in rows] == ["en:shared-id", "hi:shared-id"]
    for row in rows:
        original = source[row["language"]][0]
        assert row["question"] == original["prompt"]
        assert row["language_name"] == original["language"]
        assert row["subset_for_metrics"] == row["language_code"] == row["language"]
        assert row["category"] == "hard_prompt"
        assert row["baseline_answer"] == ""
        for key in ["row_id", "row_index", "judge_pass_stage", "human_evaluation_pending", "human_review_reason"]:
            assert row[key] == original[key]
    assert "गुरुत्वाकर्षण" in output.read_text(encoding="utf-8")
    module.datasets.get_dataset_config_names.assert_called_once_with(module.HF_DATASET, revision=module.HF_REVISION)
    for language in source:
        module.datasets.load_dataset.assert_any_call(
            module.HF_DATASET, name=language, split="test", revision=module.HF_REVISION
        )


def test_baselines_join_by_language_and_id(source: dict, tmp_path: Path) -> None:
    path = _baseline_file(
        tmp_path,
        [
            {"language": "hi", "row_id": "shared-id", "generation": "हिंदी उत्तर"},
            {"language": "en", "row_id": "shared-id", "generation": "English answer"},
        ],
    )
    rows = [json.loads(line) for line in module.prepare(baseline_file=path).read_text().splitlines()]
    assert [(row["language"], row["baseline_answer"]) for row in rows] == [
        ("en", "English answer"),
        ("hi", "हिंदी उत्तर"),
    ]


def test_missing_translation_baseline_does_not_replace_output(source: dict, tmp_path: Path) -> None:
    path = _baseline_file(tmp_path, [{"language": "en", "row_id": "shared-id", "generation": "English answer"}])
    module.OUTPUT_FPATH.parent.mkdir()
    module.OUTPUT_FPATH.write_text("existing prepared data\n")
    with pytest.raises(ValueError, match="Missing baseline.*hi"):
        module.prepare(baseline_file=path)
    assert module.OUTPUT_FPATH.read_text() == "existing prepared data\n"


@pytest.mark.parametrize("answer", ["", "   ", None, 42])
def test_rejects_empty_or_nontext_baseline(source: dict, tmp_path: Path, answer: object) -> None:
    path = _baseline_file(tmp_path, [{"language": "hi", "row_id": "shared-id", "generation": answer}])
    with pytest.raises(ValueError, match="Empty baseline"):
        module.prepare(languages=["hi"], baseline_file=path)


def test_rejects_duplicate_baselines(source: dict, tmp_path: Path) -> None:
    row = {"language": "hi", "row_id": "shared-id", "generation": "उत्तर"}
    path = _baseline_file(tmp_path, [row, row])
    with pytest.raises(ValueError, match="Duplicate baseline"):
        module.prepare(languages=["hi"], baseline_file=path)


def test_language_selection(source: dict) -> None:
    with pytest.warns(UserWarning):
        output = module.prepare(languages=["hi", "hi"])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["language"] == "hi"
    assert module.datasets.load_dataset.call_count == 1


@pytest.mark.parametrize("languages, message", [([], "at least one"), (["xx"], "Unsupported languages")])
def test_invalid_language_selection(source: dict, languages: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        module.prepare(languages=languages)
    module.datasets.load_dataset.assert_not_called()


@pytest.mark.parametrize("problem", ["empty_split", "duplicate_id", "language_mismatch", "empty_prompt"])
def test_rejects_invalid_source_rows(source: dict, problem: str) -> None:
    if problem == "empty_split":
        source["hi"].clear()
        message = "No test rows"
    elif problem == "duplicate_id":
        source["hi"].append(dict(source["hi"][0]))
        message = "Duplicate question"
    elif problem == "language_mismatch":
        source["hi"][0]["language_code"] = "en"
        message = "Language mismatch"
    else:
        source["hi"][0]["prompt"] = " "
        message = "Empty prompt"
    with pytest.warns(UserWarning), pytest.raises(ValueError, match=message):
        module.prepare(languages=["hi"])
    assert not module.OUTPUT_FPATH.exists()


@pytest.mark.parametrize("languages", [None, ["hi"]])
def test_local_parquet_preparation_without_hub(source: dict, tmp_path: Path, languages: list[str] | None) -> None:
    local_dir = tmp_path / "download"
    for language, rows in source.items():
        path = local_dir / "data" / language / "test.parquet"
        path.parent.mkdir(parents=True)
        Dataset.from_list(rows).to_parquet(path)
    module.datasets.get_dataset_config_names.side_effect = AssertionError("Local preparation must not query the Hub")
    module.datasets.load_dataset.side_effect = AssertionError("Local preparation must not load from the Hub")
    baseline_file = _baseline_file(
        tmp_path,
        [{"language": language, "row_id": "shared-id", "generation": f"{language} baseline"} for language in source],
    )
    output = module.prepare(dataset_dir=str(local_dir), languages=languages, baseline_file=baseline_file)
    with output.open(encoding="utf-8") as fin:
        rows = [json.loads(line) for line in fin]
    assert {row["language"] for row in rows} == set(languages or source)
    for row in rows:
        assert row["question"] == source[row["language"]][0]["prompt"]
        assert row["baseline_answer"] == f"{row['language']} baseline"
        assert row["human_evaluation_pending"] is True
    module.datasets.get_dataset_config_names.assert_not_called()
    module.datasets.load_dataset.assert_not_called()


def test_local_directory_without_parquet_fails_without_hub(source: dict, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No data/<language>/test.parquet"):
        module.prepare(dataset_dir=str(tmp_path))
    module.datasets.get_dataset_config_names.assert_not_called()
    module.datasets.load_dataset.assert_not_called()


def test_local_missing_language_fails_without_hub(source: dict, tmp_path: Path) -> None:
    path = tmp_path / "data" / "en" / "test.parquet"
    path.parent.mkdir(parents=True)
    Dataset.from_list(source["en"]).to_parquet(path)
    with pytest.raises(ValueError, match="Unsupported languages.*hi"):
        module.prepare(dataset_dir=str(tmp_path), languages=["hi"])
    module.datasets.get_dataset_config_names.assert_not_called()
    module.datasets.load_dataset.assert_not_called()
