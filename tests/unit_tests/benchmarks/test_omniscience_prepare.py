# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from string import Formatter
from unittest.mock import patch

import pytest
import yaml

from benchmarks.omniscience import prepare
from nemo_gym.data.task_data import find_server_dir, load_task_data_schema, validate_jsonl_rows


V1_1_REVISION = "1f9cbcec472000b28f9782205f1ef9498b45bd21"  # pragma: allowlist secret


def _v1_1_rows(num_rows: int) -> list[dict]:
    return [
        {
            "domain": "Science Engineering and Mathematics",
            "subtopic": "Biology",
            "question_id": question_id,
            "question": f"Question {question_id}?",
            "answer": f"Answer {question_id}",
        }
        for question_id in range(1, num_rows + 1)
    ]


def _prepare(tmp_path: Path, num_rows: int = 300) -> Path:
    output_fpath = tmp_path / "omniscience_benchmark.jsonl"
    with (
        patch("datasets.load_dataset", return_value=_v1_1_rows(num_rows)),
        patch.object(prepare, "get_hf_token", return_value=None),
        patch.object(prepare, "DATA_DIR", tmp_path),
        patch.object(prepare, "OUTPUT_FPATH", output_fpath),
    ):
        prepare.prepare()
    return output_fpath


def test_prepare_pins_the_v1_1_public_set_and_records_its_revision(tmp_path) -> None:
    output_fpath = tmp_path / "omniscience_benchmark.jsonl"
    with (
        patch("datasets.load_dataset", return_value=_v1_1_rows(300)) as load_dataset,
        patch.object(prepare, "get_hf_token", return_value="token"),
        patch.object(prepare, "DATA_DIR", tmp_path),
        patch.object(prepare, "OUTPUT_FPATH", output_fpath),
    ):
        assert prepare.prepare() == output_fpath

    load_dataset.assert_called_once_with(
        "ArtificialAnalysis/AA-Omniscience-Public-v1.1", split="test", revision=V1_1_REVISION, token="token"
    )
    rows = [json.loads(line) for line in output_fpath.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 300
    assert rows[0] == {
        "id": 1,
        "domain": "Science Engineering and Mathematics",
        "subtopic": "Biology",
        "question": "Question 1?",
        "expected_answer": "Answer 1",
        "aa_omniscience_version": "1.1",
        "aa_omniscience_dataset_revision": V1_1_REVISION,
    }
    assert prepare.is_prepared_data_current(output_fpath)
    # A complete file from another revision (for example the initial v1.1 upload) is prepared again.
    with patch.object(prepare, "HF_REVISION", "0" * 40):
        assert not prepare.is_prepared_data_current(output_fpath)


def test_prepare_rejects_a_dataset_with_the_wrong_row_count(tmp_path) -> None:
    sidecar_fpath = tmp_path / "omniscience_benchmark_metrics.json"
    sidecar_fpath.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="has 299 rows, expected 300"):
        _prepare(tmp_path, num_rows=299)

    assert not (tmp_path / "omniscience_benchmark.jsonl").exists()
    assert sidecar_fpath.exists()


def test_a_cached_v1_or_partial_file_is_prepared_again(tmp_path) -> None:
    v1_row = {"id": 1, "domain": "Law", "topic": "Contract law", "question": "Q?", "expected_answer": "A"}
    v1_fpath = tmp_path / "v1.jsonl"
    v1_fpath.write_text("".join(json.dumps(v1_row) + "\n" for _ in range(600)), encoding="utf-8")
    v1_1_row = {**v1_row, "aa_omniscience_dataset_revision": V1_1_REVISION}
    partial_fpath = tmp_path / "partial.jsonl"
    partial_fpath.write_text("".join(json.dumps(v1_1_row) + "\n" for _ in range(299)), encoding="utf-8")

    assert not prepare.is_prepared_data_current(v1_fpath)
    assert not prepare.is_prepared_data_current(partial_fpath)
    assert not prepare.is_prepared_data_current(tmp_path / "missing.jsonl")


def test_prepare_removes_the_metrics_sidecars_of_the_rows_it_replaces(tmp_path) -> None:
    sidecar_fpaths = [
        tmp_path / "omniscience_benchmark_metrics.json",
        tmp_path / "omniscience_benchmark_metrics_conflict.json",
    ]
    for sidecar_fpath in sidecar_fpaths:
        sidecar_fpath.write_text('{"topic": {"unique_count": 600}}', encoding="utf-8")

    _prepare(tmp_path)

    assert not any(sidecar_fpath.exists() for sidecar_fpath in sidecar_fpaths)


def test_prepared_rows_match_the_omniscience_task_data_schema(tmp_path) -> None:
    lines = _prepare(tmp_path).read_text(encoding="utf-8").splitlines()
    adapter = load_task_data_schema(find_server_dir("omniscience"))

    report = validate_jsonl_rows("omniscience", adapter, "omniscience_benchmark.jsonl", lines)

    assert report.rows == 300
    assert report.clean, report.summary()


def test_the_answer_prompt_uses_only_fields_the_prepared_rows_have() -> None:
    prompt_config = yaml.safe_load((prepare.BENCHMARK_DIR / "prompts" / "generation.yaml").read_text())
    fields = {name for template in prompt_config.values() for _, name, _, _ in Formatter().parse(template) if name}

    assert fields == {"domain", "subtopic", "question"}
