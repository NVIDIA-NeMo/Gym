# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json

import pytest

from benchmarks.babyvision_opencode import prepare as babyvision
from benchmarks.babyvision_opencode_question_only import prepare as question_only


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 8
HEADER = ["index", "image", "question", "answer", "category", "l2_category", "ans_type"]


def _row(index: int, image: bytes) -> dict:
    return {
        "index": str(index),
        "image": base64.b64encode(image).decode(),
        "question": f"  How many dots are in picture {index}?  ",
        "answer": str(index),
        "category": "counting",
        "l2_category": "dots",
        "ans_type": "number",
    }


def _write_tsv(path, rows) -> None:
    lines = ["\t".join(HEADER)] + ["\t".join(row[key] for key in HEADER) for row in rows]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def tsv(tmp_path, monkeypatch):
    path = tmp_path / "BabyVision.tsv"
    _write_tsv(path, [_row(0, PNG), _row(1, JPEG)])
    monkeypatch.setenv("BABYVISION_TSV", str(path))
    monkeypatch.setattr(babyvision, "NUM_ROWS", 2)
    monkeypatch.setattr(babyvision, "KNOWN_MD5", {hashlib.md5(path.read_bytes()).hexdigest()})
    return path


def _content(row: dict) -> list[dict]:
    return row["responses_create_params"]["input"][0]["content"]


def test_convert_puts_the_image_before_the_instruction_and_question() -> None:
    row = babyvision.convert(_row(3, PNG))
    image, text = _content(row)
    assert image == {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{_row(3, PNG)['image']}",
        "detail": "auto",
    }
    assert text["text"] == f"{babyvision.INSTRUCTION}\n\n---\n\nHow many dots are in picture 3?{babyvision.SUFFIX}"
    assert row["question"] == "How many dots are in picture 3?"
    assert row["expected_answer"] == "3"


def test_convert_detects_jpeg_images() -> None:
    image, _ = _content(babyvision.convert(_row(0, JPEG)))
    assert image["image_url"].startswith("data:image/jpeg;base64,")


def test_question_only_drops_the_instruction() -> None:
    _, text = _content(babyvision.convert(_row(2, PNG), question_only=True))
    assert text["text"] == f"How many dots are in picture 2?{babyvision.SUFFIX}"


def test_prepare_writes_one_row_per_tsv_row(tsv, tmp_path) -> None:
    output = babyvision.prepare(tmp_path / "out.jsonl")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["source_id"] for row in rows] == ["0", "1"]
    assert babyvision.INSTRUCTION in _content(rows[0])[1]["text"]


def test_question_only_prepare_writes_bare_questions(tsv, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(question_only, "OUTPUT_FPATH", tmp_path / "question_only.jsonl")
    rows = [json.loads(line) for line in question_only.prepare().read_text().splitlines()]
    assert len(rows) == 2
    assert all(babyvision.INSTRUCTION not in _content(row)[1]["text"] for row in rows)


@pytest.mark.parametrize("variant", ["default", "question_only"])
def test_rows_carry_the_tsv_index_as_a_stable_task_id(tsv, tmp_path, monkeypatch, variant: str) -> None:
    """`gym eval plan` and task selection key rows by task_id; the TSV index does not change with row order."""
    from resources_servers.visual_coding_opencode.task_data import TaskData

    if variant == "question_only":
        monkeypatch.setattr(question_only, "OUTPUT_FPATH", tmp_path / "question_only.jsonl")
        output = question_only.prepare()
    else:
        output = babyvision.prepare(tmp_path / "out.jsonl")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row["task_id"], row["source_id"]) for row in rows] == [("0", "0"), ("1", "1")]
    assert [TaskData.model_validate(row).task_id for row in rows] == ["0", "1"]
    assert TaskData.model_validate({"expected_answer": "3"}).task_id is None  # rows without one stay valid


def test_prepare_rejects_an_unknown_copy(tsv, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(babyvision, "KNOWN_MD5", {"0" * 32})
    with pytest.raises(ValueError, match="md5"):
        babyvision.prepare(tmp_path / "out.jsonl")


def test_prepare_rejects_a_wrong_row_count(tsv, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(babyvision, "NUM_ROWS", 388)
    with pytest.raises(ValueError, match="rows"):
        babyvision.prepare(tmp_path / "out.jsonl")


def test_tsv_path_needs_an_environment_variable(monkeypatch) -> None:
    monkeypatch.delenv("BABYVISION_TSV", raising=False)
    monkeypatch.delenv("LMUData", raising=False)
    with pytest.raises(FileNotFoundError):
        babyvision.tsv_path()
