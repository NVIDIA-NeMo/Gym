# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import io
import json
import sys
import types

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from benchmarks.babyvision_opencode import prepare as babyvision
from benchmarks.babyvision_opencode_question_only import prepare as question_only


def _image(fmt: str, mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (4, 3), (10, 20, 30, 255)[: len(mode)]).save(buffer, format=fmt)
    return buffer.getvalue()


def _blank(task_id: int, image: bytes) -> dict:
    return {
        "taskId": task_id,
        "type": "Visual Tracking",
        "subtype": "Maze",
        "image": {"bytes": image, "path": f"{task_id}.png"},
        "question": f"  How many dots are in picture {task_id}?  ",
        "ansType": "blank",
        "options": [],
        "choiceAns": None,
        "blankAns": str(task_id),
    }


def _choice(task_id: int, image: bytes) -> dict:
    return {
        **_blank(task_id, image),
        "question": "Which one matches?",
        "ansType": "choice",
        "options": ["the left one", "the right one"],
        "choiceAns": 1,
        "blankAns": None,
    }


def _content(row: dict) -> list[dict]:
    return row["responses_create_params"]["input"][0]["content"]


def _decoded(image_part: dict) -> Image.Image:
    prefix = "data:image/png;base64,"
    assert image_part["image_url"].startswith(prefix)
    return Image.open(io.BytesIO(base64.b64decode(image_part["image_url"][len(prefix) :])))


@pytest.fixture
def parquet(tmp_path, monkeypatch):
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([_blank(0, _image("PNG")), _choice(1, _image("JPEG"))]), path)
    monkeypatch.setenv("BABYVISION_PARQUET", str(path))
    monkeypatch.setattr(babyvision, "NUM_ROWS", 2)
    monkeypatch.setattr(babyvision, "PARQUET_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_convert_puts_the_image_before_the_instruction_and_question() -> None:
    record = _blank(3, _image("PNG"))
    row = babyvision.convert(record)
    image, text = _content(row)
    assert image == {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{babyvision.png_base64(record['image']['bytes'])}",
        "detail": "auto",
    }
    assert text["text"] == f"{babyvision.INSTRUCTION}\n\n---\n\nHow many dots are in picture 3?{babyvision.SUFFIX}"
    assert row["question"] == "How many dots are in picture 3?"
    assert row["expected_answer"] == "3"
    assert (row["category"], row["l2_category"], row["ans_type"]) == ("Visual Tracking", "Maze", "blank")


def test_convert_lists_the_choices_and_answers_with_a_letter() -> None:
    row = babyvision.convert(_choice(5, _image("PNG")))
    assert row["question"] == "Which one matches?\nChoices:\n(A) the left one\n(B) the right one"
    assert row["expected_answer"] == "B"
    assert row["ans_type"] == "choice"


@pytest.mark.parametrize(("fmt", "mode"), [("JPEG", "RGB"), ("WEBP", "RGB"), ("PNG", "RGBA"), ("PNG", "RGB")])
def test_convert_writes_every_image_as_an_rgb_png(fmt: str, mode: str) -> None:
    image, _ = _content(babyvision.convert(_blank(0, _image(fmt, mode))))
    decoded = _decoded(image)
    assert (decoded.format, decoded.mode, decoded.size) == ("PNG", "RGB", (4, 3))


def test_question_only_drops_the_instruction() -> None:
    _, text = _content(babyvision.convert(_blank(2, _image("PNG")), question_only=True))
    assert text["text"] == f"How many dots are in picture 2?{babyvision.SUFFIX}"


def test_prepare_writes_one_row_per_release_row(parquet, tmp_path) -> None:
    output = babyvision.prepare(tmp_path / "out.jsonl")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["source_id"] for row in rows] == ["0", "1"]
    # `gym eval plan` needs a stable task id.
    assert [row["task_id"] for row in rows] == ["0", "1"]
    assert [row["expected_answer"] for row in rows] == ["0", "B"]
    assert babyvision.INSTRUCTION in _content(rows[0])[1]["text"]


def test_question_only_prepare_writes_bare_questions(parquet, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(question_only, "OUTPUT_FPATH", tmp_path / "question_only.jsonl")
    rows = [json.loads(line) for line in question_only.prepare().read_text().splitlines()]
    assert len(rows) == 2
    assert all(babyvision.INSTRUCTION not in _content(row)[1]["text"] for row in rows)


def test_prepare_rejects_a_different_file(parquet, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(babyvision, "PARQUET_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        babyvision.prepare(tmp_path / "out.jsonl")


def test_prepare_rejects_a_wrong_row_count(parquet, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(babyvision, "NUM_ROWS", 388)
    with pytest.raises(ValueError, match="rows"):
        babyvision.prepare(tmp_path / "out.jsonl")


def test_parquet_path_downloads_the_pinned_revision(tmp_path, monkeypatch) -> None:
    calls = []

    def hf_hub_download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(tmp_path / "train.parquet")

    monkeypatch.delenv("BABYVISION_PARQUET", raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download))
    assert babyvision.parquet_path() == tmp_path / "train.parquet"
    assert calls == [
        (
            "UnipatAI/BabyVision",
            "data/train-00000-of-00001.parquet",
            {"repo_type": "dataset", "revision": babyvision.REVISION},
        )
    ]
    assert babyvision.PARQUET_URL == (
        f"https://huggingface.co/datasets/UnipatAI/BabyVision/resolve/{babyvision.REVISION}/"
        "data/train-00000-of-00001.parquet"
    )
