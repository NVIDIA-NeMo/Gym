# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for ``benchmarks/perceptionbench/prepare.py``."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.perceptionbench import prepare as pb_prepare
from benchmarks.perceptionbench.prepare import build_content_items, normalize_image_data_url


# Minimal valid 1x1 PNG and JPEG headers/payloads for offline MIME-detection tests.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
_TINY_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def _data_url(mime: str, raw: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


class TestContentItems:
    def test_interleaves_images_at_placeholders_and_preserves_bytes(self) -> None:
        img1 = _data_url("image/png", _TINY_PNG)
        img2 = _data_url("image/jpeg", _TINY_JPEG)
        items = build_content_items("Compare <|image_2|> with <|image_1|>.", [img1, img2])
        assert items == [
            {"type": "input_text", "text": "Compare "},
            {"type": "input_image", "image_url": img2, "detail": "high"},
            {"type": "input_text", "text": " with "},
            {"type": "input_image", "image_url": img1, "detail": "high"},
            {"type": "input_text", "text": "."},
        ]

    def test_leading_placeholder_drops_empty_leading_text(self) -> None:
        img = _data_url("image/png", _TINY_PNG)
        items = build_content_items("<|image_1|>What is the score?", [img])
        assert items == [
            {"type": "input_image", "image_url": img, "detail": "high"},
            {"type": "input_text", "text": "What is the score?"},
        ]

    def test_unreferenced_images_append_at_end_and_out_of_range_stays_text(self) -> None:
        img1 = _data_url("image/png", _TINY_PNG)
        img2 = _data_url("image/jpeg", _TINY_JPEG)
        items = build_content_items("See <|image_2|> and <|image_5|>", [img1, img2])
        assert items == [
            {"type": "input_text", "text": "See "},
            {"type": "input_image", "image_url": img2, "detail": "high"},
            {"type": "input_text", "text": " and <|image_5|>"},
            {"type": "input_image", "image_url": img1, "detail": "high"},
        ]

    def test_mislabeled_png_header_is_normalized_without_changing_image_bytes(self) -> None:
        mislabeled = _data_url("image/jpeg", _TINY_PNG)
        fixed = normalize_image_data_url(mislabeled)
        assert fixed.startswith("data:image/png;base64,")
        assert base64.b64decode(fixed.split(",", 1)[1]) == _TINY_PNG


@pytest.fixture
def synthetic_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    mislabeled_png = _data_url("image/jpeg", _TINY_PNG)
    jpeg = _data_url("image/jpeg", _TINY_JPEG)
    records = [
        {
            "index": 0,
            "problem": "How many squares are in <|image_1|>?",
            "image": [mislabeled_png],
            "answer": "1",
            "hint": "",
            "error_category": "visual_counting_error",
            "source_bmk": "NA",
            "source_idx": None,
        },
        {
            "index": 1,
            "problem": "<|image_1|> or <|image_2|>: which is brighter?",
            "image": [mislabeled_png, jpeg],
            "answer": "the first",
            "hint": "Image 1 is 1x1.",
            "error_category": "visual_comparison_error",
            "source_bmk": "BLINK",
            "source_idx": 7,
        },
        {
            "index": 2,
            "problem": "Read the text in <|image_1|>.",
            "image": [jpeg],
            "answer": "NA",
            "hint": "",
            "error_category": "ocr_error",
            "source_bmk": "OCRBench",
            "source_idx": 3,
        },
    ]
    release_file = tmp_path / "PerceptionBench.jsonl"
    release_file.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    monkeypatch.setattr(pb_prepare, "EXPECTED_SHA256", hashlib.sha256(release_file.read_bytes()).hexdigest())
    monkeypatch.setattr(pb_prepare, "EXPECTED_TASK_COUNT", len(records))
    return release_file


class TestPrepare:
    def test_full_prepare_builds_expected_rows_and_omits_hints_and_answers_from_prompt(
        self, synthetic_release: Path, tmp_path: Path
    ) -> None:
        out = pb_prepare.prepare(source_path=synthetic_release, output_path=tmp_path / "out.jsonl")
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert [row["task_id"] for row in rows] == [
            "perceptionbench_0",
            "perceptionbench_1",
            "perceptionbench_2",
        ]

        second = rows[1]
        assert second["verifier_metadata"] == {
            "index": 1,
            "question": "<|image_1|> or <|image_2|>: which is brighter?",
            "answer": "the first",
            "category": "visual_comparison_error",
            "source_bmk": "BLINK",
            "source_idx": 7,
        }
        [msg] = second["responses_create_params"]["input"]
        assert msg["role"] == "user"
        assert [item["type"] for item in msg["content"]] == [
            "input_image",
            "input_text",
            "input_image",
            "input_text",
        ]
        # Hint and reference answer must never leak into model-visible input.
        visible_text = " ".join(item.get("text", "") for item in msg["content"])
        assert "1x1" not in visible_text
        assert "the first" not in visible_text
        assert msg["content"][0]["image_url"].startswith("data:image/png;base64,")

        # 'NA' answers and source benchmarks remain strings.
        assert rows[2]["verifier_metadata"]["answer"] == "NA"
        assert rows[0]["verifier_metadata"]["source_bmk"] == "NA"

    def test_sha256_mismatch_is_rejected(
        self, synthetic_release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pb_prepare, "EXPECTED_SHA256", "0" * 64)
        with pytest.raises(ValueError, match="expected pinned"):
            pb_prepare.prepare(source_path=synthetic_release, output_path=tmp_path / "out.jsonl")

    def test_task_count_mismatch_is_rejected(
        self, synthetic_release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pb_prepare, "EXPECTED_TASK_COUNT", 3000)
        with pytest.raises(ValueError, match="expected 3000"):
            pb_prepare.prepare(source_path=synthetic_release, output_path=tmp_path / "out.jsonl")

    def test_limit_writes_subset_without_requiring_full_count(
        self, synthetic_release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pb_prepare, "EXPECTED_SHA256", "0" * 64)
        monkeypatch.setattr(pb_prepare, "EXPECTED_TASK_COUNT", 3000)
        out = pb_prepare.prepare(source_path=synthetic_release, output_path=tmp_path / "smoke.jsonl", limit=2)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
