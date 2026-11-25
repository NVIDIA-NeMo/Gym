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

    def test_include_hints_appends_non_empty_hints_to_prompt_only(
        self, synthetic_release: Path, tmp_path: Path
    ) -> None:
        out = pb_prepare.prepare(
            source_path=synthetic_release,
            output_path=tmp_path / "hints.jsonl",
            include_hints=True,
        )
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        first_text = "".join(
            item.get("text", "") for item in rows[0]["responses_create_params"]["input"][0]["content"]
        )
        second_text = "".join(
            item.get("text", "") for item in rows[1]["responses_create_params"]["input"][0]["content"]
        )
        assert first_text == "How many squares are in ?"
        assert second_text.endswith("\n\nImage 1 is 1x1.")
        # Judge question stays clean without appended hint.
        assert "1x1" not in rows[1]["verifier_metadata"]["question"]

    def test_include_hints_and_normalize_coords_1000_are_mutually_exclusive(
        self, synthetic_release: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="cannot be used together"):
            pb_prepare.prepare(
                source_path=synthetic_release,
                output_path=tmp_path / "invalid.jsonl",
                include_hints=True,
                normalize_coords_1000=True,
            )
        with pytest.raises(ValueError, match="cannot be used together"):
            pb_prepare.convert_record(
                {
                    "index": 0,
                    "problem": "<|image_1|>Q?",
                    "image": [_data_url("image/png", _TINY_PNG)],
                    "answer": "A",
                    "error_category": "ocr_error",
                },
                include_hints=True,
                normalize_coords_1000=True,
            )


def _png_with_size(width: int, height: int) -> str:
    # Header-only PNG with IHDR width/height bytes sufficient for dimension parsing.
    header = (
        b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big")
    )
    return _data_url("image/png", header)


def _jpeg_with_size(width: int, height: int, *, sof_marker: int = 0xC0) -> str:
    # SOI + APP0 (16 bytes) + SOF0/SOF2 segment carrying height/width + EOI.
    app0 = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof_body = b"\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03\x01\x11\x00"
    sof = bytes([0xFF, sof_marker]) + (len(sof_body) + 2).to_bytes(2, "big") + sof_body
    return _data_url("image/jpeg", b"\xff\xd8" + app0 + sof + b"\xff\xd9")


def _gif_with_size(width: int, height: int, *, version: bytes = b"GIF89a") -> str:
    return _data_url(
        "image/gif", version + width.to_bytes(2, "little") + height.to_bytes(2, "little") + b"\x00\x00\x00"
    )


def _webp_vp8_with_size(width: int, height: int) -> str:
    # Lossy VP8 bitstream: 12-byte RIFF header + 'VP8 ' chunk header + 6-byte frame header + 16-bit W/H.
    chunk_data = b"\x00\x00\x00\x9d\x01\x2a" + width.to_bytes(2, "little") + height.to_bytes(2, "little")
    riff = b"RIFF" + (4 + 8 + len(chunk_data)).to_bytes(4, "little") + b"WEBP"
    return _data_url("image/webp", riff + b"VP8 " + len(chunk_data).to_bytes(4, "little") + chunk_data)


def _webp_vp8l_with_size(width: int, height: int) -> str:
    # Lossless VP8L bitstream: signature byte 0x2f + packed 14-bit (width-1) and 14-bit (height-1).
    packed = ((width - 1) & 0x3FFF) | (((height - 1) & 0x3FFF) << 14)
    chunk_data = b"\x2f" + packed.to_bytes(4, "little") + b"\x00\x00\x00\x00\x00"
    riff = b"RIFF" + (4 + 8 + len(chunk_data)).to_bytes(4, "little") + b"WEBP"
    return _data_url("image/webp", riff + b"VP8L" + len(chunk_data).to_bytes(4, "little") + chunk_data)


def _webp_vp8x_with_size(width: int, height: int) -> str:
    # Extended VP8X header: 4-byte flags + 24-bit (width-1) + 24-bit (height-1).
    chunk_data = b"\x00\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    riff = b"RIFF" + (4 + 8 + len(chunk_data)).to_bytes(4, "little") + b"WEBP"
    return _data_url("image/webp", riff + b"VP8X" + len(chunk_data).to_bytes(4, "little") + chunk_data)


class TestImageDimensions:
    @pytest.mark.parametrize(
        ("image_url", "expected"),
        [
            (_png_with_size(2048, 1536), (2048, 1536)),
            (_jpeg_with_size(1600, 1067, sof_marker=0xC0), (1600, 1067)),
            (_jpeg_with_size(1365, 2047, sof_marker=0xC2), (1365, 2047)),
            (_gif_with_size(640, 480, version=b"GIF87a"), (640, 480)),
            (_gif_with_size(320, 240, version=b"GIF89a"), (320, 240)),
            (_webp_vp8_with_size(1280, 720), (1280, 720)),
            (_webp_vp8l_with_size(1920, 1080), (1920, 1080)),
            (_webp_vp8x_with_size(4096, 2160), (4096, 2160)),
        ],
        ids=["png", "jpeg_baseline", "jpeg_progressive", "gif87a", "gif89a", "webp_vp8", "webp_vp8l", "webp_vp8x"],
    )
    def test_parses_dimensions_across_all_supported_image_headers(
        self, image_url: str, expected: tuple[int, int]
    ) -> None:
        assert pb_prepare._image_dimensions(image_url) == expected

    def test_rejects_unsupported_or_truncated_image_bytes(self) -> None:
        with pytest.raises(ValueError, match="Unable to read image dimensions"):
            pb_prepare._image_dimensions(_data_url("image/bmp", b"BM\x00\x00\x00\x00"))
        with pytest.raises(ValueError, match="Unable to read image dimensions"):
            pb_prepare._image_dimensions(_data_url("image/jpeg", b"\xff\xd8\xff\xd9"))


class TestNormalizeCoords1000:
    def test_converts_row_col_and_unlabeled_pixel_coords_to_1000(self) -> None:
        img = _png_with_size(2000, 1000)
        row_col = pb_prepare.convert_record(
            {
                "index": 10,
                "problem": (
                    "<|image_1|>The origin of the pixel coordinate system is located at the top-left corner of the image, "
                    "with row increasing vertically downward and col increasing horizontally to the right. "
                    "Where is pixel position (row=250, col=1500)?\nA. Sky\nB. Road"
                ),
                "image": [img],
                "answer": "B",
                "hint": "The image dimensions are width=2000 px and height=1000 px.",
                "error_category": "visual_localization_error",
                "source_bmk": "DA-2K",
            },
            normalize_coords_1000=True,
        )
        q = row_col["verifier_metadata"]["question"]
        assert "[750, 250]" in q
        assert "row=" not in q
        assert "pixel" not in q.lower()
        assert "[0, 1000]" in q

        bracket_xy = pb_prepare.convert_record(
            {
                "index": 11,
                "problem": "<|image_1|>Given that the coordinates use the pixel coordinate system, what is at pixel coordinates [1000, 500]?",
                "image": [img],
                "answer": "A",
                "hint": "The image dimensions are width=2000 px and height=1000 px.",
                "error_category": "visual_localization_error",
                "source_bmk": "DA-2K",
            },
            normalize_coords_1000=True,
        )
        assert "[500, 500]" in bracket_xy["verifier_metadata"]["question"]
        assert "pixel" not in bracket_xy["verifier_metadata"]["question"].lower()

        paren_rc = pb_prepare.convert_record(
            {
                "index": 12,
                "problem": "<|image_1|>Which object does the pixel coordinate (250, 1000) fall on?",
                "image": [img],
                "answer": "A",
                "hint": "",
                "error_category": "visual_localization_error",
                "source_bmk": "DA-2K",
            },
            normalize_coords_1000=True,
        )
        assert "[500, 250]" in paren_rc["verifier_metadata"]["question"]

    def test_converts_0_1_points_bboxes_and_scalar_options(self) -> None:
        img = _data_url("image/png", _TINY_PNG)
        bbox_row = pb_prepare.convert_record(
            {
                "index": 20,
                "problem": (
                    "<|image_1|>Use [x, y] in normalized coordinates (0-1 range). Read [0.678, 0.119, 1.000, 0.195]."
                ),
                "image": [img],
                "answer": "hi",
                "hint": "Coordinates are given in [x1, y1, x2, y2] format.",
                "error_category": "ocr_error",
            },
            normalize_coords_1000=True,
        )
        assert "(0-1000 range)" in bbox_row["verifier_metadata"]["question"]
        assert "[678, 119, 1000, 195]" in bbox_row["verifier_metadata"]["question"]

        paren_row = pb_prepare.convert_record(
            {
                "index": 21,
                "problem": "<|image_1|>Range over [0,1], coordinate (0.135,0.324) is on?",
                "image": [img],
                "answer": "A",
                "hint": "Coordinates are given in [x, y] format.",
                "error_category": "ocr_error",
            },
            normalize_coords_1000=True,
        )
        assert "[0, 1000]" in paren_row["verifier_metadata"]["question"]
        assert "[135, 324]" in paren_row["verifier_metadata"]["question"]

        scalar_row = pb_prepare.convert_record(
            {
                "index": 22,
                "problem": (
                    "<|image_1|>Using normalized coordinates (0-1 range), find x:\nA.0.801\nB.0.901\nC.0.701\nD.0.601"
                ),
                "image": [img],
                "answer": "B",
                "hint": "",
                "error_category": "visual_localization_error",
            },
            normalize_coords_1000=True,
        )
        sq = scalar_row["verifier_metadata"]["question"]
        assert "(0-1000 range)" in sq
        assert "A.801\nB.901\nC.701\nD.601" in sq

    def test_leaves_already_1000_and_cartesian_math_untouched(self) -> None:
        img = _data_url("image/png", _TINY_PNG)
        already_1000 = "<|image_1|>Count lines in [171, 239, 799, 293] (ranging from 0 to 1000)."
        row_1000 = pb_prepare.convert_record(
            {
                "index": 30,
                "problem": already_1000,
                "image": [img],
                "answer": "2",
                "hint": "Coordinates are given in [x1, y1, x2, y2] format.",
                "error_category": "visual_counting_error",
            },
            normalize_coords_1000=True,
        )
        assert row_1000["verifier_metadata"]["question"] == already_1000

        cartesian = "<|image_1|>Where does the curve cross (0.5, 0.25) on the figure's axes?"
        row_math = pb_prepare.convert_record(
            {
                "index": 31,
                "problem": cartesian,
                "image": [img],
                "answer": "(2, -1)",
                "hint": "",
                "error_category": "visual_localization_error",
            },
            normalize_coords_1000=True,
        )
        assert row_math["verifier_metadata"]["question"] == cartesian

        cartesian_endpoints = (
            "<|image_1|>Where does the curve cross (0, 1) and (0, 0.5) on the figure's axes?\n"
            "A. 0\nB. 0.25\nC. 1\nD. 0.5"
        )
        row_math_endpoints = pb_prepare.convert_record(
            {
                "index": 32,
                "problem": cartesian_endpoints,
                "image": [img],
                "answer": "C",
                "hint": "",
                "error_category": "visual_localization_error",
            },
            normalize_coords_1000=True,
        )
        assert row_math_endpoints["verifier_metadata"]["question"] == cartesian_endpoints

    def test_converts_integer_endpoint_points_bboxes_and_scalar_options(self) -> None:
        img = _data_url("image/png", _TINY_PNG)
        mixed_points = pb_prepare.convert_record(
            {
                "index": 40,
                "problem": (
                    "<|image_1|>Using normalized coordinates (0-1 range), compare (0, 0.5), [0.25, 0.5], "
                    "(1, 0), and [0, 1]."
                ),
                "image": [img],
                "answer": "A",
                "hint": "",
                "error_category": "visual_localization_error",
            },
            normalize_coords_1000=True,
        )
        mpq = mixed_points["verifier_metadata"]["question"]
        assert "(0-1000 range)" in mpq
        assert "compare [0, 500], [250, 500], [1000, 0], and [0, 1000]." in mpq

        mixed_bbox = pb_prepare.convert_record(
            {
                "index": 41,
                "problem": "<|image_1|>Compare [0, 0.25, 1, 0.75] with full frame [0, 0, 1, 1].",
                "image": [img],
                "answer": "A",
                "hint": "Coordinates are given in [x1, y1, x2, y2] format.",
                "error_category": "visual_localization_error",
            },
            normalize_coords_1000=True,
        )
        mbq = mixed_bbox["verifier_metadata"]["question"]
        assert "[0, 250, 1000, 750]" in mbq
        assert "[0, 0, 1000, 1000]" in mbq
        assert "Coordinates are normalized to [0, 1000] in [x1, y1, x2, y2] format" in mbq

        corner_point_only = pb_prepare.convert_record(
            {
                "index": 42,
                "problem": "<|image_1|>What is at [0, 1]?",
                "image": [img],
                "answer": "floor",
                "hint": "Coordinates are given in [x, y] format.",
                "error_category": "visual_localization_error",
            },
            normalize_coords_1000=True,
        )
        cpq = corner_point_only["verifier_metadata"]["question"]
        assert "What is at [0, 1000]?" in cpq
        assert "Coordinates are normalized to [0, 1000] in [x, y] format" in cpq

        scalar_problem = "<|image_1|>Using normalized coordinates (0-1 range), find x:\nA. 0\nB. 0.25\nC. 1\nD. 0.5"
        scalar_row = pb_prepare.convert_record(
            {
                "index": 43,
                "problem": scalar_problem,
                "image": [img],
                "answer": "C",
                "hint": "",
                "error_category": "visual_localization_error",
            },
            normalize_coords_1000=True,
        )
        sq = scalar_row["verifier_metadata"]["question"]
        assert "(0-1000 range)" in sq
        assert "A. 0\nB. 250\nC. 1000\nD. 500" in sq

        scalar_with_interval = pb_prepare.convert_record(
            {
                "index": 44,
                "problem": (
                    "<|image_1|>Using normalized coordinates (0-1 range) in [0, 1], find x:\n"
                    "A. 0\nB. 0.25\nC. 1\nD. 0.5"
                ),
                "image": [img],
                "answer": "C",
                "hint": "",
                "error_category": "visual_localization_error",
            },
            normalize_coords_1000=True,
        )
        siq = scalar_with_interval["verifier_metadata"]["question"]
        assert "(0-1000 range) in [0, 1000]" in siq
        assert "A. 0\nB. 250\nC. 1000\nD. 500" in siq

        default_row = pb_prepare.convert_record(
            {
                "index": 45,
                "problem": scalar_problem,
                "image": [img],
                "answer": "C",
                "hint": "",
                "error_category": "visual_localization_error",
            },
            normalize_coords_1000=False,
        )
        assert default_row["verifier_metadata"]["question"] == scalar_problem

        interval_only = "<|image_1|>Give the normalized coordinates in [0, 1] of the red dot."
        interval_only_row = pb_prepare.convert_record(
            {
                "index": 46,
                "problem": interval_only,
                "image": [img],
                "answer": "[0.25, 0.5]",
                "hint": "Coordinates are given in [x, y] format.",
                "error_category": "visual_localization_error",
            },
            normalize_coords_1000=True,
        )
        assert interval_only_row["verifier_metadata"]["question"] == interval_only

    def test_does_not_partially_convert_non_endpoint_numeric_tokens(self) -> None:
        img = _data_url("image/png", _TINY_PNG)
        bad_tuples = (
            "<|image_1|>Using normalized coordinates (0-1 range), check (0, 1.5), [0, 10], [01, 0.5], "
            "[0, 0, 1.05, 1], and [0, 1)."
        )
        bad_tuple_row = pb_prepare.convert_record(
            {
                "index": 50,
                "problem": bad_tuples,
                "image": [img],
                "answer": "A",
                "hint": "",
                "error_category": "visual_localization_error",
            },
            normalize_coords_1000=True,
        )
        assert bad_tuple_row["verifier_metadata"]["question"] == bad_tuples

        for idx, options in enumerate(
            [
                "A. 0\nB. 0.25\nC. 1.5\nD. 0.5",
                "A. 0\nB. 0.25\nC. 1.05\nD. 0.5",
                "A. 0\nB. 1\nC. 10\nD. 2",
                "A. 0\nB. 0.25\nC. 01\nD. 0.5",
            ],
            start=51,
        ):
            problem = f"<|image_1|>Using normalized coordinates (0-1 range), find x:\n{options}"
            row = pb_prepare.convert_record(
                {
                    "index": idx,
                    "problem": problem,
                    "image": [img],
                    "answer": "A",
                    "hint": "",
                    "error_category": "visual_localization_error",
                },
                normalize_coords_1000=True,
            )
            assert row["verifier_metadata"]["question"] == problem
