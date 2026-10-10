# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import json
from pathlib import Path

from benchmarks.facts_multimodal.prepare import (
    ImageDownloadResult,
    _image_dimensions_error,
    clear_aggregate_metrics,
    parse_rubric_items,
    prepare,
)


def test_rejects_implausibly_small_image() -> None:
    tiny_gif = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"

    assert _image_dimensions_error(tiny_gif) == "image dimensions 1x1 are below 16x16"


def test_parse_rubric_items_preserves_facts_and_tags() -> None:
    rubric = """rubric_items {
  fact_text: "A blue car is visible."
  tags: "Sentence"
  tags: "essential"
}
rubric_items {
  fact_text: "The sky is cloudy."
  tags: "non-essential"
}"""

    assert parse_rubric_items(rubric) == [
        {"fact": "A blue car is visible.", "tags": ["Sentence", "essential"]},
        {"fact": "The sky is cloudy.", "tags": ["non-essential"]},
    ]


def test_parse_rubric_items_supports_protobuf_apostrophe_escaping() -> None:
    rubric = r"""rubric_items {
  fact_text: "There is a Rubik\'s Cube in the image."
  tags: "Sentence"
}"""

    assert parse_rubric_items(rubric) == [
        {"fact": "There is a Rubik's Cube in the image.", "tags": ["Sentence"]},
    ]


def test_clear_aggregate_metrics_removes_only_derived_sidecars(tmp_path: Path) -> None:
    output = tmp_path / "prepared.jsonl"
    metrics = tmp_path / "prepared_metrics.json"
    conflict = tmp_path / "prepared_metrics_conflict.json"
    unrelated = tmp_path / "other_metrics.json"
    for path in (metrics, conflict, unrelated):
        path.write_text("{}", encoding="utf-8")

    clear_aggregate_metrics(output)

    assert not metrics.exists()
    assert not conflict.exists()
    assert unrelated.exists()


def test_prepare_excludes_failed_image_downloads_and_writes_a_report(tmp_path: Path) -> None:
    source = tmp_path / "facts.csv"
    output = tmp_path / "prepared.jsonl"
    report = tmp_path / "image_download_report.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt",
                "item_id",
                "rubrics",
                "img_url",
                "prompt_category",
                "image_category",
                "user_intent_majority",
                "external_information_majority",
                "reasoning_requirement_majority",
            ],
        )
        writer.writeheader()
        for item_id in ("available", "missing"):
            writer.writerow(
                {
                    "prompt": "Describe the image.",
                    "item_id": item_id,
                    "rubrics": 'rubric_items { fact_text: "A fact." tags: "essential" }',
                    "img_url": f"https://example.test/{item_id}.jpg",
                    "prompt_category": "captioning",
                    "image_category": "other",
                    "user_intent_majority": "no_majority",
                    "external_information_majority": "no_majority",
                    "reasoning_requirement_majority": "no_majority",
                }
            )

    def fake_download(_url: str, item_id: str, image_dir: Path) -> ImageDownloadResult:
        if item_id == "available":
            image_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / "available.jpg").write_bytes(b"\xff\xd8\xff")
            return ImageDownloadResult(image_path="available.jpg", mime_type="image/jpeg")
        return ImageDownloadResult(error="HTTPError: 404")

    prepare(
        source_fpath=source,
        output_fpath=output,
        example_output_fpath=tmp_path / "example.jsonl",
        image_dir=tmp_path / "images",
        download_report_fpath=report,
        download_image_fn=fake_download,
    )

    prepared_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in prepared_rows] == ["facts_multimodal_available"]
    assert prepared_rows[0]["image_data_url"] == "data:image/jpeg;base64,/9j/"
    policy_image_url = prepared_rows[0]["responses_create_params"]["input"][0]["content"][0]["image_url"]
    assert policy_image_url == prepared_rows[0]["image_data_url"]
    report_rows = list(csv.DictReader(report.open(encoding="utf-8", newline="")))
    assert [(row["item_id"], row["status"]) for row in report_rows] == [
        ("available", "ok"),
        ("missing", "failed"),
    ]


def test_prepare_can_emit_remote_image_urls(tmp_path: Path) -> None:
    source = tmp_path / "facts.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt",
                "item_id",
                "rubrics",
                "img_url",
                "prompt_category",
                "image_category",
                "user_intent_majority",
                "external_information_majority",
                "reasoning_requirement_majority",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "prompt": "Describe it.",
                "item_id": "available",
                "rubrics": 'rubric_items { fact_text: "A fact." tags: "essential" }',
                "img_url": "https://example.test/available.jpg",
                "prompt_category": "captioning",
                "image_category": "other",
                "user_intent_majority": "no_majority",
                "external_information_majority": "no_majority",
                "reasoning_requirement_majority": "no_majority",
            }
        )

    def fake_download(_url: str, item_id: str, image_dir: Path) -> ImageDownloadResult:
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{item_id}.jpg").write_bytes(b"\xff\xd8\xff")
        return ImageDownloadResult(image_path=f"{item_id}.jpg", mime_type="image/jpeg")

    output = tmp_path / "prepared.jsonl"
    prepare(
        source_fpath=source,
        output_fpath=output,
        example_output_fpath=tmp_path / "example.jsonl",
        image_dir=tmp_path / "images",
        download_report_fpath=tmp_path / "report.csv",
        use_base64_images=False,
        download_image_fn=fake_download,
    )

    prepared_row = json.loads(output.read_text(encoding="utf-8"))
    assert prepared_row["image_data_url"] == ""
    assert prepared_row["responses_create_params"]["input"][0]["content"][0]["image_url"] == (
        "https://example.test/available.jpg"
    )
