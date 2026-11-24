# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""benchmarks/videomme_v2_agentic and videomme_v1_agentic prepare.py on invented annotations (no dataset download)."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.videomme_v1_agentic import prepare as v1
from benchmarks.videomme_v2_agentic import prepare as v2
from resources_servers.videomme_v2.app import CohortManifest


def v2_record(video: str, position: int, level: int, group_type: str = "logic", **updates: str) -> dict:
    return {
        "video_id": video,
        "url": "https://example.invalid/video",
        "group_type": group_type,
        "group_structure": "[1,2,3,4]" if group_type == "logic" else "4",
        "question_id": f"{video}-{position}",
        "question": f"Which invented shape moves in clip {video}, part {position}?",
        "options": "A. Circle\nB. Square\nC. Triangle",
        "answer": "B",
        "level": str(level),
        "second_head": "Invented",
        "third_head": "Invented",
        **updates,
    }


def v1_record(video: str, position: int) -> dict:
    return {
        "video_id": "001",
        "duration": "short",
        "domain": "Knowledge",
        "sub_category": "Invented",
        "url": "https://example.invalid/video",
        "videoID": video,
        "question_id": f"001-{position}",
        "task_type": "Counting Problem",
        "question": "How many invented dots appear?",
        "options": ["A. One.", "B. Two.", "C. Three.", "D. Four."],
        "answer": "C",
    }


def stage(tmp_path, monkeypatch, module, records: list[dict], videos: list[str], env: str) -> str:
    """Write the annotation parquet and FILES.sha256 of ``videos``; point the module's outputs into tmp_path."""
    parquet = tmp_path / "annotation.parquet"
    pq.write_table(pa.Table.from_pylist(records), parquet)
    monkeypatch.setenv(env, str(parquet))
    monkeypatch.setattr(module, "PARQUET_SHA256", hashlib.sha256(parquet.read_bytes()).hexdigest())
    monkeypatch.setattr(module, "NUM_QUESTIONS", len(records))
    monkeypatch.setattr(module, "OUTPUT_FPATH", tmp_path / "out" / "rows.jsonl")
    monkeypatch.setattr(module, "COHORT_FPATH", tmp_path / "out" / "cohort.json")
    source = tmp_path / "source"
    source.mkdir()
    (source / "FILES.sha256").write_text(
        "".join(f"{hashlib.sha256(video.encode()).hexdigest()}  ./video/{video}.mp4\n" for video in videos)
    )
    return str(source)


def read_rows(module) -> list[dict]:
    return [json.loads(line) for line in module.OUTPUT_FPATH.read_text().splitlines()]


def test_videomme_v2_rows_bind_videos_and_keep_each_questions_level(tmp_path, monkeypatch) -> None:
    records = [v2_record("001", position, level) for position, level in enumerate((1, 2, 3, 3), start=1)]
    records += [v2_record("002", position, 2, group_type="relevance") for position in range(1, 5)]
    records[5]["options"] = "A. Circle\nB. Circle\nC. Square"
    source = stage(tmp_path, monkeypatch, v2, records, ["001", "002"], "VIDEOMME_V2_PARQUET")

    assert v2.prepare(source_dir=source, video_dir="/data/video") == v2.OUTPUT_FPATH
    rows = read_rows(v2)
    assert [row["task_id"] for row in rows[:4]] == [
        "videommev2-L1/001-1",
        "videommev2-L2/001-2",
        "videommev2-L3/001-3",
        "videommev2-L3/001-4",
    ]
    (part,) = rows[0]["responses_create_params"]["input"][0]["content"]
    assert part["text"] == (
        "Video: /testbed/videos/001.mp4\nWhich invented shape moves in clip 001, part 1?\nA. Circle\nB. Square\n"
        "C. Triangle\nRespond with only the letter (A, B, or C) of the correct option.\nAnswer: "
    )
    sha256 = hashlib.sha256(b"001").hexdigest()
    assert rows[0]["video_binding"] == {
        "source": "/data/video/001.mp4",
        "sandbox_path": "videos/001.mp4",
        "sha256": sha256,
    }
    metadata = rows[0]["verifier_metadata"]
    assert (metadata["expected_answer"], metadata["video_sha256"], metadata["level"], metadata["group_id"]) == (
        "B",
        sha256,
        1,
        "001",
    )
    # The gold label stays out of the model's input.
    assert "expected_answer" not in rows[0]["responses_create_params"]
    assert rows[5]["annotation_warnings"] == ["duplicate_option_text"]

    cohort = CohortManifest.model_validate_json(v2.COHORT_FPATH.read_text())
    assert (cohort.setting, cohort.answer_policy, cohort.dataset_revision) == (
        "agentic",
        "llm_extraction_v1",
        v2.DATASET_REVISION,
    )
    logic, relevance = cohort.groups
    # A group takes its fourth question's level; each question keeps its own.
    assert (logic.level, logic.group_type, logic.group_structure) == (3, "logic", "[1,2,3,4]")
    assert (relevance.level, relevance.group_type, relevance.group_structure) == (2, "relevance", None)
    assert [cohort.task_levels[task_id] for task_id in logic.task_ids] == [1, 2, 3, 3]


def test_videomme_v2_refuses_a_changed_annotation_or_a_missing_checksum(tmp_path, monkeypatch) -> None:
    records = [v2_record("001", position, 1) for position in range(1, 5)]
    source = stage(tmp_path, monkeypatch, v2, records, ["001"], "VIDEOMME_V2_PARQUET")
    monkeypatch.setattr(v2, "PARQUET_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        v2.prepare(source_dir=source)
    (tmp_path / "source" / "FILES.sha256").write_text("")
    monkeypatch.setattr(
        v2, "PARQUET_SHA256", hashlib.sha256((tmp_path / "annotation.parquet").read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="no SHA-256 for video/001.mp4"):
        v2.prepare(source_dir=source)
    with pytest.raises(ValueError, match="videomme_v2_agentic_source_dir"):
        v2.prepare()


@pytest.mark.parametrize(
    "options", ["B. Circle\nA. Square", "A. Circle\nB.Square", "A. Circle\nB. Square\nfirst continued line"]
)
def test_videomme_v2_options_must_be_labelled_lines_in_order(options) -> None:
    with pytest.raises(ValueError, match="labelled in order"):
        v2.parse_options(options)


def test_videomme_v1_rows_use_the_letter_prompt_and_a_judged_cohort(tmp_path, monkeypatch) -> None:
    records = [v1_record("clipA", 1), v1_record("clipA", 2), v1_record("clipB", 3)]
    source = stage(tmp_path, monkeypatch, v1, records, ["clipA", "clipB"], "VIDEOMME_V1_PARQUET")

    assert v1.prepare(source_dir=source) == v1.OUTPUT_FPATH
    rows = read_rows(v1)
    assert [row["task_id"] for row in rows] == ["videomme_v1:001-1", "videomme_v1:001-2", "videomme_v1:001-3"]
    (part,) = rows[2]["responses_create_params"]["input"][0]["content"]
    assert part["text"] == (
        "Video: /testbed/videos/clipB.mp4\nSelect the best answer to the following multiple-choice question based on "
        "the video. Respond with only the letter (A, B, C, or D) of the correct option.\nHow many invented dots "
        "appear?\nA. One.\nB. Two.\nC. Three.\nD. Four.\nThe best answer is:"
    )
    assert rows[2]["video_binding"] == {
        "source": f"{source}/video/clipB.mp4",
        "sandbox_path": "videos/clipB.mp4",
        "sha256": hashlib.sha256(b"clipB").hexdigest(),
    }
    cohort = json.loads(v1.COHORT_FPATH.read_text())
    assert (cohort["benchmark"], cohort["setting"], cohort["answer_policy"]) == (
        "videomme_v1",
        "agentic",
        "llm_extraction_v1",
    )
    assert cohort["task_subsets"]["videomme_v1:001-3"] == {
        "duration": ["short"],
        "domain": ["Knowledge"],
        "sub_category": ["Invented"],
        "task_type": ["Counting Problem"],
    }
    (tmp_path / "source" / "FILES.sha256").write_text(f"{'0' * 64}  video/clipA.mp4\n")
    with pytest.raises(ValueError, match="no SHA-256 for video/clipB.mp4"):
        v1.prepare(source_dir=source)
