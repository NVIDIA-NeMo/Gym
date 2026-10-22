# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest
from pydantic import ValidationError

from resources_servers.video_mcqa.dataset import (
    SOURCES,
    VideoSource,
    load_lvbench,
    load_videomme_v1,
    text,
    validate_subsets,
)
from resources_servers.video_mcqa.prepare import prepare_dataset
from resources_servers.video_mcqa.tests.helpers import REVISION, lv_record, media, v1_record


@pytest.mark.parametrize(
    "benchmark,factory,loader",
    [
        ("videomme_v1", v1_record, load_videomme_v1),
        ("lvbench", lv_record, load_lvbench),
    ],
)
def test_flat_annotation_identity_and_input_immutability(benchmark, factory, loader) -> None:
    records = [factory()]
    before = copy.deepcopy(records)
    normalized = loader(records, dataset_revision=REVISION, media=media())
    item = normalized[0]
    assert records == before
    assert item.question.task_id == f"{benchmark}:one"
    assert item.question.dataset_source == SOURCES[benchmark]
    assert item.question.video_id == "shapes" and item.question.dataset_revision == REVISION
    assert item.question.options == ("Circle", "Square", "Triangle", "Star")
    assert item.expected_answer == "B"
    assert item.question.question == "Which invented shape moves?"
    assert not hasattr(item.question, "expected_answer")


def test_nested_v1_and_fallback_explicit_video_id() -> None:
    flat = v1_record()
    nested = {key: flat[key] for key in ("duration", "domain", "sub_category")}
    nested.update(video_id="shapes", questions=[flat])
    assert load_videomme_v1([nested], dataset_revision=REVISION, media=media()) == load_videomme_v1(
        [flat], dataset_revision=REVISION, media=media()
    )


def test_nested_lv_and_overlapping_labels() -> None:
    flat = lv_record()
    nested = {"key": "shapes", "type": "Invented shapes", "qa": [flat], "video_info": {"caption": "HIDDEN"}}
    normalized = load_lvbench([nested], dataset_revision=REVISION, media=media())
    assert normalized == load_lvbench([flat], dataset_revision=REVISION, media=media())
    assert normalized[0].subsets["task_type"] == ("entity recognition", "reasoning")


@pytest.mark.parametrize("loader,factory", [(load_videomme_v1, v1_record), (load_lvbench, lv_record)])
def test_missing_media_and_duplicates_are_not_silently_dropped(loader, factory) -> None:
    with pytest.raises(ValueError, match="Missing explicit media"):
        loader([factory()], dataset_revision=REVISION, media={})
    for records in ([], [factory(), factory()]):
        with pytest.raises(ValueError, match="unique question IDs"):
            loader(records, dataset_revision=REVISION, media=media())


@pytest.mark.parametrize("value", [None, 1, "", "  ", "bad\x00text", "bad\rtext"])
def test_invalid_text(value) -> None:
    with pytest.raises(ValueError, match="nonempty text"):
        text(value, field="fixture")


@pytest.mark.parametrize(
    "updates",
    [
        {"question_id": 1},
        {"videoID": None},
        {"question": ""},
        {"answer": "E"},
        {"answer": 1},
        {"duration": "all"},
        {"domain": ""},
        {"task_type": "x\ny"},
        {"options": ["A. X", "C. Y", "B. Z", "D. W"]},
        {"options": ["A. X", "Y", "Z", "W"]},
        {"options": ["X", "Y", "Z"]},
        {"options": None},
        {"options": ["A. ", "B. Y", "C. Z", "D. W"]},
        {"questions": "bad"},
    ],
)
def test_invalid_v1_rows(updates) -> None:
    with pytest.raises(ValueError):
        load_videomme_v1([v1_record(**updates)], dataset_revision=REVISION, media=media())


def test_unlabeled_v1_options() -> None:
    row = v1_record(options=["Circle", "Square", "Triangle", "Star"])
    assert load_videomme_v1([row], dataset_revision=REVISION, media=media())[0].question.options[1] == "Square"


def test_source_duplicate_options_are_preserved_flagged_and_never_relabelled() -> None:
    record = v1_record(options=["A. Circle", "B. Circle", "C. Triangle", "D. Star"])
    item = load_videomme_v1([record], dataset_revision=REVISION, media=media())[0]
    assert item.question.options == ("Circle", "Circle", "Triangle", "Star")
    assert item.expected_answer == "B" and item.annotation_warnings == ("duplicate_option_text",)
    rows, _ = prepare_dataset(
        [record], benchmark="videomme_v1", dataset_revision=REVISION, media=media(), setting="single_turn"
    )
    assert rows[0]["annotation_warnings"] == ["duplicate_option_text"]
    assert "duplicate_option_text" not in json.dumps(rows[0]["responses_create_params"])


def test_duplicate_option_opt_in_does_not_change_default_v2_validation() -> None:
    from dataclasses import replace

    item = load_videomme_v1([v1_record()], dataset_revision=REVISION, media=media())[0].question
    with pytest.raises(ValueError, match="distinct"):
        replace(item, options=("X", "X", "Y", "Z"))
    with pytest.raises(ValueError, match="explicit boolean"):
        replace(item, allow_duplicate_options=1)


@pytest.mark.parametrize(
    "updates",
    [
        {"uid": True},
        {"uid": -1},
        {"uid": 1.5},
        {"type": ""},
        {"key": ""},
        {"question_type": "reasoning"},
        {"question_type": []},
        {"question_type": ["unknown"]},
        {"question_type": ["reasoning", "reasoning"]},
        {"question": "Question without choices"},
        {"question": "(A) X\n(B) Y\n(C) Z\n(D) W"},
        {"question": "Stem\n(A) X\n(C) Y\n(B) Z\n(D) W"},
        {"question": "Stem\n(A) X\n(B) Y\n(C) Z\n(D) "},
        {"qa": "wrong"},
        {"qa": [1]},
    ],
)
def test_invalid_lv_rows(updates) -> None:
    with pytest.raises(ValueError):
        load_lvbench([lv_record(**updates)], dataset_revision=REVISION, media=media())


def test_lv_integer_uids_match_official_string_lookup_without_coercing_other_fields() -> None:
    item = load_lvbench([lv_record(uid=73)], dataset_revision=REVISION, media=media())[0]
    assert item.question.task_id == "lvbench:73"
    with pytest.raises(ValueError, match="unique question IDs"):
        load_lvbench([lv_record(uid=73), lv_record(uid="73")], dataset_revision=REVISION, media=media())


@pytest.mark.parametrize("loader", [load_videomme_v1, load_lvbench])
def test_record_must_be_an_object(loader) -> None:
    for record in (1, {1: "bad"}):
        with pytest.raises(ValueError, match="must be an object"):
            loader([record], dataset_revision=REVISION, media=media())


@pytest.mark.parametrize(
    "updates",
    [
        {"sha256": "x"},
        {"source_url": ""},
        {"source": "bad\npath"},
        {"video_url": "bad\rurl"},
        {"sandbox_path": "bad\npath"},
        {"unexpected": True},
    ],
)
def test_invalid_media_bindings(updates) -> None:
    with pytest.raises(ValidationError):
        media(**updates)


def test_minimal_media_binding_and_explicit_hex_revision() -> None:
    binding = VideoSource(source_url="https://example.invalid/source", sha256="b" * 64)
    assert binding.source is None and binding.video_url is None
    with pytest.raises(ValueError, match="dataset_revision"):
        load_videomme_v1([v1_record()], dataset_revision="main", media=media())


@pytest.mark.parametrize("benchmark,factory", [("videomme_v1", v1_record), ("lvbench", lv_record)])
@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
def test_both_projections_keep_gold_evidence_and_source_metadata_private(benchmark, factory, setting) -> None:
    records = [factory()]
    records[0]["response"] = "SAVED_ANSWER_DO_NOT_FORWARD"
    rows, cohort = prepare_dataset(
        records,
        benchmark=benchmark,
        dataset_revision=REVISION,
        media=media(source="/data/original.mp4", sandbox_path="video/shapes.mp4"),
        setting=setting,
    )
    row = rows[0]
    visible = json.dumps(row["responses_create_params"])
    for hidden in ("SECRET_EVIDENCE", "SAVED_ANSWER", "expected_answer", SOURCES[benchmark], "Invented shapes"):
        assert hidden not in visible
    assert row["verifier_metadata"]["expected_answer"] == "B"
    assert cohort.task_ids == (f"{benchmark}:one",) and cohort.setting == setting
    if setting == "single_turn":
        assert row["responses_create_params"]["tools"] == []
        assert row["responses_create_params"]["tool_choice"] == "none"
        assert row["responses_create_params"]["input"][0]["content"][0]["type"] == "input_video"
        assert "video_binding" not in row
    else:
        assert row["video_binding"]["source"] == "/data/original.mp4"
        assert "video/shapes.mp4" in visible and "/data/original.mp4" not in visible
        assert "tools" not in row["responses_create_params"]


@pytest.mark.parametrize(
    "updates",
    [
        {"source": None},
        {"sandbox_path": None},
        {"sandbox_path": "/absolute.mp4"},
        {"sandbox_path": "../escape.mp4"},
        {"sandbox_path": "."},
    ],
)
def test_agentic_requires_safe_explicit_media(updates) -> None:
    values = {"source": "/data/video.mp4", "sandbox_path": "video/shapes.mp4", **updates}
    with pytest.raises(ValueError, match="safe sandbox"):
        prepare_dataset(
            [v1_record()], benchmark="videomme_v1", dataset_revision=REVISION, media=media(**values), setting="agentic"
        )


def test_single_turn_never_falls_back_to_raw_video_or_a_fixed_frame_count() -> None:
    with pytest.raises(ValueError, match="prepared VideoAnyRes"):
        prepare_dataset(
            [v1_record()],
            benchmark="videomme_v1",
            dataset_revision=REVISION,
            media=media(video_url=None, source="/data/video.mp4"),
            setting="single_turn",
        )


@pytest.mark.parametrize("benchmark,setting", [("other", "single_turn"), ("lvbench", "other")])
def test_unknown_projection_mode(benchmark, setting) -> None:
    with pytest.raises(ValueError, match="Unknown benchmark or setting"):
        prepare_dataset([], benchmark=benchmark, dataset_revision=REVISION, media={}, setting=setting)


def test_subset_validation_rejects_missing_axes_and_multivalued_non_capability() -> None:
    with pytest.raises(ValueError, match="subset axes"):
        validate_subsets("lvbench", {"task_type": ("reasoning",)})
    with pytest.raises(ValueError, match="multiple labels"):
        validate_subsets("lvbench", {"domain": ("x", "y"), "task_type": ("reasoning",)})
    with pytest.raises(ValueError, match="single-line"):
        validate_subsets("lvbench", {"domain": (" x",), "task_type": ("reasoning",)})
