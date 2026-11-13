# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The video servers' task-data schemas accept the rows their verifiers and agent accept, and no others."""

import dataclasses
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.data import TaskDataValidator, load_task_data_schema, validate_jsonl_rows
from resources_servers.video_mcqa.app import VideoMCQAVerifyRequest
from resources_servers.video_mcqa.prepare import prepare_dataset
from resources_servers.video_mcqa.task import VerifierMetadata, VideoQuestion, prepare_agentic, prepare_single_turn
from resources_servers.video_mcqa.task_data import VideoBinding, VideoMCQATaskData
from resources_servers.video_mcqa.tests.helpers import REVISION, lv_record, media, response, v1_record
from resources_servers.videomme_v2.app import VideoMMEV2VerifyRequest
from responses_api_agents.opencode_video_agent.app import OpenCodeVideoRunRequest
from responses_api_agents.opencode_video_agent.app import VideoBinding as AgentVideoBinding


SERVERS = Path(__file__).resolve().parents[2]
WIRE = {
    "videomme_v1": VideoMCQAVerifyRequest,
    "lvbench": VideoMCQAVerifyRequest,
    "videomme_v2": VideoMMEV2VerifyRequest,
}
OWNERS = list(WIRE)


def report(owner: str, rows: list[dict]):
    validator = TaskDataValidator(owner, load_task_data_schema(SERVERS / owner), "rows.jsonl")
    for index, row in enumerate(rows):
        validator.validate_row(index, row)
    return validator.report


def accepted_by_runtime(owner: str, row: dict) -> bool:
    """Whether the owner's verify request and, for an agentic row, the video agent's run request accept it."""
    try:
        WIRE[owner].model_validate({**row, "response": response()})
        if row.get("setting") == "agentic":
            OpenCodeVideoRunRequest.model_validate(row)
    except ValidationError:
        return False
    return True


def prepared_rows(tmp_path: Path, owner: str, setting: str) -> list[dict]:
    """Rows as preparation writes them: VideoMME-v2 with a level and group, the others with a duplicate option."""
    video = tmp_path / "shapes.mp4"
    video.write_bytes(b"invented bytes, not a video")
    if owner == "videomme_v2":
        question = VideoQuestion(
            task_id="shape-1",
            dataset_source="https://example.invalid/invented-shapes",
            dataset_revision=REVISION,
            video_id="shapes",
            video_source="https://example.invalid/shapes.mp4",
            video_sha256="b" * 64,
            question="Which shape moves?",
            options=("Circle", "Square", "Triangle"),
            level=2,
            group_id="group-1",
        )
        if setting == "single_turn":
            rows = [prepare_single_turn(question, expected_answer="B", video_url="https://example.invalid/prepared")]
        else:
            row = prepare_agentic(question, expected_answer="B", video_path="video/shapes.mp4")
            binding = {"source": str(video), "sandbox_path": "video/shapes.mp4", "sha256": "b" * 64}
            rows = [{**row, "video_binding": binding}]
    else:
        records = (
            [v1_record(), v1_record("two", options=["A. Circle", "B. Circle", "C. Triangle", "D. Star"])]
            if owner == "videomme_v1"
            else [lv_record(), lv_record("two", question="Which?\n(A) Circle\n(B) Circle\n(C) Triangle\n(D) Star")]
        )
        rows, _ = prepare_dataset(
            records,
            benchmark=owner,
            dataset_revision=REVISION,
            media=media(source=str(video), sandbox_path="video/shapes.mp4"),
            setting=setting,
        )
    return json.loads(json.dumps(rows))


@pytest.mark.parametrize("owner", ["video_mcqa", *OWNERS])
def test_each_server_ships_its_schema(owner):
    metadata = VerifierMetadata(
        task_id="invented",
        expected_answer="B",
        option_count=3,
        dataset_source="https://example.invalid/source",
        dataset_revision=REVISION,
        video_id="shapes",
        video_source="https://example.invalid/shapes.mp4",
        video_sha256="b" * 64,
        level=2,
    )
    flat = {**dataclasses.asdict(metadata), "setting": "single_turn"}
    schema = load_task_data_schema(SERVERS / owner)
    # Only VideoMME-v2 takes other option counts and levels; every schema takes four ungrouped choices.
    assert schema.validate_python({**flat, "option_count": 4, "level": None}).option_count == 4
    if owner == "videomme_v2":
        assert schema.validate_python(flat).level == 2
    else:
        with pytest.raises(ValidationError, match="four choices"):
            schema.validate_python(flat)


@pytest.mark.parametrize("owner", OWNERS)
def test_committed_examples_validate_clean(owner):
    path = SERVERS / owner / "data" / "example.jsonl"
    result = validate_jsonl_rows(
        owner, load_task_data_schema(SERVERS / owner), str(path), path.read_text().splitlines()
    )
    assert result.rows == 5 and result.clean, result.summary()


@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
def test_prepared_rows_validate_for_their_owner_and_the_runtime(tmp_path, owner, setting):
    rows = prepared_rows(tmp_path, owner, setting)
    result = report(owner, rows)
    assert result.rows == len(rows) and result.clean, result.summary()
    assert all(accepted_by_runtime(owner, row) for row in rows)
    if owner != "videomme_v2":
        # Repeated option text is kept as provenance, not repaired or rejected.
        assert rows[1]["annotation_warnings"] == ["duplicate_option_text"]


def test_videomme_v2_levels_and_groups_are_not_four_choice_rows(tmp_path):
    rows = prepared_rows(tmp_path, "videomme_v2", "single_turn")
    assert report("videomme_v2", rows).clean
    assert report("videomme_v1", rows).error_rows == report("lvbench", rows).error_rows == 1


def _without(key):
    return lambda row: row["verifier_metadata"].pop(key)


def _binding(**updates):
    return lambda row: row["video_binding"].update(updates)


@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize(
    "change",
    [
        _without("expected_answer"),
        _without("dataset_revision"),
        _without("video_sha256"),
        lambda row: row.pop("setting"),
        lambda row: row["verifier_metadata"].update(expected_answer="Z"),
        lambda row: row.pop("video_binding"),
        _binding(sandbox_path="/etc/video.mp4"),
        _binding(sandbox_path="video/../../outside.mp4"),
        _binding(sha256="not-a-digest"),
        _binding(url="https://example.invalid/unexpected"),
    ],
    ids=[
        "no_expected_answer",
        "no_dataset_revision",
        "no_video_sha256",
        "no_setting",
        "unavailable_answer",
        "agentic_without_binding",
        "absolute_binding_path",
        "escaping_binding_path",
        "malformed_binding_digest",
        "unknown_binding_key",
    ],
)
def test_rows_the_runtime_refuses_fail_validation(tmp_path, owner, change):
    row = prepared_rows(tmp_path, owner, "agentic")[0]
    change(row)
    assert not accepted_by_runtime(owner, row)
    assert report(owner, [row]).error_rows == 1


@pytest.mark.parametrize("owner", OWNERS)
def test_a_nested_only_field_found_top_level_is_misplaced(tmp_path, owner):
    row = prepared_rows(tmp_path, owner, "single_turn")[0]
    row["expected_answer"] = row["verifier_metadata"].pop("expected_answer")
    assert not accepted_by_runtime(owner, row)
    assert report(owner, [row]).misplaced_keys == {"expected_answer": 1}


def test_schema_fields_match_the_runtime_models():
    nested = {field.name for field in dataclasses.fields(VerifierMetadata)}
    marked = {
        name
        for name, info in VideoMCQATaskData.model_fields.items()
        if isinstance(info.json_schema_extra, dict)
        and info.json_schema_extra.get("legacy_location") == "verifier_metadata"
    }
    # task_id is read in both places, so it is the one VerifierMetadata field not marked nested-only.
    assert marked == nested - {"task_id"}
    assert set(VideoMCQATaskData.model_fields) == nested | {"setting", "video_binding", "annotation_warnings"}
    assert set(VideoBinding.model_fields) == set(AgentVideoBinding.model_fields)
    assert set(OpenCodeVideoRunRequest.model_fields) & set(VideoMCQATaskData.model_fields) == {
        "setting",
        "video_binding",
    }
    for request in WIRE.values():
        assert {"task_id", "setting", "verifier_metadata"} <= set(request.model_fields)
