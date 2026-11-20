# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.client import ServerClient
from nemo_gym.data import load_task_data_schema, validate_jsonl_rows
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.got10k.app import score_sequence
from resources_servers.got10k.task_data import Got10kVerifierMetadata
from resources_servers.multi_object_tracking.task_data import TaskData

from ..app import (
    VERIFIER_FIXTURE,
    MultiObjectTrackingResourcesServer,
    MultiObjectTrackingResourcesServerConfig,
    MultiObjectTrackingVerifierMetadata,
    MultiObjectTrackingVerifyRequest,
    parse_tracks,
    score_objects,
)
from .synthetic import example_rows, ground_truth_answer


# A 1000x500 image, so a grid box [x1,y1,x2,y2] is the pixel box [x1, y1/2, x2, y2/2].
# Object 2 is hidden in frame 2.
METADATA = {
    "sequence": "seq",
    "image_width": 1000,
    "image_height": 500,
    "objects": [
        {
            "id": 1,
            "targets": [
                {"frame": 1, "bbox": [100.0, 50.0, 200.0, 100.0], "visible": 1.0},
                {"frame": 2, "bbox": [400.0, 250.0, 300.0, 200.0], "visible": 1.0},
            ],
        },
        {
            "id": 2,
            "targets": [
                {"frame": 1, "bbox": [600.0, 100.0, 200.0, 150.0], "visible": 1.0},
                {"frame": 2, "bbox": [0.0, 0.0, 100.0, 50.0], "visible": 0.0},
            ],
        },
    ],
}
A1 = [100, 100, 300, 300]
A2 = [400, 500, 700, 900]
B1 = [600, 200, 800, 500]
DISJOINT = [900, 900, 950, 950]


def _answer(*entries: dict, key: str = "tracks") -> str:
    return "<answer>" + json.dumps({key: list(entries)}) + "</answer>"


def _entry(box: int, frame: int, bbox: list[int], **extra) -> dict:
    return {"box": box, "frame": frame, "bbox": bbox, **extra}


EXACT = _answer(
    _entry(1, 1, A1, visible=True),
    _entry(1, 2, A2, visible=True),
    _entry(2, 1, B1, visible=True),
    _entry(2, 2, [0, 0, 0, 0], visible=False),
)


def _metadata(**overrides) -> MultiObjectTrackingVerifierMetadata:
    return MultiObjectTrackingVerifierMetadata.model_validate({**METADATA, **overrides})


def _single_object_metadata() -> MultiObjectTrackingVerifierMetadata:
    return _metadata(objects=METADATA["objects"][:1])


def _response(*texts: str, reasoning: str | None = None) -> dict:
    output = []
    if reasoning is not None:
        output.append(
            {
                "id": "rs",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning}],
                "content": None,
            }
        )
    for index, text in enumerate(texts):
        output.append(
            {
                "id": f"msg_{index}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    return {
        "id": "resp",
        "created_at": 0,
        "model": "test",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


def _request(*texts: str, **kwargs) -> MultiObjectTrackingVerifyRequest:
    return MultiObjectTrackingVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "track"},
            "response": _response(*texts, **kwargs),
            "verifier_metadata": METADATA,
        }
    )


@pytest.fixture
def server() -> MultiObjectTrackingResourcesServer:
    config = MultiObjectTrackingResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return MultiObjectTrackingResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="unknown",
        )
    )


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        EXACT,
        f"Following both objects.\n{EXACT}\nDone.",
        # No answer tags; other list keys; other id and box keys.
        json.dumps({"tracks": [_entry(1, 1, A1), _entry(1, 2, A2), _entry(2, 1, B1)]}),
        _answer(_entry(1, 1, A1), _entry(1, 2, A2), _entry(2, 1, B1), key="boxes"),
        _answer(
            {"id": 1, "frame": 1, "bbox_2d": A1},
            {"object_id": "1", "frame": "2", "bounding_box": A2},
            {"track_id": 2, "frame": 1, "pred_bbox": B1},
        ),
        # A bare list of entries.
        "<answer>" + json.dumps([_entry(1, 1, A1), _entry(1, 2, A2), _entry(2, 1, B1)]) + "</answer>",
        # Entries in any order, and an entry for an object that was not asked about.
        _answer(_entry(2, 1, B1), _entry(7, 1, A1), _entry(1, 2, A2), _entry(1, 1, A1)),
    ],
)
def test_answer_shapes_that_recover_every_visible_box(text: str) -> None:
    result = score_objects(text, _metadata())

    assert result["reward"] == 1.0
    assert result["parse_status"] == "structured"
    assert (result["parsed_frames"], result["visible_frames"]) == (3, 3)


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("I could not follow the objects.", "parse_failed"),
        ('<answer>{"tracks":[]}</answer>', "parse_failed"),
        # Entries without a frame or without a box.
        ('<answer>{"tracks":[{"box":1,"bbox":[100,100,300,300]},{"box":1,"frame":1}]}</answer>', "parse_failed"),
    ],
)
def test_answers_without_usable_entries_score_zero(text: str, status: str) -> None:
    result = score_objects(text, _metadata())

    assert (result["reward"], result["parse_status"], result["parsed_frames"]) == (0.0, status, 0)


def test_an_entry_without_an_id_is_dropped_when_there_are_several_objects() -> None:
    text = _answer({"frame": 1, "bbox": A1}, {"frame": 2, "bbox": A2}, _entry(2, 1, B1))

    tracks, _ = parse_tracks(text, [1, 2])

    assert tracks == {2: {1: [float(value) for value in B1]}}


def test_an_entry_without_an_id_belongs_to_the_only_object() -> None:
    text = '<answer>{"boxes":[{"frame":1,"bbox":[100,100,300,300]},{"frame":2,"bbox":[400,500,700,900]}]}</answer>'

    assert score_objects(text, _single_object_metadata())["reward"] == 1.0


def test_box_holds_the_id_or_the_coordinates() -> None:
    as_id, _ = parse_tracks(_answer({"box": 2, "frame": 1, "bbox": B1}), [1, 2])
    as_coordinates, _ = parse_tracks(_answer({"id": 2, "frame": 1, "box": B1}), [1, 2])

    assert as_id == as_coordinates == {2: {1: [float(value) for value in B1]}}


def test_the_first_entry_for_an_object_and_frame_wins() -> None:
    tracks, _ = parse_tracks(_answer(_entry(1, 1, A1), _entry(1, 1, DISJOINT)), [1, 2])

    assert tracks == {1: {1: [float(value) for value in A1]}}


def test_a_hidden_claim_is_not_a_box() -> None:
    # Object 1 is visible in both frames; claiming it hidden in frame 2 scores 0 for that frame.
    text = _answer(_entry(1, 1, A1), _entry(1, 2, A2, visible=False), _entry(2, 1, B1))

    result = score_objects(text, _metadata())

    assert [score.reward for score in result["object_scores"]] == [0.5, 1.0]
    assert result["reward"] == 0.75


def test_a_box_on_a_hidden_frame_is_neither_rewarded_nor_penalized() -> None:
    text = _answer(_entry(1, 1, A1), _entry(1, 2, A2), _entry(2, 1, B1), _entry(2, 2, DISJOINT))

    assert score_objects(text, _metadata())["reward"] == 1.0


def test_an_answer_cut_off_mid_entry_keeps_the_complete_entries() -> None:
    text = EXACT[: EXACT.index('{"box": 2')] + '{"box": 2, "frame": 1, "bbox": [600, 2'

    result = score_objects(text, _metadata())

    assert result["parse_status"] in ("structured", "salvaged")
    assert [score.reward for score in result["object_scores"]] == [1.0, 0.0]
    assert result["reward"] == 0.5


def test_swapped_ids_are_scored_against_the_wrong_object() -> None:
    text = _answer(_entry(2, 1, A1), _entry(2, 2, A2), _entry(1, 1, B1))

    assert score_objects(text, _metadata())["reward"] == 0.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_the_reward_is_the_mean_over_objects_not_over_frames() -> None:
    # Object 1 (two visible frames) is perfect; object 2 (one visible frame) is missed.
    result = score_objects(_answer(_entry(1, 1, A1), _entry(1, 2, A2)), _metadata())

    assert result["reward"] == 0.5
    assert (result["visible_frames"], result["visible_iou_sum"]) == (3, 2.0)


def test_an_object_that_is_never_visible_is_left_out_of_the_reward() -> None:
    hidden = {"id": 3, "targets": [{"frame": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "visible": 0.0}]}
    metadata = _metadata(objects=[*METADATA["objects"], hidden])

    result = score_objects(EXACT, metadata)

    assert result["reward"] == 1.0
    assert result["scored_objects"] == 2
    assert result["object_scores"][2].reward is None


def test_a_task_where_no_object_is_ever_visible_has_zero_reward_and_no_weight() -> None:
    hidden = {"id": 1, "targets": [{"frame": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "visible": 0.0}]}

    result = score_objects(_answer(_entry(1, 1, A1)), _metadata(objects=[hidden]))

    assert (result["reward"], result["scored_objects"], result["visible_frames"]) == (0.0, 0, 0)


@pytest.mark.parametrize(
    "text",
    [
        '<answer>{"boxes":[{"frame":1,"bbox":[100,100,300,300]},{"frame":2,"bbox":[400,500,700,900]}]}</answer>',
        '<answer>{"boxes":[{"frame":1,"bbox":[100,100,200,300]},{"frame":2,"bbox":[450,500,700,800]}]}</answer>',
        '<answer>{"boxes":[{"frame":2,"bbox":[400,500,700,900]}]}</answer>',
        "",
    ],
)
def test_a_single_object_task_scores_the_same_as_the_got10k_server(text: str) -> None:
    single = Got10kVerifierMetadata.model_validate(
        {"image_width": 1000, "image_height": 500, "targets": METADATA["objects"][0]["targets"]}
    )

    expected = score_sequence(text, single)
    result = score_objects(text, _single_object_metadata())

    for key in ("reward", "visible_frames", "visible_iou_sum", "visible_sr50_count", "visible_sr75_count"):
        assert result[key] == expected[key]


def test_metrics_pool_frames_over_objects_and_tasks(server: MultiObjectTrackingResourcesServer) -> None:
    perfect = score_objects(EXACT, _metadata())
    missed = score_objects("", _single_object_metadata())
    tasks = [[perfect], [missed]]

    metrics = server.compute_metrics(tasks)

    assert metrics == {
        "visible_gt_samples": 5,
        "visible_gt_AO": 0.6,
        "visible_gt_SR@0.5": 0.6,
        "visible_gt_SR@0.75": 0.6,
        "parse_pass_ratio": 0.5,
    }
    assert server.compute_repeat_metrics(tasks) == metrics
    assert server.get_key_metrics({**metrics, "mean/reward": 0.5}) == {
        key: value for key, value in metrics.items() if key != "visible_gt_samples"
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def test_verify_scores_the_assistant_message_and_ignores_reasoning(server: MultiObjectTrackingResourcesServer) -> None:
    body = _request("Working on it.", EXACT, reasoning=_answer(_entry(1, 1, DISJOINT)))

    result = asyncio.run(server.verify(body))

    assert result.reward == 1.0
    assert result.parse_status == "structured"
    assert (result.scored_objects, result.visible_frames, result.visible_iou_sum) == (2, 3, 3.0)
    assert result.verifier_metadata["sequence"] == "seq"


def test_verify_handles_a_response_with_no_message(server: MultiObjectTrackingResourcesServer) -> None:
    result = asyncio.run(server.verify(_request(reasoning="thinking only")))

    assert result.reward == 0.0
    assert result.parse_status == "empty"


# ---------------------------------------------------------------------------
# Task data schema and example rows
# ---------------------------------------------------------------------------


def test_committed_example_rows_validate_against_the_task_data_schema() -> None:
    server_dir = Path(__file__).parents[1]
    path = server_dir / "data" / "example.jsonl"
    lines = path.read_text().splitlines()

    report = validate_jsonl_rows("multi_object_tracking", load_task_data_schema(server_dir), str(path), lines)

    assert len(lines) == 5
    assert report.clean, report.summary()


def test_committed_example_rows_are_the_synthetic_ones_and_their_ground_truth_scores_one() -> None:
    path = Path(__file__).parents[1] / "data" / "example.jsonl"
    committed = [json.loads(line) for line in path.read_text().splitlines()]

    assert committed == example_rows()
    for row in committed:
        metadata = TaskData.model_validate(row["verifier_metadata"])
        assert score_objects(ground_truth_answer(row), metadata)["reward"] == pytest.approx(1.0, abs=0.05)
        assert score_objects("", metadata)["reward"] == 0.0


def test_the_request_model_and_the_task_data_schema_are_one_class() -> None:
    assert TaskData is MultiObjectTrackingVerifierMetadata
    assert MultiObjectTrackingVerifyRequest.model_fields["verifier_metadata"].annotation is TaskData


def test_task_data_defaults_and_required_fields() -> None:
    minimal = TaskData.model_validate(
        {
            "image_width": 10,
            "image_height": 10,
            "objects": [{"id": 1, "targets": [{"frame": 1, "bbox": [1, 2, 3, 4]}]}],
        }
    )
    assert (minimal.sequence, minimal.split) == (None, None)
    assert minimal.objects[0].targets[0].visible == 1.0

    for broken in (
        {key: value for key, value in METADATA.items() if key != "image_width"},
        {**METADATA, "objects": []},
        {**METADATA, "objects": [{"id": 1, "targets": []}]},
        {**METADATA, "objects": [{"targets": [{"frame": 1, "bbox": [1.0, 2.0, 3.0, 4.0]}]}]},
        {**METADATA, "objects": [{"id": 1, "targets": [{"frame": 1, "bbox": [1.0, 2.0, 3.0]}]}]},
    ):
        with pytest.raises(ValidationError):
            TaskData.model_validate(broken)
