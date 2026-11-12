# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.verification import exercise_verifier_fixture

from ..app import (
    VERIFIER_FIXTURE,
    Got10kResourcesServer,
    Got10kResourcesServerConfig,
    Got10kVerifierMetadata,
    Got10kVerifyRequest,
    aggregate_frame_metrics,
    clip_box_to_image,
    grid_box_to_pixel_xywh,
    iou_xywh,
    parse_boxes,
    score_sequence,
)


# A 1000x500 image, so a grid box [x1,y1,x2,y2] is the pixel box [x1, y1/2, x2, y2/2].
METADATA = {
    "sequence": "seq",
    "image_width": 1000,
    "image_height": 500,
    "targets": [
        {"frame": 1, "bbox": [100.0, 50.0, 200.0, 100.0], "visible": 8.0},
        {"frame": 2, "bbox": [400.0, 250.0, 300.0, 200.0], "visible": 8.0},
    ],
}
BOX_1 = [100, 100, 300, 300]
BOX_2 = [400, 500, 700, 900]
EXACT = f'<answer>{{"boxes":[{{"frame":1,"bbox":{BOX_1}}},{{"frame":2,"bbox":{BOX_2}}}]}}</answer>'


def _metadata(**overrides) -> Got10kVerifierMetadata:
    return Got10kVerifierMetadata.model_validate({**METADATA, **overrides})


def _response(*texts: str | None, reasoning: str | None = None) -> dict:
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


def _request(*texts: str, **kwargs) -> Got10kVerifyRequest:
    return Got10kVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "track"},
            "response": _response(*texts, **kwargs),
            "verifier_metadata": METADATA,
        }
    )


@pytest.fixture
def server() -> Got10kResourcesServer:
    config = Got10kResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return Got10kResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


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
        f"Tracking the boat.\n{EXACT}\nDone.",
        # No answer tags at all.
        f'{{"boxes":[{{"frame":1,"bbox":{BOX_1}}},{{"frame":2,"bbox":{BOX_2}}}]}}',
        # Alternative list keys and box keys.
        f'<answer>{{"predictions":[{{"frame":1,"box":{BOX_1}}},{{"frame":2,"bbox_2d":{BOX_2}}}]}}</answer>',
        '<answer>{"boxes":[{"frame":1,"x1":100,"y1":100,"x2":300,"y2":300},'
        '{"frame":2,"bbox":{"x1":400,"y1":500,"x2":700,"y2":900}}]}</answer>',
        # A bare list of entries, and of boxes in frame order.
        f'<answer>[{{"frame":1,"bbox":{BOX_1}}},{{"frame":2,"bbox":{BOX_2}}}]</answer>',
        f"<answer>[{BOX_1},{BOX_2}]</answer>",
        # Entries without frame ids are taken in order.
        f'<answer>{{"boxes":[{{"bbox":{BOX_1}}},{{"bbox":{BOX_2}}}]}}</answer>',
        # Frame ids as object keys, with or without a label.
        f'<answer>{{"1":{BOX_1},"frame 2":{BOX_2}}}</answer>',
        # Every frame crammed into one object with repeated keys.
        f'<answer>{{"frame":1,"bbox":{BOX_1},"frame":2,"bbox":{BOX_2}}}</answer>',
        # Python literal rather than JSON; string-valued boxes and ids.
        f"<answer>{{'boxes':[{{'frame':1,'bbox':({BOX_1[0]},100,300,300)}},{{'frame':2,'bbox':{BOX_2}}}]}}</answer>",
        f'<answer>{{"boxes":[{{"frame":"1","bbox":"{BOX_1}"}},{{"frame":"2","bbox":"({BOX_2[0]},500,700,900)"}}]}}</answer>',
        # An unusable first answer block is skipped in favour of a later one.
        f"<answer>not json</answer> {EXACT}",
        # Objects embedded in prose.
        f'First {{"boxes":[{{"frame":1,"bbox":{BOX_1}}}]}} and then {{"boxes":[{{"frame":2,"bbox":{BOX_2}}}]}}',
        # Truncated JSON, recovered pair by pair.
        f'<answer>{{"boxes":[{{"frame":1,"bbox":{BOX_1}}},{{"frame":2,"bbox":{BOX_2}}},{{"frame":3,"bbox":[1,2',
        # Labeled lines and bare numbers (4 per frame, or 5 with a leading frame id).
        f"Frame 1: {BOX_1}\nImage 2 is at {BOX_2}",
        "100 100 300 300 400 500 700 900",
        "1 100 100 300 300 2 400 500 700 900",
    ],
)
def test_lenient_answer_shapes_all_recover_both_boxes(text: str) -> None:
    boxes, status = parse_boxes(text, [1, 2])

    assert status == "structured"
    assert boxes == {1: BOX_1, 2: BOX_2}
    assert score_sequence(text, _metadata())["reward"] == 1.0


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("I cannot see the target.", "parse_failed"),
        # Numbers that fit neither 4 nor 5 per frame.
        ("The object is near 120, 340.", "ambiguous_numbers"),
        # Five per frame, but the leading numbers are not the frame ids.
        ("7 100 100 300 300 9 400 500 700 900", "ambiguous_numbers"),
        ('<answer>{"boxes":"none"}</answer>', "parse_failed"),
        ('<answer>"just a string"</answer>', "parse_failed"),
        # Entries that name no requested frame or carry no usable box.
        (
            '<answer>{"boxes":[{"frame":7,"bbox":[1,2,3,4]},{"frame":[1],"bbox":[1,2,3,4]}]}</answer>',
            "ambiguous_numbers",
        ),
        ('<answer>{"boxes":[{"frame":1,"bbox":[1,2]},{"frame":2,"bbox":["a",2,3,4]}]}</answer>', "ambiguous_numbers"),
        (
            '<answer>{"boxes":[{"frame":1,"x1":"a","y1":2,"x2":3,"y2":4},{"frame":2,"bbox":null}]}</answer>',
            "ambiguous_numbers",
        ),
        ('<answer>{"boxes":[{"frame":1,"bbox":7},{"frame":2,"bbox":""}]}</answer>', "ambiguous_numbers"),
        ("<answer>[[1,2,3,4],5]</answer>", "ambiguous_numbers"),
    ],
)
def test_answers_without_recoverable_boxes_score_zero(text: str, status: str) -> None:
    result = score_sequence(text, _metadata())

    assert result["parse_status"] == status
    assert result["reward"] == 0.0
    assert result["parsed_frames"] == 0
    assert [score.status for score in result["frame_scores"]] == ["missing", "missing"]


def test_only_requested_frames_are_kept_and_first_recovered_pair_wins() -> None:
    truncated = (
        f'{{"boxes":[{{"frame":1,"bbox":{BOX_1}}},{{"frame":1,"bbox":[0,0,5,5]}},{{"frame":9,"bbox":{BOX_2}}},{{'
    )

    assert parse_boxes(truncated, [1, 2]) == ({1: BOX_1}, "structured")


def test_a_single_requested_frame_accepts_a_bare_box() -> None:
    assert parse_boxes(f"<answer>{BOX_1}</answer>", [1]) == ({1: BOX_1}, "structured")
    assert parse_boxes(f'<answer>{{"bbox":{BOX_1}}}</answer>', [1]) == ({1: BOX_1}, "structured")


def test_a_frame_missing_from_the_answer_scores_zero_for_that_frame_only() -> None:
    result = score_sequence(f'<answer>{{"boxes":[{{"frame":2,"bbox":{BOX_2}}}]}}</answer>', _metadata())

    assert [(score.status, score.iou) for score in result["frame_scores"]] == [("missing", 0.0), ("ok", 1.0)]
    assert result["reward"] == 0.5
    assert result["parsed_frames"] == 1


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_grid_boxes_are_clamped_to_the_grid_before_scaling() -> None:
    assert grid_box_to_pixel_xywh([-50, 100, 1200, 300], 1000, 500) == pytest.approx([0.0, 50.0, 1000.0, 100.0])


def test_boxes_are_clipped_to_the_image_and_degenerate_ones_dropped() -> None:
    assert clip_box_to_image([900.0, 400.0, 300.0, 300.0], 1000, 500) == ([900.0, 400.0, 100.0, 100.0], "ok")
    assert clip_box_to_image([10.0, 10.0, -5.0, 20.0], 1000, 500) == (None, "non_positive")
    assert clip_box_to_image([float("nan"), 0.0, 5.0, 5.0], 1000, 500) == (None, "non_finite")


def test_iou() -> None:
    assert iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(1 / 3)
    assert iou_xywh([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0
    assert iou_xywh([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


@pytest.mark.parametrize(
    ("bbox", "status"),
    [
        ([300, 300, 100, 100], "non_positive"),  # corners swapped
        ([100, 100, 100, 300], "non_positive"),  # zero width
        ('["NaN",100,300,300]', "non_finite"),
    ],
)
def test_unusable_predicted_boxes_score_zero(bbox, status: str) -> None:
    text = f'<answer>{{"boxes":[{{"frame":1,"bbox":{json.dumps(bbox)}}},{{"frame":2,"bbox":{BOX_2}}}]}}</answer>'
    result = score_sequence(text, _metadata())

    assert [score.status for score in result["frame_scores"]] == [status, "ok"]
    assert result["frame_scores"][0].predicted_bbox is None
    assert result["reward"] == 0.5


def test_out_of_range_predictions_are_clamped_rather_than_rejected() -> None:
    # [-100,-100,300,300] clamps to [0,0,300,300]: pixel box [0,0,300,150] against GT [100,50,200,100].
    text = f'<answer>{{"boxes":[{{"frame":1,"bbox":[-100,-100,300,300]}},{{"frame":2,"bbox":{BOX_2}}}]}}</answer>'
    result = score_sequence(text, _metadata())

    assert result["frame_scores"][0].iou == pytest.approx(20000 / 45000)
    assert result["frame_scores"][0].predicted_bbox == [0.0, 0.0, 300.0, 150.0]


# ---------------------------------------------------------------------------
# Visibility and aggregation
# ---------------------------------------------------------------------------


def test_frames_where_the_target_is_not_visible_are_not_scored() -> None:
    targets = [{**METADATA["targets"][0], "visible": 0.0}, METADATA["targets"][1]]
    result = score_sequence(
        '<answer>{"boxes":[{"frame":2,"bbox":[400,500,700,900]}]}</answer>', _metadata(targets=targets)
    )

    assert result["reward"] == 1.0
    assert (result["visible_frames"], result["target_frames"], result["parsed_frames"]) == (1, 2, 1)


def test_a_sequence_with_no_visible_frame_has_zero_reward_and_no_weight() -> None:
    targets = [{**target, "visible": 0.0} for target in METADATA["targets"]]
    result = score_sequence(EXACT, _metadata(targets=targets))

    assert result["reward"] == 0.0
    assert result["visible_frames"] == 0
    assert aggregate_frame_metrics([result]) == {"visible_gt_samples": 0, "parse_pass_ratio": 1.0}
    assert aggregate_frame_metrics([]) == {"visible_gt_samples": 0}


def test_success_rates_use_strict_thresholds() -> None:
    # Frame 1: pixel box [100,50,100,100] covers half of GT -> IoU 0.5, which is not a success at 0.5.
    text = f'<answer>{{"boxes":[{{"frame":1,"bbox":[100,100,200,300]}},{{"frame":2,"bbox":{BOX_2}}}]}}</answer>'
    result = score_sequence(text, _metadata())

    assert [score.iou for score in result["frame_scores"]] == [0.5, 1.0]
    assert (result["visible_sr50_count"], result["visible_sr75_count"]) == (1, 1)


def test_metrics_are_weighted_by_frames_not_by_sequences(server: Got10kResourcesServer) -> None:
    long_sequence = {
        "visible_frames": 9,
        "visible_iou_sum": 9.0,
        "visible_sr50_count": 9,
        "visible_sr75_count": 9,
        "target_frames": 10,
        "parsed_frames": 10,
    }
    short_sequence = {
        "visible_frames": 1,
        "visible_iou_sum": 0.0,
        "visible_sr50_count": 0,
        "visible_sr75_count": 0,
        "target_frames": 2,
        "parsed_frames": 0,
    }
    tasks = [[long_sequence], [short_sequence]]

    metrics = server.compute_metrics(tasks)

    # Sequence-weighted AO would be 0.5.
    assert metrics == {
        "visible_gt_samples": 10,
        "visible_gt_AO": 0.9,
        "visible_gt_SR@0.5": 0.9,
        "visible_gt_SR@0.75": 0.9,
        "parse_pass_ratio": 10 / 12,
    }
    assert server.compute_repeat_metrics(tasks) == metrics
    assert server.get_key_metrics({**metrics, "mean/reward": 0.5}) == {
        key: value for key, value in metrics.items() if key != "visible_gt_samples"
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def test_verify_scores_the_assistant_message_and_ignores_reasoning(server: Got10kResourcesServer) -> None:
    body = _request("Working on it.", EXACT, reasoning='maybe {"boxes":[{"frame":1,"bbox":[0,0,1,1]}]}')

    result = asyncio.run(server.verify(body))

    assert result.reward == 1.0
    assert result.parse_status == "structured"
    assert (result.visible_frames, result.visible_iou_sum) == (2, 2.0)
    assert result.verifier_metadata["sequence"] == "seq"


def test_verify_handles_a_response_with_no_message(server: Got10kResourcesServer) -> None:
    result = asyncio.run(server.verify(_request(reasoning="thinking only")))

    assert result.reward == 0.0
    assert result.parse_status == "empty"
