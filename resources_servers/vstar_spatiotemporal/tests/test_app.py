# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.verification import JudgeError, exercise_verifier_fixture

from .. import app as app_module
from ..app import (
    JUDGE_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    VERIFIER_FIXTURE,
    VstarResourcesServer,
    VstarResourcesServerConfig,
    VstarVerifierMetadata,
    VstarVerifyRequest,
    aggregate_metrics,
    answer_text,
    balance_brackets,
    close_truncated_json,
    parse_judge_score,
    parse_span,
    parse_tracks,
    score_reply,
    spatial_scores,
    temporal_iou,
)


# A 1000x500 video, so a grid box [x1,y1,x2,y2] is the pixel box [x1, y1/2, x2, y2/2].
METADATA = {
    "item_index": 7,
    "sub_task": "spatial",
    "domain": "Animals",
    "fps": 25.0,
    "frame_count": 250,
    "width": 1000,
    "height": 500,
    "question": "what is the dog chasing?",
    "answer": "a red ball",
    "timestamps": [2.0, 6.0],
    "bboxes": [
        {"timestamp": 2, "xmin": 100, "ymin": 50, "xmax": 300, "ymax": 150},
        {"timestamp": 3, "xmin": 400, "ymin": 250, "xmax": 700, "ymax": 450},
    ],
}
GRID_2, GRID_3 = [100, 100, 300, 300], [400, 500, 700, 900]
EXACT_TRACKS = f'<answer>{{"tracks":[{{"id":1,"t":2,"box":{GRID_2}}},{{"id":1,"t":3,"box":{GRID_3}}}]}}</answer>'
PIXELS = {"2": [100.0, 50.0, 300.0, 150.0], "3": [400.0, 250.0, 700.0, 450.0]}


def _metadata(sub_task: str = "spatial", **overrides) -> VstarVerifierMetadata:
    return VstarVerifierMetadata.model_validate({**METADATA, "sub_task": sub_task, **overrides})


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


def _request(sub_task: str, *texts: str, **kwargs) -> VstarVerifyRequest:
    return VstarVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "watch"},
            "response": _response(*texts, **kwargs),
            "verifier_metadata": {**METADATA, "sub_task": sub_task},
        }
    )


@pytest.fixture
def server() -> VstarResourcesServer:
    config = VstarResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server={"type": "responses_api_models", "name": "judge"},
    )
    return VstarResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _judged(server: VstarResourcesServer, body: VstarVerifyRequest, judge_reply: str):
    """Verify `body` with the judge model answering `judge_reply`; returns (result, the judge call's kwargs)."""
    from nemo_gym.core import NeMoGymResponse

    judged = NeMoGymResponse.model_validate(_response(judge_reply))
    with patch.object(app_module, "call_judge", new=AsyncMock(return_value=judged)) as call:
        result = asyncio.run(server.verify(body))
    return result, call.call_args.kwargs if call.call_args else None


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


def test_answer_text_is_the_first_answer_tag_or_the_whole_reply() -> None:
    assert answer_text("so <answer> a ball \n</answer> and <answer>other</answer>") == "a ball"
    assert answer_text("a ball") == "a ball"
    assert answer_text("<answer>never closed") == "<answer>never closed"
    assert answer_text("<answer></answer>") == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<answer>From <t>2</t>s to <t>6.5</t>s</answer>", [2.0, 6.5]),
        ("From <t>2</t>s to <t>6</t>s", [2.0, 6.0]),
        ("<answer>From 1:05 to 2:10</answer>", [65.0, 130.0]),  # m:ss is read as seconds
        ("<answer>[3, 4]</answer>", [3.0, 4.0]),
        ("<answer>6 then 2</answer>", [6.0, 2.0]),  # kept in the order given
        ("<think>frames 1 2 3</think><answer>From <t>2</t>s to <t>6</t>s</answer>", [2.0, 6.0]),
    ],
)
def test_a_span_is_the_two_numbers_of_the_answer(text: str, expected: list[float]) -> None:
    assert parse_span(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "<answer>around 5 seconds</answer>",  # one number
        "<answer>From <t>2</t>s to <t>6</t>s, about 4 seconds</answer>",  # three numbers
        "<answer>From 2s to 6s</answer>",  # a number directly followed by a letter is not read
        "From <t>2</t>s to <t>6</t>s in frame 3",  # no answer tag, so the whole reply is read
        "<answer>unknown</answer>",
    ],
)
def test_a_reply_without_exactly_two_numbers_has_no_span(text: str) -> None:
    assert parse_span(text) == []


@pytest.mark.parametrize(
    "text",
    [
        EXACT_TRACKS,
        f'{{"tracks":[{{"id":1,"t":2,"box":{GRID_2}}},{{"id":1,"t":3,"box":{GRID_3}}}]}}',
        f'<answer>```json\n{{"tracks":[{{"id":1,"t":2.0,"box":{GRID_2}}},{{"id":1,"t":"3","box":{GRID_3}}}]}}\n```</answer>',
        # One entry per object with a box per time.
        f'<answer>{{"from":2,"to":3,"tracks":[{{"id":1,"boxes":{{"2":{GRID_2},"3.0":{GRID_3}}}}}]}}</answer>',
        # A bare list of entries.
        f'<answer>[{{"id":1,"t":2,"box":{GRID_2}}},{{"id":1,"t":3,"box":{GRID_3}}}]</answer>',
        # Fractions of the frame instead of the 0-1000 grid.
        '<answer>{"tracks":[{"t":2,"box":[0.1,0.1,0.3,0.3]},{"t":3,"box":[0.4,0.5,0.7,0.9]}]}</answer>',
        # Cut off after the second entry: the open brackets are closed.
        f'<answer>{{"tracks":[{{"id":1,"t":2,"box":{GRID_2}}},{{"id":1,"t":3,"box":{GRID_3}}},',
        # Cut off inside a third entry, which is dropped.
        f'<answer>{{"tracks":[{{"id":1,"boxes":{{"2":{GRID_2},"3":{GRID_3},"4":[1,2',
    ],
)
def test_track_answers_in_several_shapes_give_pixel_boxes_per_second(text: str) -> None:
    boxes = parse_tracks(text, 1000, 500)

    assert boxes is not None and set(boxes) == {"2", "3"}
    assert boxes["2"] == pytest.approx(PIXELS["2"]) and boxes["3"] == pytest.approx(PIXELS["3"])


@pytest.mark.parametrize(
    "text",
    [
        "",
        "<answer>no object visible</answer>",
        '<answer>{"boxes":[[1,2,3,4]]}</answer>',  # no "tracks" list
        '<answer>{"tracks":[]}</answer>',
        '<answer>{"tracks":[{"id":1,"t":2,"box":[1,2,3]}]}</answer>',  # not four coordinates
        '<answer>{"tracks":[{"id":1,"t":"soon","box":[1,2,3,4]}]}</answer>',
        '<answer>{"tracks":[{"id":1,"t":NaN,"box":[1,2,3,4]}]}</answer>',
        '<answer>{"tracks":[{"id":1,"box":[1,2,3,4]}]}</answer>',  # no time
        '<answer>{"tracks":[[2,[1,2,3,4]]]}</answer>',  # entries must be objects
    ],
)
def test_replies_without_usable_boxes_give_none(text: str) -> None:
    assert parse_tracks(text, 1000, 500) is None


def test_a_time_is_rounded_to_a_whole_second_and_the_closest_one_wins() -> None:
    text = (
        '{"tracks":[{"t":2.4,"box":[0,0,100,100]},{"t":2.1,"box":[200,200,400,400]},'
        '{"t":1.8,"box":[500,500,600,600]},{"t":3.5,"box":[1,1,2,2]}]}'
    )

    boxes = parse_tracks(text, 1000, 500)

    # 2.4, 2.1 and 1.8 all round to second 2; 2.1 is the closest. 3.5 rounds to the even second 4.
    assert boxes == {"2": [200.0, 100.0, 400.0, 200.0], "4": [1.0, 0.5, 2.0, 1.0]}


def test_bracket_repairs() -> None:
    assert balance_brackets('{"a":[1,2') == '{"a":[1,2]}'
    assert balance_brackets('"a":[1,2]}') == '{"a":[1,2]}'
    # Closing by count gets nested shapes wrong; closing innermost first does not.
    cut = '{"tracks":[{"id":1,"boxes":{"2":[1,2,3,4],"3":'
    with pytest.raises(json.JSONDecodeError):
        json.loads(balance_brackets(cut))
    assert json.loads(close_truncated_json(cut)) == {"tracks": [{"id": 1, "boxes": {"2": [1, 2, 3, 4]}}]}
    assert json.loads(close_truncated_json('{"tracks":[{"id":1,"t":')) == {"tracks": [{"id": 1}]}
    assert json.loads(close_truncated_json('{"tracks":[{"id":1,"note":"cut he')) == {
        "tracks": [{"id": 1, "note": "cut he"}]
    }


def test_an_answer_cut_off_inside_an_entry_keeps_the_entries_before_it() -> None:
    cut = f'<answer>{{"tracks":[{{"id":1,"t":2,"box":{GRID_2}}},{{"id":1,"t":3,"box":[400, 500'

    assert parse_tracks(cut, 1000, 500) == {"2": PIXELS["2"]}
    # No closing brace anywhere: the text up to the last complete box is repaired instead.
    assert parse_tracks('<answer>{"tracks":[{"id":1,"boxes":{"2":[1,2,3,4],"3":[5,6', 1000, 500) == {
        "2": [1.0, 1.0, 3.0, 2.0]
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_temporal_iou() -> None:
    assert temporal_iou([2.0, 6.0], [2.0, 6.0]) == 1.0
    assert temporal_iou([2.0, 6.0], [4.0, 8.0]) == pytest.approx(2 / 6)
    assert temporal_iou([2.0, 6.0], [7.0, 9.0]) == 0.0
    # A reversed span has no overlap.
    assert temporal_iou([2.0, 6.0], [6.0, 2.0]) == 0.0
    assert temporal_iou([2.0, 6.0], []) == 0.0
    assert temporal_iou([2.0, 2.0], [2.0, 2.0]) == 0.0


def test_boxes_are_scored_at_every_annotated_second() -> None:
    boxes = _metadata().bboxes

    assert spatial_scores(boxes, PIXELS) == ([1.0] * 5, 1.0)
    # Second 3 is missing from the answer and scores 0; an extra second is ignored.
    aps, miou = spatial_scores(boxes, {"2": PIXELS["2"], "9": [0.0, 0.0, 10.0, 10.0]})
    assert (aps, miou) == ([0.5] * 5, 0.5)
    # Half of the box at second 2: IoU 0.5 passes the 0.1, 0.3 and 0.5 thresholds.
    aps, miou = spatial_scores(boxes, {"2": [100.0, 50.0, 200.0, 150.0], "3": PIXELS["3"]})
    assert aps == [1.0, 1.0, 1.0, 0.5, 0.5] and miou == pytest.approx(0.75)
    assert spatial_scores(boxes, None) == ([0.0] * 5, 0.0)
    assert spatial_scores([], PIXELS) == ([0.0] * 5, 0.0)


def test_each_sub_task_has_its_own_reward() -> None:
    spatial = score_reply(EXACT_TRACKS, _metadata("spatial_2"))
    assert (spatial["reward"], spatial["spatial_miou"], spatial["parsed"]) == (1.0, 1.0, True)
    assert spatial["predicted_boxes"] == PIXELS

    temporal = score_reply("<answer>From <t>4</t>s to <t>8</t>s</answer>", _metadata("temporal_2"))
    assert temporal["reward"] == temporal["temporal_iou"] == pytest.approx(2 / 6)
    assert temporal["predicted_span"] == [4.0, 8.0]

    unreadable = score_reply("<answer>later</answer>", _metadata("temporal"))
    assert (unreadable["reward"], unreadable["parsed"], unreadable["predicted_span"]) == (0.0, False, [])


@pytest.mark.parametrize(
    ("reply", "score"), [("3", 3), (" 2\n", 2), ("0", 0), ("2.0", -1), ("Rating: 3", -1), ("", -1)]
)
def test_a_judge_reply_must_be_a_single_integer(reply: str, score: int) -> None:
    assert parse_judge_score(reply) == score


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def test_vqa_answers_are_rated_by_the_judge(server: VstarResourcesServer) -> None:
    body = _request("vqa", "<answer>a ball that is red</answer>", reasoning="<answer>a cat</answer>")

    result, call = _judged(server, body, " 2\n")

    assert (result.reward, result.vqa_score, result.answered, result.judge_output) == (1.0, 2, True, "2")
    assert result.answer_vqa == "a ball that is red"
    assert call["server_name"] == "judge" and call["url_path"] == "/v1/responses"
    params = call["json"]
    assert (params.temperature, params.max_output_tokens) == (0.0, 512)
    assert [(message.role, message.content) for message in params.input] == [
        ("system", JUDGE_SYSTEM_PROMPT),
        ("user", JUDGE_PROMPT.format("what is the dog chasing?", "a red ball", "a ball that is red")),
    ]


@pytest.mark.parametrize(("judge_reply", "score"), [("1", 1), ("0", 0), ("I think 3", -1)])
def test_low_or_unreadable_ratings_earn_nothing(server: VstarResourcesServer, judge_reply: str, score: int) -> None:
    result, _ = _judged(server, _request("vqa", "a stick"), judge_reply)

    assert (result.reward, result.vqa_score, result.answered) == (0.0, score, True)
    # The answer was not inside <answer> tags, so the whole reply was sent to the judge.
    assert (result.answer_vqa, result.parsed) == ("a stick", False)


def test_an_empty_answer_tag_is_an_unanswered_item_and_is_not_judged(server: VstarResourcesServer) -> None:
    result, call = _judged(server, _request("vqa", "<answer> </answer>"), "3")

    assert call is None
    assert (result.reward, result.answered, result.vqa_score) == (0.0, False, None)


def test_a_reply_with_no_text_is_rated_zero_without_the_judge(server: VstarResourcesServer) -> None:
    result, call = _judged(server, _request("vqa", reasoning="ran out of tokens"), "3")

    assert call is None
    assert (result.reward, result.answered, result.vqa_score) == (0.0, True, 0)


def test_vqa_needs_a_judge() -> None:
    config = VstarResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    server = VstarResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    with pytest.raises(JudgeError, match="judge_model_server"):
        asyncio.run(server.verify(_request("vqa", "a ball")))
    # The other sub-tasks are scored without one.
    assert asyncio.run(server.verify(_request("spatial", EXACT_TRACKS))).reward == 1.0


def test_box_and_span_tasks_read_the_message_not_the_reasoning(server: VstarResourcesServer) -> None:
    spatial = asyncio.run(server.verify(_request("spatial", "Here.", EXACT_TRACKS, reasoning='{"tracks":[]}')))
    assert (spatial.reward, spatial.spatial_aps, spatial.item_index) == (1.0, [1.0] * 5, 7)

    temporal = asyncio.run(server.verify(_request("temporal", reasoning="From 2s to 6s")))
    assert (temporal.reward, temporal.parsed) == (0.0, False)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _item(index: int, vqa: int | None, t1: float, s1: float, t2: float, s2: float, **metadata) -> list[dict]:
    """The five rollouts of one item. `vqa=None` is an item whose VQA answer was empty."""
    meta = {**METADATA, "item_index": index, **metadata}

    def rollout(sub_task: str, **fields) -> dict:
        return {"sub_task": sub_task, "item_index": index, "verifier_metadata": meta, "parsed": True, **fields}

    def spatial(miou: float) -> dict:
        return {
            "spatial_miou": miou,
            "spatial_aps": [float(miou >= threshold) for threshold in (0.1, 0.3, 0.5, 0.7, 0.9)],
        }

    return [
        rollout("vqa", answered=vqa is not None, vqa_score=vqa, parsed=vqa is not None),
        rollout("temporal", temporal_iou=t1),
        rollout("spatial", **spatial(s1)),
        rollout("spatial_2", **spatial(s2)),
        rollout("temporal_2", temporal_iou=t2),
    ]


def test_metrics_fold_the_five_tasks_of_each_item() -> None:
    rollouts = _item(0, 3, 0.8, 0.6, 0.4, 0.2) + _item(1, 1, 0.2, 0.0, 0.0, 0.0, domain="Sports", frame_count=2500)

    metrics = aggregate_metrics(rollouts)

    assert metrics["items"] == 2
    assert (metrics["VQA_acc"], metrics["VQA_avg_score"]) == (0.5, 2.0)
    assert metrics["temporal_mIoU"] == pytest.approx(0.5) and metrics["spatial_mIoU"] == pytest.approx(0.3)
    assert metrics["temporal_2_mIoU"] == pytest.approx(0.2) and metrics["spatial_2_mIoU"] == pytest.approx(0.1)
    assert metrics["AM1"] == pytest.approx((0.5 + 0.5 + 0.3) / 3)
    assert metrics["AM2"] == pytest.approx((0.5 + 0.2 + 0.1) / 3)
    assert metrics["mAM"] == pytest.approx((metrics["AM1"] + metrics["AM2"]) / 2)
    assert metrics["LGM1"] == pytest.approx(-(math.log(0.5) + math.log(0.5) + math.log(0.7)) / 3)
    assert metrics["mLGM"] == pytest.approx((metrics["LGM1"] + metrics["LGM2"]) / 2)
    assert (metrics["temporal_R1@0.3"], metrics["temporal_R1@0.7"], metrics["temporal_2_R1@0.5"]) == (0.5, 0.5, 0.0)
    assert (metrics["spatial_mAP@0.5"], metrics["spatial_mAP@0.7"], metrics["spatial_2_mAP@0.1"]) == (0.5, 0.0, 0.5)
    # Item 0 passes every part of both chains; item 1 passes none.
    assert metrics["VQA&Temp&Spat_1"] == metrics["VQA&Temp&Spat_2"] == metrics["Temp&Spat_1"] == 0.5
    assert metrics["parsed_rate_mean"] == 1.0
    # A 10-second video is Short, a 100-second one Medium.
    assert (metrics["length/Short/items"], metrics["length/Medium/items"]) == (1, 1)
    assert (metrics["domain/Animals/VQA_acc"], metrics["domain/Sports/VQA_acc"]) == (1.0, 0.0)
    assert metrics["domain/Animals/mAM"] == pytest.approx(((1 + 0.8 + 0.6) / 3 + (1 + 0.4 + 0.2) / 3) / 2)
    # A group whose accuracy is 1 has an infinite LGM, which is left out.
    assert "domain/Animals/mLGM" not in metrics and "domain/Sports/mLGM" in metrics


def test_an_unanswered_item_counts_only_in_the_overall_denominator() -> None:
    rollouts = _item(0, 2, 0.6, 0.6, 0.6, 0.6) + _item(1, None, 0.9, 0.9, 0.9, 0.9)

    metrics = aggregate_metrics(rollouts)

    assert metrics["items"] == 2
    # Accuracy is over both items; the temporal and spatial means are over the answered one only.
    assert metrics["VQA_acc"] == 0.5
    assert metrics["temporal_mIoU"] == pytest.approx(0.6) and metrics["spatial_2_mIoU"] == pytest.approx(0.6)
    assert metrics["VQA&Temp&Spat_1"] == 0.5
    assert metrics["parsed_rate_vqa"] == 0.5 and metrics["parsed_rate_temporal"] == 1.0
    # The group holds the answered item only.
    assert metrics["length/Short/items"] == 1 and metrics["length/Short/VQA_acc"] == 1.0


def test_an_unreadable_judge_reply_counts_as_zero_but_not_as_a_valid_rating() -> None:
    metrics = aggregate_metrics(_item(0, -1, 0.5, 0.5, 0.5, 0.5) + _item(1, 2, 0.5, 0.5, 0.5, 0.5))

    assert (metrics["VQA_avg_score"], metrics["VQA_avg_valid_score"], metrics["VQA_acc"]) == (1.0, 2.0, 0.5)


def test_items_with_missing_tasks_are_still_aggregated() -> None:
    # A limited run: only the VQA and chain-1 temporal tasks of the item were run.
    metrics = aggregate_metrics(_item(0, 3, 0.5, 0.0, 0.0, 0.0)[:2])

    assert (metrics["items"], metrics["temporal_mIoU"], metrics["spatial_mIoU"]) == (1, 0.5, 0.0)
    assert metrics["parsed_rate_spatial"] == 0.0
    assert aggregate_metrics([]) == {}
    assert aggregate_metrics([{"reward": 1.0}]) == {}


def test_repeats_are_aggregated_separately_and_averaged(server: VstarResourcesServer) -> None:
    first = _item(0, 3, 1.0, 0.5, 0.5, 0.5)
    second = _item(0, 0, 0.0, 0.5, 0.5, 0.5)
    tasks = [[a, b] for a, b in zip(first, second)]

    metrics = server.compute_metrics(tasks)

    assert (metrics["VQA_acc"], metrics["temporal_mIoU"], metrics["spatial_mIoU"]) == (0.5, 0.5, 0.5)
    # mLGM is infinite in the first repeat (accuracy 1), so it is the second repeat's value:
    # LGM1 = -(log 1 + log 1 + log 0.5) / 3 and LGM2 = -(log 1 + log 0.5 + log 0.5) / 3.
    assert metrics["mLGM"] == pytest.approx(-math.log(0.5) / 2)
    assert server.compute_repeat_metrics(tasks) == metrics
    assert server.get_key_metrics({**metrics, "mean/reward": 0.5}) == {
        key: metrics[key]
        for key in (
            "mAM",
            "mLGM",
            "VQA_acc",
            "temporal_mIoU",
            "spatial_mIoU",
            "temporal_2_mIoU",
            "spatial_2_mIoU",
            "parsed_rate_mean",
        )
    }
