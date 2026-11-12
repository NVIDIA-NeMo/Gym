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
    Molmo2Object,
    Molmo2VerifierMetadata,
    Molmo2VideoTrackResourcesServer,
    Molmo2VideoTrackResourcesServerConfig,
    Molmo2VideoTrackVerifyRequest,
    aggregate_metrics,
    parse_tracks,
    point_in_mask,
    rle_runs,
    score_expression,
    score_frame,
)


WIDTH, HEIGHT = 100, 50


def _mask(x0: int, y0: int, w: int, h: int) -> dict:
    """Uncompressed column-major RLE of a filled box in a WIDTH x HEIGHT image."""
    runs = [0]
    for index in range(WIDTH * HEIGHT):
        column, row = divmod(index, HEIGHT)
        inside = x0 <= column < x0 + w and y0 <= row < y0 + h
        if inside == (len(runs) % 2 == 0):
            runs[-1] += 1
        else:
            runs.append(1)
    return {"size": [HEIGHT, WIDTH], "counts": runs}


# Object A sits at pixels x 10-19, y 10-19; object B at x 60-69, y 30-39.
A = {"xy": [15.0, 15.0], "mask": _mask(10, 10, 10, 10)}
B = {"xy": [65.0, 35.0], "mask": _mask(60, 30, 10, 10)}
# The same points on the 0-1000 grid: pixel (x, y) is grid (10 x, 20 y).
A_GRID, B_GRID = [150, 300], [650, 700]
METADATA = {
    "category": "misc",
    "image_width": WIDTH,
    "image_height": HEIGHT,
    "frames": [
        {"frame": 1, "objects": [A, B]},
        {"frame": 2, "objects": [A]},
        {"frame": 3, "objects": []},
    ],
}


def _metadata(**overrides) -> Molmo2VerifierMetadata:
    return Molmo2VerifierMetadata.model_validate({**METADATA, **overrides})


def _answer(*entries: tuple[int, list]) -> str:
    tracks = [{"id": index, "frame": frame, "xy": xy} for index, (frame, xy) in enumerate(entries, start=1)]
    return f"<answer>{json.dumps({'tracks': tracks})}</answer>"


EXACT = _answer((1, A_GRID), (1, B_GRID), (2, A_GRID))
EXACT_POINTS = {1: [(150.0, 300.0), (650.0, 700.0)], 2: [(150.0, 300.0)]}


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


def _request(*texts: str, **kwargs) -> Molmo2VideoTrackVerifyRequest:
    return Molmo2VideoTrackVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "track"},
            "response": _response(*texts, **kwargs),
            "verifier_metadata": METADATA,
        }
    )


@pytest.fixture
def server() -> Molmo2VideoTrackResourcesServer:
    config = Molmo2VideoTrackResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return Molmo2VideoTrackResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


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
        f"Tracking.\n{EXACT}\nDone.",
        # No answer tags; "points"/"point" keys; a bare list.
        EXACT.removeprefix("<answer>").removesuffix("</answer>"),
        '<answer>{"points":[{"frame":1,"point":[150,300]},{"frame":1,"point":[650,700]},'
        '{"frame":2,"point":[150,300]}]}</answer>',
        '<answer>[{"frame":1,"xy":[150,300]},{"frame":1,"xy":[650,700]},{"frame":2,"xy":[150,300]}]</answer>',
        # Frame numbers and coordinates as strings; extra keys and coordinates are ignored.
        '<answer>{"tracks":[{"frame":"1","xy":["150","300"]},{"frame":1,"xy":[650,700,9],"conf":0.4},'
        '{"frame":2.0,"xy":[150,300]}]}</answer>',
        # Text before the closing think tag is dropped, even when it holds an answer.
        f'<answer>{{"tracks":[{{"frame":3,"xy":[1,1]}}]}}</answer></think>{EXACT}',
        # An unusable first answer block is skipped in favour of a later one.
        f"<answer>not json</answer> {EXACT}",
        # The object is embedded in prose.
        f"Here you go: {EXACT.removeprefix('<answer>').removesuffix('</answer>')} Hope that helps.",
        # Molmo2's own markup: "frame:id visible x y", "frame: [x,y]" and "frame id x y".
        '<tracks coords="1:1 1 150 300 2 1 650 700;2:1 1 150 300;3:1 0 150 300">tracks</tracks>',
        '<tracks coords="1: [150,300] [650 700];2: [150, 300]">tracks</tracks>',
        '<points coords="1 1 150 300 2 650 700,2 1 150 300">points</points>',
        '<tracks coords="1:1 150 300 2 650 700;2.0:1 150 300;x:1 150 300;nocolon">tracks</tracks>',
    ],
)
def test_accepted_answer_shapes_recover_every_point(text: str) -> None:
    assert parse_tracks(text) == EXACT_POINTS
    assert score_expression(text, _metadata())["reward"] == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I cannot see any matching object.",
        '<answer>{"tracks":"none"}</answer>',
        '<answer>{"tracks":[]}</answer>',
        '<answer>"just a string"</answer>',
        "<answer>[1, 2, 3]</answer>",
        # Entries without a frame, without a point pair, or with an unusable frame.
        '<answer>{"tracks":[{"xy":[1,2]},{"frame":1},{"frame":1,"xy":[5]},{"frame":"a","xy":[1,2]},'
        '{"frame":1e999,"xy":[1,2]},{"frame":null,"xy":[1,2]}]}</answer>',
        # Truncated JSON is not repaired.
        '<answer>{"tracks":[{"id":1,"frame":1,"xy":[150,300]},{"id":1,"fra',
        '<tracks coords="">tracks</tracks>',
        '<tracks coords="1:nothing here;2:">tracks</tracks>',
        '<points coords="x y z">points</points>',
    ],
)
def test_answers_without_any_point_are_parse_failures(text: str) -> None:
    result = score_expression(text, _metadata())

    assert parse_tracks(text) == {}
    assert result["parse_ok"] is False
    # Frames 1 and 2 have visible objects and no prediction; frame 3 has neither and is skipped.
    assert (result["reward"], result["num_scored_frames"]) == (0.0, 2)


def test_an_entry_with_an_unusable_point_still_counts_as_a_parsed_answer() -> None:
    text = '<answer>{"tracks":[{"frame":1,"xy":["a",2]},{"frame":2,"xy":[NaN,300]}]}</answer> <tracks coords="1: [150,300]">'
    result = score_expression(text, _metadata())

    # The frames are recorded with no points, so neither the whole text nor the markup is consulted.
    assert parse_tracks(text) == {1: [], 2: []}
    assert (result["parse_ok"], result["reward"]) == (True, 0.0)


def test_native_markup_drops_points_flagged_invisible() -> None:
    assert parse_tracks('<tracks coords="1:1 0 150 300 2 1 650 700;2:1 0 150 300">t</tracks>') == {1: [(650.0, 700.0)]}


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def test_compressed_rle_is_decoded() -> None:
    # A 4x3 mask whose pixels 2-4 and 8-9 (column-major) are set. From the fourth run on, a run is
    # stored as its difference from the run two places back, here -1 twice.
    assert rle_runs("233OO") == [2, 3, 3, 2, 2]
    assert rle_runs(b"233OO") == [2, 3, 3, 2, 2]
    # Runs above 31 take several characters; equal alternating runs become zeros.
    assert rle_runs("5X1500") == [5, 40, 5, 40, 5]
    # A difference of -32 also takes two characters.
    assert rle_runs("5X15PO") == [5, 40, 5, 8]
    assert rle_runs([1, 2, 3]) == [1, 2, 3]


def test_point_in_mask_uses_the_nearest_pixel_and_rejects_points_off_the_image() -> None:
    mask = _mask(10, 10, 10, 10)

    assert point_in_mask(mask, 10, 10) and point_in_mask(mask, 19, 19)
    assert not point_in_mask(mask, 20, 10) and not point_in_mask(mask, 10, 9)
    assert point_in_mask(mask, 9.6, 19.4) and not point_in_mask(mask, 9.4, 15)
    assert (
        not point_in_mask(mask, -1, 15) and not point_in_mask(mask, 15, HEIGHT) and not point_in_mask(mask, WIDTH, 15)
    )
    # Runs that stop short of the image leave the rest unset.
    assert not point_in_mask({"size": [HEIGHT, WIDTH], "counts": [10, 5]}, 99, 49)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_each_prediction_is_matched_to_its_nearest_object() -> None:
    objects = [Molmo2Object.model_validate(A), Molmo2Object.model_validate(B)]

    assert score_frame([(65.0, 35.0), (15.0, 15.0)], objects) == (2, 1.0, 1.0, 1.0)
    # One point on A, one in empty space: the stray point is matched to B and misses its mask.
    assert score_frame([(15.0, 15.0), (90.0, 5.0)], objects) == (1, 0.5, 0.5, 0.5)
    # Two points on A: only one can be matched to it; the other is matched to B and misses.
    assert score_frame([(15.0, 15.0), (16.0, 16.0)], objects) == (1, 0.5, 0.5, 0.5)
    # A single prediction for two objects: full precision, half recall.
    assert score_frame([(15.0, 15.0)], objects) == (1, 1.0, 0.5, pytest.approx(2 / 3))
    # Three predictions for two objects: the unmatched one lowers precision.
    assert score_frame([(15.0, 15.0), (65.0, 35.0), (40.0, 5.0)], objects) == (2, pytest.approx(2 / 3), 1.0, 0.8)


def test_frames_with_nothing_to_find_and_nothing_reported_are_skipped() -> None:
    assert score_frame([], []) is None
    assert score_frame([(1.0, 1.0)], []) == (0, 0.0, 0.0, 0.0)
    assert score_frame([], [Molmo2Object.model_validate(A)]) == (0, 0.0, 0.0, 0.0)


def test_reward_is_f1_averaged_over_scored_frames() -> None:
    # Frame 1: one of two objects found -> P 1, R 0.5, F1 2/3. Frame 2: missed -> 0. Frame 3: skipped.
    result = score_expression(_answer((1, A_GRID), (2, [900, 900])), _metadata())

    assert result["num_scored_frames"] == 2
    assert (result["precision"], result["recall"]) == (0.5, 0.25)
    assert result["reward"] == result["f1"] == pytest.approx(1 / 3)
    assert [
        (score.frame, score.predicted, score.visible_objects, score.correct) for score in result["frame_scores"]
    ] == [
        (1, 1, 2, 1),
        (2, 1, 1, 0),
    ]


def test_a_point_reported_where_no_object_is_visible_scores_that_frame_zero() -> None:
    result = score_expression(_answer((1, A_GRID), (1, B_GRID), (2, A_GRID), (3, A_GRID)), _metadata())

    assert result["num_scored_frames"] == 3
    assert result["reward"] == pytest.approx(2 / 3)


def test_points_for_frames_that_were_not_shown_are_ignored() -> None:
    assert score_expression(_answer((1, A_GRID), (1, B_GRID), (2, A_GRID), (7, A_GRID)), _metadata())["reward"] == 1.0


def test_an_expression_never_visible_scores_one_only_when_nothing_is_reported() -> None:
    empty = _metadata(frames=[{"frame": 1, "objects": []}, {"frame": 2, "objects": []}])

    silent = score_expression('<answer>{"tracks":[]}</answer>', empty)
    assert (silent["reward"], silent["precision"], silent["recall"], silent["num_scored_frames"]) == (1.0, 1.0, 1.0, 0)
    assert score_expression(_answer((1, A_GRID)), empty)["reward"] == 0.0


def test_metrics_are_means_over_expressions_overall_and_per_category(server: Molmo2VideoTrackResourcesServer) -> None:
    def rollout(category: str | None, f1: float, parse_ok: bool = True) -> dict:
        return {"category": category, "f1": f1, "precision": f1 / 2, "recall": f1, "parse_ok": parse_ok}

    tasks = [
        [rollout("animal", 1.0)],
        [rollout("animal", 0.5)],
        [rollout("misc", 0.0, parse_ok=False)],
        [rollout(None, 0.5)],
    ]

    metrics = server.compute_metrics(tasks)

    assert metrics == {
        "f1": 0.5,
        "precision": 0.25,
        "recall": 0.5,
        "parse_fail_rate": 0.25,
        "animal/f1": 0.75,
        "animal/precision": 0.375,
        "animal/recall": 0.75,
        "animal/samples": 2,
        "misc/f1": 0.0,
        "misc/precision": 0.0,
        "misc/recall": 0.0,
        "misc/samples": 1,
    }
    assert server.compute_repeat_metrics(tasks) == metrics
    assert server.get_key_metrics({**metrics, "mean/reward": 0.5}) == {
        "f1": 0.5,
        "precision": 0.25,
        "recall": 0.5,
        "parse_fail_rate": 0.25,
        "animal/f1": 0.75,
        "misc/f1": 0.0,
    }
    assert aggregate_metrics([]) == {}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def test_verify_scores_the_assistant_message_and_ignores_reasoning(server: Molmo2VideoTrackResourcesServer) -> None:
    body = _request(EXACT, reasoning=_answer((3, A_GRID)))

    result = asyncio.run(server.verify(body))

    assert (result.reward, result.f1, result.precision, result.recall) == (1.0, 1.0, 1.0, 1.0)
    assert (result.parse_ok, result.num_scored_frames, result.category) == (True, 2, "misc")


def test_verify_handles_a_response_with_no_message(server: Molmo2VideoTrackResourcesServer) -> None:
    result = asyncio.run(server.verify(_request(reasoning="thinking only")))

    assert (result.reward, result.parse_ok) == (0.0, False)
