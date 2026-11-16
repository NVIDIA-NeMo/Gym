# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from resources_servers.video_caption.reward import (
    compute_video_caption_reward,
    degeneration_response_format,
    normalize_atomic_events,
    parse_degeneration_judgment,
    parse_visual_events,
    parse_visual_hits,
    visual_events_response_format,
    visual_hits_response_format,
)


@pytest.mark.parametrize("tokens,length", [(99, 0), (100, 1), (4000, 1), (4001, 0)])
def test_caption_reward_formula_and_bounds(tokens, length):
    result = compute_video_caption_reward(
        "A person walks.",
        tokens,
        {"visual_events": ["walks", "sits"]},
        '{"visual_hits":[1,0]}',
        visual_reward_type="recall",
        penalize_degeneration=False,
    )
    assert result.visual_recall_reward == 0.5
    assert result.length_reward == length
    assert result.reward == (0.5 + length) / 2


def test_audio_fields_are_ignored_even_when_malformed():
    raw = {
        "visual_events": ["Walks."],
        "pure_audio_events": 3,
        "synergistic_events": "malformed",
    }
    assert normalize_atomic_events({"atomic_events": raw}) == {"visual_events": ["Walks."]}
    with pytest.raises(ValueError):
        normalize_atomic_events({"atomic_events": {"visual_events": [], "pure_audio_events": ["speech"]}})


def test_exact_schema_matches_strict_parser():
    array = visual_hits_response_format(190)["json_schema"]["schema"]["properties"]["visual_hits"]
    assert array["minItems"] == array["maxItems"] == 190
    assert parse_visual_hits('{"visual_hits":[1,0]}', 2) == [1, 0]


@pytest.mark.parametrize(
    "text",
    [
        '{"visual_hits":[1]}',
        '{"visual_hits":[1,0,1]}',
        '{"visual_hits":[1,',
        '{"visual_hits":[2,0]}',
        '{"visual_hits":[true,0]}',
        '{"visual_hits":[1.0,0]}',
        '{"visual_hits":[1,0],"extra":1}',
        '```json\n{"visual_hits":[1,0]}\n```',
    ],
)
def test_schema_violation_is_an_error_not_padded_or_truncated_reward(text):
    with pytest.raises(ValueError):
        parse_visual_hits(text, 2)


@pytest.mark.parametrize("verdict", ["pass", "fail"])
@pytest.mark.parametrize("tokens", [99, 100, 4000, 4001])
def test_degeneration_pass_fail_gates_total_reward(verdict, tokens):
    caption = "A person walks."
    result = compute_video_caption_reward(
        caption,
        tokens,
        {"visual_events": ["walks", "sits"]},
        '{"visual_hits":[1,1]}',
        visual_reward_type="recall",
        penalize_degeneration=True,
        degeneration_judge_output_text=json.dumps({"verdict": verdict}),
    )
    assert result.degeneration_reward == int(verdict == "pass")
    assert result.reward_before_degeneration == (1 + int(100 <= tokens <= 4000)) / 2
    assert result.reward == result.reward_before_degeneration * int(verdict == "pass")
    assert result.visual_recall_reward == 1
    assert result.degeneration_checked
    assert result.degeneration_detected is (verdict == "fail")
    assert result.is_valid_caption  # Nonempty output remains in ordinary RL advantage calculation.


def test_healthy_degeneration_verdict_preserves_reward_and_disabled_contract():
    args = ("A person walks.", 100, {"visual_events": ["walks", "sits"]})
    legacy = compute_video_caption_reward(
        *args, '{"visual_hits":[1,0]}', visual_reward_type="recall", penalize_degeneration=False
    )
    checked = compute_video_caption_reward(
        *args,
        '{"visual_hits":[1,0]}',
        visual_reward_type="recall",
        penalize_degeneration=True,
        degeneration_judge_output_text='{"verdict":"pass"}',
    )
    assert legacy.reward == checked.reward == checked.reward_before_degeneration == 0.75
    assert legacy.degeneration_reward == checked.degeneration_reward == 1
    assert not legacy.degeneration_checked and checked.degeneration_checked
    assert not legacy.degeneration_detected and not checked.degeneration_detected
    assert visual_hits_response_format(2)["json_schema"]["schema"]["required"] == ["visual_hits"]


@pytest.mark.parametrize(
    "value",
    [
        None,
        False,
        [],
        {},
        {"degeneration": []},
        {"visual_hits": [1, 0], "verdict": "pass"},
        {"verdict": "pass", "evidence": "anything"},
        {"verdict": "unknown"},
        {"verdict": "PASS"},
        {"verdict": ""},
        {"verdict": True},
        {"verdict": 1},
        {"verdict": ["pass"]},
    ],
)
def test_degeneration_schema_and_parser_reject_invalid_decisions(value):
    validator = Draft202012Validator(degeneration_response_format()["json_schema"]["schema"])
    with pytest.raises(ValidationError):
        validator.validate(value)
    with pytest.raises(ValueError):
        parse_degeneration_judgment(json.dumps(value))


@pytest.mark.parametrize(
    "verdict,passed",
    [("pass", True), ("fail", False)],
)
def test_degeneration_schema_and_parser_accept_pass_fail(verdict, passed):
    value = {"verdict": verdict}
    Draft202012Validator(degeneration_response_format()["json_schema"]["schema"]).validate(value)
    assert parse_degeneration_judgment(json.dumps(value)) is passed


def test_enabled_gate_cannot_silently_skip_missing_degeneration_response():
    with pytest.raises(ValueError, match="requires a separate judge response"):
        compute_video_caption_reward(
            "A person walks.",
            100,
            {"visual_events": ["walks"]},
            '{"visual_hits":[1]}',
            visual_reward_type="recall",
            penalize_degeneration=True,
        )


@pytest.mark.parametrize("mode,expected", [("recall", 0.5), ("precision", 0.25), ("f1", 1 / 3)])
@pytest.mark.parametrize("verdict", [None, "pass", "fail"])
def test_visual_score_selection_and_unchanged_length_and_gate(mode, expected, verdict):
    result = compute_video_caption_reward(
        "A person walks, sits, waves, and leaves.",
        100,
        {"visual_events": ["Walking", "Sitting"]},
        '{"visual_hits":[1,0]}' if mode != "precision" else None,
        visual_reward_type=mode,
        predicted_visual_events=["Walking", "Sitting", "Waving", "Leaving"] if mode != "recall" else None,
        precision_judge_output_text='{"visual_hits":[1,0,0,0]}' if mode != "recall" else None,
        penalize_degeneration=verdict is not None,
        degeneration_judge_output_text=json.dumps({"verdict": verdict}) if verdict else None,
    )
    assert result.visual_reward_type == mode
    assert result.visual_reward == pytest.approx(expected)
    assert result.visual_recall_reward == (0.5 if mode != "precision" else None)
    assert result.visual_precision_reward == (0.25 if mode != "recall" else None)
    assert result.visual_f1_reward == (pytest.approx(1 / 3) if mode == "f1" else None)
    assert result.length_reward == 1
    assert result.reward_before_degeneration == pytest.approx((1 + expected) / 2)
    assert result.reward == pytest.approx((1 + expected) / 2 if verdict != "fail" else 0)


@pytest.mark.parametrize("events", [[], ["Unsupported detail"]])
@pytest.mark.parametrize("mode", ["precision", "f1"])
def test_no_supported_events_has_zero_visual_score_with_existing_length_bonus(mode, events):
    result = compute_video_caption_reward(
        "Uninformative caption.",
        100,
        {"visual_events": ["Walking"]},
        '{"visual_hits":[0]}' if mode == "f1" else None,
        visual_reward_type=mode,
        predicted_visual_events=events,
        precision_judge_output_text='{"visual_hits":[0]}' if events else None,
        penalize_degeneration=False,
    )
    assert result.visual_reward == result.visual_precision_reward == 0
    assert result.visual_f1_reward == (0 if mode == "f1" else None)
    assert result.reward == 0.5 and result.length_reward == 1


@pytest.mark.parametrize("mode", ["recall", "precision", "f1"])
def test_empty_caption_has_no_visual_score_length_or_judgments(mode):
    result = compute_video_caption_reward(" \n\t", 0, {}, None, visual_reward_type=mode, penalize_degeneration=True)
    assert result.reward == result.visual_reward == result.length_reward == 0
    assert result.visual_reward_type == mode
    assert not result.is_valid_caption and not result.degeneration_checked


@pytest.mark.parametrize(
    "mode,recall,events,precision,error",
    [
        ("recall", None, None, None, "Recall requires a judge response"),
        ("precision", None, None, None, "requires extracted prediction events"),
        ("precision", None, ["Walk"], None, "requires a judge response"),
        ("precision", None, ["Walk", "Sit"], '{"visual_hits":[1]}', "exactly 2 entries"),
        ("unsupported", None, None, None, "Unknown visual reward type"),
    ],
)
def test_missing_or_invalid_visual_judgments_are_errors(mode, recall, events, precision, error):
    with pytest.raises(ValueError, match=error):
        compute_video_caption_reward(
            "Caption",
            100,
            {"visual_events": ["Walking"]},
            recall,
            visual_reward_type=mode,
            predicted_visual_events=events,
            precision_judge_output_text=precision,
            penalize_degeneration=False,
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"visual_events": None},
        {"visual_events": "Walk"},
        {"visual_events": [1]},
        {"visual_events": [{"event": "Walk"}]},
        {"visual_events": [], "pure_audio_events": []},
    ],
)
def test_extraction_schema_and_parser_reject_wrong_structure(value):
    validator = Draft202012Validator(visual_events_response_format()["json_schema"]["schema"])
    with pytest.raises(ValidationError):
        validator.validate(value)
    with pytest.raises(ValueError):
        parse_visual_events(json.dumps(value))


@pytest.mark.parametrize("text", ['{"visual_events":[" "]}', '{"visual_events":[""]}', "```json\n{}\n```"])
def test_extraction_parser_rejects_empty_event_strings_and_malformed_json(text):
    with pytest.raises(ValueError):
        parse_visual_events(text)


@pytest.mark.parametrize("events", [[], ["A person walks.", "A person sits."]])
def test_extraction_accepts_visual_only_events_and_empty_list(events):
    value = {"visual_events": events}
    Draft202012Validator(visual_events_response_format()["json_schema"]["schema"]).validate(value)
    assert parse_visual_events(json.dumps(value)) == events
    assert parse_visual_events('{"visual_events":[" Walk. "]}') == ["Walk."]
