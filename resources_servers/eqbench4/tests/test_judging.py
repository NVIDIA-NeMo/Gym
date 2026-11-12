# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from resources_servers.eqbench4.judging import (
    PromptParts,
    combine_readings,
    pairwise_transcript,
    parse_dimension_verdict,
    parse_json_object,
    parse_trait_scores,
    persona_internal,
    respondent_codes,
    strip_thinking,
    trait_transcript,
    weighted_margins,
)


WEIGHTS = {"rapport": 1.0, "attunement": 1.0, "sensemaking": 0.5}


def test_prompt_parts_fill_repeated_slots() -> None:
    parts = PromptParts(["Hi ", ", meet ", ". Bye ", "."], ["a", "b", "a"])
    assert parts.fill({"a": "Ann", "b": "Bo"}) == "Hi Ann, meet Bo. Bye Ann."


def test_prompt_parts_rejects_bad_shapes_and_missing_values() -> None:
    with pytest.raises(ValueError, match="one more literal"):
        PromptParts(["only"], ["slot"])
    with pytest.raises(KeyError, match="b"):
        PromptParts(["", " and ", ""], ["a", "b"]).fill({"a": "x"})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>plan</think>Hello", "Hello"),
        ("<THINKING>a\nb</THINKING> Hi <reasoning>x</reasoning>there", "Hi there"),
        ("leaked plan</think>\nAnswer", "Answer"),
        ("Answer<think>cut off before closing", "Answer"),
        ("<thinking>never finished", ""),
        ("  plain  ", "plain"),
    ],
)
def test_strip_thinking(text: str, expected: str) -> None:
    assert strip_thinking(text) == expected


def test_persona_internal_sources() -> None:
    assert persona_internal({"internal": "given"}) == "given"
    assert persona_internal({"raw_response": "<INTERNAL>\n  feels seen \n</internal><dialogue>x</dialogue>"}) == (
        "feels seen"
    )
    assert persona_internal({"raw_response": "[INTERNAL] bracketed [/INTERNAL]"}) == "bracketed"
    assert persona_internal({"raw_response": "<dialogue>no monologue</dialogue>"}) is None
    assert persona_internal({"content": "no raw"}) is None


def test_pairwise_transcript_layout() -> None:
    messages = [
        {"role": "assistant", "content": "Hello there."},
        {"role": "user", "content": "Hi.", "internal": "Wary."},
        {"role": "assistant", "content": "How are you?"},
        {"role": "user", "content": "Fine.", "raw_response": "<dialogue>Fine.</dialogue>"},
    ]
    assert pairwise_transcript(messages) == (
        "---\nAssistant:\nHello there.\n"
        "---\nUser (internal monologue):\n<internal>\nWary.\n</internal>\n"
        "---\nUser:\nHi.\n"
        "---\nAssistant:\nHow are you?\n"
        "---\nUser:\nFine."
    )


def test_trait_transcript_strips_state_blocks() -> None:
    messages = [
        {"role": "assistant", "content": " Hi. "},
        {
            "role": "user",
            "content": "ignored",
            "raw_response": "<internal>x</internal>\n<state>{}</state>\n\n\n[STATE]y[/STATE]<dialogue>Hey</dialogue>",
        },
        {"role": "user", "content": "plain reply"},
    ]
    assert trait_transcript(messages, "Chatbot") == (
        "Chatbot: Hi.\n\nPerson: <internal>x</internal>\n\n<dialogue>Hey</dialogue>\n\nPerson: plain reply\n\n"
    )


def test_respondent_codes_are_seeded_distinct_and_well_formed() -> None:
    first, second = respondent_codes("scenario|model|policy_first")
    assert (first, second) == respondent_codes("scenario|model|policy_first")
    assert first != second
    for code in (first, second):
        assert len(code) == 5 and code[0].isupper() and code[1:].isdigit()
    assert respondent_codes("another seed") != (first, second)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"rapport": "tie"}', {"rapport": "tie"}),
        ('Thoughts first.\n```json\n{"rapport": "A1234+"}\n```', {"rapport": "A1234+"}),
        ('preamble {"rapport": "B0001++"} trailer', {"rapport": "B0001++"}),
        ("no json here", {}),
        ("[1, 2]", {}),
        ("{broken", {}),
    ],
)
def test_parse_json_object(text: str, expected: dict) -> None:
    assert parse_json_object(text) == expected


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("A1111+++", (0, 3)),
        (" B2222+ ", (1, 1)),
        ("TIE", (None, 0)),
        ("C3333++", (None, 0)),
        ("A1111", (0, 0)),
    ],
)
def test_parse_dimension_verdict(verdict: str, expected: tuple) -> None:
    assert parse_dimension_verdict(verdict, "A1111", "B2222") == expected


def test_weighted_margins_apply_weights_and_skip_non_strings() -> None:
    verdicts = {"rapport": "A1111+++", "attunement": "B2222++", "sensemaking": "A1111++++", "extra": "A1111+++++"}
    assert weighted_margins(verdicts, "A1111", "B2222", WEIGHTS) == (5.0, 2.0)
    assert weighted_margins({"rapport": 3, "attunement": None}, "A1111", "B2222", WEIGHTS) == (0.0, 0.0)


def test_combine_readings_turns_margins_into_fractions() -> None:
    largest = 5 * sum(WEIGHTS.values())
    win = combine_readings([(5.0, 2.0), (4.0, 1.0)], WEIGHTS)
    assert (win.policy_margin, win.opponent_margin) == (4.5, 1.5)
    assert win.fraction == pytest.approx(0.5 + 3.0 / largest / 2)
    loss = combine_readings([(0.0, 3.0)], WEIGHTS)
    assert loss.fraction == pytest.approx(0.5 - 3.0 / largest / 2)
    assert combine_readings([(2.0, 2.0)], WEIGHTS).fraction == 0.5
    assert combine_readings([(99.0, 0.0)], WEIGHTS).fraction == 1.0
    assert combine_readings([(1.0, 0.0)], {}).fraction == 0.5
    with pytest.raises(ValueError, match="at least one"):
        combine_readings([], WEIGHTS)


def test_parse_trait_scores() -> None:
    assert parse_trait_scores(
        '[ANALYSIS]x[/ANALYSIS]\n[SCORES]\n```json\n{"warmth": 7, "challenge": "3"}\n```\n[/SCORES]'
    ) == {
        "warmth": 7.0,
        "challenge": 3.0,
    }
    # An unreadable [SCORES] block falls back to the first flat JSON object, keeping its numeric values only.
    assert parse_trait_scores('[SCORES]not json[/SCORES] then {"warmth": 6, "note": "x", "flag": true}') == {
        "warmth": 6.0
    }
    assert parse_trait_scores('[SCORES]{"warmth": "high"}[/SCORES]') is None
    assert parse_trait_scores('{"note": "only text"}') is None
    assert parse_trait_scores("{not json}") is None
    assert parse_trait_scores("nothing") is None
