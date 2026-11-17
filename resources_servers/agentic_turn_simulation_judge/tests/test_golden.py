# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The copied ATS pieces must keep agreeing with the ATS judge they were copied from.

Agreement rule: on every text ATS `parse_verdict` reads without salvaging, the lenient parse returns the same dict;
on every other text (ATS returns None or a salvaged, `truncated` verdict) it returns None. A strict parse that passes
schema validation reads the same verdict as ATS.

Set AGENTIC_TURN_JUDGE_GOLDEN_RESULTS to judge result JSONL files (os.pathsep-separated; each line with a
`judge_content` string) to run the same check on stored judge outputs. Those files stay outside git.
"""

import itertools
import json
import os
import re
from pathlib import Path

import pytest

from resources_servers.agentic_turn_simulation import judge as ats
from resources_servers.agentic_turn_simulation_judge import verdict as ours


def _v(**overrides) -> dict:
    base = {
        "analysis": "The agent ran the tests and reported the one failure it saw {exit code 1}.",
        "addresses_concern": "yes",
        "correct": "yes",
        "blocked_by": "none",
        "request_questionable": "ok",
        "discard": "no",
        "discard_reason": "none",
        "reason": "Did what was asked and reported it truthfully.",
    }
    base.update(overrides)
    return base


GOOD = _v()
BAD = _v(correct="no", blocked_by="model", analysis="Announced a fix it never made.")
STEER_NO = _v(addresses_concern="no", correct="yes")
G, B = json.dumps(GOOD), json.dumps(BAD)

REPRESENTATIVE = [
    G,
    B,
    json.dumps(STEER_NO),
    json.dumps(GOOD, indent=2, ensure_ascii=False),
    f"```json\n{G}\n```",
    f"```\n{B}\n```",
    f"```json{G}```",
    f"\n\n  {G}  \n",
    f"<think>Maybe {B}? No.</think>\n{G}",
    f"<thinking>long thoughts</thinking>{B}",
    f"lost the opening tag</think>\n{G}",
    f"Here is my verdict:\n{G}\nThanks.",
    f"Verdict:\n```json\n{B}\n```",
    json.dumps({"audit": GOOD}),
    json.dumps({"x": 1, "command": BAD}),
    json.dumps({"x": {"y": 1}}),
    json.dumps(_v(addresses_concern=True, discard=False, request_questionable=["ok"], blocked_by=None)),
    json.dumps(_v(extra_key={"nested": [1, 2]})),
    json.dumps(_v(analysis="Unicode: café — 日本語 ✓")),
    json.dumps([GOOD]),
    json.dumps({"verdict": "yes"}),
    f"{G}\n{B}",
    f"{G}{B}",
    G[:-30],
    B[: len(B) // 2],
    '{"analysis": "x", "correct": "yes", "blocked_by": "none"',
    '{"addresses_concern": "yes", "correct": "yes",}',
    "",
    "   ",
    "The agent did fine. correct: yes",
    "}{",
    "{" * 2000,
    "<think>only reasoning</think>",
    json.dumps(_v(correct="maybe")),
    json.dumps({k: v for k, v in GOOD.items() if k != "reason"}),
]


def _check_agreement(text: str) -> None:
    expected = ats.parse_verdict(text)
    lenient, error = ours.parse_lenient(text)
    if expected is None or expected.get("truncated"):
        assert lenient is None and error is not None, text[:200]
    else:
        assert (lenient, error) == (expected, None), text[:200]
        _check_ladder(lenient)

    strict, _ = ours.parse_strict(text)
    if strict is not None and ours.schema_error(strict) is None:
        assert strict == expected, text[:200]
        _check_ladder(strict)


def _check_ladder(verdict: dict) -> None:
    """A verdict that passes our schema check is one ATS can score (`usable`), and both read the same ladder from
    it. ATS reads no ladder from any other verdict; ours still may (it is only paid with validate_schema off)."""
    if ours.schema_error(verdict) is None:
        assert ats.usable(verdict)
    if ats.usable(verdict):
        assert ours.correct_ladder(verdict) == ats.correct_ladder(verdict)
    else:
        assert ats.correct_ladder(verdict) is None


def test_schema_copy_matches_ats() -> None:
    copy = (Path(ours.__file__).parent / "prompts" / "evaluation_schema.json").read_text()
    original = (Path(ats.__file__).parent / "prompts" / "evaluation_schema.json").read_text()
    assert copy == original
    assert ours.evaluation_schema() == ats.evaluation_schema()


def test_copied_constants_match_ats() -> None:
    assert ours.VERDICT_FIELDS == ats.VERDICT_FIELDS
    assert ours.REQUIRED_KEY == ats.REQUIRED_KEY
    assert ours._THINK_RE.pattern == ats._THINK_RE.pattern


def test_ladder_matches_ats_on_every_combination() -> None:
    values = ["yes", "no", "maybe", None]
    usable = 0
    for addresses, correct in itertools.product(values, values):
        verdict = {k: v for k, v in _v(addresses_concern=addresses, correct=correct).items() if v is not None}
        _check_ladder(verdict)
        usable += ats.usable(verdict)
    assert usable == 4  # the yes / no combinations
    assert ours.correct_ladder(None) == ats.correct_ladder(None)


@pytest.mark.parametrize("text", REPRESENTATIVE, ids=range(len(REPRESENTATIVE)))
def test_parse_agrees_with_ats(text: str) -> None:
    _check_agreement(text)


@pytest.mark.parametrize("text", REPRESENTATIVE, ids=range(len(REPRESENTATIVE)))
def test_outermost_braces_equals_ats_regex(text: str) -> None:
    assert ours._outermost_braces(text) == re.findall(r"\{.*\}", text, re.S)


def test_salvaged_outputs_score_nothing() -> None:
    salvaged = [t for t in REPRESENTATIVE if (ats.parse_verdict(t) or {}).get("truncated")]
    assert len(salvaged) >= 3  # the representative set exercises the salvage path
    for text in salvaged:
        assert ours.read_verdict(text, parse_mode="lenient").ladder is None
        assert ours.read_verdict(text).ladder is None


def _stored_outputs() -> list[str]:
    paths = [p for p in os.environ.get("AGENTIC_TURN_JUDGE_GOLDEN_RESULTS", "").split(os.pathsep) if p]
    texts = []
    for path in paths:
        with open(path) as f:
            for line in f:
                content = json.loads(line).get("judge_content")
                if isinstance(content, str):
                    texts.append(content)
    return texts


@pytest.mark.skipif(
    not os.environ.get("AGENTIC_TURN_JUDGE_GOLDEN_RESULTS"), reason="AGENTIC_TURN_JUDGE_GOLDEN_RESULTS not set"
)
def test_parse_agrees_with_ats_on_stored_judge_outputs() -> None:
    texts = _stored_outputs()
    assert texts
    for text in texts:
        _check_agreement(text)
