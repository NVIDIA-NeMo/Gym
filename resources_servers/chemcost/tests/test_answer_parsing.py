# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""answer_parsing.parse_final_answer against the pinned upstream ReActAgent._try_parse_final_answer."""

import json
import random
import time
from pathlib import Path

import pytest
from chemcost.baselines.react_agent import ReActAgent

from resources_servers.chemcost.answer_parsing import parse_final_answer


ANSWER = '{"predicted_cost_per_gram": 12.5, "predicted_components": [{"name": "a", "price_per_gram": 1.0}]}'
TEXTS = [
    "",
    "no answer here",
    ANSWER,
    "Working {draft}\n" + ANSWER,
    ANSWER + "\nOn reflection:\n" + ANSWER.replace("12.5", "25"),
    '{"predicted_cost_per_gram": null, "predicted_components": []}',
    '{"outer": {"predicted_cost_per_gram": 3}, "predicted_cost_per_gram": 4}',
    '{"x": 1} {"predicted_cost_per_gram": 7',
    "{ { " + ANSWER,
    "} } " + ANSWER + " }",
    '{"note": "a } inside a string", "predicted_cost_per_gram": 9}',
    '{"predicted_cost_per_gram": "12"}',
    "[1, 2, 3]",
    '{"predicted_cost_per_gram": 5} trailing {"a": {"predicted_cost_per_gram": 6}}',
    "Cost per gram: $1,234.56",
    "cost per gram of step 1 product = 3; predicted_cost_per_gram = 41.2",
    "Per-gram cost (USD): 2e3",
    '{"predicted_cost_per_gram": 1e400}',
    '{"predicted_cost_per_gram": ' + "9" * 309 + "}",
    # A long component list puts the key far from the end of the object.
    '{"predicted_cost_per_gram": 77, "predicted_components": ['
    + ", ".join(f'{{"name": "c{i}", "price_per_gram": {i}}}' for i in range(500))
    + "]}",
]


@pytest.mark.parametrize("text", TEXTS)
def test_matches_upstream_on_crafted_answers(text: str) -> None:
    assert parse_final_answer(text) == ReActAgent._try_parse_final_answer(text)


def test_matches_upstream_on_random_brace_soup() -> None:
    rng = random.Random(0)
    pieces = ["{", "}", '"predicted_cost_per_gram"', ": ", "12", ", ", '"x"', "[", "]", "cost per gram = 3", " "]
    for _ in range(2000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 30)))
        assert parse_final_answer(text) == ReActAgent._try_parse_final_answer(text), text


def test_possessive_fallback_pattern_matches_upstream() -> None:
    rng = random.Random(1)
    heads = ["cost per gram", "Per-gram cost", "predicted_cost_per_gram", "cost  per\ngram"]
    pieces = [
        " ",
        "\t",
        "\n",
        "*",
        '"',
        "(USD)",
        "(in $/g, approx)",
        ":",
        "=",
        "USD",
        "$",
        "12.5",
        "1,234",
        "2e3",
        "x",
    ]
    for _ in range(3000):
        text = rng.choice(heads) + "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
        assert parse_final_answer(text) == ReActAgent._try_parse_final_answer(text), repr(text)


def test_matches_upstream_on_saved_example_answers() -> None:
    path = Path(__file__).parents[1] / "data" / "example_rollouts.jsonl"
    for line in path.read_text().splitlines():
        for item in json.loads(line)["response"]["output"]:
            if item["type"] == "message":
                text = "".join(part.get("text", "") for part in item["content"])
                assert parse_final_answer(text) == ReActAgent._try_parse_final_answer(text)


@pytest.mark.parametrize(
    "prefix",
    [
        "cost per gram",
        "cost per gram:",
        "cost per gram: USD",
        "cost per gram: $",
        "cost per gram = USD $",
        "Per-gram cost (x)",
    ],
)
def test_whitespace_runs_do_not_backtrack(prefix: str) -> None:
    started = time.monotonic()
    assert parse_final_answer(prefix + " " * 200_000 + "x") is None
    assert parse_final_answer(prefix + " " * 200_000 + "x") == ReActAgent._try_parse_final_answer(prefix + "   x")
    assert time.monotonic() - started < 5


def test_long_whitespace_still_reaches_the_figure() -> None:
    text = "cost per gram:" + " " * 100_000 + "USD" + " " * 100_000 + "$" + " " * 100_000 + "7"
    assert parse_final_answer(text)["predicted_cost_per_gram"] == 7.0


def test_balanced_invalid_nesting_is_bounded() -> None:
    # Every candidate holds the key and is valid JSON up to its last character.
    depth = 4000
    text = '{"a": ' * depth + '{"predicted_cost_per_gram": 1 x}' + " x}" * depth
    started = time.monotonic()
    assert parse_final_answer(text) is None
    assert time.monotonic() - started < 5


def test_a_refused_candidate_gives_no_answer_and_never_a_draft() -> None:
    # Upstream's parser raises on these and its evaluator records a null prediction.
    huge = '{"predicted_cost_per_gram": ' + "9" * 5000 + "}"
    assert parse_final_answer(huge) is None
    assert parse_final_answer(ANSWER + "\nFinal:\n" + huge) is None
    assert parse_final_answer(huge + "\nFinal:\n" + ANSWER) is None
    with pytest.raises(ValueError):
        ReActAgent._try_parse_final_answer(ANSWER + "\nFinal:\n" + huge)


def test_brace_flood_is_linear() -> None:
    started = time.monotonic()
    assert parse_final_answer("{" * 200_000 + ANSWER)["predicted_cost_per_gram"] == 12.5
    assert parse_final_answer("{" * 200_000) is None
    assert time.monotonic() - started < 5
