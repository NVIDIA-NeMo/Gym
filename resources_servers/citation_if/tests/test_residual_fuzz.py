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
# SPDX-License-Identifier: Apache-2.0
"""Residual-scan property fuzz.

Property: take N compliant answers and mutate each; ANY ID-shaped token (broad
ATTEMPT_REGEX) left outside well-formed citation markup ⇒ reward 0.
PASS bar: zero violations.

Two mutation families, so the test pins behaviour in both directions:
  LEAK   — injects a residual ID token or breaks markup ⇒ reward MUST be 0
  BENIGN — whitespace/punctuation that touches no citation ⇒ reward MUST stay 1

A BENIGN failure means the gate became over-strict (false positives on ordinary
prose); a LEAK failure means the gate leaks. Both are regressions.

Usage:
    pytest resources_servers/citation_if/tests/test_residual_fuzz.py -q
"""

import random
import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from resources_servers.citation_if.scorer import (  # noqa: E402
    ATTEMPT_REGEX,
    GRAMMAR_TEMPLATES,
    RESERVED_EXAMPLE_ID,
    score_citation_if,
)


N_SAMPLES = 1000
SEED = 20260804

TRAINED = [g for g, e in GRAMMAR_TEMPLATES.items() if not e["holdout"]]
TAG_GRAMMARS = {g for g, e in GRAMMAR_TEMPLATES.items() if "open_re" in e}
_ATTEMPT = re.compile(ATTEMPT_REGEX, re.IGNORECASE)

# Random-ID scheme, matching the r1 reference (digits 0/1 excluded so a numeric
# guess can never collide with a valid ID).
ID_ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"
ID_REGEX_FULL = r"citation_[a-z2-9]{4}"
ID_REGEX_SNIP = r"citation_[a-z2-9]{4}:snippet_[0-9]+"

PROSE = [
    "The transfer completed in late 1999.",
    "Two independent sources agree on the date.",
    "The buyer was a small development firm.",
    "Reporting at the time was inconsistent.",
]


def render(grammar, inner):
    return {
        "cite_xml": f"<cite>{inner}</cite>",
        "ascii_brackets": f"[{inner}]",
        "claim_wrap_xml": f'<cite index="{inner}">the supported claim</cite>',
        "fullwidth_brackets": f"【{inner}】",
        "ref_colon": f"[ref:{inner}]",
        "double_angle": f"<<{inner}>>",
        "web_brackets": f"[web:{inner}]",
        "paren_part": f"({inner})",
        "markdown_footnote": f"[^{inner}]",
    }[grammar]


def make_row(rng):
    """A compliant (answer, verifier) pair on a random grammar/granularity."""
    grammar = rng.choice(TRAINED)
    granularity = rng.choice(["full_source", "snippet"])
    ids = set()
    while len(ids) < 3:
        ids.add("citation_" + "".join(rng.choice(ID_ALPHABET) for _ in range(4)))
    docs = sorted(ids)
    if granularity == "snippet":
        valid = [f"{d}:snippet_{k}" for d in docs for k in (1, 2)]
        id_regex = ID_REGEX_SNIP
    else:
        valid = docs
        id_regex = ID_REGEX_FULL
    gold = valid[0]
    verifier = {
        "type": "citation_if",
        "mode": "cite",
        "grammar": grammar,
        "id_kind": granularity,
        "id_regex": id_regex,
        "valid_id_set": valid,
        "expected_ids": [gold],
        "expected_slack": 1,
        "min_valid_citations": 1,
    }
    answer = f"{rng.choice(PROSE)} {render(grammar, gold)}"
    return answer, verifier, grammar, valid


# ── Mutations ────────────────────────────────────────────────────────────────


def leak_mutations(rng, answer, grammar, valid):
    """Each returns text that MUST score 0 under the residual-scan property."""
    other = rng.choice(valid)
    out = [
        ("raw_id_appended", f"{answer} See also {other}."),
        ("sources_trailer", f"{answer}\nSources: {other}"),
        ("raw_id_prefix", f"Per {other}, {answer}"),
        ("reserved_example_mention", f"{answer} (format example: {RESERVED_EXAMPLE_ID})"),
        ("raw_hallucinated_id", f"{answer} Compare citation_zzzz."),
        ("bare_id_in_parens_prose", f"{answer} The source label was {other} throughout."),
    ]
    # Half-broken markup: drop the closing delimiter so the span no longer parses
    # and its ID falls into the residual.
    broken = render(grammar, other)
    for tail in ("]", ">", "】", ")"):
        if broken.endswith(tail):
            out.append(("unclosed_markup", f"{answer} {broken[:-1]}"))
            break
    if grammar in TAG_GRAMMARS:
        out.append(("empty_tag_added", f"{answer} <cite></cite>"))
        out.append(("stray_open_tag", f"{answer} <cite>"))
    return out


def benign_mutations(rng, answer):
    """Each returns text that MUST still score 1 (touches no citation)."""
    return [
        ("trailing_ws", answer + "   "),
        ("leading_ws", "  " + answer),
        ("internal_newlines", answer.replace(". ", ".\n\n", 1)),
        ("extra_punctuation", answer + " Indeed!"),
        ("unicode_quotes", "“" + answer + "”"),
    ]


def test_leak_mutations_all_score_zero():
    rng = random.Random(SEED)
    violations, checked = [], 0
    for _ in range(N_SAMPLES):
        answer, verifier, grammar, valid = make_row(rng)
        base, _ = score_citation_if(answer, 0, verifier)
        assert base == 1.0, f"seed row not compliant: {answer!r}"
        name, text = rng.choice(leak_mutations(rng, answer, grammar, valid))
        reward, details = score_citation_if(text, 0, verifier)
        checked += 1
        if reward != 0.0:
            violations.append((name, grammar, text, details["gate_failed"]))
    assert not violations, f"{len(violations)}/{checked} leak mutations scored non-zero. First: {violations[0]}"


def test_benign_mutations_stay_compliant():
    """Guards the other direction: the strict gate must not fire on ordinary prose."""
    rng = random.Random(SEED + 1)
    violations, checked = [], 0
    for _ in range(N_SAMPLES):
        answer, verifier, _, _ = make_row(rng)
        name, text = rng.choice(benign_mutations(rng, answer))
        reward, details = score_citation_if(text, 0, verifier)
        checked += 1
        if reward != 1.0:
            violations.append((name, text, details["gate_failed"]))
    assert not violations, (
        f"{len(violations)}/{checked} benign mutations lost reward (gate over-strict). First: {violations[0]}"
    )


def test_residual_property_holds_directly():
    """The property itself, stated independently of mutation families:
    if an ID-shaped token survives outside well-formed markup, reward is 0."""
    rng = random.Random(SEED + 2)
    for _ in range(N_SAMPLES):
        answer, verifier, grammar, valid = make_row(rng)
        name, text = rng.choice(leak_mutations(rng, answer, grammar, valid) + benign_mutations(rng, answer))
        reward, details = score_citation_if(text, 0, verifier)
        # Independent oracle: blank the spans the scorer reports, then rescan.
        residual = text
        for cid in details["parsed_ids"]:
            residual = residual.replace(cid, " ")
        if _ATTEMPT.search(residual):
            assert reward == 0.0, f"residual ID survived but reward={reward}: {text!r}"
