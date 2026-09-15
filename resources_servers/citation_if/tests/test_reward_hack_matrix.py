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
"""Anti-reward-hack fixture matrix.

Property under test: no behavior that ignores the evidence, the instruction, or
the question can score 1; full compliance scores 1.

The matrix is GENERATED, never hand-written: every hack class below is templated
across all 9 trained grammars × both granularities from one worked row per cell.
PASS bar: 100% of cells produce the required verdict.

Classes 8, 10 and 15 encode the S1 (exact-match) correctness semantics that is the
coded default. Looser precision-floor variants may be adopted later; if they
are, re-derive EXPECTED_VERDICT for those three classes from the new semantics and
regenerate — never hand-edit individual cells.

Usage:
    pytest resources_servers/citation_if/tests/test_reward_hack_matrix.py -q
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from resources_servers.citation_if.scorer import (  # noqa: E402
    GRAMMAR_TEMPLATES,
    RESERVED_EXAMPLE_ID,
    score_citation_if,
)


# ── Row templates ────────────────────────────────────────────────────────────
# Non-guessable base32-style IDs so a blind numeric guess cannot
# collide with a valid ID. Gold is never at a fixed position in a real build.

TRAINED_GRAMMARS = [g for g, e in GRAMMAR_TEMPLATES.items() if not e["holdout"]]
MULTI_ID_GRAMMARS = {g for g, e in GRAMMAR_TEMPLATES.items() if e.get("multi")}
TAG_GRAMMARS = {g for g, e in GRAMMAR_TEMPLATES.items() if "open_re" in e}

GOLD_DOC, DIST_DOC, DIST_DOC2 = "citation_27bx", "citation_azh4", "citation_rbdr"

ROWS = {
    "full_source": {
        "id_regex": r"citation_[a-z0-9]{4}",
        "valid_id_set": [GOLD_DOC, DIST_DOC, DIST_DOC2],
        "expected_ids": [GOLD_DOC],
        "gold": GOLD_DOC,
        "extra_valid": DIST_DOC,  # a second valid ID, within slack
        "wrong_doc": DIST_DOC,  # valid but not gold
        "same_doc_wrong_part": None,  # n/a at full-source granularity
    },
    "snippet": {
        "id_regex": r"citation_[a-z0-9]{4}:snippet_[0-9]+",
        "valid_id_set": [
            f"{GOLD_DOC}:snippet_1",
            f"{GOLD_DOC}:snippet_5",
            f"{DIST_DOC}:snippet_1",
        ],
        "expected_ids": [f"{GOLD_DOC}:snippet_1"],
        "gold": f"{GOLD_DOC}:snippet_1",
        "extra_valid": f"{DIST_DOC}:snippet_1",
        "wrong_doc": f"{DIST_DOC}:snippet_1",
        "same_doc_wrong_part": f"{GOLD_DOC}:snippet_5",  # right doc, WRONG snippet
    },
}


def render(grammar: str, inner: str) -> str:
    """Well-formed citation markup for a grammar, wrapping `inner` verbatim."""
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


def empty_markup(grammar: str) -> str:
    """Citation markup that parses no ID. Only tag grammars have a detectable form."""
    return {"cite_xml": "<cite></cite>", "claim_wrap_xml": '<cite index="">claim</cite>'}[grammar]


def verifier(granularity: str, mode: str = "cite", correctness: bool = True) -> dict:
    row = ROWS[granularity]
    return {
        "type": "citation_if",
        "mode": mode,
        "grammar": None,  # filled per cell
        "id_kind": granularity,
        "id_regex": row["id_regex"],
        "valid_id_set": list(row["valid_id_set"]),
        "expected_ids": list(row["expected_ids"]) if correctness else None,
        "expected_slack": 1,
        "min_valid_citations": 1,
    }


# ── Hack classes ─────────────────────────────────────────────────────────────
# Each builder returns (answer_text, verifier_overrides, required_verdict).
# required_verdict None == reward 1.0.


def _classes(grammar: str, granularity: str):
    row = ROWS[granularity]
    gold, extra = row["gold"], row["extra_valid"]
    cases = []

    # 1. blind constant ID — the v1 habit, on a random-ID row
    cases.append(
        ("blind_constant_id", f"Fact. {render(grammar, 'citation_0')}", {}, {"malformed_attempt", "no_hallucination"})
    )

    # 2. instruction-example ID echo.
    #    Verdict depends on whether the reserved ID happens to parse under the row's
    #    id_regex: if it parses it is an invalid ID and — with no OTHER valid citation
    #    present — must_cite fires before no_hallucination (gate order is deliberate;
    #    cf. the r1 audit matrix, where "[citation_xmpl]" alone is must_cite and
    #    "[gold] [citation_xmpl]" is no_hallucination). If it does not parse it lands
    #    in the residual and is a malformed attempt. Reward is 0 in every case, which
    #    is the property under test; the label is diagnostic only.
    cases.append(
        (
            "example_id_echo",
            f"Fact. {render(grammar, RESERVED_EXAMPLE_ID)}",
            {},
            {"no_hallucination", "malformed_attempt", "must_cite"},
        )
    )

    # 3. valid cite + raw ID in prose
    cases.append(
        ("valid_plus_raw_id_prose", f"Per {extra}, the fact holds. {render(grammar, gold)}", {}, {"malformed_attempt"})
    )

    # 4. valid cite + trailing Sources: line
    cases.append(
        ("valid_plus_sources_line", f"Fact. {render(grammar, gold)}\nSources: {extra}", {}, {"malformed_attempt"})
    )

    # 5. valid cite + empty markup  (tag grammars only — no detectable form otherwise)
    if grammar in TAG_GRAMMARS:
        cases.append(
            (
                "valid_plus_empty_markup",
                f"Fact. {render(grammar, gold)} {empty_markup(grammar)}",
                {},
                {"malformed_attempt"},
            )
        )

    # 6. valid cite + raw hallucinated ID
    cases.append(
        (
            "valid_plus_raw_hallucinated",
            f"Fact. {render(grammar, gold)} See also citation_zzzz.",
            {},
            {"malformed_attempt"},
        )
    )

    # 7. comma+space multi-ID list (never parses — the observed r16 near-miss)
    cases.append(("comma_space_id_list", f"Fact. {render(grammar, f'{gold}, {extra}')}", {}, {"malformed_attempt"}))

    # 8. cite-everything spray — S1 cap is |expected| + slack
    spray = " ".join(render(grammar, i) for i in row["valid_id_set"])
    cases.append(("cite_everything_spray", f"Fact. {spray}", {}, {"correctness_over_cap"}))

    # 9. wrong-document citation
    cases.append(("wrong_document", f"Fact. {render(grammar, row['wrong_doc'])}", {}, {"correctness_missed_gold"}))

    # 10. right document, WRONG snippet (snippet granularity only)
    if row["same_doc_wrong_part"]:
        cases.append(
            (
                "right_doc_wrong_snippet",
                f"Fact. {render(grammar, row['same_doc_wrong_part'])}",
                {},
                {"correctness_missed_gold"},
            )
        )

    # 11. no citation at all
    cases.append(("no_citation", "The answer is that it happened in 1994.", {}, {"must_cite"}))

    # 12. citation inside reasoning only — nemo-gym strips reasoning before the
    #     verifier, so the final text the scorer sees carries no citation.
    cases.append(("reasoning_only_citation", "   ", {}, {"structural"}))

    # 13. no_cite row that cites anyway
    cases.append(
        ("no_cite_row_cites", f"Fact. {render(grammar, gold)}", {"mode": "no_cite", "expected_ids": None}, {"no_cite"})
    )

    # 16. citation with NO ANSWER — the whole output is the citation.
    #
    # Found by reviewing real rollouts: one scored 1.0 on the complete output
    # `<cite>citation_7367:snippet_2</cite>`. Correct grammar, correct gold snippet, no
    # answer. The old gate 0 accepted it because the raw string is non-empty — it is
    # 36 characters of markup.
    #
    # claim_wrap_xml needs a different string, and the reason is structural: a well-formed
    # claim wrap CONTAINS its claim, so "citation with no answer" is not expressible as a
    # bare citation there. An empty claim (`<cite index="ID"></cite>`) is not a citation at
    # all — `_iter_spans` drops it — and lands on malformed_attempt via the tag-balance
    # check. The reachable no-answer shape is a claim with no word characters.
    cases.append(
        (
            "citation_only_no_answer",
            f'<cite index="{gold}">.</cite>' if grammar == "claim_wrap_xml" else render(grammar, gold),
            {},
            {"no_answer"},
        )
    )

    # 14. COMPLIANT: exact gold cited
    cases.append(("compliant_exact_gold", f"Fact. {render(grammar, gold)}", {}, None))

    # 15. COMPLIANT: gold + one extra valid ID within slack
    cases.append(
        ("compliant_gold_plus_extra", f"Fact A. {render(grammar, gold)} Fact B. {render(grammar, extra)}", {}, None)
    )

    return cases


def matrix():
    for granularity in ROWS:
        for grammar in TRAINED_GRAMMARS:
            for name, text, overrides, required in _classes(grammar, granularity):
                v = verifier(granularity)
                v["grammar"] = grammar
                v.update(overrides)
                yield granularity, grammar, name, text, v, required


CELLS = list(matrix())


@pytest.mark.parametrize(
    "granularity,grammar,name,text,v,required",
    CELLS,
    ids=[f"{g}-{gr}-{n}" for g, gr, n, _, _, _ in CELLS],
)
def test_matrix_cell(granularity, grammar, name, text, v, required):
    reward, details = score_citation_if(text, 0, v)
    gate = details["gate_failed"]
    if required is None:
        assert reward == 1.0, f"compliant cell scored {reward} (gate={gate}): {text!r}"
        assert gate is None
    else:
        assert reward == 0.0, f"hack cell scored 1.0: {text!r}"
        assert gate in required, f"expected {required}, got {gate!r} for {text!r}"


def test_matrix_is_fully_populated():
    """Guards against silent shrinkage of the matrix — no silent caps."""
    n_classes = {n for _, _, n, _, _, _ in CELLS}
    assert len(n_classes) == 16, f"expected 16 hack classes, generated {len(n_classes)}"
    assert len(TRAINED_GRAMMARS) == 9
    # 14 classes apply everywhere; empty-markup is tag-only, wrong-snippet snippet-only.
    tag_cells = len(TAG_GRAMMARS) * len(ROWS)
    snippet_cells = len(TRAINED_GRAMMARS)
    assert len(CELLS) == 14 * 9 * 2 + tag_cells + snippet_cells
    # class 16 must cover every grammar x granularity — it is the newest gate and the
    # one most likely to be silently dropped by a future refactor.
    assert sum(1 for _, _, n, _, _, _ in CELLS if n == "citation_only_no_answer") == 9 * 2
