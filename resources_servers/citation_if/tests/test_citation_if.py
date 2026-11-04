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
"""Hack-fixture test suite for the citation_if reward checker.

Covers every known failure mode (must score 0) and fully-compliant cases
(must score 1). Run on every server change as a regression gate.

Usage:
    pytest resources_servers/citation_if/tests/test_citation_if.py -v
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from resources_servers.citation_if.scorer import score_citation_if


VALID_IDS = ["citation_0", "citation_1", "citation_2"]
EXPECTED_IDS = ["citation_0", "citation_1"]
# Larger valid set so a spray of all IDs exceeds the correctness cap (|expected|+slack).
SPRAY_VALID_IDS = ["citation_0", "citation_1", "citation_2", "citation_3", "citation_4"]

BASE_VERIFIER = {
    "type": "citation_if",
    "mode": "cite",
    "grammar": "cite_xml",
    "id_kind": "full_source",
    "id_regex": "citation_[0-9]+",
    "valid_id_set": VALID_IDS,
    "expected_ids": None,
    "min_valid_citations": 1,
}


def v(**kwargs):
    """Build a verifier dict from BASE_VERIFIER with overrides."""
    return {**BASE_VERIFIER, **kwargs}


def score(text, function_call_count=0, verifier=None):
    verifier = verifier or BASE_VERIFIER
    reward, details = score_citation_if(text, function_call_count, verifier)
    return reward, details


# ── Gate 0: Structural ────────────────────────────────────────────────────────


def test_empty_text_scores_zero():
    reward, d = score("")
    assert reward == 0.0
    assert d["gate_failed"] == "structural"


def test_tool_call_scores_zero():
    reward, d = score("Claim. <cite>citation_0</cite>", function_call_count=1)
    assert reward == 0.0
    assert d["gate_failed"] == "structural"


def test_whitespace_only_scores_zero():
    reward, d = score("   \n  ")
    assert reward == 0.0
    assert d["gate_failed"] == "structural"


# ── Gate 0b: Answer presence, and the do-not-grade-content boundary ───────────
#
# These two tests are a PAIR and must be read together. Item 5 says the scorer never
# grades answer text; gate 0b says an answer must exist. Before 2026-08-05 only the first
# property was intended and neither was tested, and the gap between them was a live reward
# hole: a pilot rollout scored 1.0 on the complete output `<cite>citation_..</cite>`.
#
# Keeping both tests adjacent is deliberate. Someone "fixing" a future empty-answer report
# by making the scorer inspect answer content would break the first test; someone
# simplifying gate 0b away would break the second.


def test_item5_scorer_does_not_grade_answer_quality():
    """Item 5: a WRONG, irrelevant answer with the correct citation still scores 1.0.

    The reward judges citation behaviour, not truth. It could not do otherwise — the data
    carries no gold answer (upstream verifiers hold reference markers like `[ref:2]`).
    """
    reward, _ = score(
        "Bananas are a kind of fish and the moon is square. <cite>citation_0</cite>",
        verifier=v(expected_ids=["citation_0"]),
    )
    assert reward == 1.0


def test_citation_with_no_answer_scores_zero():
    """Gate 0b: the whole output being a citation is not an answer.

    Regression test from a real rollout. Note gate 0 passes here — the raw string is
    not empty, it is citation markup — which is exactly why a separate gate was needed.
    """
    reward, d = score("<cite>citation_0</cite>", verifier=v(expected_ids=["citation_0"]))
    assert reward == 0.0
    assert d["gate_failed"] == "no_answer"
    assert d["structural_ok"] is True, "gate 0 must pass; the text is non-empty markup"


def test_one_word_answer_with_citation_scores_one():
    """Gate 0b is a PRESENCE test, not a length floor.

    Short answers are correct answers for this pool: measured over the pilot, the real
    answer content has a median of 5 words. A floor would reject them as soon as the
    policy sheds its SFT boilerplate.
    """
    reward, _ = score("Roosterville. <cite>citation_0</cite>", verifier=v(expected_ids=["citation_0"]))
    assert reward == 1.0


def test_claim_wrap_answer_inside_span_counts_as_answer():
    """Gate 0b must not use `_strip_spans`: claim_wrap_xml holds its answer INSIDE the span."""
    reward, _ = score(
        '<cite index="citation_0">The Sun is 9.731 Jupiter radii.</cite>',
        verifier=v(grammar="claim_wrap_xml", expected_ids=["citation_0"]),
    )
    assert reward == 1.0


# ── Gate 1: Malformed-attempt ─────────────────────────────────────────────────


def test_raw_id_in_prose_scores_zero():
    """citation_N appears in prose with no wrapper -> malformed attempt."""
    reward, d = score("citation_0 says the earth is round.")
    assert reward == 0.0
    assert d["gate_failed"] == "malformed_attempt"
    assert d["bare_id_hits"] is True


def test_opened_tag_not_closed_scores_zero():
    """<cite>citation_0 — opened but not closed -> bare id_regex hit, grammar parse fails."""
    reward, d = score("Claim. <cite>citation_0")
    assert reward == 0.0
    assert d["gate_failed"] == "malformed_attempt"


def test_wrong_grammar_format_scores_zero():
    """Model uses ascii_brackets but grammar expects cite_xml."""
    reward, d = score("Claim [citation_0].")
    assert reward == 0.0
    assert d["gate_failed"] == "malformed_attempt"


# ── Gate 2: Must-cite ─────────────────────────────────────────────────────────


def test_uncited_prose_scores_zero():
    """Non-empty text, no citations at all."""
    reward, d = score("The answer is that the sky is blue.")
    assert reward == 0.0
    assert d["gate_failed"] == "must_cite"
    assert d["bare_id_hits"] is False


def test_empty_cite_tag_scores_zero():
    """<cite></cite> — citation markup that parses no ID.

    Markup with no valid ID inside is a malformed ATTEMPT, not a
    silent non-citation. The tag-balance check (open == parsed == close) fires.
    """
    reward, d = score("Claim. <cite></cite>")
    assert reward == 0.0
    assert d["gate_failed"] == "malformed_attempt"


def test_whitespace_cite_tag_scores_zero():
    """<cite>  </cite> — only whitespace inside tag (malformed attempt, see above)."""
    reward, d = score("Claim. <cite>   </cite>")
    assert reward == 0.0
    assert d["gate_failed"] == "malformed_attempt"


# ── Gate 3: No-hallucination ──────────────────────────────────────────────────


def test_hallucinated_id_scores_zero():
    """citation_99 is not in valid_id_set.

    When the ONLY parsed citation is hallucinated, valid_cited is empty so
    must_cite fires before no_hallucination. The mixed case (valid + hallucinated)
    tests gate 3 directly — see test_mixed_valid_and_hallucinated_scores_zero.
    """
    reward, d = score("Claim. <cite>citation_99</cite>")
    assert reward == 0.0
    assert d["gate_failed"] == "must_cite"
    assert "citation_99" in d["parsed_ids"]


def test_mixed_valid_and_hallucinated_scores_zero():
    """One valid citation + one hallucinated — gate 3 fires."""
    reward, d = score("Claim A. <cite>citation_0</cite> Claim B. <cite>citation_99</cite>")
    assert reward == 0.0
    assert d["gate_failed"] == "no_hallucination"


# ── Gate 4: Correctness (when expected_ids set) ──────────────────────────────


def test_correct_citation_with_expected_ids_scores_one():
    """Cites all expected IDs -> reward 1."""
    reward, _ = score(
        "Claim A. <cite>citation_0</cite> Claim B. <cite>citation_1</cite>",
        verifier=v(expected_ids=EXPECTED_IDS),
    )
    assert reward == 1.0


def test_one_missed_expected_id_scores_zero():
    """Proposal §5: every gold must be cited (subset). Missing 1 of 2 -> gate 4 fires.

    (Previously this was allowed under a miss-tolerant reading of expected_slack;
    slack is over-citation slack only, never permission to miss a required id.)
    """
    reward, d = score(
        "Claim A. <cite>citation_0</cite>",
        verifier=v(expected_ids=EXPECTED_IDS),
    )
    assert reward == 0.0
    assert d["gate_failed"] == "correctness_missed_gold"
    assert d["missed_ids"] == ["citation_1"]
    assert d["over_cap"] is False


def test_both_expected_ids_missed_scores_zero():
    """Misses both expected IDs (cites irrelevant citation_2) -> gate 4 fires."""
    reward, d = score(
        "Claim. <cite>citation_2</cite>",
        verifier=v(expected_ids=EXPECTED_IDS),
    )
    assert reward == 0.0
    assert d["gate_failed"] == "correctness_missed_gold"
    assert len(d["missed_ids"]) == 2


def test_wrong_source_cited_scores_zero():
    """Cites the wrong source (citation_2 only, expected citation_0 and citation_1)."""
    reward, d = score(
        "Claim. <cite>citation_2</cite>",
        verifier=v(expected_ids=["citation_0", "citation_1"]),
    )
    assert reward == 0.0
    assert d["gate_failed"] == "correctness_missed_gold"


def test_all_golds_plus_one_extra_within_cap_scores_one():
    """All golds cited + 1 extra valid ID — within slack cap (|cited| = |expected|+1) -> reward 1."""
    reward, d = score(
        "A. <cite>citation_0</cite> B. <cite>citation_1</cite> C. <cite>citation_2</cite>",
        verifier=v(expected_ids=EXPECTED_IDS, valid_id_set=SPRAY_VALID_IDS),
    )
    assert reward == 1.0
    assert d["over_cap"] is False


def test_spray_all_valid_ids_over_cap_scores_zero():
    """Cites all golds but sprays every valid ID — exceeds |expected|+slack -> gate 4 fires.

    Blocks the 'cite-everything' hack: citing the whole valid set to guarantee coverage.
    valid set = 5 IDs, expected = 2, slack = 1 -> cap = 3; spray of 5 > 3 -> 0.
    """
    reward, d = score(
        "A. <cite>citation_0</cite> B. <cite>citation_1</cite> C. <cite>citation_2</cite> "
        "D. <cite>citation_3</cite> E. <cite>citation_4</cite>",
        verifier=v(expected_ids=EXPECTED_IDS, valid_id_set=SPRAY_VALID_IDS),
    )
    assert reward == 0.0
    assert d["gate_failed"] == "correctness_over_cap"
    assert d["over_cap"] is True
    assert d["missed_ids"] == []


def test_spray_all_valid_ids_with_null_expected_scores_one():
    """When expected_ids is null, the correctness gate (which holds the anti-spray cap
    |valid_cited| <= |expected|+slack) does not run — so citing every valid ID scores 1.
    This is intentional: with no expected set, the reward only checks that citations are
    well-formed and valid, never how many. The over-cap counterpart, where expected_ids IS
    set and a spray exceeds the cap, is test_spray_all_valid_ids_over_cap_scores_zero.

    Lock: a failure here means an anti-spray cap was added to the null-expected path — a
    deliberate semantics change, not a bugfix.
    """
    spray = "Claim " + "".join(f"<cite>{c}</cite>" for c in SPRAY_VALID_IDS) + "."
    reward, d = score(spray, verifier=v(expected_ids=None, valid_id_set=SPRAY_VALID_IDS))
    assert reward == 1.0
    assert d["gate_failed"] is None


def test_expected_slack_zero_forbids_any_extra():
    """expected_slack=0 -> exact-set: any extra valid ID over the golds fails."""
    reward, d = score(
        "A. <cite>citation_0</cite> B. <cite>citation_1</cite> C. <cite>citation_2</cite>",
        verifier=v(expected_ids=EXPECTED_IDS, valid_id_set=SPRAY_VALID_IDS, expected_slack=0),
    )
    assert reward == 0.0
    assert d["gate_failed"] == "correctness_over_cap"
    assert d["over_cap"] is True


# ── No-cite mode ──────────────────────────────────────────────────────────────


def test_no_cite_mode_clean_answer_scores_one():
    """No-cite mode: answer with no citation tokens -> reward 1."""
    reward, _ = score("The sky is blue.", verifier=v(mode="no_cite"))
    assert reward == 1.0


def test_no_cite_mode_with_citation_scores_zero():
    """No-cite mode: response contains a citation -> reward 0."""
    reward, d = score("Claim. <cite>citation_0</cite>", verifier=v(mode="no_cite"))
    assert reward == 0.0
    assert d["gate_failed"] == "no_cite"


def test_no_cite_mode_bare_id_scores_zero():
    """No-cite mode: bare citation_N in prose -> reward 0."""
    reward, d = score("citation_0 says so.", verifier=v(mode="no_cite"))
    assert reward == 0.0
    assert d["gate_failed"] == "no_cite"


# ── Compliant cases — must score 1 ───────────────────────────────────────────


def test_single_valid_citation_cite_xml():
    reward, _ = score("The answer is X. <cite>citation_0</cite>")
    assert reward == 1.0


def test_multiple_valid_citations_cite_xml():
    reward, _ = score("Claim A. <cite>citation_0</cite> Claim B. <cite>citation_1</cite>")
    assert reward == 1.0


def test_multi_id_in_single_tag_cite_xml():
    """<cite>citation_0,citation_1</cite> — comma-separated multi-ID."""
    reward, d = score("Claim. <cite>citation_0,citation_1</cite>")
    assert reward == 1.0
    assert set(d["valid_cited"]) == {"citation_0", "citation_1"}


def test_ascii_brackets_grammar():
    reward, _ = score(
        "Claim [citation_0].",
        verifier=v(grammar="ascii_brackets"),
    )
    assert reward == 1.0


def test_ascii_brackets_no_markdown_link_collision():
    """[citation_0](https://...) must NOT parse as a citation."""
    reward, d = score(
        "Claim [citation_0](https://example.com).",
        verifier=v(grammar="ascii_brackets"),
    )
    assert reward == 0.0
    assert d["gate_failed"] in ("malformed_attempt", "must_cite")


def test_claim_wrap_xml_valid():
    reward, _ = score(
        '<cite index="citation_0">The claim text here.</cite>',
        verifier=v(grammar="claim_wrap_xml"),
    )
    assert reward == 1.0


def test_claim_wrap_xml_empty_content_scores_zero():
    """<cite index="citation_0"> </cite> — whitespace-only content -> does not parse."""
    reward, d = score(
        '<cite index="citation_0">  </cite>',
        verifier=v(grammar="claim_wrap_xml"),
    )
    assert reward == 0.0


def test_fullwidth_brackets_grammar():
    reward, _ = score(
        "Claim 【citation_1】.",
        verifier=v(grammar="fullwidth_brackets"),
    )
    assert reward == 1.0


def test_ref_colon_grammar():
    reward, _ = score("Claim [ref:citation_0].", verifier=v(grammar="ref_colon"))
    assert reward == 1.0


def test_double_angle_grammar():
    reward, _ = score("Claim <<citation_0>>.", verifier=v(grammar="double_angle"))
    assert reward == 1.0


def test_web_brackets_grammar():
    reward, _ = score("Claim [web:citation_1].", verifier=v(grammar="web_brackets"))
    assert reward == 1.0


def test_paren_part_grammar():
    reward, _ = score("Claim (citation_0).", verifier=v(grammar="paren_part"))
    assert reward == 1.0


def test_markdown_footnote_grammar():
    reward, _ = score("Claim[^citation_0].", verifier=v(grammar="markdown_footnote"))
    assert reward == 1.0


def test_snippet_id_kind_cite_xml():
    reward, _ = score(
        "Claim. <cite>citation_0:snippet_2</cite>",
        verifier=v(
            id_kind="snippet",
            # id_regex is authoritative for grammar parsing, so a
            # snippet row must carry the snippet ID scheme (a real v2 row always does).
            id_regex=r"citation_[0-9]+:snippet_[0-9]+",
            valid_id_set=["citation_0:snippet_2", "citation_1:snippet_0"],
        ),
    )
    assert reward == 1.0


def test_snippet_bare_source_id_scores_zero():
    """In snippet mode, citing citation_0 without :snippet_N must not parse."""
    reward, d = score(
        "Claim. <cite>citation_0</cite>",
        verifier=v(id_kind="snippet", valid_id_set=["citation_0:snippet_0"]),
    )
    assert reward == 0.0


# ── Holdout grammar smoke test ────────────────────────────────────────────────


def test_curly_double_holdout_grammar_parses():
    """Holdout grammar must still parse correctly (it's just never trained on)."""
    reward, _ = score(
        "Claim {{citation_0}}.",
        verifier=v(grammar="curly_double"),
    )
    assert reward == 1.0


def test_angle_pipe_holdout_grammar_parses():
    reward, _ = score(
        "Claim <ref|citation_1>.",
        verifier=v(grammar="angle_pipe"),
    )
    assert reward == 1.0


# ── Unknown grammar ───────────────────────────────────────────────────────────


def test_unknown_grammar_scores_zero():
    reward, d = score("Claim.", verifier=v(grammar="nonexistent_format"))
    assert reward == 0.0
    assert d["gate_failed"] == "unknown_grammar"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
