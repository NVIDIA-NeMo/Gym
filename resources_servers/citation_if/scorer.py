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
"""Pure scoring logic for citation_if — no FastAPI dependency.

Importable standalone for tests, trajectory builders, and offline analysis.
The server (app.py) imports from here.

Design points worth knowing before editing:
  - the malformed-attempt gate is STRICT: strip well-formed spans, then scan the residual
  - GRAMMAR_TABLE is parameterized by verifier.id_regex, so ID schemes coexist in one pool
  - claim_wrap_xml parsing matches the convention used by citation evaluation suites
  - the correctness gate works at row granularity; cited_ids is logged in match_details
  - the scorer NEVER reads answer text — citations only
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple


# Broad attempt detector for the malformed-attempt gate. Deliberately a SUPERSET of every row-level
# id_regex: it must catch numeric-style leakage on random-ID rows, e.g.
# "… [citation_27bx]. Sources: citation_1". Never narrow this to the row's id_regex.
ATTEMPT_REGEX = r"citation_[A-Za-z0-9_:]+"
_ATTEMPT_RE = re.compile(ATTEMPT_REGEX, re.IGNORECASE)

# Fallback id_regex for rows that omit one (keeps legacy numeric-ID rows scoring
# identically). Rows produced by current build tooling always carry an explicit id_regex.
DEFAULT_ID_REGEX: Dict[str, str] = {
    "full_source": r"citation_[0-9]+",
    "snippet": r"citation_[0-9]+:snippet_[0-9]+",
}

# Reserved instruction-example ID. Never valid in any row's
# valid_id_set — echoing it fails the no-hallucination gate.
RESERVED_EXAMPLE_ID = "citation_xmpl"

TERMINAL_TOOL_RESPONSE = "No further tool results are available. Provide your final answer."


# Grammar templates. "{IDS}" is replaced with the row's ID sub-pattern, built from
# verifier.id_regex. Group(1) always captures the citation ID(s); multi-ID
# grammars may hold a comma-separated list there.
#
#   multi       — grammar admits comma-separated ID lists inside one span
#   claim_group — group index holding the wrapped claim text (empty ⇒ not a citation)
#   open_re/close_re — tag markers, used for the balance check on tag grammars
#   holdout     — reserved dev-holdout grammar, never assigned to training rows
GRAMMAR_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # Trained grammars
    "cite_xml": {
        "template": r"<cite>({IDS})</cite>",
        "multi": True,
        "open_re": re.compile(r"<cite\b", re.IGNORECASE),
        "close_re": re.compile(r"</cite\s*>", re.IGNORECASE),
        "holdout": False,
    },
    "ascii_brackets": {
        # Negative lookahead excludes markdown links [citation_N](url)
        "template": r"\[({IDS})\](?!\()",
        "multi": False,
        "holdout": False,
    },
    "claim_wrap_xml": {
        # (.*?) plus an empty-claim post-filter, NOT an inlined \s*(\S.*?)\s*.
        # The inlined form crossed tag boundaries when an empty claim-wrap preceded a
        # valid one, misattributing the ID. Citation evaluators parse this grammar the
        # same way, so changing it here would silently diverge train-time reward from
        # eval-time scoring.
        "template": r'<cite index="({IDS})">(.*?)</cite>',
        "multi": True,
        "flags": re.DOTALL,
        "claim_group": 2,
        "open_re": re.compile(r"<cite\b", re.IGNORECASE),
        "close_re": re.compile(r"</cite\s*>", re.IGNORECASE),
        "holdout": False,
    },
    "fullwidth_brackets": {
        # 【】 = U+3010/U+3011 (lenticular brackets). These exact codepoints are what
        # citation evaluators match — do not substitute visually similar characters.
        "template": r"【({IDS})】",
        "multi": False,
        "holdout": False,
    },
    "ref_colon": {
        "template": r"\[ref:({IDS})\]",
        "multi": False,
        "holdout": False,
    },
    "double_angle": {
        "template": r"<<({IDS})>>",
        "multi": False,
        "holdout": False,
    },
    "web_brackets": {
        "template": r"\[web:({IDS})\]",
        "multi": False,
        "holdout": False,
    },
    "paren_part": {
        "template": r"\(({IDS})\)",
        "multi": False,
        "holdout": False,
    },
    "markdown_footnote": {
        "template": r"\[\^({IDS})\]",
        "multi": False,
        "holdout": False,
    },
    # Holdout grammars — never assign to training rows
    "curly_double": {
        "template": r"\{\{({IDS})\}\}",
        "multi": False,
        "holdout": True,
    },
    "angle_pipe": {
        "template": r"<ref\|({IDS})>",
        "multi": False,
        "holdout": True,
    },
}

# Back-compat alias: callers that only need grammar names / holdout flags.
GRAMMAR_TABLE = GRAMMAR_TEMPLATES


@lru_cache(maxsize=512)
def compile_grammar(grammar_name: str, id_regex: str) -> re.Pattern:
    """Compile a grammar's citation pattern for a specific row ID scheme."""
    entry = GRAMMAR_TEMPLATES[grammar_name]
    if entry.get("multi"):
        ids = rf"(?:{id_regex})(?:,(?:{id_regex}))*"
    else:
        ids = rf"(?:{id_regex})"
    # .replace, not .format — templates contain literal regex braces (e.g. \{\{).
    return re.compile(entry["template"].replace("{IDS}", ids), entry.get("flags", 0))


def _row_id_regex(verifier: Dict[str, Any]) -> str:
    explicit = verifier.get("id_regex")
    if explicit:
        return explicit
    return DEFAULT_ID_REGEX.get(verifier.get("id_kind", "full_source"), DEFAULT_ID_REGEX["full_source"])


def _iter_spans(pattern: re.Pattern, entry: Dict[str, Any], text: str) -> List[re.Match]:
    """Well-formed citation spans. Claim-wrap grammars drop empty-claim matches."""
    claim_group = entry.get("claim_group")
    spans = []
    for m in pattern.finditer(text):
        if claim_group is not None and not (m.group(claim_group) or "").strip():
            continue  # empty claim wrap is not a citation
        spans.append(m)
    return spans


def _ids_from_spans(spans: List[re.Match]) -> List[str]:
    """Flatten span group(1) into IDs. Deduplicated, order preserved."""
    ids: List[str] = []
    for m in spans:
        ids.extend(m.group(1).split(","))
    return list(dict.fromkeys(ids))


def _strip_spans(spans: List[re.Match], text: str) -> str:
    """Blank out well-formed citation spans, preserving offsets elsewhere."""
    if not spans:
        return text
    out = list(text)
    for m in spans:
        for i in range(m.start(), m.end()):
            out[i] = " "
    return "".join(out)


def _answer_text(spans: List[re.Match], entry: Dict[str, Any], text: str) -> str:
    """The answer prose: citation markup blanked, but the CLAIM KEPT.

    Sibling of `_strip_spans`, and the difference matters. `_strip_spans` blanks whole
    spans, which is right for leakage scanning but wrong here: `claim_wrap_xml` has the
    form `<cite index="ID">the claim</cite>` (`claim_group=2`), so the answer text lives
    *inside* the span. Blanking it would report "no answer" for an answer that is entirely
    a wrapped claim — a legitimate form the model is free to use.

    So: blank each span except the bytes covered by its claim group, if the grammar
    defines one. For the eight ID-only grammars there is no claim group and this reduces
    to `_strip_spans`.
    """
    if not spans:
        return text
    claim_group = entry.get("claim_group")
    out = list(text)
    for m in spans:
        keep = None
        if claim_group and m.lastindex and claim_group <= m.lastindex:
            keep = m.span(claim_group)
        for i in range(m.start(), m.end()):
            if keep and keep[0] <= i < keep[1]:
                continue
            out[i] = " "
    return "".join(out)


def _tag_balance_ok(entry: Dict[str, Any], text: str, n_spans: int) -> bool:
    """Tag grammars: open == parsed == close."""
    open_re, close_re = entry.get("open_re"), entry.get("close_re")
    if not open_re or not close_re:
        return True
    return len(open_re.findall(text)) == n_spans == len(close_re.findall(text))


def score_citation_if(
    text: str,
    function_call_count: int,
    verifier: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Apply the citation_if gate sequence and return (reward, details).

    The reward reads CITATIONS ONLY — answer text is never checked.

    Gate sequence (cite mode):
        0. Structural         — non-empty text, no tool call
        0b. Answer presence   — >=1 word character outside citation markup (claim text
                                counts). A citation with no answer scores 0. Existence
                                only, never quality.
        1. Malformed-attempt  — STRICT: strip well-formed spans, then ANY residual
                                ID-shaped token (broad ATTEMPT_REGEX) ⇒ 0. Tag
                                grammars additionally require open == parsed == close.
        2. Must-cite          — >= min_valid_citations parsed IDs in valid_id_set
        3. No-hallucination   — every parsed citation ID in valid_id_set
        4. Correctness        — (when expected_ids set, at ROW granularity)
                                expected_ids ⊆ cited        → else correctness_missed_gold
                                |cited| ≤ |expected| + slack → else correctness_over_cap

    No-cite mode:
        Reward 1 iff zero attempt-regex matches AND zero citation markup.
    """
    grammar_name = verifier.get("grammar", "")
    id_kind = verifier.get("id_kind", "full_source")
    valid_id_set = set(verifier.get("valid_id_set") or [])
    expected_ids = verifier.get("expected_ids")  # list[str] or None
    min_valid = int(verifier.get("min_valid_citations", 1))
    mode = verifier.get("mode", "cite")

    if grammar_name not in GRAMMAR_TEMPLATES:
        return 0.0, {"gate_failed": "unknown_grammar", "grammar": grammar_name}

    entry = GRAMMAR_TEMPLATES[grammar_name]
    id_regex = _row_id_regex(verifier)
    pattern = compile_grammar(grammar_name, id_regex)

    # Gate 0: Structural
    structural_ok = bool(text.strip()) and function_call_count == 0
    if not structural_ok:
        return 0.0, _make_details(
            gate_failed="structural",
            text=text,
            function_call_count=function_call_count,
            grammar=grammar_name,
            id_kind=id_kind,
            mode=mode,
            id_regex=id_regex,
            parsed_ids=[],
            valid_cited=[],
            invalid_cited=[],
            attempt_hits=False,
        )

    spans = _iter_spans(pattern, entry, text)
    parsed_ids = _ids_from_spans(spans)
    valid_cited = [i for i in parsed_ids if i in valid_id_set]
    invalid_cited = [i for i in parsed_ids if i not in valid_id_set]
    attempt_hits = bool(_ATTEMPT_RE.search(text))

    # No-cite mode (negatives): no citation of any shape may appear.
    if mode == "no_cite":
        any_markup = bool(spans) or (entry.get("open_re") is not None and entry["open_re"].search(text))
        reward = 1.0 if (not attempt_hits and not any_markup) else 0.0
        return reward, _make_details(
            gate_failed=None if reward == 1.0 else "no_cite",
            text=text,
            function_call_count=function_call_count,
            grammar=grammar_name,
            id_kind=id_kind,
            mode=mode,
            id_regex=id_regex,
            parsed_ids=parsed_ids,
            valid_cited=valid_cited,
            invalid_cited=invalid_cited,
            attempt_hits=attempt_hits,
        )

    # Gate 0b: ANSWER PRESENCE. A citation with no answer must not earn reward.
    #
    # Found by reviewing real rollouts: one scored 1.0 on the complete output
    # `<cite>citation_7367:snippet_2</cite>` — correct grammar, correct gold snippet,
    # and no answer at all. Its reasoning had derived the answer ("9.731 Jupiter radii")
    # and then emitted only the citation. Gate 0 passed it because the raw string is not
    # empty: it is 36 characters of markup.
    #
    # This is answer EXISTENCE, not answer quality, and the distinction is what keeps it
    # inside the "no answer-text checks in scorer" rule. That rule bars grading what the
    # answer says — and we could not grade it anyway, since no gold answer exists in the
    # data (upstream carries reference markers like `[ref:2]`, never the answer string).
    # Asking whether an answer was written is the same question gate 0 already asks of the
    # raw string, applied to the part that is actually the answer.
    #
    # Deliberately a presence test, not a length floor. Measured over the pilot's 483
    # reward-1 cite rollouts, every threshold from 1 to 12 words catches exactly this one
    # case, so a floor buys nothing — while the real answer content (the `Exact Answer:`
    # line) has a median of 5 words, so any floor above 1 would start rejecting correct
    # terse answers the moment the policy sheds its SFT boilerplate.
    #
    # Cite mode only: `no_cite` rows are an abstain task whose gate is "no citation of any
    # shape", and gate 0 already guarantees their text is non-empty.
    if not re.search(r"\w", _answer_text(spans, entry, text)):
        return 0.0, _make_details(
            gate_failed="no_answer",
            text=text,
            function_call_count=function_call_count,
            grammar=grammar_name,
            id_kind=id_kind,
            mode=mode,
            id_regex=id_regex,
            parsed_ids=parsed_ids,
            valid_cited=valid_cited,
            invalid_cited=invalid_cited,
            attempt_hits=attempt_hits,
        )

    # Gate 1: STRICT well-formed attempts.
    # Universal property: every ID-shaped token must sit inside a well-formed span.
    # Strip the spans, then any residual hit is a malformed attempt — raw-ID leakage,
    # "Sources:" trailers, comma+space lists, empty markup, raw hallucinated IDs.
    residual = _strip_spans(spans, text)
    if _ATTEMPT_RE.search(residual) or not _tag_balance_ok(entry, text, len(spans)):
        return 0.0, _make_details(
            gate_failed="malformed_attempt",
            text=text,
            function_call_count=function_call_count,
            grammar=grammar_name,
            id_kind=id_kind,
            mode=mode,
            id_regex=id_regex,
            parsed_ids=parsed_ids,
            valid_cited=valid_cited,
            invalid_cited=invalid_cited,
            attempt_hits=attempt_hits,
        )

    # Gate 2: Must-cite
    if len(valid_cited) < min_valid:
        return 0.0, _make_details(
            gate_failed="must_cite",
            text=text,
            function_call_count=function_call_count,
            grammar=grammar_name,
            id_kind=id_kind,
            mode=mode,
            id_regex=id_regex,
            parsed_ids=parsed_ids,
            valid_cited=valid_cited,
            invalid_cited=invalid_cited,
            attempt_hits=attempt_hits,
        )

    # Gate 3: No-hallucination (also catches reserved-example echo)
    if invalid_cited:
        return 0.0, _make_details(
            gate_failed="no_hallucination",
            text=text,
            function_call_count=function_call_count,
            grammar=grammar_name,
            id_kind=id_kind,
            mode=mode,
            id_regex=id_regex,
            parsed_ids=parsed_ids,
            valid_cited=valid_cited,
            invalid_cited=invalid_cited,
            attempt_hits=attempt_hits,
        )

    # Gate 4: Correctness at ROW granularity.
    # expected_ids already carries the row's granularity — document IDs on
    # full-source rows, exact gold-snippet composite IDs on snippet rows. No mapping
    # here: parsed snippet citations already carry the ":snippet_K" suffix, so citing
    # a different snippet of the gold document fails correctness_missed_gold.
    #
    # Exact match is the coded default. Looser precision-floor variants are
    # answered from pilot data — every rollout stays re-scorable offline because
    # cited_ids is logged in match_details below.
    missed_ids: List[str] = []
    over_cap = False
    if expected_ids is not None:
        expected_slack = int(verifier.get("expected_slack", 1))
        expected_set = set(expected_ids)
        missed_ids = sorted(expected_set - set(valid_cited))
        over_cap = len(valid_cited) > len(expected_set) + expected_slack
        if missed_ids or over_cap:
            return 0.0, _make_details(
                gate_failed="correctness_missed_gold" if missed_ids else "correctness_over_cap",
                text=text,
                function_call_count=function_call_count,
                grammar=grammar_name,
                id_kind=id_kind,
                mode=mode,
                id_regex=id_regex,
                parsed_ids=parsed_ids,
                valid_cited=valid_cited,
                invalid_cited=invalid_cited,
                attempt_hits=attempt_hits,
                expected_ids=list(expected_ids),
                missed_ids=missed_ids,
                over_cap=over_cap,
            )

    return 1.0, _make_details(
        gate_failed=None,
        text=text,
        function_call_count=function_call_count,
        grammar=grammar_name,
        id_kind=id_kind,
        mode=mode,
        id_regex=id_regex,
        parsed_ids=parsed_ids,
        valid_cited=valid_cited,
        invalid_cited=invalid_cited,
        attempt_hits=attempt_hits,
        expected_ids=list(expected_ids) if expected_ids is not None else None,
        missed_ids=missed_ids,
        over_cap=over_cap,
    )


def _make_details(
    gate_failed: Optional[str],
    text: str,
    function_call_count: int,
    grammar: str,
    id_kind: str,
    mode: str,
    id_regex: str,
    parsed_ids: List[str],
    valid_cited: List[str],
    invalid_cited: List[str],
    attempt_hits: bool,
    expected_ids: Optional[List[str]] = None,
    missed_ids: Optional[List[str]] = None,
    over_cap: bool = False,
) -> Dict[str, Any]:
    return {
        "gate_failed": gate_failed,
        "mode": mode,
        "grammar": grammar,
        "id_kind": id_kind,
        "id_regex": id_regex,
        "structural_ok": bool(text.strip()) and function_call_count == 0,
        "function_call_count": function_call_count,
        # attempt_hits: any ID-shaped token anywhere in the answer (broad detector).
        # bare_id_hits kept as an alias for downstream analysis built against v1.
        "attempt_hits": attempt_hits,
        "bare_id_hits": attempt_hits,
        "parsed_ids": parsed_ids,
        # cited_ids: the parsed valid citation set, logged on EVERY
        # verdict so rollouts are re-scorable offline under different correctness
        # semantics without re-rolling. Do not remove.
        "cited_ids": valid_cited,
        "valid_cited": valid_cited,
        "invalid_cited": invalid_cited,
        "num_valid_cited": len(valid_cited),
        "expected_ids": expected_ids,
        "missed_ids": missed_ids or [],
        "over_cap": over_cap,
    }
