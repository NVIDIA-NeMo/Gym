# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""API-free checks for interleaved long-context document QA.

Every function here is a hard filter: it decides whether a question-answer pair is
admissible at all, before any judge call. They run on the *growing causal prefix* —
the part of the document the model has read when the question is asked — because that
prefix, not the whole document, is what the answer may depend on.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


# Natural-form answers: the question decides the length, so the caps are generous and
# only catch runaway generations.
MAX_QUESTION_CHARS = 600
MAX_ANSWER_CHARS = 2500

REFUSAL_PHRASES: Tuple[str, ...] = (
    "does not specify",
    "doesn't specify",
    "not mentioned",
    "no mention",
    "i don't know",
    "cannot be determined",
    "can't be determined",
    "not provided",
    "no information",
    "not specified",
    "not stated",
    "not given",
    "unable to determine",
    "insufficient information",
    "not enough information",
    "the document does not",
    "the text does not",
    "the code does not",
    "not possible to determine",
    "there is no information",
)

# References to how the source is presented rather than to what it says. The seed output
# rules forbid them: a question that says "the chunk above" leaks the chunking scheme and
# stops making sense once the record is rendered.
# The presentation sense is a complete noun phrase: "summarise this passage." or "the
# chunk above". Followed by another noun it is content -- "the passage of the bill", "the
# passage rate", "which chunk of the budget".
_AS_NOUN_PHRASE = (
    r"(?=\s*(?:[,.?!;:]|$|above\b|below\b|shown\b|quoted\b|provided\b|says?\b|said\b|"
    r"states?\b|mentions?\b|describes?\b|contains?\b|lists?\b))"
)

_META_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        # "chunk", "passage" and "above" are content words too ("the passage of the bill",
        # "above-market returns", "which chunk of the budget"), so each needs a
        # presentation sense before it counts as a meta-reference.
        r"\b(?:the|this|that|each|previous|preceding|first|last|next|current|same)\s+chunks?\b" + _AS_NOUN_PHRASE,
        r"\bchunks?\s+(?:above|below|\d+|one|two|three|number)\b",
        r"\b(?:the|this|that|previous|preceding|first|last|next|current|same)\s+passages?\b" + _AS_NOUN_PHRASE,
        r"\bpassages?\s+(?:above|below|quoted|shown|provided)\b",
        r"\bsection number\b",
        r"\b(?:the|this) (?:text|document|excerpt|material|snippet)\b",
        r"\b(?:the|these) (?:code|files?|documents?|tables?|rows?) (?:above|below|so far|provided|shown|given)\b",
        r"\b(?:earlier|later|previously|before) in the (?:text|document|code|files?)\b",
        r"\brecently (?:read|added|shown)\b",
        r"\b(?:mentioned|listed|shown|described|defined|introduced|quoted|given|provided) (?:above|below|earlier|so far)\b",
        r"\bso far\b",
        r"\b(?:the|this|that)\s+\w+\s+above\b",
        r"\b(?:above|below)[, ]+(?:the|this|that|it|we|you)\b",
        r"\b(?:stated|said|seen|read|given|noted) above\b",
    )
]

# A single long-context document is one text, so a bare referent to it is ordinary English
# ("according to the text", "what the excerpt says"). Only the position words above stay
# banned. Derived from the same nouns the pattern matches: listing them by hand let the
# two drift, so "the excerpt" and "the snippet" were rejected while "the book" and "the
# repository" sat in the allowlist although no pattern could ever produce them.
_PLAIN_REFERENTS = frozenset(
    f"{determiner} {noun}"
    for determiner in ("the", "this")
    for noun in ("text", "document", "excerpt", "material", "snippet")
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip surrounding quote marks and backticks."""
    return re.sub(r"\s+", " ", text).strip().strip("`\"'“”‘’ ").strip()


def check_constraint(answer: str, kind: str, param: Optional[str]) -> Tuple[Optional[bool], str]:
    """Check an explicit response requirement embedded in the question.

    Returns ``(ok, reason)`` where ``ok`` is ``None`` when the requirement is not
    mechanically checkable, in which case the judge is the only arbiter.
    """
    text = answer.strip()
    if kind == "start_word" and param:
        return (text.lstrip("\"'“‘*#- ").lower().startswith(param.strip().lower()), "constraint_start_word")
    if kind == "end_sentence" and param:
        tail = param.strip().rstrip("\"'”’ ").lower()
        return (text.rstrip("\"'”’* ").lower().endswith(tail), "constraint_end_sentence")
    if kind == "wrap_quotes":
        wrapped = (text.startswith('"') and text.endswith('"')) or (text.startswith("“") and text.endswith("”"))
        return (wrapped, "constraint_wrap_quotes")
    if kind == "n_sentences" and param:
        match = re.search(r"\d+", param)
        if match:
            count = len([s for s in _SENTENCE_SPLIT.split(text) if s.strip()])
            return (count == int(match.group()), "constraint_n_sentences")
    if kind == "max_words" and param:
        match = re.search(r"\d+", param)
        if match:
            return (len(text.split()) <= int(match.group()), "constraint_max_words")
    if kind == "one_word":
        return (len(re.findall(r"[A-Za-z0-9][\w'’-]*", text)) == 1, "constraint_one_word")
    if kind == "lowercase":
        return (text == text.lower(), "constraint_lowercase")
    if kind == "list_format" and param:
        lowered = param.lower()
        if "comma" in lowered:
            return ("," in text and "\n" not in text.strip(), "constraint_list_commas")
        if "semicolon" in lowered:
            return (";" in text, "constraint_list_semicolons")
        if "per line" in lowered:
            return ("\n" in text.strip(), "constraint_list_lines")
        if "alphabetical" in lowered:
            items = [x.strip().lower() for x in re.split(r"[,;\n]", text) if x.strip()]
            return (items == sorted(items) and len(items) >= 2, "constraint_list_alpha")
    return (None, "")


def meta_reference(question: str, *, allow_plain_referents: bool = True) -> Optional[str]:
    """Return the first presentation-level reference in ``question``, or ``None``."""
    for pattern in _META_PATTERNS:
        match = pattern.search(question)
        if match is None:
            continue
        hit = match.group(0)
        if allow_plain_referents and hit.lower() in _PLAIN_REFERENTS:
            continue
        return hit
    return None


def degeneracy_reasons(answer: str) -> List[str]:
    """Cheap detectors for answers that are empty, evasive, or degenerate text."""
    reasons: List[str] = []
    text = answer.strip()
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    if not text or not words:
        reasons.append("empty")
        return reasons
    lowered = text.lower()
    if any(phrase in lowered for phrase in REFUSAL_PHRASES):
        reasons.append("refusal")
    run = 1
    for current, following in zip(words, words[1:]):
        run = run + 1 if current == following else 1
        if run >= 6:
            reasons.append("repetition_run")
            break
    if len(words) >= 12:
        trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        duplicates = sum(count - 1 for count in Counter(trigrams).values())
        if duplicates / len(trigrams) > 0.5:
            reasons.append("repetition_trigram")
    if len(text) >= 12:
        punctuation = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        if punctuation / len(text) > 0.45 and not text.startswith("<answer>"):
            reasons.append("char_salad")
    if "­" in text:
        reasons.append("soft_hyphen")
    return reasons


def refusal_is_early(answer: str) -> bool:
    """True when the answer *opens* with a refusal.

    A long grounded answer may legitimately note "X does not mention Y" in passing, so a
    refusal phrase only disqualifies the answer when it appears in the opening.
    """
    head = answer[:160].lower()
    return any(phrase in head for phrase in REFUSAL_PHRASES)


# ---- fuzzy quote location -------------------------------------------------------------
# Quotes that a generator could not place verbatim (unicode punctuation, ligatures,
# line-wrapped text) get one more chance here before the pair is rejected.

_FOLD_MAP = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    "…": ".",
}
_FOLD_DROP = frozenset("_*")  # markdown emphasis


# Generous next to the 5-40 words the prompt asks for, tight enough to bound the scan.
MAX_FUZZY_QUOTE_CHARS = 2_000


def _fold(text: str) -> Tuple[str, List[int]]:
    """Folded, whitespace-collapsed, lower-cased text plus folded index -> original index."""
    out: List[str] = []
    index: List[int] = []
    previous_space = True
    for position, char in enumerate(text):
        if char in _FOLD_DROP:
            continue
        char = _FOLD_MAP.get(char, char)
        if char.isspace():
            if previous_space:
                continue
            out.append(" ")
            index.append(position)
            previous_space = True
        else:
            out.append(char.lower())
            index.append(position)
            previous_space = False
    return "".join(out), index


def fuzzy_locate(quote: str, prefix: str, *, min_ratio: float = 0.8) -> Optional[int]:
    """Approximate character offset of ``quote`` in ``prefix``, or ``None``.

    Unicode punctuation is folded, markdown emphasis dropped and whitespace collapsed;
    then either the folded quote is found outright, or the densest window of three-word
    shingle hits is aligned against it and accepted at ``min_ratio`` similarity.
    """
    folded_quote, _ = _fold(quote.strip().strip("\"“”'"))
    folded_quote = folded_quote.strip()
    if len(folded_quote) < 12:
        return None
    # The shingle search costs one prefix scan per three-word window, so a runaway quote
    # is slow against a long document: 2,000 words over a 585 kB prefix took 2.7 s. The
    # prompt asks for 5-40 words, so a quote this long is not usable evidence anyway and
    # the exact `find` above has already had its chance.
    if len(folded_quote) > MAX_FUZZY_QUOTE_CHARS:
        return None
    folded_prefix, index = _fold(prefix)
    position = folded_prefix.find(folded_quote)
    if position == -1:
        words = folded_quote.split()
        if len(words) < 4:
            return None
        hits: List[int] = []
        for start in range(len(words) - 2):
            shingle = " ".join(words[start : start + 3])
            found = folded_prefix.find(shingle)
            while found != -1 and len(hits) < 400:
                hits.append(found)
                found = folded_prefix.find(shingle, found + 1)
        if not hits:
            return None
        hits.sort()
        span = len(folded_quote) + 40
        best, best_count, low = hits[0], 0, 0
        for high in range(len(hits)):
            while hits[high] - hits[low] > span:
                low += 1
            if high - low + 1 > best_count:
                best_count, best = high - low + 1, hits[low]
        window_start = max(0, best - 30)
        window = folded_prefix[window_start : window_start + len(folded_quote) + 80]
        matcher = difflib.SequenceMatcher(None, window, folded_quote, autojunk=False)
        blocks = [b for b in matcher.get_matching_blocks() if b.size]
        if not blocks:
            return None
        position = window_start + max(0, blocks[0].a - blocks[0].b)
        aligned = folded_prefix[position : position + len(folded_quote)]
        if difflib.SequenceMatcher(None, aligned, folded_quote, autojunk=False).ratio() < min_ratio:
            return None
    return index[position] if position < len(index) else None


def locate_quote(quote: str, prefix: str) -> Tuple[Optional[int], str]:
    """Locate ``quote`` in ``prefix``. Returns ``(offset, match_kind)``.

    ``match_kind`` is ``"exact"``, ``"fuzzy"`` or ``""`` when the quote was not found.
    """
    if not quote:
        return None, ""
    position = prefix.find(quote)
    if position != -1:
        return position, "exact"
    position = fuzzy_locate(quote, prefix)
    return (position, "fuzzy") if position is not None else (None, "")


def verbatim_in_prefix(answer: str, prefix: str) -> bool:
    """Whitespace-normalised substring test for verbatim-reproduction answers."""
    normalized = normalize_whitespace(answer)
    return len(normalized) >= 8 and normalized in re.sub(r"\s+", " ", prefix)


# ---- guessability ---------------------------------------------------------------------

_GUESS_STOPWORDS = re.compile(r"\b(the|a|an|of|and|in|on|at|to|for|by|is|was|were|are)\b")


def _guess_normalize(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^\w\s.%$-]", " ", lowered)
    lowered = _GUESS_STOPWORDS.sub(" ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    # Trailing sentence punctuation is not part of an answer ("The Peruvians." = "Peruvians").
    return lowered.strip(" .,;:!?")


def _numbers(text: str) -> set[float]:
    without_separators = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    values: set[float] = set()
    for token in re.findall(r"\d+(?:\.\d+)?", without_separators):
        try:
            values.add(float(token))
        except ValueError:
            pass
    return values


def _contains_at_boundary(haystack: str, needle: str) -> bool:
    """Substring match that will not fire inside a longer word or number.

    A bare `in` test rejects a question as guessable whenever the digits of its key appear
    anywhere: key "3" matched "around 1993", key "12" matched "120 million". Those are
    different answers, and a false guessable verdict silently discards a good question.
    """
    position = haystack.find(needle)
    while position != -1:
        before = haystack[position - 1] if position else " "
        after_index = position + len(needle)
        after = haystack[after_index] if after_index < len(haystack) else " "
        if not (before.isalnum() or after.isalnum()):
            return True
        position = haystack.find(needle, position + 1)
    return False


def answer_guessed(guess: str, reference: str) -> bool:
    """True when ``guess``, produced without the document, recovers ``reference``.

    A question whose answer can be guessed from parametric knowledge teaches nothing about
    reading a long context, so it is rejected no matter how well grounded the answer is.
    """
    normalized_guess, normalized_reference = _guess_normalize(guess), _guess_normalize(reference)
    if not normalized_reference or "unknown" in normalized_guess[:40]:
        return False
    if _contains_at_boundary(normalized_guess, normalized_reference):
        return True
    reference_numbers = _numbers(reference)
    if reference_numbers and reference_numbers <= _numbers(guess) and len(normalized_reference.split()) <= 4:
        return True
    reference_tokens = set(normalized_reference.split()) - {"", "-", "."}
    if len(reference_tokens) >= 2:
        return len(reference_tokens & set(normalized_guess.split())) / len(reference_tokens) >= 0.8
    return False


# ---- answer shapes --------------------------------------------------------------------
# Shown to the judge so "matches_type" is judged against the shape the family asks for.

ANSWER_SHAPE: Dict[str, str] = {
    "recall": "a value or name, alone or in one sentence",
    "multi_hop": "a phrase or one or two sentences linking the facts",
    "state": "the final value, optionally with one sentence of context",
    "aggregation": "a list or count, optionally with a sentence",
    "comparison": "one to three sentences (or a short list)",
    "ordering": "an ordered list, optionally with a sentence",
    "synthesis": "a short paragraph (3-6 sentences)",
    "coreference": "a name, optionally with the identifying phrase",
    "rag": "a value or one sentence",
    "recency": "a value or one sentence",
    "disambiguation": "a value or name, or the exact verbatim span when one was requested",
    "closure": "the complete list of named items or the chain endpoint, optionally explained in a sentence",
    "multi_evidence": "several sentences combining the cited facts",
    "enumeration": "the complete list of members (optionally with the count), as a list or a sentence",
    "chain": "short direct answer for this step",
    "multistep": "short derived value / name / date / short list, in the format the question states",
    "unanswerable": "one or two sentences saying the text so far does not give this, naming what it does say",
    "verbatim_quote": "the exact passage reproduced word for word, nothing else",
}

# Families whose answer is derived rather than quoted: a quote may legitimately fail to
# locate, and the answer is verified by the judge instead.
DERIVED_FAMILIES = frozenset({"multistep", "unanswerable"})
