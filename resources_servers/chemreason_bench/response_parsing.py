# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover the model's JSON object and apply upstream's field coercions.

Derived from ``predict/predict.py`` in https://github.com/Khadaz/ChemReason-Bench
at commit ``c0b9ac2933708fcca47b1795492952cbf280e194`` (Apache-2.0): the
post-processors, the parse-failure contract and the lm probability rules.
Departures are marked DEPARTURE in place. Prompts ask for a bare JSON object;
replies arrive with think blocks, prose and fences.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from slot_canonicalization import canonicalize_slots


# Thinking models emit these; the JSON we want is always after them.
_THINK_BLOCK_RE = re.compile(r"<(think|thinking)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
# An unterminated block (truncated by the output budget) leaves no closing tag.
_OPEN_THINK_RE = re.compile(r"<(think|thinking)\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_UPSTREAM_SPAN_RE = re.compile(r"\{[\s\S]*\}\s*$")
_NO_PARSE = object()

# Upstream's threshold for turning the requested score into a discrete label
# (predict.py, `label = bool(score >= threshold)`).
BINARY_LABEL_THRESHOLD = 0.5

# Bound anything model-controlled before it reaches a parser, and bound the
# rationale before it reaches the O(n*m) token comparison downstream.
MAX_RESPONSE_CHARS = 200_000
MAX_RATIONALE_CHARS = 20_000


def _iter_json_objects(text: str):
    """Candidate JSON objects, leftmost-first. Callers take the LAST that parses:
    a later object is the model's correction of an earlier draft.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        # Only inside an object: a prose quote before any "{" would otherwise
        # swallow the rest of the reply and hide a valid trailing object.
        if ch == '"' and depth > 0:
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]


def clean_text(raw: Optional[str]) -> str:
    """Bounded text with reasoning blocks removed.

    DEPARTURE, deliberate: upstream's ``_raw`` is the whole answer, think blocks
    included. Every raw-text fallback here uses this instead, because scoring a
    reasoning model's trace measures the trace and not the answer. One rule for
    all three fallbacks -- ordering, contrastive choice and rationalization.
    """
    if not raw:
        return ""
    text = raw[:MAX_RESPONSE_CHARS]
    text = _THINK_BLOCK_RE.sub(" ", text)
    return _OPEN_THINK_RE.sub(" ", text)


def extract_json(raw: Optional[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return ``(object, status)``; ``object`` is None when no object parsed.

    Status ``non_object_json`` marks a reply that IS valid JSON but not an object
    (``[1, 2]``, ``null``, ``"YES"``). Upstream's ``chat_json_hf`` returns such a
    value unchanged and ``post_binary`` labels a non-dict negative, whereas a
    failed parse becomes ``{"_raw": ...}`` and defaults positive. The two must stay
    distinguishable.
    """
    if not raw or not raw.strip():
        return None, "empty_output"
    text = clean_text(raw)
    if not text.strip():
        # The whole reply was reasoning: the budget ran out before any answer.
        return None, "no_json_found"

    # Upstream's single candidate (chat_json_hf): first fenced block, else the span from
    # the first "{" to a closing "}" at the end, else the whole reply. When it parses, its
    # outcome is upstream's outcome: a dict scores, anything else is a non-object reply.
    fence = _FENCE_RE.search(text)
    span = _UPSTREAM_SPAN_RE.search(text)
    upstream_candidate = fence.group(1) if fence else (span.group(0) if span else text)
    try:
        value = json.loads(upstream_candidate.strip())
    except (ValueError, RecursionError):
        value = _NO_PARSE
    if value is not _NO_PARSE:
        return (value, "ok") if isinstance(value, dict) else (None, "non_object_json")

    # DEPARTURE: upstream stops here with {"_raw": ...}. This port also scans for embedded
    # objects and takes the rightmost, so a multi-object or prose-wrapped reply still scores.
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)]
    candidates.extend(_iter_json_objects(text))

    parsed = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, dict):
            parsed = obj  # keep going; the rightmost valid object wins
    if parsed is not None:
        return parsed, "ok"
    # Upstream's last candidate is the whole reply; valid non-object JSON parses here.
    try:
        value = json.loads(text.strip())
    except (ValueError, RecursionError):
        return None, "no_json_found"
    return (value, "ok") if isinstance(value, dict) else (None, "non_object_json")


def _as_float(value: Any, default: float) -> float:
    """Coerce to float, treating every conversion limit as malformed output.

    A JSON number with hundreds of digits raises OverflowError, not ValueError,
    and an unhandled one escapes verify() as a 500 that aborts the run.
    """
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
        return default
    return out


_ID_TOKEN_RE = re.compile(r"id(\d+)")
_DIGITS_RE = re.compile(r"\d+")
_RAW_ID_RE = re.compile(r"[\"\']?([A-Za-z0-9_\-]+)[\"\']?")


def canonicalize_step_token(token: Any) -> str:
    """Upstream's ``_canonicalize_step_token``: 'id2' and 'step_2' both mean '2'."""
    text = str(token).strip()
    if text.isdigit():
        return text
    match = _ID_TOKEN_RE.fullmatch(text)
    if match:
        return match.group(1)
    match = _DIGITS_RE.search(text)
    return match.group(0) if match else text


def post_ordering(obj: Optional[Dict[str, Any]], expected: List[str], raw: str = "") -> List[str]:
    """Port of upstream ``post_ordering``: canonicalize ids, drop repeats, and --
    once one legal id has matched -- append the unmentioned ids in presentation
    order. An answer matching nothing yields [], never a fabricated order.
    """
    obj = obj if isinstance(obj, dict) else {}
    got = obj.get("predicted_order")
    if got is None:
        got = obj.get("order")
    if got is None:
        got = []
    if not isinstance(got, list) or not all(isinstance(x, (str, int)) for x in got):
        ids = _RAW_ID_RE.findall(raw or "")
        got = [str(x) for x in ids] if ids else []

    seen = set()
    clean: List[str] = []
    for item in map(str, got):
        canonical = canonicalize_step_token(item)
        if canonical in expected and canonical not in seen:
            clean.append(canonical)
            seen.add(canonical)

    if not clean:
        return []
    for expected_id in expected:
        if expected_id not in seen:
            clean.append(expected_id)
    return clean


def post_contrastive(obj: Optional[Dict[str, Any]], options: List[Any], raw: str = "") -> int:
    """Port of upstream ``post_contrastive``, index only. Recovers the choice from
    raw text and range-checks; an invalid index becomes -1, never option 0.
    """
    obj = obj if isinstance(obj, dict) else {}
    choice = obj.get("predicted_choice") or obj.get("choice")
    index = obj.get("predicted_option_idx", obj.get("pred_idx"))

    if choice not in options:
        for option in options:
            if isinstance(option, str) and option in (raw or ""):
                choice = option
                break

    if not isinstance(index, int) or isinstance(index, bool) or not (0 <= index < len(options)):
        index = options.index(choice) if choice in options else -1
    return int(index)


def to_prediction(
    task_type: str,
    obj: Optional[Dict[str, Any]],
    expected_step_ids: Optional[List[str]] = None,
    options: Optional[List[Any]] = None,
    raw: str = "",
    legend: Optional[Dict[str, str]] = None,
    non_object: bool = False,
) -> Dict[str, Any]:
    """Coerce a parsed object into the shape ``metrics.score_row`` expects.

    UPSTREAM'S PARSE CONTRACT (JSON eval path, ``chat_json_hf`` -> ``post_process``):
    a FAILED parse becomes ``{"_raw": answer}``, a dict with no score, so an
    unparseable binary reply defaults to ``score=0.5`` -> ``label=True``. A reply
    that parses to a NON-OBJECT (``[1, 2]``, ``null``, ``"YES"``) is returned as is
    and ``post_binary`` labels it False. ``non_object`` carries that second case.

    ``_raw`` exists only when parsing failed, so a successfully parsed object is
    never re-scanned as text: ``{"predicted_order": "1 2 0"}`` scores empty.
    """
    no_object = not isinstance(obj, dict)
    obj = obj if isinstance(obj, dict) else {}
    # Upstream only has _raw when parsing failed, so recovery text is available
    # only then. Cleaned first; see clean_text on why that departs.
    recovery = clean_text(raw).strip() if no_object else ""

    if task_type == "ordering":
        return {"predicted_order": post_ordering(None if no_object else obj, expected_step_ids or [], recovery)}

    if task_type == "contrastive_choice":
        return {"predicted_option_idx": post_contrastive(None if no_object else obj, options or [], recovery)}

    if task_type in ("step_validation", "condition_validation"):
        if non_object:
            return {"score": 0.5, "label": False}  # post_binary's non-dict branch
        # A failed parse is {"_raw": ...} upstream: a dict with no score, i.e. this path.
        score = obj.get("score", obj.get("prob_positive"))
        score = min(1.0, max(0.0, _as_float(score, 0.5)))
        return {"score": score, "label": bool(score >= BINARY_LABEL_THRESHOLD)}

    if task_type == "step_completion":
        slots = obj.get("slots")
        if not isinstance(slots, dict):
            slots = {}
        # Upstream canonicalizes against the row's legend before scoring; without
        # it a matched reagent reads as an unrecognised key, or an unnormalised
        # unit trips the fatal flag and zeroes the task.
        slots = canonicalize_slots({str(k): v for k, v in slots.items()}, legend)
        return {"action": str(obj.get("action", "")), "slots": slots}

    if task_type == "rationalization":
        for key in ("gold_rationale", "rationale", "predicted_rationale", "answer"):
            value = obj.get(key)
            if isinstance(value, list):
                value = " ".join(str(item) for item in value)
            if isinstance(value, str) and value.strip():
                return {"gold_rationale": value[:MAX_RATIONALE_CHARS]}
        # Upstream sets _raw ONLY when parsing fails (predict.py:303), so a dict
        # that parsed but lacks every rationale key scores "" there -- not its own
        # JSON text. Mirror that: fall back to the reply only when nothing parsed.
        return {"gold_rationale": recovery[:MAX_RATIONALE_CHARS]}

    raise ValueError(f"unknown task_type: {task_type!r}")


# --------------------------------------------------------------------------
# lm protocol
# --------------------------------------------------------------------------

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)
# An lm option index is a bare small integer. Bounded and delimiter-guarded so a
# `$5$` reagent placeholder is not read as option 5, and a 5,000-digit run does
# not reach int(), whose 4,300-digit limit raises ValueError.
_INDEX_RE = re.compile(r"(?<![\w$.])-?\d{1,6}(?![\w$.])")
_DIGITS_ONLY_RE = re.compile(r"[0-9]+")
# Upstream's defaults: predict.py --lm_eps / --lm_binary_eps (1e-12) and
# --binary_threshold (0.5).
_LM_EPS = 1e-12
_BINARY_THRESHOLD = 0.5


def _norm_token(token: str) -> str:
    """Upstream's provider-token normalizer: strip spaces and stray quotes."""
    return str(token).strip().strip('"').strip("'").strip()


def _first_token_mass(logprobs: Any) -> List[Tuple[str, float]]:
    """``(token, probability)`` at the first generated position.

    vLLM repeats the sampled token inside ``top_logprobs``; count it once.
    """
    if not isinstance(logprobs, list) or not logprobs:
        return []
    first = logprobs[0]
    if not isinstance(first, dict):
        return []
    entries = first.get("top_logprobs") or [first]
    out: List[Tuple[str, float]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        token, logprob = entry.get("token"), entry.get("logprob")
        if isinstance(token, str) and isinstance(logprob, (int, float)):
            out.append((token, math.exp(float(logprob))))
    return out


def _lm_binary_prob_yes(alternatives: List[Tuple[str, float]]) -> Optional[float]:
    """P(YES) per the paper's appendix F.3.2 eq. (1): softmax over Y = {YES, NO}.

    DEPARTURE from ``predict.py``, which matches any token ENDING in YES/NO over
    the full vocabulary. That rule is asymmetric (Llama-3.1: 153 tokens count as
    NO vs 21 as YES) and its tail is invisible through a top-k API. Cost of
    following the paper instead: <1 point on Llama-3.1-8B.
    """
    m_yes = m_no = 0.0
    for token, probability in alternatives:
        if probability <= 0.0:
            continue
        normalized = _norm_token(token).upper()
        if normalized == "YES":
            m_yes += probability
        elif normalized == "NO":
            m_no += probability
    if m_yes <= 0.0 and m_no <= 0.0:
        return None
    # DEPARTURE: upstream requires both candidates, over the full vocabulary where both
    # always exist. Over a top-k window the unseen side is not absent, it is below the
    # smallest returned probability; use that floor as its mass rather than abstaining,
    # which would turn a confident NO into the positive default.
    floor = min(probability for _, probability in alternatives if probability > 0.0)
    m_yes = m_yes or floor
    m_no = m_no or floor
    return float((m_yes + _LM_EPS) / (m_yes + m_no + 2.0 * _LM_EPS))


def _lm_choice_probs(alternatives: List[Tuple[str, float]], num_options: int) -> Optional[List[float]]:
    """Upstream's ``_lm_choice_probs_from_prompt_ids``, over visible tokens."""
    if num_options <= 0:
        return None
    mass = [0.0] * num_options
    seen = set()
    for token, probability in alternatives:
        if probability <= 0.0:
            continue
        normalized = _norm_token(token)
        if not _DIGITS_ONLY_RE.fullmatch(normalized):
            continue
        index = int(normalized)
        if 0 <= index < num_options:
            mass[index] += probability
            seen.add(index)
    # DEPARTURE: predict.py requires >=2 distinct indices. Over a top-k window
    # that zeroed every Phi-3-mini contrastive row; the paper's eq. (2) has no
    # such condition, so one visible index is enough to rank.
    if not seen:
        return None
    mass = [m + _LM_EPS for m in mass]
    total = sum(mass)
    if total <= 0.0:
        return None
    return [m / total for m in mass]


def to_prediction_lm(
    task_type: str,
    raw: Optional[str],
    logprobs: Any = None,
    options: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Parse an lm-protocol reply, whose contract is one bare decision token.

    Upstream decides from token probabilities; this agrees whenever the reply
    opens with a decision token. Replies opening with neither take the
    conservative default rather than leaving the denominator. See README.
    """
    alternatives = _first_token_mass(logprobs)
    if alternatives and task_type in ("step_validation", "condition_validation"):
        score = _lm_binary_prob_yes(alternatives)
        if score is None:
            # Upstream: post_binary({"score": None}) -> float(None) raises -> 0.5,
            # which thresholds POSITIVE.
            return {"score": 0.5, "label": True, "status": "lm_abstained"}
        return {"score": score, "label": bool(score >= _BINARY_THRESHOLD), "status": "ok_logprobs"}
    if alternatives and task_type == "contrastive_choice":
        probabilities = _lm_choice_probs(alternatives, len(options or []))
        if probabilities is not None:
            best = int(max(range(len(probabilities)), key=lambda i: probabilities[i]))
            return {"predicted_option_idx": best, "status": "ok_logprobs"}
        # Upstream would index its None result and raise; -1 is its own
        # "no valid index" sentinel.
        return {"predicted_option_idx": -1, "status": "lm_abstained"}

    text = _norm_token(_THINK_BLOCK_RE.sub(" ", raw or ""))
    text = _OPEN_THINK_RE.sub(" ", text).strip()

    if task_type in ("step_validation", "condition_validation"):
        if not text:
            return {"score": 0.5, "label": False, "status": "empty_output"}
        yes, no = _YES_RE.search(text), _NO_RE.search(text)
        if yes and (not no or yes.start() < no.start()):
            return {"score": 1.0, "label": True, "status": "ok"}
        if no:
            return {"score": 0.0, "label": False, "status": "ok"}
        # Neither token present. Deliberately unlike the gen path, where 0.5 is
        # upstream's own >= 0.5 default and counts positive: here there is no
        # upstream default to match, since upstream reads probability mass and
        # abstains outright. Negative is the conservative reading. See README.
        return {"score": 0.5, "label": False, "status": "no_decision_token"}

    if task_type == "contrastive_choice":
        if not text:
            return {"predicted_option_idx": -1, "status": "empty_output"}
        match = _INDEX_RE.search(text)
        if match is None:
            return {"predicted_option_idx": -1, "status": "no_decision_token"}
        return {"predicted_option_idx": int(match.group()), "status": "ok"}

    raise ValueError(f"task_type {task_type!r} has no lm protocol")
