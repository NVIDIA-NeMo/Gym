# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BunsenBench Chemistry MCQ verifier and grouped aggregate metrics."""

from __future__ import annotations

import html
import re
import unicodedata
from collections import defaultdict
from typing import Any, Optional

from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from resources_servers.mcqa.app import (
    MCQAResourcesServer,
    MCQAResourcesServerConfig,
    MCQAVerifyRequest,
    MCQAVerifyResponse,
    _get_allowed_letters_from_options,
)


class BunsenChemResourcesServerConfig(MCQAResourcesServerConfig):
    pass


class BunsenChemVerifyRequest(MCQAVerifyRequest):
    bunsen_id: Optional[str] = None
    choices: Optional[list[str | dict[str, Any]]] = None
    source: Optional[str] = None
    bct_field: Optional[str] = None
    bct_subfield: Optional[str] = None


class BunsenChemVerifyResponse(MCQAVerifyResponse):
    uuid: Optional[str] = None
    bunsen_id: Optional[str] = None
    choices: Optional[list[str | dict[str, Any]]] = None
    no_answer: bool
    error_mode: Optional[str] = None
    source: Optional[str] = None
    bct_field: Optional[str] = None
    bct_subfield: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


ANSWER_LINE_PATTERN = re.compile(r"(?im)^\s*(?:(?:[-*•]|\d+[.)])\s*)?(?:\*\*)?answer(?:\*\*)?\s*:\s*(.+?)\s*$")
INLINE_ANSWER_PATTERN = re.compile(r"(?is)(?:\*\*)?answer(?:\*\*)?\s*:\s*(.+?)(?=$|[\r\n])")
BOXED_PATTERN = re.compile(r"\\boxed\{\s*(.*?)\s*\}", re.S)
ANSWER_PHRASE_PATTERN = re.compile(
    r"(?is)(?:the\s+(?:correct\s+)?answer\s+is|my\s+answer\s+is|i\s+(?:choose|select))\s+(.+?)(?:\n|$)"
)
XML_ANSWER_PATTERN = re.compile(
    r"<\s*(answer|choice|response|final_answer|final-answer)\b[^>]*>\s*(.*?)\s*"
    r"<\s*/\s*\1\s*>",
    re.I | re.S,
)
CHOICES_BLOCK_PATTERN = re.compile(r"<\s*choices\b[^>]*>.*?<\s*/\s*choices\s*>", re.I | re.S)
CHOICE_TAG_PATTERN = re.compile(r"<\s*choice\b[^>]*>\s*(.*?)\s*<\s*/\s*choice\s*>", re.I | re.S)
THINK_OPEN_PATTERN = re.compile(r"<\s*think\b[^>]*>", re.I)
THINK_CLOSE_PATTERN = re.compile(r"<\s*/\s*think\s*>", re.I)
REFUSAL_PATTERN = re.compile(
    r"(?i)(?:"
    r"i\s*(?:'|’)?\s*m\s+sorry[, ]+(?:but\s+)?i\s+(?:can(?:'|’)?t|cannot|am\s+unable|won(?:'|’)?t)"
    r"|i\s+(?:can(?:'|’)?t|cannot)\s+(?:help|assist|comply|provide|answer|do\s+that|continue)"
    r"|i\s+am\s+(?:not\s+able|unable)\s+to\s+(?:help|assist|comply|provide|answer)"
    r"|i\s+won(?:'|’)?t\s+be\s+able\s+to\s+(?:help|assist|provide|answer)"
    r"|as\s+an\s+ai(?:\s+language)?\s+model[, ]+i"
    r"|i\s+(?:must|have\s+to|will\s+have\s+to)\s+(?:decline|refuse)"
    r"|i\s+do\s+not\s+feel\s+comfortable"
    r")"
)

# Ordered, mutually exclusive classification buckets for every rollout.
ERROR_MODES: tuple[str, ...] = (
    "correct",
    "wrong_answer",
    "refusal",
    "early_termination",
    "malformed_choice",
    "format_violation",
)

SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
        "₊": "+",
        "₋": "-",
        "＋": "+",
        "−": "-",
        "–": "-",
        "—": "-",
        "‒": "-",
        "‑": "-",
        "×": "x",
        "✕": "x",
        "·": ".",
        "∙": ".",
        "⋅": ".",
        "•": ".",
        "µ": "u",
        "μ": "u",
        "⁄": "/",
        "∕": "/",
    }
)


class BunsenChemResourcesServer(MCQAResourcesServer):
    config: BunsenChemResourcesServerConfig

    def compute_metrics(self, tasks):
        metrics = super().compute_metrics(tasks)
        for group_key in ("source", "bct_field"):
            metrics.update(_grouped_metrics(tasks, group_key, f"by_{group_key}"))
        metrics.update(_grouped_bct_subfield_metrics(tasks))
        metrics.update(_error_mode_metrics(tasks))
        return metrics

    def get_key_metrics(self, agent_metrics):
        key = super().get_key_metrics(agent_metrics)
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        if "pass@1/accuracy" in agent_metrics:
            key["pass@1/accuracy"] = agent_metrics["pass@1/accuracy"]
        if "pass@1/no_answer" in agent_metrics:
            key["pass@1/no_answer"] = agent_metrics["pass@1/no_answer"]
        for mode in ERROR_MODES:
            metric_key = f"error_modes/{mode}"
            if metric_key in agent_metrics:
                key[metric_key] = agent_metrics[metric_key]
        return key

    async def verify(self, body: BunsenChemVerifyRequest) -> BunsenChemVerifyResponse:
        options = _options_from_body(body)
        allowed_letters = _get_allowed_letters_from_options(options)
        gold = _expected_answer_letter(body.expected_answer or "", options)
        if len(gold) == 1 and gold.isalpha():
            allowed_letters.add(gold)

        raw_text = body.response.output_text or ""
        text = raw_text.strip()
        pred: Optional[str] = None

        if text:
            pred = extract_bunsen_answer(text, options, allowed_letters)

        reward = 1.0 if pred is not None and pred == gold else 0.0
        error_mode = classify_error_mode(body.response, raw_text, pred, gold)

        response_payload = body.model_dump(exclude={"expected_answer", "extracted_answer"})
        response_payload.update(
            {
                "reward": reward,
                "expected_answer": gold,
                "extracted_answer": pred,
                "no_answer": pred is None,
                "error_mode": error_mode,
                "bunsen_id": _metadata_value(body, "bunsen_id"),
                "source": _metadata_value(body, "source"),
                "bct_field": _metadata_value(body, "bct_field"),
                "bct_subfield": _metadata_value(body, "bct_subfield"),
            }
        )
        return BunsenChemVerifyResponse(
            **response_payload,
        )


def extract_bunsen_answer(text: str, options: list[dict[str, str]], allowed_letters: set[str]) -> Optional[str]:
    for _, candidate in sorted(_answer_candidates(text), reverse=True):
        parsed = _parse_candidate(candidate, options, allowed_letters)
        if parsed is not None:
            return parsed

    # Last resort: exact option text or a bare letter on boundary lines.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        parsed = _parse_candidate(lines[-1], options, allowed_letters)
        if parsed is not None:
            return parsed
        parsed = _parse_candidate(lines[0], options, allowed_letters)
        if parsed is not None:
            return parsed

    return None


def classify_error_mode(response: Any, text: str, pred: Optional[str], gold: str) -> str:
    """Classify a rollout into a single, mutually exclusive error mode.

    The buckets are diagnostic: ``correct``/``wrong_answer`` cover responses where we
    could identify the model's choice, while the remaining buckets explain *why* no
    valid choice was recovered (refusal, truncated/early termination, a ``<choice>``
    tag whose content matched nothing, or a response that ignored the format entirely).
    """
    if pred is not None:
        return "correct" if pred == gold else "wrong_answer"

    # No parseable answer below. Explicit API-level refusals are the most definitive.
    if _has_refusal_content(response):
        return "refusal"

    # Truncation / unfinished generation: the model never reached a final choice.
    if _is_truncated(response) or _has_unclosed_think(text):
        return "early_termination"

    if REFUSAL_PATTERN.search(text):
        return "refusal"

    # A well-formed <choice> tag was emitted, but its content matched no option.
    if _choice_tag_present(text):
        return "malformed_choice"

    # Anything else: no recognizable answer and the required format was not followed.
    return "format_violation"


def _has_refusal_content(response: Any) -> bool:
    for item in getattr(response, "output", None) or []:
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            if getattr(part, "type", None) == "refusal" or (isinstance(part, dict) and part.get("type") == "refusal"):
                return True
    return False


def _is_truncated(response: Any) -> bool:
    if getattr(response, "status", None) == "incomplete":
        return True
    details = getattr(response, "incomplete_details", None)
    reason = None
    if isinstance(details, dict):
        reason = details.get("reason")
    elif details is not None:
        reason = getattr(details, "reason", None)
    if reason == "max_output_tokens":
        return True
    for item in getattr(response, "output", None) or []:
        status = item.get("status") if isinstance(item, dict) else getattr(item, "status", None)
        if status == "incomplete":
            return True
    return False


def _has_unclosed_think(text: str) -> bool:
    return len(THINK_OPEN_PATTERN.findall(text)) > len(THINK_CLOSE_PATTERN.findall(text))


def _choice_tag_present(text: str) -> bool:
    choices_spans = [match.span() for match in CHOICES_BLOCK_PATTERN.finditer(text)]
    for match in CHOICE_TAG_PATTERN.finditer(text):
        if not any(start <= match.start() < end for start, end in choices_spans):
            return True
    return False


def _answer_candidates(text: str) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    choices_spans = [match.span() for match in CHOICES_BLOCK_PATTERN.finditer(text)]
    for match in XML_ANSWER_PATTERN.finditer(text):
        tag = match.group(1).lower()
        if tag == "choice" and any(start <= match.start() < end for start, end in choices_spans):
            continue
        candidates.append((match.start(), match.group(2)))
    candidates.extend(_boxed_candidate_matches(text))
    for pattern in (ANSWER_LINE_PATTERN, INLINE_ANSWER_PATTERN, ANSWER_PHRASE_PATTERN):
        candidates.extend((match.start(), match.group(1)) for match in pattern.finditer(text))
    return candidates


def normalize_chemistry_text(value: Any) -> str:
    value = html.unescape(str(value))
    value = unicodedata.normalize("NFKC", value).translate(SUBSCRIPT_TRANSLATION)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Cf")
    return " ".join(value.split())


def _last_parseable_candidate(
    candidates: list[str],
    options: list[dict[str, str]],
    allowed_letters: set[str],
) -> Optional[str]:
    for candidate in reversed(candidates):
        parsed = _parse_candidate(candidate, options, allowed_letters)
        if parsed is not None:
            return parsed
    return None


def _parse_candidate(candidate: str, options: list[dict[str, str]], allowed_letters: set[str]) -> Optional[str]:
    candidate = candidate.strip()
    if not candidate:
        return None

    letter_match = _parse_answer_letter(candidate, allowed_letters)
    if letter_match is not None:
        return letter_match

    option_match = _parse_option_text(candidate, options, allowed_letters)
    if option_match is not None:
        return option_match

    candidate = _strip_wrappers(candidate)
    if not candidate:
        return None

    letter_match = _parse_answer_letter(candidate, allowed_letters)
    if letter_match is not None:
        return letter_match

    return _parse_option_text(candidate, options, allowed_letters)


def _parse_answer_letter(candidate: str, allowed_letters: set[str]) -> Optional[str]:
    if len(candidate) == 1 and candidate.isalpha():
        letter = candidate.upper()
        return letter if letter in allowed_letters else None

    option_letter = re.fullmatch(r"(?i)(?:option|choice)\s+([A-Z])\s*[\)\].:]?", candidate)
    if option_letter:
        letter = option_letter.group(1).upper()
        if letter in allowed_letters:
            return letter

    leading_letter = re.match(r"(?i)^([A-Z])[\)\].:](?:\s|$)", candidate)
    if leading_letter:
        letter = leading_letter.group(1).upper()
        if letter in allowed_letters:
            return letter

    # Handle strings like "A." or "(B)".
    letter_match = re.fullmatch(r"[^A-Za-z]*([A-Za-z])[^A-Za-z]*", candidate)
    if letter_match:
        letter = letter_match.group(1).upper()
        if letter in allowed_letters:
            return letter

    return None


def _parse_option_text(candidate: str, options: list[dict[str, str]], allowed_letters: set[str]) -> Optional[str]:
    candidate_norm = normalize_chemistry_text(candidate)
    for entry in options:
        for letter, option_text in entry.items():
            if letter.upper() in allowed_letters and normalize_chemistry_text(str(option_text)) == candidate_norm:
                return letter.upper()
    return None


def _strip_wrappers(candidate: str) -> str:
    candidate = candidate.strip(" \t\r\n`*_\"'")
    boxed = BOXED_PATTERN.fullmatch(candidate)
    if boxed:
        candidate = boxed.group(1).strip()

    while True:
        text_wrapper = re.fullmatch(r"\\(?:text|mathrm|operatorname)\{\s*(.*?)\s*\}", candidate, re.S)
        if not text_wrapper:
            break
        candidate = text_wrapper.group(1).strip()

    while len(candidate) >= 2 and candidate[0] in "([{\"'<" and candidate[-1] in ")]}\"'>":
        candidate = candidate[1:-1].strip(" \t\r\n`*_\"'")

    return candidate.strip(" \t\r\n`*_\"'.;,")


def _boxed_candidates(text: str) -> list[str]:
    return [candidate for _, candidate in _boxed_candidate_matches(text)]


def _boxed_candidate_matches(text: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    search_pos = 0
    while True:
        match = re.search(r"\\boxed\s*\{", text[search_pos:])
        if not match:
            return matches

        brace_start = search_pos + match.end() - 1
        match_start = search_pos + match.start()
        depth = 1
        pos = brace_start + 1
        while pos < len(text) and depth:
            if text[pos] == "\\":
                pos += 2
                continue
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1

        if depth == 0:
            matches.append((match_start, text[brace_start + 1 : pos - 1]))
            search_pos = pos
        else:
            return matches


def _options_from_body(body: BunsenChemVerifyRequest) -> list[dict[str, str]]:
    if body.options:
        return [{letter: text for letter, text in entry.items() if text is not None} for entry in body.options]

    options: list[dict[str, str]] = []
    for index, choice in enumerate(body.choices or []):
        if index >= 26:
            break
        fallback_letter = chr(ord("A") + index)
        if isinstance(choice, str):
            options.append({fallback_letter: choice})
            continue

        letter = (
            choice.get("letter") or choice.get("label") or choice.get("key") or choice.get("id") or fallback_letter
        )
        text = choice.get("text") or choice.get("content") or choice.get("value") or choice.get("choice")
        if isinstance(letter, str) and isinstance(text, str) and len(letter.strip()) == 1:
            options.append({letter.strip().upper(): text})
    return options


def _expected_answer_letter(expected_answer: str, options: list[dict[str, str]]) -> str:
    expected = _strip_wrappers(expected_answer).upper()
    if len(expected) == 1 and expected.isalpha():
        return expected
    parsed = _parse_candidate(expected_answer, options, _get_allowed_letters_from_options(options))
    return parsed or expected


def _metadata_value(body: BunsenChemVerifyRequest, key: str) -> Any:
    direct = getattr(body, key, None)
    if direct is not None:
        return direct
    metadata = body.metadata or {}
    return metadata.get(key)


def _first_value(rollouts: list[dict[str, Any]], key: str) -> Any:
    for rollout in rollouts:
        value = rollout.get(key)
        if value:
            return value
    return None


def _grouped_metrics(tasks: list[list[dict[str, Any]]], group_key: str, prefix: str) -> dict[str, float]:
    buckets: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for rollouts in tasks:
        value = _first_value(rollouts, group_key)
        if value:
            buckets[_metric_segment(value)].append(rollouts)

    metrics: dict[str, float] = {}
    for value, subset_tasks in buckets.items():
        subset_metrics, _, _, _ = compute_pass_majority_metrics(
            subset_tasks,
            score_fn=lambda r: {"accuracy": r["reward"]},
            answer_key="extracted_answer",
        )
        for metric_key, metric_value in subset_metrics.items():
            if metric_key != "per_sample_aggregate":
                metrics[f"{prefix}/{value}/{metric_key}"] = metric_value
    return metrics


def _grouped_bct_subfield_metrics(tasks: list[list[dict[str, Any]]]) -> dict[str, float]:
    buckets: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for rollouts in tasks:
        field = _first_value(rollouts, "bct_field")
        subfield = _first_value(rollouts, "bct_subfield")
        if field and subfield:
            buckets[f"{_metric_segment(field)}/{_metric_segment(subfield)}"].append(rollouts)

    metrics: dict[str, float] = {}
    for value, subset_tasks in buckets.items():
        subset_metrics, _, _, _ = compute_pass_majority_metrics(
            subset_tasks,
            score_fn=lambda r: {"accuracy": r["reward"]},
            answer_key="extracted_answer",
        )
        for metric_key, metric_value in subset_metrics.items():
            if metric_key != "per_sample_aggregate":
                metrics[f"by_bct_subfield/{value}/{metric_key}"] = metric_value
    return metrics


def _error_mode_metrics(tasks: list[list[dict[str, Any]]]) -> dict[str, float]:
    counts: dict[str, int] = {mode: 0 for mode in ERROR_MODES}
    total = 0
    for rollouts in tasks:
        for rollout in rollouts:
            mode = rollout.get("error_mode") or _fallback_error_mode(rollout)
            counts[mode] = counts.get(mode, 0) + 1
            total += 1

    metrics: dict[str, float] = {}
    for mode in counts:
        count = counts[mode]
        metrics[f"error_modes/{mode}/count"] = float(count)
        metrics[f"error_modes/{mode}"] = (100.0 * count / total) if total else 0.0
    return metrics


def _fallback_error_mode(rollout: dict[str, Any]) -> str:
    """Best-effort mode for rollouts produced before error_mode was recorded."""
    if rollout.get("reward"):
        return "correct"
    if rollout.get("extracted_answer") is not None:
        return "wrong_answer"
    return "format_violation"


def _metric_segment(value: Any) -> str:
    segment = str(value).strip()
    segment = re.sub(r"\s+", "_", segment)
    segment = segment.replace("/", "__")
    segment = re.sub(r"[^A-Za-z0-9_.:+-]+", "_", segment)
    return segment.strip("_") or "unknown"


if __name__ == "__main__":
    BunsenChemResourcesServer.run_webserver()
