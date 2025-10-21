# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import Any, Literal, Optional

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class MCQAResourcesServerConfig(BaseResourcesServerConfig):
    pass


class MCQARunRequest(BaseRunRequest):
    uuid: Optional[str] = None
    # Preferred dataset format: top-level `metadata` carries arbitrary data and
    # is not interpreted by the verifier. Only the fields below are used for
    # grading.
    options: Optional[list[dict[str, str]]] = None
    expected_answer: Optional[str] = None
    # Optional additional metadata for the request; if provided, may contain
    # fields like options/expected_answer as an alternative location.
    metadata: Optional[dict[str, Any]] = None
    # Grading mode selector
    grading_mode: Literal[
        "strict_single_letter_boxed",
        "lenient_boxed",
        "lenient_answer_colon",
    ] = "strict_single_letter_boxed"
    # Template metadata with custom regex support
    template_metadata: Optional[dict[str, Any]] = None


class MCQAVerifyRequest(MCQARunRequest, BaseVerifyRequest):
    pass


class MCQAVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    # body.response.output is a list of union types; we only want assistant message texts
    # TODO: @fsoares should we just assume we are always receiving the last message only? Not sure if this is always true.
    texts: list[str] = []
    for o in body.response.output:
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            # Each message has content which can be text parts; normalize to string
            content = getattr(o, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


def _extract_options_and_expected(
    body: MCQARunRequest,
) -> tuple[Optional[list[dict[str, str]]], Optional[str]]:
    # Only use explicit top-level fields for grading; ignore metadata entirely.
    return body.options, body.expected_answer  # type: ignore[return-value]


CHOICE_LETTER_PATTERN = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")
# Strict boxed: capture a single UPPERCASE letter, allowing non-letter chars around it inside the box
STRICT_BOXED_PATTERN = re.compile(r"\\boxed\{\s*[^A-Za-z]*([A-Z])[^A-Za-z]*\s*\}")
ANSWER_COLON_PATTERN = re.compile(r"(?i)answer\s*:\s*(.+)")


def _parse_answer_letter_strict_boxed(text: str, allowed_letters: set[str]) -> tuple[Optional[str], str, bool]:
    parsed_text = text
    m = STRICT_BOXED_PATTERN.search(text)
    if not m:
        return None, parsed_text, True
    letter = m.group(1).upper()
    if letter not in allowed_letters:
        return None, parsed_text, True
    return letter, parsed_text, False


BOXED_CONTENT_PATTERN = re.compile(r"\\boxed\{\s*(.*?)\s*\}", re.S)
LATEX_TEXT_WRAP_PATTERN = re.compile(r"\\text\{\s*(.*?)\s*\}", re.S)


def _strip_latex_wrappers(s: str) -> str:
    """Remove successive \\text{...} wrappers from a LaTeX string."""
    while True:
        m = LATEX_TEXT_WRAP_PATTERN.fullmatch(s)
        if not m:
            break
        s = m.group(1)
    return s


def _normalize_for_match(s: str) -> str:
    """Lowercase and collapse whitespace for robust substring/equality checks."""
    return " ".join(s.lower().split())


def _match_option_text(text: str, options: list[dict[str, str]], allowed_letters: set[str]) -> Optional[str]:
    """Match boxed content against option texts and return the option letter.

    - Looks ONLY inside the first \boxed{...} region; returns None if absent.
    - Normalizes (lowercase, collapse whitespace) both boxed content and option texts.
    - Treats a match as substring containment of an option's text in the boxed content.
    - Returns the option letter only if EXACTLY ONE option matches; otherwise returns None.
    """
    # Only match within boxed content; if no boxed content, return None
    boxed = BOXED_CONTENT_PATTERN.search(text)
    if not boxed:
        return None
    inner = boxed.group(1)
    candidate_texts = [inner, _strip_latex_wrappers(inner)]
    normalized_candidates = [_normalize_for_match(t) for t in candidate_texts]

    # Build list of (letter, normalized_option_text)
    normalized_options: list[tuple[str, str]] = []
    for entry in options or []:
        for k, v in entry.items():
            if isinstance(k, str) and len(k) == 1 and k.upper() in allowed_letters:
                normalized_options.append((k.upper(), _normalize_for_match(v)))
                break

    matched_letters: set[str] = set()
    for cand in normalized_candidates:
        for letter, opt_norm in normalized_options:
            if opt_norm and opt_norm in cand:
                matched_letters.add(letter)
    if len(matched_letters) == 1:
        return next(iter(matched_letters))
    return None


def _parse_answer_with_custom_regex(
    text: str, regex_pattern: str, allowed_letters: set[str], options: Optional[list[dict[str, str]]]
) -> Optional[str]:
    """Parse answer using custom regex from template_metadata.

    Uses rightmost (last) match to handle reasoning before final answer.
    Case-insensitive matching to handle capitalization variations.

    When using template_metadata with custom regex, we trust the regex pattern
    and allow extracted letters even if options metadata is incomplete.
    """
    try:
        # Use IGNORECASE flag and findall to get all matches
        matches = re.findall(regex_pattern, text, re.IGNORECASE)
        if not matches:
            return None

        # Take the LAST match (rightmost)
        captured = matches[-1].strip().upper()

        # Try direct letter match first
        if len(captured) == 1 and captured.isalpha():
            # If we have options metadata, validate against it
            if allowed_letters and captured in allowed_letters:
                return captured
            # If options metadata is missing/incomplete, trust the regex
            # This handles cases where template_metadata regex is used but options are incomplete
            elif not allowed_letters:
                return captured
            # If captured letter is not in allowed_letters but allowed_letters exists,
            # it might be a data quality issue - still return it when using template_metadata
            else:
                # Trust the regex when using template_metadata (this function is only called for template_metadata)
                return captured

        # Try matching against option text (normalized)
        normalized_captured = _normalize_for_match(captured)
        for entry in options or []:
            for k, v in entry.items():
                if k.upper() in allowed_letters and _normalize_for_match(v) == normalized_captured:
                    return k.upper()

        return None
    except re.error:
        # Invalid regex pattern, return None
        return None


class MCQAResourcesServer(SimpleResourcesServer):
    config: MCQAResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: MCQAVerifyRequest) -> MCQAVerifyResponse:
        text = _extract_last_assistant_text(body)
        # Pull options/expected_answer from dataset-style metadata if available
        options, expected_answer = _extract_options_and_expected(body)
        # Derive allowed letters from option keys
        allowed_letters = _get_allowed_letters_from_options(options)

        pred: Optional[str] = None

        # Check for template_metadata first (highest priority)
        if body.template_metadata and "output_regex" in body.template_metadata:
            regex_pattern = body.template_metadata["output_regex"]
            pred = _parse_answer_with_custom_regex(text, regex_pattern, allowed_letters, options)

        # Fallback to existing grading_mode logic if template_metadata didn't work
        if pred is None:
            if body.grading_mode == "strict_single_letter_boxed":
                pred, _, _ = _parse_answer_letter_strict_boxed(text, allowed_letters)
            elif body.grading_mode == "lenient_boxed":
                # Try strict boxed first
                pred, _, _ = _parse_answer_letter_strict_boxed(text, allowed_letters)
                if pred is None:
                    # Then try to match option text inside boxed content only
                    letter_from_text = _match_option_text(text, options, allowed_letters)
                    if letter_from_text is not None:
                        pred = letter_from_text
            elif body.grading_mode == "lenient_answer_colon":
                # Look for Answer: <...>
                m = ANSWER_COLON_PATTERN.search(text)
                if m:
                    candidate = _strip_latex_wrappers(m.group(1)).strip()
                    # Letter case
                    if len(candidate) == 1 and candidate.isalpha():
                        letter_up = candidate.upper()
                        if letter_up in allowed_letters:
                            pred = letter_up
                    # Option text equality (normalized)
                    if pred is None:
                        cand_norm = _normalize_for_match(candidate)
                        for entry in options or []:
                            for k, v in entry.items():
                                k_up = k.upper()
                                if k_up in allowed_letters and _normalize_for_match(v) == cand_norm:
                                    pred = k_up
                                    break
                            if pred is not None:
                                break

        gold = (expected_answer or "").strip().upper()
        is_correct = (pred == gold) if (pred is not None and gold) else False
        reward = 1.0 if is_correct else 0.0

        return MCQAVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            expected_answer=gold,
            extracted_answer=pred,
        )


def _get_allowed_letters_from_options(
    options: Optional[list[dict[str, str]]],
) -> set[str]:
    """Collect uppercase option letters from list of single-key dicts."""
    letters: set[str] = set()
    if options:
        for entry in options:
            for k in entry.keys():
                if isinstance(k, str) and len(k) == 1 and k.isalpha():
                    letters.add(k.upper())
                break
    return letters


if __name__ == "__main__":
    MCQAResourcesServer.run_webserver()
