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

import json
import os
import re
import string
from collections import Counter
from fractions import Fraction
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from pydantic import BaseModel, Field, ValidationError


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from ..data_loader import LLM_JUDGE_QUESTION_PROMPT


if load_dotenv:
    load_dotenv()


class JudgeResponse(BaseModel):
    """
    Defines the expected JSON structure for the LLM Judge's response.
    """

    verdict: Literal["YES", "NO"] = Field(..., description="The binary decision from the judge.")
    reasoning: str = Field(..., description="The explanation for the decision.")


def check_relation(count, relation: str, expected: int) -> Tuple[bool, Optional[str]]:
    """
    Check if a count satisfies a relation against an expected value.

    Args:
        count: The actual count to check (must be numeric)
        relation: One of "at least", "equal to", "less than"
        expected: The expected value to compare against (must be numeric)

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid,
        or an error string if validation fails.
    """
    # Validate count is numeric
    if not isinstance(count, (int, float)):
        return (False, f"Invalid count type: expected numeric, got {type(count).__name__}.")

    # Validate expected is numeric
    if not isinstance(expected, (int, float)):
        return (False, f"Invalid expected value type: expected numeric, got {type(expected).__name__}.")

    # Validate relation
    if relation == "at least":
        return (count >= expected, None)
    elif relation == "equal to":
        return (count == expected, None)
    elif relation == "less than":
        return (count < expected, None)
    else:
        return (False, f"Invalid relation: '{relation}'. Must be 'at least', 'equal to', or 'less than'.")


# =========================
# Brazilian Portuguese (PT-BR) language helpers
# =========================

# Latin letters incl. Portuguese diacritics (no Ñ, added Ç, Ã, Õ, À, circumflexes)
_PT_LETTERS = "A-Za-zÁÉÍÓÚÜáéíóúüÀàÃãÕõÂâÊêÔôÇç"

# Digits
_PT_DIGITS = "0-9"

# In-word joiners (apostrophes are used in pt-br like d'água, hyphens very common)
_PT_JOINERS = r"['’´\-\.]"

# Word char class for boundaries
_PT_WORDCHARS = f"{_PT_LETTERS}{_PT_DIGITS}"

# Sentence-ending punctuation (Standard + ASCII). PT doesn't use ¿ or ¡.
_PT_SENT_END = r"\.\!\?\…"

# Portuguese/opening punctuation we also need to consider for cleaning
# (Guillemets «» exist but “” are more common in BR, inverted marks removed)
_PT_PUNCT_EXTRA = r"«»“”„\"'—–…"

# Abbreviation list where a dot should NOT split sentences (PT-BR context)
_PT_ABBREVIATIONS = {
    # titles
    "Sr.",
    "Sra.",
    "Srta.",
    "Dr.",
    "Dra.",
    "Eng.",
    "Prof.",
    "Profa.",
    "Gen.",
    "Cel.",
    "Cap.",
    "Ten.",
    # misc
    "pág.",
    "págs.",
    "aprox.",
    "etc.",
    "ex.",
    "p.ex.",
    "obs.",
    "tel.",
    "av.",
    "r.",
    # formal pronouns
    "V.Exa.",
    "V.Sa.",
    "V.M.",
    # ordinal indicators/companies
    "Nº.",
    "n.º",
    "nº",
    "núm.",
    "Ltda.",
    "S.A.",
    "Cia.",
}

# Precompiled token regex: words may contain diacritics, digits, and inner joiners
_RE_PT_WORD = re.compile(rf"[{_PT_WORDCHARS}]+(?:{_PT_JOINERS}[{_PT_WORDCHARS}]+)*", flags=re.UNICODE)

# Numbered-item (supports 1. 1) 1º. 1.º 1ª 1.- )
# Added support for feminine ordinal indicator (ª) common in PT
_RE_PT_NUMBERED = re.compile(
    r"^\s*(?:\d+|[0-9]+)(?:[\.\)\-º°ª]*|\.º|\.°|º\.|°\.|ª\.)\s+", flags=re.MULTILINE | re.UNICODE
)

# Bullet points
_RE_PT_BULLET = re.compile(r"^[*-]\s+", flags=re.MULTILINE)

# Sentence end finder (one or more enders; treat ellipsis as ender)
_RE_PT_SENT_ENDERS = re.compile(rf"[{_PT_SENT_END}]+", flags=re.UNICODE)


def _pt_casefold(s: str) -> str:
    """Case-insensitive comparison helper that preserves diacritics."""
    return s.casefold()


def _strip_opening_quotes_punct(s: str) -> str:
    return s.lstrip(string.punctuation + _PT_PUNCT_EXTRA + " ").rstrip()


def _unicode_boundary_phrase_pattern(phrase: str) -> str:
    """
    Build Portuguese-aware boundary pattern using custom char class.
    """
    escaped = [re.escape(part) for part in phrase.split()]
    joined = r"\s+".join(escaped)
    return rf"(?<![{_PT_WORDCHARS}]){joined}(?![{_PT_WORDCHARS}])"


def _mask_abbrev_dots(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace periods in known abbreviations with a token to avoid sentence split.
    """
    token = "§DOT§"
    mapping = {}

    def repl(m):
        w = m.group(0)
        masked = w.replace(".", token)
        mapping[masked] = w
        return masked

    # Build pattern of the abbreviations, escape dots
    abbr_sorted = sorted(_PT_ABBREVIATIONS, key=len, reverse=True)
    if not abbr_sorted:
        return text, mapping

    pat = r"(?<!\w)(" + "|".join(re.escape(a) for a in abbr_sorted) + r")(?!\w)"
    return re.sub(pat, repl, text, flags=re.IGNORECASE), mapping


def _unmask_abbrev_dots(text: str, mapping: Dict[str, str]) -> str:
    for masked, real in mapping.items():
        text = text.replace(masked, real)
    return text


def judge_llm_api(
    user_content, system_content="You are a chatbot", temperature=0.7, seed=42, top_p=1, top_k=40, max_tokens=10000
):
    url = os.getenv("OPENROUTER_API_BASE_URL")
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "anthropic/claude-sonnet-4.5",
        "messages": [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}],
        "temperature": temperature,
        "seed": seed,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }
    print("Calling OpenRouter API")
    response = requests.post(f"{url}/chat/completions", headers=headers, json=payload)
    if response.status_code in (200, 201):
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"


# llm_judge questions validation
def validate_custom_llm_judge(response: str, question_text: str) -> Tuple[bool, str]:
    """
    Validates a response against a free-form LLM Judge question.
    Returns (True, reasoning) if verdict is YES, otherwise (False, reasoning).
    """
    try:
        judge_prompt = LLM_JUDGE_QUESTION_PROMPT.format(question=question_text, model_response=response)
        evaluation = judge_llm_api(user_content="Evaluate the response.", system_content=judge_prompt)
        # Parse Response
        evaluation = evaluation.strip()

        # Handle Markdown code blocks
        if evaluation.startswith("```"):
            evaluation = re.sub(r"^```(?:\w+)?\s*", "", evaluation, flags=re.DOTALL)
            evaluation = re.sub(r"\s*```$", "", evaluation, flags=re.DOTALL)
        # Extract JSON
        json_match = re.search(r"(\{.*\})", evaluation, re.DOTALL)
        if json_match:
            evaluation = json_match.group(1)
        json_data = json.loads(evaluation)
        judge_response = JudgeResponse(**json_data)
        # Determine Status
        flag = judge_response.verdict == "YES"
        message = judge_response.reasoning
        return flag, message
    except (json.JSONDecodeError, ValidationError) as e:
        return False, f"Error parsing Judge response: {e}. Raw: {evaluation}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def is_strict_alternating(word: str) -> bool:
    """Check if a word has strictly alternating case."""
    prev_is_upper = None
    for ch in word:
        if ch.isalpha():
            cur_is_upper = ch.isupper()
            if prev_is_upper is not None and cur_is_upper == prev_is_upper:
                return False
            prev_is_upper = cur_is_upper
        else:
            prev_is_upper = None
    return True


def count_numbered_items(response: str) -> int:
    """Count number of numbered items in response."""
    return len(_RE_PT_NUMBERED.findall(response))


def count_bullet_points(response: str) -> int:
    """Count number of bullet points in response."""
    return len(_RE_PT_BULLET.findall(response))


def count_placeholders(response: str) -> int:
    """Count number of placeholders in response."""
    return len(re.findall(r"\[.*?\]", response))


def count_all_caps_words(response: str) -> int:
    """Count number of all-caps words in response."""
    return sum(1 for w in response.split() if w.isupper())


def count_lowercase_words(response: str) -> int:
    """Count number of lowercase words in response."""
    return sum(1 for w in response.split() if w.islower())


def keyword_frequency(response: str, keyword: str) -> int:
    """
    Portuguese-aware full-word/phrase frequency using custom boundaries (handles diacritics).
    Rejects tokens with invalid outer punctuation.
    """
    keyword = keyword.strip()
    if not keyword:
        return 0
    # Build boundary-aware pattern
    pattern = _unicode_boundary_phrase_pattern(keyword)
    return len(re.findall(pattern, response, flags=re.UNICODE | re.IGNORECASE))


def is_first_letter_cap(token: str) -> bool:
    """
    Check if token follows first-letter-capitalization rule (Unicode-aware).
    """
    first_alpha_seen = False
    first = token[0]
    if first.isdigit():
        return all((not ch.isalpha()) or ch.islower() for ch in token[1:])
    if len(token) == 1:
        if token.isalpha():
            return first.isupper()
        else:
            return True
    for ch in token:
        if ch.isalpha():
            if not first_alpha_seen:
                if not ch.isupper():
                    return False
                first_alpha_seen = True
            else:
                if not ch.islower():
                    return False
    return True


def parse_fraction_or_inf(input_str: str):
    """
    Parses a string into a Fraction object or float('inf').
    """
    if isinstance(input_str, (int, float)):
        return input_str

    if not isinstance(input_str, str):
        raise TypeError(f"Input must be a string, not {type(input_str)}")
    input_str = input_str.strip().lower()
    if input_str == "inf":
        return float("inf")

    try:
        frac = Fraction(input_str)
        return frac
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Invalid input: '{input_str}'. Not a valid fraction or 'inf'.")


def extract_clean_sentences(text: str) -> List[str]:
    """
    Portuguese-aware sentence extraction.
    - Removes markdown tables, rules, list markers.
    - Preserves common abbreviations so their trailing period doesn't split a sentence.
    - Splits on '.', '!', '?', '…' (ellipsis).
    """
    # Remove markdown tables
    table_pattern = r"(?:^\s*\|.*\|.*\n){2,}"
    cleaned_text = re.sub(table_pattern, "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    rule_pattern = r"^\s*([*_-])\s*\1\s*\1+\s*$"
    text = re.sub(rule_pattern, "", cleaned_text, flags=re.MULTILINE)

    # Mask abbreviations
    text, mapping = _mask_abbrev_dots(text)

    all_sentences = []
    for line in text.split("\n"):
        line = line.lstrip()
        # Remove bullets/numbering/headings
        # Pattern updated for PT markers including ª
        line = re.sub(r"^\s*(?:[\-\*\+]\s+|(?:\d+|[0-9]+)(?:[\.\)\-º°ª]*|\.º|\.°|º\.|°\.|ª\.)\s+|#+\s+)", "", line)

        if not line:
            continue

        # Split by enders (., !, ?, …)
        parts = _RE_PT_SENT_ENDERS.split(line)
        for sentence in parts:
            s = sentence.strip()
            if s:
                s = _unmask_abbrev_dots(s, mapping)
                all_sentences.append(s)
    return all_sentences


def extract_clean_words(response: str) -> List[str]:
    """
    Portuguese-aware tokenizer: letters with diacritics, digits, allows inner ' - . ’
    Returns lowercased tokens (casefolded).
    """
    text_without_lists = re.sub(
        r"^\s*(?:\d+|[0-9]+)(?:[\.\)\-º°ª]*|\.º|\.°|º\.|°\.|ª\.)\s+", "", response, flags=re.MULTILINE
    )
    return [_pt_casefold(w) for w in _RE_PT_WORD.findall(text_without_lists)]


def analyze_lists(text: str, pattern: str) -> list[dict]:
    """
    Analyzes a text to find lists (numbered or bulleted) based on a
    provided regex pattern, noting their nesting level and item count.
    """
    lists_found = []
    current_list_stack = []  # Tracks lists at different nesting levels

    item_pattern = re.compile(pattern, re.MULTILINE)
    all_items = item_pattern.finditer(text)

    for item in all_items:
        indentation, marker, item_text = item.groups()
        indent_level = len(indentation.strip("\n"))

        while current_list_stack and indent_level < current_list_stack[-1]["indent"]:
            lists_found.append(current_list_stack.pop())

        if not current_list_stack or indent_level == current_list_stack[-1]["indent"]:
            if not current_list_stack:
                nesting_level = 1
                current_list_stack.append({"level": nesting_level, "indent": indent_level, "items": 1})
            else:
                current_list_stack[-1]["items"] += 1

        elif indent_level > current_list_stack[-1]["indent"]:
            nesting_level = current_list_stack[-1]["level"] + 1
            current_list_stack.append({"level": nesting_level, "indent": indent_level, "items": 1})

    lists_found.extend(current_list_stack)
    return lists_found


def find_markdown_tables(text: str) -> list[dict]:
    """
    Finds all markdown tables in a text and determines their dimensions.
    """
    tables_found = []
    lines = text.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # potential header row (must contain '|')
        if "|" not in line:
            i += 1
            continue
        # divider line immediately after the header
        if i + 1 >= len(lines):
            break
        divider = lines[i + 1].strip()

        if "|" not in divider or not re.match(r"^[\s|: -]+$", divider):
            i += 1
            continue

        header_cols = [col.strip() for col in line.split("|") if col.strip()]
        num_cols = len(header_cols)
        divider_cols = [col.strip() for col in divider.split("|") if col.strip()]
        if len(divider_cols) != num_cols:
            i += 1
            continue

        num_rows = 0
        j = i + 2
        while j < len(lines) and "|" in lines[j]:
            num_rows += 1
            j += 1

        tables_found.append({"rows": num_rows, "columns": num_cols})

        i = j

    return tables_found


def find_punctuations(text: str) -> list[str]:
    """
    Find sentence-ending punctuation sequences. Includes '…' (ellipsis).
    """
    cleaned_text = re.sub(
        r"^\s*(?:[\-\*\+]\s+|(?:\d+|[0-9]+)(?:[\.\)\-º°ª]*|\.º|\.°|º\.|°\.|ª\.)\s+|#+\s+)",
        "",
        text,
        flags=re.MULTILINE,
    )
    punctuations = _RE_PT_SENT_ENDERS.findall(cleaned_text)
    return punctuations


def extract_clean_paragraphs(text: str) -> List[str]:
    """
    Extracts clean paragraphs from a text by removing markdown elements.
    """
    # Remove custom title tags like <<Title>>
    cleaned_text = re.sub(r"^\s*<<.*>>\s*$", "", text, flags=re.MULTILINE)
    # Remove markdown tables
    table_pattern = r"(?:^\s*\|.*\|.*\n){2,}"
    cleaned_text = re.sub(table_pattern, "", cleaned_text, flags=re.MULTILINE)
    # Remove markdown headings
    heading_pattern = r"^\s*#+\s+.*$"
    cleaned_text = re.sub(heading_pattern, "", cleaned_text, flags=re.MULTILINE)
    # Remove horizontal rules
    rule_pattern = r"^\s*([*_-])\s*\1\s*\1+\s*$"
    cleaned_text = re.sub(rule_pattern, "", cleaned_text, flags=re.MULTILINE)

    if not cleaned_text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", cleaned_text.strip())

    clean_paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return clean_paragraphs


def validate_instruction(
    response: str, inst_type: str, kwargs: Dict[str, Any], all_instructions: Dict = None
) -> Tuple[bool, str]:
    """Validate a response against a specific instruction type and its kwargs."""
    try:
        response = response.strip()
        if inst_type == "change_case:all_caps":
            # Unicode-aware
            return (response.isupper(), "No error" if response.isupper() else "Response is not all uppercase.")
        if inst_type == "change_case:lowercase":
            return (response.islower(), "No error" if response.islower() else "Response is not all lowercase.")
        if inst_type == "change_case:alternating":
            valid = all(is_strict_alternating(w) for w in response.split() if any(c.isalpha() for c in w))
            return (valid, "No error" if valid else "Response is not strictly alternating.")
        if inst_type == "change_case:first_letter_cap":
            # Unicode-aware token check
            valid = all(is_first_letter_cap(tok) for tok in response.split())
            return (
                valid,
                "No error"
                if valid
                else "Each word must start with one uppercase letter followed only by lowercase letters.",
            )
        if inst_type == "change_case:capital_word_frequency":
            count = count_all_caps_words(response)
            rel, val = kwargs["capital_relation"], kwargs["capital_frequency"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} all-cap words, found {count}.")
        if inst_type == "change_case:lowercase_word_frequency":
            count = count_lowercase_words(response)
            rel, val = kwargs["lowercase_relation"], kwargs["lowercase_frequency"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} lowercase words, found {count}.")
        if "_target" in inst_type:
            target = kwargs["target_string"].strip()
            pattern = _unicode_boundary_phrase_pattern(target)
            matches = re.findall(pattern, response, re.IGNORECASE | re.UNICODE)
            if not matches:
                return (False, f"Target '{target}' not found in response.")
            for match in matches:
                raw_text = match.strip('"').strip("'")
                if inst_type == "change_case:all_caps_target" and not raw_text.isupper():
                    return (False, f"'{raw_text}' should be ALL CAPS.")
                elif inst_type == "change_case:lowercase_target" and not raw_text.islower():
                    return (False, f"'{raw_text}' should be all lowercase.")
                elif inst_type == "change_case:alternating_target" and not is_strict_alternating(raw_text):
                    return (False, f"'{raw_text}' is not in alternating caps.")
                elif inst_type == "change_case:first_letter_cap_target" and not raw_text.istitle():
                    return (False, f"'{raw_text}' is not first-letter capitalized.")
            return (True, "No error")
        if inst_type == "detectable_content:number_placeholders":
            count = count_placeholders(response)
            rel, val = kwargs["relation"], kwargs["num_placeholders"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} placeholders, found {count}.")
        if inst_type == "detectable_content:postscript":
            marker = kwargs.get("postscript_marker", "PS:").strip()
            lines = response.splitlines()
            for line in reversed(lines):
                if line.strip():
                    last_line = line.strip()
                    break
            else:
                last_line = ""
            has_postscript = last_line.startswith(marker) and len(last_line) > len(marker)
            return (
                has_postscript,
                "No error"
                if has_postscript
                else f"Postscript must start with '{marker}' and contain content. Found: '{last_line}'",
            )
        if inst_type == "detectable_format:json_format":
            try:
                json_part = response[response.find("{") : response.rfind("}") + 1]
                json.loads(json_part)
                return (True, "No error")
            except:
                return (False, "Response is not valid JSON format.")
        if inst_type == "detectable_format:multiple_sections":
            splitter = (kwargs.get("section_splitter") or "").strip()
            rel = kwargs.get("relation")
            val = kwargs.get("num_sections")
            if not splitter:
                return (False, "section_splitter cannot be empty.")
            if re.search(r"[#*]", splitter):
                return (
                    False,
                    "section_splitter must be a plain section name without '#' or '*'. The validator adds Markdown headers automatically.",
                )
            splitter_clean = splitter
            header_re = re.compile(
                r"^\s*#{1,6}\s+" + re.escape(splitter_clean) + r"\s+\d+\b",
                re.MULTILINE | re.IGNORECASE,
            )
            sections = header_re.findall(response)
            count = len(sections)
            if count == 0:
                nospace_re = re.compile(
                    r"^\s*#{1,6}" + re.escape(splitter_clean) + r"\s+\d+\b",
                    re.MULTILINE | re.IGNORECASE,
                )
                if nospace_re.search(response):
                    return (False, f"Markdown headers require a space after '#'. Use e.g. '### {splitter_clean} 1'.")
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (
                valid,
                "No error"
                if valid
                else f"Expected {rel} {val} sections like '### {splitter_clean} 1', found {count}.",
            )
        if inst_type == "detectable_format:numbered_list":
            count = count_numbered_items(response)
            rel, val = kwargs["relation"], kwargs["num_numbered_items"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} numbered items, found {count}.")
        if inst_type == "detectable_format:number_bullet_lists":
            count = count_bullet_points(response)
            rel, val = kwargs["relation"], kwargs["num_bullets"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} bullet points, found {count}.")
        if inst_type == "detectable_format:title":
            line = response.splitlines()[0] if response.splitlines() else ""
            found_title = line.strip().startswith("<<") and line.strip().endswith(">>")
            return (found_title, "No error" if found_title else "Title not wrapped in << >> on first line.")
        if inst_type == "keywords:existence":
            missing = [kw for kw in kwargs["keywords"] if keyword_frequency(response, kw) == 0]
            return (not missing, "No error" if not missing else f"Missing keyword(s): {missing}")
        if inst_type == "keywords:frequency":
            keyword = kwargs["keyword"].strip()
            count = keyword_frequency(response, keyword)
            rel = kwargs["relation"]
            val = kwargs["frequency"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} of '{keyword}', found {count}.")
        if inst_type == "keywords:forbidden_words":
            present = [w for w in kwargs["forbidden_words"] if keyword_frequency(response, w)]
            return (not present, "No error" if not present else f"Forbidden words found: {present}")
        if inst_type == "keywords:letter_frequency":
            """
            Count letter frequency case-insensitively and include Portuguese accented variants.
            """
            base = kwargs["letter"]
            base_fold = _pt_casefold(base)
            # Portuguese mapping
            letter_map = {
                "a": "aáàãâAÁÀÃÂ",
                "c": "cçCÇ",
                "e": "eéêEÉÊ",
                "i": "iíIÍ",
                "o": "oóõôOÓÕÔ",
                "u": "uúüUÚÜ",
                "n": "nN",  # PT has no ñ
            }
            if base_fold in letter_map:
                pattern = f"[{letter_map[base_fold]}]"
                count = len(re.findall(pattern, response))
            else:
                count = response.lower().count(base_fold)
            rel, val = kwargs["let_relation"], kwargs["let_frequency"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} occurrences of '{base}', found {count}.")
        if inst_type == "punctuation:no_comma":
            # PT uses same comma glyph
            return ("," not in response, "No error" if "," not in response else "Commas found in response.")
        if inst_type == "length_constraints:number_characters":
            count = len(response)
            rel, val = kwargs["relation"], kwargs["num_chars"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} characters, found {count}.")
        if inst_type == "length_constraints:number_words":
            count = len(_RE_PT_WORD.findall(response))
            rel, val = kwargs["relation"], kwargs["num_words"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} words, found {count}.")
        if inst_type == "startend:start_checker":
            starts_correctly = (
                _strip_opening_quotes_punct(response).lower().startswith(kwargs.get("start_phrase", "").lower())
            )
            return (
                starts_correctly,
                "No error" if starts_correctly else "Response does not start with required phrase.",
            )
        if inst_type == "startend:end_checker":
            required = kwargs["end_phrase"].strip()
            ends_with_punctuation = required[-1] in (string.punctuation + _PT_PUNCT_EXTRA) if required else False

            actual_words = _strip_opening_quotes_punct(response).split()
            if not actual_words:
                return (False, "Empty response")

            if ends_with_punctuation:
                actual_phrase = " ".join(actual_words[-len(required.split()) :])
                if actual_phrase.lower() != required.lower():
                    return (False, f"End phrase mismatch: expected '{required}', but found '{actual_phrase}'")
            else:
                actual_phrase = " ".join(actual_words).rstrip(string.punctuation + _PT_PUNCT_EXTRA + " ")[
                    -len(required) :
                ]
                if actual_phrase.lower() != required.lower():
                    return (False, f"End phrase mismatch: expected '{required}', but found '{actual_phrase}'")
            return (True, "No error")
        if inst_type == "startend:wrap_checker":
            wrap = kwargs["wrap_phrase"]
            return (
                response.startswith(wrap) and response.endswith(wrap),
                "No error" if response.startswith(wrap) and response.endswith(wrap) else f"Not wrapped with: {wrap}",
            )

        if inst_type == "startend:quotation":
            return (
                response.startswith('"') and response.endswith('"'),
                "No error"
                if response.startswith('"') and response.endswith('"')
                else "Response not wrapped in double quotes.",
            )

        if inst_type == "change_case:case_ratio":
            try:
                minR = parse_fraction_or_inf(kwargs["min_fraction"])
                maxR = parse_fraction_or_inf(kwargs["max_fraction"])
            except (ValueError, ZeroDivisionError) as e:
                raise ValueError(f"Invalid fraction input: {e}")

            if minR > maxR:
                return (False, "Validation failed: Minimum ratio greater than maximum ratio.")
            lower_count = sum(1 for ch in response if ch.islower())
            upper_count = sum(1 for ch in response if ch.isupper())
            if lower_count == 0 and upper_count == 0:
                return (False, "Validation failed: No letters found in the string.")

            if upper_count == 0:
                ratio = float("inf")
                ratio_str = "inf"
            else:
                ratio = Fraction(lower_count, upper_count)
                ratio_str = f"{ratio.numerator}/{ratio.denominator}"
            valid = minR <= ratio <= maxR
            message = (
                f"Lowercase count: {lower_count}, Uppercase count: {upper_count}. "
                f"Ratio is {ratio_str}({float(ratio):.2f}). Required range: [{minR}({float(minR):.2f}), {maxR}({float(maxR):.2f})]."
            )
            return (valid, "No error" if valid else message)
        if inst_type == "change_case:first_letter_sentence":
            """
            All sentences must start with an uppercase letter (after stripping leading punctuation like « “ ).
            """
            sentences = extract_clean_sentences(response)
            if not sentences:
                return (True, "No sentences found to validate.")
            for sentence in sentences:
                # Stripping PT opening punctuations
                s = sentence.strip("()[]{}\"'“”‘’«»").lstrip()
                # find first alphabetic char
                first_alpha = next((c for c in s if c.isalpha()), "")
                if first_alpha and not first_alpha.isupper():
                    return (False, f"Fails at: '{sentence}'")
            return (True, "No error.")
        if inst_type == "change_case:last_letter":
            cleaned_text = re.sub(r"[.!?…]+$", "", response.strip())
            if not cleaned_text:
                return (False, "Empty response")
            words = cleaned_text.split()
            # Stripping PT closing punctuations
            last_word = words[-1].strip("()[]{}\"'«»“”")
            if not last_word:
                return (False, "Empty last word")
            last_char = last_word[-1]
            case_type = kwargs["case"]
            if case_type == "uppercase":
                valid = last_char.isupper()
            elif case_type == "lowercase":
                valid = last_char.islower()
            elif case_type == "digit":
                valid = last_char.isdigit()
            elif case_type == "special":
                valid = not last_char.isalnum()
            else:
                valid = False

            return (valid, "No error." if valid else f"Last character of the response: {last_char}")
        if inst_type == "change_case:vowel_consonant_balance":
            try:
                minR = parse_fraction_or_inf(kwargs["min_fraction"])
                maxR = parse_fraction_or_inf(kwargs["max_fraction"])
            except (ValueError, ZeroDivisionError) as e:
                raise ValueError(f"Invalid fraction input: {e}")

            if minR > maxR:
                return (False, "Validation failed: Minimum ratio greater than maximum ratio.")

            # PT Vowels incl. accents
            vowels = set("aáàãâeéêiíoóõôuúüAÁÀÃÂEÉÊIÍOÓÕÔUÚÜ")
            vowel_count = sum(1 for ch in response if ch.isalpha() and ch in vowels)
            consonant_count = sum(1 for ch in response if ch.isalpha() and ch not in vowels)
            if vowel_count == 0 and consonant_count == 0:
                return (False, "Validation failed: No letters found in the response.")
            if consonant_count == 0:
                ratio = float("inf")
                ratio_str = "inf"
            else:
                ratio = Fraction(vowel_count, consonant_count)
                ratio_str = f"{ratio.numerator}/{ratio.denominator}"
            valid = minR <= ratio <= maxR
            message = (
                f"Vowel count: {vowel_count}, Consonant count: {consonant_count}. "
                f"Ratio is {ratio_str}({float(ratio):.2f}). Required range: [{minR}({float(minR):.2f}), {maxR}({float(maxR):.2f})]."
            )
            return (valid, "No error" if valid else message)
        if inst_type == "detectable_format:number_paragraphs":
            cleaned_response = response.strip().replace("\r\n", "\n")
            paragraphs = extract_clean_paragraphs(response)
            actual_paragraph_count = len([p for p in paragraphs if p.strip()])
            if not cleaned_response:
                actual_paragraph_count = 0
            relation = kwargs["relation"]
            num_paragraphs = kwargs["num_paragraphs"]
            valid, err = check_relation(actual_paragraph_count, relation, num_paragraphs)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {actual_paragraph_count} paragraphs. Expected {num_paragraphs}"
                return (False, message)
            return (True, "No error.")
        if inst_type == "detectable_format:max_paragraph_length":
            max_chars = kwargs["max_chars"]
            paragraphs = extract_clean_paragraphs(response)
            for p in paragraphs:
                p = re.sub(
                    r"^\s*(?:[\-\*\+]\s+|(?:\d+|[0-9]+)(?:[\.\)\-º°ª]*|\.º|\.°|º\.|°\.|ª\.)\s+|#+\s+)", "", p.lstrip()
                )
                char_count = len(p.strip())
                if char_count > max_chars:
                    return (False, f"Found a paragraph containing {char_count} characters.\n '{p}'")
            return (True, "No error.")
        if inst_type == "detectable_format:sentences_per_paragraph":
            num_sentences = kwargs["num_sentences"]
            relation = kwargs["relation"]
            paragraphs = extract_clean_paragraphs(response)
            for p in paragraphs:
                sentences = extract_clean_sentences(p)
                sentence_count = len([s for s in sentences if s.strip()])
                if sentence_count == 0 and p.strip():
                    sentence_count = 1
                valid, err = check_relation(sentence_count, relation, num_sentences)
                if err is not None:
                    return (False, err)
                if not valid:
                    return (False, f"Found {sentence_count} sentences. Expected {num_sentences}\n '{p}'")
            return (True, "No error.")

        if inst_type == "length_constraints:sentence_length":
            sentences = extract_clean_sentences(response)
            max_words = kwargs["max_words"]
            if not sentences:
                return (True, "No sentences found to validate.")
            for s in sentences:
                word_count = len(extract_clean_words(s))
                if word_count > max_words:
                    return (False, f"Expected at most {max_words} words. Found {word_count} words in '{s}'")
            return (True, "No error.")

        if inst_type == "length_constraints:word_repetition":
            max_repeats = kwargs["max_repeats"]
            words = extract_clean_words(response)
            word_counts = Counter(words)
            for word, count in word_counts.items():
                if count > max_repeats:
                    return (False, f"Word '{word}' appears {count} times (limit {max_repeats})")
            return (True, "No error.")

        if inst_type == "length_constraints:unique_words":
            relation = kwargs["relation"]
            num_unique = kwargs["num_unique"]
            words = extract_clean_words(response)
            unique_words_count = len(set(words))
            valid, err = check_relation(unique_words_count, relation, num_unique)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {unique_words_count} unique words. Expected {relation} {num_unique}."
                return (False, message)
            return (True, "No error.")

        if inst_type == "punctuation:question_exclaim":
            is_valid = True
            relation = kwargs["relation"]
            num_marks = kwargs["num_marks"]
            # Portuguese uses standard marks. No inverted.
            punctuations = re.findall(r"[?!]", response)
            count = len(punctuations)
            valid, err = check_relation(count, relation, num_marks)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {count} marks. Expected {relation} {num_marks}."
                return (False, message)
            return (True, "No error.")
        if inst_type == "punctuation:no_period":
            # PT also uses ellipsis, but checking strictly for '.' char
            return ("." not in response, "No error" if "." not in response else "Periods found in response.")
        if inst_type == "punctuation:end_rule":
            allowed = kwargs["allowed"]
            punctuations = set(find_punctuations(response))
            for p in punctuations:
                if p not in allowed:
                    return (False, f"'{p}' not in the list of allowed punctuations.")
            return (True, "No error.")
        if inst_type == "keywords:alliteration":
            relation = kwargs["relation"]
            target = _pt_casefold(kwargs["target_letter"])
            num_alliteration = kwargs["num_alliteration"]
            words = extract_clean_words(response)

            def starts_with_target(w: str) -> bool:
                # Digraphs like 'ch', 'lh', 'nh' if needed, otherwise base char
                if len(target) > 1:
                    return w.startswith(target)
                return w.startswith(target)

            all_count = sum(1 for w in words if starts_with_target(w))
            valid, err = check_relation(all_count, relation, num_alliteration)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {all_count} alliteration words. Expected {relation} {num_alliteration}."
                return (False, message)
            return (True, "No error.")
        if inst_type == "keywords:palindrome_word":
            min_length = kwargs["min_length"]
            words = extract_clean_words(response)
            for word in words:
                wf = _pt_casefold(word)
                if len(wf) >= min_length and wf == wf[::-1]:
                    return (True, f"No error. Word: {word}")
            return (False, "No valid palindrome words found.")

        if inst_type == "keywords:positioning":
            keyword = _pt_casefold(kwargs["keyword"])
            position = kwargs["position"]
            words = extract_clean_words(response)
            if 0 <= position < len(words) and words[position] == keyword:
                return (True, "No error.")
            found = words[position] if 0 <= position < len(words) else "<out-of-range>"
            return (False, f"'{found}' found after {position} words instead of '{keyword}'.")

        if inst_type == "detectable_format:nested_list":
            min_depth = kwargs["min_depth"]
            num_subitems = kwargs["num_subitems"]
            bullet_pattern = r"^(\s*)([*+-])[ \t]+(.*)"
            # Pattern matching PT numbering
            numbered_pattern = r"^(\s*)((?:\d+|[0-9]+)(?:[\.\)\-º°ª]*|\.º|\.°|º\.|°\.|ª\.))[ \t]+(.*)"
            lists = analyze_lists(response, bullet_pattern) + analyze_lists(response, numbered_pattern)
            for lst in lists:
                if lst["level"] == min_depth and lst["items"] >= num_subitems:
                    return (True, "No error.")
            return (False, f"List at level {min_depth} with at least {num_subitems} items not found.")
        if inst_type == "detectable_format:table":
            min_rows = kwargs["min_rows"]
            min_cols = kwargs["min_cols"]
            tables = find_markdown_tables(response)
            for table in tables:
                if table["rows"] >= min_rows and table["columns"] >= min_cols:
                    return (True, "No error.")
            return (False, f"Could not find a table with at least {min_rows} rows and {min_cols} columns.")
        if inst_type == "detectable_format:heading_depth":
            levels = kwargs["levels"]
            if not levels:
                return (False, "No levels provided.")
            heading_pattern = re.compile(r"^\s*(#+)[ \t]+(.*)", re.MULTILINE)
            all_headings = heading_pattern.findall(response)
            all_headings = set([len(item[0]) for item in all_headings])
            for level in levels:
                if level not in all_headings:
                    return (False, f"Heading of level {level} not found")
            return (True, "No error.")

        if inst_type == "length_constraints:word_length":
            max_length = kwargs["max_length"]
            min_length = kwargs["min_length"]
            if min_length > max_length:
                return (False, "Validation failed: Minimum length greater than maximum length.")
            words = set(extract_clean_words(response))
            if not words:
                return (True, "No words found to validate.")
            shortest_word = min(words, key=len)
            longest_word = max(words, key=len)
            if len(shortest_word) < min_length:
                return (
                    False,
                    f"Validation failed: The word '{shortest_word}' with length {len(shortest_word)} is shorter than the minimum of {min_length}.",
                )
            if len(longest_word) > max_length:
                return (
                    False,
                    f"Validation failed: The word '{longest_word}' with length {len(longest_word)} is longer than the maximum of {max_length}.",
                )
            return (True, "No error.")
        if inst_type == "length_constraints:avg_word_length":
            min_ratio = kwargs["min_ratio"]
            max_ratio = kwargs["max_ratio"]
            if min_ratio > max_ratio:
                return (False, "Validation failed: Minimum length greater than maximum length.")
            words = extract_clean_words(response)
            if not words:
                is_valid = min_ratio == 0
                return (is_valid, "No words found to validate.")
            avg_count = sum(len(word) for word in words) / len(words)
            is_valid = min_ratio <= avg_count <= max_ratio
            return (
                is_valid,
                "No error"
                if is_valid
                else f"Found average of {avg_count}. Expected between {min_ratio} and {max_ratio}",
            )
        if inst_type == "detectable_format:sentence_count":
            relation = kwargs["relation"]
            num_sentences = kwargs["num_sentences"]
            sentence_count = len(extract_clean_sentences(response))
            valid, err = check_relation(sentence_count, relation, num_sentences)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {sentence_count} sentences. Expected {relation} {num_sentences}"
                return (False, message)
            return (True, "No error.")

        if inst_type == "length_constraints:paragraph_length":
            words_per_paragraph = kwargs["words_per_paragraph"]
            relation = kwargs["relation"]
            paragraphs = extract_clean_paragraphs(response)
            for p in paragraphs:
                words = extract_clean_words(p)
                word_count = len([s for s in words if s.strip()])
                valid, err = check_relation(word_count, relation, words_per_paragraph)
                if err is not None:
                    return (False, err)
                if not valid:
                    return (False, f"Found {word_count} words. Expected {relation} {words_per_paragraph}\n '{p}'")
            return (True, "No error.")

        if inst_type == "detectable_content:numeric_inclusion":
            num_numbers = kwargs["num_numbers"]
            relation = kwargs["relation"]
            num_count = sum(1 for ch in response if ch.isdigit())
            valid, err = check_relation(num_count, relation, num_numbers)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {num_count} digits. Expected {relation} {num_numbers}"
                return (False, message)
            return (True, "No error.")
        if inst_type == "detectable_format:sentence_endings":
            min_variants = kwargs["min_variants"]
            # Distinct types among sentence-ending punctuation sequences
            ending_punctuations = find_punctuations(response)
            distinct = set()
            for s in ending_punctuations:
                for ch in s:
                    if ch in ".!?…":
                        distinct.add(ch)
            if len(distinct) < min_variants:
                return (
                    False,
                    f"Found {len(distinct)} types of sentence-ending punctuations ({distinct}). Expected at least {min_variants}.",
                )
            return (True, "No error.")

        if inst_type == "keywords:vowel_count":
            num_vowels = kwargs["num_vowels"]
            relation = kwargs["relation"]
            # PT Vowels
            vowels = set("aáàãâeéêiíoóõôuúüAÁÀÃÂEÉÊIÍOÓÕÔUÚÜ")
            vowel_count = sum(1 for ch in response if ch in vowels)
            valid, err = check_relation(vowel_count, relation, num_vowels)
            if err is not None:
                return (False, err)
            if not valid:
                return (False, f"Found {vowel_count} vowels. Expected {relation} {num_vowels}.")
            return (True, "No error.")
        if inst_type == "keywords:consonant_count":
            num_consonants = kwargs["num_consonants"]
            relation = kwargs["relation"]
            # Consonants = alphabetic letters that are NOT PT vowels
            vowels = set("aáàãâeéêiíoóõôuúüAÁÀÃÂEÉÊIÍOÓÕÔUÚÜ")
            consonant_count = sum(1 for ch in response if ch.isalpha() and ch not in vowels)
            valid, err = check_relation(consonant_count, relation, num_consonants)
            if err is not None:
                return (False, err)
            if not valid:
                return (False, f"Found {consonant_count} consonants. Expected {relation} {num_consonants}.")
            return (True, "No error.")
    except Exception as e:
        return (False, f"Validation error: {str(e)}")
    return (False, "Invalid Instruction")
