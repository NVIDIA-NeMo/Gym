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
from ..validator import _get_strategy


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


_IT_LETTERS = "A-Za-zÀÈÉÌÒÙàèéìòù"


def _unicode_boundary_phrase_pattern(phrase: str) -> str:
    """
    Creates a regex pattern to match a phrase with word boundaries,
    allowing for multi-token phrases with flexible whitespace.
    """
    escaped = [re.escape(part) for part in phrase.split()]
    joined = r"\s+".join(escaped)
    return rf"(?<![{_IT_LETTERS}]){joined}(?![{_IT_LETTERS}])"


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
    # print("Judge Prompt: ", system_content)
    # print("Message: ", user_content)
    response = requests.post(f"{url}/chat/completions", headers=headers, json=payload)

    if response.status_code in (200, 201):
        data = response.json()
        # print("Response: ", data["choices"][0]["message"]["content"])
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
    return len(re.findall(r"^\s*\d+\.", response, re.MULTILINE))


def count_bullet_points(response: str) -> int:
    """Count number of bullet points in response. Supports Markdown format (* or -)."""
    # Match bullet points: * or - followed by space or tab, at start of line (with optional leading whitespace)
    return len(re.findall(r"^\s*[*-]\s+", response, re.MULTILINE))


def count_placeholders(response: str) -> int:
    """Count number of placeholders in response."""
    return len(re.findall(r"\[.*?\]", response))


def count_all_caps_words(response: str) -> int:
    """Count number of all-caps words in response."""
    return sum(1 for w in response.split() if w.isupper())


def count_lowercase_words(response: str) -> int:
    """Count number of lowercase words in response."""
    return sum(1 for w in response.split() if w.islower())


def keyword_frequency(response: str, keyword: str, language: str = "it") -> int:
    """Count frequency of a keyword in response, using Unicode-aware boundaries for Italian.
    Handles elisions by treating apostrophes as word separators.
    Example: "dell'Italia" contains "Italia", "L'amico" contains "amico"
    """
    strategy = _get_strategy(language)
    keyword_normalized = strategy.casefold(keyword.strip())
    response_normalized = strategy.casefold(response)

    # For CJK scripts, fall back to substring counting with non-letter/digit guards
    if strategy.word_script == "cjk":
        pattern = rf"(?<![\w]){re.escape(keyword_normalized)}(?![\w])"
        return len(re.findall(pattern, response_normalized, flags=re.UNICODE))

    # For Italian, handle elisions: apostrophes act as word separators
    # Tokenize the response to handle elisions properly
    # Tokenization splits "dell'Italia" into ["dell", "Italia"], so we can count "Italia" directly
    words = strategy.tokenize_words(response_normalized)

    # Strip punctuation from words before comparison (handles "amico." -> "amico")
    punctuation_chars = string.punctuation.replace("'", "")  # Keep apostrophes as they're handled by tokenization

    # Count exact matches (case-insensitive, normalized)
    # Strip punctuation from each word before comparing
    count = 0
    for w in words:
        # Remove punctuation from the word (both leading and trailing)
        w_clean = w.rstrip(punctuation_chars).lstrip(punctuation_chars)
        if strategy.casefold(w_clean) == keyword_normalized:
            count += 1

    return count


def is_first_letter_cap(token: str) -> bool:
    """
    Check if token follows first-letter-capitalization rule.
    For Italian, handles elisions: "L'Amico" -> True (A is capitalized after apostrophe).
    """
    first_alpha_seen = False
    first = token[0]
    if first.isdigit():
        return all((not ch.isalpha()) or ch.islower() or ch in "àèéìòù" for ch in token[1:])
    if len(token) == 1:
        if token.isalpha():
            return first.isupper() or first in "ÀÈÉÌÒÙ"
        else:
            return True

    for ch in token:
        if ch.isalpha() or ch in "ÀÈÉÌÒÙàèéìòù":
            if not first_alpha_seen:
                # First letter after apostrophe or start should be uppercase
                if not (ch.isupper() or ch in "ÀÈÉÌÒÙ"):
                    return False
                first_alpha_seen = True
            else:
                # Subsequent letters should be lowercase
                if not (ch.islower() or ch in "àèéìòù"):
                    return False
    return True


def parse_fraction_or_inf(input_str: str):
    """
    Parses a string into a Fraction object or float('inf').
    Handles 'inf' or formats like '1/0' as infinity.
    """
    if isinstance(input_str, (int, float)):
        return input_str  # Return numbers directly

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


def extract_clean_sentences(text: str, language: str = "it") -> List[str]:
    """
    Takes a raw text string and returns a clean list of sentences.
    This version correctly handles list items that do not end with punctuation
    by treating each cleaned line as a source for one or more sentences.
    Uses Italian language strategy for sentence delimiters and abbreviations.
    """
    strategy = _get_strategy(language)

    # Remove markdown tables
    table_pattern = r"(?:^\s*\|.*\|.*\n){2,}"
    cleaned_text = re.sub(table_pattern, "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    rule_pattern = r"^\s*([*_-])\s*\1\s*\1+\s*$"
    text = re.sub(rule_pattern, "", cleaned_text, flags=re.MULTILINE)

    all_sentences = []

    # Process the text line by line
    for line in text.split("\n"):
        # Clean the line by removing markdown markers and leading space
        line = line.lstrip()
        cleaned_line = re.sub(r"^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)", "", line)

        if not cleaned_line:
            continue

        # Use strategy's sentence_split which handles Italian abbreviations
        line_parts = strategy.sentence_split(cleaned_line)

        # 3. Add the resulting parts to our main list after cleaning them.
        for sentence in line_parts:
            stripped_sentence = sentence.strip()
            if stripped_sentence:
                all_sentences.append(stripped_sentence)

    return all_sentences


def extract_clean_words(response: str, language: str = "it") -> List[str]:
    """Extract clean words using Italian language strategy for tokenization."""
    strategy = _get_strategy(language)
    text_without_lists = re.sub(r"^\s*\d+\.\s", "", response, flags=re.MULTILINE)
    return strategy.tokenize_words(text_without_lists)


def analyze_lists(text: str, pattern: str) -> list[dict]:
    """
    Analyzes a text to find lists (numbered or bulleted) based on a
    provided regex pattern, noting their nesting level and item count.
    Correctly handles indentation by counting spaces/tabs.
    """
    lists_found = []
    current_list_stack = []  # Tracks lists at different nesting levels

    # Compile the pattern provided by the user
    item_pattern = re.compile(pattern, re.MULTILINE)

    # Find all list items in the text
    all_items = item_pattern.finditer(text)

    for item in all_items:
        indentation, marker, item_text = item.groups()
        # Calculate indentation level properly:
        # - Remove newlines
        # - Count spaces (each space = 1)
        # - Count tabs (each tab = 4 spaces for standard indentation)
        indent_str = indentation.replace("\n", "").replace("\r", "")
        indent_level = 0
        for char in indent_str:
            if char == " ":
                indent_level += 1
            elif char == "\t":
                indent_level += 4  # Standard tab width
            # Ignore other whitespace characters

        # If the stack is empty or indentation is less than the last list,
        # it means all previous lists have ended.
        while current_list_stack and indent_level < current_list_stack[-1]["indent"]:
            lists_found.append(current_list_stack.pop())

        # If the stack is empty or indentation is the same, it's a new top-level list
        # or another item in the current list.
        if not current_list_stack or indent_level == current_list_stack[-1]["indent"]:
            if not current_list_stack:  # A new top-level list starts
                nesting_level = 1
                current_list_stack.append({"level": nesting_level, "indent": indent_level, "items": 1})
            else:  # Another item in the current-level list
                current_list_stack[-1]["items"] += 1

        # If indentation increases, a new nested list has started
        elif indent_level > current_list_stack[-1]["indent"]:
            nesting_level = current_list_stack[-1]["level"] + 1
            current_list_stack.append({"level": nesting_level, "indent": indent_level, "items": 1})

    # Add any remaining lists from the stack to the results
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
            break  # Reached end of text

        divider = lines[i + 1].strip()
        # A valid divider must contain '|' and be made of '-', '|', ':', and whitespace.
        if "|" not in divider or not re.match(r"^[\s|: -]+$", divider):
            i += 1
            continue

        # Determine the number of columns from the header.
        # We count the segments between pipes, ignoring empty segments from start/end pipes.
        header_cols = [col.strip() for col in line.split("|") if col.strip()]
        num_cols = len(header_cols)

        # The divider line must have a number of segments that matches the header.
        divider_cols = [col.strip() for col in divider.split("|") if col.strip()]
        if len(divider_cols) != num_cols:
            i += 1
            continue

        # Count the data rows
        num_rows = 0
        j = i + 2  # Start counting from the line after the divider
        while j < len(lines) and "|" in lines[j]:
            num_rows += 1
            j += 1

        tables_found.append({"rows": num_rows, "columns": num_cols})

        # Move the main index past this entire table
        i = j

    return tables_found


def find_punctuations(text: str, language: str = "it") -> list[str]:
    """Find punctuation marks using Italian language strategy."""
    strategy = _get_strategy(language)
    delims = strategy.punctuation_marks
    cleaned_text = re.sub(r"^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)", "", text, flags=re.MULTILINE)
    punctuations = re.findall(f"[{re.escape(delims)}]+", cleaned_text)
    return punctuations


def extract_clean_paragraphs(text: str) -> List[str]:
    """
    Extracts clean paragraphs from a text by removing markdown elements.

    This function removes:
    - Markdown tables
    - Markdown headings (#, ##, etc.)
    - Horizontal rules (---, ***, ___)
    - Custom title tags (<<...>>)

    Paragraphs are defined as blocks of text separated by one or more blank lines.
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

    # Split the fully cleaned text into paragraphs
    # A paragraph is a block of text separated by one or more blank lines.
    if not cleaned_text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", cleaned_text.strip())

    # Final filter to remove any empty strings that might remain
    clean_paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return clean_paragraphs


def validate_instruction(
    response: str, inst_type: str, kwargs: Dict[str, Any], all_instructions: Dict = None
) -> Tuple[bool, str]:
    """Validate a response against a specific instruction type and its kwargs."""
    try:
        response = response.strip()
        if inst_type == "change_case:all_caps":
            # Check if all letters are uppercase, including accented uppercase
            # Must have at least one letter
            has_letters = any(ch.isalpha() for ch in response)
            if not has_letters:
                return (False, "Response contains no letters.")

            # All alphabetic characters must be uppercase (including accented)
            is_all_upper = all((ch.isupper() or ch in "ÀÈÉÌÒÙ") if ch.isalpha() else True for ch in response)
            return (is_all_upper, "No error" if is_all_upper else "Response is not all uppercase.")

        if inst_type == "change_case:lowercase":
            return (response.islower(), "No error" if response.islower() else "Response is not all lowercase.")

        if inst_type == "change_case:alternating":
            valid = all(is_strict_alternating(w) for w in response.split() if w.isalpha())
            return (valid, "No error" if valid else "Response is not strictly alternating.")

        if inst_type == "change_case:first_letter_cap":
            # Use Italian tokenization which splits elisions
            words = extract_clean_words(response, "it")
            if not words:
                return (True, "No words found to validate.")

            # Check each tokenized word (elisions are already split)
            invalid_words = []
            for tok in words:
                if not is_first_letter_cap(tok):
                    invalid_words.append(tok)

            valid = len(invalid_words) == 0
            return (valid, "No error" if valid else f"Words that don't follow first-letter-cap rule: {invalid_words}")

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
            matches = re.findall(pattern, response, re.IGNORECASE)

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
            # Accept both custom format <<Title>> and Markdown headers # Title
            found_title = (line.strip().startswith("<<") and line.strip().endswith(">>")) or (
                line.strip().startswith("#") and len(line.strip()) > 1
            )
            return (
                found_title,
                "No error"
                if found_title
                else "Title not wrapped in << >> or Markdown header (# Title) on first line.",
            )

        if inst_type == "keywords:existence":
            missing = [kw for kw in kwargs["keywords"] if keyword_frequency(response, kw, "it") == 0]
            return (not missing, "No error" if not missing else f"Missing keyword(s): {missing}")

        if inst_type == "keywords:frequency":
            keyword = kwargs["keyword"].strip().lower()
            count = keyword_frequency(response, keyword, "it")
            rel = kwargs["relation"]
            val = kwargs["frequency"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} of '{keyword}', found {count}.")

        if inst_type == "keywords:forbidden_words":
            present = [w for w in kwargs["forbidden_words"] if keyword_frequency(response, w, "it")]
            return (not present, "No error" if not present else f"Forbidden words found: {present}")

        if inst_type == "keywords:letter_frequency":
            letter = kwargs["letter"].lower()
            # Map base letter to all accented variants for Italian
            letter_map = {"a": "aàAÀ", "e": "eèéEÈÉ", "i": "iìIÌ", "o": "oòOÒ", "u": "uùUÙ"}
            # Count base letter and all accented variants
            if letter in letter_map:
                pattern = f"[{letter_map[letter]}]"
                count = len(re.findall(pattern, response))
            else:
                # For other letters, count case-insensitively
                count = len(re.findall(re.escape(letter), response, re.IGNORECASE))
            rel, val = kwargs["let_relation"], kwargs["let_frequency"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (
                valid,
                "No error"
                if valid
                else f"Expected {rel} {val} '{letter}' (case-insensitive, including accented variants), found {count}.",
            )

        if inst_type == "punctuation:no_comma":
            return ("," not in response, "No error" if "," not in response else "Commas found in response.")

        if inst_type == "length_constraints:number_characters":
            count = len(response)
            rel, val = kwargs["relation"], kwargs["num_chars"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} characters, found {count}.")

        if inst_type == "length_constraints:number_words":
            count = len(re.compile(r"\b(?=\S*[A-Za-z0-9])\S+\b").findall(response))
            rel, val = kwargs["relation"], kwargs["num_words"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} words, found {count}.")

        if inst_type == "length:max_word_count":
            """
            Checks if the total word count is at most max_words.
            """
            max_words = kwargs.get("max_words", 0)
            count = len(extract_clean_words(response, "it"))
            is_valid = count <= max_words
            return (is_valid, "No error" if is_valid else f"Found {count} words. Expected at most {max_words}.")

        if inst_type == "startend:start_checker":
            starts_correctly = (
                response.lstrip(string.punctuation + " ").lower().startswith(kwargs.get("start_phrase", "").lower())
            )
            return (
                starts_correctly,
                "No error" if starts_correctly else "Response does not start with required phrase.",
            )

        if inst_type == "startend:end_checker":
            required = kwargs["end_phrase"].strip()
            # Check if required phrase ends with punctuation
            ends_with_punctuation = required[-1] in string.punctuation if required else False

            # Get the actual end of the response
            actual_words = response.lstrip(string.punctuation).strip().split()
            if not actual_words:
                return (False, "Empty response")

            # If required phrase ends with punctuation, we need exact match
            if ends_with_punctuation:
                actual_phrase = " ".join(actual_words[-len(required.split()) :])
                if actual_phrase.lower() != required.lower():
                    return (False, f"End phrase mismatch: expected '{required}', but found '{actual_phrase}'")
            else:
                # If no punctuation, strip trailing punctuation and whitespace
                actual_phrase = " ".join(actual_words).rstrip(string.punctuation + " ")[-len(required) :]
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
            """
            Returns True if the ratio of lowercase to uppercase letters lies between
            minR and maxR (inclusive). Otherwise, returns False.

            If there are no letters, returns False.
            If there are no uppercase letters, ratio is considered float('inf').
            Handles Italian accented characters properly.
            """

            try:
                minR = parse_fraction_or_inf(kwargs["min_fraction"])
                maxR = parse_fraction_or_inf(kwargs["max_fraction"])
            except (ValueError, ZeroDivisionError) as e:
                raise ValueError(f"Invalid fraction input: {e}")

            if minR > maxR:
                return (False, "Validation failed: Minimum ratio greater than maximum ratio.")
            # Count lowercase including accented lowercase (à, è, é, ì, ò, ù)
            lower_count = sum(1 for ch in response if ch.islower() or ch in "àèéìòù")
            # Count uppercase including accented uppercase (À, È, É, Ì, Ò, Ù)
            upper_count = sum(1 for ch in response if ch.isupper() or ch in "ÀÈÉÌÒÙ")

            if lower_count == 0 and upper_count == 0:
                print("Validation failed: No letters found in the string.")
                return False

            # The ratio variable will hold either a Fraction object or float('inf')
            if upper_count == 0:
                ratio = float("inf")
                ratio_str = "inf"
            else:
                # Convert the calculated ratio directly into a Fraction
                ratio = Fraction(lower_count, upper_count)
                ratio_str = f"{ratio.numerator}/{ratio.denominator}"

            valid = minR <= ratio <= maxR

            # Construct a detailed message for both pass and fail cases
            message = (
                f"Lowercase count: {lower_count}, Uppercase count: {upper_count}. "
                f"Ratio is {ratio_str}({float(ratio):.2f}). Required range: [{minR}({float(minR):.2f}), {maxR}({float(maxR):.2f})]."
            )
            return (valid, "No error" if valid else f"{message}")

        if inst_type == "change_case:first_letter_sentence":
            """
            Checks if all sentences in the text start with an uppercase alphabet.
            Paragraphs are separated by newlines.
            Sentences are split using '.', '!', or '?' as delimiters.
            Handles Italian abbreviations and elisions.
            """

            sentences = extract_clean_sentences(response, "it")

            if not sentences:
                return (True, "No sentences found to validate.")

            # print(sentences)
            for sentence in sentences:
                sentence = sentence.strip("()[]{}\"'")

                # Get the first word of the sentence (after tokenization to handle elisions)
                words = extract_clean_words(sentence, "it")
                if not words:
                    continue

                first_word = words[0]
                # Find first alphabetic character in the first word
                first_alpha = None
                for ch in first_word:
                    if ch.isalpha() or ch in "ÀÈÉÌÒÙàèéìòù":
                        first_alpha = ch
                        break

                if first_alpha and not (first_alpha.isupper() or first_alpha in "ÀÈÉÌÒÙ"):
                    return (False, f"Fails at: '{sentence}' (first word: '{first_word}')")

            return (True, "No error.")

        if inst_type == "change_case:last_letter":
            """
            Checks if the last character of the last word in the text matches the given case.
            The last word may contain letters, numbers, or symbols (e.g., '45%').
            Trailing sentence-ending punctuation (.!? ) and wrapping symbols ()[]{},"' are ignored.
            """

            cleaned_text = re.sub(r"[.!?]+$", "", response.strip())

            if not cleaned_text:
                return (False, "Empty response")  # Empty after cleaning

            # Extract last word
            words = cleaned_text.split()
            last_word = words[-1]

            # Strip wrapping punctuation like (), [] , {} , quotes
            last_word = last_word.strip("()[]{}\"'")

            if not last_word:
                return False

            last_char = last_word[-1]
            valid = True

            # print(sentences)
            case_type = kwargs["case"]
            if case_type == "uppercase":
                valid = last_char.isupper() or last_char in "ÀÈÉÌÒÙ"
            elif case_type == "lowercase":
                valid = last_char.islower() or last_char in "àèéìòù"
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

            # Use Italian vowels from language strategy
            strategy = _get_strategy("it")
            vowels = strategy.vowels
            vowel_count = sum(1 for ch in response if ch.isalpha() and ch in vowels)
            consonant_count = sum(1 for ch in response if ch.isalpha() and ch not in vowels)

            # Handle the case where there are no letters at all
            if vowel_count == 0 and consonant_count == 0:
                return (False, "Validation failed: No letters found in the response.")

            # Handle the case where there are no consonants (infinite ratio)
            if consonant_count == 0:
                ratio = float("inf")
                ratio_str = "inf"
            else:
                # Convert the calculated ratio directly into a Fraction
                ratio = Fraction(vowel_count, consonant_count)
                ratio_str = f"{ratio.numerator}/{ratio.denominator}"

            valid = minR <= ratio <= maxR

            # Create a detailed message for both pass and fail cases
            message = (
                f"Vowel count: {vowel_count}, Consonant count: {consonant_count}. "
                f"Ratio is {ratio_str}({float(ratio):.2f}). Required range: [{minR}({float(minR):.2f}), {maxR}({float(maxR):.2f})]."
            )
            # print(message)
            return (valid, "No error" if valid else f"{message}")

        if inst_type == "detectable_format:number_paragraphs":
            """
            Checks if the number of paragraphs in the given text
            satisfies the relation with the expected_count.
            Paragraphs are defined as blocks of text separated by one or more empty lines.
            """

            cleaned_response = response.strip().replace("\r\n", "\n")

            # Treat multiple "Enters" as a single paragraph break.
            paragraphs = extract_clean_paragraphs(response)

            # Filter out any potential empty strings
            actual_paragraph_count = len([p for p in paragraphs if p.strip()])

            # If the input was empty, the count is 0, not 1.
            if not cleaned_response:
                actual_paragraph_count = 0

            relation = kwargs["relation"]
            num_paragraphs = kwargs["num_paragraphs"]
            is_valid = False

            valid, err = check_relation(actual_paragraph_count, relation, num_paragraphs)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {actual_paragraph_count} paragraphs. Expected {num_paragraphs}"
                return (False, message)
            return (True, "No error.")

        if inst_type == "detectable_format:max_paragraph_length":
            """
            Checks if the number of characters in each paragraph (including spaces and special characters)
            is at most the given expected_count.
            """
            max_chars = kwargs["max_chars"]
            paragraphs = extract_clean_paragraphs(response)

            # print(paragraphs)

            for p in paragraphs:
                p = re.sub(r"^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)", "", p.lstrip())
                # print(p)
                char_count = len(p.strip())
                if char_count > max_chars:
                    return (False, f"Found a paragraph containing {char_count} characters.\n '{p}'")

            return (True, "No error.")

        if inst_type == "detectable_format:sentences_per_paragraph":
            """
            Checks if the number of sentences in each paragraph satisfies relation with a given number.
            """
            num_sentences = kwargs["num_sentences"]
            relation = kwargs["relation"]
            paragraphs = extract_clean_paragraphs(response)

            # print(paragraphs)
            is_valid = True

            for p in paragraphs:
                # print(p)
                sentences = extract_clean_sentences(p, "it")

                sentence_count = len([s for s in sentences if s.strip()])
                if sentence_count == 0 and p.strip():
                    sentence_count = 1

                # print(sentence_count, relation, num_sentences)

                valid, err = check_relation(sentence_count, relation, num_sentences)
                if err is not None:
                    return (False, err)
                if not valid:
                    message = f"Found {sentence_count} sentences. Expected {num_sentences}\n '{p}'"
                    return (False, message)

            return (True, "No error.")

        if inst_type == "detectable_format:indentation":
            """
            Checks if text has indentation of specified type and size.
            indent_type: "spaces" or "tabs"
            size: number of spaces/tabs per indent level
            """
            indent_type = kwargs.get("indent_type", "spaces")
            size = kwargs.get("size", 4)

            lines = response.split("\n")
            has_indentation = False

            for line in lines:
                stripped = line.lstrip()
                if not stripped:  # Skip empty lines
                    continue

                indent = line[: len(line) - len(stripped)]
                if indent_type == "spaces":
                    # Check if line starts with spaces (multiple of size)
                    if indent and all(c == " " for c in indent):
                        indent_count = len(indent)
                        if indent_count >= size:
                            has_indentation = True
                            break
                elif indent_type == "tabs":
                    # Check if line starts with tabs
                    if indent and all(c == "\t" for c in indent):
                        indent_count = len(indent)
                        if indent_count >= size:
                            has_indentation = True
                            break

            return (
                has_indentation,
                "No error" if has_indentation else f"No indentation found with {indent_type} of size {size}.",
            )

        if inst_type == "length_constraints:sentence_length":
            """
            Checks if the number of words in each sentence (including bullet list items: '-' and numbered lists '1.')
            must be less than or equal to max_words.

            """
            sentences = extract_clean_sentences(response, "it")
            max_words = kwargs["max_words"]

            if not sentences:
                return (True, "No sentences found to validate.")

            for s in sentences:
                word_count = len(extract_clean_words(s, "it"))
                if word_count > max_words:
                    return (False, f"Expected at most {max_words} words. Found {word_count} words in '{s}'")

            return (True, "No error.")

        if inst_type == "length_constraints:word_repetition":
            max_repeats = kwargs["max_repeats"]
            words = extract_clean_words(response, "it")
            # print(words)
            # flag=0

            # Count occurrences
            word_counts = Counter(words)

            # Check if any word exceeds max_repeats
            for word, count in word_counts.items():
                if count > max_repeats:
                    return (False, f"Word '{word}' appears {count} times (limit {max_repeats})")
                    # flag=1

            return (True, "No error.")

        if inst_type == "length_constraints:unique_words":
            relation = kwargs["relation"]
            num_unique = kwargs["num_unique"]
            words = extract_clean_words(response, "it")
            # Normalize words to lowercase for comparison (handles elisions correctly)
            # After tokenization, "L'arte" -> ["L", "arte"] and "dell'arte" -> ["dell", "arte"]
            # So "arte" will be counted once
            strategy = _get_strategy("it")
            normalized_words = [strategy.casefold(w) for w in words]

            # Convert to set to get unique words
            unique_words_count = len(set(normalized_words))

            valid, err = check_relation(unique_words_count, relation, num_unique)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {unique_words_count} unique words. Expected {relation} {num_unique}."
                return (False, message)
            return (True, "No error.")

        if inst_type == "punctuation:frequency":
            """
            Checks if a specific punctuation mark appears with the specified frequency.
            """
            punctuation = kwargs.get("punctuation", "")
            relation = kwargs.get("relation", "at least")
            frequency = kwargs.get("frequency", 0)

            count = response.count(punctuation)

            valid, err = check_relation(count, relation, frequency)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {count} occurrences of '{punctuation}'. Expected {relation} {frequency}."
                return (False, message)
            return (True, "No error.")

        if inst_type == "punctuation:balance":
            """
            Checks if opening and closing punctuation marks are balanced.
            Validates pairs: () [] {} "" ''
            """
            pairs = {"(": ")", "[": "]", "{": "}", '"': '"', "'": "'"}

            stack = []
            in_double_quote = False
            in_single_quote = False

            for char in response:
                if char == '"' and not in_single_quote:
                    if in_double_quote and stack and stack[-1] == '"':
                        stack.pop()
                        in_double_quote = False
                    else:
                        stack.append('"')
                        in_double_quote = True
                elif char == "'" and not in_double_quote:
                    if in_single_quote and stack and stack[-1] == "'":
                        stack.pop()
                        in_single_quote = False
                    else:
                        stack.append("'")
                        in_single_quote = True
                elif char in pairs and char not in ['"', "'"]:
                    stack.append(char)
                elif char in pairs.values() and char not in ['"', "'"]:
                    if not stack:
                        return (False, f"Unmatched closing '{char}' found.")
                    opening = stack.pop()
                    if pairs[opening] != char:
                        return (False, f"Mismatched punctuation: expected '{pairs[opening]}', found '{char}'.")

            if stack:
                unmatched = stack[0]
                return (False, f"Unmatched opening '{unmatched}' found.")

            return (True, "No error.")

        if inst_type == "punctuation:question_exclaim":
            is_valid = True
            relation = kwargs["relation"]
            num_marks = kwargs["num_marks"]

            punctuation_pattern = r"[?!]"

            # Find all punctuation characters
            punctuations = re.findall(punctuation_pattern, response)

            count = len(punctuations)
            # print("Count of punctuations: ", count)

            valid, err = check_relation(count, relation, num_marks)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {count} marks. Expected {relation} {num_marks}."
                return (False, message)
            return (True, "No error.")

        if inst_type == "punctuation:no_period":
            return ("." not in response, "No error" if "." not in response else "Periods found in response.")

        if inst_type == "punctuation:end_rule":
            """
            Validates that only allowed punctuation marks appear at sentence endings.
            Checks sentence-ending punctuation specifically, not all punctuation in text.
            """
            allowed = kwargs.get("allowed", [])
            if not isinstance(allowed, list):
                allowed = list(allowed) if allowed else []

            # Find all sentence-ending punctuation marks
            strategy = _get_strategy("it")
            delims = strategy.sentence_delims

            # Extract sentences and check their ending punctuation
            sentences = extract_clean_sentences(response, "it")

            for sentence in sentences:
                # Find punctuation at the end of the sentence
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Check the last character(s) for punctuation
                end_punct = ""
                for i in range(len(sentence) - 1, -1, -1):
                    if sentence[i] in delims or sentence[i] in string.punctuation:
                        end_punct = sentence[i] + end_punct
                    else:
                        break

                # If sentence ends with punctuation, check if it's allowed
                if end_punct:
                    # Check each punctuation character
                    for punct_char in end_punct:
                        if punct_char in delims or punct_char in string.punctuation:
                            # Normalize to single character for comparison
                            if punct_char not in allowed:
                                return (
                                    False,
                                    f"Found disallowed punctuation '{punct_char}' at sentence end. Allowed: {allowed}",
                                )

            return (True, "No error.")

        if inst_type == "keywords:alliteration":
            """
            Checks alliteration using phonetic logic for Italian.
            Handles hard/soft C and G sounds:
            - C before a,o,u = hard /k/; C before e,i = soft /tʃ/
            - G before a,o,u = hard /g/; G before e,i = soft /dʒ/
            Only counts words that start with target_letter AND have the same phonetic sound.
            """
            relation = kwargs["relation"]
            num_alliteration = kwargs["num_alliteration"]
            target_letter = kwargs["target_letter"].lower()

            def get_phonetic_sound(word: str) -> str:
                """Get the phonetic sound of the first letter(s) of a word."""
                if not word:
                    return ""
                word_lower = word.lower()
                first_char = word_lower[0]

                # Handle C sounds
                if first_char == "c":
                    if len(word_lower) > 1:
                        next_char = word_lower[1]
                        if next_char in "ei":
                            return "tʃ"  # Soft C (like "ch")
                        elif next_char == "h":
                            # "ch" before a,o,u is hard /k/, before e,i is soft /tʃ/
                            if len(word_lower) > 2 and word_lower[2] in "ei":
                                return "tʃ"
                            else:
                                return "k"  # Hard C
                    return "k"  # Default hard C

                # Handle G sounds
                elif first_char == "g":
                    if len(word_lower) > 1:
                        next_char = word_lower[1]
                        if next_char in "ei":
                            return "dʒ"  # Soft G (like "j")
                        elif next_char == "h":
                            # "gh" is always hard /g/
                            return "g"
                    return "g"  # Default hard G

                # For other letters, use the letter itself as the sound
                return first_char

            words = extract_clean_words(response, "it")

            # Find all words starting with target_letter and their phonetic sounds
            matching_words = []
            for word in words:
                word_lower = word.lower()
                if word_lower.startswith(target_letter):
                    matching_words.append((word, get_phonetic_sound(word)))

            if not matching_words:
                all_count = 0
            else:
                # For alliteration, ALL words must have the SAME phonetic sound
                # If words have different sounds, they don't alliterate
                sounds = [sound for _, sound in matching_words]
                from collections import Counter

                sound_counts = Counter(sounds)

                # Check if all words have the same phonetic sound
                if len(sound_counts) == 1:
                    # All words have the same sound - perfect alliteration
                    all_count = len(matching_words)
                else:
                    # Words have different sounds - they do NOT alliterate
                    # Example: "Chicco" (hard /k/) and "cerca" (soft /tʃ/) don't alliterate
                    all_count = 0

            valid, err = check_relation(all_count, relation, num_alliteration)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {all_count} alliteration words starting with '{target_letter}' with consistent phonetic sound. Expected {relation} {num_alliteration}."
                return (False, message)
            return (True, "No error.")

        if inst_type == "keywords:palindrome_word":
            """
            Checks if response contains at least one palindrome word of minimum length.
            Handles elisions and case-insensitive comparison.
            Example: "Anna" is a palindrome (case-insensitive), "oro" is a palindrome.
            """
            min_length = kwargs["min_length"]
            words = extract_clean_words(response, "it")

            # Strip punctuation from words and check for palindromes (case-insensitive)
            punctuation_chars = string.punctuation.replace("'", "")
            strategy = _get_strategy("it")

            for word in words:
                # Remove punctuation from the word
                word_clean = word.rstrip(punctuation_chars).lstrip(punctuation_chars)

                # Check if word meets minimum length requirement
                if len(word_clean) < min_length:
                    continue

                # Check if word is a palindrome (case-insensitive)
                word_normalized = strategy.casefold(word_clean)
                if word_normalized == word_normalized[::-1]:
                    return (True, f"No error. Word: {word_clean}")

            return (False, "No valid palindrome words found.")

        if inst_type == "keywords:positioning":
            keyword = kwargs["keyword"]
            position = kwargs["position"]

            words = extract_clean_words(response)

            if words[position] == keyword:
                return (True, "No error.")

            return (False, f"'{words[position]}' found after {position} words instead of '{keyword}'.")

        if inst_type == "detectable_format:nested_list":
            min_depth = kwargs["min_depth"]
            num_subitems = kwargs["num_subitems"]

            # Combined pattern that matches all list types: bullets, numbered, and letter-based
            # This ensures nested lists with different markers are analyzed together
            combined_pattern = r"^(\s*)([*+-]|\d+\.|[a-z]\.)[ \t]+(.*)"

            # Analyze all list items together to correctly detect nesting levels
            lists = analyze_lists(response, combined_pattern)

            # Check if any list at the required depth has enough items
            for lst in lists:
                if lst["level"] == min_depth and lst["items"] >= num_subitems:
                    return (True, "No error.")

            return (False, f"List at level {min_depth} with at least {num_subitems} items not found.")

        if inst_type == "detectable_format:table":
            min_rows = kwargs["min_rows"]
            min_cols = kwargs["min_cols"]

            tables = find_markdown_tables(response)

            # print(tables)

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
            # print(all_headings)

            for level in levels:
                if not level in all_headings:
                    return (False, f"Heading of level {level} not found")

            return (True, "No error.")

        if inst_type == "detectable_format:section_balance":
            """
            Checks if sections of a specific element type are balanced (equal count).
            element_type: "paragraph", "list", "heading", "bullets", etc.
            count: expected number of elements
            """
            element_type = kwargs.get("element_type", "paragraph")
            expected_count = kwargs.get("count", 0)

            if element_type == "paragraph":
                actual_count = len(extract_clean_paragraphs(response))
            elif element_type == "list":
                # Count both bullet and numbered lists
                bullet_count = count_bullet_points(response)
                numbered_count = count_numbered_items(response)
                actual_count = bullet_count + numbered_count
            elif element_type == "bullets":
                # Count only bullet points
                actual_count = count_bullet_points(response)
            elif element_type == "heading":
                # Count markdown headings
                heading_pattern = re.compile(r"^\s*#+\s+", re.MULTILINE)
                actual_count = len(heading_pattern.findall(response))
            else:
                return (False, f"Unsupported element_type: {element_type}")

            is_valid = actual_count == expected_count
            return (
                is_valid,
                "No error" if is_valid else f"Found {actual_count} {element_type}(s). Expected {expected_count}.",
            )

        if inst_type == "length_constraints:word_length":
            max_length = kwargs["max_length"]
            min_length = kwargs["min_length"]

            if min_length > max_length:
                return (False, "Validation failed: Minimum length greater than maximum length.")

            words = set(extract_clean_words(response, "it"))

            if not words:
                return (True, "No words found to validate.")

            # Find the shortest and longest words in the set
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
            is_valid = True
            min_ratio = kwargs["min_ratio"]
            max_ratio = kwargs["max_ratio"]

            if min_ratio > max_ratio:
                return (False, "Validation failed: Minimum length greater than maximum length.")

            words = extract_clean_words(response)
            avg_count = sum(len(word) for word in words) / len(words)

            if not words:
                is_valid = min_ratio == 0
                return (is_valid, "No words found to validate.")

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
            sentence_count = len(extract_clean_sentences(response, "it"))

            valid, err = check_relation(sentence_count, relation, num_sentences)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {sentence_count} sentences. Expected {relation} {num_sentences}"
                return (False, message)
            return (True, "No error.")

        if inst_type == "length_constraints:paragraph_length":
            """
            Checks if the number of words in each paragraph satisfies relation with a given number.
            """
            words_per_paragraph = kwargs["words_per_paragraph"]
            relation = kwargs["relation"]

            # Treat multiple "Enters" as a single paragraph break.
            paragraphs = extract_clean_paragraphs(response)

            # print(paragraphs)
            is_valid = True

            for p in paragraphs:
                # print(p)
                words = extract_clean_words(p, "it")

                word_count = len([s for s in words if s.strip()])
                # if word_count == 0 and p.strip():
                #     word_count = 1

                # print(word_count, relation, words_per_paragraph)

                valid, err = check_relation(word_count, relation, words_per_paragraph)
                if err is not None:
                    return (False, err)
                if not valid:
                    message = f"Found {word_count} words. Expected {relation} {words_per_paragraph}\n '{p}'"
                    return (False, message)

            return (True, "No error.")

        if inst_type == "punctuation:variety":
            """
            Checks if text contains at least min_types distinct punctuation marks.
            """
            min_types = kwargs.get("min_types", 1)

            # Find all punctuation marks in the text
            strategy = _get_strategy("it")
            delims = strategy.punctuation_marks
            # Also include common punctuation: , ; : - ( ) [ ] { } " ' etc.
            all_punct = delims + ",;:—–-()[]{}\"'"

            found_punctuation = set()
            for char in response:
                if char in all_punct:
                    found_punctuation.add(char)

            is_valid = len(found_punctuation) >= min_types
            return (
                is_valid,
                "No error"
                if is_valid
                else f"Found {len(found_punctuation)} distinct punctuation types. Expected at least {min_types}.",
            )

        if inst_type == "detectable_content:numeric_inclusion":
            num_numbers = kwargs["num_numbers"]
            relation = kwargs["relation"]

            num_count = sum(1 for ch in response if ch.isdigit())
            # print("Numeric Count:", num_count)

            valid, err = check_relation(num_count, relation, num_numbers)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {num_count} digits. Expected {relation} {num_numbers}"
                return (False, message)
            return (True, "No error.")

        if inst_type == "detectable_format:sentence_endings":
            min_variants = kwargs["min_variants"]
            # Find distinct punctuation types at sentence endings
            # Look for punctuation at end of sentences (after word, before whitespace/newline)
            strategy = _get_strategy("it")
            delims = strategy.punctuation_marks
            # Match punctuation at end of sentences (word boundary + punctuation + end of sentence)
            ending_punct_pattern = rf"[{re.escape(delims)}]+(?=\s|$)"
            ending_punctuations = re.findall(ending_punct_pattern, response)
            # Get distinct types (normalize multiple occurrences to single char)
            distinct_types = set()
            for punct in ending_punctuations:
                # Extract the base punctuation character(s)
                for char in punct:
                    if char in delims:
                        distinct_types.add(char)

            if len(distinct_types) < min_variants:
                return (
                    False,
                    f"Found {len(distinct_types)} types of sentence-ending punctuations ({distinct_types}). Expected at least {min_variants}.",
                )

            return (True, "No error.")

        if inst_type == "keywords:vowel_count":
            num_vowels = kwargs["num_vowels"]
            relation = kwargs["relation"]

            # Use Italian vowels from language strategy
            strategy = _get_strategy("it")
            vowels = strategy.vowels
            vowel_count = sum(1 for ch in response if ch in vowels)

            # print("Vowel count:", vowel_count)
            valid, err = check_relation(vowel_count, relation, num_vowels)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {vowel_count} vowels. Expected {relation} {num_vowels}"
                return (False, message)
            return (True, "No error.")

        if inst_type == "keywords:consonant_count":
            num_consonants = kwargs["num_consonants"]
            relation = kwargs["relation"]

            # Use Italian vowels from language strategy
            strategy = _get_strategy("it")
            vowels = strategy.vowels
            # For Italian, count all alphabetic characters that are not vowels
            consonant_count = sum(1 for ch in response if ch.isalpha() and ch not in vowels)

            # print("consonant count:", consonant_count)
            valid, err = check_relation(consonant_count, relation, num_consonants)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {consonant_count} consonants. Expected {relation} {num_consonants}"
                return (False, message)
            return (True, "No error.")

    except Exception as e:
        return (False, f"Validation error: {str(e)}")

    return (False, "Invalid Instruction")
