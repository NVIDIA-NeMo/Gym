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


def count_numbered_items(response: str) -> int:
    """Count number of numbered items in response."""
    return len(re.findall(r"^\s*\d+\.", response, re.MULTILINE))


def count_bullet_points(response: str) -> int:
    """Count number of bullet points in response."""
    return len(re.findall(r"^[*-]\s", response, re.MULTILINE))


def count_placeholders(response: str) -> int:
    """Count number of placeholders in response."""
    return len(re.findall(r"\[.*?\]", response))


def keyword_frequency(response: str, keyword: str) -> int:
    """Count frequency of a keyword in response, allowing any Unicode tokens (Hindi)."""
    keyword = keyword.strip()
    escaped_tokens = [re.escape(part) for part in keyword.split()]
    phrase_pattern = r"\s+".join(escaped_tokens)

    pattern = rf"(?<![\w\'\-\\u0964\\u0965]){phrase_pattern}(?![\w\'\-\\u0964\\u0965])"

    return len(re.findall(pattern, response, flags=re.IGNORECASE | re.UNICODE))


def extract_clean_sentences(text: str) -> List[str]:
    """
    Takes a raw text string and returns a clean list of sentences.
    This version correctly handles list items that do not end with punctuation
    by treating each cleaned line as a source for one or more sentences.
    Treats Devanagari Danda (।) and Double Danda (॥) as sentence delimiters.
    """

    # Remove markdown tables
    table_pattern = r"(?:^\s*\|.*\|.*\n){2,}"
    cleaned_text = re.sub(table_pattern, "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    rule_pattern = r"^\s*([*_-])\s*\1\s*\1+\s*$"
    text = re.sub(rule_pattern, "", cleaned_text, flags=re.MULTILINE)

    # print(text)

    all_sentences = []

    # Process the text line by line
    for line in text.split("\n"):
        # Clean the line by removing markdown markers and leading space
        line = line.lstrip()
        cleaned_line = re.sub(r"^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)", "", line)

        if not cleaned_line:
            continue

        # Split the individual cleaned line into sentences.
        # This handles both multi-sentence lines and single-sentence list items.
        # Include Devanagari Danda (। U+0964) and Double Danda (॥ U+0965) as sentence delimiters
        # Also include ! and ? which are commonly used in Hindi
        line_parts = re.split(r"[!?\u0964\u0965]+", cleaned_line)

        # 3. Add the resulting parts to our main list after cleaning them.
        for sentence in line_parts:
            stripped_sentence = sentence.strip()
            if stripped_sentence:
                all_sentences.append(stripped_sentence)

    # print(all_sentences)

    return all_sentences


def extract_clean_words(response: str) -> List[str]:
    text_without_lists = re.sub(r"^\s*\d+\.\s", "", response, flags=re.MULTILINE)

    # Replace Danda characters with spaces to create word boundaries
    text_with_word_boundaries = re.sub(r"[\u0964\u0965]+", " ", text_without_lists)

    # Match contiguous Devanagari characters (U+0900 to U+097F)
    # EXCLUDING Danda (U+0964) and Double Danda (U+0965) to prevent punctuation being counted as words
    words = re.findall(r"[\u0900-\u0963\u0966-\u097F]+", text_with_word_boundaries)

    return words


def analyze_lists(text: str, pattern: str) -> list[dict]:
    """
    Analyzes a text to find lists (numbered or bulleted) based on a
    provided regex pattern, noting their nesting level and item count.
    """
    lists_found = []
    current_list_stack = []  # Tracks lists at different nesting levels

    # Compile the pattern provided by the user
    item_pattern = re.compile(pattern, re.MULTILINE)

    # Find all list items in the text
    all_items = item_pattern.finditer(text)

    for item in all_items:
        indentation, marker, item_text = item.groups()
        indent_level = len(indentation.strip("\n"))

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


def find_punctuations(text: str) -> list[str]:
    cleaned_text = re.sub(r"^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)", "", text, flags=re.MULTILINE)

    # [!?\u0964\u0965]+ matches one or more characters that are '!', '?', Danda (।), or Double Danda (॥).
    # Period (.) is excluded as it's not used as sentence-ending punctuation in Hindi.
    punctuations = re.findall(r"[!?\u0964\u0965]+", cleaned_text)

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


def normalize_hindi_digits(text: str) -> str:
    """
    Replaces Hindi Devanagari digits (०-९) with standard Arabic digits (0-9).
    """
    hindi_digits = "०१२३४५६७८९"
    english_digits = "0123456789"

    # Create a translation table
    translation_table = str.maketrans(hindi_digits, english_digits)

    return text.translate(translation_table)


def validate_instruction(
    response: str, inst_type: str, kwargs: Dict[str, Any], all_instructions: Dict = None
) -> Tuple[bool, str]:
    """Validate a response against a specific instruction type and its kwargs."""
    try:
        response = response.strip()
        response = normalize_hindi_digits(response)

        NOT_APPLICABLE_FOR_HINDI = {
            "change_case:all_caps",
            "change_case:lowercase",
            "change_case:alternating",
            "change_case:first_letter_cap",
            "change_case:capital_word_frequency",
            "change_case:lowercase_word_frequency",
            "change_case:all_caps_target",
            "change_case:lowercase_target",
            "change_case:alternating_target",
            "change_case:first_letter_cap_target",
            "change_case:case_ratio",
            "change_case:first_letter_sentence",
            "change_case:last_letter",
            "change_case:vowel_consonant_balance",
            "keywords:vowel_count",
            "keywords:consonant_count",
            "punctuation:no_period",
        }

        if inst_type in NOT_APPLICABLE_FOR_HINDI:
            return (False, "Invalid Instruction")

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
            line = response.splitlines()[0]
            found_title = line.strip().startswith("<<") and line.strip().endswith(">>")
            return (found_title, "No error" if found_title else "Title not wrapped in << >> on first line.")

        if inst_type == "keywords:existence":
            missing = [kw for kw in kwargs["keywords"] if keyword_frequency(response, kw) == 0]
            return (not missing, "No error" if not missing else f"Missing keyword(s): {missing}")

        if inst_type == "keywords:frequency":
            keyword = kwargs["keyword"].strip().lower()
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
            letter = kwargs["letter"].lower()
            count = response.lower().count(letter)
            rel, val = kwargs["let_relation"], kwargs["let_frequency"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (
                valid,
                "No error" if valid else f"Expected {rel} {val} '{letter}' (case-insensitive), found {count}.",
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
            count = len(extract_clean_words(response))
            rel, val = kwargs["relation"], kwargs["num_words"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} words, found {count}.")

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

            for p in paragraphs:
                p = re.sub(r"^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)", "", p.lstrip())
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

            is_valid = True

            for p in paragraphs:
                sentences = extract_clean_sentences(p)

                sentence_count = len([s for s in sentences if s.strip()])
                if sentence_count == 0 and p.strip():
                    sentence_count = 1

                valid, err = check_relation(sentence_count, relation, num_sentences)
                if err is not None:
                    return (False, err)
                if not valid:
                    message = f"Found {sentence_count} sentences. Expected {num_sentences}\n '{p}'"
                    return (False, message)

            return (True, "No error.")

        if inst_type == "length_constraints:sentence_length":
            """
            Checks if the number of words in each sentence (including bullet list items: '-' and numbered lists '1.')
            must be less than or equal to max_words.

            """
            sentences = extract_clean_sentences(response)
            max_words = kwargs["max_words"]

            if not sentences:
                return (True, "No sentences found to validate.")

            for s in sentences:
                word_count = len(s.split())
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

            punctuation_pattern = r"[?!]"

            # Find all punctuation characters
            punctuations = re.findall(punctuation_pattern, response)

            count = len(punctuations)

            valid, err = check_relation(count, relation, num_marks)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {count} marks. Expected {relation} {num_marks}."
                return (False, message)
            return (True, "No error.")

        if inst_type == "punctuation:end_rule":
            allowed = kwargs["allowed"]

            punctuations = set(find_punctuations(response))

            for p in punctuations:
                if not p in allowed:
                    return (False, f"'{p}' not in the list of allowed punctuations.")

            return (True, "No error.")

        if inst_type == "keywords:alliteration":
            relation = kwargs["relation"]
            num_alliteration = kwargs["num_alliteration"]
            target_letter = kwargs["target_letter"]

            words = extract_clean_words(response)
            all_count = sum(1 for word in words if word.startswith(target_letter))

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
                if word == word[::-1] and len(word) >= min_length:
                    return (True, f"No error. Word: {word}")

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

            bullet_pattern = r"^(\s*)([*+-])[ \t]+(.*)"
            numbered_pattern = r"^(\s*)(\d+\.)[ \t]+(.*)"

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
                if not level in all_headings:
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
            sentence_count = len(extract_clean_sentences(response))

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

            is_valid = True

            for p in paragraphs:
                words = extract_clean_words(p)

                word_count = len([s for s in words if s.strip()])

                valid, err = check_relation(word_count, relation, words_per_paragraph)
                if err is not None:
                    return (False, err)
                if not valid:
                    message = f"Found {word_count} words. Expected {relation} {words_per_paragraph}\n '{p}'"
                    return (False, message)

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
            punctuations = set(find_punctuations(response))

            if len(punctuations) < min_variants:
                return (
                    False,
                    f"Found {len(punctuations)} types of punctuations. Expected at least {min_variants}.\n {punctuations}",
                )

            return (True, "No error.")

    except Exception as e:
        return (False, f"Validation error: {str(e)}")

    return (False, "Invalid Instruction")
