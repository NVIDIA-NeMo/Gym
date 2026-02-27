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
import unicodedata
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


def extract_clean_sentences(text: str) -> List[str]:
    """
    Takes a raw text string and returns a clean list of sentences.
    This version correctly handles list items that do not end with punctuation
    by treating each cleaned line as a source for one or more sentences.
    """

    # Remove markdown tables
    table_pattern = r"(?:^\s*[|｜].*[|｜].*\n){2,}"
    cleaned_text = re.sub(table_pattern, "", text, flags=re.MULTILINE)

    # Remove horizontal rules (both half-width and full-width characters)
    # Handle both ASCII and full-width versions: ---, ***, ___, ーーー, ＊＊＊, ＿＿＿
    rule_pattern = r"^\s*([*_\-＊＿ー])\s*\1\s*\1+\s*$"
    cleaned_text = re.sub(rule_pattern, "", cleaned_text, flags=re.MULTILINE)

    all_sentences = []

    # Process the text line by line
    for line in cleaned_text.split("\n"):
        # Clean the line by removing markdown markers and leading space
        line = line.lstrip()
        clean_pattern = r"^\s*(?:[\-\*\+・]\s+|(?:\d+|[０-９]+)[.．]\s+|[#＃]+\s+)"
        cleaned_line = re.sub(clean_pattern, "", line)

        if not cleaned_line:
            continue

        # Split the individual cleaned line into sentences.
        # This handles both multi-sentence lines and single-sentence list items.
        line_parts = re.split(r"[.!?。！？‼⁉⁈⁇]+", cleaned_line)

        # 3. Add the resulting parts to our main list after cleaning them.
        for sentence in line_parts:
            stripped_sentence = sentence.strip()
            if stripped_sentence:
                all_sentences.append(stripped_sentence)

    # print(all_sentences)

    return all_sentences


def extract_clean_words(response: str) -> List[str]:
    text_without_lists = re.sub(r"^\s*\d+\.\s", "", response, flags=re.MULTILINE)
    pattern = r"[一-龯]+|[ぁ-ん]+|[ァ-ンー]+"
    return re.findall(pattern, text_without_lists.lower())


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
    cleaned_text = re.sub(r"^\s*(?:[\-\*\+・]\s+|(?:\d+|[０-９]+)[.．]\s+|[#＃]+\s+)", "", text, flags=re.MULTILINE)

    # [.!?]+ matches one or more characters that are '.', '!', or '?'.
    punctuations = re.findall(r"[.!?。！？‼⁉⁈⁇]+", cleaned_text)

    return punctuations


def extract_clean_paragraphs(text: str) -> List[str]:
    """
    Extracts clean paragraphs from Japanese text by removing markdown elements.

    This function removes:
    - Markdown tables
    - Markdown headings (#, ##, etc.)
    - Horizontal rules (---, ***, ___)
    - Custom title tags (<<...>>)
    - List markers (numbered lists with digits, bullet points)

    Paragraphs are defined as blocks of text separated by one or more blank lines.
    Japanese text is handled correctly as it doesn't use spaces between words.
    """

    # Remove custom title tags like <<Title>>
    cleaned_text = re.sub(r"^\s*<<.*>>\s*$", "", text, flags=re.MULTILINE)

    # Remove markdown tables
    table_pattern = r"(?:^\s*[|｜].*[|｜].*\n){2,}"
    cleaned_text = re.sub(table_pattern, "", cleaned_text, flags=re.MULTILINE)

    # Remove markdown headings (both half-width # and full-width ＃)
    heading_pattern = r"^\s*[#＃]+\s+.*$"
    cleaned_text = re.sub(heading_pattern, "", cleaned_text, flags=re.MULTILINE)

    # Remove horizontal rules (both half-width and full-width characters)
    # Handle both ASCII and full-width versions: ---, ***, ___, ーーー, ＊＊＊, ＿＿＿
    rule_pattern = r"^\s*([*_\-＊＿ー])\s*\1\s*\1+\s*$"
    cleaned_text = re.sub(rule_pattern, "", cleaned_text, flags=re.MULTILINE)

    # Remove list markers (numbered lists and bullet points)
    # Handle both half-width and full-width numbers, and various bullet markers
    list_pattern = r"^\s*(?:[\-\*\+・]\s+|[\d０-９]+[\.．]\s+)"
    cleaned_text = re.sub(list_pattern, "", cleaned_text, flags=re.MULTILINE)

    # Split the fully cleaned text into paragraphs
    # A paragraph is a block of text separated by one or more blank lines.
    # This works for Japanese as blank lines are language-independent.
    if not cleaned_text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", cleaned_text.strip())

    # Final filter to remove any empty strings that might remain
    # Also remove paragraphs that are only whitespace or punctuation
    clean_paragraphs = []
    for p in paragraphs:
        stripped = p.strip()
        # Keep paragraph if it has actual content (not just whitespace/punctuation)
        if stripped and re.search(r"[\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", stripped):
            clean_paragraphs.append(stripped)

    return clean_paragraphs


# Japanese numeral characters (including daiji and multipliers) for counting
JAPANESE_NUMERAL_CHARS = {
    # --- Standard 0-9 (Zero is special) ---
    "零",
    "〇",
    # --- Units (1-9) ---
    # Standard | Daiji (Common) | Daiji (Rare)
    "一",
    "壱",
    "二",
    "弐",
    "三",
    "参",
    "四",
    "肆",
    "五",
    "伍",
    "六",
    "陸",
    "七",
    "漆",
    "八",
    "捌",
    "九",
    "玖",
    # --- Multipliers (Power of 10) ---
    "十",
    "拾",  # 10
    "百",
    "陌",  # 100
    "千",
    "阡",  # 1,000
    "万",
    "萬",  # 10,000
    "億",  # 100,000,000 (100 Million)
    "兆",  # 1,000,000,000,000 (1 Trillion)
}


def normalize_japanese_numbers(text: str) -> str:
    """
    Replaces full-width numbers (０-９) and Japanese kanji numbers with standard Arabic digits.
    Handles basic kanji digits: 一(1), 二(2), 三(3), 四(4), 五(5), 六(6), 七(7), 八(8), 九(9), 十(10).
    """
    # Replace full-width numbers (０-９) with regular numbers (0-9)
    fullwidth_digits = "０１２３４５６７８９"
    english_digits = "0123456789"
    translation_table = str.maketrans(fullwidth_digits, english_digits)
    text = text.translate(translation_table)

    # Replace basic kanji digits
    JAPANESE_NUMERAL_MAP = {
        # --- Standard 0-9 (Zero is special) ---
        "零": "0",
        "〇": "0",
        # --- Units (1-9) ---
        # Standard | Daiji (Common) | Daiji (Rare)
        "一": "1",
        "壱": "1",
        "二": "2",
        "弐": "2",
        "三": "3",
        "参": "3",
        "四": "4",
        "肆": "4",
        "五": "5",
        "伍": "5",
        "六": "6",
        "陸": "6",
        "七": "7",
        "漆": "7",
        "八": "8",
        "捌": "8",
        "九": "9",
        "玖": "9",
    }

    for kanji, digit in JAPANESE_NUMERAL_MAP.items():
        text = text.replace(kanji, digit)

    return text


def validate_instruction(
    response: str, inst_type: str, kwargs: Dict[str, Any], all_instructions: Dict = None
) -> Tuple[bool, str]:
    """Validate a response against a specific instruction type and its kwargs."""
    try:
        response = response.strip()
        response = unicodedata.normalize("NFC", response)
        response = normalize_japanese_numbers(response)

        NOT_APPLICABLE_FOR_JAPANESE = {
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
            "keywords:existence",
            "keywords:frequency",
            "keywords:forbidden_words",
            "length_constraints:number_words",
            "startend:quotation",
            "change_case:case_ratio",
            "change_case:first_letter_sentence",
            "change_case:last_letter",
            "change_case:vowel_consonant_balance",
            "length_constraints:sentence_length",
            "length_constraints:word_repetition",
            "length_constraints:unique_words",
            "keywords:alliteration",
            "keywords:positioning",
            "length_constraints:word_length",
            "length_constraints:avg_word_length",
            "length_constraints:paragraph_length",
            "detectable_format:sentence_endings",
            "keywords:vowel_count",
            "keywords:consonant_count",
        }

        if inst_type in NOT_APPLICABLE_FOR_JAPANESE:
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
            return (
                "、" not in response and "," not in response,
                "No error" if "、" not in response and "," not in response else "Commas(、,) found in response.",
            )

        if inst_type == "length_constraints:number_characters":
            count = len(response)
            rel, val = kwargs["relation"], kwargs["num_chars"]
            valid, err = check_relation(count, rel, val)
            if err is not None:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} characters, found {count}.")

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

            valid, err = check_relation(actual_paragraph_count, relation, num_paragraphs)
            if err is not None:
                return (False, err)
            if not valid:
                message = f"Found {actual_paragraph_count} paragraphs. Expected {num_paragraphs}"
                return (False, message)
            return (True, "No error.")

        if inst_type == "detectable_format:sentences_per_paragraph":
            """
            Checks if the number of sentences in each paragraph satisfies relation with a given number.
            """
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
                    message = f"Found {sentence_count} sentences. Expected {num_sentences}\n '{p}'"
                    return (False, message)

            return (True, "No error.")

        if inst_type == "punctuation:question_exclaim":
            relation = kwargs["relation"]
            num_marks = kwargs["num_marks"]

            punctuation_pattern = r"[!?！？‼⁉⁈⁇]"

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

        if inst_type == "punctuation:no_period":
            return (
                "。" not in response and "．" not in response,
                "No error" if "。" not in response and "．" not in response else "Periods(。) found in response.",
            )

        if inst_type == "punctuation:end_rule":
            allowed = kwargs["allowed"]

            punctuations = set(find_punctuations(response))

            for p in punctuations:
                if not p in allowed:
                    return (False, f"'{p}' not in the list of allowed punctuations.")

            return (True, "No error.")

        if inst_type == "keywords:palindrome_word":
            min_length = kwargs["min_length"]
            words = extract_clean_words(response)
            for word in words:
                if word == word[::-1] and len(word) >= min_length:
                    return (True, f"No error. Word: {word}")

            return (False, "No valid palindrome words found.")

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

        if inst_type == "detectable_format:section_balance":
            # Logic for this instruction to be added here
            return (False, "Invalid Instruction")

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

        if inst_type == "punctuation:variety":
            # Logic for this instruction to be added here
            return (False, "Invalid Instruction")

        if inst_type == "detectable_content:numeric_inclusion":
            num_numbers = kwargs["num_numbers"]
            relation = kwargs["relation"]

            # Count regular digits (0-9, including full-width ０-９)
            regular_digit_count = sum(1 for ch in response if ch.isdigit() or ch in "０１２３４５６７８９")
            # Count Japanese numerals (including daiji and multipliers)
            japanese_numeral_count = sum(1 for ch in response if ch in JAPANESE_NUMERAL_CHARS)
            num_count = regular_digit_count + japanese_numeral_count
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
            punctuations = set(find_punctuations(response))
            # print(punctuations)

            if len(punctuations) < min_variants:
                return (
                    False,
                    f"Found {len(punctuations)} types of punctuations. Expected at least {min_variants}.\n {punctuations}",
                )

            return (True, "No error.")

    except Exception as e:
        return (False, f"Validation error: {str(e)}")

    return (False, "Invalid Instruction")
