from fractions import Fraction
import re
import string
import json
import os
from typing import Dict, List, Literal, Tuple, Any

from collections import Counter

from .. import validator as base_validator

import requests
from ..data_loader import DEFINITION_GENERATOR_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT, eval_modes, subinst_def, inst_def, LLM_JUDGE_QUESTION_PROMPT

from pydantic import BaseModel, ValidationError, Field
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

class JudgeResponse(BaseModel):
    """
    Defines the expected JSON structure for the LLM Judge's response.
    """
    verdict: Literal["YES", "NO"] = Field(..., description="The binary decision from the judge.")
    reasoning: str = Field(..., description="The explanation for the decision.")

class DefintionResponse(BaseModel):
    """
    Defines the expected JSON structure for the LLM Judge's response.
    """
    status: Literal["PASS", "FAIL"] = Field(..., description="The binary decision from the generator.")
    definition: str = Field(..., description="The definition of the term.")


# ===== Arabic-aware Unicode Helpers =====
# Arabic letters: Arabic + Supplement + Extended-A + Presentation Forms A/B
_AR_LETTERS_RANGES = "\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"
# Western + Arabic-Indic + Eastern Arabic-Indic digits
_AR_DIGITS_RANGES = "0-9\u0660-\u0669\u06F0-\u06F9"
# Word characters we consider inside tokens
_WORD_CHARS = f"A-Za-z{_AR_LETTERS_RANGES}{_AR_DIGITS_RANGES}"
# Sentence-ending punctuation (Arabic + Latin + Urdu)
_SENT_END = r"\.\!\?؟؛۔"
# Arabic punctuation to treat as punctuation alongside string.punctuation
_AR_PUNCT_EXTRA = "،؛؟۔"

# Precompiled patterns
_RE_WORD = re.compile(rf"[{_WORD_CHARS}]+(?:[\'’\-\.][{_WORD_CHARS}]+)*", flags=re.UNICODE)
_RE_SENT_SPLIT = re.compile(rf"[{_SENT_END}]+", flags=re.UNICODE)
_RE_SENT_ENDINGS = re.compile(rf"[{_SENT_END}]+", flags=re.UNICODE)
_RE_NUMBERED_ITEM = re.compile(
    rf"^\s*([{_AR_DIGITS_RANGES}]+)[\.\)\،]\s+",
    flags=re.MULTILINE | re.UNICODE
)  # supports 1. ١. ۱. 1) ١) ۱) 1، …

def _unicode_boundary_phrase_pattern(phrase: str) -> str:
    """
    Build Arabic-aware boundary pattern using custom char-class instead of \b.
    """
    escaped = [re.escape(part) for part in phrase.split()]
    joined = r"\s+".join(escaped)
    return rf"(?<![{_WORD_CHARS}]){joined}(?![{_WORD_CHARS}])"

# ========= New: English-letter sanitizer =========
_EN_LETTERS_RE = re.compile(r"[A-Za-z]")

def _strip_english_letters(text: str) -> str:
    """Remove only English alphabets; keep digits and punctuation."""
    if not isinstance(text, str):
        return text
    return _EN_LETTERS_RE.sub("", text)
# ================================================


def judge_llm_api(user_content, system_content="You are a chatbot", temperature=0.7, seed=42, top_p=1, top_k=40,
                  max_tokens=10000):
    url = os.getenv("OPENROUTER_API_BASE_URL")

    headers={
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json",
  }
    payload = {
        "model": "anthropic/claude-sonnet-4.5",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],

        "temperature": temperature,
        "seed": seed,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens
        
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
        # Sanitize input text (why: enforce Arabic-only validation logic)
        response = _strip_english_letters(response)

        judge_prompt = LLM_JUDGE_QUESTION_PROMPT.format(
            question=question_text,
            model_response=response
        )

        evaluation = judge_llm_api(
            user_content="Evaluate the response.",
            system_content=judge_prompt
        )

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
        flag = (judge_response.verdict == "YES")
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

def char_frequency(response: str, char: str) -> int:
    """Count frequency of a character in response."""
    return response.count(char)

def count_numbered_items(response: str) -> int:
    """Count number of numbered items in response (Arabic/Latin digits; ., ), Arabic comma)."""
    return len(_RE_NUMBERED_ITEM.findall(response))

def count_bullet_points(response: str) -> int:
    """Count number of bullet points in response."""
    return len(re.findall(r'^[*-]\s', response, re.MULTILINE))

def count_placeholders(response: str) -> int:
    """Count number of placeholders in response."""
    return len(re.findall(r'\[.*?\]', response))

def count_all_caps_words(response: str) -> int:
    """Count number of all-caps words in response."""
    return sum(1 for w in response.split() if w.isupper())

def count_lowercase_words(response: str) -> int:
    """Count number of lowercase words in response."""
    return sum(1 for w in response.split() if w.islower())

def word_frequency(response: str, word: str) -> int:
    """Count frequency of a (raw) token in response (case-insensitive for Latin only)."""
    words = re.findall(r'[^\s]+', response.lower())
    return words.count(word.lower())

def keyword_frequency(response: str, keyword: str) -> int:
    """Arabic-aware full-word/phrase frequency using custom boundaries."""
    keyword = keyword.strip()
    if not keyword:
        return 0
    pattern = _unicode_boundary_phrase_pattern(keyword)
    return len(re.findall(pattern, response, flags=re.IGNORECASE | re.UNICODE))

def is_first_letter_cap(token: str) -> bool:
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
    Handles 'inf' or formats like '1/0' as infinity.
    """
    if isinstance(input_str, (int, float)):
        return input_str # Return numbers directly
        
    if not isinstance(input_str, str):
        raise TypeError(f"Input must be a string, not {type(input_str)}")

    input_str = input_str.strip().lower()
    if input_str == 'inf':
        return float('inf')
    
    try:
        frac=Fraction(input_str)
        return frac
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Invalid input: '{input_str}'. Not a valid fraction or 'inf'.")
    
def extract_clean_sentences(text: str) -> List[str]:
    """
    Arabic-aware: split on . ! ? ؟ ؛ ۔ and handle list items without punctuation.
    """

    # Remove markdown tables
    table_pattern = r'(?:^\s*\|.*\|.*\n){2,}'
    cleaned_text = re.sub(table_pattern, '', text, flags=re.MULTILINE)

    # Remove horizontal rules
    rule_pattern = r'^\s*([*_-])\s*\1\s*\1+\s*$'
    text = re.sub(rule_pattern, '', cleaned_text, flags=re.MULTILINE)
    
    all_sentences = []
    for line in text.split('\n'):
        line = line.lstrip()
        # Remove list markers (bullets, headings, numbered with Arabic/Latin digits)
        line = re.sub(r'^\s*(?:[\-\*\+]\s+|[\d\u0660-\u0669\u06F0-\u06F9]+[.)،]\s+|#+\s+)', '', line)
        if not line:
            continue

        parts = _RE_SENT_SPLIT.split(line)
        for sentence in parts:
            s = sentence.strip()
            if s:
                all_sentences.append(s)
    return all_sentences

def extract_clean_words(response: str)-> List[str]:
    """
    Arabic-aware tokenizer: Arabic/Latin letters and digits, keeps in-word ' - . ’
    """
    text_without_lists = re.sub(r'^\s*(?:[\d\u0660-\u0669\u06F0-\u06F9]+[.)،]\s+)', '', response, flags=re.MULTILINE)
    return [w.lower() for w in _RE_WORD.findall(text_without_lists)]

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
        indent_level = len(indentation.strip('\n'))

        while current_list_stack and indent_level < current_list_stack[-1]['indent']:
            lists_found.append(current_list_stack.pop())

        if not current_list_stack or indent_level == current_list_stack[-1]['indent']:
            if not current_list_stack:
                nesting_level = 1
                current_list_stack.append({
                    'level': nesting_level,
                    'indent': indent_level,
                    'items': 1
                })
            else:
                current_list_stack[-1]['items'] += 1
        elif indent_level > current_list_stack[-1]['indent']:
            nesting_level = current_list_stack[-1]['level'] + 1
            current_list_stack.append({
                'level': nesting_level,
                'indent': indent_level,
                'items': 1
            })

    lists_found.extend(current_list_stack)
    
    return lists_found

def find_markdown_tables(text: str) -> list[dict]:
    """
    Finds all markdown tables in a text and determines their dimensions.
    """
    tables_found = []
    lines = text.strip().split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if '|' not in line:
            i += 1
            continue

        if i + 1 >= len(lines):
            break

        divider = lines[i + 1].strip()
        if '|' not in divider or not re.match(r'^[\s|: -]+$', divider):
            i += 1
            continue
       
        header_cols = [col.strip() for col in line.split('|') if col.strip()]
        num_cols = len(header_cols)

        divider_cols = [col.strip() for col in divider.split('|') if col.strip()]
        if len(divider_cols) != num_cols:
            i += 1
            continue

        num_rows = 0
        j = i + 2
        while j < len(lines) and '|' in lines[j]:
            num_rows += 1
            j += 1
        
        tables_found.append({
            'rows': num_rows,
            'columns': num_cols
        })
        i = j
    
    return tables_found

def find_punctuations(text: str)-> list[str]:
    cleaned_text = re.sub(r'^\s*(?:[\-\*\+]\s+|[\d\u0660-\u0669\u06F0-\u06F9]+[.)،]\s+|#+\s+)', '', text, flags=re.MULTILINE)
    # Arabic + Latin sentence enders
    punctuations = _RE_SENT_ENDINGS.findall(cleaned_text)
    return punctuations

def extract_clean_paragraphs(text: str) -> List[str]:
    """
    Extracts clean paragraphs from a text by removing markdown elements.
    """
    cleaned_text = re.sub(r'^\s*<<.*>>\s*$', '', text, flags=re.MULTILINE)

    table_pattern = r'(?:^\s*\|.*\|.*\n){2,}'
    cleaned_text = re.sub(table_pattern, '', cleaned_text, flags=re.MULTILINE)

    heading_pattern = r'^\s*#+\s+.*$'
    cleaned_text = re.sub(heading_pattern, '', cleaned_text, flags=re.MULTILINE)

    rule_pattern = r'^\s*([*_-])\s*\1\s*\1+\s*$'
    cleaned_text = re.sub(rule_pattern, '', cleaned_text, flags=re.MULTILINE)

    if not cleaned_text.strip():
        return []
    
    paragraphs = re.split(r'\n\s*\n', cleaned_text.strip())
    clean_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return clean_paragraphs

def validate_instruction(response: str, inst_type: str, kwargs: Dict[str, Any], all_instructions: Dict = None) -> Tuple[bool, str]:
    """Validate a response against a specific instruction type and its kwargs."""
    try:
        response = response.strip()
        # Sanitize input text (why: enforce pre-validation English-letter removal)
        response = _strip_english_letters(response)

        # ===== Not-applicable instructions for Arabic: always invalid at runtime =====
        NOT_APPLICABLE_FOR_ARABIC = {
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
        }
        if inst_type in NOT_APPLICABLE_FOR_ARABIC:
            return (False, "Invalid Instruction")

        if inst_type == "detectable_content:number_placeholders":
            count = count_placeholders(response)
            rel, val = kwargs["relation"], kwargs["num_placeholders"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
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
                "No error" if has_postscript else f"Postscript must start with '{marker}' and contain content. Found: '{last_line}'"
            )

        if inst_type == "detectable_format:json_format":
            try:
                json_part = response[response.find("{"):response.rfind("}")+1]
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
                return (False, "section_splitter must be a plain section name without '#' or '*'. The validator adds Markdown headers automatically.")

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

            if rel in ("at least", ">="):
                valid = count >= val
            elif rel in ("equal to", "==", "equals"):
                valid = count == val
            elif rel in ("less than", "<"):
                valid = count < val
            else:
                valid = count == val

            return (valid, "No error" if valid else f"Expected {rel} {val} sections like '### {splitter_clean} 1', found {count}.")

        if inst_type == "detectable_format:numbered_list":
            count = count_numbered_items(response)
            rel, val = kwargs["relation"], kwargs["num_numbered_items"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} numbered items, found {count}.")

        if inst_type == "detectable_format:number_bullet_lists":
            count = count_bullet_points(response)
            rel, val = kwargs["relation"], kwargs["num_bullets"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} bullet points, found {count}.")

        if inst_type == "detectable_format:title":
            line = response.splitlines()[0]
            found_title = line.strip().startswith("<<") and line.strip().endswith(">>")
            return (
                found_title,
                "No error" if found_title else "Title not wrapped in << >> on first line."
            )

        if inst_type == "keywords:existence":
            missing = [kw for kw in kwargs["keywords"] if keyword_frequency(response, kw) == 0]
            return (not missing, "No error" if not missing else f"Missing keyword(s): {missing}")

        if inst_type == "keywords:frequency":
            keyword = kwargs["keyword"].strip().lower()
            count = keyword_frequency(response, keyword)
            rel = kwargs["relation"]
            val = kwargs["frequency"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (
                valid,
                "No error" if valid else f"Expected {rel} {val} of '{keyword}', found {count}."
            )

        if inst_type == "keywords:forbidden_words":
            present = [w for w in kwargs["forbidden_words"] if keyword_frequency(response, w)]
            return (not present, "No error" if not present else f"Forbidden words found: {present}")

        if inst_type == "keywords:letter_frequency":
            letter = kwargs["letter"]
            # Case-insensitive for Latin, literal match otherwise (works for Arabic)
            count = response.count(letter) + (response.count(letter.swapcase()) if hasattr(letter, "swapcase") and letter.swapcase() != letter else 0)
            rel, val = kwargs["let_relation"], kwargs["let_frequency"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (
                valid,
                "No error" if valid else f"Expected {rel} {val} '{letter}', found {count}."
            )

        if inst_type == "punctuation:no_comma":
            # Arabic comma '،' counts as comma
            has_comma = (',' in response) or ('،' in response)
            return (not has_comma, "No error" if not has_comma else "Commas found in response (',' or '،').")

        if inst_type == "length_constraints:number_characters":
            count = len(response)
            rel, val = kwargs["relation"], kwargs["num_chars"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} characters, found {count}.")

        if inst_type == "length_constraints:number_words":
            # Use Arabic-aware tokenizer
            count = len(_RE_WORD.findall(response))
            rel, val = kwargs["relation"], kwargs["num_words"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} words, found {count}.")

        if inst_type == "startend:start_checker":
            # Extend punctuation to include Arabic punctuations when stripping
            starts_correctly = response.lstrip(string.punctuation + _AR_PUNCT_EXTRA + " ").lower().startswith(kwargs.get("start_phrase", "").lower())
            return (
                starts_correctly,
                "No error" if starts_correctly else "Response does not start with required phrase."
            )

        if inst_type == "startend:end_checker":
            required = kwargs["end_phrase"].strip()
            # punctuation-aware
            ends_with_punctuation = required[-1] in (string.punctuation + _AR_PUNCT_EXTRA) if required else False
            actual_words = response.lstrip(string.punctuation + _AR_PUNCT_EXTRA).strip().split()
            if not actual_words:
                return (False, "Empty response")
            if ends_with_punctuation:
                actual_phrase = " ".join(actual_words[-len(required.split()):])
                if actual_phrase.lower() != required.lower():
                    return (False, f"End phrase mismatch: expected '{required}', but found '{actual_phrase}'")
            else:
                actual_phrase = " ".join(actual_words).rstrip(string.punctuation + _AR_PUNCT_EXTRA + " ")[-len(required):]
                if actual_phrase.lower() != required.lower():
                    return (False, f"End phrase mismatch: expected '{required}', but found '{actual_phrase}'")
            return (True, "No error")

        if inst_type == "startend:wrap_checker":
            wrap = kwargs["wrap_phrase"]
            return (response.startswith(wrap) and response.endswith(wrap),
                    "No error" if response.startswith(wrap) and response.endswith(wrap) else f"Not wrapped with: {wrap}")

        if inst_type == "startend:quotation":
            return (response.startswith('"') and response.endswith('"'),
                    "No error" if response.startswith('"') and response.endswith('"') else "Response not wrapped in double quotes.")
            
        if inst_type == "detectable_format:number_paragraphs":
            cleaned_response = response.strip().replace("\r\n", "\n")
            paragraphs = extract_clean_paragraphs(response)
            actual_paragraph_count = len([p for p in paragraphs if p.strip()])
            if not cleaned_response:
                actual_paragraph_count = 0

            relation = kwargs["relation"]
            num_paragraphs = kwargs["num_paragraphs"]

            if relation == "equal to":
                is_valid = actual_paragraph_count == num_paragraphs
            elif relation == "at least":
                is_valid = actual_paragraph_count >= num_paragraphs
            elif relation == "less than":
                is_valid = actual_paragraph_count < num_paragraphs
            else:
                return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
            
            message = f"Found {actual_paragraph_count} paragraphs. Expected {num_paragraphs}"
            return (is_valid, "No error." if is_valid else message)
 
        if inst_type == "detectable_format:max_paragraph_length":
            max_chars=kwargs["max_chars"]
            paragraphs = extract_clean_paragraphs(response)
            for p in paragraphs:
                p = re.sub(r'^\s*(?:[\-\*\+]\s+|[\d\u0660-\u0669\u06F0-\u06F9]+[.)،]\s+|#+\s+)', '', p.lstrip())
                char_count=len(p.strip())
                if char_count>max_chars:
                    return (False, f"Found a paragraph containing {char_count} characters.\n '{p}'")
            return (True, "No error.")

        if inst_type == "detectable_format:sentences_per_paragraph":
            num_sentences=kwargs["num_sentences"]
            relation=kwargs["relation"]
            paragraphs = extract_clean_paragraphs(response)
            for p in paragraphs:
                sentences=extract_clean_sentences(p)
                sentence_count=len([s for s in sentences if s.strip()])
                if sentence_count == 0 and p.strip():
                    sentence_count = 1
                if relation == "equal to":
                    is_valid = sentence_count == num_sentences
                elif relation == "at least":
                    is_valid = sentence_count >= num_sentences
                elif relation == "less than":
                    is_valid = sentence_count < num_sentences
                else:
                    return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
                if not is_valid:
                    return (False, f"Found {sentence_count} sentences. Expected {num_sentences}\n '{p}'")
            return (True, "No error.")

        if inst_type == "detectable_format:indentation":
            # Logic for this instruction to be added here
            return (False, "Invalid Instruction")

        if inst_type == "length_constraints:sentence_length":
            sentences = extract_clean_sentences(response)
            max_words=kwargs["max_words"]
            if not sentences:
                return (True, "No sentences found to validate.")
            for s in sentences:
                word_count=len(_RE_WORD.findall(s))
                if word_count > max_words:
                    return (False, f"Expected at most {max_words} words. Found {word_count} words in '{s}'")
            return (True, "No error.")
            
        if inst_type == "length_constraints:word_repetition":
            max_repeats=kwargs["max_repeats"]
            words= extract_clean_words(response)
            word_counts = Counter(words)
            for word, count in word_counts.items():
                if count > max_repeats:
                    return(False, f"Word '{word}' appears {count} times (limit {max_repeats})")
            return (True, "No error.")
            
        if inst_type == "length_constraints:unique_words":
            relation=kwargs["relation"]
            num_unique=kwargs["num_unique"]
            words= extract_clean_words(response)
            unique_words_count = len(set(words))
            if relation == "equal to":
                is_valid = unique_words_count == num_unique
            elif relation == "at least":
                is_valid = unique_words_count >= num_unique
            elif relation == "less than":
                is_valid = unique_words_count < num_unique
            else:
                return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
            if not is_valid:
                return (False, f"Found {unique_words_count} unique words. Expected {relation} {num_unique}.")
            return (True, "No error.")        

        if inst_type == "punctuation:frequency":
            # Logic for this instruction to be added here
            return (False, "Invalid Instruction")

        if inst_type == "punctuation:balance":
            # Logic for this instruction to be added here
            return (False, "Invalid Instruction")

        if inst_type == "punctuation:question_exclaim":
            relation=kwargs["relation"]
            num_marks=kwargs["num_marks"]
            punctuations = re.findall(r"[?!؟]", response)
            count = len(punctuations)
            if relation == "equal to":
                is_valid= count == num_marks
            elif relation == "less than":
                is_valid= count < num_marks
            elif relation == "at least":
                is_valid= count >= num_marks
            else:
                raise ValueError("Invalid relation. Use 'equal to', 'less than', or 'at least'")
            if not is_valid:
                return (False, f"Found {count} marks. Expected {relation} {num_marks}.")
            return (True, "No error.")   

        if inst_type == "punctuation:no_period":
            # Treat both '.' and Urdu '۔' as periods
            no_period = ('.' not in response) and ('۔' not in response)
            return (no_period, "No error" if no_period else "Periods found in response ('.' or '۔').")

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
            words=extract_clean_words(response)
            all_count= sum(1 for word in words if word.startswith(target_letter))
            if relation == "equal to":
                is_valid = all_count == num_alliteration
            elif relation == "at least":
                is_valid = all_count >= num_alliteration
            elif relation == "less than":
                is_valid = all_count < num_alliteration
            else:
                return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
            if not is_valid:
                return (False, f"Found {all_count} alliteration words. Expected {relation} {num_alliteration}.")
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
            if 0 <= position < len(words) and words[position] == keyword:
                return (True, "No error.")
            found = words[position] if 0 <= position < len(words) else "<out-of-range>"
            return (False, f"'{found}' found after {position} words instead of '{keyword}'.")
            
        if inst_type == "detectable_format:nested_list":
            min_depth = kwargs["min_depth"]
            num_subitems = kwargs["num_subitems"]
            bullet_pattern = r"^(\s*)([*+-])[ \t]+(.*)"
            numbered_pattern = rf"^(\s*)([{_AR_DIGITS_RANGES}]+[.)،])[ \t]+(.*)"
            lists = analyze_lists(response, bullet_pattern) + analyze_lists(response, numbered_pattern)
            for l in lists:
                if l['level'] == min_depth and l['items'] >= num_subitems:
                    return (True, "No error.")
            return (False, f"List at level {min_depth} with at least {num_subitems} items not found.")

        if inst_type == "detectable_format:table":
            min_rows = kwargs["min_rows"]
            min_cols = kwargs["min_cols"]
            tables = find_markdown_tables(response)
            for table in tables:
                if table['rows'] >= min_rows and table['columns'] >= min_cols:
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

        if inst_type == "length_constraints:word_length":
            max_length=kwargs["max_length"]
            min_length=kwargs["min_length"]
            if min_length>max_length:
                return (False, "Validation failed: Minimum length greater than maximum length.")
            words = set(extract_clean_words(response))
            if not words:
                return (True, "No words found to validate.")
            shortest_word = min(words, key=len)
            longest_word = max(words, key=len)
            if len(shortest_word) < min_length:
                return (False, f"Validation failed: The word '{shortest_word}' with length {len(shortest_word)} is shorter than the minimum of {min_length}.")
            if len(longest_word) > max_length:
                return (False, f"Validation failed: The word '{longest_word}' with length {len(longest_word)} is longer than the maximum of {max_length}.")
            return (True, "No error.")   

        if inst_type == "length_constraints:avg_word_length":
            min_ratio=kwargs["min_ratio"]
            max_ratio=kwargs["max_ratio"]
            if min_ratio>max_ratio:
                return (False, "Validation failed: Minimum length greater than maximum length.")
            words = extract_clean_words(response)
            if not words:
                is_valid= min_ratio==0
                return (is_valid, "No words found to validate.")
            avg_count=sum(len(word) for word in words)/len(words)
            is_valid= min_ratio<=avg_count<=max_ratio
            return (is_valid, "No error" if is_valid else f"Found average of {avg_count}. Expected between {min_ratio} and {max_ratio}")

        if inst_type == "detectable_format:sentence_count":
            relation= kwargs["relation"]
            num_sentences= kwargs["num_sentences"]
            sentence_count = len(extract_clean_sentences(response))
            if relation == "equal to":
                is_valid = sentence_count == num_sentences
            elif relation == "at least":
                is_valid = sentence_count >= num_sentences
            elif relation == "less than":
                is_valid = sentence_count < num_sentences
            else:
                return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
            if not is_valid:
                return (False, f"Found {sentence_count} sentences. Expected {relation} {num_sentences}")
            return (True, "No error.")
        
        if inst_type == "length_constraints:paragraph_length":
            words_per_paragraph=kwargs["words_per_paragraph"]
            relation=kwargs["relation"]
            paragraphs = extract_clean_paragraphs(response)
            for p in paragraphs:
                words=extract_clean_words(p)
                word_count=len([s for s in words if s.strip()])
                if relation == "equal to":
                    is_valid = word_count == words_per_paragraph
                elif relation == "at least":
                    is_valid = word_count >= words_per_paragraph
                elif relation == "less than":
                    is_valid = word_count < words_per_paragraph
                else:
                    return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
                if not is_valid:
                    return (False, f"Found {word_count} words. Expected {relation} {words_per_paragraph}\n '{p}'")
            return (True, "No error.")

        if inst_type == "punctuation:variety":
            # Logic for this instruction to be added here
            return (False, "Invalid Instruction")

        if inst_type == "detectable_content:numeric_inclusion":
            num_numbers=kwargs["num_numbers"]
            relation=kwargs["relation"]
            num_count = sum(1 for ch in response if ch.isdigit())  # Unicode-aware
            if relation == "equal to":
                is_valid = num_count == num_numbers
            elif relation == "at least":
                is_valid = num_count >= num_numbers
            elif relation == "less than":
                is_valid = num_count < num_numbers
            else:
                return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
            if not is_valid:
                return (False, f"Found {num_count} digits. Expected {relation} {num_numbers}")
            return (True, "No error.")

        if inst_type == "detectable_format:sentence_endings":
            min_variants = kwargs["min_variants"]; 
            punctuations = set(find_punctuations(response))
            if len(punctuations) < min_variants:
                return (False, f"Found {len(punctuations)} types of punctuations. Expected at least {min_variants}.\n {punctuations}")
            return (True, "No error.")

    except Exception as e:
        return (False, f"Validation error: {str(e)}")

    return (False, "Invalid Instruction")

def _get_dynamic_definition(inst_type: str, term: str, cache: Dict[Tuple[str, str], Tuple[str, bool]]) -> Tuple[str, bool]:
    """
    Calls an LLM to dynamically define a sub-instruction term.
    Returns (definition, is_valid)
    """
    cache_key = (inst_type, term)
    if cache_key in cache:
        return cache[cache_key]
    try:
        instruction_name = inst_def.get(inst_type, {}).get("instruction_name", inst_type)
        context_terms_list = list(subinst_def.get(inst_type, {}).keys())
        context_terms_str = ", ".join(context_terms_list) if context_terms_list else "none"

        system_prompt = DEFINITION_GENERATOR_SYSTEM_PROMPT.format(
            instruction=instruction_name,
            inst_label=inst_type,
            term=term,
            context_related_terms=context_terms_str
        )

        response_str = judge_llm_api(
            user_content=f"Define the term: {term}",
            system_content=system_prompt
        )

        evaluation = response_str.strip()
        if evaluation.strip().startswith("```"):
            evaluation = re.sub(r"^```(?:\w+)?\s*", "", evaluation, flags=re.DOTALL)
            evaluation = re.sub(r"\s*```$", "", evaluation, flags=re.DOTALL)

        json_match = re.search(r"(\{.*\})", evaluation, re.DOTALL)
        if json_match:
            evaluation = json_match.group(1)

        json_data = json.loads(evaluation)

        definition = json_data.get("definition", "definition not found")
        status = json_data.get("status", "FAIL")
        
        print("Definition: ", definition)
        print("Status: ", status)

        if status == "PASS":
            result = (definition, True)
        else:
            result = (definition, False)

        cache[cache_key] = result
        return result
    
    except (json.JSONDecodeError, KeyError) as e:
        return (f"Error parsing definition response: {e}. Raw: '{evaluation}'", False)
    except Exception as e:
        return (f"Error in _get_dynamic_definition: {e}", False)

def validate_prompt_against_instructions(user_prompt: str, turn_instructions: Dict) -> Tuple[bool, str]:
    return base_validator.validate_prompt_against_instructions(
        user_prompt, turn_instructions, language="ar"
    )

