from fractions import Fraction
import re
import string
import json
from typing import Dict, List, Literal, Tuple, Any
import copy
import json
import re
import os

from collections import (Counter, defaultdict)

import requests
from data_loader import DEFINITION_GENERATOR_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT, PROMPT_VALIDATION_JUDGE_SYSTEM_PROMPT, THINKING_VALIDATION_JUDGE_SYSTEM_PROMPT, template_json, conflict_dict, LLM_INSTRUCTIONS, eval_modes, subinst_def, inst_def, LLM_JUDGE_QUESTION_PROMPT

from pydantic import BaseModel, ValidationError, Field
from dotenv import load_dotenv

import validators.validator as base_validator

from notebook_processing.processor import NotebookParsingError

# Import language strategy functions from base validator
from validators.validator import _get_strategy

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


# Map of expected kwargs for each instruction ID
EXPECTED_ARGUMENTS = {
    # --- PRIORITY (must be first, before any other instruction entries) ---
    # "length_constraints:number_characters": ["relation", "num_chars"],
    # "length_constraints:number_words": ["relation", "num_words"],
    # "change_case:lowercase_word_frequency": ["lowercase_relation", "lowercase_frequency"],
    # "keywords:letter_frequency": ["letter", "let_relation", "let_frequency"],
    # "change_case:capital_word_frequency": ["capital_relation", "capital_frequency"],
    # "change_case:vowel_consonant_balance": ["min_fraction", "max_fraction"],
    # "detectable_format:max_paragraph_length": ["max_chars"],
    # "length_constraints:sentence_length": ["max_words"],
    # "punctuation:question_exclaim": ["relation", "num_marks"],
    # "keywords:alliteration": ["target_letter", "relation", "num_alliteration"],
    # "keywords:palindrome_word": ["min_length"],
    # "keywords:positioning": ["keyword", "position"],
    # "length_constraints:word_length": ["min_length", "max_length"],
    # "length_constraints:avg_word_length": ["min_ratio", "max_ratio"],
    # "length_constraints:paragraph_length": ["relation", "words_per_paragraph"],
    # "detectable_content:numeric_inclusion": ["relation", "num_numbers"],
    # "keywords:vowel_count": ["relation", "num_vowels"],
    # "keywords:consonant_count": ["relation", "num_consonants"],

    # --- NON-PRIORITY (original order, with priority keys NOT repeated) ---
    "change_case:all_caps": [],
    "change_case:lowercase": [],
    "change_case:alternating": [],
    "change_case:first_letter_cap": [],
    "change_case:all_caps_target": ["target_string"],
    "change_case:lowercase_target": ["target_string"],
    "change_case:alternating_target": ["target_string"],
    "change_case:first_letter_cap_target": ["target_string"],
    "detectable_content:number_placeholders": ["relation", "num_placeholders"],
    "detectable_content:postscript": ["postscript_marker"],
    "detectable_format:json_format": [],
    "detectable_format:multiple_sections": ["section_splitter", "relation", "num_sections"],
    "detectable_format:numbered_list": ["relation", "num_numbered_items"],
    "detectable_format:number_bullet_lists": ["relation", "num_bullets"],
    "detectable_format:title": [],
    "keywords:existence": ["keywords"],
    "keywords:frequency": ["keyword", "relation", "frequency"],
    "keywords:forbidden_words": ["forbidden_words"],
    "punctuation:no_comma": [],
    "length:max_word_count": ["max_words"],
    "startend:start_checker": ["start_phrase"],
    "startend:end_checker": ["end_phrase"],
    "startend:wrap_checker": ["wrap_phrase"],
    "startend:quotation": [],

    # New VIF Instructions (excluding any priority duplicates)
    "change_case:case_ratio": ["min_fraction", "max_fraction"],
    "change_case:first_letter_sentence": [],
    "change_case:last_letter": ["case"],
    "detectable_format:number_paragraphs": ["relation", "num_paragraphs"],
    "detectable_format:sentences_per_paragraph": ["relation", "num_sentences"],
    "detectable_format:indentation": ["indent_type", "size"],
    "length_constraints:word_repetition": ["max_repeats"],
    "length_constraints:unique_words": ["relation", "num_unique"],
    "punctuation:frequency": ["punctuation", "relation", "frequency"],
    "punctuation:balance": [],
    "punctuation:no_period": [],
    "punctuation:end_rule": ["allowed"],
    "detectable_format:nested_list": ["min_depth", "num_subitems"],
    "detectable_format:table": ["min_rows", "min_cols"],
    "detectable_format:heading_depth": ["levels"],
    "detectable_format:section_balance": ["element_type", "count"],
    "detectable_format:sentence_count": ["relation", "num_sentences"],
    "punctuation:variety": ["min_types"],
    "detectable_format:sentence_endings": ["min_variants"],

    # New LLM Judge Instructions
    "stylistic:tone_formality": ["tone_level"],
    "stylistic:emotional_tone": ["emotion_type"],
    "stylistic:politeness": ["politeness_degree"],
    "stylistic:descriptive_level": ["description_degree"],
    "stylistic:literary_style": ["style_type"],
    "stylistic:sentence_tone_consistency": ["tone_type"],
    "stylistic:voice": ["voice_type"],
    "stylistic:figurative_language": ["figure_type", "relation", "num_occurrences"],
    "stylistic:tone_transition": ["from_tone", "to_tone", "transition_position"],
    "stylistic:emotive_adjectives": ["relation", "num_adjectives"],
    "stylistic:sensory_detail": ["sense_type", "relation", "num_details"],
    "stylistic:rhythm_pattern": ["rhythm_type"],
    "linguistic:pragmatic_context": ["context_type"],
    "linguistic:speech_act": ["act_type"],
    "linguistic:syntactic_pattern": ["pattern_type"],
    "linguistic:grammatical_mood": ["mood_type"],
    "linguistic:morphological_form": ["form_type"],
    "linguistic:phonological_pattern": ["phonology_type"],
    "linguistic:sound_symbolism": ["relation", "num_symbolisms"],
    "situation:role_based": ["role_type"],
    "situation:task_specific": ["task_type"],
    "situation:audience_alignment": ["audience_type"],
    "situation:contextual_scenario": ["scenario_type"],
    "situation:perspective": ["perspective_type"],
    "situation:emotional_alignment": ["emotion_type"],
    "situation:cultural_context": ["culture_type", "adaptation_level"],
    "situation:temporal_context": ["time_frame"],
    "situation:environment_setting": ["environment_type"],
}


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
    """Count number of numbered items in response."""
    return len(re.findall(r'^\s*\d+\.', response, re.MULTILINE))

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
    """Count frequency of a word in response."""
    words = re.findall(r'[^\s]+', response.lower())
    return words.count(word.lower())

def tokenize_korean_words(text: str, use_spacing: bool = True) -> List[str]:
    """
    Tokenize Korean text into words.
    
    For word counting instructions (number_words, unique_words, etc.):
    - Use SPACING to split words (Korean words are separated by spaces)
    - Each space-separated segment is a word
    
    For other instructions (alliteration):
    - Can use syllable-based tokenization if needed
    
    Args:
        use_spacing: If True, split on whitespace (for word counting).
                     If False, segment by syllables (for alliteration).
    """
    # Remove list markers
    text_without_lists = re.sub(r'^\s*\d+\.\s', '', text, flags=re.MULTILINE)
    
    if use_spacing:
        # For word counting: split on whitespace
        # Each space-separated segment is a word
        # This is the correct approach for Korean word counting
        words = re.split(r'\s+', text_without_lists)
        # Filter out empty strings and clean each word
        # Remove any remaining punctuation that might be attached
        words = [w.strip() for w in words if w.strip()]
        return words
    else:
        # For alliteration: segment by Hangul syllables
        # Korean Hangul range: AC00-D7AF
        hangul_pattern = re.compile(r'[\uAC00-\uD7AF]')
        words = []
        segments = re.split(r'\s+', text_without_lists)
        
        for segment in segments:
            if not segment.strip():
                continue
            
            if hangul_pattern.search(segment):
                # Segment by Hangul syllables
                current_word = []
                for char in segment:
                    if hangul_pattern.match(char):
                        if current_word:
                            words.append(''.join(current_word))
                            current_word = []
                        words.append(char)
                    elif char.isalnum() or char in "'-":
                        current_word.append(char)
                    else:
                        if current_word:
                            words.append(''.join(current_word))
                            current_word = []
                if current_word:
                    words.append(''.join(current_word))
            else:
                standard_words = re.findall(r'[^\W_]+(?:[\'\-][^\W_]+)*', segment, flags=re.UNICODE)
                words.extend(standard_words)
        
        return [w for w in words if w.strip()]

def keyword_frequency(response: str, keyword: str, language: str = "ko") -> int:
    """Count frequency of a keyword in response, using substring matching for Korean."""
    strategy = _get_strategy(language)
    keyword = keyword.strip()
    keyword_normalized = strategy.casefold(keyword)

    # For Korean (CJK), use substring matching instead of word boundaries
    # Korean words can be compound and keywords should match even when part of a larger word
    if strategy.word_script == "cjk" and language == "ko":
        # Use simple substring matching for Korean
        response_normalized = strategy.casefold(response)
        # Count all occurrences of the keyword as a substring
        return response_normalized.count(keyword_normalized)

    # For other languages, use word boundaries
    escaped_tokens = [re.escape(part) for part in keyword.split()]
    phrase_pattern = r"\s+".join(escaped_tokens)
    pattern = rf"(?<![\w]){phrase_pattern}(?![\w])"

    return len(re.findall(pattern, strategy.casefold(response), flags=re.UNICODE))

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
    
def extract_clean_sentences(text: str, language: str = "ko") -> List[str]:
    """
    Takes a raw text string and returns a clean list of sentences.
    This version correctly handles list items that do not end with punctuation
    by treating each cleaned line as a source for one or more sentences.
    Uses Korean language strategy for sentence delimiters.
    """
    strategy = _get_strategy(language)
    delims = strategy.sentence_delims

    # Remove markdown tables
    table_pattern = r'(?:^\s*\|.*\|.*\n){2,}'
    cleaned_text = re.sub(table_pattern, '', text, flags=re.MULTILINE)

    # Remove horizontal rules
    rule_pattern = r'^\s*([*_-])\s*\1\s*\1+\s*$'
    text = re.sub(rule_pattern, '', cleaned_text, flags=re.MULTILINE)
    
    
    # print(text)
    
    all_sentences = []
    
    # Process the text line by line
    for line in text.split('\n'):
        # Clean the line by removing markdown markers and leading space
        line = line.lstrip()
        cleaned_line = re.sub(r'^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)', '', line)

        if not cleaned_line:
            continue

        # Split the individual cleaned line into sentences by language delimiters.
        line_parts = re.split(f"[{re.escape(delims)}]+", cleaned_line)

        # 3. Add the resulting parts to our main list after cleaning them.
        for sentence in line_parts:
            stripped_sentence = sentence.strip()
            if stripped_sentence:
                all_sentences.append(stripped_sentence)
        
    # print(all_sentences)
                
    return all_sentences

def extract_clean_words(response: str, language: str = "ko", use_spacing: bool = True)-> List[str]:
    """
    Extract clean words using Korean language strategy for tokenization.
    
    Args:
        language: Language code
        use_spacing: For Korean, if True use spacing-based tokenization (for word counting),
                     if False use syllable-based (for alliteration)
    """
    if language == "ko":
        # Use Korean-specific tokenization
        return tokenize_korean_words(response, use_spacing=use_spacing)
    
    strategy = _get_strategy(language)
    text_without_lists = re.sub(r'^\s*\d+\.\s', '', response, flags=re.MULTILINE)
    return strategy.tokenize_words(text_without_lists)

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

        # If the stack is empty or indentation is less than the last list,
        # it means all previous lists have ended.
        while current_list_stack and indent_level < current_list_stack[-1]['indent']:
            lists_found.append(current_list_stack.pop())

        # If the stack is empty or indentation is the same, it's a new top-level list
        # or another item in the current list.
        if not current_list_stack or indent_level == current_list_stack[-1]['indent']:
            if not current_list_stack: # A new top-level list starts
                nesting_level = 1
                current_list_stack.append({
                    'level': nesting_level,
                    'indent': indent_level,
                    'items': 1
                })
            else: # Another item in the current-level list
                current_list_stack[-1]['items'] += 1
        
        # If indentation increases, a new nested list has started
        elif indent_level > current_list_stack[-1]['indent']:
            nesting_level = current_list_stack[-1]['level'] + 1
            current_list_stack.append({
                'level': nesting_level,
                'indent': indent_level,
                'items': 1
            })

    # Add any remaining lists from the stack to the results
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

        # potential header row (must contain '|')
        if '|' not in line:
            i += 1
            continue

        # divider line immediately after the header
        if i + 1 >= len(lines):
            break # Reached end of text

        divider = lines[i + 1].strip()
        # A valid divider must contain '|' and be made of '-', '|', ':', and whitespace.
        if '|' not in divider or not re.match(r'^[\s|: -]+$', divider):
            i += 1
            continue
            
       
        # Determine the number of columns from the header.
        # We count the segments between pipes, ignoring empty segments from start/end pipes.
        header_cols = [col.strip() for col in line.split('|') if col.strip()]
        num_cols = len(header_cols)

        # The divider line must have a number of segments that matches the header.
        divider_cols = [col.strip() for col in divider.split('|') if col.strip()]
        if len(divider_cols) != num_cols:
            i += 1
            continue

        # Count the data rows
        num_rows = 0
        j = i + 2 # Start counting from the line after the divider
        while j < len(lines) and '|' in lines[j]:
            num_rows += 1
            j += 1
        
        tables_found.append({
            'rows': num_rows,
            'columns': num_cols
        })
        
        # Move the main index past this entire table
        i = j
    
    return tables_found

def find_punctuations(text: str, language: str = "ko")-> list[str]:
    """Find punctuation marks using Korean language strategy."""
    strategy = _get_strategy(language)
    delims = strategy.punctuation_marks
    cleaned_text = re.sub(r'^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)', '', text, flags=re.MULTILINE)
    # For Korean, also include period (.) as punctuation
    if language == "ko":
        # Korean uses both Korean and Latin punctuation
        korean_punct = "。！？"  # Korean punctuation marks
        all_punct = delims + korean_punct
        punctuations = re.findall(f"[{re.escape(all_punct)}]+", cleaned_text)
    else:
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
    cleaned_text = re.sub(r'^\s*<<.*>>\s*$', '', text, flags=re.MULTILINE)

    # Remove markdown tables
    table_pattern = r'(?:^\s*\|.*\|.*\n){2,}'
    cleaned_text = re.sub(table_pattern, '', cleaned_text, flags=re.MULTILINE)

    # Remove markdown headings
    heading_pattern = r'^\s*#+\s+.*$'
    cleaned_text = re.sub(heading_pattern, '', cleaned_text, flags=re.MULTILINE)

    # Remove horizontal rules
    rule_pattern = r'^\s*([*_-])\s*\1\s*\1+\s*$'
    cleaned_text = re.sub(rule_pattern, '', cleaned_text, flags=re.MULTILINE)

    # Split the fully cleaned text into paragraphs
    # A paragraph is a block of text separated by one or more blank lines.
    if not cleaned_text.strip():
        return []
    
    paragraphs = re.split(r'\n\s*\n', cleaned_text.strip())
    
    # Final filter to remove any empty strings that might remain
    clean_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return clean_paragraphs

def validate_instruction(response: str, inst_type: str, kwargs: Dict[str, Any], all_instructions: Dict = None) -> Tuple[bool, str]:
    """Validate a response against a specific instruction type and its kwargs."""
    try:
        response = response.strip()
        strategy = _get_strategy("ko")
        
        # Korean doesn't support case rules - return False for all case-related instructions
        if inst_type.startswith("change_case:"):
            if not strategy.supports_case_rules:
                return (False, f"Korean language does not support case-related instructions. '{inst_type}' is not applicable.")
            # If somehow case rules are supported, continue with normal validation
            if inst_type == "change_case:all_caps":
                return (response.isupper(), "No error" if response.isupper() else "Response is not all uppercase.")
            if inst_type == "change_case:lowercase":
                return (response.islower(), "No error" if response.islower() else "Response is not all lowercase.")
            if inst_type == "change_case:alternating":
                valid = all(is_strict_alternating(w) for w in response.split() if w.isalpha())
                return (valid, "No error" if valid else "Response is not strictly alternating.")
            if inst_type == "change_case:first_letter_cap":
                valid = all(is_first_letter_cap(tok) for tok in response.split())
                return (valid, "No error" if valid else "Each word must start with one uppercase letter followed only by lowercase letters.")
            if inst_type == "change_case:capital_word_frequency":
                count = count_all_caps_words(response)
                rel, val = kwargs['capital_relation'], kwargs['capital_frequency']
                valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
                return (valid, "No error" if valid else f"Expected {rel} {val} all-cap words, found {count}.")
            if inst_type == "change_case:lowercase_word_frequency":
                count = count_lowercase_words(response)
                rel, val = kwargs['lowercase_relation'], kwargs['lowercase_frequency']
                valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
                return (valid, "No error" if valid else f"Expected {rel} {val} lowercase words, found {count}.")

        if "_target" in inst_type and inst_type.startswith("change_case:"):
            if not strategy.supports_case_rules:
                return (False, f"Korean language does not support case-related instructions. '{inst_type}' is not applicable.")
            target = kwargs["target_string"].strip()
            strategy_ko = _get_strategy("ko")
            target_normalized = strategy_ko.casefold(target)
            response_normalized = strategy_ko.casefold(response)
            
            # For CJK, use substring matching
            if strategy_ko.word_script == "cjk":
                if target not in response:
                    return (False, f"Target '{target}' not found in response.")
                # For CJK, we can't validate case, so just check existence
                return (True, "No error")
            else:
                target_escaped = re.escape(target)
                pattern = rf'\b{target_escaped}\b'
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
            missing = [kw for kw in kwargs["keywords"] if keyword_frequency(response, kw, "ko") == 0]
            return (not missing, "No error" if not missing else f"Missing keyword(s): {missing}")

        if inst_type == "keywords:frequency":
            keyword = kwargs["keyword"].strip()
            strategy_ko = _get_strategy("ko")
            keyword_normalized = strategy_ko.casefold(keyword)
            count = keyword_frequency(response, keyword, "ko")
            rel = kwargs["relation"]
            val = kwargs["frequency"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (
                valid,
                "No error" if valid else f"Expected {rel} {val} of '{keyword}', found {count}."
            )

        if inst_type == "keywords:forbidden_words":
            present = [w for w in kwargs["forbidden_words"] if keyword_frequency(response, w, "ko")]
            return (not present, "No error" if not present else f"Forbidden words found: {present}")

        if inst_type == "keywords:letter_frequency":
            letter = kwargs["letter"].lower()
            count = response.lower().count(letter)
            rel, val = kwargs["let_relation"], kwargs["let_frequency"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (
                valid,
                "No error" if valid else f"Expected {rel} {val} '{letter}' (case-insensitive), found {count}."
            )

        if inst_type == "punctuation:no_comma":
            return (',' not in response, "No error" if ',' not in response else "Commas found in response.")

        if inst_type == "length_constraints:number_characters":
            count = len(response)
            rel, val = kwargs["relation"], kwargs["num_chars"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} characters, found {count}.")

        if inst_type == "length_constraints:number_words":
            """
            For Korean: word counting is based on spacing.
            Each space-separated segment is a word.
            """
            # Use spacing-based tokenization for Korean
            count = len(extract_clean_words(response, "ko", use_spacing=True))
            rel, val = kwargs["relation"], kwargs["num_words"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} words, found {count}.")

        if inst_type == "startend:start_checker":
            strategy_ko = _get_strategy("ko")
            start_phrase = kwargs.get("start_phrase", "")
            response_normalized = strategy_ko.casefold(response.lstrip(string.punctuation + " "))
            phrase_normalized = strategy_ko.casefold(start_phrase)
            starts_correctly = response_normalized.startswith(phrase_normalized)
            return (
                starts_correctly,
                "No error" if starts_correctly else "Response does not start with required phrase."
            )

        if inst_type == "startend:end_checker":
            """
            For Korean: Check if response ends with the exact required phrase.
            The last phrase should start from the last spacing (word boundary).
            - If required phrase has punctuation, it must match exactly (including punctuation type).
            - The phrase must match from the last word boundary, not as a substring.
            - Example: "배고프다" should match, but "프다" or "다" should NOT match.
            """
            required = kwargs["end_phrase"].strip()
            strategy_ko = _get_strategy("ko")
            
            response_stripped = response.rstrip()
            required_stripped = required.rstrip()
            
            # All possible punctuation marks
            all_punctuation = string.punctuation + "。！？"
            
            # Check if required phrase ends with punctuation
            required_ends_with_punct = required_stripped and required_stripped[-1] in all_punctuation
            
            if required_ends_with_punct:
                # Exact match including punctuation - punctuation type must match exactly
                # Get the last word(s) from response based on spacing
                # Split required phrase to get words (spacing-based)
                required_words = required_stripped.split()
                if not required_words:
                    return (False, "Required phrase is empty")
                
                # Get last words from response (same count as required)
                response_words = response_stripped.split()
                if len(response_words) < len(required_words):
                    return (False, f"Response has fewer words than required phrase")
                
                # Get the last N words from response (where N = number of words in required phrase)
                actual_words = response_words[-len(required_words):]
                actual_phrase = ' '.join(actual_words)
                
                # Check exact match including punctuation
                if actual_phrase == required_stripped:
                    return (True, "No error")
                else:
                    return (False, f"End phrase mismatch: expected '{required_stripped}', but found '{actual_phrase}'")
            else:
                # Required doesn't end with punctuation - check text part, ignoring trailing punctuation
                # Get words from required phrase (spacing-based)
                required_words = required_stripped.split()
                if not required_words:
                    return (False, "Required phrase is empty")
                
                # Get last words from response (same count as required)
                response_words = response_stripped.split()
                if len(response_words) < len(required_words):
                    return (False, f"Response has fewer words than required phrase")
                
                # Get the last N words from response
                actual_words = response_words[-len(required_words):]
                actual_phrase = ' '.join(actual_words)
                
                # Remove trailing punctuation from both for comparison
                actual_text = actual_phrase.rstrip(all_punctuation + " \n\t")
                required_text = required_stripped.rstrip(all_punctuation + " \n\t")
                
                # Check if they match (case-insensitive for Korean)
                response_normalized = strategy_ko.casefold(actual_text)
                required_normalized = strategy_ko.casefold(required_text)
                
                if response_normalized == required_normalized:
                    return (True, "No error")
                else:
                    return (False, f"End phrase mismatch: expected '{required_text}', but found '{actual_text}'")

        if inst_type == "startend:wrap_checker":
            wrap = kwargs["wrap_phrase"]
            return (response.startswith(wrap) and response.endswith(wrap),
                    "No error" if response.startswith(wrap) else f"Not wrapped with: {wrap}")

        if inst_type == "startend:quotation":
            return (response.startswith('"') and response.endswith('"'),
                    "No error" if response.startswith('"') else "Response not wrapped in double quotes.")
            
        if inst_type == "change_case:case_ratio":
            """
            Returns True if the ratio of lowercase to uppercase letters lies between
            minR and maxR (inclusive). Otherwise, returns False.
            Korean doesn't support case rules.
            """
            if not strategy.supports_case_rules:
                return (False, "Korean language does not support case-related instructions. 'change_case:case_ratio' is not applicable.")
            
            try:
                minR = parse_fraction_or_inf(kwargs["min_fraction"])
                maxR = parse_fraction_or_inf(kwargs["max_fraction"])
            except (ValueError, ZeroDivisionError) as e:
                raise ValueError(f"Invalid fraction input: {e}")
            
            if minR>maxR:
                return (False, "Validation failed: Minimum ratio greater than maximum ratio.")
            lower_count = sum(1 for ch in response if ch.islower())
            upper_count = sum(1 for ch in response if ch.isupper())

            if lower_count == 0 and upper_count == 0:
                print("Validation failed: No letters found in the string.")
                return False
            

            # The ratio variable will hold either a Fraction object or float('inf')
            if upper_count == 0:
                ratio = float('inf')
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
            return (
                valid,
                "No error" if valid else f"{message}"
            )

        if inst_type == "change_case:first_letter_sentence":
            """
            Checks if all sentences in the text start with an uppercase alphabet.
            Korean doesn't support case rules.
            """
            if not strategy.supports_case_rules:
                return (False, "Korean language does not support case-related instructions. 'change_case:first_letter_sentence' is not applicable.")
            
            sentences = extract_clean_sentences(response, "ko")

            if not sentences:
                return (True, "No sentences found to validate.")

            # print(sentences)
            for sentence in sentences:
                sentence = sentence.strip("()[]{}\"'")
                
                if not sentence[0].isupper():  # check first char
                    return (False, f"Fails at: '{sentence}'")
            
            return (True, "No error.")

        if inst_type == "change_case:last_letter":
            """
            Checks if the last character of the last word in the text matches the given case.
            For Korean, we support digit and special checks, but not uppercase/lowercase.
            """
            strategy_ko = _get_strategy("ko")
            delims = strategy_ko.sentence_delims
            # Remove sentence-ending punctuation
            cleaned_text = re.sub(f'[{re.escape(delims)}]+$', '', response.strip())

            if not cleaned_text:
                return (False, "Empty response")  # Empty after cleaning

            # For Korean, get the last space-separated word
            # Split on whitespace to get words (spacing-based)
            words = cleaned_text.split()
            if not words:
                return (False, "No words found in response")
            
            last_word = words[-1]

            # Strip wrapping punctuation like (), [] , {} , quotes
            last_word = last_word.strip("()[]{}\"'")

            if not last_word:
                return (False, "Last word is empty after cleaning")

            last_char = last_word[-1]
            valid = True
            case_type = kwargs["case"]

            if case_type == "uppercase":
                if not strategy_ko.supports_case_rules:
                    return (False, "Korean language does not support case-related instructions. 'uppercase' case check is not applicable.")
                valid = last_char.isupper()
            elif case_type == "lowercase":
                if not strategy_ko.supports_case_rules:
                    return (False, "Korean language does not support case-related instructions. 'lowercase' case check is not applicable.")
                valid = last_char.islower()
            elif case_type == "digit":
                valid = last_char.isdigit()
            elif case_type == "special":
                # For Korean, special includes punctuation and non-alphanumeric
                # Korean characters (Hangul) are considered alphanumeric
                valid = not (last_char.isalnum() or '\uAC00' <= last_char <= '\uD7AF')
            else:
                return (False, f"Invalid case type: {case_type}")
            
            return (valid, "No error." if valid else f"Last character of the response: {last_char}")

        if inst_type == "change_case:vowel_consonant_balance":
            """
            Korean doesn't support vowel rules.
            """
            if not strategy.supports_vowel_rules:
                return (False, "Korean language does not support vowel/consonant balance instructions. 'change_case:vowel_consonant_balance' is not applicable.")
            
            try:
                minR = parse_fraction_or_inf(kwargs["min_fraction"])
                maxR = parse_fraction_or_inf(kwargs["max_fraction"])
            except (ValueError, ZeroDivisionError) as e:
                raise ValueError(f"Invalid fraction input: {e}")
            
            if minR>maxR:
                return (False, "Validation failed: Minimum ratio greater than maximum ratio.")
            
            # Use Korean vowels from language strategy (though Korean doesn't support this)
            vowels = strategy.vowels
            vowel_count = sum(1 for ch in response if ch.isalpha() and ch in vowels)
            consonant_count = sum(1 for ch in response if ch.isalpha() and ch not in vowels)

            # Handle the case where there are no letters at all
            if vowel_count == 0 and consonant_count == 0:
                return (False, "Validation failed: No letters found in the response.")
            

            # Handle the case where there are no consonants (infinite ratio)
            if consonant_count == 0:
                ratio = float('inf')
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
            return (
                valid,
                "No error" if valid else f"{message}"
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


            if relation == "equal to":
                is_valid = actual_paragraph_count == num_paragraphs
            elif relation == "at least":
                is_valid = actual_paragraph_count >= num_paragraphs
            elif relation == "less than":
                is_valid = actual_paragraph_count < num_paragraphs
            else:
                return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
            
            message = (
                f"Found {actual_paragraph_count} paragraphs. Expected {num_paragraphs}"
            )
            
            return (is_valid, "No error." if is_valid else message)
 
        if inst_type == "detectable_format:max_paragraph_length":
            """
            Checks if the number of characters in each paragraph (including spaces and special characters)
            is at most the given expected_count.
            """
            max_chars=kwargs["max_chars"]
            paragraphs = extract_clean_paragraphs(response)


            # print(paragraphs)
            
            for p in paragraphs:
                p = re.sub(r'^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)', '', p.lstrip())
                # print(p)
                char_count=len(p.strip())
                if char_count>max_chars:
                    return (False, f"Found a paragraph containing {char_count} characters.\n '{p}'")
                
            return (True, "No error.")

        if inst_type == "detectable_format:sentences_per_paragraph":
            """
            Checks if the number of sentences in each paragraph satisfies relation with a given number.
            """
            num_sentences=kwargs["num_sentences"]
            relation=kwargs["relation"]
            paragraphs = extract_clean_paragraphs(response)


            # print(paragraphs)
            is_valid=True
            
            for p in paragraphs:

                # print(p)
                sentences=extract_clean_sentences(p, "ko")
                
                sentence_count=len([s for s in sentences if s.strip()])
                if sentence_count == 0 and p.strip():
                    sentence_count = 1
                    
                # print(sentence_count, relation, num_sentences)
                
                if relation == "equal to":
                    is_valid = sentence_count == num_sentences
                elif relation == "at least":
                    is_valid = sentence_count >= num_sentences
                elif relation == "less than":
                    is_valid = sentence_count < num_sentences
                else:
                    return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
                
                if not is_valid:
                    message = (
                        f"Found {sentence_count} sentences. Expected {num_sentences}\n '{p}'"
                    )
                    return (False, message)
                    
            return (True, "No error.")

        if inst_type == "detectable_format:indentation":
            # Logic for this instruction to be added here
            return (False, "Invalid Instruction")

        if inst_type == "length_constraints:sentence_length":
            """
            Checks if the number of words in each sentence (including bullet list items: '-' and numbered lists '1.')
            must be less than or equal to max_words.

            """
            sentences = extract_clean_sentences(response, "ko")
            max_words=kwargs["max_words"]
            
            if not sentences:
                return (True, "No sentences found to validate.")
            
            for s in sentences:
                # Use spacing-based tokenization for Korean
                word_count=len(extract_clean_words(s, "ko", use_spacing=True))
                if word_count > max_words:
                    return (False, f"Expected at most {max_words} words. Found {word_count} words in '{s}'")
            
            return (True, "No error.")
            

        if inst_type == "length_constraints:word_repetition":
            """
            For Korean: word counting is based on spacing.
            Each space-separated segment is a word.
            """
            max_repeats=kwargs["max_repeats"]
            # Use spacing-based tokenization for Korean
            words = extract_clean_words(response, "ko", use_spacing=True)
            
            # Count occurrences
            word_counts = Counter(words)

            # Check if any word exceeds max_repeats
            for word, count in word_counts.items():
                if count > max_repeats:
                    return(False, f"Word '{word}' appears {count} times (limit {max_repeats})")
                    
            return (True, "No error.")
            

        if inst_type == "length_constraints:unique_words":
            """
            For Korean: word counting is based on spacing.
            Each space-separated segment is a word.
            Example: "배고픈 고양이" = 2 words
            """
            relation=kwargs["relation"]
            num_unique=kwargs["num_unique"]
            # Use spacing-based tokenization for Korean
            words = extract_clean_words(response, "ko", use_spacing=True)
            
            # Convert to set to get unique words
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
                message = (
                    f"Found {unique_words_count} unique words. Expected {relation} {num_unique}."
                )
                return (False, message)
                    
            return (True, "No error.")        

        if inst_type == "punctuation:frequency":
            # Logic for this instruction to be added here
            return (False, "Invalid Instruction")


        if inst_type == "punctuation:balance":
            # Logic for this instruction to be added here
            return (False, "Invalid Instruction")

        if inst_type == "punctuation:question_exclaim":
            is_valid=True
            relation=kwargs["relation"]
            num_marks=kwargs["num_marks"]
            
            punctuation_pattern = r"[?!]"

            # Find all punctuation characters
            punctuations = re.findall(punctuation_pattern, response)

            count = len(punctuations)
            # print("Count of punctuations: ", count)

            if relation == "equal to":
                is_valid= count == num_marks
            elif relation == "less than":
                is_valid= count < num_marks
            elif relation == "at least":
                is_valid= count >= num_marks
            else:
                raise ValueError("Invalid relation. Use 'equal to', 'less than', or 'at least'")
            
            if not is_valid:
                message = (
                    f"Found {count} marks. Expected {relation} {num_marks}."
                )
                return (False, message)
                    
            return (True, "No error.")   

        if inst_type == "punctuation:no_period":
            return ('.' not in response, "No error" if '.' not in response else "Periods found in response.")

        if inst_type == "punctuation:end_rule":
            allowed = kwargs["allowed"]
            
            punctuations = find_punctuations(response, "ko")
            # Normalize punctuation marks - '.' should match '.' in allowed list
            # For Korean, period (.) is commonly used and should be recognized
            normalized_allowed = set(str(a) for a in allowed)
            # If period is in allowed list (as '.' or '。'), accept both
            if '.' in normalized_allowed or '。' in normalized_allowed:
                normalized_allowed.add('.')
                normalized_allowed.add('。')

            for p in punctuations:
                # Check if punctuation is in allowed list
                if p not in normalized_allowed and p not in allowed:
                    return (False, f"'{p}' not in the list of allowed punctuations.")
            
            return (True, "No error.")

        if inst_type == "keywords:alliteration":
            """
            For Korean: alliteration is based on syllables (1 Korean letter = 1 syllable).
            The target_letter should be the syllable itself (like '다'), not the consonant.
            Example: "다음은 요청하신..." - first syllable is '다'
            If target_letter is '다', it should match syllables starting with '다'.
            """
            relation = kwargs["relation"]
            num_alliteration = kwargs["num_alliteration"]
            target_letter = kwargs["target_letter"]
            
            # For alliteration, use syllable-based tokenization (not spacing-based)
            words = extract_clean_words(response, "ko", use_spacing=False)
            
            # For Korean, check if the syllable itself matches the target_letter
            # The target_letter should be a Korean syllable (like '다'), not a consonant
            all_count = 0
            for word in words:
                if not word:
                    continue
                # Check if it's a Korean character (Hangul syllable)
                if '\uAC00' <= word[0] <= '\uD7AF':
                    # Check if the syllable itself matches the target_letter
                    # target_letter should be a syllable like '다', not a consonant like 'ㄷ'
                    if word[0] == target_letter:
                        all_count += 1
                else:
                    # For non-Korean words, check if starts with target letter (case-insensitive)
                    if word[0].lower() == target_letter.lower():
                        all_count += 1
            
            if relation == "equal to":
                is_valid = all_count == num_alliteration
            elif relation == "at least":
                is_valid = all_count >= num_alliteration
            elif relation == "less than":
                is_valid = all_count < num_alliteration
            else:
                return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
            
            if not is_valid:
                message = (
                    f"Found {all_count} alliteration words. Expected {relation} {num_alliteration}."
                )
                return (False, message)
                    
            return (True, "No error.")

        if inst_type == "keywords:palindrome_word":
            """
            For Korean: word counting is based on spacing.
            Each space-separated segment is a word.
            """
            min_length = kwargs["min_length"]
            # Use spacing-based tokenization for Korean
            words = extract_clean_words(response, "ko", use_spacing=True)
            for word in words:
                if word == word[::-1] and len(word) >= min_length:
                    return (True, f"No error. Word: {word}")
            
            return (False, "No valid palindrome words found.")
            
        if inst_type == "keywords:positioning":
            """
            For Korean: positioning is based on spacing.
            Each space-separated segment is a word.
            """
            keyword = kwargs["keyword"]
            position = kwargs["position"]
            
            # Use spacing-based tokenization for Korean
            words = extract_clean_words(response, "ko", use_spacing=True)
            
            if position >= len(words):
                return (False, f"Position {position} is out of range. Response has {len(words)} words.")
            
            # Check if the word at position matches the keyword (case-insensitive for Korean)
            word_at_position = words[position]
            strategy_ko = _get_strategy("ko")
            
            # Check exact match (case-insensitive)
            if strategy_ko.casefold(word_at_position) == strategy_ko.casefold(keyword):
                return (True, "No error.")
            
            return (False, f"'{word_at_position}' found after {position} words instead of '{keyword}'.")
            
            

        if inst_type == "detectable_format:nested_list":
            min_depth = kwargs["min_depth"]
            num_subitems = kwargs["num_subitems"]
            
            bullet_pattern = r"^(\s*)([*+-])[ \t]+(.*)"
            numbered_pattern = r"^(\s*)(\d+\.)[ \t]+(.*)"
            
            lists = analyze_lists(response, bullet_pattern) + analyze_lists(response, numbered_pattern)
            
            # print(bullet_lists, numbered_lists)
            
            for l in lists:
                if l['level'] == min_depth and l['items'] >= num_subitems:
                    return (True, "No error.")
            
            return (False, f"List at level {min_depth} with at least {num_subitems} items not found.")

        if inst_type == "detectable_format:table":
            min_rows = kwargs["min_rows"]
            min_cols = kwargs["min_cols"]
            
            tables = find_markdown_tables(response)
            
            # print(tables)
            
            for table in tables:
                if table['rows'] >= min_rows and table['columns'] >= min_cols:
                    return (True, "No error.")
            
            return (False, f"Could not find a table with at least {min_rows} rows and {min_cols} columns.")

        if inst_type == "detectable_format:heading_depth":
            levels = kwargs["levels"]
            
            if not levels:
                return (False, "No levels provided.")
            
            # Handle both list and single value (string or int)
            # Convert to list if it's a single value
            if not isinstance(levels, list):
                levels = [levels]
            
            # Convert all levels to integers for comparison
            try:
                levels = [int(level) for level in levels]
            except (ValueError, TypeError):
                return (False, f"Invalid levels format: {levels}. Expected list of integers or single integer.")
            
            # Markdown heading pattern: # followed by space or tab, then text
            # Pattern should match: ### heading (with space/tab after #)
            # Use same pattern as other validators: [ \t]+ allows space or tab
            heading_pattern = re.compile(r"^\s*(#+)[ \t]+(.*)", re.MULTILINE)

            all_headings = heading_pattern.findall(response)
            all_headings = set([len(item[0]) for item in all_headings])
            
            for level in levels:
                if level not in all_headings:
                    return (False, f"Heading of level {level} not found")
            
            return (True, "No error.")
                    
        if inst_type == "detectable_format:section_balance":
            # Logic for this instruction to be added here
            return (False, "Invalid Instruction")

        if inst_type == "length_constraints:word_length":
            """
            For Korean: word counting is based on spacing.
            Each space-separated segment is a word.
            """
            max_length=kwargs["max_length"]
            min_length=kwargs["min_length"]
            
            # Ensure these are numbers
            try:
                min_length = int(min_length) if not isinstance(min_length, int) else min_length
                max_length = int(max_length) if not isinstance(max_length, int) else max_length
            except (ValueError, TypeError):
                return (False, f"Invalid length values: min_length={min_length}, max_length={max_length}")
            
            if min_length>max_length:
                return (False, "Validation failed: Minimum length greater than maximum length.")

            # Use spacing-based tokenization for Korean
            words = extract_clean_words(response, "ko", use_spacing=True)

            if not words:
                return (True, "No words found to validate.")

            # Find words that violate the constraints
            too_short = [w for w in words if len(w) < min_length]
            too_long = [w for w in words if len(w) > max_length]

            if too_short:
                shortest_word = min(too_short, key=len)
                return (
                    False, 
                    f"Validation failed: The word '{shortest_word}' with length {len(shortest_word)} is shorter than the minimum of {min_length}."
                )
            if too_long:
                longest_word = max(too_long, key=len)
                return (
                    False, 
                    f"Validation failed: The word '{longest_word}' with length {len(longest_word)} is longer than the maximum of {max_length}."
                )
            return (True, "No error.")   

        if inst_type == "length_constraints:avg_word_length":
            """
            For Korean: word counting is based on spacing.
            Each space-separated segment is a word.
            """
            is_valid=True
            min_ratio=kwargs["min_ratio"]
            max_ratio=kwargs["max_ratio"]
            
            # Ensure min_ratio and max_ratio are numbers
            try:
                min_ratio = float(min_ratio) if not isinstance(min_ratio, (int, float)) else min_ratio
                max_ratio = float(max_ratio) if not isinstance(max_ratio, (int, float)) else max_ratio
            except (ValueError, TypeError):
                return (False, f"Invalid ratio values: min_ratio={min_ratio}, max_ratio={max_ratio}")
            
            if min_ratio>max_ratio:
                return (False, "Validation failed: Minimum length greater than maximum length.")

            # Use spacing-based tokenization for Korean
            words = extract_clean_words(response, "ko", use_spacing=True)
            
            if not words:
                is_valid= min_ratio==0
                return (is_valid, "No words found to validate.")
            
            avg_count = sum(len(word) for word in words) / len(words)
            is_valid= min_ratio<=avg_count<=max_ratio
            
            return (is_valid, "No error" if is_valid else f"Found average of {avg_count:.2f}. Expected between {min_ratio} and {max_ratio}")

        if inst_type == "detectable_format:sentence_count":
            relation= kwargs["relation"]
            num_sentences= kwargs["num_sentences"]
            sentence_count = len(extract_clean_sentences(response, "ko"))
            
            if relation == "equal to":
                is_valid = sentence_count == num_sentences
            elif relation == "at least":
                is_valid = sentence_count >= num_sentences
            elif relation == "less than":
                is_valid = sentence_count < num_sentences
            else:
                return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
            
            if not is_valid:
                message = (
                    f"Found {sentence_count} sentences. Expected {relation} {num_sentences}"
                )
                return (False, message)
                    
            return (True, "No error.")
        
        if inst_type == "length_constraints:paragraph_length":
            """
            Checks if the number of words in each paragraph satisfies relation with a given number.
            """
            words_per_paragraph=kwargs["words_per_paragraph"]
            relation=kwargs["relation"]
            
            # Treat multiple "Enters" as a single paragraph break.
            paragraphs = extract_clean_paragraphs(response)


            # print(paragraphs)
            is_valid=True
            
            for p in paragraphs:

                # print(p)
                # Use spacing-based tokenization for Korean
                words = extract_clean_words(p, "ko", use_spacing=True)
                word_count = len(words)
                # if word_count == 0 and p.strip():
                #     word_count = 1
                    
                # print(word_count, relation, words_per_paragraph)
                
                if relation == "equal to":
                    is_valid = word_count == words_per_paragraph
                elif relation == "at least":
                    is_valid = word_count >= words_per_paragraph
                elif relation == "less than":
                    is_valid = word_count < words_per_paragraph
                else:
                    return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
                
                if not is_valid:
                    message = (
                        f"Found {word_count} words. Expected {relation} {words_per_paragraph}\n '{p}'"
                    )
                    return (False, message)
                    
            return (True, "No error.")

        if inst_type == "punctuation:variety":
            # Logic for this instruction to be added here
            return (False, "Invalid Instruction")

        if inst_type == "detectable_content:numeric_inclusion":
            
            num_numbers=kwargs["num_numbers"]
            relation=kwargs["relation"]
            
            num_count = sum(1 for ch in response if ch.isdigit())
            # print("Numeric Count:", num_count)
            
            if relation == "equal to":
                is_valid = num_count == num_numbers
            elif relation == "at least":
                is_valid = num_count >= num_numbers
            elif relation == "less than":
                is_valid = num_count < num_numbers
            else:
                return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
            
            if not is_valid:
                message = (
                    f"Found {num_count} digits. Expected {relation} {num_numbers}"
                )
                return (False, message)
                    
            return (True, "No error.")

        if inst_type == "detectable_format:sentence_endings":
            min_variants = kwargs["min_variants"]; 
            
            punctuations = set(find_punctuations(response, "ko"))
            # print(punctuations)

            if len(punctuations) < min_variants:
                return (False, f"Found {len(punctuations)} types of punctuations. Expected at least {min_variants}.\n {punctuations}")
            
            return (True, "No error.")

        if inst_type == "keywords:vowel_count":
            """
            Korean doesn't support vowel rules, but we can still count Latin vowels if present.
            """
            if not strategy.supports_vowel_rules:
                # Still allow counting Latin vowels in mixed text, but warn that Korean vowels aren't counted
                pass
            
            num_vowels=kwargs["num_vowels"]
            relation=kwargs["relation"]
            
            # Use Korean vowels from language strategy (though Korean doesn't support this)
            # For Korean, we'll count Latin vowels only
            vowels = set("aeiouAEIOU")
            vowel_count = sum(1 for ch in response if ch in vowels)

            # print("Vowel count:", vowel_count)
            if relation == "equal to":
                is_valid = vowel_count == num_vowels
            elif relation == "at least":
                is_valid = vowel_count >= num_vowels
            elif relation == "less than":
                is_valid = vowel_count < num_vowels
            else:
                return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
            
            if not is_valid:
                message = (
                    f"Found {vowel_count} vowels. Expected {relation} {num_vowels}"
                )
                return (False, message)
                    
            return (True, "No error.")

        if inst_type == "keywords:consonant_count":
            """
            Korean doesn't support vowel rules, but we can still count Latin consonants if present.
            """
            if not strategy.supports_vowel_rules:
                # Still allow counting Latin consonants in mixed text, but warn that Korean consonants aren't counted
                pass
            
            num_consonants=kwargs["num_consonants"]
            relation=kwargs["relation"]
            
            # Use Korean vowels from language strategy (though Korean doesn't support this)
            # For Korean, we'll count Latin consonants only
            vowels = set("aeiouAEIOU")
            consonants = set(string.ascii_letters) - vowels
            consonant_count = sum(1 for ch in response if ch in consonants)

            # print("consonant count:", consonant_count)
            if relation == "equal to":
                is_valid = consonant_count == num_consonants
            elif relation == "at least":
                is_valid = consonant_count >= num_consonants
            elif relation == "less than":
                is_valid = consonant_count < num_consonants
            else:
                return (False, "Invalid 'relation' argument. Use 'equal to', 'at least', or 'less than'.")
            
            if not is_valid:
                message = (
                    f"Found {consonant_count} consonants. Expected {relation} {num_consonants}"
                )
                return (False, message)
                    
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
        # Check cache first. If found, return immediately.
        return cache[cache_key]
    try:
        # 1. Get context for the definition prompt
        instruction_name = inst_def.get(inst_type, {}).get("instruction_name", inst_type)
        context_terms_list = list(subinst_def.get(inst_type, {}).keys())
        context_terms_str = ", ".join(context_terms_list) if context_terms_list else "none"

        # 2. Format the system prompt
        system_prompt = DEFINITION_GENERATOR_SYSTEM_PROMPT.format(
            instruction=instruction_name,
            inst_label=inst_type,
            term=term,
            context_related_terms=context_terms_str
        )

        # 3. Call the LLM API
        # We use the existing judge_llm_api, passing our new prompt as system_content
        # and a simple user_content to trigger the response.
        response_str = judge_llm_api(
            user_content=f"Define the term: {term}",
            system_content=system_prompt
        )

        # 4. Parse the LLM's JSON response
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

        # 5. Store the new result in the cache before returning
        cache[cache_key] = result
        return result
    
    except (json.JSONDecodeError, KeyError) as e:
        return (f"Error parsing definition response: {e}. Raw: '{evaluation}'", False)
    except Exception as e:
        return (f"Error in _get_dynamic_definition: {e}", False)

def validate_llm_instruction(
    response: str,
    inst_type: str,
    kwargs: Dict[str, Any],
    all_instructions: Dict = None,
    definition_cache: Dict[Tuple[str, str], Tuple[bool, str]] = None,
) -> Tuple[bool, str]:
    if inst_type not in LLM_INSTRUCTIONS:
        return False, f"Instruction '{inst_type}' is not an LLM instruction."
    return base_validator.validate_llm_instruction(
        response, inst_type, kwargs, all_instructions, definition_cache, language="ko"
    )

def validate_prompt_against_instructions(user_prompt: str, turn_instructions: Dict) -> Tuple[bool, str]:
    return base_validator.validate_prompt_against_instructions(
        user_prompt, turn_instructions, language="ko"
    )

def validate_thinking_against_instructions(thinking: str, turn_instructions: Dict) -> Tuple[bool, str]:
    return base_validator.validate_thinking_against_instructions(
        thinking, turn_instructions, language="ko"
    )


def check_contradicting_instructions(instructions_list: List[Dict]) -> List[Dict]:
    """Check for contradicting instruction IDs in the list (order-insensitive)."""
    errors, seen_pairs = set(), set()
    # Collect all instruction IDs
    ids = {inst["instruction_id"] for inst in instructions_list if isinstance(inst, dict) and "instruction_id" in inst}

    # Check each instruction for conflicts
    for instr_id in ids:
        for conflicting_id in conflict_dict.get(instr_id, []):
            pair = frozenset([instr_id, conflicting_id])
            if conflicting_id in ids and pair not in seen_pairs:
                errors.add(f"{instr_id} and {conflicting_id} are contradicting")
                seen_pairs.add(pair)
    return errors

def validate_instruction_schema(instructions: Dict) -> List[Dict]:
    """Validate the schema of instructions against expected arguments and check for contradicting instructions."""
    mismatches = []
    
    # Validate metadata field
    metadata = instructions.get("metadata", [])
    if not isinstance(metadata, list):
        mismatches.append({
            "field": "metadata",
            "expected": "list",
            "actual": type(metadata).__name__
        })
    
    # Validate instructions array
    instructions_list = instructions.get("instructions", [])
    if not isinstance(instructions_list, list):
        mismatches.append({
            "field": "instructions",
            "expected": "list",
            "actual": type(instructions_list).__name__
        })
        return mismatches

    # Check for contradicting instructions
    contradiction_errors = check_contradicting_instructions(instructions_list)
    mismatches.extend(contradiction_errors)


    # Validate each instruction object
    for i, inst in enumerate(instructions_list):
        if not isinstance(inst, dict):
            mismatches.append({
                "instruction_index": i,
                "expected": "dict",
                "actual": type(inst).__name__
            })
            continue

        # Check for required instruction_id field
        if "instruction_id" not in inst:
            mismatches.append({
                "instruction_index": i,
                "missing_field": "instruction_id"
            })
            continue

        # Validate kwargs against expected arguments
        expected_args = set(EXPECTED_ARGUMENTS.get(inst["instruction_id"], []))
        actual_args = set(k for k in inst.keys() if k != "instruction_id")

        if expected_args != actual_args:
            mismatches.append({
                "instruction": inst["instruction_id"],
                "expected_args": sorted(expected_args),
                "actual_args": sorted(actual_args)
            })

    return mismatches 

def extract_notebook_sections_as_dict(ipynb_path):
    with open(ipynb_path, 'r', encoding='utf-8') as file:
        notebook_data = json.load(file)

    result = defaultdict(list)

    for cell_idx, cell in enumerate(notebook_data.get('cells', [])):
        if cell.get('cell_type') != 'markdown':
            continue

        content = ''.join(cell.get('source', [])).strip()
        split_lines = content.splitlines()
        if split_lines[0] == '# Metadata':
            result['task_metadata'].append(content)
            continue
        elif cell_idx == 0:
            raise NotebookParsingError("The first cell must be a markdown cell with the title '# Metadata'.")

        match = re.search(r'\*\*\[([\w.]+)]\*\*', split_lines[0])
        title = match.group(1)

        result[title].append('\n'.join(split_lines[1:]))

    return result


def validate_notebook_schema(notebook, template_json, log_filename):
    logs = []
    try:
        dict_turn_metadata = turn_metadata_json_to_dict(notebook['turn_metadata'])
        correct_turn_metadata = compare_consecutive_metadata_items(dict_turn_metadata)
        
        conflicting_instructions = find_conflicting_instructions(dict_turn_metadata)
        issues_in_keys_against_template = validate_keys_against_template(template_json, dict_turn_metadata)
        issues_in_instruction_kwargs_datatype = validate_instruction_kwargs_datatype(dict_turn_metadata)

        logs.append(f'CONFLICTING INSTRUCTIONS FOUND - {conflicting_instructions}')
        logs.append(f'INSTRUCTION ARGUMENT MISMATCHES IN TURN JSON - {issues_in_keys_against_template}')
        logs.append(f'VALIDATING JSON SCHEMA - {issues_in_instruction_kwargs_datatype}')

        i, flag = 1, False
        for t, f in zip(correct_turn_metadata, dict_turn_metadata):
            if t['metadata'] != f['metadata']:
                logs.append(f"TURN {i} METADATA SHOULD BE {t['metadata']}, BUT IS {f['metadata']}")
                flag = True
            i += 1
        if not flag:
            logs.append("TURN METADATA IS CORRECT")
            if any([conflicting_instructions, issues_in_keys_against_template, issues_in_instruction_kwargs_datatype]):
                logs.append('False')
            else:
                logs.append('True')
        else:
            logs.append('False')
    except Exception as e:
        logs.append(f'Some error occurred while validating the notebook - {e}')
    finally:
        with open(log_filename, "w", encoding="utf-8") as f:
            f.writelines(line + '\n' for line in logs)


def turn_metadata_json_to_dict(turn_metadata):
    parsed_json_metadata = []
    for item in turn_metadata:
        # Extract the JSON block between triple backticks
        match = re.search(r"```(?:\w+)?\n(.*?)```", item, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            # Parse the JSON string into a dictionary
            data = json.loads(json_str)
            data['metadata'] = set(data['metadata'])
            parsed_json_metadata.append(data)
        else:
            raise "No JSON found in item."
    return parsed_json_metadata


def compare_consecutive_metadata_items(dict_turn_metadata):
    def to_dict(instructions):
        return {instr['instruction_id']: instr for instr in instructions}

    updated = []

    for idx, current_turn in enumerate(dict_turn_metadata):
        if idx == 0:
            updated.append(copy.deepcopy(current_turn))  # Keep the first as-is
            continue

        prev_instr = to_dict(dict_turn_metadata[idx - 1]['instructions'])
        curr_instr = to_dict(current_turn['instructions'])

        metadata = set()

        # Check for additions and modifications
        for instr_id, instr in curr_instr.items():
            if instr_id not in prev_instr:
                metadata.add("add")
            elif instr != prev_instr[instr_id]:
                metadata.add("modify")

        # Check for removals
        for instr_id in prev_instr:
            if instr_id not in curr_instr:
                metadata.add("remove")

        # Avoid duplicates
        current_copy = copy.deepcopy(current_turn)
        current_copy['metadata'] = metadata
        updated.append(current_copy)

    return updated


def find_conflicting_instructions(dict_turn_metadata):
    conflicts_found = []

    for data in dict_turn_metadata:
        instruction_ids = {instr["instruction_id"] for instr in data.get("instructions", [])}
        current_conflicts = []

        for instr_id in instruction_ids:
            if instr_id in conflict_dict:
                for conflicting_id in conflict_dict[instr_id]:
                    if conflicting_id in instruction_ids:
                        pair = tuple(sorted((instr_id, conflicting_id)))
                        if pair not in current_conflicts:
                            current_conflicts.append(pair)

        if current_conflicts:
            conflicts_found.append(current_conflicts)

    return conflicts_found


def validate_keys_against_template(template_json, dict_turn_metadata):
    # Map template instruction_id to expected key set
    template_keys = {
        instr["instruction_id"]: set(instr.keys())
        for instr in template_json.get("instructions", [])
    }
    idx, res = 1, []

    for input_json in dict_turn_metadata:
        mismatches = {}

        # Check each instruction in input_json
        for instr in input_json.get("instructions", []):
            instr_id = instr.get("instruction_id")
            input_keys = set(instr.keys())

            if instr_id not in template_keys:
                mismatches[instr_id] = {
                    "error": "instruction_id not in template"
                }
            elif input_keys != template_keys[instr_id]:
                mismatches[instr_id] = {
                    "missing_keys": list(template_keys[instr_id] - input_keys),
                    "extra_keys": list(input_keys - template_keys[instr_id])
                }
        res.append({f"TURN {idx}": mismatches}) if mismatches else ''
        idx += 1
    return res


# def is_valid_str(val):
    return isinstance(val, str)

def is_valid_str(val):
    return isinstance(val, str)

def is_valid_int(val):
    return isinstance(val, int)

def is_valid_float(val):
    return isinstance(val, float)

def is_valid_list_str(val):
    return isinstance(val, list) and all(isinstance(i, str) for i in val)

def is_valid_list_int(val):
    return isinstance(val, list) and all(isinstance(i, int) for i in val)

def is_valid_relation(val):
    return val in {"at least", "equal to", "less than"}

# Map type names to the validation functions
TYPE_VALIDATORS = {
    "str": is_valid_str,
    "int": is_valid_int,
    "float": is_valid_float,
    "list_int": is_valid_list_int,
    "list_str": is_valid_list_str,
    "relation": is_valid_relation,
}

type_map = {
        "int": "int",
        "float": "float",
        "str": "str",
        "list(str)": "list_str",
        "list(int)": "list_int",
        "{at least, equal to, less than}": "relation"
    }

def generate_schema_from_template(template):
    """Dynamically creates the validation schema from the template_json."""
    schema = {}
    for instruction in template.get("instructions", []):
        iid = instruction.get("instruction_id")
        if not iid:
            continue
        args = {}
        for key, value in instruction.items():
            if key == "instruction_id":
                continue
            expected_type = type_map.get(value)
            if expected_type:
                args[key] = expected_type
        
        if args:
            schema[iid] = args
    return schema


def validate_instruction_kwargs_datatype(dict_turn_metadata):
    # Dynamically generate the validation rules from the imported template
    validation_schema = generate_schema_from_template(template_json)
    
    turn, issues = 1, []
    for data in dict_turn_metadata:
        errors = []
        instructions = data.get("instructions", [])

        for inst in instructions:
            iid = inst.get("instruction_id")
            if not iid or iid not in validation_schema:
                continue

            expected_args = validation_schema[iid]
            
            for arg_name, expected_type in expected_args.items():
                validator_func = TYPE_VALIDATORS.get(expected_type)
                # Check if the validator exists and if the argument value is valid
                if not validator_func or not validator_func(inst.get(arg_name)):
                    errors.append(f"{iid}: Argument '{arg_name}' must be a valid {expected_type}.")

        if errors:
            issues.append({f'TURN {turn}': errors})
        turn += 1
        
    return issues

def analyze_instruction_statuses_by_turn(data):
    results_per_turn, frontier_fail_rates = [], []
    task_fail, nova_fail, resp = False, None, []

    for item in data:
        turn_index = item.get('turn_index')
        response_type = item.get('response_type')
        results = item.get('results', [])

        passed = sum(r.get('status') == 'Passed' for r in results)
        failed = sum(r.get('status') == 'Failed' for r in results)
        total = passed + failed

        results_per_turn.append({
            'turn_index': turn_index,
            'response_type': response_type,
            'total': total,
            'passed': passed,
            'failed': failed
        })

        if response_type == 'response' and failed > 0:
            resp.append(f'❗FINAL TURN FAILING ON {failed} INSTRUCTIONS ❗')
            task_fail = True

        classification = 'Failure for each frontier model should be >= 50%'
        min_fail = 100
        if total > 0 and response_type != 'response' and response_type != 'prompt_validation':
            fail_rate = (failed * 100 / total)
            min_fail = min(min_fail, fail_rate)
            # if response_type == 'nova_response':
            #     nova_fail = fail_rate
            # elif response_type.endswith('_response'):
            #     frontier_fail_rates.append(fail_rate)
            if fail_rate < 50:
                task_fail = True
            elif fail_rate > 50:
                classification = 'EXPERT'
            elif classification != 'EXPERT':
                classification = 'HARD'

    # frontier_fail = round(sum(frontier_fail_rates) / len(frontier_fail_rates)) if frontier_fail_rates else 0

    # Classification logic

    # if frontier_fail >= 50:
    #     if frontier_fail > 50:
    #         classification = 'EXPERT'
    #     else:
    #         classification = 'HARD'
    # else:
    #     task_fail = True

    resp.append(f"Minimum Frontier Fail: {round(min_fail, 2)}%")
    result = {'task_fail': task_fail, 'text': resp, 'results_per_turn': results_per_turn, 'classification': classification}
    return result
