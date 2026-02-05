"""
VIF Validators adapted for NeMo Gym integration.
Includes both fast rule-based validators and async LLM judge validators.
"""

"""
VIF Validators adapted for NeMo Gym integration.
Includes both fast rule-based validators and async LLM judge validators.
"""

from fractions import Fraction
import re
import string
import json
import unicodedata
from typing import Dict, List, Literal, Tuple, Any, Optional

from collections import Counter, defaultdict
from pydantic import BaseModel, ValidationError, Field

from .data_loader import (
    JUDGE_SYSTEM_PROMPT,
    DEFINITION_GENERATOR_SYSTEM_PROMPT,
    LLM_JUDGE_QUESTION_PROMPT,
    LLM_INSTRUCTIONS,
    EXPECTED_ARGUMENTS,
    eval_modes,
    inst_def,
    subinst_def,
)


class JudgeResponse(BaseModel):
    """Defines the expected JSON structure for the LLM Judge's response."""
    verdict: Literal["YES", "NO"] = Field(..., description="The binary decision from the judge.")
    reasoning: str = Field(..., description="The explanation for the decision.")


class DefinitionResponse(BaseModel):
    """Defines the expected JSON structure for the definition generator's response."""
    status: Literal["PASS", "FAIL"] = Field(..., description="The binary decision from the generator.")
    definition: str = Field(..., description="The definition of the term.")


# ============================================================================
# Language Strategy Classes for Multi-Language Support
# ============================================================================

class LanguageStrategy:
    """Base strategy for language-specific text handling."""
    code = "en"
    has_case = True
    supports_case_rules = True
    supports_vowel_rules = True
    sentence_delims = ".!?"
    punctuation_marks = ".!?"
    vowels = set("aeiouAEIOU")
    word_script = "latin"  # latin | cjk

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def casefold(self, text: str) -> str:
        return self.normalize(text).casefold()

    def tokenize_words(self, text: str) -> List[str]:
        text_without_lists = re.sub(r'^\s*\d+\.\s', '', text, flags=re.MULTILINE)
        return re.findall(r"[^\W_]+(?:['-][^\W_]+)*", text_without_lists, flags=re.UNICODE)

    def sentence_split(self, text: str) -> List[str]:
        return re.split(f"[{re.escape(self.sentence_delims)}]+", text)

    def punctuation_list(self, text: str) -> List[str]:
        cleaned_text = re.sub(r'^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)', '', text, flags=re.MULTILINE)
        return re.findall(f"[{re.escape(self.punctuation_marks)}]+", cleaned_text)


class EnglishStrategy(LanguageStrategy):
    code = "en"


class FrenchStrategy(LanguageStrategy):
    code = "fre"
    vowels = set("aàâäeéèêëiîïoôöuùûüAEÉÈÊËÀÂÄÎÏÔÖÙÛÜ")


class SpanishStrategy(LanguageStrategy):
    code = "es"
    sentence_delims = ".!¿?"
    punctuation_marks = ".!¿?"
    vowels = set("aáeéiíoóuúüAÁEÉIÍOÓUÚÜ")


class ItalianStrategy(LanguageStrategy):
    code = "it"
    vowels = set("aàeèéiìoòuùAÀEÈÉIÌOÒUÙ")
    
    # Common Italian abbreviations that end with period but don't end sentences
    ABBREVIATIONS = {'sig.', 'dott.', 'prof.', 'avv.', 'pag.', 'fig.', 'tab.', 'ecc.', 'gen.', 
                     'sig', 'dott', 'prof', 'avv', 'pag', 'fig', 'tab', 'ecc', 'gen',
                     'Sig.', 'Dott.', 'Prof.', 'Avv.', 'Pag.', 'Fig.', 'Tab.', 'Ecc.', 'Gen.',
                     'SIG.', 'DOTT.', 'PROF.', 'AVV.', 'PAG.', 'FIG.', 'TAB.', 'ECC.', 'GEN.'}
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize Italian text, treating apostrophes as word separators for elisions.
        Example: "L'amico" -> ["L", "amico"]
        """
        text_without_lists = re.sub(r'^\s*\d+\.\s', '', text, flags=re.MULTILINE)
        words = re.split(r"[\s']+", text_without_lists)
        return [w for w in words if w and re.search(r'[a-zA-ZÀ-ÿ]', w)]
    
    def sentence_split(self, text: str) -> List[str]:
        """
        Split Italian text into sentences, avoiding splits at abbreviations.
        """
        protected_text = text
        abbr_map = {}
        for i, abbr in enumerate(self.ABBREVIATIONS):
            placeholder = f"__ABBR_{i}__"
            protected_text = protected_text.replace(abbr, placeholder)
            abbr_map[placeholder] = abbr
        
        delims_escaped = re.escape(self.sentence_delims)
        pattern = rf"([{delims_escaped}]+)\s*(?=[A-ZÀÈÉÌÒÙ]|$)"
        
        matches = list(re.finditer(pattern, protected_text))
        if not matches:
            restored = protected_text
            for placeholder, abbr in abbr_map.items():
                restored = restored.replace(placeholder, abbr)
            return [restored.strip()] if restored.strip() else [text]
        
        sentences = []
        last_pos = 0
        for match in matches:
            if match.start() > last_pos:
                sentences.append(protected_text[last_pos:match.start()].strip())
            end_pos = match.end()
            if end_pos < len(protected_text):
                next_char = protected_text[end_pos]
                if next_char.isupper() or next_char in 'ÀÈÉÌÒÙ':
                    last_pos = end_pos
                else:
                    last_pos = match.end()
            else:
                last_pos = match.end()
        
        if last_pos < len(protected_text):
            sentences.append(protected_text[last_pos:].strip())
        
        restored_sentences = []
        for sent in sentences:
            for placeholder, abbr in abbr_map.items():
                sent = sent.replace(placeholder, abbr)
            if sent.strip():
                restored_sentences.append(sent.strip())
        
        return restored_sentences if restored_sentences else [text]


class GermanStrategy(LanguageStrategy):
    code = "de"
    vowels = set("aäeëiïoöuüyAÄEËIÏOÖUÜY")


class PortugueseBRStrategy(LanguageStrategy):
    code = "pt-BR"
    vowels = set("aáàâãeéêiíoóõôuúüAÁÀÂÃEÉÊIÍOÓÕÔUÚÜ")


class CJKStrategy(LanguageStrategy):
    """Base strategy for CJK languages (Chinese, Japanese, Korean)."""
    has_case = False
    supports_case_rules = False
    supports_vowel_rules = False
    word_script = "cjk"

    def tokenize_words(self, text: str) -> List[str]:
        text = self.normalize(text)
        return [ch for ch in text if ch.strip()]

    def sentence_split(self, text: str) -> List[str]:
        return re.split(f"[{re.escape(self.sentence_delims)}]+", text)


class JapaneseStrategy(CJKStrategy):
    code = "ja"
    sentence_delims = "。！？.!?"
    punctuation_marks = "。！？.!?"


class ChineseStrategy(CJKStrategy):
    code = "zh"
    sentence_delims = "。！？.!?"
    punctuation_marks = "。！？.!?"


class KoreanStrategy(CJKStrategy):
    code = "ko"
    sentence_delims = "?!？.!"
    punctuation_marks = "?!？.!"


class HindiStrategy(LanguageStrategy):
    code = "hi"
    has_case = False
    supports_case_rules = False
    supports_vowel_rules = False
    sentence_delims = "।.?!"
    punctuation_marks = "।.?!"

    def tokenize_words(self, text: str) -> List[str]:
        text_without_lists = re.sub(r'^\s*\d+\.\s', '', text, flags=re.MULTILINE)
        words = re.findall(r"[\u0900-\u097F]+", text_without_lists)
        if not words:
            words = re.findall(r"[^\W_]+", text_without_lists, flags=re.UNICODE)
        return words


class ArabicStrategy(LanguageStrategy):
    code = "ar"
    has_case = False
    supports_case_rules = False
    supports_vowel_rules = False
    sentence_delims = "؟!.?"
    punctuation_marks = "؟!.?"

    def normalize(self, text: str) -> str:
        text = super().normalize(text)
        return re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)


# Language strategy registry
LANG_STRATEGIES: Dict[str, LanguageStrategy] = {
    cls.code: cls() for cls in [
        EnglishStrategy,
        FrenchStrategy,
        SpanishStrategy,
        ItalianStrategy,
        GermanStrategy,
        PortugueseBRStrategy,
        JapaneseStrategy,
        ChineseStrategy,
        KoreanStrategy,
        HindiStrategy,
        ArabicStrategy
    ]
}

SUPPORTED_LANGS = set(LANG_STRATEGIES.keys())


def _get_strategy(language: str) -> LanguageStrategy:
    """Get the language strategy for a given language code."""
    return LANG_STRATEGIES.get(language, LANG_STRATEGIES["en"])


def _supports_case(language: str) -> bool:
    """Check if a language supports case-based rules."""
    return _get_strategy(language).supports_case_rules


def _get_vowels(language: str) -> set:
    """Get the vowel set for a language."""
    return _get_strategy(language).vowels


# ============================================================================
# Instruction Capability Sets for Multi-Language Support
# ============================================================================

# Instructions that require case support (not available for CJK, Hindi, Arabic)
CASE_INSTRUCTIONS = {
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
    "keywords:letter_frequency",
    "keywords:alliteration",
}

# Instructions that require vowel/consonant distinction (not available for CJK, Hindi, Arabic)
VOWEL_INSTRUCTIONS = {
    "change_case:vowel_consonant_balance",
    "keywords:vowel_count",
    "keywords:consonant_count",
    "keywords:alliteration",
}


def is_instruction_supported(inst_type: str, language: str) -> bool:
    """
    Check if an instruction type is supported for a given language.
    
    Args:
        inst_type: The instruction type ID (e.g., "change_case:all_caps")
        language: The language code (e.g., "en", "ja", "hi")
        
    Returns:
        True if the instruction is supported for the language, False otherwise.
    """
    # English supports all instructions
    if language == "en":
        return True
    
    strategy = _get_strategy(language)
    
    # Case-sensitive instructions are not supported for languages without case
    if not strategy.supports_case_rules and inst_type in CASE_INSTRUCTIONS:
        return False
    
    # Vowel/consonant instructions are not supported for languages without vowel rules
    if not strategy.supports_vowel_rules and inst_type in VOWEL_INSTRUCTIONS:
        return False
    
    return True


def get_supported_instructions(language: str) -> List[str]:
    """
    Get list of instruction IDs supported for a given language.
    
    Args:
        language: The language code
        
    Returns:
        List of supported instruction IDs for the language.
    """
    return [iid for iid in EXPECTED_ARGUMENTS.keys() if is_instruction_supported(iid, language)]


# ============================================================================
# Helper Functions for Text Processing
# ============================================================================

def check_relation(count, relation: str, expected: int ) -> Tuple[bool, Optional[str]]:
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


def keyword_frequency(response: str, keyword: str, language: str = "en") -> int:
    """Count frequency of a keyword in response, using Unicode-aware boundaries."""
    strategy = _get_strategy(language)
    keyword = strategy.casefold(keyword.strip())

    # For CJK scripts, fall back to substring counting with non-letter/digit guards
    if strategy.word_script == "cjk":
        pattern = rf"(?<![\w]){re.escape(keyword)}(?![\w])"
        return len(re.findall(pattern, strategy.casefold(response), flags=re.UNICODE))

    # For Latin scripts, use word boundary matching
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
    """Parses a string into a Fraction object or float('inf')."""
    if isinstance(input_str, (int, float)):
        return input_str
        
    if not isinstance(input_str, str):
        raise TypeError(f"Input must be a string, not {type(input_str)}")

    input_str = input_str.strip().lower()
    if input_str == 'inf':
        return float('inf')
    
    try:
        frac = Fraction(input_str)
        return frac
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Invalid input: '{input_str}'. Not a valid fraction or 'inf'.")


def extract_clean_sentences(text: str, language: str = "en") -> List[str]:
    """Takes a raw text string and returns a clean list of sentences."""
    strategy = _get_strategy(language)
    delims = strategy.sentence_delims

    # Remove markdown tables
    table_pattern = r'(?:^\s*\|.*\|.*\n){2,}'
    cleaned_text = re.sub(table_pattern, '', text, flags=re.MULTILINE)

    # Remove horizontal rules
    rule_pattern = r'^\s*([*_-])\s*\1\s*\1+\s*$'
    text = re.sub(rule_pattern, '', cleaned_text, flags=re.MULTILINE)
    
    all_sentences = []
    for line in text.split('\n'):
        line = line.lstrip()
        cleaned_line = re.sub(r'^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)', '', line)
        if not cleaned_line:
            continue
        # Split by language-specific delimiters
        line_parts = re.split(f"[{re.escape(delims)}]+", cleaned_line)
        for sentence in line_parts:
            stripped_sentence = sentence.strip()
            if stripped_sentence:
                all_sentences.append(stripped_sentence)
    return all_sentences


def extract_clean_words(response: str, language: str = "en") -> List[str]:
    """Extract clean words from text using language-specific tokenization."""
    strategy = _get_strategy(language)
    text_without_lists = re.sub(r'^\s*\d+\.\s', '', response, flags=re.MULTILINE)
    return strategy.tokenize_words(text_without_lists)


def analyze_lists(text: str, pattern: str) -> list:
    """Analyzes a text to find lists based on a provided regex pattern."""
    lists_found = []
    current_list_stack = []
    item_pattern = re.compile(pattern, re.MULTILINE)
    all_items = item_pattern.finditer(text)

    for item in all_items:
        indentation, marker, item_text = item.groups()
        indent_level = len(indentation.strip('\n'))

        while current_list_stack and indent_level < current_list_stack[-1]['indent']:
            lists_found.append(current_list_stack.pop())

        if not current_list_stack or indent_level == current_list_stack[-1]['indent']:
            if not current_list_stack:
                nesting_level = 1
                current_list_stack.append({'level': nesting_level, 'indent': indent_level, 'items': 1})
            else:
                current_list_stack[-1]['items'] += 1
        elif indent_level > current_list_stack[-1]['indent']:
            nesting_level = current_list_stack[-1]['level'] + 1
            current_list_stack.append({'level': nesting_level, 'indent': indent_level, 'items': 1})

    lists_found.extend(current_list_stack)
    return lists_found


def find_markdown_tables(text: str) -> list:
    """Finds all markdown tables in a text and determines their dimensions."""
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
        
        tables_found.append({'rows': num_rows, 'columns': num_cols})
        i = j
    
    return tables_found


def find_punctuations(text: str, language: str = "en") -> List[str]:
    """Find punctuation marks using language-specific patterns."""
    strategy = _get_strategy(language)
    cleaned_text = re.sub(r'^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)', '', text, flags=re.MULTILINE)
    return strategy.punctuation_list(cleaned_text)


def extract_clean_paragraphs(text: str) -> List[str]:
    """Extracts clean paragraphs from a text by removing markdown elements."""
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


# ============================================================================
# Fast Rule-Based Validators (Synchronous)
# ============================================================================

def validate_instruction(response: str, inst_type: str, kwargs: Dict[str, Any], all_instructions: Dict = None, language: str = "en") -> Tuple[bool, str]:
    """
    Validate a response against a specific instruction type and its kwargs.
    
    Args:
        response: The text response to validate
        inst_type: The instruction type ID (e.g., "change_case:all_caps")
        kwargs: Dictionary of instruction arguments
        all_instructions: Optional dictionary of all instructions for context
        language: Language code for multi-language support (default: "en")
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        response = response.strip()
        strategy = _get_strategy(language)
        
        # Note: Language compatibility is pre-validated at the app level.
        # Unsupported instructions should never reach here as the rollout is skipped.
        
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
            valid, err = check_relation(count, rel, val)
            if err:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} all-cap words, found {count}.")

        if inst_type == "change_case:lowercase_word_frequency":
            count = count_lowercase_words(response)
            rel, val = kwargs['lowercase_relation'], kwargs['lowercase_frequency']
            valid, err = check_relation(count, rel, val)
            if err:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} lowercase words, found {count}.")

        if "_target" in inst_type:
            target = kwargs["target_string"].strip().lower()
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
            valid, err = check_relation(count, rel, val)
            if err:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} placeholders, found {count}.")

        if inst_type == "detectable_content:postscript":
            marker = kwargs.get("postscript_marker", "PS:").strip()
            lines = response.splitlines()
            last_line = ""
            for line in reversed(lines):
                if line.strip():
                    last_line = line.strip()
                    break

            has_postscript = last_line.startswith(marker) and len(last_line) > len(marker)
            return (has_postscript, "No error" if has_postscript else f"Postscript must start with '{marker}' and contain content.")

        if inst_type == "detectable_format:json_format":
            try:
                json_part = response[response.find("{"):response.rfind("}")+1]
                json.loads(json_part)
                return (True, "No error")
            except Exception:
                return (False, "Response is not valid JSON format.")

        if inst_type == "detectable_format:multiple_sections":
            splitter = (kwargs.get("section_splitter") or "").strip()
            rel = kwargs.get("relation")
            val = kwargs.get("num_sections")

            if not splitter:
                return (False, "section_splitter cannot be empty.")
            if re.search(r"[#*]", splitter):
                return (False, "section_splitter must be a plain section name without '#' or '*'.")

            header_re = re.compile(
                r"^\s*#{1,6}\s+" + re.escape(splitter) + r"\s+\d+\b",
                re.MULTILINE | re.IGNORECASE,
            )
            sections = header_re.findall(response)
            count = len(sections)

            valid, err = check_relation(count, rel, val)
            if err:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} sections, found {count}.")

        if inst_type == "detectable_format:numbered_list":
            count = count_numbered_items(response)
            rel, val = kwargs["relation"], kwargs["num_numbered_items"]
            valid, err = check_relation(count, rel, val)
            if err:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} numbered items, found {count}.")

        if inst_type == "detectable_format:number_bullet_lists":
            count = count_bullet_points(response)
            rel, val = kwargs["relation"], kwargs["num_bullets"]
            valid, err = check_relation(count, rel, val)
            if err:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} bullet points, found {count}.")

        if inst_type == "detectable_format:title":
            line = response.splitlines()[0]
            found_title = line.strip().startswith("<<") and line.strip().endswith(">>")
            return (found_title, "No error" if found_title else "Title not wrapped in << >> on first line.")

        if inst_type == "keywords:existence":
            missing = [kw for kw in kwargs["keywords"] if keyword_frequency(response, kw, language) == 0]
            return (not missing, "No error" if not missing else f"Missing keyword(s): {missing}")

        if inst_type == "keywords:frequency":
            keyword = strategy.casefold(kwargs["keyword"].strip())
            count = keyword_frequency(response, keyword, language)
            rel = kwargs["relation"]
            val = kwargs["frequency"]
            valid, err = check_relation(count, rel, val)
            if err:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} of '{keyword}', found {count}.")

        if inst_type == "keywords:forbidden_words":
            present = [w for w in kwargs["forbidden_words"] if keyword_frequency(response, w, language)]
            return (not present, "No error" if not present else f"Forbidden words found: {present}")

        if inst_type == "keywords:letter_frequency":
            letter = kwargs["letter"].lower()
            count = response.lower().count(letter)
            rel, val = kwargs["let_relation"], kwargs["let_frequency"]
            valid, err = check_relation(count, rel, val)
            if err:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} '{letter}', found {count}.")

        if inst_type == "punctuation:no_comma":
            return (',' not in response, "No error" if ',' not in response else "Commas found in response.")

        if inst_type == "length_constraints:number_characters":
            count = len(response)
            rel, val = kwargs["relation"], kwargs["num_chars"]
            valid, err = check_relation(count, rel, val)
            if err:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} characters, found {count}.")

        if inst_type == "length_constraints:number_words":
            words = extract_clean_words(response, language)
            count = len(words)
            rel, val = kwargs["relation"], kwargs["num_words"]
            valid, err = check_relation(count, rel, val)
            if err:
                return (False, err)
            return (valid, "No error" if valid else f"Expected {rel} {val} words, found {count}.")

        if inst_type == "startend:start_checker":
            starts_correctly = response.lstrip(string.punctuation + " ").lower().startswith(kwargs.get("start_phrase", "").lower())
            return (starts_correctly, "No error" if starts_correctly else "Response does not start with required phrase.")

        if inst_type == "startend:end_checker":
            required = kwargs["end_phrase"].strip()
            ends_with_punctuation = required[-1] in string.punctuation if required else False
            actual_words = response.lstrip(string.punctuation).strip().split()
            
            if not actual_words:
                return (False, "Empty response")
                
            if ends_with_punctuation:
                actual_phrase = " ".join(actual_words[-len(required.split()):])
                if actual_phrase.lower() != required.lower():
                    return (False, f"End phrase mismatch: expected '{required}', but found '{actual_phrase}'")
            else:
                actual_phrase = " ".join(actual_words).rstrip(string.punctuation + " ")[-len(required):]
                if actual_phrase.lower() != required.lower():
                    return (False, f"End phrase mismatch: expected '{required}', but found '{actual_phrase}'")
            return (True, "No error")

        if inst_type == "startend:wrap_checker":
            wrap = kwargs["wrap_phrase"]
            return (response.startswith(wrap) and response.endswith(wrap),
                    "No error" if response.startswith(wrap) else f"Not wrapped with: {wrap}")

        if inst_type == "startend:quotation":
            return (response.startswith('"') and response.endswith('"'),
                    "No error" if response.startswith('"') else "Response not wrapped in double quotes.")

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
                return (False, "No letters found in the string.")

            if upper_count == 0:
                ratio = float('inf')
            else:
                ratio = Fraction(lower_count, upper_count)

            valid = minR <= ratio <= maxR
            return (valid, "No error" if valid else f"Case ratio out of range.")

        if inst_type == "change_case:first_letter_sentence":
            sentences = extract_clean_sentences(response, language)
            if not sentences:
                return (True, "No sentences found to validate.")

            for sentence in sentences:
                sentence = sentence.strip("()[]{}\"'")
                if not sentence[0].isupper():
                    return (False, f"Fails at: '{sentence}'")
            return (True, "No error.")

        if inst_type == "change_case:last_letter":
            cleaned_text = re.sub(r'[.!?]+$', '', response.strip())
            if not cleaned_text:
                return (False, "Empty response")

            words = cleaned_text.split()
            last_word = words[-1].strip("()[]{}\"'")
            if not last_word:
                return (False, "No valid last word")

            last_char = last_word[-1]
            valid = True
            case = kwargs["case"]
            
            if case == "uppercase":
                valid = last_char.isupper()
            elif case == "lowercase":
                valid = last_char.islower()
            elif case == "digit":
                valid = last_char.isdigit()
            elif case == "special":
                valid = not last_char.isalnum()
            
            return (valid, "No error." if valid else f"Last character: {last_char}")

        if inst_type == "change_case:vowel_consonant_balance":
            try:
                minR = parse_fraction_or_inf(kwargs["min_fraction"])
                maxR = parse_fraction_or_inf(kwargs["max_fraction"])
            except (ValueError, ZeroDivisionError) as e:
                raise ValueError(f"Invalid fraction input: {e}")
            
            if minR > maxR:
                return (False, "Minimum ratio greater than maximum ratio.")
            
            vowels = _get_vowels(language)
            vowel_count = sum(1 for ch in response if ch.isalpha() and ch in vowels)
            consonant_count = sum(1 for ch in response if ch.isalpha() and ch not in vowels)

            if vowel_count == 0 and consonant_count == 0:
                return (False, "No letters found in the response.")

            if consonant_count == 0:
                ratio = float('inf')
            else:
                ratio = Fraction(vowel_count, consonant_count)

            valid = minR <= ratio <= maxR
            return (valid, "No error" if valid else f"Vowel/consonant ratio out of range.")

        if inst_type == "detectable_format:number_paragraphs":
            paragraphs = extract_clean_paragraphs(response)
            actual_count = len([p for p in paragraphs if p.strip()])
            
            if not response.strip():
                actual_count = 0

            relation = kwargs["relation"]
            num_paragraphs = kwargs["num_paragraphs"]

            is_valid, err = check_relation(actual_count, relation, num_paragraphs)
            if err:
                return (False, err)
            return (is_valid, "No error." if is_valid else f"Found {actual_count} paragraphs, expected {num_paragraphs}")

        if inst_type == "detectable_format:max_paragraph_length":
            max_chars = kwargs["max_chars"]
            paragraphs = extract_clean_paragraphs(response)

            for p in paragraphs:
                p = re.sub(r'^\s*(?:[\-\*\+]\s+|\d+\.\s+|#+\s+)', '', p.lstrip())
                char_count = len(p.strip())
                if char_count > max_chars:
                    return (False, f"Found paragraph with {char_count} characters (max: {max_chars})")
            return (True, "No error.")

        if inst_type == "detectable_format:sentences_per_paragraph":
            num_sentences = kwargs["num_sentences"]
            relation = kwargs["relation"]
            paragraphs = extract_clean_paragraphs(response)

            for p in paragraphs:
                sentences = extract_clean_sentences(p, language)
                sentence_count = len([s for s in sentences if s.strip()])
                if sentence_count == 0 and p.strip():
                    sentence_count = 1

                is_valid, err = check_relation(sentence_count, relation, num_sentences)
                if err:
                    return (False, err)
                if not is_valid:
                    return (False, f"Found {sentence_count} sentences, expected {relation} {num_sentences}")
            return (True, "No error.")

        if inst_type == "length_constraints:sentence_length":
            sentences = extract_clean_sentences(response, language)
            max_words = kwargs["max_words"]
            
            if not sentences:
                return (True, "No sentences found to validate.")
            
            for s in sentences:
                word_count = len(strategy.tokenize_words(s))
                if word_count > max_words:
                    return (False, f"Found sentence with {word_count} words (max: {max_words})")
            return (True, "No error.")

        if inst_type == "length_constraints:word_repetition":
            max_repeats = kwargs["max_repeats"]
            words = extract_clean_words(response, language)
            word_counts = Counter([strategy.casefold(w) for w in words])

            for word, count in word_counts.items():
                if count > max_repeats:
                    return (False, f"Word '{word}' appears {count} times (limit: {max_repeats})")
            return (True, "No error.")

        if inst_type == "length_constraints:unique_words":
            relation = kwargs["relation"]
            num_unique = kwargs["num_unique"]
            words = extract_clean_words(response, language)
            unique_words_count = len(set(strategy.casefold(w) for w in words))

            is_valid, err = check_relation(unique_words_count, relation, num_unique)
            if err:
                return (False, err)
            return (is_valid, "No error" if is_valid else f"Found {unique_words_count} unique words, expected {relation} {num_unique}")

        if inst_type == "punctuation:question_exclaim":
            relation = kwargs["relation"]
            num_marks = kwargs["num_marks"]
            punctuations = re.findall(r"[?!]", response)
            count = len(punctuations)

            is_valid, err = check_relation(count, relation, num_marks)
            if err:
                return (False, err)
            return (is_valid, "No error" if is_valid else f"Found {count} marks, expected {relation} {num_marks}")

        if inst_type == "punctuation:no_period":
            return ('.' not in response, "No error" if '.' not in response else "Periods found in response.")

        if inst_type == "punctuation:end_rule":
            allowed = kwargs["allowed"]
            punctuations = set(find_punctuations(response, language))

            for p in punctuations:
                if p not in allowed:
                    return (False, f"'{p}' not in the list of allowed punctuations.")
            return (True, "No error.")

        if inst_type == "keywords:alliteration":
            relation = kwargs["relation"]
            num_alliteration = kwargs["num_alliteration"]
            target_letter = strategy.casefold(kwargs["target_letter"])
            
            words = extract_clean_words(response, language)
            all_count = sum(1 for word in words if strategy.casefold(word).startswith(target_letter))

            is_valid, err = check_relation(all_count, relation, num_alliteration)
            if err:
                return (False, err)
            return (is_valid, "No error" if is_valid else f"Found {all_count} alliteration words, expected {relation} {num_alliteration}")

        if inst_type == "keywords:palindrome_word":
            min_length = kwargs["min_length"]
            words = extract_clean_words(response, language)
            for word in words:
                normalized_word = strategy.casefold(word)
                if normalized_word == normalized_word[::-1] and len(word) >= min_length:
                    return (True, f"No error. Word: {word}")
            return (False, "No valid palindrome words found.")

        if inst_type == "keywords:positioning":
            keyword = strategy.casefold(kwargs["keyword"])
            position = kwargs["position"]
            words = extract_clean_words(response, language)
            
            if position >= len(words):
                return (False, f"Position {position} is out of range (only {len(words)} words found).")
            
            if strategy.casefold(words[position]) == keyword:
                return (True, "No error.")
            return (False, f"'{words[position]}' found at position {position} instead of '{kwargs['keyword']}'.")

        if inst_type == "detectable_format:nested_list":
            min_depth = kwargs["min_depth"]
            num_subitems = kwargs["num_subitems"]
            
            bullet_pattern = r"^(\s*)([*+-])[ \t]+(.*)"
            numbered_pattern = r"^(\s*)(\d+\.)[ \t]+(.*)"
            
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
            return (False, f"Could not find table with at least {min_rows} rows and {min_cols} columns.")

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
                return (False, "Minimum length greater than maximum length.")

            words = set(extract_clean_words(response, language))
            if not words:
                return (True, "No words found to validate.")

            shortest_word = min(words, key=len)
            longest_word = max(words, key=len)

            if len(shortest_word) < min_length:
                return (False, f"Word '{shortest_word}' is shorter than minimum {min_length}.")
            if len(longest_word) > max_length:
                return (False, f"Word '{longest_word}' is longer than maximum {max_length}.")
            return (True, "No error.")

        if inst_type == "length_constraints:avg_word_length":
            min_ratio = kwargs["min_ratio"]
            max_ratio = kwargs["max_ratio"]
            
            if min_ratio > max_ratio:
                return (False, "Minimum ratio greater than maximum ratio.")

            words = extract_clean_words(response, language)
            if not words:
                is_valid = min_ratio == 0
                return (is_valid, "No words found to validate.")
            
            avg_count = sum(len(word) for word in words) / len(words)
            is_valid = min_ratio <= avg_count <= max_ratio
            return (is_valid, "No error" if is_valid else f"Average word length {avg_count:.2f} not in range [{min_ratio}, {max_ratio}]")

        if inst_type == "detectable_format:sentence_count":
            relation = kwargs["relation"]
            num_sentences = kwargs["num_sentences"]
            sentence_count = len(extract_clean_sentences(response, language))

            is_valid, err = check_relation(sentence_count, relation, num_sentences)
            if err:
                return (False, err)
            return (is_valid, "No error" if is_valid else f"Found {sentence_count} sentences, expected {relation} {num_sentences}")

        if inst_type == "length_constraints:paragraph_length":
            words_per_paragraph = kwargs["words_per_paragraph"]
            relation = kwargs["relation"]
            paragraphs = extract_clean_paragraphs(response)

            for p in paragraphs:
                words = extract_clean_words(p, language)
                word_count = len([s for s in words if s.strip()])

                is_valid, err = check_relation(word_count, relation, words_per_paragraph)
                if err:
                    return (False, err)
                if not is_valid:
                    return (False, f"Found {word_count} words in paragraph, expected {relation} {words_per_paragraph}")
            return (True, "No error.")

        if inst_type == "detectable_content:numeric_inclusion":
            num_numbers = kwargs["num_numbers"]
            relation = kwargs["relation"]
            num_count = sum(1 for ch in response if ch.isdigit())

            is_valid, err = check_relation(num_count, relation, num_numbers)
            if err:
                return (False, err)
            return (is_valid, "No error" if is_valid else f"Found {num_count} digits, expected {relation} {num_numbers}")

        if inst_type == "detectable_format:sentence_endings":
            min_variants = kwargs["min_variants"]
            punctuations = set(find_punctuations(response, language))

            if len(punctuations) < min_variants:
                return (False, f"Found {len(punctuations)} types of punctuations, expected at least {min_variants}")
            return (True, "No error.")

        if inst_type == "keywords:vowel_count":
            num_vowels = kwargs["num_vowels"]
            relation = kwargs["relation"]
            
            vowels = _get_vowels(language)
            vowel_count = sum(1 for ch in response if ch in vowels)

            is_valid, err = check_relation(vowel_count, relation, num_vowels)
            if err:
                return (False, err)
            return (is_valid, "No error" if is_valid else f"Found {vowel_count} vowels, expected {relation} {num_vowels}")

        if inst_type == "keywords:consonant_count":
            num_consonants = kwargs["num_consonants"]
            relation = kwargs["relation"]
            
            vowels = _get_vowels(language)
            # For languages with case, use ascii letters minus vowels
            # For caseless languages, just count alphabetic chars that aren't vowels
            if strategy.has_case:
                consonants = set(string.ascii_letters) - vowels
                consonant_count = sum(1 for ch in response if ch in consonants)
            else:
                consonant_count = sum(1 for ch in response if ch.isalpha() and ch not in vowels)

            is_valid, err = check_relation(consonant_count, relation, num_consonants)
            if err:
                return (False, err)
            return (is_valid, "No error" if is_valid else f"Found {consonant_count} consonants, expected {relation} {num_consonants}")

        # Unsupported instructions
        if inst_type in ["detectable_format:indentation", "punctuation:frequency", "punctuation:balance", 
                         "detectable_format:section_balance", "punctuation:variety"]:
            return (False, f"Instruction '{inst_type}' not yet implemented.")

    except Exception as e:
        return (False, f"Validation error: {str(e)}")

    return (False, f"Unknown instruction: {inst_type}")


def validate_instruction_schema(instructions: Dict) -> List[Dict]:
    """Validate the schema of instructions against expected arguments."""
    mismatches = []
    
    metadata = instructions.get("metadata", [])
    if not isinstance(metadata, list):
        mismatches.append({"field": "metadata", "expected": "list", "actual": type(metadata).__name__})
    
    instructions_list = instructions.get("instructions", [])
    if not isinstance(instructions_list, list):
        mismatches.append({"field": "instructions", "expected": "list", "actual": type(instructions_list).__name__})
        return mismatches

    contradiction_errors = check_contradicting_instructions(instructions_list)
    mismatches.extend(contradiction_errors)

    for i, inst in enumerate(instructions_list):
        if not isinstance(inst, dict):
            mismatches.append({"instruction_index": i, "expected": "dict", "actual": type(inst).__name__})
            continue

        if "instruction_id" not in inst:
            mismatches.append({"instruction_index": i, "missing_field": "instruction_id"})
            continue

        expected_args = set(EXPECTED_ARGUMENTS.get(inst["instruction_id"], []))
        actual_args = set(k for k in inst.keys() if k != "instruction_id")

        if expected_args != actual_args:
            mismatches.append({
                "instruction": inst["instruction_id"],
                "expected_args": sorted(expected_args),
                "actual_args": sorted(actual_args)
            })

    return mismatches


def check_contradicting_instructions(instructions_list: List[Dict]) -> List[str]:
    """Check for contradicting instruction IDs in the list."""
    from .data_loader import conflict_dict
    
    errors = set()
    seen_pairs = set()
    ids = {inst["instruction_id"] for inst in instructions_list if isinstance(inst, dict) and "instruction_id" in inst}

    for instr_id in ids:
        for conflicting_id in conflict_dict.get(instr_id, []):
            pair = frozenset([instr_id, conflicting_id])
            if conflicting_id in ids and pair not in seen_pairs:
                errors.add(f"{instr_id} and {conflicting_id} are contradicting")
                seen_pairs.add(pair)
    return list(errors)

