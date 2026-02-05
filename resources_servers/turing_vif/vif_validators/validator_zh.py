import json
import re
from typing import Any, Dict, List, Tuple

import validators.validator as base_validator
from validators.validator import LLM_INSTRUCTIONS, _normalize_casefold, find_punctuations

# Only expose the instructions requested for Chinese; all others are omitted.
EXPECTED_ARGUMENTS: Dict[str, List[str]] = {
    # --- PRIORITY (must be first, before any other instruction entries; do not repeat later) ---
    # "length_constraints:number_characters": ["relation", "num_chars"],
    # "length_constraints:number_words": ["relation", "num_words"],
    # "keywords:letter_frequency": ["letter", "let_relation", "let_frequency"],
    # "detectable_format:max_paragraph_length": ["max_chars"],
    # "length_constraints:sentence_length": ["max_words"],
    # "punctuation:question_exclaim": ["relation", "num_marks"],
    # "length_constraints:word_length": ["min_length", "max_length"],
    # "length_constraints:avg_word_length": ["min_ratio", "max_ratio"],
    # "detectable_content:numeric_inclusion": ["relation", "num_numbers"],

    # --- NON-PRIORITY (original order, with priority keys NOT repeated) ---
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
    "startend:start_checker": ["start_phrase"],
    "startend:wrap_checker": ["wrap_phrase"],
    "startend:end_checker": ["end_phrase"],
    "startend:quotation": [],
    "detectable_format:number_paragraphs": ["relation", "num_paragraphs"],
    "detectable_format:sentences_per_paragraph": ["relation", "num_sentences"],
    "length_constraints:word_repetition": ["max_repeats"],
    "length_constraints:unique_words": ["relation", "num_unique"],
    "punctuation:no_period": [],
    "punctuation:end_rule": ["allowed"],
    "detectable_format:nested_list": ["min_depth", "num_subitems"],
    "detectable_format:table": ["min_rows", "min_cols"],
    "detectable_format:sentence_count": ["relation", "num_sentences"],
    "length_constraints:paragraph_length": ["relation", "words_per_paragraph"],
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


CHINESE_COMMA_SET = {",", "，", "、"}
CHINESE_PERIOD_SET = {".", "。", "．"}
QUESTION_EXCLAIM_PATTERN = re.compile(r"[?？!！]")


def _relation_ok(count: int, relation: str, target: int) -> bool:
    if relation in {"equal to", "=="}:
        return count == target
    if relation in {"at least", ">="}:
        return count >= target
    if relation in {"less than", "<"}:
        return count < target
    raise ValueError("Invalid relation. Use 'equal to', 'at least', or 'less than'.")


def _keyword_count(text: str, keyword: str) -> int:
    kw = (keyword or "").strip()
    if not kw:
        return 0
    normalized_text = _normalize_casefold(text, "zh")
    normalized_kw = _normalize_casefold(kw, "zh")
    return normalized_text.count(normalized_kw)


def _count_placeholders(text: str) -> int:
    """
    Counts placeholder tokens of the form {TOKEN} or [TOKEN].
    """
    curly = re.findall(r"\{[^{}]+\}", text)
    square = re.findall(r"\[[^\[\]]+\]", text)
    return len(curly) + len(square)


def validate_instruction(response: str, inst_type: str, kwargs: Dict[str, Any], all_instructions: Dict = None) -> Tuple[bool, str]:
    if inst_type not in EXPECTED_ARGUMENTS:
        return False, f"Instruction '{inst_type}' is not supported for language 'zh'."

    response = response.strip()

    if inst_type == "detectable_format:json_format":
        return base_validator.validate_instruction(response, inst_type, kwargs, all_instructions, language="zh")

    if inst_type == "detectable_content:number_placeholders":
        count = _count_placeholders(response)
        rel, val = kwargs["relation"], kwargs["num_placeholders"]
        valid = _relation_ok(count, rel, val)
        return valid, ("No error" if valid else f"Expected {rel} {val} placeholders, found {count}.")

    if inst_type == "detectable_format:multiple_sections":
        splitter = (kwargs.get("section_splitter") or "").strip()
        rel = kwargs.get("relation")
        val = kwargs.get("num_sections")

        if not splitter:
            return False, "section_splitter cannot be empty."

        header_re = re.compile(
            rf"^\s*(?:#+\s*)?{re.escape(splitter)}\s+\d+\b",
            re.MULTILINE,
        )
        sections = header_re.findall(response)
        count = len(sections)

        if count == 0 and splitter == "---":
            hr_matches = re.findall(r"^\s*---+\s*$", response, re.MULTILINE)
            count = len(hr_matches)

        if count == 0 and splitter.startswith("<<") and splitter.endswith(">>"):
            generic_marks = re.findall(r"^\s*<<[^>]+>>\s*$", response, re.MULTILINE)
            count = len(generic_marks)

        if count == 0:
            parts = [p for p in re.split(r"\n\s*\n", response) if p.strip()]
            if splitter and len(parts) > 1 and any(splitter in p for p in parts):
                count = len(parts)

        if rel in ("at least", ">="):
            valid = count >= val
        elif rel in ("equal to", "==", "equals"):
            valid = count == val
        elif rel in ("less than", "<"):
            valid = count < val
        else:
            valid = count == val

        return valid, ("No error" if valid else f"Expected {rel} {val} sections, found {count}.")

    if inst_type == "detectable_format:number_bullet_lists":
        bullet_pattern = re.compile(r"^[ \t]*[*\-+]\s+", re.MULTILINE)
        numbered_pattern = re.compile(r"^[ \t]*\d+[.)]\s+", re.MULTILINE)
        count = len(bullet_pattern.findall(response)) + len(numbered_pattern.findall(response))
        rel, val = kwargs["relation"], kwargs["num_bullets"]
        valid = _relation_ok(count, rel, val)
        return valid, ("No error" if valid else f"Expected {rel} {val} bullet points, found {count}.")

    if inst_type == "detectable_format:title":
        first_line = response.splitlines()[0].strip() if response.splitlines() else ""
        is_marker = first_line.startswith("<<") and first_line.endswith(">>")
        is_heading = first_line.startswith("#")
        is_standalone = bool(first_line) and (len(response.splitlines()) <= 2)
        if is_marker or is_heading or is_standalone:
            return True, "No error"
        return False, "Title not detected on first line (accepts << >>, Markdown heading, or first-line title)."

    if inst_type == "keywords:existence":
        missing = [kw for kw in kwargs["keywords"] if _keyword_count(response, kw) == 0]
        return (not missing, "No error" if not missing else f"Missing keyword(s): {missing}")

    if inst_type == "keywords:frequency":
        count = _keyword_count(response, kwargs["keyword"])
        relation, target = kwargs["relation"], kwargs["frequency"]
        valid = _relation_ok(count, relation, target)
        return valid, ("No error" if valid else f"Expected {relation} {target} of '{kwargs['keyword']}', found {count}.")

    if inst_type == "keywords:forbidden_words":
        present = [kw for kw in kwargs["forbidden_words"] if _keyword_count(response, kw) > 0]
        return (not present, "No error" if not present else f"Forbidden words found: {present}")

    if inst_type == "keywords:letter_frequency":
        letter = kwargs["letter"]
        count = _keyword_count(response, letter)
        relation, target = kwargs["let_relation"], kwargs["let_frequency"]
        valid = _relation_ok(count, relation, target)
        return valid, ("No error" if valid else f"Expected {relation} {target} '{letter}', found {count}.")

    if inst_type == "punctuation:no_comma":
        has_comma = any(p in response for p in CHINESE_COMMA_SET)
        return (not has_comma, "No error" if not has_comma else "Commas found in response.")

    if inst_type == "punctuation:no_period":
        has_period = any(p in response for p in CHINESE_PERIOD_SET)
        return (not has_period, "No error" if not has_period else "Periods found in response.")

    if inst_type == "punctuation:question_exclaim":
        count = len(QUESTION_EXCLAIM_PATTERN.findall(response))
        relation, target = kwargs["relation"], kwargs["num_marks"]
        valid = _relation_ok(count, relation, target)
        return valid, ("No error" if valid else f"Found {count} marks. Expected {relation} {target}.")

    if inst_type == "punctuation:frequency":
        punct = kwargs.get("punctuation", "")
        relation, target = kwargs.get("relation", "at least"), kwargs.get("frequency", 0)
        count = response.count(punct)
        valid = _relation_ok(count, relation, target)
        return valid, ("No error" if valid else f"Found {count} occurrences of '{punct}'. Expected {relation} {target}.")

    if inst_type == "punctuation:end_rule":
        allowed = kwargs["allowed"]
        stripped = response.rstrip()
        if not stripped:
            return False, "Empty response."
        m = re.search(r"[。．.!?！？?!；;：:，,]$", stripped)
        if not m:
            return False, "Required ending punctuation not found."
        last = m.group(0)
        if last not in allowed:
            return False, f"Ending punctuation '{last}' not allowed. Allowed: {allowed}"
        return True, "No error."

    if inst_type == "startend:wrap_checker":
        wrap = kwargs["wrap_phrase"]
        WRAP_PAIRS = [
            ("《", "》"), ("〈", "〉"), ("「", "」"), ("『", "』"),
            ("【", "】"), ("[", "]"), ("(", ")"), ("（", "）"),
            ("{", "}"), ("<", ">"),
        ]
        if len(wrap) == 1:
            return (response.startswith(wrap) and response.endswith(wrap),
                    "No error" if response.startswith(wrap) and response.endswith(wrap) else f"Not wrapped with: {wrap}")
        if len(wrap) == 2:
            left, right = wrap[0], wrap[1]
            if response.startswith(left) and response.endswith(right):
                return True, "No error"
        for l, r in WRAP_PAIRS:
            if response.startswith(l) and response.endswith(r):
                return True, "No error"
        return False, f"Not wrapped with: {wrap}"

    if inst_type == "startend:end_checker":
        required = kwargs["end_phrase"].strip()
        if not required:
            return False, "End phrase is empty."
        normalized = response.rstrip()
        if normalized.endswith(required):
            return True, "No error"
        stripped = re.sub(r"[。．.?!！？…\s]+$", "", normalized)
        if stripped.endswith(required):
            return True, "No error"
        return False, f"End phrase mismatch: expected to end with '{required}'."

    if inst_type == "length_constraints:sentence_length":
        max_words = kwargs["max_words"]
        sentences = re.split(r"[。．.!?！？]+", response)
        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue
            char_count = len(re.sub(r"\s+", "", s))
            if char_count > max_words:
                return False, f"Sentence length {char_count} exceeds max {max_words}."
        return True, "No error."

    if inst_type == "startend:quotation":
        quote_pairs = [('"', '"'), ("“", "”"), ("「", "」"), ("『", "』"), ("《", "》"), ("〈", "〉")]
        for start, end in quote_pairs:
            if response.startswith(start) and response.endswith(end):
                return True, "No error"
        return False, "Response not wrapped in accepted quotation marks."

    return base_validator.validate_instruction(response, inst_type, kwargs, all_instructions, language="zh")


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
        response, inst_type, kwargs, all_instructions, definition_cache, language="zh"
    )


def validate_prompt_against_instructions(user_prompt: str, turn_instructions: Dict) -> Tuple[bool, str]:
    return base_validator.validate_prompt_against_instructions(
        user_prompt, turn_instructions, language="zh"
    )

def validate_thinking_against_instructions(thinking: str, turn_instructions: Dict) -> Tuple[bool, str]:
    return base_validator.validate_thinking_against_instructions(
        thinking, turn_instructions, language="zh"
    )

# Re-export generic helpers used elsewhere
validate_instruction_schema = base_validator.validate_instruction_schema
analyze_instruction_statuses_by_turn = base_validator.analyze_instruction_statuses_by_turn

