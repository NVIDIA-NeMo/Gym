"""This module provides functions for parsing mathematical expressions from text."""

import re
from enum import Enum
from fractions import Fraction
from functools import total_ordering
from typing import Any, Optional
from sympy import simplify, Symbol, Mul

import regex
import sympy
from loguru import logger
from sympy import N
from sympy.parsing.latex import parse_latex

from matharena.parse_manual import complete_mapper, manual_mapper


APPROX_RE = re.compile(r"\\approx|≈|approximately|approx\.?|about", re.I)
EXACT_BOX_TOKENS = (r"\frac", r"\sqrt", r"\binom", "/", "^")
SYMPY_LOCALS = {
    "binomial": sympy.binomial,
    "floor": sympy.floor,
    "ceiling": sympy.ceiling,
    "ceil": sympy.ceiling,
    "pi": sympy.pi,
    "E": sympy.E,
    "e": sympy.E,
    "I": sympy.I,
}


@total_ordering
class WarningType(Enum):
    """An enumeration for warning levels."""

    NONE = 0
    MINOR = 1
    POSSIBLE = 2
    MAJOR = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return self.value < other


def latex2sympy_fixed(latex: str):
    # if _integer is present, replace it with _{integer} for any integer
    latex = re.sub(r"_([0-9]+)", r"_{\1}", latex)
    latex_parsed = parse_latex(latex)
    # replace constants like pi and e with their numerical value
    known_constants = {"pi": sympy.pi, "e": sympy.E, "I": 1j, "i": 1j}

    # Replace any symbol in expr that is in our known_constants dictionary.
    expr = latex_parsed.xreplace(
        {s: known_constants[s.name] for s in latex_parsed.free_symbols if s.name in known_constants}
    )
    return expr


def remove_inner_boxed(match: str):
    """Removes inner `\boxed` or `\fbox` commands from a string.

    Args:
        match (str): The string to process.

    Returns:
        str: The string with inner `\boxed` or `\fbox` commands removed.
    """
    pattern = r"(\\boxed|\\fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    matches = list(regex.finditer(pattern, match))
    if not matches:
        return match
    for m in matches:
        match = match.replace(m.group(0), m.group(2))
    return match


def find_last_boxed_content(text: str, list_answer: bool = False) -> Optional[str]:
    """Finds the content of the last `\boxed` or `\fbox` command in a string.

    Args:
        text (str): The string to search.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the content of the last `\boxed` or `\fbox` command
            and a warning level.
    """
    pattern = r"(boxed|fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    matches = list(regex.finditer(pattern, text))
    if not matches:
        return None, WarningType.NONE

    if len(matches) > 1 and list_answer:
        # find all boxed content on the same line (no \n in between) as the last boxed
        split_text = text.split("\n")
        for i in range(len(split_text) - 1, -1, -1):
            matches_line = list(regex.finditer(pattern, split_text[i]))
            if len(matches_line) > 0:
                # If a full list answer is boxed repeatedly on the final line,
                # joining all boxes duplicates it. Multiple scalar boxes are
                # still joined, e.g. \boxed{2}, \boxed{3}.
                if any("," in match.group(2) for match in matches_line):
                    returned_boxed = matches_line[-1].group(2)
                else:
                    returned_boxed = ",".join([match.group(2) for match in matches_line])
                return remove_inner_boxed(returned_boxed), WarningType.POSSIBLE

    selected_match = matches[-1]
    if len(matches) > 1 and should_prefer_previous_boxed_over_approximation(selected_match.group(2)):
        for match in reversed(matches[:-1]):
            if not contains_approximation(match.group(2)):
                selected_match = match
                break
    if len(matches) > 1 and is_decimal_approximation_box(selected_match.group(2)):
        for match in reversed(matches[:-1]):
            if looks_like_exact_boxed_expression(match.group(2)):
                selected_match = match
                break

    last_match = remove_inner_boxed(selected_match.group(2))
    return last_match, WarningType.NONE


def extract_boxed_answer(text: str, list_answer: bool = False) -> Optional[str]:
    """Extracts the content of the last `\boxed` or `\fbox` command in a string.

    Args:
        text (str): The string to search.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the content of the last `\boxed` or `\fbox` command
            and a warning level.
    """
    answer, warning = find_last_boxed_content(text, list_answer)
    if answer is not None:
        return answer, warning
    else:
        return None, warning


def replace_and_or(s: str) -> str:
    """Replaces 'and' or 'or' with commas in a string.

    1) If 'and/or' (or their \text{} forms) is NOT right next to a comma
       (ignoring spaces) -> replace it by a single ','.
    2) Otherwise (comma already on at least one side) -> delete it.

    Args:
        s (str): The string to process.

    Returns:
        str: The processed string.
    """
    TOKEN = re.compile(
        r"""
        (?:\\text\s*\{\s*)?      # optional '\text{' and any leading blanks
        (and|or)                 # the word itself
        (?:\s*\})?               # optional closing '}' with any blanks
        """,
        re.I | re.VERBOSE,
    )
    # We build a fresh output string piece-by-piece so that each check
    # uses the **current** comma layout, not the one from the original text.
    out, idx = [], 0
    for m in TOKEN.finditer(s):
        start, end = m.span()
        # copy text *before* the token
        out.append(s[idx:start])

        # look to the left of the token, skipping blanks
        j = start - 1
        while j >= 0 and s[j].isspace():
            j -= 1
        comma_left = j >= 0 and s[j] == ","

        # look to the right of the token, skipping blanks
        k = end
        while k < len(s) and s[k].isspace():
            k += 1
        comma_right = k < len(s) and s[k] == ","

        # choose replacement
        out.append("" if (comma_left or comma_right) else ",")
        idx = end  # advance cursor

    out.append(s[idx:])  # tail of string
    return "".join(out)


def contains_approximation(s: str) -> bool:
    """Returns whether a boxed answer looks like a trailing approximation."""
    return APPROX_RE.search(s) is not None


def should_prefer_previous_boxed_over_approximation(s: str) -> bool:
    if not contains_approximation(s):
        return False

    prefix = APPROX_RE.split(s, maxsplit=1)[0].strip()
    if not prefix:
        return True
    if "=" in prefix or any(token in prefix for token in EXACT_BOX_TOKENS):
        return False
    return re.fullmatch(r"[\s$(){}\[\].,0-9+\-]+", prefix) is not None


def is_decimal_approximation_box(s: str) -> bool:
    """Returns whether a boxed value is only a decimal approximation."""
    s = s.replace(r"\displaystyle", "")
    s = s.replace("$", "").replace(",", "").strip()
    s = re.sub(r"\\[,;:! ]", "", s)
    s = re.sub(r"\s+", "", s)
    return re.fullmatch(r"[-+]?\d+\.\d+(?:\.\.\.)?", s) is not None


def looks_like_exact_boxed_expression(s: str) -> bool:
    """Returns whether a boxed expression is a plausible exact answer."""
    if contains_approximation(s) or is_decimal_approximation_box(s):
        return False
    return any(token in s for token in EXACT_BOX_TOKENS)


def split_top_level(s: str, delimiter: str) -> list[str]:
    """Splits on delimiters that are not inside braces or parentheses."""
    if delimiter != ",":
        return s.split(delimiter)

    parts = []
    start = 0
    depth = 0
    openers = "({["
    closers = ")}]"
    for i, char in enumerate(s):
        if char in openers:
            depth += 1
        elif char in closers and depth > 0:
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return parts


def expand_pm_list_terms(s: str) -> str:
    r"""Expands top-level list entries containing \pm into plus and minus entries."""
    if r"\pm" not in s:
        return s

    parts = split_top_level(s, ",")
    if len(parts) == 1:
        return s

    expanded = []
    for part in parts:
        if r"\pm" in part:
            expanded.append(part.replace(r"\pm", "+"))
            expanded.append(part.replace(r"\pm", "-"))
        else:
            expanded.append(part)
    return ",".join(expanded)


def canonicalize_frac_shorthand(s: str) -> str:
    """Normalizes common LaTeX fraction and binomial shorthands to braced form."""
    frac_cmd = r"(\\(?:dfrac|tfrac|frac))"
    atom = r"(\\[a-zA-Z]+|[A-Za-z0-9])"
    s = regex.sub(frac_cmd + r"\{((?:[^{}]|\{(?2)\})*)\}\s*" + atom, r"\1{\2}{\3}", s)
    s = re.sub(frac_cmd + r"\s*" + atom + r"\s*" + atom, r"\1{\2}{\3}", s)
    s = re.sub(frac_cmd + r"\s*\{([^{}]+)\}\s*" + atom, r"\1{\2}{\3}", s)
    s = re.sub(frac_cmd + r"\s*" + atom + r"\s*\{", r"\1{\2}{", s)
    s = re.sub(r"(\\binom)\s*" + atom + r"\s*" + atom, r"\1{\2}{\3}", s)
    return s


def strip_trailing_qualifiers(s: str) -> str:
    """Removes conditions/units/approximations appended after the answer."""
    for marker in [r"\qquad", r"\quad"]:
        idx = s.find(marker)
        if idx == -1:
            continue
        tail = s[idx + len(marker) :].lower()
        if any(
            token in tail
            for token in [
                r"\text",
                r"\ge",
                r"\le",
                r"\to",
                ">=",
                "<=",
                "approximately",
                "approx",
                "about",
                "where",
                " for ",
                " as ",
                " if ",
                " when ",
            ]
        ):
            return s[:idx]

    return re.sub(
        r"\s*\((?:[^()]*(?:\\ge|\\le|\\to|>=|<=|>|<|where|for|as|if|when)[^()]*)\)\s*$",
        "",
        s,
        flags=re.I,
    )


def extract_boxed_answer_parse(
    text: str, parse: bool = True, list_answer: bool = False, typed_delimiters: bool = True
) -> Optional[int]:
    """Extracts and parses the content of the last `\boxed` or `\fbox` command.

    Args:
        text (str): The string to search.
        parse (bool, optional): Whether to parse the answer. Defaults to True.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the parsed answer and a warning level.
    """
    answer, warning = extract_boxed_answer(text, list_answer)
    if answer is not None:
        if answer.count("=") > 1:
            warning = max(warning, WarningType.MAJOR)  # this is a major warning, we should not have more than one "="
        try:
            return sympy.Integer(int(answer)), warning
        except:  # noqa: E722
            # logger.info(f"Could not parse answer {answer} as integer")
            if parse:
                parsed_answer, warning = parse_answer(
                    answer, list_answer=list_answer, typed_delimiters=typed_delimiters
                )
                return parsed_answer, warning
            return answer, warning
    return None, WarningType.MAJOR


def extract_last_integer(text: str) -> Optional[int]:
    """Extracts the last integer from a string.

    Args:
        text (str): The string to search.

    Returns:
        tuple: A tuple containing the last integer and a warning level.
    """
    pattern = r"\b\d+\b"
    matches = list(regex.finditer(pattern, text))
    if not matches:
        return None, WarningType.MAJOR
    try:
        return int(matches[-1].group()), WarningType.MAJOR
    except Exception as e:
        logger.warning(f"Error extracting last integer: {e}")
        return None, WarningType.MAJOR


def extract_answer(
    text: str, strict_parsing: bool = True, parse: bool = True, list_answer: bool = False, typed_delimiters: bool = True
):
    """Extracts and parses the final answer from a string.

    Args:
        text (str): The string to search.
        strict_parsing (bool, optional): Whether to use strict parsing. Defaults to True.
        parse (bool, optional): Whether to parse the answer. Defaults to True.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the parsed answer and a warning level.
    """
    if text is None or len(text) == 0:
        return None, WarningType.MAJOR
    warning_old = WarningType.NONE
    if text in complete_mapper:
        text = complete_mapper[text]
        warning_old = WarningType.MAJOR
    text, warning = replace_unicode(text)
    warning = max(warning, warning_old)
    answer, warning_new = extract_boxed_answer_parse(text, parse, list_answer, typed_delimiters=typed_delimiters)
    if isinstance(answer, AnswerList) and len(answer.answers) == 1:
        answer = answer.answers[0]
    warning = max(warning, warning_new)
    if answer is not None or strict_parsing:
        return answer, warning

    return extract_last_integer(text)

def parse_answer(s: str, primitive_type: type = None, list_answer: bool = False, typed_delimiters: bool = True):
    """Parses a string into a mathematical expression.

    Args:
        s (str): The string to parse.
        primitive_type (type, optional): The primitive type to parse into. Defaults to None.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the parsed answer and a warning level.
    """
    warning = WarningType.NONE
    if s in manual_mapper:
        logger.warning(f"Applying manual parsing to {s}")
        s = manual_mapper[s]
        warning = WarningType.MAJOR
    if typed_delimiters:
        typed_output = parse_typed_delimited_answer(s, primitive_type=primitive_type)
        if typed_output is not None:
            return typed_output
    s = remove_invalid_characters(s)
    s = remove_outer_brackets(normalize_string(s, list_answer))
    # s = insert_implicit_mul(s)
    output, warning_new = ParseList.parse("(" + s + ")", primitive_type=primitive_type)
    warning = max(warning, warning_new)
    if output is None:
        logger.warning(f"Could not parse {s}, returning None")
        return None, max(warning, WarningType.MAJOR)
    if len(output) == 1:
        output = output[0]

    if isinstance(output, (list, tuple)):
        output = AnswerList(output)
    return output, warning


def normalize_string(s, list_answer=False):
    """Normalizes a string for parsing.

    Args:
        s (str): The string to normalize.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        str: The normalized string.
    """
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\Bigl", "").replace(r"\Bigr", "")
    s = s.replace(r"\bigl", "").replace(r"\bigr", "")
    s = s.replace(r"\Big", "").replace(r"\big", "").replace(r"\Large", "").replace(r"\large", "")
    s = canonicalize_frac_shorthand(s)
    s = remove_aligns(s)
    s = s.replace("[", "(")
    s = s.replace("]", ")")
    s = s.replace("\\{", "(")  # sets will be converted to lists
    s = s.replace("\\}", ")")  # sets will be converted to lists
    s = s.replace("$", "")
    s = s.replace("\\ ", " ")
    # remove hline and vline
    s = s.replace(r"\hline", "")
    s = s.replace(r"\vline", "")
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace("·", " \\cdot ")
    s = s.replace("^\\circ", " ")
    s = s.replace("^{\\circ}", " ")
    s = s.replace("\\displaystyle", "")
    s = s.replace("\\(", "(")
    s = s.replace("\\)", ")")
    s = s.replace("{,}", "")  # o4-mini does this
    # remove \\begin{anything} and \\end{anything}
    if s.endswith("."):
        s = s[:-1]

    if list_answer and s is not None:
        s = replace_and_or(s)
        s = expand_pm_list_terms(s)

    if not list_answer:
        # remove thousands separators without eating function/list commas like max(1,2s)
        s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)
        s = s.replace("{,}", "")
    if list_answer:
        s = s.replace(";", ",")
        s = s.replace("{,}", ",")
    # if we see \sqrt 123ea pi\frac -> \sqrt{123ea}pi\frac
    if "\\sqrt " in s:
        s = re.sub(r"\\sqrt\s*([^\s{}]*)", r"\\sqrt{\1}", s)
    # remove everything that appears within \text{...}
    s = re.sub(r"\\text\{.*?\}", "", s)
    # replace \mathrm{...} with ...
    s = re.sub(r"\\mathrm\{(.*?)\}", r" \1 ", s)

    s = s.replace("F_{30}", "832040")  # Fibonacci number present in one problem
    if "\\approx" in s:
        s = s.split("\\approx")[0]
        if s.endswith("("):  # in case it was put in brackets
            s = s[:-1]
    if "=" in s:
        if list_answer:
            parts = split_top_level(s, ",")
            if len(parts) > 1:
                s = ",".join(part.split("=")[-1] for part in parts)
            else:
                s = s.split("=")[-1]
        else:
            s = s.split("=")[-1]
    s = strip_trailing_qualifiers(s)
    for relation in [r"\longrightarrow", r"\rightarrow", r"\to", r"\sim"]:
        if relation in s:
            s = s.split(relation)[-1]
    if r"\in" in s and list_answer:
        s = s.split(r"\in")[-1]

    s = re.sub(r"\s*\+\s*o\s*\([^)]*\)\s*$", "", s)
    s = re.sub(r"\s*\+\s*O\s*\([^)]*\)\s*$", "", s)
    s = s.replace(r"\qquad", " ").replace(r"\quad", " ")
    return strip(s)


def remove_outer_brackets(s):
    """Removes the outermost matching brackets from the string if they encompass the entire string.

    Parameters:
    s (str): The input string potentially wrapped with brackets.

    Returns:
    str: The string with the outermost brackets removed if they match and encompass the entire string.
    """
    while True:
        if not s:
            return s
        opening = s[0]
        closing = s[-1]

        if opening == "(" and closing == ")":
            count = 0
            matched = True
            for i, char in enumerate(s):
                if char == opening:
                    count += 1
                elif char == closing:
                    count -= 1
                if count == 0 and i != len(s) - 1:
                    matched = False
                    break

            if matched:
                s = s[1:-1]
                continue
        break

    return s


def parse_typed_delimited_answer(s: str, primitive_type: type = None):
    s = remove_invalid_characters(re.sub(r"\\(?:left|right|Bigl|Bigr|bigl|bigr)", "", s))
    if "=" in s:
        s = s.split("=")[-1]
    s = strip(s)
    if s.startswith(r"\{") and s.endswith(r"\}"):
        kind, inner, brackets, unordered, accepts_bare_list = "set", s[2:-2], ("{", "}"), True, True
    elif len(s) >= 2 and s[0] in "([" and s[-1] in ")]":
        parts = split_top_level(s[1:-1], ",")
        if len(parts) < 2:
            return None
        if (s[0] == "[" or s[-1] == "]") and len(parts) == 2:
            kind, unordered, accepts_bare_list = "interval", False, False
        else:
            kind, unordered, accepts_bare_list = "tuple", False, True
        inner, brackets = s[1:-1], (s[0], s[-1])
    else:
        return None

    parts = split_top_level(inner, ",")
    parsed_parts, warning = [], WarningType.NONE
    for part in parts:
        parsed, new_warning = parse_answer(part, primitive_type=primitive_type)
        if parsed is None:
            return None, max(warning, WarningType.MAJOR)
        parsed_parts.append(parsed)
        warning = max(warning, new_warning)
    return DelimitedAnswer(kind, parsed_parts, brackets, unordered, accepts_bare_list), warning


def remove_aligns(s: str) -> str:
    """Removes `\\begin{align}` and `\\end{align}` environments from a string.

    Args:
        s (str): The string to process.

    Returns:
        str: The processed string.
    """
    # This pattern captures:
    #   \begin{align followed by any non-} characters (like align*, alignat, etc.)
    #   then any content (non-greedily) up to
    #   \\end{align...} with the same "align" prefix
    pattern = r"\\begin{align[^}]*}(.*?)\\end{align[^}]*}"

    # Use a callback to remove '&' from the matched group before returning it
    return re.sub(pattern, lambda m: m.group(1).replace("&", "").replace("\\\\", ""), s, flags=re.DOTALL)


def replace_unicode(text: str) -> str:
    """Replaces unicode characters with their LaTeX equivalents.

    Args:
        text (str): The string to process.

    Returns:
        tuple: A tuple containing the processed string and a warning level.
    """
    text_old = text
    for old, new in {
        "\u23a7": r"\boxed{",
        "\u23ab": r"}",
        "\n\u2502": r"\boxed{",
        "\u2502": r"}",
        "\n\u2503": r"\boxed{",
        "\u2503": r"}",
        "\n\uf8f0": r"\boxed{",
        "\uf8fb": r"}",
        "<|begin_of_box|>": r"\boxed{",
        "<|end_of_box|>": r"}",
    }.items():
        text = text.replace(old, new)
    warning = WarningType.NONE if text == text_old else WarningType.POSSIBLE
    for old, new in {
        "\u221a": r"\sqrt",
        "\u00d7": r"\cdot",
        "\u202f": r" ",
        "\u2212": "-",
        "\u03c0": r"\pi",
        "\u00b1": r"\pm",
    }.items():
        text = text.replace(old, new)
    return text, warning


def remove_invalid_characters(text):
    """Removes invalid characters from a string.

    Args:
        text (str): The string to process.

    Returns:
        str: The processed string.
    """
    return re.sub(r"\\[;:,!]", "", text)


def strip(s: str):
    s = s.strip()
    # be careful with this, it can also remove the "\" in "\begin" if just done with strip
    while s.startswith(r"\n"):
        s = s[2:]
    while s.endswith(r"\n"):
        s = s[:-2]
    while s.startswith("\\ "):
        s = s[2:]
    # if s starts with any thing of the form \\\ and then a bracket, or \\\n and then a bracket, remove it
    while re.match(r"\\{2,}\n?\(", s):
        s = s[3:]
    return s


def split_multiletter_symbols(expr):
    reps = {}
    for s in list(expr.free_symbols):
        name = s.name
        if name.isalpha() and len(name) > 1 and not all(ch in "ABCDE" for ch in name):
            reps[s] = Mul(*[Symbol(ch) for ch in name])
    return expr.xreplace(reps)


def _typed_element_equal(left, right):
    try:
        if left == right:
            return True
    except Exception:
        pass
    return check_answers(left, right)


def _typed_sequence_equal(left, right, unordered=False):
    if len(left) != len(right):
        return False
    if not unordered:
        return all(_typed_element_equal(a, b) for a, b in zip(left, right))
    matched = set()
    for answer in left:
        for i, other in enumerate(right):
            if i not in matched and _typed_element_equal(answer, other):
                matched.add(i)
                break
        else:
            return False
    return True


class DelimitedAnswer:
    def __init__(self, kind, answers, brackets, unordered=False, accepts_bare_list=True):
        self.kind = kind
        self.answers = list(answers)
        self.brackets = brackets
        self.unordered = unordered
        self.accepts_bare_list = accepts_bare_list
        if kind == "interval":
            self.left, self.right = self.answers
            self.left_closed = brackets[0] == "["
            self.right_closed = brackets[1] == "]"

    def equals(self, other):
        if isinstance(other, DelimitedAnswer):
            if (self.kind, self.brackets) != (other.kind, other.brackets):
                return False
            other = other.answers
        elif self.accepts_bare_list and isinstance(other, AnswerList):
            other = other.answers
        elif not (self.accepts_bare_list and isinstance(other, (list, tuple))):
            return False
        return _typed_sequence_equal(self.answers, other, unordered=self.unordered)

    def __str__(self):
        left, right = self.brackets
        return left + ",".join(str(ans) for ans in self.answers) + right

    def __len__(self):
        return len(self.answers)

    def __iter__(self):
        return iter(self.answers)


TYPED_DELIMITED_ANSWER_TYPES = (DelimitedAnswer,)


def check_answers(ans1, ans2):
    """Checks if two answers are equal.

    Args:
        ans1: The first answer.
        ans2: The second answer.

    Returns:
        bool: True if the answers are equal, False otherwise.
    """
    if ans1 is None or ans2 is None:
        return False
    if isinstance(ans2, TYPED_DELIMITED_ANSWER_TYPES):
        return bool(ans2.equals(ans1))
    if isinstance(ans1, TYPED_DELIMITED_ANSWER_TYPES):
        return bool(ans1.equals(ans2))
    if (type(ans1) in [list, AnswerList]) != (type(ans2) in [list, AnswerList]):
        return False
    try:
        if not (hasattr(ans1, "equals") and callable(ans1.equals)) or not (
            hasattr(ans2, "equals") and callable(ans2.equals)
        ):
            if isinstance(ans1, str) or isinstance(ans2, str):
                # sympy check equality
                return bool(ans1 == ans2)
                        
            # do approximate equal here
            err = abs(N(ans1 - ans2))
            # Exact equality includes 0 == 0, for which relative error
            # would otherwise divide by zero and produce NaN.
            if err == 0:
                return True
            if err < 1e-10 and err / max(abs(N(ans1)), abs(N(ans2))) < 1e-10:
                return True
            return False
        
        if not isinstance(ans1, AnswerList):
            ans1 = split_multiletter_symbols(ans1)
        if not isinstance(ans2, AnswerList):
            ans2 = split_multiletter_symbols(ans2)
        return bool(ans1.equals(ans2))
    except Exception as e:
        logger.warning(f"Error comparing answers {ans1} and {ans2}: {e}")
        return False


class AnswerList:
    """A class for representing a list of answers."""

    def __init__(self, answers: list[Any]):
        """Initializes the AnswerList.

        Args:
            answers (list[Any]): A list of answers.
        """
        if not isinstance(answers, (list, tuple)):
            raise ValueError(f"Expected passed answers to be list or tuple, received {type(answers)}")

        valid_answers = []
        for answer in answers:
            if bool(re.search(r"\d", str(answer))):
                valid_answers.append(answer)
            else:
                logger.warning(f"Could not find any numbers in {answer}, removed from list")

        self.answers = list(valid_answers)

    def equals(self, other: list[Any]):
        """Checks if this AnswerList is equal to another list of answers.

        Args:
            other (list[Any]): The other list of answers.

        Returns:
            bool: True if the lists are equal, False otherwise.
        """
        if len(self.answers) != len(other):
            # logger.info(f"Lists {self.answers} and {other} do not have the same length.")
            return False

        match_ids = set()
        for ans1 in self.answers:
            match_found = False
            for i, ans2 in enumerate(other):
                if i not in match_ids and check_answers(ans1, ans2):
                    match_ids.add(i)
                    match_found = True
                    break
            if not match_found:
                # logger.info(f"Could not find a match for element {ans1} in {other}")
                return False
        return True

    def __str__(self):
        return "[" + ",".join(str(ans) for ans in self.answers) + "]"

    def __len__(self):
        return len(self.answers)

    def __iter__(self):
        return iter(self.answers)


class ParseObject:
    """A base class for parsing objects."""

    @classmethod
    def is_at_start(cls, string):
        """Checks if the object is at the start of a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is at the start of the string, False otherwise.
        """
        return False

    @classmethod
    def is_complete(cls, string):
        """Checks if the object is complete in a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is complete in the string, False otherwise.
        """
        return string.count("{") == string.count("}") and string.count("(") == string.count(")")

    @classmethod
    def is_finished(cls, string):
        """Checks if the object is finished in a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is finished in the string, False otherwise.
        """
        return True

    @classmethod
    def parse(cls, string):
        """Parses a string into an object.

        Args:
            string (str): The string to parse.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError


class ParsePrimitive(ParseObject):
    """A class for parsing primitive types."""

    @staticmethod
    def _rewrite_latex(latex_str, brace_digits=False):
        if brace_digits:
            latex_str = re.sub(r"\{(\d+)\}", r"(\1)", latex_str)
        latex_str = re.sub(r"\\lfloor\s*(.*?)\s*\\rfloor", r"floor(\1)", latex_str)
        latex_str = re.sub(r"\\lceil\s*(.*?)\s*\\rceil", r"ceiling(\1)", latex_str)
        latex_str = re.sub(r"\\*(?:dfrac|tfrac|frac)\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", latex_str)
        latex_str = re.sub(r"\\*binom\{([^{}]*)\}\{([^{}]*)\}", r"binomial(\1, \2)", latex_str)
        latex_str = re.sub(r"\\*sqrt\[(.*?)\]\{(.*?)\}", r"(\2)**(1/(\1))", latex_str)
        latex_str = re.sub(r"\\*sqrt\((\d+)\)\{([^{}]*)\}", r"(\2)**(1/(\1))", latex_str)
        latex_str = re.sub(r"\\*sqrt\{(.*?)\}", r"(\1)**(1/2)", latex_str)
        latex_str = latex_str.replace("^", "**")
        latex_str = latex_str.replace("\\cdot", "*").replace("\\times", "*")
        latex_str = latex_str.replace("\\pi", " pi ").replace("\\e", " E ").replace("\\i", " I ")
        return re.sub(r"\bi\b", "I", latex_str)

    @classmethod
    def parse(cls, string, primitive_type):
        """Parses a string into a primitive type.

        Args:
            string (str): The string to parse.
            primitive_type (type): The primitive type to parse into.

        Returns:
            tuple: A tuple containing the parsed primitive and a warning level.
        """
        warning = WarningType.NONE
        string = canonicalize_frac_shorthand(string)
        # Integer
        if string.isdigit():
            if primitive_type == Fraction:
                return Fraction(int(string), 1)
            return int(string), warning
        # Float
        try:
            float_string = float(string)
            if int(float_string) == float_string:
                if primitive_type == Fraction:
                    return Fraction(int(float_string), 1)
                return int(float_string), warning
            return float_string, warning
        except ValueError:
            # logger.info(f"Couldn't configure floating point to fraction for {string}")
            pass
        # Expression
        if bool(re.search(r"sqrt(\d+)", string)):
            string = re.sub(r"sqrt(\d+)", r"sqrt{\1}", string)
        if bool(re.search(r"frac(\d)", string)):
            string = re.sub(r"frac(\d)", r"frac{\1}", string)
        try:
            latex_str = string
            for _ in range(5):
                init_str = latex_str
                latex_str = cls._rewrite_latex(latex_str)
                if init_str == latex_str:
                    break

            for _ in range(5):
                init_str = latex_str
                latex_str = cls._rewrite_latex(latex_str, brace_digits=True)
                if init_str == latex_str:
                    break

            # Handle implcit multiplication
            latex_str = re.sub(r"(\d|(?<![a-zA-Z])[a-zA-Z]{1,2}(?![a-zA-Z]))\(", r"\1*(", latex_str)
            latex_str = re.sub(r"\)(\d|(?<![a-zA-Z])[a-zA-Z]{1,2}(?![a-zA-Z]))", r")*\1", latex_str)
            latex_str = re.sub(r"(?<=\d)((?<![a-zA-Z])[a-zA-Z]{1,2}(?![a-zA-Z]))", r"*\1", latex_str)
            latex_str = re.sub(r"((?<![a-zA-Z])[a-zA-Z]{1,2}(?![a-zA-Z]))(?=\d)", r"\1*", latex_str)
            latex_str = re.sub(r"\)\s*\(", r")*(", latex_str)
            latex_str = re.sub(r"\{([^{}]*)\}", lambda m: "[" + m.group(1).replace(",", ", ") + "]", latex_str)

            if latex_str == "None":
                string = sympy.core.symbol.Symbol("None")
            else:
                string = sympy.sympify(latex_str, locals=SYMPY_LOCALS)
        except Exception as e:
            try:
                string_no_eq = string
                if "=" in string_no_eq:
                    # rfind is used to remove the last occurence of "="
                    string_no_eq = string_no_eq[string_no_eq.rfind("=") + 1 :]
                output_val = latex2sympy_fixed(string_no_eq)
                # print complex and real part separately

                try:
                    simplified = simplify(output_val)
                    if simplified.is_integer:
                        return int(simplified), warning
                    if isinstance(simplified, sympy.Rational):
                        return simplified, warning
                    float_val = float(N(output_val, 101))
                    if float("inf") == float_val or float("-inf") == float_val:
                        return int(N(latex2sympy_fixed(string_no_eq), 50001)), warning  # important for large ints
                    return float_val, warning
                except:  # noqa: E722
                    try:
                        complex_val = complex(N(output_val, 101))
                        return complex_val, warning
                    except:  # noqa: E722
                        return output_val, warning
            except Exception as e:
                logger.warning(f"Error: Custom parsing error {e}, {string_no_eq}")
                warning = max(warning, WarningType.MAJOR)
                return None, warning

        return string, warning

    @classmethod
    def is_at_start(cls, string):
        return True


class ParseList(ParseObject):
    """A class for parsing lists."""

    @classmethod
    def is_at_start(cls, string):
        """Checks if the object is at the start of a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is at the start of the string, False otherwise.
        """
        return string.startswith(r"(")

    @classmethod
    def is_finished(cls, string):
        """Checks if the object is finished in a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is finished in the string, False otherwise.
        """
        # safe condition for finishing a list
        return string.strip().strip(",").endswith(")")

    @classmethod
    def is_complete(cls, string):
        """Checks if the object is complete in a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is complete in the string, False otherwise.
        """
        return string.count("(") == string.count(")")

    @classmethod
    def never_zero_count(cls, string):
        """Checks if the parenthesis count never reaches zero before the end of the string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the parenthesis count never reaches zero, False otherwise.
        """
        # says wheter count "(" - count ")" for every string[:i] is never zero
        count = 0
        ever_zero = False
        for char in string:
            if char == "(":
                count += 1
            if char == ")":
                count -= 1
            if count == 0:
                ever_zero = True
        return not ever_zero

    @classmethod
    def parse(cls, string, delimiter=[r"\n", ","], primitive_type=None, depth=0):
        """Parses a string into a list.

        Args:
            string (str): The string to parse.
            delimiter (list[str], optional): The delimiter to use. Defaults to [r"\n", ","].
            primitive_type (type, optional): The primitive type to parse into. Defaults to None.
            depth (int, optional): The recursion depth. Defaults to 0.

        Returns:
            tuple: A tuple containing the parsed list and a warning level.
        """
        if isinstance(delimiter, str):
            delimiter = [delimiter]
        output = []
        if not string.startswith("("):
            return None
        string = string.strip().strip(",")
        if cls.never_zero_count(string[:-1]):
            string = string[1:-1]
        string = strip(string)
        used_delim = delimiter[0]
        if not any(len(split_top_level(string, delim)) > 1 for delim in delimiter):
            parsed, warning = ParsePrimitive.parse(string, primitive_type=primitive_type)
            return [parsed], warning
        for delim in delimiter:
            comma_separated = split_top_level(string, delim)
            if len(comma_separated) > 1:
                used_delim = delim
                break
        warning = WarningType.NONE
        while len(string) > 0:
            previous_string = string
            comma_separated = split_top_level(string, used_delim)
            allowed_objects = [ParseList, ParsePrimitive]
            if depth > 50:
                allowed_objects = [ParsePrimitive]
            for obj in allowed_objects:
                if obj.is_at_start(strip(string)):
                    current_index = 1
                    segment = strip(used_delim.join(comma_separated[:current_index]))
                    while not obj.is_complete(segment) or not obj.is_finished(segment):
                        current_index += 1
                        if current_index >= len(comma_separated):
                            break
                        segment = strip(used_delim.join(comma_separated[:current_index]))
                    if not obj.is_complete(segment) or not obj.is_finished(segment):
                        continue

                    if obj == ParseList:
                        parsed, new_warning = obj.parse(
                            segment, primitive_type=primitive_type, depth=depth + 1
                        )
                    else:
                        parsed, new_warning = obj.parse(segment, primitive_type=primitive_type)
                    warning = max(warning, new_warning)
                    output.append(parsed)
                    string = strip(used_delim.join(comma_separated[current_index:]))
                    break
            if previous_string == string:
                if depth > 50:
                    logger.error(f"Response {string} reached depth > 50")
                    raise ValueError(f"Failed to parse '{string}'")
                return None, WarningType.MAJOR
        return output, warning
