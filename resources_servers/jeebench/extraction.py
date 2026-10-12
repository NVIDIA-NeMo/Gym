# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gold-blind extraction of explicitly stated answers, including Indic and LaTeX forms.

This module never solves a question or receives its reference answer. Unsupported
or ambiguous conclusions remain unparsed; automatic extraction stays provisional.
"""

from __future__ import annotations

import ast
import math
import re
import unicodedata
from collections.abc import Callable


PARSER_VERSION = "indic-formats-v2.6"
MAX_PAYLOAD_CHARS = 4096
# Answer/option vocabulary covers the fourteen languages in the Indic campaign.
ANSWER_WORDS = (
    "उत्तर",
    "उत्तरे",
    "उत्तरम्",
    "উত্তর",
    "উত্তৰ",
    "જવાબ",
    "ઉત્તર",
    "ಉತ್ತರ",
    "ಉತ್ತರಗಳು",
    "ഉത്തരം",
    "ഉത്തരങ്ങൾ",
    "ଉତ୍ତର",
    "ਉੱਤਰ",
    "ਜਵਾਬ",
    "விடை",
    "பதில்",
    "సమాధానం",
    "జవాబు",
    "جواب",
)
OPTION_WORDS = (
    "विकल्प",
    "विकल्पहरू",
    "पर्याय",
    "বিকল্প",
    "বিকল্পসমূহ",
    "વિકલ્પ",
    "ಆಯ್ಕೆ",
    "ಆಯ್ಕೆಗಳು",
    "ഓപ്ഷൻ",
    "ഓപ്ഷനുകൾ",
    "ବିକଳ୍ପ",
    "ਵਿਕਲਪ",
    "விருப்பம்",
    "விருப்பங்கள்",
    "தேர்வு",
    "ఎంపిక",
    "ఎంపికలు",
    "اختیار",
    "آپشن",
    "सही कथन",
    "सत्य कथन",
)
CONNECTORS = (
    "and",
    "और",
    "तथा",
    "एवं",
    "व",
    "र",
    "च",
    "ও",
    "এবং",
    "আৰু",
    "અને",
    "ಮತ್ತು",
    "ഉം",
    "കൂടാതെ",
    "ଏବଂ",
    "ଓ",
    "ਅਤੇ",
    "மற்றும்",
    "மరియు",
    "اور",
)
COPULAS = (
    "is",
    "are",
    "equals",
    "है",
    "हैं",
    "होता है",
    "होती है",
    "হ'ল",
    "হল",
    "হয়",
    "হৈছে",
    "হবে",
    "છે",
    "ಆಗಿದೆ",
    "ಇದೆ",
    "ಆಗಿವೆ",
    "ആണ്",
    "ആകുന്നു",
    "ଅଟେ",
    "ହେଉଛି",
    "ਹੈ",
    "ਹਨ",
    "ஆகும்",
    "ஆகிறது",
    "உள்ளது",
    "అవుతుంది",
    "ఉంది",
    "ہے",
    "ہیں",
)
POSITIVE_WORDS = (
    "correct",
    "true",
    "valid",
    "right",
    "सही",
    "सत्य",
    "योग्य",
    "সঠিক",
    "শুদ্ধ",
    "সত্য",
    "સાચો",
    "સાચા",
    "સાચું",
    "ಸರಿಯಾದ",
    "ಸರಿ",
    "ശരി",
    "ശരിയായ",
    "ସଠିକ",
    "ଠିକ୍",
    "ਸਹੀ",
    "சரி",
    "சரியான",
    "సరైన",
    "సత్య",
    "صحیح",
    "درست",
)
NEGATIVE_WORDS = (
    "incorrect",
    "false",
    "wrong",
    "not",
    "गलत",
    "असत्य",
    "नहीं",
    "ভুল",
    "অশুদ্ধ",
    "মিথ্যা",
    "নহয়",
    "ખોટો",
    "ತಪ್ಪು",
    "തെറ്റ്",
    "ଭୁଲ",
    "ਗਲਤ",
    "தவறு",
    "తప్పు",
    "غلط",
)
NEGATIVE_PATTERN = "|".join(sorted(map(re.escape, NEGATIVE_WORDS), key=len, reverse=True))
OPTION_NOUNS = (
    "option",
    "options",
    "choice",
    "choices",
    "statement",
    "statements",
    "relation",
    "relations",
    "विकल्प",
    "कथन",
    "उक्ति",
    "বিকল্প",
    "উক্তি",
    "উক্তিবোৰ",
    "উক্তিসমূহ",
    "উক্তিগুলি",
    "વિકલ્પ",
    "ಆಯ್ಕೆ",
    "ಹೇಳಿಕೆ",
    "ഓപ്ഷൻ",
    "പ്രസ്താവന",
    "ବିକଳ୍ପ",
    "କଥନ",
    "ਵਿਕਲਪ",
    "ਕਥਨ",
    "விருப்பம்",
    "கூற்று",
    "ఎంపిక",
    "ప్రకటన",
    "اختیار",
    "بیان",
)
VALUE_WORDS = (
    "value",
    "answer",
    "result",
    "minimum",
    "maximum",
    "number",
    "total",
    "sum",
    "product",
    "ratio",
    "exponent",
    "pressure",
    "energy",
    "speed",
    "rate",
    "mass",
    "मान",
    "मूल्य",
    "संख्या",
    "उत्तर",
    "মান",
    "সংখ্যা",
    "উত্তর",
    "મૂલ્ય",
    "સંખ્યા",
    "ಮೌಲ್ಯ",
    "ಸಂಖ್ಯೆ",
    "മൂല്യം",
    "എണ്ണം",
    "ମୂଲ୍ୟ",
    "ସଂଖ୍ୟା",
    "ਮੁੱਲ",
    "ਗਿਣਤੀ",
    "மதிப்பு",
    "எண்ணிக்கை",
    "విలువ",
    "సంఖ్య",
    "قدر",
    "تعداد",
)
UNCERTAIN = re.compile(
    r"\b(?:or|maybe|perhaps|uncertain|might|either|not|neither|incorrect|false)\b"
    r"|या|अथवा|नहीं|गलत|অথবা|ভুল|અથવા|ಅಥವಾ|അല്ലെങ്കിൽ|ଅଥବା|ਜਾਂ|அல்லது|లేదా|یا",
    re.IGNORECASE,
)
LOCAL_WORDS = "|".join(sorted(map(re.escape, ANSWER_WORDS + OPTION_WORDS), key=len, reverse=True))
ENGLISH_MARKER = re.compile(
    r"(?i)\b(?:(?:final\s+|correct\s+)?answers?|correct\s+(?:options?|choices?|statements?))"
    r"(?:\s*\([^\n)]{1,100}\))?\s*(?::|=|\bis\b|\bare\b)\s*"
)
LOCAL_MARKER = re.compile(
    r"(?:"
    + LOCAL_WORDS
    + r")\s*(?::|=|(?:"
    + "|".join(sorted(map(re.escape, COPULAS), key=len, reverse=True))
    + r")\s*:?)?\s*",
    re.I,
)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
# Explicit units only; never discard arbitrary prose, alternatives, or inequalities.
UNITS = re.compile(
    r"(?:(?:bar|atm|Pa|kPa|MPa|GPa|Torr|torr|K|°C|°|deg|degrees?|rad|radians?|mol|moles?|"
    r"m|cm|mm|nm|pm|km|s|ms|min|h|kg|g|mg|N|J|kJ|eV|MeV|V|A|W|Hz|C|F|H|T|M|L|l|"
    r"litres?|liters?|ohms?|Ω|बार|केल्विन|मोल|मीटर|सेकंड|जूल|न्यूटन)"
    r"(?:\^?[+-]?\d+)?[\s·*/]*)+"
)


def clean_math(text: str) -> str:
    """Remove presentation markup while retaining numbers, labels and operators."""
    text = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+", lambda m: "^{" + unicodedata.normalize("NFKC", m[0]).replace("−", "-") + "}", text
    )
    text = unicodedata.normalize("NFKC", text).replace("−", "-").replace("–", "-")
    text = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
    text = re.sub(r"\\(?:left|right|displaystyle|textstyle)\b", "", text)
    text = re.sub(r"\\(?:[,;:!]|quad\b|qquad\b|enspace\b|thinspace\b| )", " ", text)
    text = re.sub(r"\\[()[\]]", "", text).replace(r"\%", "%").replace("$", "").replace("`", "")
    text = re.sub(r"(?<!\d)\*\*|\*\*(?!\d)", "", text)
    for _ in range(12):
        changed = re.sub(
            r"\\(?:text|textbf|textrm|mathrm|mathbf|mathit|operatorname|mbox|emph|underline)\{([^{}]*)\}",
            r"\1",
            text,
        )
        changed = re.sub(r"\{\\(?:rm|bf)\s+([^{}]*)\}", r"\1", changed)
        if changed == text:
            break
        text = changed
    return text.strip().strip("*_ ")


def numeric_value(text: str) -> str | None:
    """Parse a bounded literal arithmetic value, optionally followed by units."""
    text = clean_math(text).strip().rstrip(".।۔ ")
    if re.fullmatch(NUMBER, text):
        return text
    # Strip only an explicit recognized unit suffix (no scale/unit conversion).
    for match in re.finditer(r"\s+(?=[A-Za-z°Ω\u0900-\u097f])", text):
        if UNITS.fullmatch(text[match.end() :]):
            text = text[: match.start()].strip()
            break
    for _ in range(8):
        changed = re.sub(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", text)
        if changed == text:
            break
        text = changed
    text = text.replace(r"\times", "*").replace(r"\cdot", "*").replace("×", "*")
    text = re.sub(r"\^\{([+-]?\d+)\}", r"**(\1)", text)
    text = re.sub(r"\^([+-]?\d+)", r"**(\1)", text)
    if len(text) > 160 or not re.fullmatch(r"[\d.eE+*/()\s-]+", text):
        return None
    try:
        tree = ast.parse(text.strip(), mode="eval")
        if len(list(ast.walk(tree))) > 40:
            return None

        def value(node: ast.AST) -> float:
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return float(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return value(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            if isinstance(node, ast.BinOp):
                a, b = value(node.left), value(node.right)
                if isinstance(node.op, ast.Add):
                    return a + b
                if isinstance(node.op, ast.Sub):
                    return a - b
                if isinstance(node.op, ast.Mult):
                    return a * b
                if isinstance(node.op, ast.Div):
                    return a / b
                if isinstance(node.op, ast.Pow) and abs(b) <= 100 and abs(a) <= 1e100:
                    return a**b
            raise ValueError("Unsupported expression")

        result = value(tree.body)
        return str(result) if isinstance(result, (int, float)) and math.isfinite(result) else None
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


SCALAR_LITERAL = (
    rf"(?:\\(?:dfrac|tfrac|frac)\{{[^{{}}]+\}}\{{[^{{}}]+\}}|{NUMBER}\s*/\s*{NUMBER}|{NUMBER})"
    r"(?:\s*(?:\\times|\\cdot|×|\*)\s*10(?:\^\{[+-]?\d+\}|\^[+-]?\d+))?"
)
COPULA_PATTERN = "|".join(sorted(map(re.escape, COPULAS), key=len, reverse=True))
VALUE_PATTERN = "|".join(sorted(map(re.escape, VALUE_WORDS), key=len, reverse=True))
POSITIVE_PATTERN = "|".join(sorted(map(re.escape, POSITIVE_WORDS), key=len, reverse=True))
OPTION_PATTERN = "|".join(sorted(map(re.escape, OPTION_NOUNS), key=len, reverse=True))
CONCLUSION_PATTERN = re.compile(
    r"(?i)(?:^|[\n.!?।۔])\s*(?:thus|hence|therefore|so|finally|consequently|"
    r"अतः|इसलिए|इसलिये|अंततः|সুতরাং|সেয়ে|অতএব|તેથી|ಆದ್ದರಿಂದ|ಹೀಗಾಗಿ|"
    r"അതിനാൽ|അതുകൊണ്ട്|ତେଣୁ|ଅତଏବ|ਇਸ ਲਈ|எனவே|ஆகவே|కాబట్టి|అందువల్ల|لہذا|چنانچہ)"
    r"(?=\s|[,，:؛]|$)"
)


def scalar_prefix(text: str, kind: str, normalize: Callable[[str, str], str | None]) -> str | None:
    """Read one scalar at the start of an explicitly selected payload."""
    text = clean_math(text).lstrip(" :：=≈~")
    text = re.sub(r"^\([^()]{0,80}\)\s*[:：]?\s*", "", text)
    match = re.match(SCALAR_LITERAL, text)
    if not match:
        return None
    number = numeric_value(match[0])
    answer = normalize(number, kind) if number is not None else None
    if answer is None:
        return None
    suffix = text[match.end() :].lstrip()
    # A selected scalar may be followed by a unit, terminator, or explanation.
    if not suffix or suffix.startswith((".", "।", "۔", ",", ";", ")", "(", "%")):
        return answer
    unit_end = re.match(r"(?:\\?[A-Za-z°Ω]+(?:\^?\{?[+-]?\d+\}?)?[\s·*/,-]*)+", suffix)
    if unit_end:
        rest = suffix[unit_end.end() :].lstrip()
        if not rest or rest.startswith((".", "।", "۔", ",", ";", ")", "(")):
            return answer
    return None


def explicit_scalar(
    text: str, kind: str, normalize: Callable[[str, str], str | None], *, allow_assignment: bool = True
) -> str | None:
    """Extract a scalar from an answer/value assertion, never from free calculations."""
    if len(text) > MAX_PAYLOAD_CHARS:
        return None
    text = clean_math(text).strip()
    if UNCERTAIN.search(text):
        return None
    answer = scalar_prefix(text, kind, normalize)
    if answer is not None:
        return answer
    relations = []
    relation = re.compile(
        rf"(?:{VALUE_PATTERN})(?:\s*(?:=|≈|:)|[^\n.!?।۔=]{{0,60}}?(?:{COPULA_PATTERN}))\s*({SCALAR_LITERAL})",
        re.I,
    )
    for match in relation.finditer(text):
        number = numeric_value(match[1])
        answer = normalize(number, kind) if number is not None else None
        if answer is not None:
            relations.append(answer)
    if relations:
        return relations[-1]
    # A short scalar assignment in a selected answer is also explicit.
    if not allow_assignment:
        return None
    assignments = []
    for match in re.finditer(rf"(?<![\w])(?:[A-Za-zΑ-ω]\w*(?:_\{{?\w+\}}?)?)\s*(?:=|≈)\s*({SCALAR_LITERAL})", text):
        number = numeric_value(match[1])
        answer = normalize(number, kind) if number is not None else None
        if answer is not None:
            assignments.append(answer)
    return assignments[-1] if len(set(assignments)) == 1 else None


def _roman_to_option(match: re.Match[str]) -> str:
    return {"i": "A", "ii": "B", "iii": "C", "iv": "D"}[match[1].lower()]


def explicit_choices(
    text: str, kind: str, normalize: Callable[[str, str], str | None], *, selected: bool = False
) -> str | None:
    """Read a clearly selected option list, including statement numbers i--iv."""
    if kind not in {"MCQ", "MCQ(multiple)"} or len(text) > MAX_PAYLOAD_CHARS:
        return None
    text = clean_math(text)
    text = re.sub(r"[\[(]\s*(i|ii|iii|iv)\s*[\])]", _roman_to_option, text, flags=re.I)
    direct = normalize(text, kind)
    if direct is not None:
        return direct
    label = r"(?:[\[(]\s*[A-Da-d]\s*[\])]|(?<![A-Za-z])[A-Da-d](?![A-Za-z]))"
    connector = r"(?:\s*(?:,|;|&|\\&|/|(?:" + "|".join(map(re.escape, CONNECTORS)) + r"))\s*)"
    labels = label + r"(?:" + connector + label + r")*"

    def normalize_labels(value: str) -> str | None:
        found = re.findall(r"[\[(]\s*([A-Da-d])\s*[\])]|(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", value)
        value = "".join((left or right).upper() for left, right in found)
        return normalize(value, kind) if value else None

    candidates = []
    for pattern in (
        re.compile(rf"({labels})\s*(?:(?:{COPULA_PATTERN})\s*)?(?:{POSITIVE_PATTERN})", re.I),
        re.compile(
            rf"(?:{POSITIVE_PATTERN})\s*(?:{OPTION_PATTERN})?\s*(?:(?:{COPULA_PATTERN})\s*)?[:：-]?\s*({labels})",
            re.I,
        ),
        re.compile(
            rf"\b(?:this|it|answer|result)\b\s*(?:(?:{COPULA_PATTERN})\s*)"
            rf"(?:the\s+)?(?:{OPTION_PATTERN})\s*[:：-]?\s*({labels})",
            re.I,
        ),
    ):
        for match in pattern.finditer(text):
            answer = normalize_labels(match[1])
            if answer is not None:
                candidates.append((match.start(), answer))
    if selected:
        # A boxed/final payload can contain an inflected or misspelled option
        # noun between an explicit correctness word and the selected labels.
        pattern = re.compile(rf"(?:{POSITIVE_PATTERN})[^()\[\]\n]{{0,60}}?({labels})", re.I)
        for match in pattern.finditer(text):
            answer = normalize_labels(match[1])
            if answer is not None:
                candidates.append((match.start(), answer))
    return max(candidates)[1] if candidates else None


def payload_answer(
    text: str, kind: str, normalize: Callable[[str, str], str | None], *, question: str = ""
) -> str | None:
    """Interpret a selected final payload; do not search a solution for guesses."""
    if len(text) > MAX_PAYLOAD_CHARS:
        return None
    text = clean_math(text)
    direct = normalize(text, kind)
    if direct is not None:
        return direct
    if kind in {"Numeric", "Integer"}:
        # A scalar equation can state its value on the right of the last equality.
        # Do not accept lists of assignments, inequalities, or prose alternatives.
        equation = re.sub(r"\\(?:approx|simeq)(?![A-Za-z])|≈", "=", text)
        if "=" in equation and not re.search(r"[,;<>]|\\(?:leq?|geq?|pm)\b|\b(?:or|and)\b", equation):
            left, right = equation.rsplit("=", 1)
            words = re.findall(r"(?<!\\)\b[A-Za-z]{3,}\b", left)
            allowed = {
                "Product",
                "product",
                "Lines",
                "lines",
                "max",
                "min",
                "lim",
                "frac",
                "dfrac",
                "tfrac",
                "sqrt",
                "cos",
                "sin",
                "tan",
                "alpha",
                "beta",
                "gamma",
                "theta",
                "lambda",
                "cdots",
                "times",
                "cdot",
                "infty",
                "Delta",
                "sum",
                "log",
                "ln",
                "left",
                "right",
            }
            if all(word in allowed for word in words):
                text = right.strip()
        number = numeric_value(text)
        answer = normalize(number, kind) if number is not None else None
        return answer if answer is not None else explicit_scalar(text, kind, normalize)
    text = text.lstrip("→⇒➜➤• ")
    # Accept descriptors only at the start of an explicitly selected payload.
    text = re.sub(r"(?i)^(?:(?:the\s+)?(?:correct\s+)?(?:options?|choices?|answers?)\s*(?:are|is|:)?)\s*", "", text)
    text = re.sub(r"^(?:(?:सही|अंतिम|केवल|correct)\s+)?(?:" + LOCAL_WORDS + r"|कथन)\s*[:：]?\s*", "", text)
    text = re.sub(rf"^(?:(?:{COPULA_PATTERN})|হ'ল)\s*[:：-]?\s*", "", text, flags=re.I)
    text = re.sub(r"(?i)\b(?:" + "|".join(CONNECTORS) + r")\b", ",", text)
    # Explicit positive/negative lists in the same answer (common in Indic prose).
    clauses = re.fullmatch(
        r"([A-Da-d\s,()\[\]&]+)\s+(?:सही|सत्य)\s*(?:हैं|है)?[;।.]\s*"
        r"(?:विकल्प\s+|कथन\s+)?([A-Da-d\s,()\[\]&]+)\s+(?:गलत|असत्य)\s*(?:हैं|है)?[।.]*",
        text,
    )
    if clauses:
        yes = normalize(clauses[1], "MCQ(multiple)")
        no = normalize(clauses[2], "MCQ(multiple)")
        if yes and no and not set(yes) & set(no):
            return normalize(yes, kind)
        return None
    # The enclosing marker may already say "correct options", leaving a
    # payload of only the positive list followed by an explicit negative list.
    if ";" in text:
        yes_text, no_text = text.split(";", 1)
        no_match = re.match(
            rf"([A-Da-d\s,(){{}}\[\]&]+)\s+(?:{NEGATIVE_PATTERN})(?:\s+(?:{COPULA_PATTERN}))?",
            no_text.strip(),
            re.I,
        )
        yes = normalize(yes_text, "MCQ(multiple)")
        no = normalize(no_match[1], "MCQ(multiple)") if no_match else None
        if yes and no and not set(yes) & set(no):
            return normalize(yes, kind)
    # A positive assertion after a pure option list is still an explicit selection.
    positive = re.fullmatch(
        r"([A-Da-d\s,;(){}\[\]&]+)\s+(?:(?:are|is)\s+(?:the\s+)?correct(?:\s+(?:options?|answers?))?"
        r"|(?:सभी\s+)?(?:सही|सत्य)\s*(?:हैं|है)?|हैं|है|only|হয়|সঠিক|ശരിയാണ്|சரியானவை|సరైనవి)\s*[.।۔]*",
        text,
        re.I,
    )
    if positive:
        text = positive[1]
    choices = explicit_choices(text, kind, normalize, selected=True)
    if choices is not None:
        return choices
    if UNCERTAIN.search(text):
        return None
    # TeX spacing and escaped ampersands occur in multi-select lists.
    text = text.replace(r"\&", "&")
    direct = normalize(text, kind)
    if direct is not None:
        return direct
    # Explicit single option followed by its description; reject another label.
    match = re.match(r"^(?:\(([A-Da-d])\)|\[([A-Da-d])\]|([A-Da-d])\s*[.):–—-])(?:\s+|$)(.*)$", text, re.S)
    if match and not re.search(r"\([A-Da-d]\)|\[[A-Da-d]\]|\b[A-D][.)](?:\s|$)", match[4]):
        answer = next(group for group in match.groups()[:3] if group).upper()
        return normalize(answer, kind)
    return None


def match_option_text(payload: str, question: str) -> str | None:
    """Map an explicitly returned option value to one unique supplied option.

    This is exact text matching after presentation normalization, not solving or
    a semantic judge. Duplicate option values and partial matches are rejected.
    """
    labels = list(re.finditer(r"(?m)^\s*\[([A-D])\]\s*", question))
    if [m[1] for m in labels] != list("ABCD"):
        return None

    def canonical(value: str) -> str:
        value = clean_math(value).replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
        return re.sub(r"\s+", "", value.strip().rstrip(".।۔"))

    target = canonical(payload)
    matches = []
    for i, match in enumerate(labels):
        option = question[match.end() : labels[i + 1].start() if i + 1 < len(labels) else len(question)]
        if target and target == canonical(option):
            matches.append(match[1])
    return matches[0] if len(matches) == 1 else None


def option_verdicts(text: str, kind: str) -> str | None:
    """Read explicit true/false verdicts only when ALL four options are classified."""
    if kind not in {"MCQ", "MCQ(multiple)"}:
        return None
    verdicts: dict[str, set[bool]] = {}
    pattern = re.compile(
        r"[\[(]([A-D])[\])]\s*[:–—-]?\s*(?:(?:is|are)\s+)?"
        r"(incorrect|false|correct|true|गलत|असत्य|सही|सत्य)(?!\w)",
        re.I,
    )
    for match in pattern.finditer(text):
        verdicts.setdefault(match[1].upper(), set()).add(match[2].lower() in {"correct", "true", "सही", "सत्य"})
    if set(verdicts) != set("ABCD") or any(len(v) != 1 for v in verdicts.values()):
        return None
    choices = "".join(k for k in "ABCD" if verdicts[k] == {True})
    if kind == "MCQ" and len(choices) != 1:
        return None
    return choices or "None"


def extract_formatted_answer(
    text: str, kind: str, normalize: Callable[[str, str], str | None], *, question: str = ""
) -> tuple[str, str]:
    """Select the latest explicit conclusion, then a standalone final value.

    Invalid latest candidates are never replaced with earlier intermediate ones.
    Both the selection rules and payload normalization are independent of gold.
    """
    # Normalize before finding candidates so their offsets share one coordinate system.
    text = clean_math(text)
    candidates: list[tuple[int, str, str]] = []
    boxed_spans = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth, end = 1, match.end()
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        payload = text[match.end() : end - 1] if depth == 0 else ""
        candidates.append((match.start(), payload, "boxed"))
        boxed_spans.append((match.start(), end))
    for pattern in (ENGLISH_MARKER, LOCAL_MARKER):
        for match in reversed(list(pattern.finditer(text))):
            if candidates and match.start() < max(c[0] for c in candidates):
                break
            if any(start <= match.start() < end for start, end in boxed_spans):
                continue
            # Ignore a marker inside an earlier narrative sentence. A same-line
            # conclusion is accepted only when it has a type-compatible payload.
            tail = text[match.end() : match.end() + MAX_PAYLOAD_CHARS + 1].lstrip(" \t:*_\n")
            payload = tail.split("\n", 1)[0].strip()
            if "\n" not in tail and match.end() + MAX_PAYLOAD_CHARS < len(text):
                payload = ""  # Never accept a prefix cut from a looping/oversized answer.
            # The local phrase 'option C' can have a trailing sentence terminator.
            payload = payload.rstrip("।۔ ")
            # A box on the payload line carries stronger, more precise boundaries.
            if r"\boxed" in payload:
                continue
            if pattern is LOCAL_MARKER:
                # Local answer vocabulary also occurs in question/explanation
                # prose. An oversized one-line loop is never a conclusion; for
                # normal outputs retain a parseable trailing option/value.
                has_relation = ":" in match[0] or "=" in match[0] or any(word in match[0] for word in COPULAS)
                parsed_payload = payload_answer(payload, kind, normalize, question=question)
                if (":" not in match[0] and "=" not in match[0] and parsed_payload is None) or (
                    not has_relation and len(text) > MAX_PAYLOAD_CHARS and "\n" not in text
                ):
                    continue
            if pattern is ENGLISH_MARKER and payload_answer(payload, kind, normalize, question=question) is None:
                prefix = text[text.rfind("\n", 0, match.start()) + 1 : match.start()].strip()
                # Narrative references such as 'the answer is consistent' do not
                # replace an explicit answer. An actual Final Answer marker does.
                if ":" not in match[0] and "=" not in match[0] and "final" not in match[0].lower() and prefix:
                    continue
            candidates.append((match.start(), payload, "answer_marker"))
            break
    lines = [line.strip() for line in text.splitlines() if line.strip() and line.strip() not in {"---", "```"}]
    if lines:
        # A trailing standalone value can follow a heading or prose explanation.
        answer = (
            normalize(lines[-1], kind)
            if kind.startswith("MCQ")
            else payload_answer(lines[-1], kind, normalize, question=question)
        )
        # A wrapped prose sentence ending in e.g. 'for\nn=3' does not make
        # n the requested answer. Only an isolated final equation is eligible.
        last_paragraph = re.split(r"\n\s*\n", text.strip())[-1].strip()
        if kind in {"Numeric", "Integer"} and normalize(lines[-1], kind) is None and last_paragraph != lines[-1]:
            answer = None
        if answer is not None:
            candidates.append((text.rfind(lines[-1]), lines[-1], "bare_final_line"))
    if candidates:
        _, payload, method = max(candidates, key=lambda item: item[0])
        answer = payload_answer(payload, kind, normalize, question=question)
        if answer is None and kind == "MCQ" and question:
            answer = match_option_text(payload, question)
            if answer is not None:
                method = "option_text_match"
        return (answer, method) if answer is not None else ("None", "unparsed")
    if not lines:
        return "None", "unparsed"
    verdict = option_verdicts(text, kind)
    if verdict is not None:
        return verdict, "complete_option_verdicts"
    final = lines[-1]
    last_paragraph = re.split(r"\n\s*\n", text.strip())[-1].strip()
    if kind in {"MCQ", "MCQ(multiple)"}:
        verdict_lines = re.findall(r"(?m)^\s*[\[(][A-Da-d][\])]\s*", last_paragraph)
        # Multiple per-option verdict lines must classify all four options via
        # option_verdicts; never select only the last positive line.
        if len(verdict_lines) <= 1:
            answer = explicit_choices(last_paragraph, kind, normalize)
            if answer is not None:
                return answer, "concluding_choices"
    else:
        has_conclusion = bool(CONCLUSION_PATTERN.search(last_paragraph))
        has_value_relation = bool(re.search(rf"(?:{VALUE_PATTERN})", last_paragraph, re.I))
        if has_conclusion or has_value_relation:
            answer = explicit_scalar(last_paragraph, kind, normalize, allow_assignment=has_conclusion)
            if answer is not None:
                return answer, "concluding_value"
    answer = normalize(final, kind)
    if answer is not None:
        return answer, "bare_final_line"
    # Simple English conclusions with an explicit scalar, rather than arbitrary
    # last-number heuristics. Unsupported prose is left for manual extraction.
    if kind in {"Numeric", "Integer"}:
        match = re.search(
            r"(?i)\b(?:value|answer|result|minimum|maximum|product|sum|number|mass)\b[^\n]*?\bis\s+(.+)$", final
        )
        if match:
            answer = payload_answer(match[1], kind, normalize, question=question)
            if answer is not None:
                return answer, "concluding_value"
        # A final 'Thus, r^2 = 2' is a stated value, not a request to solve r.
        conclusion = re.sub(r"(?i)^(?:thus|hence|therefore|so|अतः|इसलिए)\s*[,，:]?\s*", "", final)
        if conclusion != final:
            answer = payload_answer(conclusion, kind, normalize, question=question)
            if answer is not None:
                return answer, "concluding_value"
    return "None", "unparsed"
