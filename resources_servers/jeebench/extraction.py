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


PARSER_VERSION = "indic-formats-v2.10"
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
    "হ’ল",
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
    "विवरण",
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
    r"or|maybe|perhaps|uncertain|unsure|possibly|probably|might|either|not|neither|incorrect|false"
    r"|cannot|(?:is|are|was|were|has|have|had|do|does|did|ca|could|wo|would|should|must|need)n['’]t"
    r"|if|unless|assum(?:e|ing|ption)|suppos(?:e|ing)|hypothetical|tentative|contradiction|inconsistent|rejected"
    r"|या|अथवा|नहीं|गलत|অথবা|ভুল|અથવા|ಅಥವಾ|അല്ലെങ്കിൽ|ଅଥବା|ਜਾਂ|அல்லது|లేదా|یا|" + NEGATIVE_PATTERN,
    re.IGNORECASE,
)


def uncertain(text: str) -> bool:
    """Match whole words, including combining marks and Indic/Arabic joiners."""

    def word_char(char: str) -> bool:
        return bool(char) and (unicodedata.category(char)[0] in "LMN" or char in "_\u200c\u200d")

    return any(
        not word_char(text[m.start() - 1] if m.start() else "")
        and not word_char(text[m.end()] if m.end() < len(text) else "")
        for m in UNCERTAIN.finditer(text)
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
    r"m|dm|cm|mm|nm|pm|km|s|ms|min|h|kg|g|mg|N|J|kJ|eV|MeV|V|A|W|Hz|C|F|H|T|M|L|l|"
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
    # Flatten integer powers before unwrapping nested legacy {\rm ...} units.
    text = re.sub(r"\^\{([+-]?\d+)\}", r"^\1", text)
    text = clean_math(text).strip().rstrip(".।۔ ")
    if re.fullmatch(NUMBER, text):
        return text
    # Strip only an explicit recognized unit suffix (no scale/unit conversion).
    for match in re.finditer(r"\s+(?=[A-Za-z°Ω\u0900-\u097f])", text):
        units = re.sub(r"\^\{([+-]?\d+)\}", r"^\1", text[match.end() :])
        if UNITS.fullmatch(units):
            text = text[: match.start()].strip()
            break
    for _ in range(8):
        text = re.sub(r"\\(dfrac|tfrac|frac)\s*(\d)\s*(\d)", r"\\\1{\2}{\3}", text)
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


def harmless_annotation(text: str) -> bool:
    """Accept complete supported annotations, never arbitrary non-negative prose."""
    return bool(
        re.fullmatch(
            r"(?:as required|(?:this|it) is the required (?:answer|value|coefficient|result|coplanar point)"
            r"|this follows from symmetry|(?:as|because) it yields a valid (?:configuration|solution|result)"
            r"|the calculation is consistent|thus the answer is consistent"
            r"|किया गया|کیا گیا)[.।۔]*",
            text.strip(),
            re.I,
        )
    )


def supporting_statement(text: str) -> bool:
    """Validate a whole supporting statement; unknown prose is not an annotation.

    This deliberately accepts only a small grammar, rather than assuming that
    the absence of a known negative word makes arbitrary surrounding text safe.
    """
    text = text.strip().strip(".,:;।۔ ")
    if not text or harmless_annotation(text):
        return True
    if re.fullmatch(rf"(?:so |therefore,? )?the (?:{VALUE_PATTERN}) is given by", text, re.I):
        return True
    relation = re.fullmatch(
        rf"(?:the (?:corresponding )?(?:{VALUE_PATTERN}|coefficient|drag) is\s+|i\.e\.\s+)"
        r"(?:[A-Za-z](?:_\{?\w+\}?)?\s*=\s*)?(.+)",
        text,
        re.I,
    )
    return bool(relation and numeric_value(relation[1]) is not None)


def supporting_context(text: str) -> bool:
    """Require every remaining line, including separate paragraphs, to be supported."""
    return all(supporting_statement(line) for line in text.splitlines())


def choice_gap(text: str) -> bool:
    """Validate a complete gap between verdicts, without deleting unknown words."""
    text = text.strip(" \n\t,;:.।۔")
    return bool(
        supporting_statement(text)
        or re.fullmatch(
            rf"(?:{'|'.join(map(re.escape, CONNECTORS))}|in particular|विशेष रूप से|only|एक\u00ad?मात्र|"
            rf"(?:the )?(?:{OPTION_PATTERN})|the calculation gives the required (?:ratio|value|result))",
            text,
            re.I,
        )
    )


def scalar_prefix(text: str, kind: str, normalize: Callable[[str, str], str | None]) -> str | None:
    """Require a complete scalar expression; never discard a numeric suffix."""
    text = clean_math(text).lstrip(" :：=≈~")
    number = numeric_value(text.removesuffix("%"))
    if number is not None:
        return normalize(number, kind)
    # Only recognized complete annotations may be discarded. Absence of a
    # known negative word is not evidence that arbitrary prose is affirmative.
    annotation = re.fullmatch(r"(.+?)(?:[.।۔]\s+|,\s+(?=as\b|which\b|because\b))([\w\s,.'’।۔-]+)", text)
    if annotation and harmless_annotation(annotation[2]):
        number = numeric_value(annotation[1])
        if number is not None:
            return normalize(number, kind)
    # Accept only these redundant annotations, not arbitrary prose or a second value.
    annotation = re.fullmatch(
        r"(.+?)[.।۔]\s*\((?:([0-9]+) is a non-negative integer|[A-Za-z]\s*=\s*([0-9]+))\.?\)", text
    )
    if annotation:
        number = numeric_value(annotation[1])
        answer = normalize(number, kind) if number is not None else None
        repeated = normalize(annotation[2] or annotation[3], kind)
        if answer is not None and repeated is not None and float(answer) == float(repeated):
            return answer
    return None


def explicit_scalar(
    text: str, kind: str, normalize: Callable[[str, str], str | None], *, allow_assignment: bool = True
) -> str | None:
    """Extract one complete value assertion; reject multiple asserted quantities."""
    if len(text) > MAX_PAYLOAD_CHARS:
        return None
    text = clean_math(text).strip()
    if uncertain(text):
        return None
    answer = scalar_prefix(text, kind, normalize)
    if answer is not None:
        return answer
    relation = re.compile(
        rf"(?:{VALUE_PATTERN})(?:\s*(?:=|≈|:)|[^\n.!?।۔=]{{0,60}}?(?:{COPULA_PATTERN}))\s*(?=[+\-\d.(\\])",
        re.I,
    )
    relations = list(relation.finditer(text))
    if relations:
        if len(relations) != 1 or re.search(NUMBER, text[: relations[0].start()]):
            return None
        return scalar_prefix(text[relations[0].end() :], kind, normalize)
    if not allow_assignment:
        return None
    assignments = list(re.finditer(r"(?<![\w])(?:[A-Za-zΑ-ω]\w*(?:_\{?\w+\}?)?)\s*(?:=|≈)\s*", text))
    if len(assignments) != 1:
        return None
    prefix = CONCLUSION_PATTERN.sub("", text[: assignments[0].start()]).strip(" ,:।۔.")
    if prefix:
        return None
    return scalar_prefix(text[assignments[0].end() :], kind, normalize)


def _roman_to_option(match: re.Match[str]) -> str:
    return {"i": "A", "ii": "B", "iii": "C", "iv": "D"}[match[1].lower()]


def choice_assertions(text: str, kind: str, normalize: Callable[[str, str], str | None]) -> str | None:
    """Account for every label in one conclusion, including negative verdicts.

    Only explicit lists and verdicts select options. Every unmatched fragment
    must be a supported connective or complete affirmative annotation.
    """
    label = r"(?:[\[(]\s*[A-Da-d]\s*[\])]|(?<![\w{\\])(?-i:[A-D])(?![\w}]))"
    connector = r"\s*(?:,\s*(?:and\s+)?|&|/|" + "|".join(map(re.escape, CONNECTORS)) + r")\s*"
    labels = label + rf"(?:{connector}{label})*"
    copula = rf"(?:(?:{COPULA_PATTERN}|being)\s+)?"
    noun = rf"(?:{OPTION_PATTERN})(?!\w)"
    verdict = rf"(?:{NEGATIVE_PATTERN}|{POSITIVE_PATTERN})(?!\w)"
    assertion = re.compile(
        rf"(?P<labels>{labels})\s+{copula}(?:in general\s+)?(?:the\s+)?(?:only\s+)?(?P<verdict>{verdict})"
        rf"(?:\s+(?:{POSITIVE_PATTERN})(?!\w))?(?:\s+{noun})?(?:\s+(?:{COPULA_PATTERN})(?!\w))?",
        re.I,
    )
    selection = re.compile(
        rf"(?:(?:{POSITIVE_PATTERN})(?!\w)\s+{noun}\s*{copula}[:：]?\s*"
        rf"|(?:(?:^|[,;]\s*)(?:the\s+)?|(?:\b(?:which is|this is|quoted as|corresponds to)|i\.e\.)"
        rf"\s+)\b(?:options?|choices?)\s*(?:(?:is|are)\s*)?[:：]?\s*)"
        rf"(?P<labels>{labels})(?:\s+(?:{COPULA_PATTERN})(?!\w))?",
        re.I,
    )
    positive, negative = set(), set()
    covered = [False] * len(text)
    for pattern in (assertion, selection):
        for match in pattern.finditer(text):
            if any(covered[match.start() : match.end()]):
                continue
            # Extract labels, not the letters of conjunctions such as 'and'.
            values = [re.sub(r"[^A-Da-d]", "", m[0]).upper() for m in re.finditer(label, match["labels"])]
            is_negative = pattern is assertion and match["verdict"].lower() in NEGATIVE_WORDS
            (negative if is_negative else positive).update(values)
            start, end = match.span()
            covered[start:end] = [True] * (end - start)
    remainder = "".join(" " if covered[i] else char for i, char in enumerate(text))
    # Keep gaps separate: joining text across a consumed assertion could create
    # an apparently harmless phrase that was never actually stated.
    gaps = "".join("\0" if covered[i] else char for i, char in enumerate(text)).split("\0")
    if (
        not positive
        or positive & negative
        or re.search(label, remainder)
        or uncertain(remainder)
        or not all(choice_gap(gap) for gap in gaps if gap.strip())
        or "?" in text
        or re.search(
            r"\b(?:if|unless|assuming|assume|suppose|whether|considered|possibly|probably|doubt|cannot|can't|don't|doesn't)\b",
            text,
            re.I,
        )
    ):
        return None
    return normalize("".join(sorted(positive)), kind)


def explicit_choices(
    text: str, kind: str, normalize: Callable[[str, str], str | None], *, selected: bool = False
) -> str | None:
    """Read a clearly selected option list, including statement numbers i--iv."""
    if kind not in {"MCQ", "MCQ(multiple)"} or len(text) > MAX_PAYLOAD_CHARS:
        return None
    text = clean_math(text)
    if not re.search(r"(?<!\w)[A-D](?!\w)", text):
        text = re.sub(r"[\[(]\s*(i|ii|iii|iv)\s*[\])]", _roman_to_option, text, flags=re.I)
    direct = normalize(text, kind)
    if direct is not None:
        return direct
    # Bare lowercase letters in prose (especially the article "a") are not labels.
    label = r"(?:[\[(]\s*[A-Da-d]\s*[\])]|(?<!\w)(?-i:[A-D])(?!\w))"
    connector = r"(?:\s*(?:,|&|\\&|/|(?:" + "|".join(map(re.escape, CONNECTORS)) + r"))\s*)"
    labels = label + r"(?:" + connector + label + r")*"

    def normalize_labels(value: str) -> str | None:
        found = re.findall(r"[\[(]\s*([A-Da-d])\s*[\])]|(?<!\w)([A-D])(?!\w)", value)
        value = "".join((left or right).upper() for left, right in found)
        return normalize(value, "MCQ(multiple)") if value else None

    text = CONCLUSION_PATTERN.sub("", text).strip(" ,:।۔.")
    text = re.sub(r"(?i)^(?:after discussion\s+)?I conclude(?: that)?\s+", "", text)
    assertions = choice_assertions(text, kind, normalize)
    if assertions is not None:
        return assertions
    # Split only between assertions, never within an option list.
    clauses = re.split(rf";|(?<=[a-z])\s*(?:,|\band\b)(?=\s*{label}\s+(?:{COPULA_PATTERN}))", text)
    yes, no = set(), set()
    for clause in clauses:
        clause = clause.strip(" ,:।۔.")
        negative = re.fullmatch(
            rf"({labels})\s+(?:(?:{COPULA_PATTERN})\s+)?(?:{NEGATIVE_PATTERN})(?!\w)"
            rf"(?:\s+(?:{POSITIVE_PATTERN}|{COPULA_PATTERN}))*[.।۔]*",
            clause,
            re.I,
        )
        if negative:
            no.update(normalize_labels(negative[1]) or "")
            continue
        if uncertain(clause):
            return None
        positive = re.fullmatch(
            rf"({labels})\s+(?:(?:{COPULA_PATTERN})\s+)?(?:{POSITIVE_PATTERN})(?!\w)"
            rf"(?:\s+(?:{COPULA_PATTERN}))*[.।۔]*",
            clause,
            re.I,
        )
        if positive:
            yes.update(normalize_labels(positive[1]) or "")
            continue
        # A positive introductory phrase must lead directly to the selected list.
        prefix = re.fullmatch(
            rf"(?:{POSITIVE_PATTERN})(?!\w)\s*(?:{OPTION_PATTERN})?\s*"
            rf"(?:(?:{COPULA_PATTERN})\s*)?[:：-]?\s*({labels})[.।۔]*",
            clause,
            re.I,
        )
        if prefix:
            yes.update(normalize_labels(prefix[1]) or "")
            continue
        # Inflected Indic nouns are allowed only before an explicitly bracketed list.
        local_prefix = re.fullmatch(
            rf"(?:{POSITIVE_PATTERN})(?!\w)[^A-Za-z0-9()\[\]\n]*\s*({labels})[.।۔]*", clause, re.I
        )
        if local_prefix:
            yes.update(normalize_labels(local_prefix[1]) or "")
            continue
        leading = re.match(rf"({labels})(.*)$", clause, re.S)
        if selected and leading:
            tail = leading[2].strip(" ,:।۔.")
            # A selected list may repeat a scalar value or statement indices,
            # or carry a supported affirmative annotation. Unknown prose must
            # not silently turn a rejection/condition into a positive selection.
            statement_indices = re.fullmatch(
                r"[\[(](?:I|II|III|IV)[\])]"
                r"(?:\s*(?:,|and|और|&)\s*[\[(](?:I|II|III|IV)[\])])*",
                tail,
            )
            if (
                not re.search(label, tail)
                and not uncertain(tail)
                and (not tail or numeric_value(tail) is not None or statement_indices or harmless_annotation(tail))
            ):
                yes.update(normalize_labels(leading[1]) or "")
                continue
        return None
    if not yes or yes & no:
        return None
    return normalize("".join(sorted(yes)), kind)


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
            left, *values = equation.split("=")
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
            if all(word in allowed for word in words) and not re.search(r"[^\x00-\x7fΑ-ω]", left):
                if len(values) > 1:
                    numbers = [numeric_value(value) for value in values]
                    if any(number is None for number in numbers) or len({float(n) for n in numbers}) != 1:
                        return None
                text = values[-1].strip()
        number = numeric_value(text)
        answer = normalize(number, kind) if number is not None else None
        return answer if answer is not None else explicit_scalar(text, kind, normalize)
    text = text.lstrip("→⇒➜➤• ")
    # Accept descriptors only at the start of an explicitly selected payload.
    text = re.sub(r"(?i)^(?:(?:the\s+)?(?:correct\s+)?(?:options?|choices?|answers?)\s*(?:are|is|:)?)\s*", "", text)
    text = re.sub(r"^(?:(?:सही|अंतिम|केवल|correct)\s+)?(?:" + LOCAL_WORDS + r"|कथन)\s*[:：]?\s*", "", text)
    text = re.sub(rf"^(?:(?:{COPULA_PATTERN})|হ'ল)\s*[:：-]?\s*", "", text, flags=re.I)
    # A label followed by the exact text of that supplied option is an explicit
    # selection, even when the option itself is a negative statement. Do not
    # accept arbitrary descriptors when the question provides no matching text.
    selected_option = re.fullmatch(r"(?:[\[(]([A-Da-d])[\])]|([A-Da-d])[).:]?)\s*[:：]?\s+(.+)", text, re.S)
    if question and selected_option:
        label = (selected_option[1] or selected_option[2]).upper()
        if match_option_text(selected_option[3], question) == label:
            return label
    return explicit_choices(text.replace(r"\&", "&"), kind, normalize, selected=True)


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


def option_verdicts(text: str, kind: str, normalize: Callable[[str, str], str | None]) -> str | None:
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
    # Complete coverage of labels does not license discarding a retraction.
    return choice_assertions(text, kind, normalize) if choices else None


def requested_symbol(question: str) -> str | None:
    """Recognize a terminal request for the value of one explicitly typeset symbol."""
    math_spans = list(re.finditer(r"\$([^$]+)\$|\\\((.*?)\\\)", question, re.S))
    if not math_spans:
        return None
    last = math_spans[-1]
    if len(math_spans) > 1:
        gap = question[math_spans[-2].end() : last.start()]
        if not re.search(r"[.!?।۔]", gap) and any(word in gap.split() for word in CONNECTORS):
            return None
    target = clean_math(last[1] or last[2])
    suffix = question[last.end() :].strip()
    prefix = question[max(0, last.start() - 50) : last.start()]
    # Ask for the symbol itself, not e.g. the determinant of a matrix, product of
    # its roots, or number of its possible values.
    value_request = re.match(
        r"(?:(?:का|के|की|को|નું|નો|ની|ৰ|এর|चा|ची|चे|ಯ|ದ|ന്റെ|ର|ਦਾ|ਦੀ|ਦੇ|இன்|யின்|کا|کی|کے)\s*)?"
        r"(?:value\b|मान|मूल्य|মান|મૂલ્ય|ಮೌಲ್ಯ|മൂല്യം|ମୂଲ୍ୟ|ਮੁੱਲ|மதிப்பு|విలువ|قدر)",
        suffix,
        re.I,
    )
    if value_request and value_request.end() < len(suffix):
        if unicodedata.category(suffix[value_request.end()])[0] in "LMN":
            value_request = None
    if (
        not re.fullmatch(r"[A-Za-z]", target)
        or len(suffix) > 120
        or not (value_request or (re.search(r"\bvalue\s+of\s*$", prefix, re.I) and re.fullmatch(r"[?.।۔\s]*", suffix)))
    ):
        return None
    return target


def requested_scalar(text: str, target: str, kind: str, normalize: Callable[[str, str], str | None]) -> str | None:
    """Read affirmative, agreeing assignments with their surrounding context."""
    # Do not clip away an earlier qualifier to fit a tail window.
    if len(text) > MAX_PAYLOAD_CHARS or uncertain(text):
        return None
    answers = []
    for line in text.splitlines():
        conclusion = CONCLUSION_PATTERN.sub("", line).strip(" ,:।۔.")
        match = re.fullmatch(rf"{target}\s*(?:=|≈|\\(?:approx|simeq)(?![A-Za-z]))\s*(.+)", conclusion)
        if not match:
            if not supporting_statement(line):
                return None
            continue
        # No prefix/last-number selection from lists, prose or symbolic equations.
        answer = payload_answer(conclusion, kind, normalize)
        if answer is None or answer == "None":
            return None
        answers.append(answer)
    return answers[0] if answers and all(float(a) == float(answers[0]) for a in answers) else None


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
        # A box delimits the value, not the subsequent verdict about that value.
        if not supporting_context(text[end:]):
            payload = ""
        candidates.append((match.start(), payload, "boxed"))
        boxed_spans.append((match.start(), end))
    for pattern in (ENGLISH_MARKER, LOCAL_MARKER):
        for match in reversed(list(pattern.finditer(text))):
            if candidates and match.start() < max(c[0] for c in candidates):
                break
            if any(start <= match.start() < end for start, end in boxed_spans):
                continue
            # Include the complete closing context, not just the marker's line.
            # Check its length before trimming so whitespace cannot hide a later
            # qualification beyond the payload limit.
            tail = text[match.end() :].lstrip(" \t:*_\n")
            payload = tail[: MAX_PAYLOAD_CHARS + 1].strip()
            if len(tail) > MAX_PAYLOAD_CHARS:
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
                # Keep an explicit leading option as the latest candidate even
                # when its annotation is unsupported. Skipping it could revive
                # an earlier (possibly hypothetical) answer instead.
                leading_option = kind.startswith("MCQ") and re.match(r"(?:[\[(][A-Da-d][\])]|[A-D](?!\w))", payload)
                if (":" not in match[0] and "=" not in match[0] and parsed_payload is None and not leading_option) or (
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
    if not candidates and lines and kind in {"Numeric", "Integer"} and normalize(lines[-1], kind) is None:
        target = requested_symbol(question)
        last_paragraph = re.split(r"\n\s*\n", text.strip())[-1].strip()
        standalone = last_paragraph == lines[-1] and numeric_value(lines[-1]) is not None
        if target and not standalone:
            answer = requested_scalar(text, target, kind, normalize)
            return (answer, "requested_variable") if answer is not None else ("None", "unparsed")
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
        if kind in {"Numeric", "Integer"} and answer is not None and numeric_value(lines[-1]) is None:
            # An unmarked equation has no explicit final-answer boundary. Its
            # context must be checked even when the question names no symbol.
            symbol = re.match(r"^([A-Za-z])\s*(?:=|≈|\\(?:approx|simeq)(?![A-Za-z]))", lines[-1])
            if symbol:
                answer = requested_scalar(text, symbol[1], kind, normalize)
            elif len(lines) > 1:
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
    verdict = option_verdicts(text, kind, normalize)
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
        prefix = text[: text.rfind(last_paragraph)]
        if has_conclusion or (has_value_relation and supporting_context(prefix)):
            answer = explicit_scalar(last_paragraph, kind, normalize, allow_assignment=has_conclusion)
            if answer is not None:
                return answer, "concluding_value"
    answer = normalize(final, kind)
    if answer is not None:
        return answer, "bare_final_line"
    # Simple English conclusions with an explicit scalar, rather than arbitrary
    # last-number heuristics. Unsupported prose is left for manual extraction.
    if kind in {"Numeric", "Integer"}:
        # A final 'Thus, r^2 = 2' is a stated value, not a request to solve r.
        conclusion = re.sub(r"(?i)^(?:thus|hence|therefore|so|अतः|इसलिए)\s*[,，:]?\s*", "", final)
        if conclusion != final:
            answer = payload_answer(conclusion, kind, normalize, question=question)
            if answer is not None:
                return answer, "concluding_value"
    return "None", "unparsed"
