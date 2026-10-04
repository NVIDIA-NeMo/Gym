# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact boxed-answer grading for the FrontierMath public sample.

This is a text-answer adaptation, not Epoch's Python submission protocol.
Symbolic parsing runs in a disposable process controlled by the resources server.
"""

import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from sympy import Expr, Rational, Symbol


MAX_ANSWER_CHARS = 4096
MAX_FINITE_SUM_TERMS = 1000
MAX_COEFFICIENT_DEGREE = 10000
MAX_POLYNOMIAL_DEGREE = 256


@dataclass
class GradeResult:
    reward: float
    extracted_answer: str | None
    grading_status: str


def extract_answer(text: str) -> str | None:
    """Extract the last box outside reasoning tags; reject an unfinished final box."""
    text = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.split(r"<(?:think|thinking)>", text, maxsplit=1, flags=re.IGNORECASE)[0]
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index].strip() or None
    return None


def normalize_digits(text: str) -> str:
    """Accept Indic decimal digits without converting exact numbers to floats."""
    text = unicodedata.normalize("NFKC", text).replace("−", "-")
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)


def _evaluate_finite_sums(expression: "Expr") -> "Expr":
    """Enumerate bounded sums instead of asking simplify() for a closed form."""
    from sympy import Add, Sum

    remaining_terms = MAX_FINITE_SUM_TERMS
    for summation in expression.atoms(Sum):
        if len(summation.limits) != 1 or summation.function.has(Sum):
            raise ValueError("Unsupported nested or multiple sum")
        variable, lower, upper = summation.limits[0]
        if not (lower.is_Integer and upper.is_Integer and 0 <= upper - lower < remaining_terms):
            raise ValueError("Finite sum exceeds verifier bound")
        remaining_terms -= int(upper - lower + 1)
        value = Add(*(summation.function.subs(variable, n) for n in range(int(lower), int(upper) + 1)))
        expression = expression.xreplace({summation: value})
    return expression


def _rational_coefficient(expression: "Expr", *, variable: "Symbol", degree: int) -> "Rational":
    """Solve Q(x) A(x) = P(x) coefficient by coefficient using exact arithmetic."""
    from sympy import Float, Poly, Rational, together

    if expression.has(Float):
        raise ValueError("Decimal approximations are not exact answers")
    numerator, denominator = together(expression).as_numer_denom()
    numerator, denominator = Poly(numerator, variable, domain="QQ"), Poly(denominator, variable, domain="QQ")
    if max(numerator.degree(), denominator.degree()) > MAX_POLYNOMIAL_DEGREE or denominator.nth(0) == 0:
        raise ValueError("Unsupported generating function")
    terms = [(i, denominator.nth(i)) for i in range(1, denominator.degree() + 1) if denominator.nth(i)]
    values = []
    for n in range(degree + 1):
        values.append((numerator.nth(n) - sum(c * values[n - i] for i, c in terms if i <= n)) / denominator.nth(0))
    return Rational(values[degree])


def parse_exact_expression(text: str) -> "Expr":
    """Parse exact scalars, including bounded rational generating-function coefficients."""
    from latex2sympy2_extended import latex2sympy
    from sympy import Symbol

    text = re.sub(r"\\(?:displaystyle|left|right|Biggl|Biggr|biggl|biggr|Bigl|Bigr|bigl|bigr)\b", "", text)
    text = re.sub(r"\\[,;!:]", "", text).strip()
    text = re.sub(r"\\sqrt\s*([0-9])", r"\\sqrt{\1}", text)
    coefficient = re.fullmatch(r"\[\s*([a-zA-Z])\s*\^\s*(?:\{(\d+)\}|(\d+))\s*\]\s*(.+)", text, re.DOTALL)
    if not coefficient:
        return _evaluate_finite_sums(latex2sympy(text))
    degree = int(coefficient[2] or coefficient[3])
    if degree > MAX_COEFFICIENT_DEGREE:
        raise ValueError("Coefficient degree exceeds verifier bound")
    return _rational_coefficient(latex2sympy(coefficient[4]), variable=Symbol(coefficient[1]), degree=degree)


def grade_answer(*, expected_answer: str, answer_type: str, generated_answer: str) -> GradeResult:
    """Compare exact scalars; never use numerical tolerance or an LLM judge."""
    extracted = extract_answer(generated_answer)
    if extracted is None:
        return GradeResult(0.0, None, "missing_answer")
    if len(extracted) > MAX_ANSWER_CHARS:
        return GradeResult(0.0, None, "answer_too_long")
    candidate = normalize_digits(extracted)
    if answer_type == "integer" and re.fullmatch(r"[+-]?\d+", candidate):
        correct = int(candidate) == int(expected_answer)
        return GradeResult(float(correct), str(int(candidate)), "correct" if correct else "incorrect")

    # Parse the entire boxed expression, not another number from the reasoning.
    # Reject floats: numerical closeness would erase the BMO answer's exponential term.
    from sympy import Expr, Float, Integer, simplify

    try:
        prediction = parse_exact_expression(candidate)
        if not isinstance(prediction, Expr) or prediction.free_symbols or prediction.has(Float):
            return GradeResult(0.0, extracted, "invalid_expression")
        gold = Integer(expected_answer) if answer_type == "integer" else parse_exact_expression(expected_answer)
        correct = simplify(prediction - gold) == 0
    except Exception:
        # This boundary processes arbitrary model text. Parse failures are failed answers.
        return GradeResult(0.0, extracted, "invalid_expression")
    return GradeResult(float(correct), str(prediction), "correct" if correct else "incorrect")


if __name__ == "__main__":
    print(json.dumps(asdict(grade_answer(**json.load(sys.stdin)))))
