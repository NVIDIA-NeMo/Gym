# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded mathematical syntax, constructed directly into SymPy. In grading it runs only inside the grading
sandbox of whichever provider the server is configured with, never in the server process.

Grammar: decimal numbers, ASCII names, parentheses, + - * / ** (or ^), unary signs, bounded rational
exponents, Rational(int, int), boolean literals, comma-separated tuples, the function table below, an
undefined function application name(args) built as sympy.Function(name)(*args), and the Interval
constructors Interval(a, b) and Interval.open/closed/Lopen/Ropen(a, b). i/I denote the imaginary unit.
No Python evaluator, namespace lookup, or input-supplied callable is used: an undefined function name binds
a fresh sympy.Function, never a library object, and member access parses only for those Interval constructors.

Unsupported notation is a representation error. For an expected/reference value it is preflight-unscorable,
never a candidate mismatch. An externally enforced remote process deadline is still required. Equivalence uses
algebraic identities and bounded deterministic rewrites as proof; fixed positive real sample points only disprove
(complex ones only for a symbol with an explicit non-real assumption: declared non-real, or complex of unknown
reality). An undecided pair stays None, never guessed.
"""

import ast
import io
import re
import tokenize
from decimal import Decimal
from fractions import Fraction

import sympy as sp
from sympy.core.evalf import PrecisionExhausted

from .codec import RepresentationError
from .task_data import DECIMAL_PATTERN


MAX_EXPRESSION_BYTES = 4096
# Sized to admit a long written-out polynomial, which runs to a few hundred tokens. The remote deadline is still
# the ultimate bound, and the term and growth budgets below still apply.
MAX_SYNTAX_NODES = 2048
# A written-out polynomial parses LEFT-nested, so its AST depth equals its term count. 512 admits a long
# polynomial while the node budget above caps total size. It is also a recursion guard: build()
# recurses once per level, and against SymPy 1.14 build() recurses safely well past this depth.
MAX_SYNTAX_DEPTH = 512
MAX_LITERAL_DIGITS = 128
MAX_LITERAL_EXPONENT = 128
MAX_POWER = 32
MAX_SYMBOLS = 32
MAX_TERMS = 512
MAX_RESULT_NODES = 2048
MAX_ARITHMETIC_BITS = 4096
MAX_FUNCTION_DEPTH = 4
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_CONSTANTS = {
    "pi": sp.pi,
    "E": sp.E,
    "i": sp.I,
    "I": sp.I,
    "GoldenRatio": sp.GoldenRatio,
    "EulerGamma": sp.EulerGamma,
    "Catalan": sp.Catalan,
}
_FUNCTIONS = {
    "sqrt": (sp.sqrt, 1, 1),
    "sin": (sp.sin, 1, 1),
    "cos": (sp.cos, 1, 1),
    "tan": (sp.tan, 1, 1),
    "cot": (sp.cot, 1, 1),
    "sec": (sp.sec, 1, 1),
    "csc": (sp.csc, 1, 1),
    "asin": (sp.asin, 1, 1),
    "acos": (sp.acos, 1, 1),
    "atan": (sp.atan, 1, 1),
    "atan2": (sp.atan2, 2, 2),
    "sinh": (sp.sinh, 1, 1),
    "cosh": (sp.cosh, 1, 1),
    "tanh": (sp.tanh, 1, 1),
    "asinh": (sp.asinh, 1, 1),
    "acosh": (sp.acosh, 1, 1),
    "atanh": (sp.atanh, 1, 1),
    "exp": (sp.exp, 1, 1),
    "log": (sp.log, 1, 2),
    "ln": (sp.log, 1, 2),
    "Abs": (sp.Abs, 1, 1),
    "abs": (sp.Abs, 1, 1),
    "conjugate": (sp.conjugate, 1, 1),
    "re": (sp.re, 1, 1),
    "im": (sp.im, 1, 1),
    "sign": (sp.sign, 1, 1),
    "erf": (sp.erf, 1, 1),
    "erfc": (sp.erfc, 1, 1),
    "gamma": (sp.gamma, 1, 1),
    "loggamma": (sp.loggamma, 1, 1),
    # The Riemann/Hurwitz zeta function. Binding it as the library function keeps a numeric verdict
    # (for example "zeta(4)/zeta(2)") instead of leaving it undecided.
    "zeta": (sp.zeta, 1, 2),
    "factorial": (sp.factorial, 1, 1),
    "besselj": (sp.besselj, 2, 2),
    "bessely": (sp.bessely, 2, 2),
}
# The one member-access form that parses: a SymPy Interval constructor such as Interval.open(a, b). Each
# constructor name maps to the (left_open, right_open) flags. No other attribute access parses.
_INTERVAL_BOUNDS = {
    "open": (True, True),
    "closed": (False, False),
    "Lopen": (True, False),
    "Ropen": (False, True),
}


def _number(text: str) -> Fraction:
    if not DECIMAL_PATTERN.fullmatch(text):
        raise RepresentationError("symbolic_number")
    mantissa, *parts = re.split("[eE]", text)
    exponent = parts[0] if parts else "0"
    if (
        sum(char.isdigit() for char in mantissa) > MAX_LITERAL_DIGITS
        or len(exponent.lstrip("+-")) > 3
        or abs(int(exponent)) > MAX_LITERAL_EXPONENT
    ):
        raise RepresentationError("symbolic_number_limit")
    return Fraction(Decimal(text))


def _syntax(text: str) -> tuple[str, ast.Expression]:
    if type(text) is not str or len(text) > MAX_EXPRESSION_BYTES:
        raise RepresentationError("symbolic_byte_limit")
    try:
        if len(text.encode("utf-8")) > MAX_EXPRESSION_BYTES:
            raise RepresentationError("symbolic_byte_limit")
    except UnicodeError as exc:
        raise RepresentationError("invalid_unicode") from exc
    text = text.strip().replace("^", "**")
    try:
        for index, token in enumerate(tokenize.generate_tokens(io.StringIO(text).readline)):
            if index > MAX_SYNTAX_NODES:
                raise RepresentationError("symbolic_syntax_limit")
            if token.type == tokenize.NUMBER:
                _number(token.string)
            elif token.type == tokenize.NAME:
                if not _NAME.fullmatch(token.string) or "__" in token.string:
                    raise RepresentationError("symbolic_name")
            elif token.type == tokenize.OP:
                # "." is admitted only for the Interval constructor member access. The AST
                # builder rejects every other attribute form, so the token set stays bounded.
                if token.string not in ("+", "-", "*", "/", "**", "(", ")", ",", "."):
                    raise RepresentationError("symbolic_operator")
            elif token.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER):
                raise RepresentationError("symbolic_syntax")
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, tokenize.TokenError, RecursionError, ValueError) as exc:
        if isinstance(exc, RepresentationError):
            raise
        raise RepresentationError("symbolic_syntax") from exc
    pending = [(tree, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > MAX_SYNTAX_NODES or depth > MAX_SYNTAX_DEPTH:
            raise RepresentationError("symbolic_syntax_limit")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return text, tree


def _bounded_result(expr: sp.Expr) -> sp.Expr:
    for index, node in enumerate(sp.preorder_traversal(expr)):
        if index >= MAX_RESULT_NODES:
            raise RepresentationError("symbolic_growth_limit")
        if isinstance(node, sp.Rational):
            if max(int(node.p).bit_length(), int(node.q).bit_length()) > MAX_ARITHMETIC_BITS:
                raise RepresentationError("symbolic_growth_limit")
    if expr.has(sp.nan, sp.oo, -sp.oo, sp.zoo):
        raise RepresentationError("nonfinite_expression")
    return expr


def parse_expression(text: str) -> sp.Expr:
    """Parse and normalize one bounded expression. No original assumptions survive."""
    text, tree = _syntax(text)
    symbols = {}

    def literal(node: ast.AST) -> Fraction | None:
        if type(node) is ast.Constant and type(node.value) in (int, float):
            return _number(ast.get_source_segment(text, node))
        if type(node) is ast.UnaryOp and type(node.op) in (ast.UAdd, ast.USub):
            value = literal(node.operand)
            return -value if value is not None and type(node.op) is ast.USub else value
        if type(node) is ast.BinOp and type(node.op) is ast.Div:
            left, right = literal(node.left), literal(node.right)
            if right == 0:
                raise RepresentationError("nonfinite_expression")
            if left is not None and right is not None:
                bits = max(left.numerator.bit_length(), left.denominator.bit_length())
                bits += max(right.numerator.bit_length(), right.denominator.bit_length())
                if bits > MAX_ARITHMETIC_BITS:
                    raise RepresentationError("symbolic_growth_limit")
                return left / right
        return None

    def budget(terms: int, bits: int) -> None:
        if terms > MAX_TERMS or bits > MAX_ARITHMETIC_BITS:
            raise RepresentationError("symbolic_growth_limit")

    def interval(bounds: tuple[bool, bool], arg_nodes: list, function_depth: int) -> tuple[sp.Expr, int, int]:
        # A SymPy Interval constructor. Two operands only. The flags say which ends are open.
        if function_depth >= MAX_FUNCTION_DEPTH:
            raise RepresentationError("symbolic_function_limit")
        if len(arg_nodes) != 2:
            raise RepresentationError("symbolic_arity")
        built = [build(child, function_depth + 1) for child in arg_nodes]
        terms, bits = sum(part[1] for part in built), sum(part[2] for part in built)
        budget(terms, bits)
        return sp.Interval(built[0][0], built[1][0], bounds[0], bounds[1]), terms, bits

    def build(node: ast.AST, function_depth: int = 0) -> tuple[sp.Expr, int, int]:
        number = literal(node)
        if number is not None:
            bits = max(number.numerator.bit_length(), number.denominator.bit_length())
            budget(1, bits)
            return sp.Rational(number.numerator, number.denominator), 1, bits
        if type(node) is ast.Constant and type(node.value) is bool:
            # A boolean literal: the condition of a Piecewise's catch-all branch, and nothing else.
            return (sp.true if node.value else sp.false), 1, 1
        if type(node) is ast.Tuple:
            # A tuple groups a (value, condition) pair for Piecewise-shaped notation. Empty is illegal.
            if not node.elts:
                raise RepresentationError("symbolic_syntax")
            parts = [build(child, function_depth) for child in node.elts]
            terms, bits = sum(part[1] for part in parts), sum(part[2] for part in parts)
            budget(terms, bits)
            return sp.Tuple(*(part[0] for part in parts)), terms, bits
        if type(node) is ast.Name:
            name = node.id
            if name in ("nan", "oo", "zoo", "inf", "Infinity"):
                raise RepresentationError("nonfinite_expression")
            if name in _CONSTANTS:
                return _CONSTANTS[name], 1, 1
            if name not in symbols:
                if len(symbols) >= MAX_SYMBOLS:
                    raise RepresentationError("symbolic_symbol_limit")
                symbols[name] = sp.Symbol(name)
            return symbols[name], 1, 1
        if type(node) is ast.UnaryOp and type(node.op) in (ast.UAdd, ast.USub):
            value, terms, bits = build(node.operand, function_depth)
            if type(node.op) is ast.USub:
                value = sp.Mul(sp.S.NegativeOne, value, evaluate=False)
            return value, terms, bits
        if type(node) is ast.BinOp:
            left, left_terms, left_bits = build(node.left, function_depth)
            if type(node.op) is ast.Pow:
                power = literal(node.right)
                if power is None or abs(power.numerator) > MAX_POWER or power.denominator > MAX_POWER:
                    raise RepresentationError("symbolic_exponent_limit")
                terms = left_terms ** max(1, abs(power.numerator))
                bits = left_bits * max(1, abs(power.numerator), power.denominator)
                budget(terms, bits)
                if left == 0 and power <= 0:
                    raise RepresentationError("nonfinite_expression")
                return sp.Pow(left, sp.Rational(power.numerator, power.denominator), evaluate=False), terms, bits
            right, right_terms, right_bits = build(node.right, function_depth)
            if type(node.op) in (ast.Add, ast.Sub):
                terms, bits = left_terms + right_terms, max(left_bits, right_bits) + 1
                budget(terms, bits)
                if type(node.op) is ast.Sub:
                    right = sp.Mul(sp.S.NegativeOne, right, evaluate=False)
                return sp.Add(left, right, evaluate=False), terms, bits
            if type(node.op) in (ast.Mult, ast.Div):
                terms, bits = left_terms * right_terms, left_bits + right_bits
                budget(terms, bits)
                if type(node.op) is ast.Div:
                    if right == 0:
                        raise RepresentationError("nonfinite_expression")
                    right = sp.Pow(right, sp.S.NegativeOne, evaluate=False)
                return sp.Mul(left, right, evaluate=False), terms, bits
            raise RepresentationError("symbolic_operator")
        if type(node) is ast.Call and not node.keywords:
            if function_depth >= MAX_FUNCTION_DEPTH:
                raise RepresentationError("symbolic_function_limit")
            # Member access parses only for the Interval constructors. Every other attribute
            # form is rejected, so admitting "." to the token set opens no general access path.
            if type(node.func) is ast.Attribute:
                target, method = node.func.value, node.func.attr
                if type(target) is ast.Name and target.id == "Interval" and method in _INTERVAL_BOUNDS:
                    return interval(_INTERVAL_BOUNDS[method], node.args, function_depth)
                raise RepresentationError("symbolic_operator")
            if type(node.func) is not ast.Name:
                raise RepresentationError("symbolic_syntax")
            name = node.func.id
            if name == "Rational" and len(node.args) == 2:
                left, right = literal(node.args[0]), literal(node.args[1])
                if left is None or right is None or left.denominator != 1 or right.denominator != 1 or right == 0:
                    raise RepresentationError("symbolic_rational")
                number = left / right
                bits = max(number.numerator.bit_length(), number.denominator.bit_length())
                budget(1, bits)
                return sp.Rational(number.numerator, number.denominator), 1, bits
            if name == "Interval":
                # A bare Interval(a, b) is the closed interval.
                return interval((False, False), node.args, function_depth)
            args = [build(child, function_depth + 1) for child in node.args]
            terms, bits = sum(arg[1] for arg in args), sum(arg[2] for arg in args)
            budget(terms, bits)
            if name in _FUNCTIONS:
                function, minimum, maximum = _FUNCTIONS[name]
                if not minimum <= len(node.args) <= maximum:
                    raise RepresentationError("symbolic_arity")
                return function(*(arg[0] for arg in args), evaluate=False), terms, bits
            # A name not in the table is an undefined function application built as sympy.Function(name)(*args),
            # so an application such as tr(rho) stays an opaque symbolic term instead of being rejected. A fresh
            # Function is bound, never a library object, and at least one argument is required.
            if not node.args:
                raise RepresentationError("symbolic_arity")
            return sp.Function(name)(*(arg[0] for arg in args), evaluate=False), terms, bits
        raise RepresentationError("symbolic_syntax")

    expression, _, _ = build(tree.body)
    _bounded_result(expression)
    return _simplified(expression)


def _simplified(expression: sp.Expr) -> sp.Expr:
    """Simplify within the growth bound, or keep the bounded raw expression.

    A ``simplify`` that explodes past the node or bit bound, or runs out of memory, is abandoned, not fatal,
    so a constant expression stays comparable and ``equivalent`` falls back to a numeric check. A non-finite
    result still raises, from ``_bounded_result``, and a timeout propagates to the caller.
    """
    try:
        simplified = sp.simplify(expression)
    except RepresentationError:
        raise
    except TimeoutError:
        raise  # the per-case timer belongs to the caller, never a failed simplify
    except Exception:  # memory exhaustion included: the bounded raw expression is kept
        return expression
    try:
        return _bounded_result(simplified)
    except RepresentationError as exc:
        if exc.code == "symbolic_growth_limit":
            return expression
        raise


# Numeric-equivalence fallback (below). Bounded and fully deterministic: no
# randomness, a fixed count of rational sample points, and evalf at a fixed
# precision, so a comparison never depends on machine or run.
_NUMERIC_FALLBACK_RTOL = Fraction(1, 10**9)  # default relative tolerance when the case states none
_NUMERIC_FALLBACK_ATOL = Fraction(1, 10**30)  # absolute floor, far above the evalf residue below
_NUMERIC_FALLBACK_DIGITS = 50  # evalf working precision (decimal digits)
_NUMERIC_SAMPLE_COUNT = 8  # sample points per free symbol
_NUMERIC_MIN_DECIDED = 4  # decided sample points required to accept when symbols are free
# Deterministic sample values, non-integer rationals of both signs: _numeric_equivalent uses them signed, and the
# disproof points below use their magnitudes. The values avoid 0, +-1 and every integer, so they miss common poles
# and branch points.
_NUMERIC_SAMPLE_BASE = (
    Fraction(3, 2),
    Fraction(-5, 2),
    Fraction(7, 3),
    Fraction(-11, 4),
    Fraction(13, 5),
    Fraction(-17, 6),
    Fraction(23, 7),
    Fraction(-29, 8),
)
# Functions whose value turns on which side of a branch point the argument falls, and conjugate, which real
# samples cannot tell from its argument. An evaluation can land on the wrong side of the point, so an undecided
# constant difference that contains one stays None rather than being accepted on its numeric value.
_BRANCH_FUNCTIONS = (
    sp.Abs,
    sp.sign,
    sp.floor,
    sp.ceiling,
    sp.Heaviside,
    sp.Piecewise,
    sp.Max,
    sp.Min,
    sp.re,
    sp.im,
    sp.arg,
    sp.conjugate,
)


def _sample_point(index: int, position: int) -> sp.Rational:
    """A distinct, deterministic rational for sample ``index`` at symbol ``position``."""
    value = _NUMERIC_SAMPLE_BASE[(index + position) % len(_NUMERIC_SAMPLE_BASE)] + Fraction(position, 17)
    return sp.Rational(value.numerator, value.denominator)


def _numeric_value(expression: sp.Expr, assignment: dict) -> sp.Float | None:
    """One finite real value of ``expression`` under ``assignment``, else None.

    None means undecidable numerically, never a guessed verdict."""
    try:
        substituted = expression.subs(assignment) if assignment else expression
        value = substituted.evalf(_NUMERIC_FALLBACK_DIGITS)
    except (TypeError, ValueError, ArithmeticError, RecursionError, AttributeError):
        return None
    if not (getattr(value, "is_number", False) and value.is_real and value.is_finite):
        return None
    return value


def _numeric_equivalent(left: sp.Expr, right: sp.Expr, rtol: Fraction = _NUMERIC_FALLBACK_RTOL) -> bool | None:
    """Decide equivalence by evaluation when algebra could not.

    A sample is decided only when both expressions reduce to a finite real value there. Returns False on the
    first decided sample whose values disagree beyond ``rtol`` plus a small floor, True when every decided
    sample agrees and enough were decided, and None otherwise. None is undecided, never a guessed verdict.
    """
    symbols = sorted(left.free_symbols | right.free_symbols, key=str)
    if len(symbols) > MAX_SYMBOLS:
        return None
    rtol_value = sp.Rational(rtol.numerator, rtol.denominator)
    atol_value = sp.Rational(_NUMERIC_FALLBACK_ATOL.numerator, _NUMERIC_FALLBACK_ATOL.denominator)
    count = _NUMERIC_SAMPLE_COUNT if symbols else 1
    required = min(_NUMERIC_MIN_DECIDED, count) if symbols else 1
    decided = 0
    for index in range(count):
        assignment = {symbol: _sample_point(index, position) for position, symbol in enumerate(symbols)}
        left_value = _numeric_value(left, assignment)
        right_value = _numeric_value(right, assignment)
        if left_value is None or right_value is None:
            continue
        if bool(abs(left_value - right_value) > rtol_value * max(abs(left_value), abs(right_value)) + atol_value):
            return False
        decided += 1
    return True if decided >= required else None


def rational_expression(number: Fraction) -> sp.Expr:
    """Bridge an already decoded real number without decimal string reparsing."""
    if max(number.numerator.bit_length(), number.denominator.bit_length()) > MAX_ARITHMETIC_BITS:
        raise RepresentationError("symbolic_growth_limit")
    return sp.Rational(number.numerator, number.denominator)


# expand_trig writes sin(N*x) as a degree-N multiple-angle polynomial before _bounded_result can see the result, so
# an angle carrying an integer multiple above this bound skips that one rewrite.
_MAX_TRIG_MULTIPLE = 32
_EXPANDED_TRIG = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc, sp.sinh, sp.cosh, sp.tanh, sp.coth, sp.sech, sp.csch)


def _bounded_expand_trig(expression: sp.Expr) -> sp.Expr:
    """``sp.expand_trig``, or ``expression`` unchanged when a trig angle has an integer multiple above the bound."""
    for function in expression.atoms(*_EXPANDED_TRIG):
        for term in sp.Add.make_args(function.args[0]):
            coefficient = term.as_coeff_Mul(rational=True)[0]
            if term.free_symbols and coefficient.is_Integer and abs(coefficient) > _MAX_TRIG_MULTIPLE:
                return expression
    return sp.expand_trig(expression)


# Deterministic rewrites that can PROVE a difference is zero. Each is bounded by _bounded_result. None of them can
# certify inequality: in equivalent() a False comes from SymPy's assumptions, the exact polynomial check, a constant
# difference's evaluation or a disproof point (positive real ones, and complex ones only under _declared_complex).
_PROOF_REWRITES = (sp.expand, sp.cancel, sp.trigsimp, _bounded_expand_trig, sp.powsimp, sp.radsimp, sp.logcombine)
# Disproof points: exact rationals with every coordinate positive. Physics quantities are positive by default, so a
# disagreement that only a negative coordinate would show (sqrt(z**2) against z) is no disproof. Points 0-7 take the
# magnitudes of the base values plus position/17, so no two symbols share a value. Point 8 puts every coordinate in
# (0, 1), where an expression such as sqrt(1 - beta**2) is real: (position + 1)/(2*position + 3), so 1/3, 2/5, 3/7...
_DISPROOF_POINT_COUNT = _NUMERIC_SAMPLE_COUNT + 1
# Imaginary parts added to the first two disproof points. These complex points are evaluated only for a symbol
# declared non-real, or complex with unknown reality (_declared_complex); parsed symbols carry no assumptions, so
# production pairs never use them.
_COMPLEX_SAMPLE_IMAG = (Fraction(1, 3), Fraction(-2, 5))


def _disproof_point(index: int, position: int) -> sp.Rational:
    """Coordinate ``position`` of disproof point ``index``: a fixed, exact, positive rational."""
    if index == _NUMERIC_SAMPLE_COUNT:
        value = Fraction(position + 1, 2 * position + 3)
    else:
        value = abs(_NUMERIC_SAMPLE_BASE[(index + position) % len(_NUMERIC_SAMPLE_BASE)]) + Fraction(position, 17)
    return sp.Rational(value.numerator, value.denominator)


def _declared_complex(left: sp.Expr, right: sp.Expr) -> bool:
    """True when a symbol carries an explicit non-real assumption: declared non-real, or complex of unknown reality."""
    symbols = left.free_symbols | right.free_symbols
    return any(symbol.is_real is False or (symbol.is_complex and symbol.is_real is None) for symbol in symbols)


def _disproof_value(expression: sp.Expr, assignment: dict, real_only: bool) -> sp.Expr | None:
    """One finite value of ``expression`` at a disproof point, else None.

    Evaluated strictly at 50 digits, so a value evalf cannot carry to full precision (PrecisionExhausted) is refused
    and the point is skipped rather than counted. ``real_only`` also refuses a non-real value (the both-real rule)."""
    try:
        substituted = expression.subs(assignment) if assignment else expression
        value = substituted.evalf(_NUMERIC_FALLBACK_DIGITS, strict=True)
    except (PrecisionExhausted, TypeError, ValueError, ArithmeticError, RecursionError, AttributeError):
        return None
    if not (getattr(value, "is_number", False) and value.is_finite and (value.is_real or not real_only)):
        return None
    return value


def _samples_disagree(left: sp.Expr, right: sp.Expr, rtol: Fraction = _NUMERIC_FALLBACK_RTOL) -> bool:
    """True when a fixed disproof point tells the two expressions apart. Never certifies equality.

    A positive real point counts only where both values are finite and real. A disagreement that only a negative
    coordinate or a complex point would show leaves the pair undecided; the complex points are evaluated only for a
    symbol declared non-real, or complex with unknown reality (_declared_complex).
    """
    symbols = sorted(left.free_symbols | right.free_symbols, key=str)
    if len(symbols) > MAX_SYMBOLS:
        return False
    rtol_value = sp.Rational(rtol.numerator, rtol.denominator)
    atol_value = sp.Rational(_NUMERIC_FALLBACK_ATOL.numerator, _NUMERIC_FALLBACK_ATOL.denominator)
    points = [
        {symbol: _disproof_point(index, position) for position, symbol in enumerate(symbols)}
        for index in range(_DISPROOF_POINT_COUNT if symbols else 1)
    ]
    checks = [(point, True) for point in points]  # (assignment, real values only)
    if _declared_complex(left, right):
        for imag in _COMPLEX_SAMPLE_IMAG:
            offset = sp.I * sp.Rational(imag.numerator, imag.denominator)
            checks.extend(({symbol: value + offset for symbol, value in point.items()}, False) for point in points[:2])
    for assignment, real_only in checks:
        left_value = _disproof_value(left, assignment, real_only)
        right_value = _disproof_value(right, assignment, real_only)
        if left_value is None or right_value is None:
            continue
        if bool(abs(left_value - right_value) > rtol_value * max(abs(left_value), abs(right_value)) + atol_value):
            return True
    return False


def equivalent(left: sp.Expr, right: sp.Expr) -> bool | None:
    """Equivalence by proof, disproof at fixed positive real points, else None.

    True only when the difference is structurally zero, simplifies to zero, is decided zero by SymPy's assumptions,
    is the zero polynomial over an exact coefficient domain, or one bounded deterministic rewrite reduces it to zero;
    a constant difference may also be decided by high-precision evaluation. False when the assumptions decide it, the
    exact polynomial test does, or a disproof point gives two different finite real values (_samples_disagree).
    Anything else is None: undecided, never guessed. A timeout propagates to the caller; memory exhaustion in one
    rewrite is only a failed attempt.
    """
    if left == right:
        return True
    try:
        difference = _bounded_result(sp.simplify(left - right))
    except RepresentationError as exc:
        if exc.code == "symbolic_growth_limit":
            return False if _samples_disagree(left, right) else None
        raise
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return False if _samples_disagree(left, right) else None
    if difference == 0 or difference.is_zero is True:
        return True
    if difference.is_zero is False:
        return False
    symbols = sorted(difference.free_symbols, key=lambda symbol: symbol.name)
    try:
        if symbols and difference.is_polynomial(*symbols):
            polynomial = sp.Poly(difference, *symbols)
            domain = polynomial.get_domain()
            # Only exact coefficient arithmetic gives a verdict: ZZ, QQ, the Gaussian integers or rationals, or an
            # algebraic field. Over a ring of transcendental constants a zero value can look nonzero
            # (x*(atan(1/2) + atan(1/3) - pi/4)), so such a difference goes on to the samples.
            if domain.is_ZZ or domain.is_QQ or domain.is_ZZ_I or domain.is_QQ_I or domain.is_AlgebraicField:
                return bool(polynomial.is_zero)
    except (TypeError, ValueError, AttributeError, NotImplementedError, sp.PolynomialError):
        pass  # not a polynomial for this purpose
    if not symbols:
        if difference.has(*_BRANCH_FUNCTIONS):
            return None
        # A constant difference: 50-digit evaluation decides within rtol 1e-9, at one point when the operands are
        # constants, or at the eight sample points (at least four decided) when their free symbols cancel.
        return _numeric_equivalent(left, right)
    if _samples_disagree(left, right):  # a cheap disproof before the rewrites saves the per-case budget
        return False
    for rewrite in _PROOF_REWRITES:
        try:
            if _bounded_result(rewrite(difference)) == 0:
                return True
        except TimeoutError:
            raise  # the per-case timer belongs to the caller, never a failed rewrite
        except Exception:  # memory exhaustion included: a failed rewrite, and the next one runs
            continue
    return None
