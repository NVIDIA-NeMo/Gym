# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned MathArena parser with an arithmetic-only replacement for string sympify.

The upstream source is trusted, hash-checked code. Model-produced expressions are
never passed to Python eval, exec, sympify, or parse_expr. Unsafe expression syntax
raises a BaseException subclass so upstream's broad ``except Exception`` blocks
cannot silently turn a security restriction into the last-integer fallback.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import sympy


UPSTREAM_REVISION = "b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b"
SOURCE_HASHES = {
    "parser.py.txt": "e32c1a04dd6797c13a6e60281a3ec493cb68139d77b52930068ad3712890c44a",
    "parse_manual.py.txt": "8858daf09f0b8fda2326853874db13fe32d03b506464edcd8f96215db59a3936",
}
_VENDOR = Path(__file__).parent / "_vendor"
_FUNCTIONS = {
    "sqrt": sympy.sqrt,
    "binomial": sympy.binomial,
    "floor": sympy.floor,
    "ceiling": sympy.ceiling,
    "ceil": sympy.ceiling,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "cot": sympy.cot,
    "sec": sympy.sec,
    "csc": sympy.csc,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "exp": sympy.exp,
    "log": sympy.log,
    "ln": sympy.log,
    "Abs": sympy.Abs,
    "abs": sympy.Abs,
    "factorial": sympy.factorial,
    "Mod": sympy.Mod,
    "Min": sympy.Min,
    "Max": sympy.Max,
}
_CONSTANTS = {"pi": sympy.pi, "E": sympy.E, "e": sympy.E, "I": sympy.I, "oo": sympy.oo}


class UnsafeMathExpression(BaseException):
    """The expression exceeds the explicit safe arithmetic grammar or limits."""


def safe_sympify(expression: object, **_: object) -> object:
    """Construct SymPy arithmetic directly from a bounded AST, without Python evaluation.

    Syntax that is merely LaTeX is allowed to reach upstream's ANTLR fallback.
    Python attributes, subscripts, comprehensions, strings, lambda, imports and
    arbitrary calls are rejected, even when embedded in an otherwise valid sum.
    """
    if not isinstance(expression, str):
        # Upstream calls this hook only with strings. SymPy objects do not need
        # reparsing; accepting arbitrary Python objects would broaden the boundary.
        if isinstance(expression, sympy.Basic | int | float):
            return expression
        raise UnsafeMathExpression("Only strings and already parsed arithmetic values are supported")
    if len(expression) > 8192:
        raise UnsafeMathExpression("Expression exceeds 8192 characters")
    if "__" in expression or any(char in expression for char in "'\"`;"):
        raise UnsafeMathExpression("Python strings, private names and statements are not mathematical answers")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        # The pinned ANTLR LaTeX parser builds SymPy objects directly; it does not
        # call eval, exec, sympify or parse_expr on its input.
        raise ValueError("Not Python arithmetic; try the upstream LaTeX parser") from exc
    if sum(1 for _ in ast.walk(tree)) > 512:
        raise UnsafeMathExpression("Expression exceeds 512 syntax nodes")

    def construct(node: ast.AST) -> object:
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            if isinstance(node.value, float):
                if not math.isfinite(node.value):
                    raise UnsafeMathExpression("Nonfinite literal")
                return sympy.Float(node.value)
            if node.value.bit_length() > 4096:
                raise UnsafeMathExpression("Integer literal exceeds 4096 bits")
            return sympy.Integer(node.value)
        if isinstance(node, ast.Name) and re.fullmatch(r"[^\W\d_]\w*", node.id) and "_" not in node.id:
            return _CONSTANTS.get(node.id, sympy.Symbol(node.id))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            value = construct(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.Div | ast.Pow):
            left, right = construct(node.left), construct(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if right.is_number and abs(right) > 10000:
                raise UnsafeMathExpression("Exponent exceeds the arithmetic resource limit")
            return left**right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
            if node.keywords or len(node.args) > 8:
                raise UnsafeMathExpression("Only bounded positional mathematical function calls are supported")
            args = [construct(arg) for arg in node.args]
            if node.func.id in {"factorial", "binomial"} and any(arg.is_number and abs(arg) > 10000 for arg in args):
                raise UnsafeMathExpression("Combinatorial argument exceeds the arithmetic resource limit")
            return _FUNCTIONS[node.func.id](*args)
        if isinstance(node, ast.Tuple | ast.List):
            return [construct(item) for item in node.elts]
        raise UnsafeMathExpression(f"Unsupported mathematical syntax: {type(node).__name__}")

    return construct(tree.body)


class _SympyProxy:
    """Override only upstream's string parser without mutating global SymPy state."""

    sympify = staticmethod(safe_sympify)

    def __getattr__(self, name: str) -> object:
        return getattr(sympy, name)


@lru_cache(maxsize=1)
def official_parser() -> ModuleType:
    """Load hash-checked MIT upstream code with only import and sympify substitutions."""
    sources = {}
    for name, expected in SOURCE_HASHES.items():
        content = (_VENDOR / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError(f"MathArena vendored source hash mismatch: {name}")
        sources[name] = content.decode("utf-8")
    module = ModuleType("_gym_matharena_parser")
    # Only repository-owned, digest-pinned source is executed here, never model
    # output. The manual map contains fixed upstream string-to-string corrections.
    exec(compile(sources["parse_manual.py.txt"], str(_VENDOR / "parse_manual.py.txt"), "exec"), module.__dict__)
    source = sources["parser.py.txt"].replace("from matharena.parse_manual import complete_mapper, manual_mapper", "")
    source = source.replace("import sympy\n", "")
    module.sympy = _SympyProxy()
    exec(compile(source, str(_VENDOR / "parser.py.txt"), "exec"), module.__dict__)
    return module


def parse_result(
    text: str, *, strict: bool, expected_answer: int | None = None, output_tokens: int = 0
) -> dict[str, Any]:
    """Run the pinned extraction/equality path; callers isolate this in a worker."""
    parser = official_parser()
    answer, warning = parser.extract_answer(text, strict_parsing=strict)
    result = {
        "extracted_answer": str(answer) if answer is not None else None,
        "parser_warning": warning.value,
        "needs_format_retry": answer is None,
    }
    if expected_answer is not None:
        gold, _ = parser.parse_answer(str(expected_answer))
        result["reward"] = float(parser.check_answers(answer, gold))
        # Port the AIME-relevant warning augmentation from upstream grader.py.
        # The number-proximity warning is inapplicable to a <=3-digit gold answer.
        suspicious_length = output_tokens >= 1000 and output_tokens % 1000 == 0
        remaining = output_tokens
        while remaining > 1 and remaining % 10 == 0:
            remaining //= 10
        while remaining > 1 and remaining % 2 == 0:
            remaining //= 2
        suspicious_length |= output_tokens >= 1000 and remaining == 1
        if not result["reward"] and suspicious_length:
            result["parser_warning"] = 1
        elif not result["reward"] and parser.extract_answer(text, strict_parsing=True)[0] is None:
            if str(expected_answer) in re.findall(r"\d+", text):
                result["parser_warning"] = max(result["parser_warning"], 2)
        if not text and not suspicious_length:
            result["parser_warning"] = 3
    return result
