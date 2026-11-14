# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grade answer correctness and compliance with the requested format."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import TypedDict

from resources_servers.pdf_analyst_qa.task_data import PdfAnalystQaTask
from resources_servers.pdf_analyst_qa.verifiers.answer_formats import parse_answer


class AnswerGrade(TypedDict):
    """Outcome correctness and requested-format compliance are separate checks."""

    semantic_correct: bool
    format_compliant: bool
    value: str


def _number(value: str) -> Decimal | None:
    text = value.strip().replace("−", "-")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1].strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON keys")
        result[key] = value
    return result


def _value(value: str):
    text = value.strip().replace("−", "-")
    try:
        parsed = json.loads(text, parse_int=Decimal, parse_float=Decimal, object_pairs_hook=_unique_keys)
    except ValueError:
        # Benchmark lists can contain unquoted labels and thousands separators.
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1]
            return [_value(part) for part in inner.split(",")]
        return text
    return parsed


def _equal(left, right, key: str | None = None) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k], k) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal(a, b, key) for a, b in zip(left, right, strict=True))
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if left is None or right is None:
        return left is right
    if not isinstance(left, (str, Decimal)) or not isinstance(right, (str, Decimal)):
        return False
    # Never reinterpret identifiers or unit labels as numbers.
    numeric = key not in {"entity", "period", "group", "unit", "units", "label"}
    ln, rn = _number(str(left)), _number(str(right))
    if numeric and ln is not None and rn is not None:
        return ln == rn
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    a, b = " ".join(left.split()), " ".join(right.split())
    if key in {"entity", "period"}:
        return a.casefold() == b.casefold()
    if a == b:
        return True
    # Numeric formatting may differ while the same explicit unit is retained.
    lm = re.fullmatch(r"(.+?)\s*(%|[A-Za-z][A-Za-z ]*)", a)
    rm = re.fullmatch(r"(.+?)\s*(%|[A-Za-z][A-Za-z ]*)", b)
    if lm and rm and lm[2] == rm[2]:
        ln, rn = _number(lm[1]), _number(rm[1])
        if ln is not None and rn is not None:
            return ln == rn
    if key in {"unit", "units"}:
        return False
    if lm and _number(lm[1]) is not None or rm and _number(rm[1]) is not None:
        return False
    return a.casefold() == b.casefold() or {a, b} == {"*", "asterisk"}


def answer_is_equivalent(candidate: str, reference: str) -> bool:
    """Compare complete results, preserving order, labels, units and extra claims."""
    try:
        json.loads(candidate, object_pairs_hook=_unique_keys)
    except json.JSONDecodeError:
        pass
    except ValueError:
        return False
    return _equal(_value(candidate), _value(reference))


def _unwrapped(response: str) -> str:
    text = response.strip()
    tagged = re.fullmatch(r"(?:<REASONING>.*?</REASONING>\s*)?<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", text, re.DOTALL)
    if tagged:
        text = tagged[1].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        text = fenced[1].strip()
    return text


def _string_values_follow_reference(observed, expected) -> bool:
    if isinstance(observed, dict) and isinstance(expected, dict):
        return all(
            _string_values_follow_reference(observed[k], value) for k, value in expected.items() if k in observed
        )
    if isinstance(observed, list) and isinstance(expected, list):
        return all(_string_values_follow_reference(a, b) for a, b in zip(observed, expected, strict=False))
    return not (isinstance(expected, str) and _number(expected) is not None and isinstance(observed, Decimal))


def _grade_list_answer(task, candidate: str, reference: str) -> tuple[bool, bool]:
    """Use generated slot roles so numeric-looking labels never become numbers."""
    try:
        observed = json.loads(candidate, parse_float=Decimal, object_pairs_hook=_unique_keys)
        expected = json.loads(reference, parse_float=Decimal, object_pairs_hook=_unique_keys)
    except ValueError:
        return False, False
    program = getattr(task, "answer_program", None)
    parameters = next(
        (node.parameters for node in getattr(program, "nodes", []) if "compact_columns" in node.parameters),
        None,
    )
    if parameters is None:
        # These tasks are generated with slot metadata; do not guess missing roles.
        return False, False
    plan = parameters["task"]
    shape = plan["answer_shape"]
    names = plan["outputs"]
    observed_parts = observed if shape == "multi_part" else [observed]
    expected_parts = expected if shape == "multi_part" else [expected]
    if not isinstance(observed_parts, list) or len(observed_parts) != len(names):
        return False, False
    semantic, compliant = True, True
    for name, left, right in zip(names, observed_parts, expected_parts, strict=True):
        columns = parameters["compact_columns"][name]
        left_rows, right_rows = ([left], [right]) if shape == "entity_value" else (left, right)
        if not isinstance(left_rows, list) or len(left_rows) != len(right_rows):
            return False, False
        for row, gold in zip(left_rows, right_rows, strict=True):
            cells, gold_cells = (row, gold) if columns else ([row], [gold])
            if not isinstance(cells, list) or len(cells) != len(columns) + 1:
                return False, False
            for role, cell, target in zip([*columns, "value"], cells, gold_cells, strict=True):
                if role in {"entity", "period", "group"}:
                    semantic &= isinstance(cell, str) and _equal(cell, target, role)
                    compliant &= isinstance(cell, str)
                else:
                    semantic &= type(cell) is not bool and _equal(str(cell), str(target), role)
                    compliant &= (
                        type(cell) is int
                        if role == "rank"
                        else (type(cell) is int or isinstance(cell, Decimal) and cell.is_finite())
                    )
    return semantic, compliant


def grade_task_answer(task: PdfAnalystQaTask, response: str) -> AnswerGrade:
    """Keep semantic correctness independent of the explicitly requested format."""
    reference = parse_answer(task.final_answer, answer_format=task.answer_contract, expected_unit=task.units).value
    try:
        json.loads(_unwrapped(response), object_pairs_hook=_unique_keys)
    except json.JSONDecodeError:
        pass
    except ValueError:
        return {"semantic_correct": False, "format_compliant": False, "value": response}
    try:
        observed = parse_answer(response, answer_format=task.answer_contract, expected_unit=task.units)
    except ValueError:
        compliant = False
        value = _unwrapped(response)
        # Parse conventional wrappers without accepting arbitrary accompanying prose.
        for answer_format in ("value_with_unit", "json_object", "value_with_citation"):
            try:
                parsed = parse_answer(value, answer_format=answer_format, expected_unit=task.units)
            except ValueError:
                continue
            value = parsed.value
            break
    else:
        compliant, value = True, observed.value
        if task.answer_contract == "value_with_citation":
            expected = parse_answer(task.final_answer, answer_format=task.answer_contract, expected_unit=task.units)
            compliant = observed.citations == expected.citations
    if _unwrapped(response) != response.strip():
        compliant = False if task.answer_contract != "tagged" else compliant
    value = _unwrapped(value)
    if task.answer_contract == "structured_json" and re.search(
        r"(?:numeric|numerical) values (?:as|must be) (?:JSON )?strings"
        r"|\b(?:write\s+)?value\s+as\s+a\s+decimal\s+string\b",
        getattr(task, "question", ""),
        re.IGNORECASE,
    ):
        compliant = compliant and _string_values_follow_reference(_value(value), _value(reference))
    if task.units and task.answer_contract == "bare_value":
        try:
            value = parse_answer(value, answer_format="value_with_unit", expected_unit=task.units).value
        except ValueError:
            pass
    semantic = answer_is_equivalent(value, reference)
    if task.answer_contract == "bracket_list":
        semantic, slot_compliance = _grade_list_answer(task, value, reference)
        compliant = compliant and slot_compliance
    elif _number(reference) is not None and _number(value) is None:
        # A numerical slot cannot contain an unparsed unit or currency marker.
        compliant = False
    return {"semantic_correct": semantic, "format_compliant": compliant, "value": value}
