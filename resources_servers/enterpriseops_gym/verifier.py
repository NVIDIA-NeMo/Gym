# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EnterpriseOps-Gym's SQL state verifiers.

A verifier runs a read query against a task database and compares the value it extracts from the answer with an
expected value. Extraction and comparison follow the upstream harness (``benchmark/verifier.py`` in
ServiceNow/EnterpriseOps-Gym): a one-row, one-column answer is compared as that value, a one-row answer as the row,
and a longer answer as the list of rows; ``equals`` is Python equality, ``greater_than``/``less_than`` are Python
ordering, and ``contains`` is a substring test on the answer's string form.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Awaitable, Callable, Dict, List

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from resources_servers.enterpriseops_gym.task_data import ComparisonType, DatabaseStateVerifier


class VerifierResult(BaseModel):
    """How one verifier came out on the final database state."""

    name: str
    gym_name: str
    passed: bool
    comparison_type: ComparisonType
    expected: JsonValue
    actual: JsonValue = None
    # Why the verifier could not compare, such as a query the database rejected.
    error: str | None = None


class QueryRejectedError(Exception):
    """The database refused a verifier's query, which fails that verifier."""


async def evaluate(
    verifiers: List[DatabaseStateVerifier], run_sql: Callable[[DatabaseStateVerifier], Awaitable[Dict[str, Any]]]
) -> List[VerifierResult]:
    """Run each verifier's query with ``run_sql``, which returns the ``/api/sql-runner`` answer, and compare its value.

    The queries run one at a time, as upstream runs them: concurrent queries on one gym database can race in the gym
    servers. An error other than ``QueryRejectedError`` propagates: without the database there is no verdict.
    """
    results = []
    for verifier in verifiers:
        config = verifier.validation_config
        result = VerifierResult(
            name=verifier.name,
            gym_name=verifier.gym_name,
            passed=False,
            comparison_type=config.comparison_type,
            expected=config.expected_value,
        )
        try:
            answer = await run_sql(verifier)
        except QueryRejectedError as error:
            results.append(result.model_copy(update={"error": str(error)}))
            continue
        actual = extract_value(answer)
        passed = compare(actual, config.expected_value, config.comparison_type)
        results.append(result.model_copy(update={"actual": actual, "passed": passed}))
    return results


def extract_value(answer: Dict[str, Any]) -> Any:
    """The value a ``/api/sql-runner`` answer stands for, as the upstream harness extracts it."""
    if "data" not in answer:
        return answer
    rows = answer["data"]
    if not isinstance(rows, list) or not rows:
        return rows
    if len(rows) == 1 and isinstance(rows[0], dict) and len(rows[0]) == 1:
        return next(iter(rows[0].values()))
    return rows[0] if len(rows) == 1 else rows


def compare(actual: Any, expected: Any, comparison_type: ComparisonType) -> bool:
    """Compare as the upstream harness does; values that cannot be ordered do not pass.

    One deviation: for equality and ordering, a number and a numeric string compare as numbers. 20 public oracle
    tasks expect ``"1"`` from a ``COUNT(*)``, which Python equality never matches; the same tasks in the dataset's
    distractor-tool configs expect ``1``.
    """
    if comparison_type != "contains":
        actual, expected = _as_numbers(actual, expected)
    try:
        if comparison_type == "equals":
            return bool(actual == expected)
        if comparison_type == "greater_than":
            return bool(actual > expected)
        if comparison_type == "less_than":
            return bool(actual < expected)
        return expected in str(actual)
    except TypeError:
        return False


def _as_numbers(actual: Any, expected: Any) -> tuple[Any, Any]:
    """Turn the numeric string of a (number, numeric string) pair into a number; leave other pairs alone."""

    def is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def number(text: str) -> int | float | None:
        for parse in (int, float):
            try:
                return parse(text)
            except ValueError:
                pass
        return None

    if is_number(actual) and isinstance(expected, str) and (parsed := number(expected.strip())) is not None:
        return actual, parsed
    if is_number(expected) and isinstance(actual, str) and (parsed := number(actual.strip())) is not None:
        return parsed, expected
    return actual, expected


class FinalStateVerifyRequest(BaseModel):
    """A task's verifiers and the final state of each gym server's database, as the SQL that builds it."""

    model_config = ConfigDict(extra="forbid")

    verifiers: List[DatabaseStateVerifier] = Field(min_length=1)
    final_state: Dict[str, str]


class FinalStateVerifyResponse(BaseModel):
    reward: float
    verifier_results: List[VerifierResult]


class FinalStateVerifier:
    """Scores verifiers against final database states built in SQLite: the server's scoring, without gym servers."""

    async def verify(self, body: FinalStateVerifyRequest) -> FinalStateVerifyResponse:
        databases = {}
        for gym_name, sql in body.final_state.items():
            databases[gym_name] = sqlite3.connect(":memory:")
            databases[gym_name].executescript(sql)

        async def run_sql(verifier: DatabaseStateVerifier) -> Dict[str, Any]:
            try:
                cursor = databases[verifier.gym_name].execute(verifier.validation_config.query)
            except sqlite3.Error as error:
                raise QueryRejectedError(f"SQL execution error: {error}") from error
            columns = [column[0] for column in cursor.description or []]
            return {"success": True, "data": [dict(zip(columns, row)) for row in cursor.fetchall()]}

        try:
            results = await evaluate(body.verifiers, run_sql)
        finally:
            for database in databases.values():
                database.close()
        return FinalStateVerifyResponse(reward=float(all(r.passed for r in results)), verifier_results=results)
