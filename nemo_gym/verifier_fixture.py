# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process contract tests for resources-server verifiers."""

from __future__ import annotations

import inspect
import json
import math
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


_CaseKind = Literal["full_reward", "zero_reward", "malformed", "determinism"]
_FixtureInvocation = Callable[[object, BaseModel], object | Awaitable[object]]
_FiniteReward = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class VerifierFixtureError(ValueError):
    """A fixture is invalid or does not match verifier behavior."""


@dataclass(frozen=True)
class VerifierFixture:
    server_factory: Callable[[], object]
    request_model: type[BaseModel]
    cases_path: str | Path
    invoke: _FixtureInvocation | None = None
    reseed: _FixtureInvocation | None = None


class VerifierFixtureCase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1)
    kind: _CaseKind
    request: dict[str, Any]
    expected_reward: _FiniteReward | None = None
    expected_error: str | None = None

    @model_validator(mode="after")
    def validate_expectation(self) -> "VerifierFixtureCase":
        if self.kind == "malformed":
            if not self.expected_error or self.expected_reward is not None:
                raise ValueError("a malformed case requires expected_error and forbids expected_reward")
        elif self.expected_reward is None or self.expected_error is not None:
            raise ValueError(f"a {self.kind} case requires expected_reward and forbids expected_error")
        return self


@dataclass(frozen=True)
class VerifierFixtureResult:
    name: str
    kind: _CaseKind
    observed_rewards: tuple[float, ...] = ()


def load_verifier_fixture(path: str | Path) -> list[VerifierFixtureCase]:
    """Load and validate resources-server-owned JSONL cases."""
    fixture_path = Path(path)
    try:
        lines = fixture_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise VerifierFixtureError(f"Could not read verifier fixture '{fixture_path}': {error}") from error

    cases: list[VerifierFixtureCase] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise VerifierFixtureError(
                f"Invalid JSON in verifier fixture '{fixture_path}' at line {line_number}: {error.msg}"
            ) from error
        try:
            cases.append(VerifierFixtureCase.model_validate(payload))
        except ValidationError as error:
            issue = error.errors(include_url=False, include_context=False, include_input=False)[0]
            raise VerifierFixtureError(
                f"Invalid verifier fixture '{fixture_path}' at line {line_number}: {issue['msg']}"
            ) from error
    if not cases:
        raise VerifierFixtureError(f"Verifier fixture '{fixture_path}' contains no cases")
    return cases


def _reward_from(result: object, case: VerifierFixtureCase) -> float:
    reward = result.get("reward") if isinstance(result, Mapping) else getattr(result, "reward", result)
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise VerifierFixtureError(f"Case '{case.name}' returned a non-finite numeric reward: {reward!r}")
    return float(reward)


async def _observe(
    fixture: VerifierFixture,
    case: VerifierFixtureCase,
    *,
    expect_error: bool = False,
    reseed: bool = False,
) -> float:
    if reseed and fixture.reseed is None:
        raise VerifierFixtureError("a seeded determinism case requires a reseed adapter")
    try:
        server = fixture.server_factory()
    except Exception as error:
        raise VerifierFixtureError(f"Case '{case.name}' could not create a fresh server: {error}") from error
    try:
        request = fixture.request_model.model_validate(case.request)
        if reseed:
            seeded = fixture.reseed(server, request)
            if inspect.isawaitable(seeded):
                await seeded
        if fixture.invoke is None:
            verify = getattr(server, "verify", None)
            if not callable(verify):
                raise VerifierFixtureError("server_factory returned an object without a callable verify method")
            result = verify(request)
        else:
            result = fixture.invoke(server, request)
        if inspect.isawaitable(result):
            result = await result
    except Exception as error:
        if expect_error:
            raise
        raise VerifierFixtureError(f"Case '{case.name}' failed: {error}") from error
    return _reward_from(result, case)


def _assert_equal(actual: float, expected: float, message: str) -> None:
    if actual != expected:
        raise VerifierFixtureError(f"{message}: expected {expected}, observed {actual}")


def _write_fixture(path: Path, cases: list[VerifierFixtureCase]) -> None:
    if path.is_symlink() or not path.is_file():
        raise VerifierFixtureError(f"Refusing to replace non-regular verifier fixture '{path}'")
    rendered = "".join(
        f"{json.dumps(case.model_dump(mode='json', exclude_none=True), sort_keys=True)}\n" for case in cases
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    except OSError as error:
        raise VerifierFixtureError(f"Could not update verifier fixture '{path}': {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


async def exercise_verifier_fixture(
    fixture: VerifierFixture,
    *,
    reward_range: Sequence[float],
    higher_is_better: bool = True,
    determinism: str | Enum,
    update_expected: bool = False,
) -> tuple[VerifierFixtureResult, ...]:
    """Exercise required scorer cases without starting a resources server."""
    path = Path(fixture.cases_path)
    cases = load_verifier_fixture(path)
    if len(reward_range) != 2:
        raise VerifierFixtureError("reward_range must contain exactly two endpoints")
    lower, upper = reward_range
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (lower, upper)):
        raise VerifierFixtureError("reward_range endpoints must be finite numbers")
    lower, upper = float(lower), float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise VerifierFixtureError("reward_range must contain finite endpoints with lower < upper")
    if not isinstance(higher_is_better, bool):
        raise VerifierFixtureError("higher_is_better must be a boolean")
    full_reward, zero_reward = (upper, lower) if higher_is_better else (lower, upper)
    full_endpoint, zero_endpoint = ("upper", "lower") if higher_is_better else ("lower", "upper")
    determinism_value = str(getattr(determinism, "value", determinism))
    if determinism_value not in {"seeded", "stochastic", "unknown"}:
        raise VerifierFixtureError(f"Unsupported determinism value: {determinism_value!r}")

    required = {"full_reward", "zero_reward", "malformed"} | (
        {"determinism"} if determinism_value == "seeded" else set()
    )
    missing = sorted(required - {case.kind for case in cases})
    if missing:
        raise VerifierFixtureError(f"Verifier fixture '{path}' is missing required cases: {', '.join(missing)}")

    results: list[VerifierFixtureResult] = []
    updates: dict[int, float] = {}
    for index, case in enumerate(cases):
        if case.kind == "determinism" and determinism_value != "seeded":
            continue
        if case.kind == "malformed":
            try:
                await _observe(fixture, case, expect_error=True)
            except VerifierFixtureError:
                raise
            except Exception as error:
                if case.expected_error not in str(error):
                    raise VerifierFixtureError(
                        f"Malformed case '{case.name}' error did not contain {case.expected_error!r}: {error}"
                    ) from error
            else:
                raise VerifierFixtureError(f"Malformed case '{case.name}' did not raise an error")
            results.append(VerifierFixtureResult(case.name, case.kind))
            continue

        observed = await _observe(fixture, case, reseed=case.kind == "determinism")
        if not lower <= observed <= upper:
            raise VerifierFixtureError(
                f"Case '{case.name}' returned reward {observed} outside declared range [{lower}, {upper}]"
            )
        observations = [observed]
        if case.kind == "full_reward":
            _assert_equal(
                observed,
                full_reward,
                f"Full-reward case '{case.name}' does not pin the {full_endpoint} endpoint",
            )
        elif case.kind == "zero_reward":
            _assert_equal(
                observed,
                zero_reward,
                f"Zero-reward case '{case.name}' does not pin the {zero_endpoint} endpoint",
            )
        elif case.kind == "determinism":
            repeated = await _observe(fixture, case, reseed=True)
            observations.append(repeated)
            _assert_equal(repeated, observed, f"Determinism case '{case.name}' changed on a fresh server")

        if update_expected:
            updates[index] = observed
        else:
            _assert_equal(observed, float(case.expected_reward), f"Case '{case.name}' reward mismatch")
        results.append(VerifierFixtureResult(case.name, case.kind, tuple(observations)))

    if update_expected:
        updated = [
            case.model_copy(update={"expected_reward": updates[index]}) if index in updates else case
            for index, case in enumerate(cases)
        ]
        if updated != cases:
            _write_fixture(path, updated)
    return tuple(results)
