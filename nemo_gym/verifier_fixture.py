# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise a resources server's verifier against a small JSONL fixture."""

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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError, model_validator


CaseKind = Literal["full_reward", "zero_reward", "malformed", "determinism"]
VerifyInvocation = Callable[[object, BaseModel], object | Awaitable[object]]


class VerifierFixtureError(ValueError):
    """A verifier fixture is invalid or does not match verifier behavior."""


@dataclass(frozen=True)
class VerifierFixture:
    """Everything needed to exercise one resources server's verifier in-process.

    Resources-server ``app.py`` modules export an instance as ``VERIFIER_FIXTURE``.
    ``server_factory`` must return fresh state on every call. Most verifiers use the
    default ``server.verify(body)`` invocation; ``invoke`` adapts verifiers whose
    method also needs request or session context.
    """

    server_factory: Callable[[], object]
    request_model: type[BaseModel]
    cases_path: str | Path
    invoke: VerifyInvocation | None = None


class VerifierFixtureCase(BaseModel):
    """One request and its expected verifier outcome."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: CaseKind
    request: dict[str, Any]
    expected_reward: FiniteFloat | None = None
    expected_error: str | None = None
    expected_error_type: str | None = None

    @model_validator(mode="after")
    def validate_expectation(self) -> "VerifierFixtureCase":
        if self.kind == "malformed":
            if not self.expected_error or not self.expected_error_type:
                raise ValueError("a malformed case requires expected_error and expected_error_type")
            if self.expected_reward is not None:
                raise ValueError("a malformed case may not set expected_reward")
        elif self.expected_reward is None:
            raise ValueError(f"a {self.kind} case requires expected_reward")
        elif self.expected_error is not None:
            raise ValueError(f"a {self.kind} case may not set expected_error")
        elif self.expected_error_type is not None:
            raise ValueError(f"a {self.kind} case may not set expected_error_type")
        return self


class VerifierFixtureResult(BaseModel):
    """Observed rewards for one successfully exercised fixture case."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: CaseKind
    observed_rewards: tuple[float, ...] = ()


def load_verifier_fixture(path: str | Path) -> list[VerifierFixtureCase]:
    """Load and validate resource-server-owned JSONL fixture cases."""

    fixture_path = Path(path)
    try:
        lines = fixture_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise VerifierFixtureError(f"Could not read verifier fixture '{fixture_path}': {error}") from error

    cases: list[VerifierFixtureCase] = []
    for line_number, line in enumerate(lines, start=1):
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
            raise VerifierFixtureError(
                f"Invalid verifier fixture '{fixture_path}' at line {line_number}: {error.errors()[0]['msg']}"
            ) from error

    if not cases:
        raise VerifierFixtureError(f"Verifier fixture '{fixture_path}' contains no cases")
    return cases


def _reward_from(result: object, case: VerifierFixtureCase) -> float:
    if isinstance(result, Mapping):
        reward = result.get("reward")
    else:
        reward = getattr(result, "reward", result if isinstance(result, (int, float)) else None)
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise VerifierFixtureError(f"Case '{case.name}' returned a non-finite numeric reward: {reward!r}")
    return float(reward)


async def _observe(fixture: VerifierFixture, case: VerifierFixtureCase) -> float:
    server = fixture.server_factory()
    request = fixture.request_model.model_validate(case.request)
    if fixture.invoke is None:
        verify = getattr(server, "verify", None)
        if not callable(verify):
            raise VerifierFixtureError("server_factory returned an object without a callable verify method")
        result = verify(request)
    else:
        result = fixture.invoke(server, request)
    if inspect.isawaitable(result):
        result = await result
    return _reward_from(result, case)


def _reward_bounds(reward_range: Sequence[float]) -> tuple[float, float]:
    if len(reward_range) != 2:
        raise VerifierFixtureError("reward_range must contain exactly two endpoints")
    lower, upper = reward_range
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (lower, upper)):
        raise VerifierFixtureError("reward_range endpoints must be finite numbers")
    lower, upper = float(lower), float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise VerifierFixtureError("reward_range must contain finite endpoints with lower < upper")
    return lower, upper


def _assert_close(actual: float, expected: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
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
    finally:
        temporary.unlink(missing_ok=True)


async def exercise_verifier_fixture(
    fixture: VerifierFixture,
    *,
    reward_range: Sequence[float],
    determinism: str | Enum,
    update_expected: bool = False,
) -> tuple[VerifierFixtureResult, ...]:
    """Run a verifier directly against its fixture, without starting Gym services.

    A determinism case creates two fresh servers with the same request payload. A
    stateful ``invoke`` adapter can seed each server from that payload before calling
    its verifier.
    """

    path = Path(fixture.cases_path)
    cases = load_verifier_fixture(path)
    lower, upper = _reward_bounds(reward_range)
    determinism_value = str(getattr(determinism, "value", determinism))
    if determinism_value not in {"seeded", "stochastic", "unknown"}:
        raise VerifierFixtureError(f"Unsupported determinism value: {determinism_value!r}")

    required_kinds = {"full_reward", "zero_reward", "malformed"}
    if determinism_value == "seeded":
        required_kinds.add("determinism")
    missing = sorted(required_kinds - {case.kind for case in cases})
    if missing:
        raise VerifierFixtureError(f"Verifier fixture '{path}' is missing required cases: {', '.join(missing)}")

    results: list[VerifierFixtureResult] = []
    changed = False
    for case in cases:
        if case.kind == "determinism" and determinism_value != "seeded":
            continue
        if case.kind == "malformed":
            try:
                await _observe(fixture, case)
            except Exception as error:
                if type(error).__name__ != case.expected_error_type:
                    raise VerifierFixtureError(
                        f"Malformed case '{case.name}' raised {type(error).__name__}, expected "
                        f"{case.expected_error_type}"
                    ) from error
                if case.expected_error not in str(error):
                    raise VerifierFixtureError(
                        f"Malformed case '{case.name}' raised {type(error).__name__}, but its message did not "
                        f"contain {case.expected_error!r}: {error}"
                    ) from error
            else:
                raise VerifierFixtureError(f"Malformed case '{case.name}' did not raise an error")
            results.append(VerifierFixtureResult(name=case.name, kind=case.kind))
            continue

        observed = await _observe(fixture, case)
        if not lower <= observed <= upper:
            raise VerifierFixtureError(
                f"Case '{case.name}' returned reward {observed} outside declared range [{lower}, {upper}]"
            )
        observations = [observed]
        if case.kind == "full_reward":
            _assert_close(observed, upper, f"Full-reward case '{case.name}' does not pin the upper endpoint")
        elif case.kind == "zero_reward":
            _assert_close(observed, lower, f"Zero-reward case '{case.name}' does not pin the lower endpoint")
        elif case.kind == "determinism":
            repeated = await _observe(fixture, case)
            observations.append(repeated)
            _assert_close(repeated, observed, f"Determinism case '{case.name}' changed after reseeding")

        if update_expected:
            if not math.isclose(observed, float(case.expected_reward), rel_tol=1e-9, abs_tol=1e-12):
                case.expected_reward = observed
                changed = True
        else:
            _assert_close(observed, float(case.expected_reward), f"Case '{case.name}' reward mismatch")
        results.append(VerifierFixtureResult(name=case.name, kind=case.kind, observed_rewards=tuple(observations)))

    if update_expected and changed:
        _write_fixture(path, cases)
    return tuple(results)


__all__ = [
    "VerifierFixture",
    "VerifierFixtureCase",
    "VerifierFixtureError",
    "VerifierFixtureResult",
    "exercise_verifier_fixture",
    "load_verifier_fixture",
]
