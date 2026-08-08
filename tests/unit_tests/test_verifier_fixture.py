# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel

from nemo_gym.verifier_fixture import (
    VerifierFixture,
    VerifierFixtureError,
    exercise_verifier_fixture,
    load_verifier_fixture,
)


def _write_fixture(path: Path, *, expected: float = 0.5, include_determinism: bool = True) -> str:
    cases = [
        {"name": "correct", "kind": "full_reward", "request": {"outcome": "full"}, "expected_reward": 1.0},
        {"name": "incorrect", "kind": "zero_reward", "request": {"outcome": "zero"}, "expected_reward": 0.0},
        {
            "name": "invalid",
            "kind": "malformed",
            "request": {"outcome": "malformed"},
            "expected_error": "invalid request",
            "expected_error_type": "ValueError",
        },
    ]
    if include_determinism:
        cases.append(
            {
                "name": "same seed",
                "kind": "determinism",
                "request": {"outcome": "sample", "seed": 7},
                "expected_reward": expected,
            }
        )
    rendered = "".join(f"{json.dumps(case)}\n" for case in cases)
    path.write_text(rendered, encoding="utf-8")
    return rendered


class _Request(BaseModel):
    outcome: str
    seed: int | None = None


class _Server:
    def __init__(self, sample_reward: Callable[[], float], full_reward: float):
        self.sample_reward = sample_reward
        self.full_reward = full_reward

    async def verify(self, request: _Request):
        if request.outcome == "malformed":
            raise ValueError("invalid request payload")
        if request.outcome == "sample":
            return {"reward": self.sample_reward()}
        return {"reward": self.full_reward if request.outcome == "full" else 0.0}


def _fixture(
    path: Path,
    *,
    deterministic: bool = True,
    sample_reward: float = 0.5,
    full_reward: float = 1.0,
) -> VerifierFixture:
    sample_calls = 0

    def next_sample_reward() -> float:
        nonlocal sample_calls
        sample_calls += 1
        return sample_reward if deterministic or sample_calls == 1 else sample_reward + 0.1

    return VerifierFixture(
        server_factory=lambda: _Server(next_sample_reward, full_reward),
        request_model=_Request,
        cases_path=path,
    )


async def test_exercises_required_cases_without_rewriting(tmp_path: Path) -> None:
    fixture = tmp_path / "verifier_cases.jsonl"
    original = _write_fixture(fixture)

    results = await exercise_verifier_fixture(_fixture(fixture), reward_range=(0, 1), determinism="seeded")

    assert [result.kind for result in results] == [
        "full_reward",
        "zero_reward",
        "malformed",
        "determinism",
    ]
    assert results[-1].observed_rewards == (0.5, 0.5)
    assert fixture.read_text(encoding="utf-8") == original


async def test_update_expected_is_explicit_and_atomic(tmp_path: Path) -> None:
    fixture = tmp_path / "verifier_cases.jsonl"
    original = _write_fixture(fixture, expected=0.25)

    with pytest.raises(VerifierFixtureError, match="reward mismatch"):
        await exercise_verifier_fixture(_fixture(fixture), reward_range=(0, 1), determinism="seeded")
    assert fixture.read_text(encoding="utf-8") == original

    await exercise_verifier_fixture(_fixture(fixture), reward_range=(0, 1), determinism="seeded", update_expected=True)
    updated = load_verifier_fixture(fixture)
    assert next(case for case in updated if case.kind == "determinism").expected_reward == 0.5


async def test_update_expected_preserves_endpoint_and_determinism_checks(tmp_path: Path) -> None:
    fixture = tmp_path / "verifier_cases.jsonl"
    original = _write_fixture(fixture)

    with pytest.raises(VerifierFixtureError, match="outside declared range"):
        await exercise_verifier_fixture(
            _fixture(fixture, full_reward=1.5),
            reward_range=(0, 1),
            determinism="seeded",
            update_expected=True,
        )
    assert fixture.read_text(encoding="utf-8") == original

    with pytest.raises(VerifierFixtureError, match="non-finite numeric reward"):
        await exercise_verifier_fixture(
            _fixture(fixture, full_reward=float("nan")),
            reward_range=(0, 1),
            determinism="seeded",
        )

    with pytest.raises(VerifierFixtureError, match="does not pin the upper endpoint"):
        await exercise_verifier_fixture(
            _fixture(fixture, full_reward=0.75),
            reward_range=(0, 1),
            determinism="seeded",
            update_expected=True,
        )
    assert fixture.read_text(encoding="utf-8") == original

    with pytest.raises(VerifierFixtureError, match="changed after reseeding"):
        await exercise_verifier_fixture(
            _fixture(fixture, deterministic=False),
            reward_range=(0, 1),
            determinism="seeded",
            update_expected=True,
        )
    assert fixture.read_text(encoding="utf-8") == original


async def test_optional_invocation_adapter_supports_contextual_verifiers(tmp_path: Path) -> None:
    fixture_path = tmp_path / "verifier_cases.jsonl"
    _write_fixture(fixture_path)

    class ContextServer(_Server):
        async def verify(self, context, request):
            assert context == "fixture-context"
            return await super().verify(request)

    fixture = VerifierFixture(
        server_factory=lambda: ContextServer(lambda: 0.5, 1.0),
        request_model=_Request,
        cases_path=fixture_path,
        invoke=lambda server, request: server.verify("fixture-context", request),
    )

    await exercise_verifier_fixture(fixture, reward_range=(0, 1), determinism="seeded")


async def test_non_seeded_fixture_does_not_require_determinism_case(tmp_path: Path) -> None:
    fixture = tmp_path / "verifier_cases.jsonl"
    _write_fixture(fixture, include_determinism=False)

    await exercise_verifier_fixture(_fixture(fixture), reward_range=(0, 1), determinism="unknown")

    with pytest.raises(VerifierFixtureError, match="missing required cases: determinism"):
        await exercise_verifier_fixture(_fixture(fixture), reward_range=(0, 1), determinism="seeded")

    _write_fixture(fixture)
    results = await exercise_verifier_fixture(
        _fixture(fixture, deterministic=False),
        reward_range=(0, 1),
        determinism="unknown",
    )
    assert all(result.kind != "determinism" for result in results)


async def test_malformed_case_enforces_the_declared_error(tmp_path: Path) -> None:
    fixture = tmp_path / "verifier_cases.jsonl"
    _write_fixture(fixture)
    content = fixture.read_text(encoding="utf-8").replace(
        '"expected_error_type": "ValueError"',
        '"expected_error_type": "RuntimeError"',
    )
    fixture.write_text(content, encoding="utf-8")

    with pytest.raises(VerifierFixtureError, match="raised ValueError, expected RuntimeError"):
        await exercise_verifier_fixture(_fixture(fixture), reward_range=(0, 1), determinism="unknown")

    _write_fixture(fixture)
    fixture.write_text(
        fixture.read_text(encoding="utf-8").replace(
            '"expected_error": "invalid request"', '"expected_error": "other"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(VerifierFixtureError, match="message did not contain"):
        await exercise_verifier_fixture(_fixture(fixture), reward_range=(0, 1), determinism="unknown")

    class AcceptsMalformed(_Server):
        async def verify(self, request: _Request):
            if request.outcome == "malformed":
                return {"reward": 0.0}
            return await super().verify(request)

    _write_fixture(fixture)
    accepts_malformed = VerifierFixture(
        server_factory=lambda: AcceptsMalformed(lambda: 0.5, 1.0),
        request_model=_Request,
        cases_path=fixture,
    )
    with pytest.raises(VerifierFixtureError, match="did not raise an error"):
        await exercise_verifier_fixture(accepts_malformed, reward_range=(0, 1), determinism="unknown")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "contains no cases"),
        ("not json\n", "Invalid JSON"),
        ('{"name": "only", "kind": "full_reward", "request": {}, "expected_reward": 1}\n', "missing required"),
        (
            '{"name": "bad", "kind": "malformed", "request": {}}\n',
            "requires expected_error and expected_error_type",
        ),
    ],
)
async def test_rejects_invalid_or_incomplete_fixtures(tmp_path: Path, content: str, message: str) -> None:
    fixture = tmp_path / "verifier_cases.jsonl"
    fixture.write_text(content, encoding="utf-8")

    with pytest.raises(VerifierFixtureError, match=message):
        await exercise_verifier_fixture(_fixture(fixture), reward_range=(0, 1), determinism="unknown")


@pytest.mark.parametrize(
    ("reward_range", "message"),
    [
        ((0,), "exactly two endpoints"),
        ((False, 1), "finite numbers"),
        ((1, 0), "lower < upper"),
    ],
)
async def test_rejects_invalid_reward_ranges(tmp_path: Path, reward_range: tuple, message: str) -> None:
    fixture = tmp_path / "verifier_cases.jsonl"
    _write_fixture(fixture)

    with pytest.raises(VerifierFixtureError, match=message):
        await exercise_verifier_fixture(_fixture(fixture), reward_range=reward_range, determinism="seeded")
