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


def _write_cases(
    path: Path,
    *,
    expected_full: float = 1.0,
    expected_zero: float = 0.0,
    expected_sample: float = 0.5,
    determinism: bool = True,
) -> str:
    cases = [
        {
            "name": "full",
            "kind": "full_reward",
            "request": {"outcome": "full"},
            "expected_reward": expected_full,
        },
        {
            "name": "zero",
            "kind": "zero_reward",
            "request": {"outcome": "zero"},
            "expected_reward": expected_zero,
        },
        {
            "name": "malformed",
            "kind": "malformed",
            "request": {"outcome": "malformed"},
            "expected_error": "invalid request",
        },
    ]
    if determinism:
        cases.append(
            {
                "name": "seeded",
                "kind": "determinism",
                "request": {"outcome": "sample", "seed": 7},
                "expected_reward": expected_sample,
            }
        )
    rendered = "".join(f"{json.dumps(case)}\n" for case in cases)
    path.write_text(rendered, encoding="utf-8")
    return rendered


class _Request(BaseModel):
    outcome: str
    seed: int | None = None


class _Server:
    def __init__(self, sample: Callable[[], float], full_reward: float = 1.0, zero_reward: float = 0.0):
        self._sample = sample
        self._full_reward = full_reward
        self._zero_reward = zero_reward

    async def verify(self, request: _Request) -> dict[str, float]:
        if request.outcome == "malformed":
            raise ValueError("invalid request payload")
        if request.outcome == "sample":
            return {"reward": self._sample()}
        return {"reward": self._full_reward if request.outcome == "full" else self._zero_reward}


def _fixture(
    path: Path,
    *,
    full_reward: float = 1.0,
    zero_reward: float = 0.0,
    sample_reward: float = 0.5,
    deterministic: bool = True,
    reseed: bool = True,
) -> VerifierFixture:
    calls = 0

    def sample() -> float:
        nonlocal calls
        calls += 1
        return sample_reward if deterministic or calls == 1 else sample_reward + 0.1

    return VerifierFixture(
        server_factory=lambda: _Server(sample, full_reward, zero_reward),
        request_model=_Request,
        cases_path=path,
        reseed=(lambda _server, _request: None) if reseed else None,
    )


async def test_exercises_the_four_case_floor(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    original = _write_cases(path)

    results = await exercise_verifier_fixture(_fixture(path), reward_range=(0, 1), determinism="seeded")

    assert [result.kind for result in results] == ["full_reward", "zero_reward", "malformed", "determinism"]
    assert results[-1].observed_rewards == (0.5, 0.5)
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "contains no cases"),
        ("not-json\n", "Invalid JSON"),
        ('{"name":"full","kind":"full_reward","request":{},"expected_reward":1}\n', "missing required"),
        ('{"name":"bad","kind":"malformed","request":{}}\n', "requires expected_error"),
    ],
)
async def test_rejects_invalid_or_incomplete_cases(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(VerifierFixtureError, match=message):
        await exercise_verifier_fixture(_fixture(path), reward_range=(0, 1), determinism="unknown")


async def test_seeded_fixture_requires_a_determinism_case(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_cases(path, determinism=False)

    with pytest.raises(VerifierFixtureError, match="missing required cases: determinism"):
        await exercise_verifier_fixture(_fixture(path), reward_range=(0, 1), determinism="seeded")


async def test_seeded_fixture_requires_a_reseed_adapter(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_cases(path)

    with pytest.raises(VerifierFixtureError, match="requires a reseed adapter"):
        await exercise_verifier_fixture(_fixture(path, reseed=False), reward_range=(0, 1), determinism="seeded")


async def test_enforces_range_endpoints_and_determinism(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_cases(path)

    with pytest.raises(VerifierFixtureError, match="outside declared range"):
        await exercise_verifier_fixture(_fixture(path, full_reward=1.5), reward_range=(0, 1), determinism="seeded")
    with pytest.raises(VerifierFixtureError, match="does not pin the upper endpoint"):
        await exercise_verifier_fixture(_fixture(path, full_reward=0.75), reward_range=(0, 1), determinism="seeded")
    with pytest.raises(VerifierFixtureError, match="changed"):
        await exercise_verifier_fixture(_fixture(path, deterministic=False), reward_range=(0, 1), determinism="seeded")
    with pytest.raises(VerifierFixtureError, match="lower < upper"):
        await exercise_verifier_fixture(_fixture(path), reward_range=(1, 0), determinism="seeded")


async def test_lower_is_better_reverses_the_required_endpoints(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_cases(path, expected_full=0.0, expected_zero=1.0)

    results = await exercise_verifier_fixture(
        _fixture(path, full_reward=0.0, zero_reward=1.0),
        reward_range=(0, 1),
        higher_is_better=False,
        determinism="seeded",
    )

    assert results[0].observed_rewards == (0.0,)
    assert results[1].observed_rewards == (1.0,)


async def test_update_expected_is_explicit_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    original = _write_cases(path, expected_sample=0.25)

    with pytest.raises(VerifierFixtureError, match="reward mismatch"):
        await exercise_verifier_fixture(_fixture(path), reward_range=(0, 1), determinism="seeded")
    assert path.read_text(encoding="utf-8") == original

    with pytest.raises(VerifierFixtureError, match="changed"):
        await exercise_verifier_fixture(
            _fixture(path, deterministic=False), reward_range=(0, 1), determinism="seeded", update_expected=True
        )
    assert path.read_text(encoding="utf-8") == original

    await exercise_verifier_fixture(_fixture(path), reward_range=(0, 1), determinism="seeded", update_expected=True)
    cases = load_verifier_fixture(path)
    assert next(case for case in cases if case.kind == "determinism").expected_reward == 0.5


async def test_invocation_adapter_supports_contextual_verifiers(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_cases(path)

    class ContextServer(_Server):
        async def verify_with_context(self, context: str, request: _Request) -> dict[str, float]:
            assert context == "fixture-context"
            return await super().verify(request)

    fixture = VerifierFixture(
        server_factory=lambda: ContextServer(lambda: 0.5),
        request_model=_Request,
        cases_path=path,
        invoke=lambda server, request: server.verify_with_context("fixture-context", request),
        reseed=lambda _server, _request: None,
    )

    await exercise_verifier_fixture(fixture, reward_range=(0, 1), determinism="seeded")


@pytest.mark.parametrize(
    ("reward_range", "higher_is_better", "determinism", "message"),
    [
        ((0,), True, "unknown", "exactly two"),
        ((False, 1), True, "unknown", "finite numbers"),
        ((0, 1), "yes", "unknown", "must be a boolean"),
        ((0, 1), True, "repeatable", "Unsupported determinism"),
    ],
)
async def test_rejects_invalid_fixture_contract_options(
    tmp_path: Path,
    reward_range: tuple,
    higher_is_better: object,
    determinism: str,
    message: str,
) -> None:
    path = tmp_path / "cases.jsonl"
    _write_cases(path, determinism=False)

    with pytest.raises(VerifierFixtureError, match=message):
        await exercise_verifier_fixture(
            _fixture(path),
            reward_range=reward_range,
            higher_is_better=higher_is_better,
            determinism=determinism,
        )


def test_loader_skips_blank_lines_and_reports_read_errors(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_cases(path, determinism=False)
    path.write_text("\n" + path.read_text(encoding="utf-8"), encoding="utf-8")

    assert len(load_verifier_fixture(path)) == 3
    with pytest.raises(VerifierFixtureError, match="Could not read verifier fixture"):
        load_verifier_fixture(tmp_path)


@pytest.mark.parametrize(
    ("factory", "invoke", "message"),
    [
        (lambda: (_ for _ in ()).throw(RuntimeError("factory failed")), None, "could not create"),
        (object, None, "callable verify"),
        (object, lambda _server, _request: float("nan"), "non-finite"),
        (object, lambda _server, _request: (_ for _ in ()).throw(RuntimeError("verify failed")), "Case 'full' failed"),
    ],
)
async def test_reports_invalid_server_adapters(
    tmp_path: Path,
    factory: Callable[[], object],
    invoke: Callable | None,
    message: str,
) -> None:
    path = tmp_path / "cases.jsonl"
    _write_cases(path, determinism=False)
    fixture = VerifierFixture(server_factory=factory, request_model=_Request, cases_path=path, invoke=invoke)

    with pytest.raises(VerifierFixtureError, match=message):
        await exercise_verifier_fixture(fixture, reward_range=(0, 1), determinism="unknown")


@pytest.mark.parametrize(
    ("malformed", "expected_error", "message"),
    [
        ("fixture_error", "invalid request", "fixture rejected request"),
        ("wrong_error", "different error", "did not contain"),
        ("no_error", "invalid request", "did not raise"),
    ],
)
async def test_reports_invalid_malformed_case_behavior(
    tmp_path: Path,
    malformed: str,
    expected_error: str,
    message: str,
) -> None:
    path = tmp_path / "cases.jsonl"
    _write_cases(path, determinism=False)
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    cases[2]["expected_error"] = expected_error
    path.write_text("".join(f"{json.dumps(case)}\n" for case in cases), encoding="utf-8")

    class MalformedServer(_Server):
        async def verify(self, request: _Request) -> dict[str, float]:
            if request.outcome != "malformed":
                return await super().verify(request)
            if malformed == "fixture_error":
                raise VerifierFixtureError("fixture rejected request")
            if malformed == "wrong_error":
                raise ValueError("invalid request")
            return {"reward": 0.0}

    fixture = VerifierFixture(
        server_factory=lambda: MalformedServer(lambda: 0.5),
        request_model=_Request,
        cases_path=path,
    )

    with pytest.raises(VerifierFixtureError, match=message):
        await exercise_verifier_fixture(fixture, reward_range=(0, 1), determinism="unknown")


async def test_supports_async_reseed(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_cases(path)

    async def reseed(_server: object, _request: object) -> None:
        return None

    fixture = _fixture(path)
    fixture = VerifierFixture(
        server_factory=fixture.server_factory,
        request_model=fixture.request_model,
        cases_path=fixture.cases_path,
        reseed=reseed,
    )

    await exercise_verifier_fixture(fixture, reward_range=(0, 1), determinism="seeded")
