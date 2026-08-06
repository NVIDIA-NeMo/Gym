# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.verifier_fixture import (
    VerifierFixtureError,
    exercise_verifier_fixture,
    load_verifier_fixture,
    validate_verifier_fixture,
)


def _cases(*, full: float = 1.0, zero: float = 0.0) -> list[dict[str, Any]]:
    return [
        {
            "case": "full_reward",
            "request": {"outcome": "full"},
            "expected_status": 200,
            "expected_reward": full,
        },
        {
            "case": "zero_reward",
            "request": {"outcome": "zero"},
            "expected_status": 200,
            "expected_reward": zero,
        },
        {
            "case": "malformed",
            "request": {"outcome": "malformed"},
            "expected_status": 422,
        },
        {
            "case": "determinism_reseed",
            "request": {"outcome": "full"},
            "expected_status": 200,
            "expected_reward": full,
            "reseed": True,
        },
    ]


def _write(path: Path, cases: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(case) + "\n" for case in cases))


class _Response:
    def __init__(self, status_code: int, payload: Mapping[str, Any]):
        self.status_code = status_code
        self._payload = dict(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _Client:
    def __init__(self, *, full: float = 1.0, zero: float = 0.0, malformed_status: int = 422):
        self.full = full
        self.zero = zero
        self.malformed_status = malformed_status

    def post(self, path: str, *, json: Mapping[str, Any]) -> _Response:
        assert path == "/verify"
        outcome = json["outcome"]
        if outcome == "malformed":
            return _Response(self.malformed_status, {"detail": "invalid"})
        return _Response(200, {"reward": self.full if outcome == "full" else self.zero})


def test_contract_enforces_scoring_floor_manifest_endpoints_and_seeded_reseed() -> None:
    validate_verifier_fixture(_cases(full=2, zero=-1), reward_range=(-1, 2), determinism="seeded")
    validate_verifier_fixture(
        _cases(full=-1, zero=2),
        reward_range=(-1, 2),
        higher_is_better=False,
        determinism="seeded",
    )

    with pytest.raises(VerifierFixtureError, match="missing: malformed"):
        validate_verifier_fixture([case for case in _cases() if case["case"] != "malformed"])
    with pytest.raises(VerifierFixtureError, match="full-reward case expects 1"):
        validate_verifier_fixture(_cases(), reward_range=(0, 4))
    malformed_reseed = _cases()
    malformed_reseed[-1].pop("reseed")
    with pytest.raises(VerifierFixtureError, match="must set reseed"):
        validate_verifier_fixture(malformed_reseed, determinism="seeded")


@pytest.mark.parametrize("determinism", ["stochastic", "unknown", None])
def test_non_seeded_contract_does_not_require_or_compare_reseed(determinism: str | None) -> None:
    cases = [case for case in _cases() if case["case"] != "determinism_reseed"]

    validate_verifier_fixture(cases, reward_range=(0, 1), determinism=determinism)

    with pytest.raises(VerifierFixtureError, match="missing: determinism_reseed"):
        validate_verifier_fixture(cases, reward_range=(0, 1), determinism="seeded")


def test_stochastic_fixture_does_not_run_reseed_comparison(tmp_path: Path) -> None:
    path = tmp_path / "verifier_cases.jsonl"
    _write(path, _cases())
    factory_calls = 0

    def client_factory() -> _Client:
        nonlocal factory_calls
        factory_calls += 1
        return _Client(full=float(factory_calls))

    exercise_verifier_fixture(client_factory, path, determinism="stochastic", update_expected=True)

    assert factory_calls == len(_cases())


def test_exercises_fixture_against_in_process_client(tmp_path: Path) -> None:
    path = tmp_path / "verifier_cases.jsonl"
    _write(path, _cases())

    exercise_verifier_fixture(_Client, path, reward_range=(0, 1), determinism="seeded")

    stale = _cases(full=9, zero=-9)
    _write(path, stale)
    with pytest.raises(VerifierFixtureError, match="expected reward"):
        exercise_verifier_fixture(_Client, path)


def test_update_expected_is_atomic_and_regenerates_rewards(tmp_path: Path) -> None:
    path = tmp_path / "verifier_cases.jsonl"
    _write(path, _cases())

    exercise_verifier_fixture(
        lambda: _Client(full=4, zero=-2),
        path,
        update_expected=True,
        determinism="seeded",
    )

    updated = load_verifier_fixture(path)
    assert updated[0]["expected_reward"] == 4
    assert updated[1]["expected_reward"] == -2
    assert updated[2] == {
        "case": "malformed",
        "request": {"outcome": "malformed"},
        "expected_status": 422,
    }
    validate_verifier_fixture(updated, reward_range=(-2, 4), determinism="seeded")

    before = path.read_text()
    with pytest.raises(VerifierFixtureError, match="malformed.*non-success"):
        exercise_verifier_fixture(
            lambda: _Client(malformed_status=200),
            path,
            update_expected=True,
        )
    assert path.read_text() == before


def test_stateful_fixture_replays_canned_setup_for_each_reseed(tmp_path: Path) -> None:
    path = tmp_path / "verifier_cases.jsonl"
    cases = _cases()
    cases[-1]["setup"] = [
        {
            "path": "/seed_session",
            "request": {"known_end_state": "complete"},
            "expected_status": 200,
        }
    ]
    cases[-1]["request"] = {"outcome": "session"}
    _write(path, cases)

    class StatefulClient(_Client):
        state = None

        def post(self, path: str, *, json: Mapping[str, Any]) -> _Response:
            if path == "/seed_session":
                self.state = json["known_end_state"]
                return _Response(200, {"seeded": True})
            if json["outcome"] == "session":
                return _Response(200, {"reward": 1.0 if self.state == "complete" else 0.0})
            return super().post(path, json=json)

    exercise_verifier_fixture(StatefulClient, path, reward_range=(0, 1), determinism="seeded")
