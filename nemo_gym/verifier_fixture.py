# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard offline scoring fixture for resources servers.

The fixture deliberately exercises the ASGI application in process.  It does not
start Ray, a Gym head server, or a model server.  Environment manifests use the
same JSONL file to pin the scorer's reward endpoints and determinism contract.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from nemo_gym.repository_io import atomic_write_text


UPDATE_EXPECTED_ENV_VAR = "NEMO_GYM_UPDATE_EXPECTED"
REWARD_RANGE_ENV_VAR = "NEMO_GYM_FIXTURE_REWARD_RANGE"
DETERMINISM_ENV_VAR = "NEMO_GYM_FIXTURE_DETERMINISM"
HIGHER_IS_BETTER_ENV_VAR = "NEMO_GYM_FIXTURE_HIGHER_IS_BETTER"

FULL_REWARD_CASE = "full_reward"
ZERO_REWARD_CASE = "zero_reward"
MALFORMED_CASE = "malformed"
DETERMINISM_RESEED_CASE = "determinism_reseed"
REQUIRED_CASES = (
    FULL_REWARD_CASE,
    ZERO_REWARD_CASE,
    MALFORMED_CASE,
)
SEEDED_REQUIRED_CASES = (*REQUIRED_CASES, DETERMINISM_RESEED_CASE)


class VerifierFixtureError(ValueError):
    """A verifier fixture is malformed or disagrees with its declared contract."""


def build_offline_verifier_app(
    server_class: type[Any],
    *,
    server_config: Mapping[str, Any],
    instance_name: str,
) -> Any:
    """Construct a resources-server ASGI app for the canonical offline fixture.

    Entrypoints expose a fixed ``create_offline_verifier_app`` wrapper around
    this helper.  The wrapper selects the runtime server class explicitly; this
    helper owns the otherwise shared config and ``ServerClient`` construction.
    """

    from unittest.mock import MagicMock

    from nemo_gym.base_resources_server import SimpleResourcesServer
    from nemo_gym.server_utils import ServerClient

    if not isinstance(server_class, type) or not issubclass(server_class, SimpleResourcesServer):
        raise VerifierFixtureError("Offline verifier factory must select a SimpleResourcesServer subclass.")
    config_field = server_class.model_fields.get("config")
    config_type = config_field.annotation if config_field is not None else None
    if not isinstance(config_type, type) or not hasattr(config_type, "model_validate"):
        raise VerifierFixtureError(
            f"Resources-server class {server_class.__name__} does not declare a concrete Pydantic config type."
        )

    payload = dict(server_config)
    payload.update({"host": "127.0.0.1", "port": 8080, "num_workers": None, "name": instance_name})
    try:
        config = config_type.model_validate(payload)
        server = server_class(config=config, server_client=MagicMock(spec=ServerClient))
        return server.setup_webserver()
    except VerifierFixtureError:
        raise
    except Exception as error:
        raise VerifierFixtureError(
            f"Could not construct resources-server class {server_class.__name__} for offline verification: {error}."
        ) from error


def verifier_fixture_environment(
    *,
    reward_range: tuple[int | float, int | float] | None = None,
    higher_is_better: bool = True,
    determinism: str | None = None,
    update_expected: bool = False,
) -> dict[str, str]:
    """Build the explicit environment contract consumed by verifier tests.

    ``NEMO_GYM_UPDATE_EXPECTED`` is always set, including for the ordinary
    read-only path.  This prevents a value inherited from a developer shell or
    CI runner from silently rewriting a checked-in fixture.
    """

    environment = {UPDATE_EXPECTED_ENV_VAR: "1" if update_expected else "0"}
    if reward_range is not None:
        environment[REWARD_RANGE_ENV_VAR] = json.dumps(list(reward_range), separators=(",", ":"))
        environment[HIGHER_IS_BETTER_ENV_VAR] = str(higher_is_better).lower()
    if determinism is not None:
        environment[DETERMINISM_ENV_VAR] = determinism
    return environment


class _Response(Protocol):
    status_code: int

    def json(self) -> Any: ...


class _Client(Protocol):
    def post(self, path: str, *, json: Mapping[str, Any]) -> _Response: ...


def load_verifier_fixture(path: str | Path) -> list[dict[str, Any]]:
    """Load a non-empty JSONL fixture with path-qualified diagnostics."""

    fixture_path = Path(path)
    cases: list[dict[str, Any]] = []
    try:
        lines = fixture_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise VerifierFixtureError(f"Could not read verifier fixture '{fixture_path}': {error}.") from error

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise VerifierFixtureError(
                f"Verifier fixture '{fixture_path}' line {line_number} is not valid JSON: {error.msg}."
            ) from None
        if not isinstance(case, dict):
            raise VerifierFixtureError(f"Verifier fixture '{fixture_path}' line {line_number} must be a JSON object.")
        cases.append(case)
    if not cases:
        raise VerifierFixtureError(f"Verifier fixture '{fixture_path}' contains no cases.")
    return cases


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise VerifierFixtureError(f"{label} must be a finite number; got {value!r}.")
    return float(value)


def validate_verifier_fixture(
    cases: Sequence[Mapping[str, Any]],
    *,
    reward_range: tuple[int | float, int | float] | None = None,
    higher_is_better: bool = True,
    determinism: str | None = None,
    require_expected_values: bool = True,
) -> None:
    """Validate the scoring floor and, when supplied, a manifest contract.

    Extra cases are permitted.  Three scoring sentinels are always required; a
    re-seed sentinel is additionally required only when the manifest claims
    ``determinism: seeded``.
    """

    by_name: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for index, case in enumerate(cases, 1):
        case_name = case.get("case")
        if not isinstance(case_name, str) or not case_name.strip():
            raise VerifierFixtureError(f"Verifier fixture case {index} requires a non-empty string 'case'.")
        if case_name in by_name:
            duplicates.append(case_name)
        by_name[case_name] = case
        request = case.get("request")
        if not isinstance(request, Mapping):
            raise VerifierFixtureError(f"Verifier fixture case '{case_name}' requires an object 'request'.")
        setup = case.get("setup") or []
        if not isinstance(setup, Sequence) or isinstance(setup, (str, bytes, bytearray)):
            raise VerifierFixtureError(f"Verifier fixture case '{case_name}' field 'setup' must be a list.")
        for step_index, step in enumerate(setup, 1):
            if not isinstance(step, Mapping):
                raise VerifierFixtureError(
                    f"Verifier fixture case '{case_name}' setup step {step_index} must be an object."
                )
            if not isinstance(step.get("path"), str) or not str(step["path"]).startswith("/"):
                raise VerifierFixtureError(
                    f"Verifier fixture case '{case_name}' setup step {step_index} requires an absolute API path."
                )
            if not isinstance(step.get("request"), Mapping):
                raise VerifierFixtureError(
                    f"Verifier fixture case '{case_name}' setup step {step_index} requires an object 'request'."
                )
            method = step.get("method", "post")
            if method != "post":
                raise VerifierFixtureError(
                    f"Verifier fixture case '{case_name}' setup step {step_index} only supports method='post'."
                )
            setup_status = step.get("expected_status", 200)
            if isinstance(setup_status, bool) or not isinstance(setup_status, int):
                raise VerifierFixtureError(
                    f"Verifier fixture case '{case_name}' setup step {step_index} expected_status must be an integer."
                )
        if require_expected_values:
            status = case.get("expected_status")
            if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
                raise VerifierFixtureError(
                    f"Verifier fixture case '{case_name}' requires expected_status in the range 100..599; "
                    "regenerate TODO expectations with --update-expected."
                )
    if duplicates:
        raise VerifierFixtureError(
            "Verifier fixture case names must be unique; duplicate(s): " + ", ".join(duplicates)
        )

    required_cases = SEEDED_REQUIRED_CASES if determinism == "seeded" else REQUIRED_CASES
    missing = [name for name in required_cases if name not in by_name]
    if missing:
        raise VerifierFixtureError(
            "Verifier fixture requires the scoring floor "
            f"({', '.join(required_cases)}); missing: {', '.join(missing)}."
        )

    full = by_name[FULL_REWARD_CASE]
    zero = by_name[ZERO_REWARD_CASE]
    malformed = by_name[MALFORMED_CASE]
    reseed = by_name.get(DETERMINISM_RESEED_CASE)
    if reseed is not None and reseed.get("reseed") is not True:
        raise VerifierFixtureError("Verifier fixture case 'determinism_reseed' must set reseed: true.")

    if require_expected_values:
        reward_cases = [(FULL_REWARD_CASE, full), (ZERO_REWARD_CASE, zero)]
        if reseed is not None:
            reward_cases.append((DETERMINISM_RESEED_CASE, reseed))
        for name, case in reward_cases:
            status = int(case["expected_status"])
            if not 200 <= status < 300:
                raise VerifierFixtureError(f"Verifier fixture case '{name}' must expect a successful HTTP status.")
            _finite_number(case.get("expected_reward"), label=f"Verifier fixture case '{name}' expected_reward")
        if 200 <= int(malformed["expected_status"]) < 300:
            raise VerifierFixtureError("Verifier fixture case 'malformed' must expect a non-success HTTP status.")

    if reward_range is not None and require_expected_values:
        minimum = _finite_number(reward_range[0], label="reward.range minimum")
        maximum = _finite_number(reward_range[1], label="reward.range maximum")
        full_reward = _finite_number(full.get("expected_reward"), label="full_reward expected_reward")
        zero_reward = _finite_number(zero.get("expected_reward"), label="zero_reward expected_reward")
        full_endpoint = maximum if higher_is_better else minimum
        zero_endpoint = minimum if higher_is_better else maximum
        if full_reward != full_endpoint:
            raise VerifierFixtureError(
                f"reward.range [{minimum:g}, {maximum:g}] with higher_is_better={higher_is_better!s} "
                f"but fixture full-reward case expects {full_reward:g}."
            )
        if zero_reward != zero_endpoint:
            raise VerifierFixtureError(
                f"reward.range [{minimum:g}, {maximum:g}] with higher_is_better={higher_is_better!s} "
                f"but fixture zero-reward case expects {zero_reward:g}."
            )
        if determinism == "seeded" and reseed is not None:
            reseed_reward = _finite_number(reseed.get("expected_reward"), label="determinism_reseed expected_reward")
            if reseed_reward != full_reward:
                raise VerifierFixtureError(
                    "Fixture determinism re-seed must expect the same reward as the full-reward case."
                )

    if determinism == "seeded" and reseed is None:  # defensive; covered by the required-case check
        raise VerifierFixtureError("determinism=seeded requires the determinism_reseed case.")


def _actual_reward(response: _Response, case_name: str) -> int | float:
    try:
        payload = response.json()
    except Exception as error:
        raise VerifierFixtureError(
            f"Verifier fixture case '{case_name}' returned a non-JSON successful response."
        ) from error
    if not isinstance(payload, Mapping) or "reward" not in payload:
        raise VerifierFixtureError(f"Verifier fixture case '{case_name}' response is missing numeric 'reward'.")
    reward = payload["reward"]
    _finite_number(reward, label=f"Verifier fixture case '{case_name}' response reward")
    return reward


def _atomic_write_fixture(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(json.dumps(dict(case), separators=(",", ":")) + "\n" for case in cases)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    atomic_write_text(path, content, create_parent=True, mode=mode)


def exercise_verifier_fixture(
    client_factory: Callable[[], _Client],
    fixture_path: str | Path,
    *,
    reward_range: tuple[int | float, int | float] | None = None,
    higher_is_better: bool = True,
    determinism: str | None = None,
    update_expected: bool = False,
) -> None:
    """Score every canned request and optionally regenerate its expectations.

    A fresh in-process client is used for a re-seed, which resets server-local
    state without provisioning any service.  Expectations are replaced atomically
    only after every case and the manifest-dependent scoring contract succeed.
    """

    path = Path(fixture_path)
    source_cases = load_verifier_fixture(path)
    validate_verifier_fixture(
        source_cases,
        reward_range=reward_range,
        higher_is_better=higher_is_better,
        determinism=determinism,
        require_expected_values=not update_expected,
    )

    actual_cases: list[dict[str, Any]] = []
    for source_case in source_cases:
        case = dict(source_case)
        case_name = str(case["case"])
        client = client_factory()
        for step_index, step in enumerate(case.get("setup") or [], 1):
            setup_response = client.post(str(step["path"]), json=step["request"])
            setup_status = int(step.get("expected_status", 200))
            if setup_response.status_code != setup_status:
                raise VerifierFixtureError(
                    f"Verifier fixture case '{case_name}' setup step {step_index} expected HTTP "
                    f"{setup_status}, got {setup_response.status_code}."
                )
        response = client.post("/verify", json=case["request"])
        actual_status = response.status_code
        actual_reward = (
            _actual_reward(response, case_name) if 200 <= actual_status < 300 and case_name != MALFORMED_CASE else None
        )

        if determinism == "seeded" and case_name == DETERMINISM_RESEED_CASE:
            repeated_client = client_factory()
            for step in case.get("setup") or []:
                repeated_setup = repeated_client.post(str(step["path"]), json=step["request"])
                setup_status = int(step.get("expected_status", 200))
                if repeated_setup.status_code != setup_status:
                    raise VerifierFixtureError(
                        f"Verifier fixture case '{case_name}' re-seed setup expected HTTP "
                        f"{setup_status}, got {repeated_setup.status_code}."
                    )
            repeated = repeated_client.post("/verify", json=case["request"])
            try:
                first_payload = response.json()
                repeated_payload = repeated.json()
            except Exception as error:
                raise VerifierFixtureError(
                    f"Verifier fixture case '{case_name}' cannot compare determinism for a non-JSON response."
                ) from error
            if repeated.status_code != actual_status or repeated_payload != first_payload:
                raise VerifierFixtureError(
                    f"Verifier fixture case '{case_name}' is not deterministic after re-seeding."
                )

        if update_expected:
            case["expected_status"] = actual_status
            if actual_reward is None:
                case.pop("expected_reward", None)
            else:
                case["expected_reward"] = actual_reward
        else:
            expected_status = int(case["expected_status"])
            if actual_status != expected_status:
                raise VerifierFixtureError(
                    f"Verifier fixture case '{case_name}' expected HTTP {expected_status}, got {actual_status}."
                )
            if "expected_reward" in case:
                expected_reward = _finite_number(
                    case["expected_reward"], label=f"Verifier fixture case '{case_name}' expected_reward"
                )
                if actual_reward is None or not math.isclose(float(actual_reward), expected_reward):
                    raise VerifierFixtureError(
                        f"Verifier fixture case '{case_name}' expected reward {expected_reward:g}, "
                        f"got {actual_reward!r}."
                    )
        actual_cases.append(case)

    # Status semantics and the deterministic floor cannot be regenerated away.
    # Endpoint expectations may intentionally be changing, so the manifest-aware
    # CLI checks them after the atomic update and gives the author a direct error.
    validate_verifier_fixture(
        actual_cases,
        reward_range=None if update_expected else reward_range,
        higher_is_better=higher_is_better,
        determinism=determinism,
    )
    if update_expected:
        _atomic_write_fixture(path, actual_cases)


__all__ = [
    "DETERMINISM_ENV_VAR",
    "DETERMINISM_RESEED_CASE",
    "FULL_REWARD_CASE",
    "HIGHER_IS_BETTER_ENV_VAR",
    "MALFORMED_CASE",
    "REQUIRED_CASES",
    "SEEDED_REQUIRED_CASES",
    "REWARD_RANGE_ENV_VAR",
    "UPDATE_EXPECTED_ENV_VAR",
    "VerifierFixtureError",
    "ZERO_REWARD_CASE",
    "build_offline_verifier_app",
    "exercise_verifier_fixture",
    "load_verifier_fixture",
    "validate_verifier_fixture",
    "verifier_fixture_environment",
]
