# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The resume identity saved beside a run's manifest never holds a credential value.

Every value a credential-named setting carries, at any depth, is redacted before interpolations are resolved, so an
environment credential is never read into the identity. A canary credential set literally and through the environment
must appear in no file the run writes, while a resume still accepts the unchanged configuration and refuses a semantic
change, also for a run saved by an earlier Gym whose identity file still holds credentials.
"""

import json
import os
import stat
from pathlib import Path

import pytest

from nemo_gym.config.config_types import ConfigError
from nemo_gym.rollouts.rollout_recovery import (
    RunManifest,
    _configuration_identity,
    _digest,
    _without_null_defaults,
    config_identity_path_for,
    validate_resume,
)


CANARY = "CANARY-SECRET-123"  # pragma: allowlist secret
OTHER_CANARY = "CANARY-SECRET-456"  # pragma: allowlist secret


def credentialed_config(secret: str, model: str = "demo-model") -> dict:
    return {
        "exa_api_key": secret,
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {
                    "model": model,
                    "base_url": "http://127.0.0.1:8000/v1",
                    "client": {"headers": {"Authorization": f"Bearer {secret}", "X-Session": secret}},
                }
            }
        },
        "search_tools": {
            "resources_servers": {
                "web_search": {
                    "exa_api_key": secret,
                    "search_key_ref": "${exa_api_key}",
                    "harbor": {"secrets": {"X_TOKEN": secret, "nested": {"anything": secret}}},
                    "env": [{"name": "SERVICE_TOKEN", "value": secret}, {"API_KEY": secret}, f"HF_TOKEN={secret}"],
                    "my_service_pw": secret,
                    "service_key": secret,
                    "reference_key": "expected_answer",
                    "artifact_url": f"https://reader:{secret}@artifacts.example.com/store",
                    "judge_prompt": "Grade the answer.",
                }
            }
        },
    }


@pytest.fixture(params=["literal", "environment"])
def secret(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """The canary as a literal value, or as an environment interpolation resolved from it."""
    if request.param == "literal":
        return CANARY
    monkeypatch.setenv("GYM_TEST_CANARY", CANARY)
    return "${oc.env:GYM_TEST_CANARY}"


def save_run(tmp_path: Path, global_config: dict) -> tuple[Path, Path]:
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n")
    materialized = tmp_path / "materialized.jsonl"
    materialized.write_text("")
    path = tmp_path / "rollouts_manifest.json"
    RunManifest.create(source, [], {}, global_config).write(path)
    return path, materialized


def current(tmp_path: Path, global_config: dict) -> RunManifest:
    return RunManifest.create(tmp_path / "input.jsonl", [], {}, global_config)


def files_holding_canary(root: Path) -> list[str]:
    found = []
    for directory, _, names in os.walk(root):
        for name in names:
            if CANARY.encode() in (Path(directory) / name).read_bytes():
                found.append(name)
    return found


def test_no_file_the_run_writes_holds_a_credential(tmp_path: Path, secret: str) -> None:
    path, materialized = save_run(tmp_path, credentialed_config(secret))

    assert validate_resume(path, current(tmp_path, credentialed_config(secret)), materialized) is not None

    assert config_identity_path_for(path).exists()
    assert files_holding_canary(tmp_path) == []


def test_the_identity_keeps_the_semantic_settings_and_marks_credentials_redacted(tmp_path: Path) -> None:
    identity = _configuration_identity({}, credentialed_config(CANARY), [])
    settings = identity["servers"]["search_tools"]["resources_servers"]["web_search"]

    assert settings["judge_prompt"] == "Grade the answer."
    assert settings["exa_api_key"] == settings["my_service_pw"] == settings["service_key"] == "<redacted>"
    # A `..._key` setting that selects what is measured stays, so a resume still sees it change.
    assert settings["reference_key"] == "expected_answer"
    model = identity["servers"]["policy_model"]["responses_api_models"]["vllm_model"]
    assert model["client"]["headers"] == {"Authorization": "<redacted>", "X-Session": "<redacted>"}
    assert settings["harbor"]["secrets"] == {"X_TOKEN": "<redacted>", "nested": {"anything": "<redacted>"}}
    assert CANARY not in json.dumps(identity)


def test_the_identity_sidecar_is_private_to_its_owner(tmp_path: Path) -> None:
    previous = os.umask(0o022)
    try:
        path, _ = save_run(tmp_path, credentialed_config(CANARY))
    finally:
        os.umask(previous)

    assert stat.S_IMODE(config_identity_path_for(path).stat().st_mode) == 0o600


def test_a_changed_credential_is_not_drift(tmp_path: Path) -> None:
    path, materialized = save_run(tmp_path, credentialed_config(CANARY))

    assert validate_resume(path, current(tmp_path, credentialed_config(OTHER_CANARY)), materialized) is not None


def test_a_semantic_change_still_refuses_and_is_named(tmp_path: Path) -> None:
    path, materialized = save_run(tmp_path, credentialed_config(CANARY))

    with pytest.raises(ConfigError, match="semantic settings differ: .*model"):
        validate_resume(path, current(tmp_path, credentialed_config(CANARY, model="other-model")), materialized)


def save_run_as_earlier_gym(tmp_path: Path, global_config: dict, *, with_identity: bool = True) -> tuple[Path, Path]:
    """A run saved before the identity was credential-safe: its digest and identity file hold the resolved values."""
    path, materialized = save_run(tmp_path, global_config)
    earlier = _without_null_defaults(_configuration_identity({}, global_config, [], credential_safe=False))
    manifest = RunManifest.model_validate_json(path.read_bytes())
    path.write_text(json.dumps(manifest.model_dump(mode="json") | {"config_digest": _digest(earlier)}))
    identity = config_identity_path_for(path)
    identity.unlink()
    if with_identity:
        identity.write_text(json.dumps(earlier, indent=2))
        identity.chmod(0o644)
    return path, materialized


@pytest.mark.parametrize("with_identity", [True, False])
def test_a_run_saved_by_an_earlier_gym_resumes_on_the_unchanged_configuration(
    tmp_path: Path, with_identity: bool
) -> None:
    path, materialized = save_run_as_earlier_gym(tmp_path, credentialed_config(CANARY), with_identity=with_identity)
    saved = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    assert validate_resume(path, current(tmp_path, credentialed_config(CANARY)), materialized) is not None
    # Resuming reads the earlier identity file; it never rewrites it.
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.name in saved} == saved


def test_a_run_saved_by_an_earlier_gym_is_compared_on_its_credential_safe_form(tmp_path: Path) -> None:
    path, materialized = save_run_as_earlier_gym(tmp_path, credentialed_config(CANARY))

    # A rotated credential no longer matches the earlier digest, and the earlier identity re-read without
    # credentials shows no difference.
    assert validate_resume(path, current(tmp_path, credentialed_config(OTHER_CANARY)), materialized) is not None
    with pytest.raises(ConfigError, match="semantic settings differ: .*model"):
        validate_resume(path, current(tmp_path, credentialed_config(CANARY, model="other-model")), materialized)
