# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A new collection setting with a default must not change the resume fingerprint of a run saved before it existed.

``dispatch_connect_grace_s`` was added to the collection config with a default. A run saved before it (its digest has no
such key) and a run saved after it (its digest has the default) both resume on a Gym that leaves it out of the digest.
"""

import json
from pathlib import Path

import pytest

import nemo_gym.rollouts.rollout_recovery as recovery
from nemo_gym.config.config_types import ConfigError
from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig
from nemo_gym.rollouts.rollout_recovery import (
    RunManifest,
    _configuration_identity,
    _digest,
    _without_null_defaults,
    config_identity_path_for,
    resume_drift_path_for,
    validate_resume,
)


FIELD = "dispatch_connect_grace_s"
GLOBAL = {
    "policy_model": {"responses_api_models": {"vllm_model": {"model": "demo-model", "request_timeout_s": 600}}},
}


@pytest.fixture(autouse=True)
def _fresh_digest_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registries of recorded digests are per process; each test starts with none, as a resuming job does."""
    monkeypatch.setattr(recovery, "_LEGACY_CONFIG_DIGESTS", {})
    monkeypatch.setattr(recovery, "_CONFIG_IDENTITIES", {})


def collection(**overrides) -> dict:
    return RolloutCollectionConfig(
        input_jsonl_fpath="in.jsonl", output_jsonl_fpath="out.jsonl", **overrides
    ).model_dump()


def without_field(config: dict) -> dict:
    return {key: value for key, value in config.items() if key != FIELD}


def digest_with(config: dict, global_config: dict = GLOBAL, **setting) -> str:
    """The digest a Gym that recorded the setting took: it is in the collection identity."""
    identity = _configuration_identity(config, global_config, [])
    identity["collection"] = {**identity["collection"], **setting}
    return _digest(_without_null_defaults(identity))


def saved_run(tmp_path: Path, config: dict, *, digest: str | None = None, identity: bool = False) -> tuple[Path, Path]:
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n")
    materialized = tmp_path / "materialized.jsonl"
    materialized.write_text("")
    path = tmp_path / "rollouts_manifest.json"
    manifest = RunManifest.create(source, [], config, GLOBAL)
    saved_identity = config_identity_path_for(path)
    manifest.write(path)
    if digest is not None:
        manifest = manifest.model_copy(update={"config_digest": digest})
        manifest.write(path)
    if not identity:
        saved_identity.unlink(missing_ok=True)  # a manifest that saved only its digest
    # The run is resumed by another process, which knows nothing of the digests this one recorded saving it.
    recovery._LEGACY_CONFIG_DIGESTS.clear()
    recovery._CONFIG_IDENTITIES.clear()
    return path, materialized


def current(tmp_path: Path, config: dict, global_config: dict = GLOBAL) -> RunManifest:
    return RunManifest.create(tmp_path / "input.jsonl", [], config, global_config)


def test_the_setting_has_a_default_and_is_not_in_the_digest() -> None:
    config = collection()
    assert config[FIELD] == 60.0
    assert current_digest_ignores(config)


def current_digest_ignores(config: dict) -> bool:
    return _digest(_without_null_defaults(_configuration_identity(config, GLOBAL, []))) == _digest(
        _without_null_defaults(_configuration_identity({**config, FIELD: 5.0}, GLOBAL, []))
    )


def test_a_digest_only_run_saved_before_the_setting_existed_resumes(tmp_path: Path) -> None:
    # Saved by a Gym whose collection config had no such key: its digest is the identity without it.
    before = digest_with(without_field(collection()))
    path, materialized = saved_run(tmp_path, without_field(collection()), digest=before)
    assert not config_identity_path_for(path).exists()

    resumed = validate_resume(path, current(tmp_path, collection()), materialized)

    assert resumed is not None and not resumed.identity_overridden
    assert not resume_drift_path_for(path).exists()


def test_a_digest_only_run_saved_with_the_default_still_resumes(tmp_path: Path) -> None:
    born_with_default = digest_with(collection(), **{FIELD: 60.0})
    path, materialized = saved_run(tmp_path, collection(), digest=born_with_default)

    assert validate_resume(path, current(tmp_path, collection()), materialized) is not None
    # The setting is operational: resuming it with another value is not refused either.
    assert validate_resume(path, current(tmp_path, collection(**{FIELD: 5.0})), materialized) is not None


def test_a_non_default_value_is_operational_and_never_refuses(tmp_path: Path) -> None:
    # A saved identity names the difference as operational, so a run saved with a non-default value resumes with the
    # drift recorded (first, so no digest recorded in this process knows the value).
    (tmp_path / "id").mkdir()
    born_with_five = digest_with(collection(**{FIELD: 5.0}), **{FIELD: 5.0})
    path, materialized = saved_run(tmp_path / "id", collection(**{FIELD: 5.0}), digest=born_with_five, identity=True)
    saved_identity = _configuration_identity(collection(**{FIELD: 5.0}), GLOBAL, [])
    saved_identity["collection"][FIELD] = 5.0
    config_identity_path_for(path).write_text(json.dumps(_without_null_defaults(saved_identity)))
    assert validate_resume(path, current(tmp_path / "id", collection()), materialized) is not None
    (entry,) = [json.loads(line) for line in resume_drift_path_for(path).read_text().splitlines()]
    assert entry["operational_keys"] == [f"collection.{FIELD}"]

    # A run saved before the setting existed resumes with another value, with nothing to record.
    before = digest_with(without_field(collection()))
    path, materialized = saved_run(tmp_path, without_field(collection()), digest=before)
    assert validate_resume(path, current(tmp_path, collection(**{FIELD: 5.0})), materialized) is not None
    assert not resume_drift_path_for(path).exists()


@pytest.mark.parametrize("digest_only", [True, False])
def test_a_semantic_change_still_refuses(tmp_path: Path, digest_only: bool) -> None:
    before = digest_with(without_field(collection()))
    path, materialized = saved_run(tmp_path, without_field(collection()), digest=before, identity=not digest_only)
    changed = {
        "policy_model": {"responses_api_models": {"vllm_model": {"model": "other-model", "request_timeout_s": 600}}}
    }

    with pytest.raises(ConfigError, match="incompatible resolved configuration") as raised:
        validate_resume(path, current(tmp_path, collection(), changed), materialized)

    if digest_only:
        assert "saved only a digest" in str(raised.value)
