# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.environment_manifest import EnvironmentManifest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY
from nemo_gym.rollout_reverification import _yield_inputs_and_rollouts_paired
from nemo_gym.trajectory_bundle import (
    DEFAULT_TRAJECTORY_IDENTITY_FIELDS,
    TRAJECTORY_ID_KEY,
    CapturedEnvironment,
    FailureReplaySelection,
    captured_environment_from_config,
    load_trajectory_bundle,
    read_trajectory_bundle,
    stable_trajectory_id,
    stamp_trajectory_id,
    validate_trajectory_resume,
    validate_verifier_compatibility,
    write_trajectory_bundle,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))


def _environment(**updates) -> CapturedEnvironment:
    return CapturedEnvironment(
        name="gdpval",
        kind="benchmark",
        version="1.2.3",
        composition_hash="a" * 64,
        integration_profile="custom-driver",
        resources_server="gdpval",
        grading_mode="comparison",
        **updates,
    )


def _manifest(**updates) -> EnvironmentManifest:
    return EnvironmentManifest.model_validate(
        {
            "name": "gdpval",
            "version": "2.0.0",
            "kind": "benchmark",
            "integration_profile": "custom-driver",
            "domain": "other",
            "description": "GDPVal replay contract fixture.",
            "modality": "text",
            "licensing": "Apache-2.0",
            "authors": ["fixture-owner"],
            "reward": {"range": [0, 1], "higher_is_better": True},
            "resources_server": "gdpval",
            "datasets": [
                {
                    "name": "fixture",
                    "type": "benchmark",
                    "jsonl_fpath": "fixture.jsonl",
                    "prepare_script": "prepare.py",
                    "prompt_config": "prompt.yaml",
                }
            ],
            "rollout_driver": "resources_servers.gdpval.multistage_orchestrator:run_rollout_collection",
            "canonical_split": "test",
            "standard_prompt_config": "prompt.yaml",
            "grading_mode": "comparison",
            **updates,
        }
    )


def test_bundle_is_self_describing_and_detects_artifact_drift(tmp_path: Path) -> None:
    inputs = tmp_path / "run_materialized_inputs.jsonl"
    successes = tmp_path / "run.jsonl"
    failures = failures_path_for(successes)
    row = {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 2, "stage_index": 3}
    stamp_trajectory_id(row)
    _write_jsonl(inputs, [row])
    _write_jsonl(successes, [{**row, "response": {"output": []}, "reward": 1.0}])
    _write_jsonl(failures, [])

    bundle_path = write_trajectory_bundle(
        rollouts_path=successes,
        materialized_inputs_path=inputs,
        environment=_environment(),
    )
    bundle, artifacts = load_trajectory_bundle(bundle_path)

    assert bundle.schema_version == 1
    assert bundle.environment == _environment()
    assert bundle.trajectory_identity_fields == DEFAULT_TRAJECTORY_IDENTITY_FIELDS
    assert artifacts == {"inputs": inputs, "successes": successes, "failures": failures}
    assert bundle.artifacts.inputs.rows == 1
    assert bundle.artifacts.successes.rows == 1
    assert bundle.artifacts.failures is not None and bundle.artifacts.failures.rows == 0

    successes.write_text("{}\n")
    assert read_trajectory_bundle(bundle_path).environment == _environment()
    with pytest.raises(ConfigError, match="changed after capture"):
        load_trajectory_bundle(bundle_path)


def test_resume_requires_matching_capture_provenance(tmp_path: Path) -> None:
    inputs = tmp_path / "run_materialized_inputs.jsonl"
    successes = tmp_path / "run.jsonl"
    _write_jsonl(inputs, [])
    _write_jsonl(successes, [])

    with pytest.raises(ConfigError, match="without trajectory bundle"):
        validate_trajectory_resume(
            rollouts_path=successes,
            materialized_inputs_path=inputs,
            environment=_environment(),
        )

    write_trajectory_bundle(
        rollouts_path=successes,
        materialized_inputs_path=inputs,
        environment=_environment(),
    )
    assert (
        validate_trajectory_resume(
            rollouts_path=successes,
            materialized_inputs_path=inputs,
            environment=_environment(),
        )
        is not None
    )

    with pytest.raises(ConfigError, match="captured environment or composition"):
        validate_trajectory_resume(
            rollouts_path=successes,
            materialized_inputs_path=inputs,
            environment=CapturedEnvironment.model_validate(
                {**_environment().model_dump(mode="json"), "composition_hash": "b" * 64}
            ),
        )


def test_manifest_bound_capture_records_composition_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_gym.environment_execution as environment_execution
    import nemo_gym.environment_validation as environment_validation

    manifest = _manifest()
    monkeypatch.setattr(
        environment_execution,
        "preflight_manifest_execution",
        lambda _config: SimpleNamespace(manifest=manifest),
    )
    monkeypatch.setattr(
        environment_validation,
        "inspect_workload",
        lambda *_args, **_kwargs: SimpleNamespace(composition_hash="b" * 64),
    )

    captured = captured_environment_from_config({"manifest_path": "benchmarks/gdpval/manifest.yaml"})

    assert captured == CapturedEnvironment(
        name="gdpval",
        kind="benchmark",
        version="2.0.0",
        composition_hash="b" * 64,
        integration_profile="custom-driver",
        resources_server="gdpval",
        grading_mode="comparison",
        rollout_driver="resources_servers.gdpval.multistage_orchestrator:run_rollout_collection",
    )


def test_capture_records_resolved_composition_after_component_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_gym.environment_execution as environment_execution
    import nemo_gym.environment_validation as environment_validation

    manifest = _manifest()
    monkeypatch.setattr(
        environment_execution,
        "preflight_manifest_execution",
        lambda _config: SimpleNamespace(manifest=manifest),
    )
    monkeypatch.setattr(
        environment_validation,
        "inspect_workload",
        lambda *_args, **_kwargs: SimpleNamespace(composition_hash="c" * 64),
    )

    captured = captured_environment_from_config(
        {
            "manifest_path": "benchmarks/gdpval/manifest.yaml",
            "environment_component_swaps": {"model_server": {"selected": "replacement"}},
        }
    )

    assert captured is not None
    assert captured.composition_hash == "c" * 64


def test_stable_trajectory_identity_includes_stage_and_supports_driver_fields() -> None:
    base = {TASK_INDEX_KEY_NAME: 4, ROLLOUT_INDEX_KEY_NAME: 2, "stage_index": 0, "candidate": "a"}
    later_stage = {**base, "stage_index": 1}

    assert stable_trajectory_id(base) != stable_trajectory_id(later_stage)
    custom_fields = (*DEFAULT_TRAJECTORY_IDENTITY_FIELDS, "candidate")
    custom = dict(base)
    identifier = stamp_trajectory_id(custom, custom_fields)
    assert custom[TRAJECTORY_ID_KEY] == identifier
    assert identifier != stable_trajectory_id({**base, "candidate": "b"}, custom_fields)


def test_verifier_compatibility_allows_new_version_but_rejects_wrong_contract(tmp_path: Path) -> None:
    inputs = tmp_path / "run_materialized_inputs.jsonl"
    successes = tmp_path / "run.jsonl"
    _write_jsonl(inputs, [])
    _write_jsonl(successes, [])
    bundle_path = write_trajectory_bundle(
        rollouts_path=successes,
        materialized_inputs_path=inputs,
        environment=_environment(),
    )
    bundle, _ = load_trajectory_bundle(bundle_path)

    decisions = validate_verifier_compatibility(bundle, _manifest())
    assert "gdpval@1.2.3" in decisions[0]

    with pytest.raises(ConfigError, match="resources_server"):
        validate_verifier_compatibility(bundle, _manifest(resources_server="other_verifier"))

    decisions = validate_verifier_compatibility(
        bundle,
        _manifest(resources_server="other_verifier", grading_mode="replacement"),
        allow_verifier_change=True,
    )
    assert "explicit verifier replacement accepted with --force" in decisions[-1]

    with pytest.raises(ConfigError, match="name"):
        validate_verifier_compatibility(
            bundle,
            _manifest(name="another_environment", resources_server="other_verifier"),
            allow_verifier_change=True,
        )


def test_failure_replay_uses_latest_replayable_attempt_and_stage_identity(tmp_path: Path) -> None:
    inputs_path = tmp_path / "inputs.jsonl"
    successes_path = tmp_path / "rollouts.jsonl"
    failures_path = failures_path_for(successes_path)
    stage_zero = {
        TASK_INDEX_KEY_NAME: 0,
        ROLLOUT_INDEX_KEY_NAME: 0,
        "stage_index": 0,
        "agent_ref": {"name": "agent"},
    }
    stage_one = {**stage_zero, "stage_index": 1}
    _write_jsonl(inputs_path, [stage_zero, stage_one])
    _write_jsonl(successes_path, [{**stage_zero, "response": {"attempt": "success"}}])
    _write_jsonl(
        failures_path,
        [
            {
                **stage_zero,
                "response": {"attempt": "shadowed"},
                NG_FAILURE_CLASS_KEY: "judge_failed",
            },
            {
                **stage_one,
                "response": {"attempt": "old"},
                NG_FAILURE_CLASS_KEY: "judge_failed",
            },
            {
                **stage_one,
                "response": {"attempt": "latest"},
                NG_FAILURE_CLASS_KEY: "judge_failed",
            },
            {**stage_one, NG_FAILURE_CLASS_KEY: "transport_failed"},
        ],
    )

    pairs = list(
        _yield_inputs_and_rollouts_paired(
            inputs_path,
            successes_path,
            failure_rollouts_jsonl_fpath=failures_path,
            failure_trajectories=FailureReplaySelection.LATEST_REPLAYABLE,
        )
    )

    assert [(pair.input["stage_index"], pair.rollout["response"]["attempt"]) for pair in pairs] == [
        (0, "success"),
        (1, "latest"),
    ]
