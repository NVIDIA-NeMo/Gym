# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned artifact contract for captured rollout trajectories."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import orjson
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.config_types import ConfigError
from nemo_gym.environment_manifest import EnvironmentManifest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.path_utils import failures_path_for


TRAJECTORY_BUNDLE_SCHEMA_VERSION = 1
TRAJECTORY_ID_KEY = "_ng_trajectory_id"
STAGE_INDEX_KEY = "stage_index"
DEFAULT_TRAJECTORY_IDENTITY_FIELDS = (
    STAGE_INDEX_KEY,
    TASK_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
)


class FailureReplaySelection(str, Enum):
    """Failure-sidecar rows selected for reward replay."""

    EXCLUDE = "exclude"
    JUDGE_FAILED = "judge-failed"
    LATEST_REPLAYABLE = "latest-replayable"


class _BundleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CapturedEnvironment(_BundleModel):
    name: str = Field(min_length=1)
    kind: Literal["environment", "benchmark"]
    version: str = Field(min_length=1)
    composition_hash: str
    integration_profile: str = Field(min_length=1)
    resources_server: str = Field(min_length=1)
    grading_mode: str | None = None
    session_model: str | None = None
    state: str | None = None
    rollout_driver: str | None = None

    @field_validator("composition_hash")
    @classmethod
    def _validate_composition_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("composition_hash must be a lowercase SHA-256 digest")
        return value


class CaptureArtifact(_BundleModel):
    path: str = Field(min_length=1)
    sha256: str
    rows: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("artifact paths must stay below the bundle directory")
        return path.as_posix()

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return value


class CaptureArtifacts(_BundleModel):
    inputs: CaptureArtifact
    successes: CaptureArtifact
    failures: CaptureArtifact | None = None


class FailureSemantics(_BundleModel):
    success_precedence: bool = True
    attempt_selection: Literal["latest"] = "latest"
    replay_default: FailureReplaySelection = FailureReplaySelection.LATEST_REPLAYABLE
    requires_response: bool = True
    non_persisted_failures: Literal["omitted"] = "omitted"


class TrajectoryBundle(_BundleModel):
    schema_version: Literal[1] = TRAJECTORY_BUNDLE_SCHEMA_VERSION
    environment: CapturedEnvironment | None
    trajectory_identity_fields: tuple[str, ...] = DEFAULT_TRAJECTORY_IDENTITY_FIELDS
    artifacts: CaptureArtifacts
    failure_semantics: FailureSemantics = Field(default_factory=FailureSemantics)

    @field_validator("trajectory_identity_fields")
    @classmethod
    def _validate_identity_fields(cls, fields: tuple[str, ...]) -> tuple[str, ...]:
        if not fields or any(not isinstance(field, str) or not field for field in fields):
            raise ValueError("trajectory_identity_fields must contain non-empty field names")
        if len(fields) != len(set(fields)):
            raise ValueError("trajectory_identity_fields must be unique")
        required = {TASK_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME}
        if not required.issubset(fields):
            raise ValueError("trajectory_identity_fields must include task and rollout indices")
        return fields


class TrajectoryResumeCheckpoint(_BundleModel):
    schema_version: Literal[1] = TRAJECTORY_BUNDLE_SCHEMA_VERSION
    environment: CapturedEnvironment | None
    trajectory_identity_fields: tuple[str, ...] = DEFAULT_TRAJECTORY_IDENTITY_FIELDS
    inputs: CaptureArtifact


def bundle_path_for(rollouts_path: str | Path) -> Path:
    path = Path(rollouts_path)
    return path.with_suffix(".bundle.json")


def resume_checkpoint_path_for(rollouts_path: str | Path) -> Path:
    path = Path(rollouts_path)
    return path.with_suffix(".resume.json")


def _identity_value(row: Mapping[str, Any], field: str) -> Any:
    if field == STAGE_INDEX_KEY and field not in row:
        return 0
    if field not in row:
        raise ConfigError(f"Trajectory row is missing identity field '{field}'.")
    return row[field]


def stable_trajectory_id(
    row: Mapping[str, Any],
    identity_fields: Sequence[str] = DEFAULT_TRAJECTORY_IDENTITY_FIELDS,
) -> str:
    fields = tuple(identity_fields)
    try:
        payload = orjson.dumps([[field, _identity_value(row, field)] for field in fields])
    except TypeError as error:
        raise ConfigError(f"Trajectory identity fields are not JSON-serializable: {error}.") from error
    return "ngt1-" + hashlib.sha256(payload).hexdigest()


def trajectory_identity_key(
    row: Mapping[str, Any],
    identity_fields: Sequence[str] = DEFAULT_TRAJECTORY_IDENTITY_FIELDS,
) -> str:
    expected = stable_trajectory_id(row, identity_fields)
    recorded = row.get(TRAJECTORY_ID_KEY)
    if recorded is not None and recorded != expected:
        raise ConfigError(f"Trajectory row has stale {TRAJECTORY_ID_KEY}={recorded!r}; expected {expected!r}.")
    return expected


def stamp_trajectory_id(
    row: dict[str, Any],
    identity_fields: Sequence[str] = DEFAULT_TRAJECTORY_IDENTITY_FIELDS,
    *,
    overwrite: bool = False,
) -> str:
    expected = stable_trajectory_id(row, identity_fields)
    recorded = row.get(TRAJECTORY_ID_KEY)
    if recorded is not None and recorded != expected and not overwrite:
        raise ConfigError(f"Trajectory row has stale {TRAJECTORY_ID_KEY}={recorded!r}; expected {expected!r}.")
    row[TRAJECTORY_ID_KEY] = expected
    return expected


def captured_environment_from_config(
    global_config: Mapping[str, Any] | DictConfig,
) -> CapturedEnvironment | None:
    """Resolve immutable environment provenance before rollout collection starts."""

    manifest_path = global_config.get("manifest_path")
    if not manifest_path:
        return None

    from nemo_gym.environment_execution import preflight_manifest_execution
    from nemo_gym.environment_validation import inspect_workload

    preflight = preflight_manifest_execution(global_config)
    if preflight is None:
        raise ConfigError("Manifest-bound rollout capture could not resolve its environment manifest.")
    manifest = preflight.manifest
    if manifest.resources_server is None:
        raise ConfigError(f"Environment '{manifest.name}' does not declare a verifier resources server.")
    composition_hash = inspect_workload(global_config, manifest=manifest).composition_hash
    return CapturedEnvironment(
        name=manifest.name,
        kind=manifest.kind.value,
        version=manifest.version,
        composition_hash=str(composition_hash),
        integration_profile=manifest.integration_profile.value,
        resources_server=manifest.resources_server,
        grading_mode=manifest.grading_mode,
        session_model=manifest.session_model.value if manifest.session_model is not None else None,
        state=manifest.state.value if manifest.state is not None else None,
        rollout_driver=manifest.rollout_driver,
    )


def validate_verifier_compatibility(
    bundle: TrajectoryBundle,
    manifest: EnvironmentManifest,
    *,
    allow_verifier_change: bool = False,
) -> tuple[str, ...]:
    """Reject replay through a verifier contract belonging to another environment."""

    captured = bundle.environment
    if captured is None:
        raise ConfigError(
            "This trajectory bundle was not captured from a manifest-bound environment and cannot be used by "
            "`gym env test --replay`."
        )
    current = {
        "name": manifest.name,
        "kind": manifest.kind.value,
        "resources_server": manifest.resources_server,
        "grading_mode": manifest.grading_mode,
        "session_model": manifest.session_model.value if manifest.session_model is not None else None,
        "state": manifest.state.value if manifest.state is not None else None,
    }
    original = {
        "name": captured.name,
        "kind": captured.kind,
        "resources_server": captured.resources_server,
        "grading_mode": captured.grading_mode,
        "session_model": captured.session_model,
        "state": captured.state,
    }
    mismatches = [
        f"{field}: captured={original[field]!r}, selected={current[field]!r}"
        for field in original
        if original[field] != current[field]
        and not (allow_verifier_change and field in {"resources_server", "grading_mode"})
    ]
    if mismatches:
        raise ConfigError(
            "Captured trajectories are incompatible with the selected verifier contract: " + "; ".join(mismatches)
        )
    decisions = [
        f"capture environment {captured.name}@{captured.version}",
        f"capture composition sha256:{captured.composition_hash}",
    ]
    if captured.resources_server == manifest.resources_server and captured.grading_mode == manifest.grading_mode:
        decisions.append(f"verifier contract {captured.resources_server} is compatible")
    else:
        decisions.append(
            "explicit verifier replacement accepted with --force: "
            f"{captured.resources_server}/{captured.grading_mode} -> "
            f"{manifest.resources_server}/{manifest.grading_mode}"
        )
    return tuple(decisions)


def _artifact_stats(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    rows = 0
    try:
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                if line.strip():
                    rows += 1
    except OSError as error:
        raise ConfigError(f"Could not read capture artifact '{path}': {error}.") from error
    return digest.hexdigest(), rows


def _describe_artifact(path: Path, bundle_dir: Path) -> CaptureArtifact:
    absolute = path.expanduser().resolve()
    if path.is_symlink() or not absolute.is_file():
        raise ConfigError(f"Capture artifact is missing or symbolic-linked: '{path}'.")
    try:
        relative = absolute.relative_to(bundle_dir.resolve())
    except ValueError as error:
        raise ConfigError(f"Capture artifact '{path}' must stay below '{bundle_dir}'.") from error
    digest, rows = _artifact_stats(absolute)
    return CaptureArtifact(path=relative.as_posix(), sha256=digest, rows=rows)


def _read_artifact_rows(path: Path, *, label: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = orjson.loads(line)
                if not isinstance(row, Mapping):
                    raise ConfigError(f"Trajectory {label} row {line_number} must be a JSON object.")
                rows.append(row)
    except orjson.JSONDecodeError as error:
        raise ConfigError(f"Trajectory {label} artifact '{path}' is not valid JSONL: {error}.") from error
    except OSError as error:
        raise ConfigError(f"Could not read trajectory {label} artifact '{path}': {error}.") from error
    return rows


def _validate_artifact_relationships(
    bundle: TrajectoryBundle,
    artifacts: Mapping[str, Path | None],
) -> None:
    inputs_path = artifacts["inputs"]
    successes_path = artifacts["successes"]
    assert inputs_path is not None and successes_path is not None
    input_rows = _read_artifact_rows(inputs_path, label="input")
    success_rows = _read_artifact_rows(successes_path, label="success")
    failure_path = artifacts.get("failures")
    failure_rows = _read_artifact_rows(failure_path, label="failure") if failure_path is not None else []

    input_keys = [trajectory_identity_key(row, bundle.trajectory_identity_fields) for row in input_rows]
    if len(input_keys) != len(set(input_keys)):
        raise ConfigError("Trajectory bundle materialized inputs contain duplicate identities.")
    available = set(input_keys)
    success_keys = [trajectory_identity_key(row, bundle.trajectory_identity_fields) for row in success_rows]
    if len(success_keys) != len(set(success_keys)):
        raise ConfigError("Trajectory bundle successes contain duplicate identities.")
    for label, rows in (("success", success_rows), ("failure", failure_rows)):
        for row in rows:
            identity = trajectory_identity_key(row, bundle.trajectory_identity_fields)
            if identity not in available:
                raise ConfigError(f"Trajectory bundle {label} '{identity}' has no matching materialized input.")


def write_trajectory_bundle(
    *,
    rollouts_path: str | Path,
    materialized_inputs_path: str | Path,
    environment: CapturedEnvironment | None,
    identity_fields: Sequence[str] = DEFAULT_TRAJECTORY_IDENTITY_FIELDS,
) -> Path:
    """Write bundle metadata last, after all referenced artifacts are durable."""

    rollouts = Path(rollouts_path)
    destination = bundle_path_for(rollouts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    failures = failures_path_for(rollouts)
    bundle = TrajectoryBundle(
        environment=environment,
        trajectory_identity_fields=tuple(identity_fields),
        artifacts=CaptureArtifacts(
            inputs=_describe_artifact(Path(materialized_inputs_path), destination.parent),
            successes=_describe_artifact(rollouts, destination.parent),
            failures=_describe_artifact(failures, destination.parent) if failures.is_file() else None,
        ),
    )
    _validate_artifact_relationships(
        bundle,
        {
            "inputs": Path(materialized_inputs_path).expanduser().resolve(),
            "successes": rollouts.expanduser().resolve(),
            "failures": failures.expanduser().resolve() if failures.is_file() else None,
        },
    )
    content = orjson.dumps(bundle.model_dump(mode="json"), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as error:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        raise ConfigError(f"Could not write trajectory bundle '{destination}': {error}.") from error
    return destination


def _atomic_write_metadata(destination: Path, content: bytes, *, label: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as error:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
        raise ConfigError(f"Could not write {label} '{destination}': {error}.") from error
    return destination


def write_trajectory_resume_checkpoint(
    *,
    rollouts_path: str | Path,
    materialized_inputs_path: str | Path,
    environment: CapturedEnvironment | None,
    identity_fields: Sequence[str] = DEFAULT_TRAJECTORY_IDENTITY_FIELDS,
) -> Path:
    """Persist immutable run provenance before any rollout output is appended."""

    destination = resume_checkpoint_path_for(rollouts_path)
    checkpoint = TrajectoryResumeCheckpoint(
        environment=environment,
        trajectory_identity_fields=tuple(identity_fields),
        inputs=_describe_artifact(Path(materialized_inputs_path), destination.parent),
    )
    content = (
        orjson.dumps(checkpoint.model_dump(mode="json"), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"
    )
    return _atomic_write_metadata(destination, content, label="trajectory resume checkpoint")


def _load_trajectory_resume_checkpoint(path: Path) -> tuple[TrajectoryResumeCheckpoint, Path]:
    if path.is_symlink() or not path.is_file():
        raise ConfigError(f"Trajectory resume checkpoint was not found or is symbolic-linked: '{path}'.")
    try:
        checkpoint = TrajectoryResumeCheckpoint.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ConfigError(f"Invalid trajectory resume checkpoint '{path}': {error}.") from error
    return checkpoint, _resolve_artifact(path, checkpoint.inputs)


def _resolve_artifact(bundle_path: Path, artifact: CaptureArtifact) -> Path:
    path = bundle_path.parent / artifact.path
    if path.is_symlink():
        raise ConfigError(f"Trajectory bundle artifact may not be a symbolic link: '{path}'.")
    resolved = path.resolve()
    try:
        resolved.relative_to(bundle_path.parent.resolve())
    except ValueError as error:
        raise ConfigError(f"Trajectory bundle artifact escapes its bundle directory: '{artifact.path}'.") from error
    if not resolved.is_file():
        raise ConfigError(f"Trajectory bundle artifact was not found: '{resolved}'.")
    digest, rows = _artifact_stats(resolved)
    if digest != artifact.sha256 or rows != artifact.rows:
        raise ConfigError(
            f"Trajectory bundle artifact '{resolved}' changed after capture: "
            f"expected sha256:{artifact.sha256} and {artifact.rows} rows, got sha256:{digest} and {rows} rows."
        )
    return resolved


def read_trajectory_bundle(path: str | Path) -> TrajectoryBundle:
    """Read bundle metadata without hashing its potentially large artifacts."""

    bundle_path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if bundle_path.is_symlink() or not bundle_path.is_file():
        raise ConfigError(f"Trajectory bundle was not found or is symbolic-linked: '{bundle_path}'.")
    try:
        return TrajectoryBundle.model_validate_json(bundle_path.read_bytes())
    except (OSError, ValueError) as error:
        raise ConfigError(f"Invalid trajectory bundle '{bundle_path}': {error}.") from error


def load_trajectory_bundle(path: str | Path) -> tuple[TrajectoryBundle, dict[str, Path | None]]:
    """Read bundle metadata and verify every referenced artifact."""

    bundle_path = Path(path).expanduser().resolve()
    bundle = read_trajectory_bundle(bundle_path)
    artifacts: dict[str, Path | None] = {
        "inputs": _resolve_artifact(bundle_path, bundle.artifacts.inputs),
        "successes": _resolve_artifact(bundle_path, bundle.artifacts.successes),
        "failures": (
            _resolve_artifact(bundle_path, bundle.artifacts.failures)
            if bundle.artifacts.failures is not None
            else None
        ),
    }
    _validate_artifact_relationships(bundle, artifacts)
    return bundle, artifacts


def validate_trajectory_resume(
    *,
    rollouts_path: str | Path,
    materialized_inputs_path: str | Path,
    environment: CapturedEnvironment | None,
    identity_fields: Sequence[str] = DEFAULT_TRAJECTORY_IDENTITY_FIELDS,
) -> TrajectoryBundle | None:
    """Require cached rows to retain their original capture provenance."""

    rollouts = Path(rollouts_path).expanduser().resolve()
    inputs = Path(materialized_inputs_path).expanduser().resolve()
    path = bundle_path_for(rollouts)
    checkpoint_path = resume_checkpoint_path_for(rollouts)
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        checkpoint, checkpoint_inputs = _load_trajectory_resume_checkpoint(checkpoint_path)
        mismatches: list[str] = []
        if checkpoint.environment != environment:
            mismatches.append("captured environment or composition")
        if checkpoint.trajectory_identity_fields != tuple(identity_fields):
            mismatches.append("trajectory identity fields")
        if checkpoint_inputs != inputs:
            mismatches.append("materialized inputs path")
        if mismatches:
            raise ConfigError(
                f"Cannot resume '{rollouts}' because its trajectory checkpoint does not match the current capture: "
                + ", ".join(mismatches)
                + ". Start a fresh collection or use the original environment and paths."
            )
        return None
    if not path.is_file():
        if environment is not None:
            raise ConfigError(
                f"Cannot resume manifest-bound rollouts from '{rollouts}' without trajectory bundle '{path}'. "
                "Start a fresh collection or restore the original bundle."
            )
        return None

    bundle, artifacts = load_trajectory_bundle(path)
    mismatches: list[str] = []
    if bundle.environment != environment:
        mismatches.append("captured environment or composition")
    if bundle.trajectory_identity_fields != tuple(identity_fields):
        mismatches.append("trajectory identity fields")
    if artifacts["inputs"] != inputs:
        mismatches.append("materialized inputs path")
    if artifacts["successes"] != rollouts:
        mismatches.append("rollouts path")
    failures = failures_path_for(rollouts).resolve()
    captured_failures = artifacts["failures"]
    if captured_failures != (failures if failures.is_file() else None):
        mismatches.append("failure sidecar lineage")
    if mismatches:
        raise ConfigError(
            f"Cannot resume '{rollouts}' because its trajectory bundle does not match the current capture: "
            + ", ".join(mismatches)
            + ". Start a fresh collection or use the original environment and paths."
        )
    return bundle


__all__ = [
    "DEFAULT_TRAJECTORY_IDENTITY_FIELDS",
    "FailureReplaySelection",
    "TRAJECTORY_BUNDLE_SCHEMA_VERSION",
    "TRAJECTORY_ID_KEY",
    "CaptureArtifact",
    "CapturedEnvironment",
    "TrajectoryBundle",
    "TrajectoryResumeCheckpoint",
    "bundle_path_for",
    "captured_environment_from_config",
    "load_trajectory_bundle",
    "read_trajectory_bundle",
    "resume_checkpoint_path_for",
    "stable_trajectory_id",
    "stamp_trajectory_id",
    "trajectory_identity_key",
    "validate_trajectory_resume",
    "validate_verifier_compatibility",
    "write_trajectory_bundle",
    "write_trajectory_resume_checkpoint",
]
