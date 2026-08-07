# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed contract for an environment ``manifest.yaml``."""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Mapping

import yaml
from packaging.licenses import LICENSES
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from nemo_gym.config_types import ConfigError, Domain


JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SEMVER_PATTERN = (
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)*$"
_SPDX_LICENSE_PATTERN = "|".join(
    re.escape(identifier) for identifier in sorted(entry["id"] for entry in LICENSES.values())
)
LICENSE_PATTERN = rf"^(?:internal|proprietary|unknown|{_SPDX_LICENSE_PATTERN}|LicenseRef-[A-Za-z0-9][A-Za-z0-9.-]*|DocumentRef-[A-Za-z0-9][A-Za-z0-9.-]*:LicenseRef-[A-Za-z0-9][A-Za-z0-9.-]*)$"
SOURCE_PATTERN = r"^(?:(?:https?|ssh|git|file)://\S+|[^@\s]+@[^:\s]+:\S+)$"
CALLABLE_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ManifestName = Annotated[str, StringConstraints(strip_whitespace=True, pattern=NAME_PATTERN)]
SemVer = Annotated[str, StringConstraints(strip_whitespace=True, pattern=SEMVER_PATTERN)]
License = Annotated[str, StringConstraints(strip_whitespace=True, pattern=LICENSE_PATTERN)]
Source = Annotated[str, StringConstraints(strip_whitespace=True, pattern=SOURCE_PATTERN)]
GitRef = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[^\s]+$")]
PythonCallable = Annotated[str, StringConstraints(strip_whitespace=True, pattern=CALLABLE_PATTERN)]


class EnvironmentKind(str, Enum):
    ENVIRONMENT = "environment"
    BENCHMARK = "benchmark"


class IntegrationProfile(str, Enum):
    STOCK_LOOP = "stock-loop"
    MEASURED_LOOP = "measured-loop"
    EXTERNAL_LOOP = "external-loop"
    CUSTOM_DRIVER = "custom-driver"


class Determinism(str, Enum):
    SEEDED = "seeded"
    STOCHASTIC = "stochastic"
    UNKNOWN = "unknown"


class SessionModel(str, Enum):
    EPISODE = "episode"
    STEP = "step"


class EnvironmentState(str, Enum):
    NONE = "none"
    PER_SESSION = "per_session"


class Lifecycle(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class DatasetKind(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    EXAMPLE = "example"
    BENCHMARK = "benchmark"


_PROFILE_REQUIRED_FIELDS = {
    IntegrationProfile.STOCK_LOOP: ("model_server",),
    IntegrationProfile.MEASURED_LOOP: ("model_server",),
    IntegrationProfile.EXTERNAL_LOOP: (),
    IntegrationProfile.CUSTOM_DRIVER: ("rollout_driver",),
}
_BENCHMARK_REQUIRED_FIELDS = ("canonical_split", "standard_prompt_config")


class _ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_default=True)


class Reward(_ManifestModel):
    range: tuple[FiniteFloat, FiniteFloat] = Field(description="Inclusive lower and upper reward endpoints.")
    higher_is_better: bool

    @model_validator(mode="after")
    def validate_range(self) -> "Reward":
        if self.range[0] >= self.range[1]:
            raise ValueError("reward.range must be ordered with lower < upper")
        return self


class ManifestDataset(_ManifestModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"type": {"const": "benchmark"}}, "required": ["type"]},
                    "then": {
                        "properties": {
                            "prepare_script": {"minLength": 1, "type": "string"},
                            "prompt_config": {"minLength": 1, "type": "string"},
                        },
                        "required": ["prepare_script", "prompt_config"],
                    },
                }
            ]
        },
    )

    name: NonEmptyString
    type: DatasetKind
    jsonl_fpath: NonEmptyString
    prepare_script: NonEmptyString | None = None
    prompt_config: NonEmptyString | None = None
    num_repeats: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_benchmark_dataset(self) -> "ManifestDataset":
        if self.type == DatasetKind.BENCHMARK:
            missing = [field for field in ("prepare_script", "prompt_config") if getattr(self, field) is None]
            if missing:
                raise ValueError("a benchmark dataset requires: " + ", ".join(missing))
        return self


class AdoptedFrom(_ManifestModel):
    source: Source
    ref: GitRef
    reconciled: date


def _profile_schema_conditions() -> list[dict[str, Any]]:
    nonempty_string = {"minLength": 1, "type": "string"}
    nonempty_datasets = {"minItems": 1, "type": "array"}

    def requires(profile: IntegrationProfile, fields: tuple[str, ...]) -> dict[str, Any]:
        return {
            "if": {
                "properties": {"integration_profile": {"const": profile.value}},
                "required": ["integration_profile"],
            },
            "then": {"properties": {field: nonempty_string for field in fields}, "required": list(fields)},
        }

    return [
        {
            "if": {"properties": {"kind": {"const": "benchmark"}}, "required": ["kind"]},
            "then": {
                "properties": {
                    "canonical_split": nonempty_string,
                    "standard_prompt_config": nonempty_string,
                    "datasets": {
                        **nonempty_datasets,
                        "contains": {
                            "properties": {"type": {"const": "benchmark"}},
                            "required": ["type"],
                        },
                    },
                },
                "required": list(_BENCHMARK_REQUIRED_FIELDS),
            },
        },
        *(requires(profile, fields) for profile, fields in _PROFILE_REQUIRED_FIELDS.items() if fields),
        {
            "if": {
                "properties": {"integration_profile": {"const": IntegrationProfile.CUSTOM_DRIVER.value}},
                "required": ["integration_profile"],
            },
            "else": {"properties": {"rollout_driver": {"type": "null"}}},
        },
    ]


class EnvironmentManifest(_ManifestModel):
    """Authored metadata plus a read-only mirror of the resolved Gym composition."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        title="NeMo Gym Environment Manifest",
        json_schema_extra={"allOf": _profile_schema_conditions()},
    )

    name: ManifestName
    version: SemVer = Field(description="Semantic version of the resolved composition.")
    kind: EnvironmentKind
    integration_profile: IntegrationProfile
    domain: Domain
    description: NonEmptyString
    modality: NonEmptyString
    licensing: License = "unknown"
    authors: list[NonEmptyString] = Field(min_length=1, json_schema_extra={"uniqueItems": True})
    reward: Reward
    determinism: Determinism = Determinism.UNKNOWN

    resources_server: NonEmptyString
    agent_server: NonEmptyString
    datasets: list[ManifestDataset] = Field(min_length=1)
    model_server: NonEmptyString | None = None
    rollout_driver: PythonCallable | None = None
    grading_mode: NonEmptyString | None = None

    session_model: SessionModel | None = None
    state: EnvironmentState | None = None
    sandbox: NonEmptyString | None = None
    canonical_split: NonEmptyString | None = None
    standard_prompt_config: NonEmptyString | None = None
    adopted_from: AdoptedFrom | None = None
    lifecycle: Lifecycle = Lifecycle.ACTIVE

    @field_validator("authors")
    @classmethod
    def validate_unique_authors(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("authors must be unique")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "EnvironmentManifest":
        missing = [
            field for field in _PROFILE_REQUIRED_FIELDS[self.integration_profile] if getattr(self, field) is None
        ]
        if self.kind == EnvironmentKind.BENCHMARK:
            missing.extend(field for field in _BENCHMARK_REQUIRED_FIELDS if getattr(self, field) is None)
        if missing:
            raise ValueError("manifest requires: " + ", ".join(dict.fromkeys(missing)))
        if self.integration_profile != IntegrationProfile.CUSTOM_DRIVER and self.rollout_driver is not None:
            raise ValueError("rollout_driver is only valid for the custom-driver profile")
        names = [dataset.name for dataset in self.datasets]
        if len(names) != len(set(names)):
            raise ValueError("dataset names must be unique")
        if self.kind == EnvironmentKind.BENCHMARK and not any(
            dataset.type == DatasetKind.BENCHMARK for dataset in self.datasets
        ):
            raise ValueError("a benchmark manifest requires a benchmark dataset")
        return self


class ManifestError(ConfigError):
    """A manifest could not be read, parsed, or validated."""


def _validation_error(path: Path, error: ValidationError) -> ManifestError:
    issues = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in item["loc"]) or "manifest"
        issues.append(f"  - {location}: {item['msg']}")
    return ManifestError(f"Invalid environment manifest '{path}':\n" + "\n".join(issues))


def load_manifest(path: str | Path) -> EnvironmentManifest:
    manifest_path = Path(path)
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(f"Environment manifest '{manifest_path}' was not found.") from error
    except (OSError, UnicodeError) as error:
        raise ManifestError(f"Could not read environment manifest '{manifest_path}': {error}") from error
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ManifestError(f"Malformed YAML in environment manifest '{manifest_path}'{location}.") from error
    if not isinstance(data, dict):
        raise ManifestError(f"Invalid environment manifest '{manifest_path}': expected a YAML mapping.")
    try:
        return EnvironmentManifest.model_validate(data)
    except ValidationError as error:
        raise _validation_error(manifest_path, error) from error


def dump_manifest(manifest: EnvironmentManifest | Mapping[str, Any]) -> str:
    if not isinstance(manifest, EnvironmentManifest):
        try:
            manifest = EnvironmentManifest.model_validate(manifest)
        except ValidationError as error:
            raise _validation_error(Path("<memory>"), error) from error
    return yaml.safe_dump(
        manifest.model_dump(mode="json", exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
    )


def manifest_json_schema() -> dict[str, Any]:
    return {"$schema": JSON_SCHEMA_DIALECT, **EnvironmentManifest.model_json_schema(mode="validation")}
