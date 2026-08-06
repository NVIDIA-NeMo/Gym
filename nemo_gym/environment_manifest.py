# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contract for NeMo Gym environment manifests.

The manifest is an inspection and discovery artifact.  Runtime composition remains
authoritative in Gym's Hydra configuration; the component and dataset fields here are
a checked, read-only mirror of that resolved configuration.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import tempfile
import warnings
from datetime import date
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Mapping
from urllib.parse import unquote, urlsplit

import yaml
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
from nemo_gym.repository_io import atomic_write_text


JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# SemVer 2.0.0, including optional pre-release and build identifiers. Numeric
# identifiers may not contain leading zeroes.
SEMVER_PATTERN = (
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)

NAME_PATTERN = r"^[a-z0-9][a-z0-9._/-]*$"
CAPABILITY_PATTERN = r"^[a-z][a-z0-9._:/-]*$"
_SPDX_ID_TOKEN = r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?"
SPDX_IDENTIFIER_PATTERN = (
    rf"^(?:{_SPDX_ID_TOKEN}\+?|LicenseRef-{_SPDX_ID_TOKEN}|DocumentRef-{_SPDX_ID_TOKEN}:LicenseRef-{_SPDX_ID_TOKEN})$"
)
ADOPTED_SOURCE_PATTERN = r"^(?:(?:https?|ssh|git|file)://[^\s?#]+|[^@\s]+@[^:\s]+:[^\s]+)$"
GIT_REF_SAFE_PATTERN = r"^[^\x00-\x20\x7f~^:?*\[\\]+$"
PYTHON_CALLABLE_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"

try:
    from packaging.licenses import canonicalize_license_expression as _canonicalize_license_expression
except ImportError:  # Packaging < 24.2 or a minimal downstream installation.
    _canonicalize_license_expression = None

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Description = Annotated[str, StringConstraints(strip_whitespace=True, min_length=10)]
ManifestName = Annotated[str, StringConstraints(strip_whitespace=True, pattern=NAME_PATTERN)]
SemVer = Annotated[str, StringConstraints(strip_whitespace=True, pattern=SEMVER_PATTERN)]
CapabilityName = Annotated[str, StringConstraints(strip_whitespace=True, pattern=CAPABILITY_PATTERN)]


class EnvironmentKind(str, Enum):
    """Whether the task set carries benchmark reproducibility obligations."""

    ENVIRONMENT = "environment"
    BENCHMARK = "benchmark"


class IntegrationProfile(str, Enum):
    """Where the episode-driving loop lives."""

    STOCK_LOOP = "stock-loop"
    MEASURED_LOOP = "measured-loop"
    EXTERNAL_LOOP = "external-loop"
    CUSTOM_DRIVER = "custom-driver"


class Determinism(str, Enum):
    """Whether reseeding a task reproduces its world."""

    SEEDED = "seeded"
    STOCHASTIC = "stochastic"
    UNKNOWN = "unknown"


class SessionModel(str, Enum):
    """Lifetime of one environment session."""

    EPISODE = "episode"
    STEP = "step"


class EnvironmentState(str, Enum):
    """State isolation exposed by the environment."""

    NONE = "none"
    PER_SESSION = "per_session"


class Lifecycle(str, Enum):
    """Authored retirement state. Validation maturity is derived elsewhere."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"


class SpecialLicense(str, Enum):
    """Non-SPDX licensing classifications supported in authored manifests."""

    INTERNAL = "internal"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


class DatasetKind(str, Enum):
    """Dataset roles currently understood by Gym config."""

    TRAIN = "train"
    VALIDATION = "validation"
    EXAMPLE = "example"
    BENCHMARK = "benchmark"


class _ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_default=True)


def _references_grading_mode(node: ast.AST) -> bool:
    return any(
        (isinstance(item, ast.Name) and item.id == "grading_mode")
        or (isinstance(item, ast.Attribute) and item.attr == "grading_mode")
        for item in ast.walk(node)
    )


def _literal_annotation_values(annotation: ast.AST) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(annotation):
        if not isinstance(node, ast.Subscript):
            continue
        annotation_name = node.value.id if isinstance(node.value, ast.Name) else getattr(node.value, "attr", None)
        if annotation_name != "Literal":
            continue
        values.update(
            item.value
            for item in ast.walk(node.slice)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )
    return values


@lru_cache(maxsize=None)
def _grading_mode_source_facts(source_path: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract grading literals and inherited resource-server modules without executing source code."""

    path = Path(source_path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return (), ()
    if "grading_mode" not in source and "resources_servers." not in source:
        return (), ()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return (), ()

    modes: set[str] = set()
    imported_bases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("resources_servers."):
            for imported in node.names:
                imported_bases[imported.asname or imported.name] = node.module

    inherited_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "grading_mode":
                modes.update(_literal_annotation_values(node.annotation))
        elif isinstance(node, ast.Compare):
            operands = (node.left, *node.comparators)
            if any(_references_grading_mode(operand) for operand in operands):
                modes.update(
                    item.value
                    for operand in operands
                    for item in ast.walk(operand)
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )
        elif isinstance(node, ast.Match) and _references_grading_mode(node.subject):
            modes.update(
                item.value
                for case in node.cases
                for item in ast.walk(case.pattern)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in imported_bases:
                    inherited_modules.add(imported_bases[base.id])

    return tuple(sorted(modes)), tuple(sorted(inherited_modules))


def grading_modes_from_source(
    source_path: str | Path,
    *,
    resources_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Infer supported ``grading_mode`` values from an entrypoint's AST.

    Only ``Literal`` annotations, comparisons/match cases, and resource-server base-class imports are
    inspected. The module is never imported, so optional dependencies and startup side effects cannot run.
    An empty tuple means the source was unavailable or no finite set could be proven.
    """

    path = Path(source_path).resolve()
    if resources_root is None:
        resources_dir = next((parent for parent in path.parents if parent.name == "resources_servers"), None)
    else:
        resources_dir = Path(resources_root).resolve()
    visited: set[Path] = set()

    def visit(candidate: Path) -> set[str]:
        resolved = candidate.resolve()
        if resolved in visited:
            return set()
        visited.add(resolved)
        modes, inherited_modules = _grading_mode_source_facts(str(resolved))
        discovered = set(modes)
        if resources_dir is None:
            return discovered
        repository_root = resources_dir.parent
        for module in inherited_modules:
            parts = module.split(".")
            inherited_path = repository_root.joinpath(*parts).with_suffix(".py")
            try:
                inherited_path.resolve().relative_to(resources_dir)
            except ValueError:
                continue
            discovered.update(visit(inherited_path))
        return discovered

    return tuple(sorted(visit(path)))


def _discover_resources_server_grading_modes(resources_root: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not resources_root.is_dir():
        return ()
    constraints: list[tuple[str, tuple[str, ...]]] = []
    for server_dir in sorted(resources_root.iterdir()):
        entrypoint = server_dir / "app.py"
        if not server_dir.is_dir() or not entrypoint.is_file():
            continue
        modes = grading_modes_from_source(entrypoint, resources_root=resources_root)
        if modes:
            constraints.append((server_dir.name, modes))
    return tuple(constraints)


@lru_cache(maxsize=1)
def _default_resources_server_grading_modes() -> tuple[tuple[str, tuple[str, ...]], ...]:
    resources_root = Path(__file__).resolve().parent.parent / "resources_servers"
    return _discover_resources_server_grading_modes(resources_root)


def resources_server_grading_modes(
    resources_root: str | Path | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return finite code-derived grading modes for locally available resources servers."""

    if resources_root is None:
        constraints = _default_resources_server_grading_modes()
    else:
        constraints = _discover_resources_server_grading_modes(Path(resources_root).resolve())
    return dict(constraints)


class Reward(_ManifestModel):
    """Declared reward scale."""

    range: tuple[FiniteFloat, FiniteFloat] = Field(
        description="Inclusive lower and upper reward endpoints.",
    )
    higher_is_better: bool

    @model_validator(mode="after")
    def validate_range(self) -> "Reward":
        lower, upper = self.range
        if lower >= upper:
            raise ValueError("reward range must be ordered as [lower, upper] with lower < upper")
        return self


class ManifestDataset(_ManifestModel):
    """The score-affecting subset of one resolved Gym dataset configuration."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"type": {"const": DatasetKind.BENCHMARK.value}},
                        "required": ["type"],
                    },
                    "then": {
                        "properties": {"prepare_script": {"minLength": 1, "type": "string"}},
                        "required": ["prepare_script"],
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
        if self.type == DatasetKind.BENCHMARK and self.prepare_script is None:
            raise ValueError("a benchmark dataset requires prepare_script")
        return self


def _valid_git_ref(value: str) -> bool:
    if (
        value == "@"
        or value.startswith("-")
        or value.startswith("/")
        or value.endswith(("/", "."))
        or "//" in value
        or ".." in value
        or "@{" in value
        or re.search(r"[\x00-\x20\x7f~^:?*\[\\]", value)
    ):
        return False
    return all(part and not part.startswith(".") and not part.endswith(".lock") for part in value.split("/"))


def parse_python_callable_reference(value: str, *, field_name: str = "callable") -> tuple[str, str]:
    """Parse the module and function names accepted by Gym's runtime loader."""

    module_name, separator, function_name = value.partition(":")
    if (
        not separator
        or ":" in function_name
        or not function_name.isidentifier()
        or not module_name
        or any(not segment.isidentifier() for segment in module_name.split("."))
    ):
        raise ValueError(f"{field_name} must be 'module.path:function' with valid Python identifiers")
    return module_name, function_name


class AdoptedFrom(_ManifestModel):
    """Provenance for an implementation adopted from an upstream project."""

    source: NonEmptyString = Field(
        json_schema_extra={
            "anyOf": [
                {"format": "uri", "pattern": ADOPTED_SOURCE_PATTERN, "type": "string"},
                {"pattern": r"^[^@\s]+@[^:\s]+:.+$", "type": "string"},
            ]
        }
    )
    ref: NonEmptyString = Field(json_schema_extra={"pattern": GIT_REF_SAFE_PATTERN})
    reconciled: date

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if re.search(r"\s", value):
            raise ValueError("source may not contain whitespace")

        # Accept cloneable absolute URLs and Git's common SCP-like SSH syntax.
        parsed = urlsplit(value)
        if parsed.scheme:
            scheme = parsed.scheme.casefold()
            if scheme not in {"file", "git", "http", "https", "ssh"}:
                raise ValueError("source URL scheme must be one of: https, http, ssh, git, file")
            if parsed.query or parsed.fragment:
                raise ValueError("source must not contain a query or fragment; record the revision in ref")
            if scheme == "file":
                if not parsed.path.startswith("/") or not parsed.path.strip("/"):
                    raise ValueError("file source must use an absolute path")
            elif not parsed.hostname or parsed.path in {"", "/"}:
                raise ValueError("source URL must include a host and repository path")
            if parsed.password is not None or (scheme in {"git", "http", "https"} and parsed.username is not None):
                raise ValueError("source URL must not embed a password or access token")
            return value

        scp_match = re.fullmatch(r"[^@\s]+@[^:\s]+:(?P<path>[^\s]+)", value)
        if scp_match is None or not scp_match.group("path").strip("/"):
            raise ValueError("source must be a cloneable URL or SCP-like Git URL (for example git@host:org/repo.git)")
        return value

    @field_validator("ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        if not _valid_git_ref(value):
            raise ValueError("ref must be a valid one-level or path-like Git ref or commit identifier")
        return value


def _profile_json_schema_conditions() -> list[dict[str, Any]]:
    """JSON Schema equivalents of :meth:`EnvironmentManifest.validate_conditions`."""

    nonempty_string = {"minLength": 1, "type": "string"}
    # The base property already supplies the item schema; conditions only need to
    # rule out the nullable branch and require a non-empty array.
    nonempty_datasets = {"minItems": 1, "type": "array"}

    def when(profile: IntegrationProfile, required: list[str]) -> dict[str, Any]:
        field_schemas = {field: nonempty_datasets if field == "datasets" else nonempty_string for field in required}
        return {
            "if": {
                "properties": {"integration_profile": {"const": profile.value}},
                "required": ["integration_profile"],
            },
            "then": {"properties": field_schemas, "required": required},
        }

    return [
        {
            "if": {"properties": {"kind": {"const": EnvironmentKind.BENCHMARK.value}}, "required": ["kind"]},
            "then": {
                "properties": {
                    "canonical_split": nonempty_string,
                    "datasets": {
                        **nonempty_datasets,
                        "contains": {
                            "properties": {"type": {"const": DatasetKind.BENCHMARK.value}},
                            "required": ["type"],
                        },
                    },
                    "standard_prompt_config": nonempty_string,
                },
                "required": ["canonical_split", "standard_prompt_config"],
            },
        },
        when(
            IntegrationProfile.STOCK_LOOP,
            ["resources_server", "agent_server", "model_server", "datasets"],
        ),
        when(
            IntegrationProfile.MEASURED_LOOP,
            ["resources_server", "agent_server", "model_server", "datasets"],
        ),
        when(
            IntegrationProfile.EXTERNAL_LOOP,
            ["resources_server", "agent_server", "datasets"],
        ),
        when(
            IntegrationProfile.CUSTOM_DRIVER,
            ["resources_server", "datasets", "rollout_driver"],
        ),
        {
            "if": {
                "properties": {"integration_profile": {"const": IntegrationProfile.CUSTOM_DRIVER.value}},
                "required": ["integration_profile"],
            },
            "else": {"properties": {"rollout_driver": {"type": "null"}}},
        },
    ]


class EnvironmentManifest(_ManifestModel):
    """Authoritative machine-readable description of one runnable environment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        title="NeMo Gym Environment Manifest",
        json_schema_extra={
            "allOf": [
                *_profile_json_schema_conditions(),
            ]
        },
    )

    name: ManifestName
    version: SemVer
    kind: EnvironmentKind
    integration_profile: IntegrationProfile
    domain: Domain
    description: Description = Field(
        json_schema_extra={"not": {"pattern": r"^[Tt][Oo][Dd][Oo](?:\b|:)"}},
    )
    modality: NonEmptyString
    licensing: NonEmptyString = Field(
        default=SpecialLicense.UNKNOWN.value,
        description="An SPDX license identifier or internal, proprietary, or unknown.",
        json_schema_extra={
            "anyOf": [
                {"enum": [member.value for member in SpecialLicense]},
                {"pattern": SPDX_IDENTIFIER_PATTERN},
            ]
        },
    )
    authors: list[NonEmptyString] = Field(min_length=1, json_schema_extra={"uniqueItems": True})
    reward: Reward
    determinism: Determinism = Determinism.UNKNOWN

    # Read-only mirrors of the resolved Hydra composition.
    resources_server: NonEmptyString | None = None
    agent_server: NonEmptyString | None = None
    model_server: NonEmptyString | None = None
    datasets: list[ManifestDataset] | None = Field(default=None, min_length=1)
    rollout_driver: NonEmptyString | None = Field(
        default=None,
        json_schema_extra={"pattern": PYTHON_CALLABLE_PATTERN},
    )
    grading_mode: NonEmptyString | None = None

    # Optional environment-runtime declarations.
    session_model: SessionModel | None = None
    state: EnvironmentState | None = None
    sandbox: NonEmptyString | None = None

    # Required for benchmarks, but intentionally permitted for environments.
    canonical_split: NonEmptyString | None = None
    standard_prompt_config: NonEmptyString | None = None

    adopted_from: AdoptedFrom | None = None
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    requires: list[CapabilityName] = Field(default_factory=list, json_schema_extra={"uniqueItems": True})
    provides: list[CapabilityName] = Field(default_factory=list, json_schema_extra={"uniqueItems": True})

    @field_validator("name")
    @classmethod
    def validate_name_segments(cls, value: str) -> str:
        if "//" in value or any(segment in {"", ".", ".."} for segment in value.split("/")):
            raise ValueError("name must contain non-empty path-safe segments and may not contain '.' or '..'")
        return value

    @field_validator("licensing")
    @classmethod
    def validate_licensing(cls, value: str) -> str:
        if value in {member.value for member in SpecialLicense}:
            return value
        if not re.fullmatch(SPDX_IDENTIFIER_PATTERN, value):
            raise ValueError("licensing must be one SPDX license identifier or one of: internal, proprietary, unknown")
        if value.startswith(("LicenseRef-", "DocumentRef-")) or _canonicalize_license_expression is None:
            return value
        try:
            canonical = str(_canonicalize_license_expression(value))
        except ValueError as error:
            raise ValueError(f"licensing contains an unknown SPDX license identifier: {value!r}") from error
        if not re.fullmatch(SPDX_IDENTIFIER_PATTERN, canonical):
            raise ValueError("licensing must contain one SPDX license identifier, not an expression")
        return canonical

    @field_validator("rollout_driver")
    @classmethod
    def validate_rollout_driver(cls, value: str | None) -> str | None:
        if value is not None:
            parse_python_callable_reference(value, field_name="rollout_driver")
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if re.match(r"^todo(?:\b|:)", value, flags=re.IGNORECASE):
            raise ValueError("description must replace the TODO placeholder with a meaningful summary")
        return value

    @field_validator("authors", "requires", "provides")
    @classmethod
    def validate_unique_lists(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("entries must be unique")
        return value

    @model_validator(mode="after")
    def validate_conditions(self) -> "EnvironmentManifest":
        missing: list[str] = []
        if self.kind == EnvironmentKind.BENCHMARK:
            if self.canonical_split is None:
                missing.append("canonical_split")
            if self.standard_prompt_config is None:
                missing.append("standard_prompt_config")

        profile_requirements = {
            IntegrationProfile.STOCK_LOOP: ("resources_server", "agent_server", "model_server", "datasets"),
            IntegrationProfile.MEASURED_LOOP: (
                "resources_server",
                "agent_server",
                "model_server",
                "datasets",
            ),
            IntegrationProfile.EXTERNAL_LOOP: ("resources_server", "agent_server", "datasets"),
            IntegrationProfile.CUSTOM_DRIVER: ("resources_server", "datasets", "rollout_driver"),
        }
        missing.extend(
            field for field in profile_requirements[self.integration_profile] if getattr(self, field) is None
        )
        if missing:
            fields = ", ".join(dict.fromkeys(missing))
            raise ValueError(
                f"manifest kind={self.kind.value!r} profile={self.integration_profile.value!r} requires: {fields}"
            )

        if self.integration_profile != IntegrationProfile.CUSTOM_DRIVER and self.rollout_driver is not None:
            raise ValueError("rollout_driver is only valid when integration_profile is 'custom-driver'")

        if self.datasets is not None:
            names = [dataset.name for dataset in self.datasets]
            if len(names) != len(set(names)):
                raise ValueError("dataset names must be unique within a manifest")
            if self.kind == EnvironmentKind.BENCHMARK:
                benchmark_datasets = [dataset for dataset in self.datasets if dataset.type == DatasetKind.BENCHMARK]
                if not benchmark_datasets:
                    raise ValueError("kind='benchmark' requires at least one dataset with type='benchmark'")
                mismatched_prompt_configs = [
                    dataset.name
                    for dataset in benchmark_datasets
                    if dataset.prompt_config is not None and dataset.prompt_config != self.standard_prompt_config
                ]
                if mismatched_prompt_configs:
                    names = ", ".join(repr(name) for name in mismatched_prompt_configs)
                    raise ValueError(
                        "benchmark dataset prompt_config must match standard_prompt_config; "
                        f"mismatched datasets: {names}"
                    )

        return self


def manifest_implied_capabilities(
    manifest: EnvironmentManifest | Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Return ``(manifest field, capability)`` pairs implied by runtime facts.

    Keeping the source field alongside each capability lets static validation
    enforce the composition topology: when Gym selects a model, ``modality``
    belongs to that model rather than to any other model in the merged config.
    """

    def field(name: str, default: object = None) -> object:
        if isinstance(manifest, Mapping):
            return manifest.get(name, default)
        return getattr(manifest, name, default)

    def value(name: str) -> str | None:
        raw = field(name)
        if raw is None:
            return None
        return str(getattr(raw, "value", raw))

    def slug(raw: str) -> str:
        normalized = re.sub(r"[^a-z0-9._:/-]+", "-", raw.strip().casefold()).strip("-._:/")
        return normalized if normalized and normalized[0].isalpha() else f"capability-{normalized or 'unknown'}"

    capabilities: list[tuple[str, str]] = []
    modality = value("modality")
    if modality and value("model_server"):
        capabilities.append(("modality", f"{slug(modality)}-model"))
    session_model = value("session_model")
    if session_model:
        capabilities.append(("session_model", f"session:{slug(session_model)}"))
    state = value("state")
    if state and state != EnvironmentState.NONE.value:
        capabilities.append(("state", f"state:{slug(state)}"))
    sandbox = value("sandbox")
    if sandbox:
        capabilities.append(("sandbox", sandbox if sandbox.startswith("sandbox:") else f"sandbox:{slug(sandbox)}"))
    return tuple(capabilities)


def manifest_required_capabilities(manifest: EnvironmentManifest | Mapping[str, Any]) -> tuple[str, ...]:
    """Combine authored requirements with capabilities implied by runtime facts.

    The spelling is deliberately mechanical so components can declare the same
    contract without a compatibility table keyed by implementation name.
    """

    if isinstance(manifest, Mapping):
        raw_requires = manifest.get("requires", ()) or ()
    else:
        raw_requires = manifest.requires
    values = raw_requires if isinstance(raw_requires, (list, tuple, set)) else (raw_requires,)
    capabilities = [capability for capability in values if isinstance(capability, str) and capability]
    capabilities.extend(capability for _field_name, capability in manifest_implied_capabilities(manifest))
    return tuple(dict.fromkeys(capabilities))


class ManifestError(ConfigError):
    """A manifest could not be read, parsed, or validated."""


def validate_adopted_from_reference(adopted_from: AdoptedFrom) -> str | None:
    """Require an adopted upstream and revision to resolve to Git objects."""

    parsed = urlsplit(adopted_from.source)
    local = parsed.scheme.casefold() == "file"
    if local:
        if parsed.netloc not in {"", "localhost"}:
            raise ManifestError(
                f"adopted_from file source '{adopted_from.source}' names a remote host; use a local file URL."
            )
        repository = Path(unquote(parsed.path))
        if not repository.is_dir():
            raise ManifestError(f"adopted_from source '{adopted_from.source}' does not resolve to a local directory.")
        command = ["git", "-C", str(repository), "rev-parse", "--verify", "--quiet", f"{adopted_from.ref}^{{commit}}"]
    elif re.fullmatch(r"[0-9a-fA-F]{40}", adopted_from.ref):
        command = ["git", "ls-remote", "--exit-code", adopted_from.source]
    else:
        command = ["git", "ls-remote", "--exit-code", adopted_from.source, adopted_from.ref]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except FileNotFoundError as error:
        raise ManifestError("git is required to validate adopted_from provenance.") from error
    except subprocess.TimeoutExpired as error:
        raise ManifestError(f"Timed out while validating adopted_from source '{adopted_from.source}'.") from error
    if not local and re.fullmatch(r"[0-9a-fA-F]{40}", adopted_from.ref):
        # A raw object ID is not a ref-pattern, so ``ls-remote URL <sha>``
        # normally returns no rows even when the commit is reachable. Query the
        # advertised refs first, then ask the server for the exact commit when
        # it is no longer a tip.
        if completed.returncode != 0:
            raise ManifestError(
                f"Could not query adopted_from source '{adopted_from.source}' (git exit {completed.returncode}); "
                "check repository access and authentication."
            )
        if any(
            line.partition("\t")[0].casefold() == adopted_from.ref.casefold() for line in completed.stdout.splitlines()
        ):
            return None
        try:
            with tempfile.TemporaryDirectory(prefix="nemo-gym-provenance-") as temporary_directory:
                initialize = subprocess.run(
                    ["git", "init", "--bare", "--quiet", temporary_directory],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=5,
                )
                fetch = (
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            temporary_directory,
                            "fetch",
                            "--quiet",
                            "--depth=1",
                            "--no-tags",
                            adopted_from.source,
                            adopted_from.ref,
                        ],
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=5,
                    )
                    if initialize.returncode == 0
                    else initialize
                )
                completed = (
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            temporary_directory,
                            "rev-parse",
                            "--verify",
                            "--quiet",
                            "FETCH_HEAD^{commit}",
                        ],
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=5,
                    )
                    if fetch.returncode == 0
                    else fetch
                )
        except subprocess.TimeoutExpired as error:
            raise ManifestError(f"Timed out while validating adopted_from source '{adopted_from.source}'.") from error
        if completed.returncode != 0:
            raise ManifestError(
                f"adopted_from commit '{adopted_from.ref}' is not advertised and could not be fetched from "
                f"'{adopted_from.source}'."
            )
        return None
    if completed.returncode != 0:
        if not local and completed.returncode != 2:
            raise ManifestError(
                f"Could not query adopted_from source '{adopted_from.source}' (git exit {completed.returncode}); "
                "check repository access and authentication."
            )
        raise ManifestError(
            f"adopted_from ref '{adopted_from.ref}' does not resolve to a commit in '{adopted_from.source}'."
        )
    return None


def _format_validation_error(path: Path, error: ValidationError) -> ManifestError:
    issues: list[str] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in item["loc"]) or "manifest"
        issues.append(f"  - {location}: {item['msg']}")
    return ManifestError(f"Invalid environment manifest '{path}':\n" + "\n".join(issues))


def load_manifest(path: str | Path) -> EnvironmentManifest:
    """Load and validate one YAML manifest with actionable, path-qualified errors."""

    manifest_path = Path(path)
    try:
        source = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ManifestError(f"Environment manifest '{manifest_path}' was not found.") from error
    except OSError as error:
        raise ManifestError(f"Could not read environment manifest '{manifest_path}': {error}") from error

    try:
        data = yaml.safe_load(source)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
        problem = getattr(error, "problem", None) or str(error).splitlines()[0]
        raise ManifestError(
            f"Malformed YAML in environment manifest '{manifest_path}'{location}: {problem}"
        ) from error

    if not isinstance(data, dict):
        actual = "null" if data is None else type(data).__name__
        raise ManifestError(
            f"Invalid environment manifest '{manifest_path}': expected a YAML mapping at the document root, got {actual}."
        )

    try:
        return EnvironmentManifest.model_validate(data)
    except ValidationError as error:
        raise _format_validation_error(manifest_path, error) from error


def dump_manifest(manifest: EnvironmentManifest | Mapping[str, Any], path: str | Path | None = None) -> str:
    """Serialize a manifest as stable YAML and optionally write it to ``path``."""

    if not isinstance(manifest, EnvironmentManifest):
        try:
            manifest = EnvironmentManifest.model_validate(manifest)
        except ValidationError as error:
            raise _format_validation_error(Path(path or "<memory>"), error) from error

    rendered = yaml.safe_dump(
        manifest.model_dump(mode="json", exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
    )
    if path is not None:
        manifest_path = Path(path)
        try:
            atomic_write_text(manifest_path, rendered)
        except OSError as error:
            raise ManifestError(f"Could not write environment manifest '{manifest_path}': {error}") from error
    return rendered


def manifest_json_schema() -> dict[str, Any]:
    """Return the JSON Schema used by editors, agents, CI, and catalog tooling."""

    schema = EnvironmentManifest.model_json_schema(mode="validation")
    grading_conditions = [
        {
            "if": {
                "properties": {"resources_server": {"const": server_name}},
                "required": ["resources_server"],
            },
            "then": {
                "properties": {
                    "grading_mode": {"enum": [*modes, None]},
                }
            },
        }
        for server_name, modes in sorted(resources_server_grading_modes().items())
    ]
    schema["allOf"] = [*schema.get("allOf", []), *grading_conditions]
    return {"$schema": JSON_SCHEMA_DIALECT, **schema}


def write_manifest_json_schema(path: str | Path) -> None:
    """Write :func:`manifest_json_schema` deterministically."""

    schema_path = Path(path)
    rendered = json.dumps(manifest_json_schema(), indent=2, sort_keys=True) + "\n"
    try:
        atomic_write_text(schema_path, rendered, create_parent=True)
    except OSError as error:
        raise ManifestError(f"Could not write environment manifest JSON Schema '{schema_path}': {error}") from error


__all__ = [
    "AdoptedFrom",
    "DatasetKind",
    "Determinism",
    "EnvironmentKind",
    "EnvironmentManifest",
    "EnvironmentState",
    "IntegrationProfile",
    "Lifecycle",
    "ManifestDataset",
    "ManifestError",
    "Reward",
    "SessionModel",
    "SpecialLicense",
    "dump_manifest",
    "grading_modes_from_source",
    "load_manifest",
    "manifest_implied_capabilities",
    "manifest_json_schema",
    "manifest_required_capabilities",
    "parse_python_callable_reference",
    "resources_server_grading_modes",
    "validate_adopted_from_reference",
    "write_manifest_json_schema",
]
