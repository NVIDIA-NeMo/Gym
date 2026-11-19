# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed contract for an environment ``manifest.yaml``."""

from __future__ import annotations

import posixpath
from datetime import date
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, NamedTuple

import yaml
from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from nemo_gym.config.config_types import ConfigError, Domain
from nemo_gym.utils.text import did_you_mean


JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
MANIFEST_SCHEMA_ID = "environment-manifest.schema.json"
_PRIVATE_LICENSE_VALUES = frozenset({"internal", "proprietary", "unknown"})
SOURCE_PATTERN = r"^(?:(?:https?|ssh|git|file)://\S+|[^@\s]+@[^:\s]+:\S+)$"
CALLABLE_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
GITHUB_LOGIN_PATTERN = r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$"
# A yes-or-no answer that an author can also give as "unknown". An absent field means nobody answered;
# "unknown" means somebody checked and nobody knows.
BoolOrUnknown = bool | Literal["unknown"]

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
License = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Source = Annotated[str, StringConstraints(strip_whitespace=True, pattern=SOURCE_PATTERN)]
GitRef = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[^\s]+$")]
PythonCallable = Annotated[str, StringConstraints(strip_whitespace=True, pattern=CALLABLE_PATTERN)]
GitHubLogin = Annotated[str, StringConstraints(strip_whitespace=True, pattern=GITHUB_LOGIN_PATTERN)]

CAPABILITIES_AND_DOMAINS_PATH = Path(__file__).with_name("capabilities_and_domains.yaml")


class _CapabilitiesAndDomains(NamedTuple):
    capability_taxonomy_version: str
    domain_list_version: str
    # Each capability with its specializations, in the file's order.
    capabilities: Mapping[str, tuple[str, ...]]
    domains: tuple[str, ...]

    def capability_names(self) -> list[str]:
        """Every value `capabilities` accepts: each capability, followed by `<capability>/<specialization>`."""
        return [
            name
            for capability, specializations in self.capabilities.items()
            for name in (capability, *(f"{capability}/{specialization}" for specialization in specializations))
        ]


@cache
def _capabilities_and_domains() -> _CapabilitiesAndDomains:
    """The capability taxonomy and domain list, read once from :data:`CAPABILITIES_AND_DOMAINS_PATH`."""
    data = yaml.safe_load(CAPABILITIES_AND_DOMAINS_PATH.read_text(encoding="utf-8"))
    return _CapabilitiesAndDomains(
        capability_taxonomy_version=data["capability_taxonomy_version"],
        domain_list_version=data["domain_list_version"],
        capabilities={name: tuple(body["specializations"]) for name, body in data["capabilities"].items()},
        domains=tuple(data["domains"]),
    )


def _check_capability(value: str) -> str:
    lists = _capabilities_and_domains()
    capability, slash, specialization = value.partition("/")
    specializations = lists.capabilities.get(capability)
    if specializations is not None and (not slash or specialization in specializations):
        return value
    if specializations is not None:
        raise ValueError(
            f"'{specialization}' is not a specialization of '{capability}' in the capability taxonomy "
            f"({lists.capability_taxonomy_version}). Its specializations are: {', '.join(specializations)}."
        )
    raise ValueError(
        f"'{value}' is not in the capability taxonomy ({lists.capability_taxonomy_version}); write "
        f"<capability> or <capability>/<specialization>.{did_you_mean(value, lists.capability_names())} "
        "The values are listed in nemo_gym/environment/capabilities_and_domains.yaml."
    )


def _check_domain(value: str) -> str:
    lists = _capabilities_and_domains()
    if value in lists.domains:
        return value
    raise ValueError(
        f"'{value}' is not in the domain list ({lists.domain_list_version}).{did_you_mean(value, lists.domains)} "
        "The values are listed in nemo_gym/environment/capabilities_and_domains.yaml."
    )


def _list_capabilities(schema: dict[str, Any]) -> None:
    schema["enum"] = _capabilities_and_domains().capability_names()


def _list_domains(schema: dict[str, Any]) -> None:
    schema["enum"] = list(_capabilities_and_domains().domains)


# Items are checked by EnvironmentManifest.validate_listed_values once the field is known to be a list, so a bad
# item gives one error rather than one per branch of `list | "unknown"`. The schema lists the values for editors.
# The lists are read when a manifest is first validated or the schema is built, not when this module is imported.
ListedCapability = Annotated[str, Field(json_schema_extra=_list_capabilities)]
ListedDomain = Annotated[str, Field(json_schema_extra=_list_domains)]


class EnvironmentKind(StrEnum):
    ENVIRONMENT = "environment"
    BENCHMARK = "benchmark"


class IntegrationProfile(StrEnum):
    CUSTOM_GYM_VERIFIER = "custom-gym-verifier"
    CUSTOM_GYM_AGENT_LOOP = "custom-gym-agent-loop"
    EXTERNAL_AGENT_LOOP = "external-agent-loop"
    EXTERNAL_ROLLOUT_DRIVER = "external-rollout-driver"


class Determinism(StrEnum):
    SEEDED = "seeded"
    STOCHASTIC = "stochastic"
    UNKNOWN = "unknown"


class SessionModel(StrEnum):
    EPISODE = "episode"
    STEP = "step"


class EnvironmentState(StrEnum):
    NONE = "none"
    PER_SESSION = "per_session"


class Lifecycle(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class DatasetKind(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    EXAMPLE = "example"
    BENCHMARK = "benchmark"


class PromptSource(StrEnum):
    TEMPLATE = "template"
    PREPARED = "prepared"
    AGENT = "agent"


class IntendedUse(StrEnum):
    EVALUATION = "evaluation"
    TRAINING = "training"


_PROFILE_REQUIRED_FIELDS = {
    IntegrationProfile.CUSTOM_GYM_VERIFIER: ("model_server", "resources_server"),
    IntegrationProfile.CUSTOM_GYM_AGENT_LOOP: ("model_server",),
    IntegrationProfile.EXTERNAL_AGENT_LOOP: (),
    IntegrationProfile.EXTERNAL_ROLLOUT_DRIVER: ("rollout_driver",),
}
_BENCHMARK_REQUIRED_FIELDS = ("canonical_split",)


class _ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_default=True)


class Reward(_ManifestModel):
    range: tuple[FiniteFloat, FiniteFloat] = Field(description="Inclusive lower and upper reward endpoints.")
    higher_is_better: bool

    @model_validator(mode="after")
    def validate_range(self) -> "Reward":
        if self.range[0] > self.range[1]:
            raise ValueError("reward.range must be ordered with lower <= upper")
        return self


class ManifestDataset(_ManifestModel):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"type": {"const": "benchmark"}}, "required": ["type"]},
                    "then": {
                        "properties": {
                            "prepare_script": {"minLength": 1, "type": "string"},
                        },
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
        if self.type == DatasetKind.BENCHMARK:
            if self.prepare_script is None:
                raise ValueError("a benchmark dataset requires: prepare_script")
        return self


class AdoptedFrom(_ManifestModel):
    source: Source
    ref: GitRef
    reconciled: date


class ManifestRequirements(_ManifestModel):
    """What a run needs beyond the model under test. Each field also takes "unknown"."""

    gpu: BoolOrUnknown | None = Field(
        default=None,
        description="Whether the environment itself needs a GPU, for example to score answers.",
    )
    # The same words as a Harbor task's `network_mode`, so a cluster with restricted egress can tell an environment
    # that calls a few named services, which it can allowlist, from one that needs the open internet.
    network: Literal["no-network", "allowlist", "public", "unknown"] | None = Field(
        default=None,
        description=(
            "Outbound network access a run needs: no-network; allowlist (only specific services, such as a search "
            "API or a hosted judge model); or public (the open internet, for example browsing or downloads)."
        ),
    )


class _ListedConfig(_ManifestModel):
    """A config that a main manifest lists. It shares the main manifest's facts and sets only those that differ.

    A field left out takes the main manifest's value; ``null`` sets it to none, as for a harness with no model server.
    ``requirements`` is combined field by field, so setting only ``network`` keeps the inherited ``gpu``; a list such as
    ``datasets`` or ``maintainers`` replaces the inherited one whole. The composition fields mirror this config as the
    main manifest's mirror the main config, and ``gym env validate --sync`` writes them here.
    """

    name: NonEmptyString = Field(
        description=(
            "Catalog name of the config: its path under benchmarks/ or environments/ without .yaml, as "
            "`gym eval run --benchmark` takes it."
        )
    )
    config_path: NonEmptyString = Field(description="Config path relative to the main manifest.")
    description: NonEmptyString | None = Field(default=None, description="What this config changes.")
    maintainers: list[GitHubLogin] | None = Field(
        default=None,
        min_length=1,
        json_schema_extra={"uniqueItems": True},
        description="GitHub logins of the people who maintain this config, when they are not the main manifest's.",
    )
    requirements: ManifestRequirements | None = Field(
        default=None,
        description=(
            "The requirements that differ, field by field: a field it sets replaces the inherited one, and a field it "
            "leaves out keeps the inherited value."
        ),
    )
    primary_metric: NonEmptyString | None = Field(
        default=None,
        description="The headline metric, when this config reports another one, for example after more repeats.",
    )
    integration_profile: IntegrationProfile | None = None
    dataset_owner: NonEmptyString | None = None
    resources_server: NonEmptyString | None = None
    agent_server: NonEmptyString | None = None
    model_server: NonEmptyString | None = None
    datasets: list[ManifestDataset] | None = None
    rollout_driver: PythonCallable | None = None
    grading_mode: NonEmptyString | None = None
    harbor_target: NonEmptyString | None = None


class ManifestSetup(_ListedConfig):
    """A config that runs the main manifest's tasks and scoring with another harness or runtime."""


class ManifestProtocol(_ListedConfig):
    """A config that scores a benchmark's tasks under another prompt, tools or scoring.

    It may list its own setups: configs that run the protocol with another harness or runtime. They take the
    protocol's facts, not only the main manifest's. A setup lists nothing, so the nesting stops there.
    """

    description: NonEmptyString = Field(description="How the protocol's prompt, tools or scoring differ.")
    standard_prompt_config: NonEmptyString | None = None
    setups: list[ManifestSetup] = Field(
        default_factory=list,
        description="Configs that run this protocol's tasks, prompt and scoring with another harness or runtime.",
    )


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

    # A Harbor-backed manifest's rows come from its target, so it declares no datasets of its own.
    no_harbor_target = {"harbor_target": {"type": "null"}}
    return [
        {
            "if": {"properties": no_harbor_target},
            "then": {"properties": {"datasets": nonempty_datasets}, "required": ["datasets"]},
            "else": {
                "properties": {
                    "datasets": {"maxItems": 0, "type": "array"},
                    "prompt_source": {"enum": [PromptSource.PREPARED.value, PromptSource.AGENT.value]},
                },
                "required": ["prompt_source"],
            },
        },
        {
            "if": {"properties": {"kind": {"const": "benchmark"}}, "required": ["kind"]},
            "then": {"properties": {"canonical_split": nonempty_string}, "required": list(_BENCHMARK_REQUIRED_FIELDS)},
        },
        {
            "if": {"properties": {"kind": {"const": "benchmark"}, **no_harbor_target}, "required": ["kind"]},
            "then": {
                "properties": {
                    "datasets": {
                        **nonempty_datasets,
                        "contains": {
                            "properties": {"type": {"const": "benchmark"}},
                            "required": ["type"],
                        },
                    },
                },
            },
        },
        {
            "if": {
                "properties": {"kind": {"const": "benchmark"}, "prompt_source": {"const": "template"}},
                "required": ["kind"],
            },
            "then": {
                "properties": {"standard_prompt_config": nonempty_string},
                "required": ["standard_prompt_config"],
            },
        },
        {
            "if": {"properties": {"prompt_source": {"const": "prepared"}}, "required": ["prompt_source"]},
            "then": {"properties": {"standard_prompt_config": {"type": "null"}}},
        },
        {
            "if": {
                "properties": {"prompt_source": {"enum": ["prepared", "agent"]}},
                "required": ["prompt_source"],
            },
            "then": {
                "properties": {
                    "datasets": {"items": {"properties": {"prompt_config": {"type": "null"}}}},
                },
            },
        },
        *(requires(profile, fields) for profile, fields in _PROFILE_REQUIRED_FIELDS.items() if fields),
        {
            "if": {
                "properties": {"integration_profile": {"const": IntegrationProfile.EXTERNAL_ROLLOUT_DRIVER.value}},
                "required": ["integration_profile"],
            },
            "else": {"properties": {"rollout_driver": {"type": "null"}}},
        },
        {
            "if": {"properties": {"kind": {"const": EnvironmentKind.ENVIRONMENT.value}}, "required": ["kind"]},
            "then": {"properties": {"protocols": {"maxItems": 0}}},
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

    name: NonEmptyString
    version: NonEmptyString = Field(
        description=(
            "Version of the resolved composition, shared by the setups and protocols listed here; Semantic "
            "Versioning is recommended."
        )
    )
    experimental: bool = Field(
        default=True,
        description="Manual experimental flag; false does not imply certificate-backed validation.",
    )
    kind: EnvironmentKind
    integration_profile: IntegrationProfile
    domain: Domain
    description: NonEmptyString
    modality: NonEmptyString
    licensing: License = Field(
        default="unknown",
        description="SPDX license expression, internal, proprietary, or unknown.",
    )
    authors: list[NonEmptyString] = Field(min_length=1, json_schema_extra={"uniqueItems": True})
    maintainers: list[GitHubLogin] | None = Field(
        default=None,
        min_length=1,
        json_schema_extra={"uniqueItems": True},
        description="GitHub logins of the people who maintain the environment now.",
    )
    intended_use: list[IntendedUse] | None = Field(
        default=None,
        min_length=1,
        json_schema_extra={"uniqueItems": True},
        description="Whether the environment is meant for evaluation, training or both; unknown when absent.",
    )
    capabilities: list[ListedCapability] | Literal["unknown"] | None = Field(
        default=None,
        json_schema_extra={"uniqueItems": True},
        description=(
            "Skills the environment exercises, from the capability taxonomy, each written <capability> or "
            "<capability>/<specialization>, for example coding/code_generation. The first entry is the main "
            "capability, and its top level (the part before the slash) is the environment's coarse category, used "
            "for listings and charts. An empty list when none applies, or unknown."
        ),
    )
    domains: list[ListedDomain] | Literal["unknown"] | None = Field(
        default=None,
        json_schema_extra={"uniqueItems": True},
        description=(
            "Fields of expertise the tasks need, from the domain list, with the main domain first; an empty "
            "list when none applies, or unknown."
        ),
    )
    overlaps_with: list[NonEmptyString] | None = Field(
        default=None,
        min_length=1,
        json_schema_extra={"uniqueItems": True},
        description=(
            "Benchmarks that training on this environment would invalidate, for example because their tasks overlap "
            "with its data."
        ),
    )
    primary_metric: NonEmptyString | None = Field(
        default=None,
        description="The key metric that is the environment's headline score.",
    )
    requirements: ManifestRequirements | None = None
    reward: Reward
    determinism: Determinism = Determinism.UNKNOWN

    config_path: NonEmptyString = Field(default="config.yaml", description="Config path relative to the manifest.")
    harbor_target: NonEmptyString | None = Field(
        default=None,
        description="Harbor task folder, entry or reference the config's harbor_target names; its tasks are the rows.",
    )
    dataset_owner: NonEmptyString | None = None
    resources_server: NonEmptyString | None
    agent_server: NonEmptyString
    datasets: list[ManifestDataset] = Field(
        default_factory=list, description="Required unless harbor_target is set, whose tasks are the rows."
    )
    model_server: NonEmptyString | None = None
    rollout_driver: PythonCallable | None = None
    grading_mode: NonEmptyString | None = None

    session_model: SessionModel | None = None
    state: EnvironmentState | None = None
    sandbox: NonEmptyString | None = None
    canonical_split: NonEmptyString | None = None
    prompt_source: PromptSource = PromptSource.TEMPLATE
    standard_prompt_config: NonEmptyString | None = Field(
        default=None,
        description="Required for template benchmarks; informational agent template; absent for prepared prompts.",
    )
    adopted_from: AdoptedFrom | None = None
    lifecycle: Lifecycle = Lifecycle.ACTIVE

    setups: list[ManifestSetup] = Field(
        default_factory=list,
        description="Configs that run these tasks and this scoring with another harness or runtime.",
    )
    protocols: list[ManifestProtocol] = Field(
        default_factory=list,
        description="Benchmark configs that score these tasks under another prompt, tools or scoring.",
    )

    @field_validator("authors", "maintainers", "intended_use", "capabilities", "domains", "overlaps_with")
    @classmethod
    def validate_unique_items(cls, value: list[str] | str | None, info: ValidationInfo) -> list[str] | str | None:
        if isinstance(value, list) and len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must be unique")
        return value

    @field_validator("capabilities", "domains")
    @classmethod
    def validate_listed_values(cls, value: list[str] | str | None, info: ValidationInfo) -> list[str] | str | None:
        if not isinstance(value, list):
            return value
        check = _check_capability if info.field_name == "capabilities" else _check_domain
        for item in value:
            if item == "unknown":
                raise ValueError(
                    f"unknown is the whole value, as in `{info.field_name}: unknown`, not an item in a list; "
                    "an empty list means none applies"
                )
            check(item)
        return value

    @field_validator("licensing")
    @classmethod
    def validate_licensing(cls, value: str) -> str:
        if value in _PRIVATE_LICENSE_VALUES:
            return value
        try:
            return canonicalize_license_expression(value)
        except InvalidLicenseExpression as error:
            raise ValueError(
                "licensing must be an SPDX license expression, internal, proprietary, or unknown"
            ) from error

    @model_validator(mode="after")
    def validate_contract(self) -> "EnvironmentManifest":
        missing = [
            field for field in _PROFILE_REQUIRED_FIELDS[self.integration_profile] if getattr(self, field) is None
        ]
        if self.harbor_target is None and not self.datasets:
            missing.append("datasets")
        if self.kind == EnvironmentKind.BENCHMARK:
            missing.extend(field for field in _BENCHMARK_REQUIRED_FIELDS if getattr(self, field) is None)
            if self.prompt_source == PromptSource.TEMPLATE and self.standard_prompt_config is None:
                missing.append("standard_prompt_config")
        if missing:
            raise ValueError("manifest requires: " + ", ".join(dict.fromkeys(missing)))
        if self.harbor_target is not None:
            if self.datasets:
                raise ValueError("a manifest with harbor_target declares no datasets: its rows are the target's tasks")
            if self.prompt_source == PromptSource.TEMPLATE:
                raise ValueError(
                    "a manifest with harbor_target needs prompt_source prepared or agent: "
                    "its rows carry each task's instruction"
                )
        if self.prompt_source == PromptSource.PREPARED and self.standard_prompt_config is not None:
            raise ValueError("standard_prompt_config is forbidden for prepared prompts")
        if self.prompt_source != PromptSource.TEMPLATE and any(
            dataset.prompt_config is not None for dataset in self.datasets
        ):
            raise ValueError("dataset.prompt_config is only valid for template prompts")
        if self.integration_profile != IntegrationProfile.EXTERNAL_ROLLOUT_DRIVER and self.rollout_driver is not None:
            raise ValueError("rollout_driver is only valid for the external-rollout-driver profile")
        names = [dataset.name for dataset in self.datasets]
        if len(names) != len(set(names)):
            raise ValueError("dataset names must be unique")
        if (
            self.kind == EnvironmentKind.BENCHMARK
            and self.harbor_target is None
            and not any(dataset.type == DatasetKind.BENCHMARK for dataset in self.datasets)
        ):
            raise ValueError("a benchmark manifest requires a benchmark dataset")
        return self

    @model_validator(mode="after")
    def validate_listed_configs(self) -> "EnvironmentManifest":
        if self.protocols and self.kind != EnvironmentKind.BENCHMARK:
            raise ValueError("protocols are only valid for benchmarks")
        listed_configs = self.listed_configs()
        names = [self.name, *(listed.name for listed in listed_configs)]
        if len(names) != len(set(names)):
            raise ValueError("the main manifest and each setup and protocol need their own name")
        config_paths = [posixpath.normpath(listed.config_path) for listed in (self, *listed_configs)]
        if len(config_paths) != len(set(config_paths)):
            raise ValueError("the main manifest and each setup and protocol need their own config_path")
        for listed in listed_configs:
            try:
                resolve_listed_manifest(self, listed)
            except ValidationError as error:
                issue = error.errors(include_url=False, include_context=False, include_input=False)[0]
                location = ".".join(str(part) for part in issue["loc"])
                inherited = "the main manifest" if self.listing_protocol(listed) is None else "its protocol"
                raise ValueError(
                    f"{listed.name}, combined with {inherited}: {f'{location}: ' if location else ''}{issue['msg']}"
                ) from error
        return self

    def listed_configs(self) -> tuple[ManifestSetup | ManifestProtocol, ...]:
        """The setups, then each protocol followed by its own setups, in the order the manifest lists them."""
        return (*self.setups, *(listed for protocol in self.protocols for listed in (protocol, *protocol.setups)))

    def listing_protocol(self, listed: ManifestSetup | ManifestProtocol) -> ManifestProtocol | None:
        """The protocol that lists ``listed`` as one of its setups; ``None`` when the main manifest lists it."""
        return next((protocol for protocol in self.protocols if listed in protocol.setups), None)


class ManifestError(ConfigError):
    """A manifest could not be read, parsed, or validated."""


def resolve_listed_manifest(
    main: EnvironmentManifest, listed: ManifestSetup | ManifestProtocol
) -> EnvironmentManifest:
    """The complete manifest of a setup or protocol: the facts it inherits with the ones it sets.

    A setup a protocol lists inherits the protocol's complete manifest; any other listed config inherits the main
    manifest's. Its ``config_path`` stays relative to the main manifest, and it lists no setups or protocols of its
    own. The catalog, validation and publication all get a setup or protocol from here, so none of them combines them.
    """
    protocol = main.listing_protocol(listed)
    inherited = main if protocol is None else resolve_listed_manifest(main, protocol)
    data = inherited.model_dump(mode="json", exclude={"setups", "protocols"})
    own = listed.model_dump(mode="json", exclude_unset=True, exclude={"setups"})
    # Requirements are separate facts, so a setup that names only its network need keeps the inherited GPU need.
    if isinstance(own.get("requirements"), dict) and isinstance(data.get("requirements"), dict):
        own["requirements"] = {**data["requirements"], **own["requirements"]}
    data.update(own)
    return EnvironmentManifest.model_validate(data)


def load_listed_manifest(path: str | Path, name: str) -> EnvironmentManifest:
    """The manifest named ``name``: the main manifest at ``path`` or a setup or protocol it lists."""
    main = load_manifest(path)
    if name == main.name:
        return main
    listed = next((listed for listed in main.listed_configs() if listed.name == name), None)
    if listed is None:
        raise ManifestError(f"Environment manifest '{path}' lists no setup or protocol named '{name}'.")
    return resolve_listed_manifest(main, listed)


def resolve_manifest_config_path(manifest_path: str | Path, manifest: EnvironmentManifest) -> Path:
    """Resolve a relative config without leaving its benchmarks/environments tree.

    Standalone manifests are confined to their own directory. Resolve symlinks
    before checking containment so a linked config cannot escape the catalog.
    """
    path = Path(manifest_path).absolute()
    config_path = Path(manifest.config_path)
    if config_path.is_absolute():
        raise ManifestError(f"Manifest '{path}' config_path must be relative.")
    catalog = next((parent for parent in path.parents if parent.name in {"benchmarks", "environments"}), path.parent)
    resolved = (path.parent / config_path).resolve()
    if not resolved.is_relative_to(catalog.resolve()):
        raise ManifestError(f"Manifest '{path}' config_path escapes catalog tree '{catalog}'.")
    return resolved


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
    data = manifest.model_dump(mode="json", exclude_none=True)
    # Keep existing scaffold/sync output stable; extensions are authored only when needed.
    if manifest.config_path == "config.yaml":
        data.pop("config_path")
    if manifest.prompt_source == PromptSource.TEMPLATE:
        data.pop("prompt_source")
    # Required nullable composition fields must survive a load/dump round trip.
    data["resources_server"] = manifest.resources_server
    # A setup or protocol keeps exactly the fields it sets, including a null that clears a main manifest's value.
    for key in ("setups", "protocols"):
        listed = [entry.model_dump(mode="json", exclude_unset=True) for entry in getattr(manifest, key)]
        if listed:
            data[key] = listed
        else:
            data.pop(key, None)
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
    )


def manifest_json_schema() -> dict[str, Any]:
    schema = EnvironmentManifest.model_json_schema(mode="validation")
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": MANIFEST_SCHEMA_ID,
        **schema,
    }
