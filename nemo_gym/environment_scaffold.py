# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile-aware scaffolding for NeMo Gym environments and benchmarks."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, Mapping, Sequence

from omegaconf import OmegaConf

from nemo_gym.environment_manifest import EnvironmentManifest, dump_manifest


EnvironmentKind = Literal["environment", "benchmark"]
IntegrationProfile = Literal["stock-loop", "measured-loop", "external-loop", "custom-driver"]

ENVIRONMENT_KINDS: tuple[EnvironmentKind, ...] = ("environment", "benchmark")
INTEGRATION_PROFILES: tuple[IntegrationProfile, ...] = (
    "stock-loop",
    "measured-loop",
    "external-loop",
    "custom-driver",
)

_ASSET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VERIFIER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)?$")
_VERIFIER_FIXTURE_RELATIVE_PATH = Path("tests/verifier_cases.jsonl")


class ScaffoldError(ValueError):
    """Base error for invalid or unfulfillable scaffold requests."""


class ScaffoldConflictError(ScaffoldError):
    """Raised when a generated file would overwrite different user content."""

    def __init__(self, paths: tuple[Path, ...]):
        self.paths = paths
        rendered = "\n".join(f"  - {path}" for path in paths)
        super().__init__(f"Scaffolding would overwrite existing files with different content:\n{rendered}")


@dataclass(frozen=True)
class ScaffoldResult:
    """The files affected by one scaffold invocation.

    ``created`` contains newly written files. ``existing`` contains files that
    already held the exact generated content. Existing files are never rewritten.
    """

    root: Path
    asset_dir: Path
    created: tuple[Path, ...]
    existing: tuple[Path, ...]

    @property
    def files(self) -> tuple[Path, ...]:
        return self.created + self.existing


@dataclass(frozen=True)
class _ReusedConfigInstance:
    name: str
    resources_servers: tuple[str, ...] = ()
    responses_api_agents: tuple[str, ...] = ()
    responses_api_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class _VerifierSelection:
    config_path: Path
    resource_instance: str
    resource_type: str
    agent_instance: str | None = None
    agent_type: str | None = None
    instances: tuple[_ReusedConfigInstance, ...] = ()
    has_rollout_driver: bool = False


_METADATA_DEFAULTS: dict[str, Any] = {
    "version": "0.1.0",
    "domain": "other",
    "description": None,
    "modality": "text",
    "licensing": "unknown",
    "authors": ("TODO",),
    "reward_range": (0.0, 1.0),
    "higher_is_better": True,
    "determinism": "unknown",
    "canonical_split": None,
}


def scaffold_environment(
    *,
    kind: EnvironmentKind,
    name: str,
    profile: IntegrationProfile = "stock-loop",
    reuse_verifier: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> ScaffoldResult:
    """Create a runnable, profile-aware environment or benchmark skeleton.

    Args:
        kind: ``environment`` or ``benchmark``. This chooses the recipe directory
            and the dataset authoring path.
        name: New recipe name. Nested paths and traversal are intentionally rejected.
        profile: Which component drives the episode.
        reuse_verifier: An existing resources-server selector (``name`` or
            ``name/flavor``). When supplied, no resources-server implementation is
            generated.
        metadata: Optional manifest values. Supported keys are the keys in
            :data:`_METADATA_DEFAULTS`; unknown keys are rejected so typos do not
            silently produce an incomplete declaration.
        root: Gym checkout or plugin root. Defaults to the current directory.

    The operation is idempotent and strictly non-destructive. A repeated identical
    request reports files through ``result.existing``. If any generated file exists
    with different contents, the function raises :class:`ScaffoldConflictError`
    before writing anything.
    """

    if kind not in ENVIRONMENT_KINDS:
        raise ScaffoldError(f"kind must be one of {ENVIRONMENT_KINDS}, got {kind!r}")
    if profile not in INTEGRATION_PROFILES:
        raise ScaffoldError(f"profile must be one of {INTEGRATION_PROFILES}, got {profile!r}")
    if not _ASSET_NAME_RE.fullmatch(name) or name in {".", ".."}:
        raise ScaffoldError(
            "name must be a single lowercase path component containing only letters, digits, '.', '_' or '-'"
        )
    if reuse_verifier is not None and not _VERIFIER_NAME_RE.fullmatch(reuse_verifier):
        raise ScaffoldError("reuse_verifier must be a resources-server name or name/config flavor")
    if profile == "custom-driver" and (not name.isidentifier() or _python_name(name) != name):
        raise ScaffoldError("custom-driver names must also be valid lowercase Python module names")

    requested_root = Path.cwd() if root is None else Path(root).expanduser()
    if requested_root.is_symlink():
        raise ScaffoldError(f"scaffold root must not be a symlink: {requested_root}")
    scaffold_root = requested_root.resolve()
    values = _manifest_metadata(kind, name, metadata)
    model_capability = _model_capability(values["modality"])
    module_name = _python_name(name)
    asset_parent = "benchmarks" if kind == "benchmark" else "environments"
    asset_dir = scaffold_root / asset_parent / name

    verifier = _resolve_reused_verifier(scaffold_root, reuse_verifier) if reuse_verifier else None
    resource_component = verifier.resource_type if verifier else module_name
    resource_instance = verifier.resource_instance if verifier else f"{module_name}_resources_server"
    agent_component = f"{module_name}_agent" if profile in {"measured-loop", "external-loop"} else "simple_agent"
    agent_instance = f"{module_name}_agent"

    files: dict[Path, str] = {}
    dataset_path = f"{asset_parent}/{name}/data/example.jsonl"
    prompt_path = f"{asset_parent}/{name}/prompt.yaml" if kind == "benchmark" else None
    prepare_path = f"{asset_parent}/{name}/prepare.py" if kind == "benchmark" else None
    driver_path = f"{asset_parent}.{module_name}.rollout_driver:run_rollout_collection"

    manifest = _build_manifest(
        kind=kind,
        name=name,
        profile=profile,
        values=values,
        model_capability=model_capability,
        resource_component=resource_component,
        agent_component=agent_component,
        dataset_name=name,
        dataset_path=dataset_path,
        prepare_path=prepare_path,
        prompt_path=prompt_path,
        rollout_driver=driver_path if profile == "custom-driver" else None,
    )
    files[asset_dir / "manifest.yaml"] = dump_manifest(manifest)
    files[asset_dir / "config.yaml"] = _asset_config_yaml(
        root=scaffold_root,
        kind=kind,
        name=name,
        module_name=module_name,
        profile=profile,
        model_capability=model_capability,
        resource_instance=resource_instance,
        agent_instance=agent_instance,
        dataset_path=dataset_path,
        prompt_path=prompt_path,
        prepare_path=prepare_path,
        verifier=verifier,
        reuse_cleanup_path=asset_dir / ".reuse_cleanup.yaml" if verifier else None,
    )
    if verifier is not None:
        files[asset_dir / ".reuse_cleanup.yaml"] = _reuse_cleanup_yaml(
            verifier=verifier,
            profile=profile,
            rollout_driver=driver_path if profile == "custom-driver" else None,
        )
    files[asset_dir / "README.md"] = _asset_readme(kind, name, profile, reuse_verifier)
    files[asset_dir / "data" / ".gitignore"] = "train.jsonl\nvalidation.jsonl\n*_metrics.json\n*_rollouts.jsonl\n"
    files[asset_dir / "__init__.py"] = _license_header()

    if kind == "benchmark":
        source_row = {"question": "What is 6 x 7?", "expected_answer": "42"}
        files[asset_dir / "data" / "source.jsonl"] = json.dumps(source_row) + "\n"
        # Keep a prepared row in the initial scaffold so it can run before the
        # contributor replaces the sample and invokes `gym eval prepare`.
        files[asset_dir / "data" / "example.jsonl"] = json.dumps(source_row) + "\n"
        files[asset_dir / "prompt.yaml"] = _benchmark_prompt_yaml()
        files[asset_dir / "prepare.py"] = _benchmark_prepare_py(name)
    else:
        files[asset_dir / "data" / "example.jsonl"] = _environment_example_jsonl(agent_instance)

    if verifier is None:
        resource_dir = scaffold_root / "resources_servers" / module_name
        files.update(
            _resources_server_files(
                scaffold_root,
                resource_dir,
                module_name,
                domain=str(values["domain"]),
                description=str(values["description"]),
                reward_range=values["reward_range"],
                higher_is_better=bool(values["higher_is_better"]),
                determinism=str(values["determinism"]),
            )
        )

    if profile in {"measured-loop", "external-loop"}:
        agent_dir = scaffold_root / "responses_api_agents" / f"{module_name}_agent"
        files.update(_agent_files(scaffold_root, agent_dir, module_name, profile))

    if profile == "custom-driver":
        files[asset_dir / "rollout_driver.py"] = _rollout_driver_py()

    return _write_plan(scaffold_root, asset_dir, files)


def default_scaffold_description(kind: EnvironmentKind | str, name: str) -> str:
    return f"Starter {kind} for {name}"


def _manifest_metadata(kind: EnvironmentKind, name: str, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    supplied = dict(metadata or {})
    unknown = sorted(set(supplied) - set(_METADATA_DEFAULTS))
    if unknown:
        raise ScaffoldError(f"unknown metadata field(s): {', '.join(unknown)}")
    if kind == "benchmark" and not supplied.get("canonical_split"):
        raise ScaffoldError("benchmark scaffolding requires metadata.canonical_split")
    values = _METADATA_DEFAULTS | supplied
    if values["description"] is None:
        values["description"] = default_scaffold_description(kind, name)

    reward_range = tuple(values["reward_range"])
    if len(reward_range) != 2 or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for value in reward_range
    ):
        raise ScaffoldError("metadata.reward_range must contain exactly two numeric endpoints")
    if reward_range[0] >= reward_range[1]:
        raise ScaffoldError("metadata.reward_range must have a minimum strictly below its maximum")
    values["reward_range"] = reward_range

    authors = values["authors"]
    if isinstance(authors, str):
        authors = (authors,)
    authors = tuple(authors)
    if not authors or not all(isinstance(author, str) and author.strip() for author in authors):
        raise ScaffoldError("metadata.authors must contain at least one non-empty name")
    values["authors"] = authors
    return values


def _python_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]", "_", name)
    return f"env_{normalized}" if normalized[0].isdigit() else normalized


def _model_capability(modality: object) -> str:
    slug = re.sub(r"[^a-z0-9._:/-]+", "-", str(modality).strip().casefold()).strip("-._:/")
    if not slug or not slug[0].isalpha():
        slug = f"modality-{slug or 'unknown'}"
    return f"{slug}-model"


def _class_name(module_name: str) -> str:
    return "".join(part.capitalize() for part in module_name.split("_") if part)


def _reused_verifier_config_path(root: Path, selector: str, server_name: str, config_stem: str) -> Path:
    local_config = root / "resources_servers" / server_name / "configs" / f"{config_stem}.yaml"
    if local_config.is_file():
        return local_config

    # Import lazily so a standalone plugin with a local config needs no global
    # registry state.
    from nemo_gym.resources_server_registry import discover_resources_servers

    entry = discover_resources_servers().get(selector)
    if entry is not None:
        return entry.config_path.resolve()
    raise ScaffoldError(
        f"reuse_verifier {selector!r} was not found; run `gym list resources-servers` to see valid selectors"
    )


def _load_reused_verifier_config(config_path: Path) -> dict[str, Any]:
    try:
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False, throw_on_missing=False)
    except Exception as exc:  # pragma: no cover - OmegaConf gives several parse exception types
        raise ScaffoldError(f"could not read reused verifier config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScaffoldError(f"reused verifier config {config_path} is not a mapping")
    return raw


def _reused_config_instances(raw: Mapping[str, Any]) -> tuple[_ReusedConfigInstance, ...]:
    instances: list[_ReusedConfigInstance] = []
    for instance_name, instance in raw.items():
        if not isinstance(instance, dict):
            continue
        inventory = _ReusedConfigInstance(
            name=str(instance_name),
            resources_servers=(
                tuple(str(item) for item in instance["resources_servers"])
                if isinstance(instance.get("resources_servers"), dict)
                else ()
            ),
            responses_api_agents=(
                tuple(str(item) for item in instance["responses_api_agents"])
                if isinstance(instance.get("responses_api_agents"), dict)
                else ()
            ),
            responses_api_models=(
                tuple(str(item) for item in instance["responses_api_models"])
                if isinstance(instance.get("responses_api_models"), dict)
                else ()
            ),
        )
        if inventory.resources_servers or inventory.responses_api_agents or inventory.responses_api_models:
            instances.append(inventory)
    return tuple(instances)


def _select_reused_resource(
    config_path: Path,
    server_name: str,
    instances: Sequence[_ReusedConfigInstance],
) -> tuple[str, str]:
    resources = [(instance.name, instance.resources_servers) for instance in instances if instance.resources_servers]
    if not resources:
        raise ScaffoldError(f"reused verifier config {config_path} defines no resources server")
    if len(resources) > 1:
        preferred = [
            candidate for candidate in resources if candidate[0] in {server_name, f"{server_name}_resources_server"}
        ]
        if len(preferred) != 1:
            rendered = ", ".join(candidate[0] for candidate in resources)
            raise ScaffoldError(f"reused verifier config {config_path} is ambiguous; resources instances: {rendered}")
        resource_instance, resource_types = preferred[0]
    else:
        resource_instance, resource_types = resources[0]

    if len(resource_types) == 1:
        resource_type = resource_types[0]
    elif server_name in resource_types:
        resource_type = server_name
    else:
        rendered = ", ".join(resource_types)
        raise ScaffoldError(
            f"reused verifier config {config_path} is ambiguous; resources implementations: {rendered}"
        )
    return resource_instance, resource_type


def _reused_resource_capabilities(
    raw: Mapping[str, Any],
    resource_instance: str,
    resource_type: str,
) -> set[str]:
    resource_instance_config = raw.get(resource_instance)
    resource_declarations = (
        resource_instance_config.get("resources_servers") if isinstance(resource_instance_config, dict) else None
    )
    resource_config = resource_declarations.get(resource_type) if isinstance(resource_declarations, dict) else None
    declared_provides = resource_config.get("provides") if isinstance(resource_config, dict) else None
    if isinstance(declared_provides, str):
        provided_capabilities = {declared_provides}
    elif isinstance(declared_provides, list):
        provided_capabilities = {str(capability) for capability in declared_provides}
    else:
        provided_capabilities = set()
    return provided_capabilities


def _reused_simple_agent(raw: Mapping[str, Any], resource_instance: str) -> tuple[str, str] | None:
    matching_agents: list[tuple[str, str]] = []
    for instance_name, instance in raw.items():
        if not isinstance(instance, dict):
            continue
        agents = instance.get("responses_api_agents")
        if not isinstance(agents, dict):
            continue
        for agent_type, agent_config in agents.items():
            if not isinstance(agent_config, dict):
                continue
            resource_ref = agent_config.get("resources_server")
            if isinstance(resource_ref, dict) and resource_ref.get("name") == resource_instance:
                matching_agents.append((str(instance_name), str(agent_type)))
    return next((candidate for candidate in matching_agents if candidate[1] == "simple_agent"), None)


def _require_reused_verifier_fixture(config_path: Path, selector: str, resource_type: str) -> None:
    fixture_path = config_path.parent.parent / _VERIFIER_FIXTURE_RELATIVE_PATH
    if fixture_path.is_file():
        return
    raise ScaffoldError(
        f"reuse_verifier {selector!r} selects resources server {resource_type!r}, but its canonical verifier "
        f"fixture is missing at {fixture_path}. Add full_reward, zero_reward, and malformed cases there "
        f"(plus determinism_reseed when seeded), then make `gym env test --resources-server {selector}` pass "
        "before reusing this scorer."
    )


def _resolve_reused_verifier(root: Path, selector: str) -> _VerifierSelection:
    server_name, slash, flavor = selector.partition("/")
    config_stem = flavor if slash else server_name
    config_path = _reused_verifier_config_path(root, selector, server_name, config_stem)
    raw = _load_reused_verifier_config(config_path)
    instances = _reused_config_instances(raw)
    resource_instance, resource_type = _select_reused_resource(config_path, server_name, instances)
    if "verification" not in _reused_resource_capabilities(raw, resource_instance, resource_type):
        raise ScaffoldError(
            f"reuse_verifier {selector!r} selects resources server {resource_type!r} in {config_path}, "
            "but that declaration does not explicitly provide the 'verification' capability. "
            "Choose a declared scorer with `gym list components --provides verification`."
        )
    _require_reused_verifier_fixture(config_path, selector, resource_type)
    simple_agent = _reused_simple_agent(raw, resource_instance)
    agent_instance, agent_type = simple_agent if simple_agent is not None else (None, None)

    return _VerifierSelection(
        config_path=config_path.resolve(),
        resource_instance=resource_instance,
        resource_type=resource_type,
        agent_instance=agent_instance,
        agent_type=agent_type,
        instances=instances,
        has_rollout_driver=bool(raw.get("rollout_collection_driver")),
    )


def _stock_agent_inheritance(verifier: _VerifierSelection, profile: IntegrationProfile) -> tuple[str, str] | None:
    """Return a safe stock-agent source, leaving mixed config instances as baggage.

    ``_inherit_from`` moves an entire top-level instance. Only inherit the common
    MCQA shape (one simple agent in an otherwise agent-only instance); moving a
    combined resource/agent/model instance would rename the selected verifier or
    retain an unrelated implementation.
    """

    if profile != "stock-loop" or not verifier.agent_instance or not verifier.agent_type:
        return None
    source = next((item for item in verifier.instances if item.name == verifier.agent_instance), None)
    if source is None:
        return None
    if source.resources_servers or source.responses_api_models:
        return None
    if source.responses_api_agents != (verifier.agent_type,):
        return None
    return verifier.agent_instance, verifier.agent_type


def _reuse_cleanup_yaml(
    *,
    verifier: _VerifierSelection,
    profile: IntegrationProfile,
    rollout_driver: str | None,
) -> str:
    """Neutralize composition bundled beside a reused verifier.

    Resources-server configs often bundle a stock agent and its datasets (and
    occasionally models or another verifier). Loading that config verbatim would
    make the scaffold compose multiple agents/datasets. The cleanup file is
    deliberately listed after the reused config so its directives see the fully
    merged instance and retain only the selected verifier implementation.
    """

    inherited_agent = _stock_agent_inheritance(verifier, profile)
    inherited_agent_instance = inherited_agent[0] if inherited_agent else None
    lines = [
        "# Keep only the selected verifier from the reused component config.",
        "# This file is loaded after that config so _delete_key sees inherited baggage.",
    ]
    for instance in verifier.instances:
        delete_groups: list[str] = []
        if instance.responses_api_agents and instance.name != inherited_agent_instance:
            delete_groups.append("responses_api_agents")
        if instance.responses_api_models:
            delete_groups.append("responses_api_models")
        if instance.resources_servers and instance.name != verifier.resource_instance:
            delete_groups.append("resources_servers")

        unselected_resources = (
            tuple(item for item in instance.resources_servers if item != verifier.resource_type)
            if instance.name == verifier.resource_instance
            else ()
        )
        if not delete_groups and not unselected_resources:
            continue
        lines.append(f"{_yaml_scalar(instance.name)}:")
        if delete_groups:
            lines.append(f"  _delete_key: {_yaml_scalar(', '.join(delete_groups))}")
        if unselected_resources:
            lines.extend(
                [
                    "  resources_servers:",
                    f"    _delete_key: {_yaml_scalar(', '.join(unselected_resources))}",
                ]
            )

    if verifier.has_rollout_driver:
        lines.extend(
            [
                "rollout_collection_driver: "
                + (_yaml_scalar(rollout_driver) if rollout_driver is not None else "null"),
            ]
        )
    return "\n".join(lines) + "\n"


def _build_manifest(
    *,
    kind: EnvironmentKind,
    name: str,
    profile: IntegrationProfile,
    values: Mapping[str, Any],
    model_capability: str,
    resource_component: str,
    agent_component: str,
    dataset_name: str,
    dataset_path: str,
    prepare_path: str | None,
    prompt_path: str | None,
    rollout_driver: str | None,
) -> EnvironmentManifest:
    minimum, maximum = values["reward_range"]
    dataset: dict[str, Any] = {
        "name": dataset_name,
        "type": "benchmark" if kind == "benchmark" else "example",
        "jsonl_fpath": dataset_path,
        "num_repeats": 1,
    }
    if kind == "benchmark":
        dataset.update(prepare_script=prepare_path, prompt_config=prompt_path)

    data: dict[str, Any] = {
        "name": name,
        "version": values["version"],
        "kind": kind,
        "integration_profile": profile,
        "domain": values["domain"],
        "description": values["description"],
        "modality": values["modality"],
        "licensing": values["licensing"],
        "authors": list(values["authors"]),
        "reward": {
            "range": [minimum, maximum],
            "higher_is_better": bool(values["higher_is_better"]),
        },
        "determinism": values["determinism"],
        "resources_server": resource_component,
        "agent_server": agent_component,
        "model_server": "policy_model",
        "datasets": [dataset],
        "requires": [model_capability],
        "provides": ["verification"],
    }
    if kind == "benchmark":
        data.update(
            canonical_split=values["canonical_split"],
            standard_prompt_config=prompt_path,
        )
    if rollout_driver is not None:
        data["rollout_driver"] = rollout_driver

    try:
        return EnvironmentManifest.model_validate(data)
    except ValueError as error:
        raise ScaffoldError(f"generated manifest is invalid: {error}") from error


def _asset_config_yaml(
    *,
    root: Path,
    kind: EnvironmentKind,
    name: str,
    module_name: str,
    profile: IntegrationProfile,
    model_capability: str,
    resource_instance: str,
    agent_instance: str,
    dataset_path: str,
    prompt_path: str | None,
    prepare_path: str | None,
    verifier: _VerifierSelection | None,
    reuse_cleanup_path: Path | None,
) -> str:
    config_path = (
        verifier.config_path
        if verifier
        else root / "resources_servers" / module_name / "configs" / f"{module_name}.yaml"
    )
    config_ref = _portable_path(config_path, root)
    config_refs = [config_ref]
    if reuse_cleanup_path is not None:
        config_refs.append(_portable_path(reuse_cleanup_path, root))
    lines = [
        "# The recipe composes deployable components; manifest.yaml declares their meaning.",
        "config_paths:",
        *(f"  - {reference}" for reference in config_refs),
        "",
    ]

    inherited_agent = _stock_agent_inheritance(verifier, profile) if verifier else None
    if inherited_agent is not None:
        inherited_agent_instance, inherited_agent_type = inherited_agent
        lines.extend(
            [
                f"{agent_instance}:",
                f"  _inherit_from: {inherited_agent_instance}",
                "  responses_api_agents:",
                f"    {inherited_agent_type}:",
                "      resources_server:",
                "        type: resources_servers",
                f"        name: {resource_instance}",
            ]
        )
    else:
        agent_type = f"{module_name}_agent" if profile in {"measured-loop", "external-loop"} else "simple_agent"
        lines.extend(
            [
                f"{agent_instance}:",
                "  responses_api_agents:",
                f"    {agent_type}:",
                "      entrypoint: app.py",
            ]
        )
        lines.append(f"      requires: [verification, {_yaml_scalar(model_capability)}]")
        lines.extend(
            [
                "      resources_server:",
                "        type: resources_servers",
                f"        name: {resource_instance}",
                "      model_server:",
                "        type: responses_api_models",
                "        name: policy_model",
            ]
        )

    lines.extend(
        [
            "      datasets:",
            f"        - name: {name}",
            f"          type: {'benchmark' if kind == 'benchmark' else 'example'}",
            f"          jsonl_fpath: {dataset_path}",
        ]
    )
    if kind == "benchmark":
        lines.extend([f"          prompt_config: {prompt_path}", f"          prepare_script: {prepare_path}"])
    lines.append("          num_repeats: 1")
    if profile == "custom-driver":
        lines.extend(
            [
                "",
                f"rollout_collection_driver: {'benchmarks' if kind == 'benchmark' else 'environments'}."
                f"{module_name}.rollout_driver:run_rollout_collection",
            ]
        )
    return "\n".join(lines) + "\n"


def _asset_readme(kind: EnvironmentKind, name: str, profile: IntegrationProfile, reuse_verifier: str | None) -> str:
    kind_title = "Benchmark" if kind == "benchmark" else "Environment"
    verifier_line = (
        f"This recipe reuses the `{reuse_verifier}` resources server."
        if reuse_verifier
        else f"The starter verifier lives in `resources_servers/{_python_name(name)}/`."
    )
    test_commands = []
    if not reuse_verifier:
        test_commands.append(f"gym env test {name} --update-expected  # generate expectations, then review the diff")
    test_commands.append(f"gym env test {name}                    # read-only check used by CI")
    rendered_test_commands = "\n        ".join(test_commands)
    return dedent(
        f"""\
        # {name}

        TODO: Describe this {kind} and its intended use.

        - Kind: `{kind}`
        - Integration profile: `{profile}`
        - {verifier_line}

        ## Start locally

        ```bash
        gym env validate {name}
        {rendered_test_commands}
        gym env start {name} --model-type openai_model
        ```

        Replace the sample task and stub behavior while keeping each checkpoint runnable.

        ## Licensing

        - Code: TODO
        - Data: TODO

        ## {kind_title} provenance

        TODO: Record the task/data source and pinned upstream revision when applicable.
        """
    )


def _benchmark_prompt_yaml() -> str:
    return dedent(
        """\
        # Standard prompt for this benchmark. Keep it separate from domain JSONL.
        user: |-
          Answer the following task. Put only the final answer on the last line.

          {question}
        """
    )


def _benchmark_prepare_py(name: str) -> str:
    return _license_header() + dedent(
        f'''\
        """Prepare raw domain rows for the {name} benchmark."""

        from pathlib import Path


        BENCHMARK_DIR = Path(__file__).parent
        SOURCE_PATH = BENCHMARK_DIR / "data" / "source.jsonl"
        OUTPUT_PATH = BENCHMARK_DIR / "data" / "example.jsonl"


        def prepare() -> Path:
            """Produce domain JSONL; prompt_config materializes Responses API input later."""
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            # TODO: Download/transform the canonical split. The starter copy keeps
            # the scaffold runnable before real data preparation is implemented.
            OUTPUT_PATH.write_text(SOURCE_PATH.read_text())
            return OUTPUT_PATH


        if __name__ == "__main__":
            prepare()
        '''
    )


def _environment_example_jsonl(agent_instance: str) -> str:
    row = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "What is 6 x 7? Reply with only the answer."}]
        },
        "expected_answer": "42",
        "agent_ref": {"type": "responses_api_agents", "name": agent_instance},
    }
    return json.dumps(row) + "\n"


def _resources_server_files(
    root: Path,
    resource_dir: Path,
    module_name: str,
    *,
    domain: str,
    description: str,
    reward_range: tuple[int | float, int | float],
    higher_is_better: bool,
    determinism: str,
) -> dict[Path, str]:
    class_name = _class_name(module_name)
    config_path = resource_dir / "configs" / f"{module_name}.yaml"
    requirements = _requirements_text(root, resource_dir)
    return {
        resource_dir / "__init__.py": _license_header(),
        resource_dir / "app.py": _resources_server_app_py(class_name, reward_range, higher_is_better),
        config_path: _resources_server_config_yaml(module_name, domain, description),
        resource_dir / "requirements.txt": requirements,
        resource_dir / "README.md": _resources_server_readme(module_name),
        resource_dir / "tests" / "__init__.py": _license_header(),
        resource_dir / "tests" / "test_app.py": _resources_server_test_py(
            module_name, reward_range, higher_is_better, determinism
        ),
        resource_dir / "tests" / "verifier_cases.jsonl": _verifier_cases_jsonl(),
    }


def _resources_server_config_yaml(module_name: str, domain: str, description: str) -> str:
    return dedent(
        f"""\
        # Reusable verifier, tools, and per-task state for this environment.
        {module_name}_resources_server:
          resources_servers:
            {module_name}:
              entrypoint: app.py
              domain: {_yaml_scalar(domain)}
              provides: [verification]
              verified: false
              description: {_yaml_scalar(description)}
        """
    )


def _resources_server_app_py(
    class_name: str,
    reward_range: tuple[int | float, int | float],
    higher_is_better: bool,
) -> str:
    minimum, maximum = reward_range
    full_reward, zero_reward = (maximum, minimum) if higher_is_better else (minimum, maximum)
    return _license_header() + dedent(
        f'''\
        from collections.abc import Mapping
        from typing import Any, ClassVar

        from fastapi import FastAPI
        from pydantic import ConfigDict

        from nemo_gym.base_resources_server import (
            BaseResourcesServerConfig,
            BaseVerifyRequest,
            BaseVerifyResponse,
            ReverifyMode,
            SimpleResourcesServer,
        )
        from nemo_gym.verifier_fixture import build_offline_verifier_app


        class {class_name}ResourcesServerConfig(BaseResourcesServerConfig):
            REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


        class {class_name}VerifyRequest(BaseVerifyRequest):
            model_config = ConfigDict(extra="allow")

            expected_answer: str


        class {class_name}ResourcesServer(SimpleResourcesServer):
            config: {class_name}ResourcesServerConfig

            async def verify(self, body: {class_name}VerifyRequest) -> BaseVerifyResponse:
                """Starter exact-match verifier; replace this while keeping fixture cases green."""
                output_text = ""
                for item in body.response.output:
                    if item.type == "message" and item.role == "assistant":
                        output_text = "".join(
                            content.text for content in item.content if content.type == "output_text"
                        ).strip()
                reward = {full_reward!r} if output_text == body.expected_answer else {zero_reward!r}
                return BaseVerifyResponse(**body.model_dump(), reward=reward)


        def create_offline_verifier_app(
            *, server_config: Mapping[str, Any], instance_name: str
        ) -> FastAPI:
            return build_offline_verifier_app(
                {class_name}ResourcesServer,
                server_config=server_config,
                instance_name=instance_name,
            )


        if __name__ == "__main__":
            {class_name}ResourcesServer.run_webserver()
        '''
    )


def _resources_server_test_py(
    module_name: str,
    reward_range: tuple[int | float, int | float],
    higher_is_better: bool,
    determinism: str,
) -> str:
    minimum, maximum = reward_range
    return _license_header() + dedent(
        f'''\
        import json
        import os
        from pathlib import Path

        from fastapi.testclient import TestClient

        from nemo_gym.verifier_fixture import (
            DETERMINISM_ENV_VAR,
            HIGHER_IS_BETTER_ENV_VAR,
            REWARD_RANGE_ENV_VAR,
            UPDATE_EXPECTED_ENV_VAR,
            exercise_verifier_fixture,
            load_verifier_fixture,
        )
        from resources_servers.{module_name}.app import create_offline_verifier_app


        CASES_PATH = Path(__file__).with_name("verifier_cases.jsonl")
        CASES = load_verifier_fixture(CASES_PATH)


        def _client() -> TestClient:
            return TestClient(
                create_offline_verifier_app(
                    server_config={{"entrypoint": "app.py"}},
                    instance_name="{module_name}_resources_server",
                )
            )


        def test_verifier_fixture() -> None:
            reward_range = tuple(
                json.loads(os.getenv(REWARD_RANGE_ENV_VAR, "[{minimum!r}, {maximum!r}]"))
            )
            exercise_verifier_fixture(
                _client,
                CASES_PATH,
                reward_range=reward_range,
                higher_is_better=os.getenv(
                    HIGHER_IS_BETTER_ENV_VAR, {str(higher_is_better).lower()!r}
                ).casefold() == "true",
                determinism=os.getenv(DETERMINISM_ENV_VAR, {determinism!r}),
                update_expected=os.getenv(UPDATE_EXPECTED_ENV_VAR) == "1",
            )
        '''
    )


def _verifier_cases_jsonl() -> str:
    def response(text: str) -> dict[str, Any]:
        return {
            "id": "response_fixture",
            "created_at": 0.0,
            "model": "fixture",
            "object": "response",
            "output": [
                {
                    "id": "message_fixture",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }

    def request(text: str, expected: str) -> dict[str, Any]:
        return {
            "responses_create_params": {"input": [{"role": "user", "content": "What is 6 x 7?"}]},
            "response": response(text),
            "expected_answer": expected,
        }

    cases = [
        {
            "case": "full_reward",
            "request": request("42", "42"),
            "expected_status": "TODO",
        },
        {
            "case": "zero_reward",
            "request": request("41", "42"),
            "expected_status": "TODO",
        },
        {
            "case": "malformed",
            "request": {"responses_create_params": {"input": [{"role": "user", "content": "missing response"}]}},
            "expected_status": "TODO",
        },
        {
            "case": "determinism_reseed",
            "request": request("42", "42"),
            "expected_status": "TODO",
            "reseed": True,
        },
    ]
    return "".join(json.dumps(case) + "\n" for case in cases)


def _resources_server_readme(module_name: str) -> str:
    return dedent(
        f"""\
        # {module_name} resources server

        This component owns verification, environment-specific tools, and per-task state.
        The generated verifier is an exact-match stub. Replace it with domain logic, then generate
        the fixture's TODO expectations and inspect the diff before publishing.

        ```bash
        gym env test --resources-server {module_name} --update-expected
        gym env test --resources-server {module_name}
        ```

        ## Licensing

        - Code: TODO
        - Data or external systems used by verification: TODO
        """
    )


def _agent_files(root: Path, agent_dir: Path, module_name: str, profile: IntegrationProfile) -> dict[Path, str]:
    class_name = _class_name(module_name)
    component_name = f"{module_name}_agent"
    return {
        agent_dir / "__init__.py": _license_header(),
        agent_dir / "app.py": _agent_app_py(class_name, profile),
        agent_dir / "requirements.txt": _requirements_text(root, agent_dir),
        agent_dir / "README.md": _agent_readme(component_name, profile),
        agent_dir / "tests" / "__init__.py": _license_header(),
        agent_dir / "tests" / "test_app.py": _agent_test_py(module_name, class_name),
    }


def _agent_app_py(class_name: str, profile: IntegrationProfile) -> str:
    if profile == "measured-loop":
        doc = "Customize responses() here; this harness behaviour is part of the measurement."
        imports = [
            "from fastapi import Request, Response",
            "",
            "from nemo_gym.base_responses_api_agent import Body",
            "from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming",
            "from responses_api_agents.simple_agent.app import SimpleAgent",
        ]
    else:
        doc = "Replace run() with the external framework adapter; the pass-through keeps the scaffold runnable."
        imports = ["from responses_api_agents.simple_agent.app import SimpleAgent"]
    lines = [
        *imports,
        "",
        "",
        f"class {class_name}Agent(SimpleAgent):",
        f'    """{doc}"""',
    ]
    if profile == "measured-loop":
        lines.extend(
            [
                "",
                "    async def responses(",
                "        self,",
                "        request: Request,",
                "        response: Response,",
                "        body: NeMoGymResponseCreateParamsNonStreaming = Body(),",
                "    ) -> NeMoGymResponse:",
                "        # TODO: Implement the harness strategy that is part of this measurement.",
                "        return await super().responses(request, response, body)",
            ]
        )
    elif profile == "external-loop":
        lines.extend(
            [
                "",
                "    async def run(self, request, body):",
                "        # TODO: Delegate the complete episode to the external framework and",
                "        # adapt its trajectory before verification.",
                "        return await super().run(request, body)",
            ]
        )
    lines.extend(
        [
            "",
            "",
            'if __name__ == "__main__":',
            f"    {class_name}Agent.run_webserver()",
        ]
    )
    return _license_header() + "\n".join(lines) + "\n"


def _agent_test_py(module_name: str, class_name: str) -> str:
    return _license_header() + dedent(
        f'''\
        from unittest.mock import MagicMock

        from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
        from nemo_gym.server_utils import ServerClient
        from responses_api_agents.{module_name}_agent.app import {class_name}Agent
        from responses_api_agents.simple_agent.app import SimpleAgentConfig


        def test_sanity() -> None:
            agent = {class_name}Agent(
                config=SimpleAgentConfig(
                    host="127.0.0.1",
                    port=8080,
                    entrypoint="app.py",
                    name="{module_name}_agent",
                    resources_server=ResourcesServerRef(type="resources_servers", name="resource"),
                    model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
                ),
                server_client=MagicMock(spec=ServerClient),
            )
            assert agent.setup_webserver() is not None
        '''
    )


def _agent_readme(component_name: str, profile: IntegrationProfile) -> str:
    responsibility = (
        "the measured harness strategy" if profile == "measured-loop" else "an external framework's episode loop"
    )
    return dedent(
        f"""\
        # {component_name}

        This agent server is the integration point for {responsibility}. The generated
        pass-through implementation runs immediately; replace only the marked integration point.
        """
    )


def _rollout_driver_py() -> str:
    return _license_header() + dedent(
        '''\
        """Custom rollout-driver integration point."""

        from collections.abc import Mapping
        from typing import Any


        async def run_rollout_collection(rollout_collection_config, global_config_dict: Mapping[str, Any]):
            """Delegate to Gym's stock collector until custom orchestration is added."""
            from nemo_gym.rollout_collection import RolloutCollectionHelper

            # TODO: Add orchestration above the agent while preserving standard output artifacts.
            await RolloutCollectionHelper().run_from_config(rollout_collection_config)
        '''
    )


def _requirements_text(root: Path, component_dir: Path) -> str:
    if (root / "pyproject.toml").is_file():
        relative_root = os.path.relpath(root, component_dir)
        return f"-e nemo-gym[dev] @ {relative_root}\n"
    return "nemo-gym[dev]\n"


def _portable_path(path: Path, root: Path) -> str:
    """Express a config path relative to one of Gym's component roots.

    Config discovery can select a verifier from an extra/plugin root. Embedding
    that root's absolute machine path would make the generated recipe
    non-portable, while the root-relative component path is resolved by Gym's
    normal ordered search at load time.
    """

    from nemo_gym import component_search_roots

    resolved_path = path.resolve()
    candidates = [root, *component_search_roots()]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved_root = candidate.expanduser().resolve()
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        try:
            relative = resolved_path.relative_to(resolved_root)
        except ValueError:
            continue
        return relative.as_posix()
    raise ScaffoldError(
        f"reused verifier config '{resolved_path}' is not under the scaffold root or a component search root; "
        "cannot write a portable config reference"
    )


def _validate_write_target(root: Path, path: Path) -> None:
    """Reject targets that escape ``root`` or traverse any symlink below it."""

    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ScaffoldError(f"scaffold write target is outside root '{root}': {path}") from error
    if ".." in relative.parts:
        raise ScaffoldError(f"scaffold write target is outside root '{root}': {path}")

    current = root
    if current.is_symlink():
        raise ScaffoldError(f"scaffold write target traverses symlink '{current}': {path}")
    for part in relative.parts:
        current /= part
        # is_symlink() also detects dangling links, unlike exists().
        if current.is_symlink():
            raise ScaffoldError(f"scaffold write target traverses symlink '{current}': {path}")

    resolved_root = root.resolve()
    resolved_target = path.resolve(strict=False)
    if not resolved_target.is_relative_to(resolved_root):
        raise ScaffoldError(
            f"scaffold write target resolves outside root '{resolved_root}': {path} -> {resolved_target}"
        )


def _write_plan(root: Path, asset_dir: Path, files: Mapping[Path, str]) -> ScaffoldResult:
    ordered = sorted(files.items(), key=lambda item: item[0].as_posix())
    # Complete containment/symlink preflight before checking content or writing
    # any file. This keeps a mixed safe/unsafe plan atomic.
    for path, _content in ordered:
        _validate_write_target(root, path)

    conflicts: list[Path] = []
    existing: list[Path] = []
    for path, content in ordered:
        if path.exists():
            if not path.is_file() or path.read_text() != content:
                conflicts.append(path)
            else:
                existing.append(path)
        parent = path.parent
        while parent != root.parent and parent != parent.parent:
            if parent.exists() and not parent.is_dir():
                conflicts.append(parent)
                break
            if parent == root:
                break
            parent = parent.parent
    if conflicts:
        raise ScaffoldConflictError(tuple(sorted(set(conflicts), key=str)))

    created: list[Path] = []
    for path, content in ordered:
        if path in existing:
            continue
        # Recheck immediately before mutation to narrow the window in which a
        # parent could be replaced after preflight.
        _validate_write_target(root, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x") as stream:
                stream.write(content)
        except FileExistsError:
            # A concurrent creator is safe only when it produced the same file.
            if not path.is_file() or path.read_text() != content:
                raise ScaffoldConflictError((path,)) from None
            existing.append(path)
        else:
            created.append(path)

    return ScaffoldResult(
        root=root,
        asset_dir=asset_dir,
        created=tuple(created),
        existing=tuple(sorted(existing, key=str)),
    )


def _yaml_scalar(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _license_header() -> str:
    return dedent(
        """\
        # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
        # SPDX-License-Identifier: Apache-2.0

        """
    )
