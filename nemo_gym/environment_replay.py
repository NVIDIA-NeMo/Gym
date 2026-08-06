# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier-only replay of previously collected rollout trajectories."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf
from pydantic import Field

import nemo_gym.global_config as global_config_module
from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.config_types import BaseNeMoGymCLIConfig, ConfigError
from nemo_gym.discovery import SERVER_GROUP_KEYS
from nemo_gym.environment_manifest import EnvironmentManifest
from nemo_gym.path_utils import failures_path_for
from nemo_gym.trajectory_bundle import (
    DEFAULT_TRAJECTORY_IDENTITY_FIELDS,
    FailureReplaySelection,
    bundle_path_for,
    load_trajectory_bundle,
    validate_verifier_compatibility,
)


_SERVER_GROUPS = frozenset(SERVER_GROUP_KEYS)


class EnvironmentReplayConfig(BaseNeMoGymCLIConfig):
    """Arguments supplied by ``gym env test NAME --replay RUN.jsonl``."""

    environment_ref: str
    manifest_path: Path
    replay_rollouts_path: Path
    output_jsonl_fpath: Path | None = None
    force: bool = False
    limit: int | None = Field(default=None, ge=1)
    num_samples_in_parallel: int | None = Field(default=None, ge=1)
    failure_trajectories: FailureReplaySelection = FailureReplaySelection.LATEST_REPLAYABLE


@dataclass(frozen=True)
class ReplayPaths:
    rollouts: Path
    materialized_inputs: Path
    failures: Path | None
    bundle: Path | None
    output: Path


@dataclass(frozen=True)
class ReplayResult:
    paths: ReplayPaths
    started_components: tuple[str, ...]
    rows: int


def _absolute_output_path(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (Path.cwd() / expanded).resolve()


def infer_replay_paths(run_path: str | Path, output_path: str | Path | None = None) -> ReplayPaths:
    """Resolve a capture bundle, or infer legacy sibling paths for an old rollout JSONL."""

    selected = _resolve_under_cwd_or_install(Path(run_path).expanduser()).resolve()
    if not selected.is_file():
        raise ConfigError(f"Replay capture was not found: '{selected}'.")
    selected_is_bundle = selected.name.endswith(".bundle.json")
    if not selected_is_bundle and selected.suffix != ".jsonl":
        raise ConfigError(f"Replay capture must be a .bundle.json or .jsonl file; got '{selected}'.")

    candidate_bundle = selected if selected_is_bundle else bundle_path_for(selected)
    if candidate_bundle.is_file():
        _bundle, artifacts = load_trajectory_bundle(candidate_bundle)
        rollouts = artifacts["successes"]
        materialized = artifacts["inputs"]
        failures = artifacts["failures"]
        assert rollouts is not None and materialized is not None
        if not selected_is_bundle and rollouts != selected:
            raise ConfigError(
                f"Trajectory bundle '{candidate_bundle}' names successes at '{rollouts}', not selected file '{selected}'."
            )
        resolved_bundle: Path | None = candidate_bundle
    else:
        rollouts = selected
        materialized = rollouts.with_stem(rollouts.stem + "_materialized_inputs").with_suffix(".jsonl")
        if not materialized.is_file():
            raise ConfigError(
                f"Replay needs the captured materialized inputs at '{materialized}'. "
                "Keep this sibling file from `gym eval run --save-trajectories`, or use `gym eval reverify` "
                "with an explicit --inputs path."
            )
        legacy_failures = failures_path_for(rollouts)
        failures = legacy_failures if legacy_failures.is_file() else None
        resolved_bundle = None

    output = (
        _absolute_output_path(Path(output_path))
        if output_path is not None
        else rollouts.with_stem(rollouts.stem + "_replayed").with_suffix(".jsonl")
    )
    protected_paths = {rollouts, materialized, *(path for path in (failures, resolved_bundle) if path is not None)}
    if output in protected_paths:
        raise ConfigError("Replay output must differ from every captured bundle artifact.")
    existing = [path for path in (output, failures_path_for(output)) if path.exists()]
    if existing:
        rendered = ", ".join(f"'{path}'" for path in existing)
        raise ConfigError(
            f"Replay never overwrites prior results; output already exists: {rendered}. "
            "Choose a different path with --output."
        )
    return ReplayPaths(
        rollouts=rollouts,
        materialized_inputs=materialized,
        failures=failures,
        bundle=resolved_bundle,
        output=output,
    )


def _server_groups(block: object) -> frozenset[str]:
    if not isinstance(block, (Mapping, DictConfig)):
        return frozenset()
    return frozenset(group for group in _SERVER_GROUPS if isinstance(block.get(group), (Mapping, DictConfig)))


def _server_references(value: object) -> Iterator[tuple[str, str]]:
    if isinstance(value, (Mapping, DictConfig)):
        ref_type = value.get("type")
        ref_name = value.get("name")
        if ref_type in _SERVER_GROUPS and isinstance(ref_name, str) and ref_name:
            yield str(ref_type), ref_name
        for item in value.values():
            yield from _server_references(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _server_references(item)


def _selected_resource_instance(config: Mapping[str, Any] | DictConfig, manifest: EnvironmentManifest) -> str:
    implementation = manifest.resources_server
    if implementation is None:  # manifest validation normally makes this unreachable
        raise ConfigError(f"Environment '{manifest.name}' does not declare a resources_server.")

    candidates = {
        str(instance)
        for instance, block in config.items()
        if isinstance(block, (Mapping, DictConfig))
        and isinstance(block.get("resources_servers"), (Mapping, DictConfig))
        and implementation in block["resources_servers"]
    }
    agent_selected: set[str] = set()
    if manifest.agent_server is not None:
        for block in config.values():
            if not isinstance(block, (Mapping, DictConfig)):
                continue
            agents = block.get("responses_api_agents")
            if not isinstance(agents, (Mapping, DictConfig)) or manifest.agent_server not in agents:
                continue
            agent_config = agents[manifest.agent_server]
            if not isinstance(agent_config, (Mapping, DictConfig)):
                continue
            reference = agent_config.get("resources_server")
            if isinstance(reference, (Mapping, DictConfig)) and str(reference.get("name") or "") in candidates:
                agent_selected.add(str(reference["name"]))
    if len(agent_selected) == 1:
        return next(iter(agent_selected))
    if len(candidates) == 1:
        return next(iter(candidates))
    rendered = ", ".join(sorted(candidates)) or "none"
    raise ConfigError(
        f"Environment '{manifest.name}' must resolve one resources-server instance for replay; found {rendered}."
    )


def verifier_only_config(
    config: Mapping[str, Any] | DictConfig,
    manifest: EnvironmentManifest,
) -> tuple[DictConfig, tuple[str, ...]]:
    """Keep only the selected verifier and its transitive server dependencies.

    Agent servers are never dependencies of verification: encountering one is a
    loud error instead of accidentally replaying the policy interaction loop.
    Model servers are retained only when the verifier explicitly references them
    (for example, an LLM judge).
    """

    selected = _selected_resource_instance(config, manifest)
    included_groups: dict[str, str] = {selected: "resources_servers"}
    included_implementations: dict[str, str] = {selected: str(manifest.resources_server)}
    pending = [selected]
    while pending:
        instance = pending.pop()
        block = config.get(instance)
        if not isinstance(block, (Mapping, DictConfig)):
            raise ConfigError(f"Verifier dependency '{instance}' is not a configured server instance.")
        required_group = included_groups[instance]
        implementations = block.get(required_group)
        if not isinstance(implementations, (Mapping, DictConfig)) or not implementations:
            raise ConfigError(
                f"Verifier '{selected}' expects {required_group}/'{instance}', but that instance does not "
                "provide the required server type."
            )
        implementation = included_implementations.get(instance) or str(next(iter(implementations)))
        required_config = implementations.get(implementation)
        if not isinstance(required_config, (Mapping, DictConfig)):
            raise ConfigError(
                f"Verifier '{selected}' expects {required_group}/'{instance}' implementation "
                f"'{implementation}', but it is not configured."
            )
        included_implementations[instance] = implementation
        # Inspect only the server type that verification actually needs. A
        # resolved Hydra block may contain several co-located server groups;
        # references under an agent group must not pull the policy graph into
        # replay, and the group itself must never be handed to RunHelper.
        for ref_type, dependency in _server_references(required_config):
            if ref_type == "responses_api_agents":
                raise ConfigError(
                    f"Verifier '{selected}' references agent server '{dependency}'. Replay refuses to start an "
                    "agent or rerun the policy; move scoring dependencies behind the resources server."
                )
            dependency_block = config.get(dependency)
            if not isinstance(dependency_block, (Mapping, DictConfig)):
                raise ConfigError(f"Verifier '{selected}' references missing dependency '{dependency}'.")
            dependency_groups = _server_groups(dependency_block)
            if ref_type not in dependency_groups:
                raise ConfigError(
                    f"Verifier '{selected}' expects {ref_type}/'{dependency}', but that instance has "
                    f"{', '.join(sorted(dependency_groups)) or 'no server type'}."
                )
            previous_group = included_groups.get(dependency)
            if previous_group is not None and previous_group != ref_type:
                raise ConfigError(
                    f"Verifier '{selected}' requires server instance '{dependency}' as both {previous_group} "
                    f"and {ref_type}; replay requires a distinct runtime name for each server."
                )
            if previous_group is None:
                included_groups[dependency] = ref_type
                pending.append(dependency)

    filtered: dict[str, Any] = {}
    for key, value in config.items():
        # Preserve command/runtime settings while excluding every unselected
        # deployable block (especially the agent and policy model).
        instance = str(key)
        required_group = included_groups.get(instance)
        if required_group is not None:
            assert isinstance(value, (Mapping, DictConfig))  # established while traversing dependencies
            # Put the one required group first because RunHelper treats each
            # top-level key as exactly one server instance and dispatches from
            # its first nested group. Preserve non-server settings, but remove
            # every co-located deployable group that replay did not require.
            implementation = included_implementations[instance]
            filtered_block = {
                required_group: {
                    implementation: deepcopy(value[required_group][implementation]),
                }
            }
            filtered_block.update(
                {
                    str(block_key): deepcopy(block_value)
                    for block_key, block_value in value.items()
                    if str(block_key) not in _SERVER_GROUPS
                }
            )
            filtered[instance] = filtered_block
        elif not _server_groups(value):
            filtered[str(key)] = deepcopy(value)
    return OmegaConf.create(filtered), tuple(sorted(included_groups))


def _validate_replay_verifier_contract(
    config: Mapping[str, Any] | DictConfig,
    *,
    selected_resource: str,
    manifest: EnvironmentManifest,
) -> None:
    """Require the projected root service to expose the verification protocol."""

    block = config.get(selected_resource)
    implementations = block.get("resources_servers") if isinstance(block, (Mapping, DictConfig)) else None
    if not isinstance(implementations, (Mapping, DictConfig)) or tuple(implementations) != (
        manifest.resources_server,
    ):
        raise ConfigError(
            f"Replay must retain exactly one resources server implementation, {manifest.resources_server!r}."
        )
    verifier = implementations[manifest.resources_server]
    provides = verifier.get("provides") if isinstance(verifier, (Mapping, DictConfig)) else None
    if (
        not isinstance(provides, Sequence)
        or isinstance(provides, (str, bytes, bytearray))
        or "verification" not in provides
    ):
        raise ConfigError(
            f"Replay resources server '{manifest.resources_server}' must declare provides: [verification]."
        )


@contextmanager
def _temporary_global_config(config: DictConfig):
    """Make the filtered config visible to RunHelper and ServerClient for one replay."""

    original = global_config_module._GLOBAL_CONFIG_DICT
    global_config_module._GLOBAL_CONFIG_DICT = config
    try:
        yield
    finally:
        global_config_module._GLOBAL_CONFIG_DICT = original


def replay_environment_rollouts(
    global_config: DictConfig,
    *,
    run_helper_factory: Callable[[], Any] | None = None,
    reverification_helper_factory: Callable[[], Any] | None = None,
) -> ReplayResult:
    """Start verifier-side services and delegate replay to the existing helper."""

    from nemo_gym.environment_execution import resolve_manifest_execution_binding
    from nemo_gym.environment_validation import validate_execution_contracts, validate_manifest_launch_sources
    from nemo_gym.rollout_reverification import (
        RolloutReverificationConfig,
        RolloutReverificationHelper,
        reverification_fingerprint,
    )

    config = EnvironmentReplayConfig.model_validate(global_config)
    preflight = resolve_manifest_execution_binding(global_config)
    if preflight is None:  # EnvironmentReplayConfig requires manifest_path.
        raise ConfigError("Replay requires an explicit environment manifest binding.")
    manifest = preflight.manifest
    paths = infer_replay_paths(config.replay_rollouts_path, config.output_jsonl_fpath)
    if paths.bundle is None:
        if not config.force:
            raise ConfigError(
                "Replay requires a self-describing .bundle.json capture. Legacy JSONL pairs can only be replayed "
                "with --force because their environment and verifier provenance cannot be checked."
            )
        compatibility = ("legacy capture provenance unavailable (--force)",)
    else:
        bundle, _artifacts = load_trajectory_bundle(paths.bundle)
        compatibility = validate_verifier_compatibility(
            bundle,
            manifest,
            allow_verifier_change=config.force,
        )
    selected_resource = _selected_resource_instance(global_config, manifest)
    filtered, started_components = verifier_only_config(global_config, manifest)
    _validate_replay_verifier_contract(
        filtered,
        selected_resource=selected_resource,
        manifest=manifest,
    )
    validate_manifest_launch_sources(filtered)
    validate_execution_contracts(filtered, None, profile=preflight.profile)
    reverify_config = RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(paths.materialized_inputs),
        rollouts_jsonl_fpath=str(paths.rollouts),
        failure_rollouts_jsonl_fpath=str(paths.failures) if paths.failures is not None else None,
        output_jsonl_fpath=str(paths.output),
        force=config.force,
        limit=config.limit,
        num_samples_in_parallel=config.num_samples_in_parallel,
        upload_rollouts_to_wandb=False,
        failure_trajectories=config.failure_trajectories,
        trajectory_identity_fields=(
            bundle.trajectory_identity_fields if paths.bundle is not None else DEFAULT_TRAJECTORY_IDENTITY_FIELDS
        ),
        verifier_fingerprint=reverification_fingerprint(
            filtered,
            resources_server_name=selected_resource,
        ),
    )

    if run_helper_factory is None:
        from nemo_gym.cli.env import RunHelper

        run_helper_factory = RunHelper
    reverification_helper_factory = reverification_helper_factory or RolloutReverificationHelper

    print(
        "Replay will start verifier-side components only "
        f"({', '.join(started_components)}); no agent or policy rollout will run.\n"
        f"Captured rollouts: {paths.rollouts}\n"
        f"Materialized inputs: {paths.materialized_inputs}\n"
        f"Failure trajectories: {config.failure_trajectories.value}"
        f"{f' from {paths.failures}' if paths.failures is not None else ' (none captured)'}\n"
        f"Capture compatibility: {'; '.join(compatibility)}\n"
        f"Replay output: {paths.output}"
    )
    runner = run_helper_factory()
    started = False
    with _temporary_global_config(filtered):
        try:
            runner.start(None, preflight_mode="launch-sources-only")
            started = True
            rows = asyncio.run(
                reverification_helper_factory().run_from_config(
                    reverify_config,
                    resources_server_name=selected_resource,
                )
            )
        finally:
            if started:
                runner.shutdown()

    # The shared helper intentionally prefixes unsafe forced replay output. It
    # does not return the selected path, so report the file it actually created
    # without reimplementing its reverify-mode guard.
    unsafe_output = paths.output.with_name("unsafe_" + paths.output.name)
    if not paths.output.exists() and unsafe_output.exists():
        paths = replace(paths, output=unsafe_output)
    return ReplayResult(paths=paths, started_components=started_components, rows=len(rows))


__all__ = [
    "EnvironmentReplayConfig",
    "ReplayPaths",
    "ReplayResult",
    "infer_replay_paths",
    "replay_environment_rollouts",
    "verifier_only_config",
]
