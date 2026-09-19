# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose an environment definition with selected sandbox and episode runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from nemo_gym.config_types import ConfigError
from nemo_gym.environment.authoring import LoadedEnvironment, materialize_tasks, tasks_to_jsonl


ENVIRONMENT_ADAPTER_NAME = "environment_adapter_resources_server"


@dataclass(frozen=True)
class SandboxRuntime:
    """A selected sandbox backend and its prepared runtime image."""

    config_paths: tuple[Path, ...]
    runtime_image: str
    sandbox_provider_ref: str
    sandbox_config: dict[str, object]


@dataclass(frozen=True)
class AgentRoleBinding:
    """One agent role required by an episode protocol."""

    resources_server_name: str


@dataclass(frozen=True)
class EpisodeProtocolRuntime:
    """Environment-server composition selected for an episode protocol."""

    config_paths: tuple[Path, ...]
    environment_server_name: str
    environment_server_config: dict[str, object]
    agent_roles: dict[str, AgentRoleBinding]


@dataclass(frozen=True)
class EnvironmentRunArtifacts:
    """Temporary config and rollout input generated for one environment run."""

    config_path: Path
    input_jsonl_path: Path
    config_paths: tuple[Path, ...]


def compose_environment_run(
    loaded: LoadedEnvironment,
    output_dir: str | Path,
    *,
    sandbox: SandboxRuntime,
    episode_protocol: EpisodeProtocolRuntime,
    adapter_config_path: Path,
    taskset: str | None = None,
) -> EnvironmentRunArtifacts:
    """Materialize tasks and compose selected sandbox and episode-protocol runtimes."""

    if loaded.definition.runtime.mcp_servers:
        raise ConfigError(
            "`gym eval run --environment` does not yet support runtime.mcp_servers. "
            "Remove the declaration or run this environment through a custom server composition."
        )

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    tasks = materialize_tasks(loaded, taskset=taskset)
    input_jsonl_path = output_path / "tasks.jsonl"
    input_jsonl_path.write_text(tasks_to_jsonl(tasks), encoding="utf-8")
    config_path = output_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _run_config(
                loaded,
                input_jsonl_path=input_jsonl_path,
                sandbox=sandbox,
                episode_protocol=episode_protocol,
                taskset_names=tuple(dict.fromkeys(task.materialized.task_id.taskset for task in tasks)),
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return EnvironmentRunArtifacts(
        config_path=config_path,
        input_jsonl_path=input_jsonl_path,
        config_paths=(
            *sandbox.config_paths,
            adapter_config_path,
            *episode_protocol.config_paths,
            config_path,
        ),
    )


def _run_config(
    loaded: LoadedEnvironment,
    *,
    input_jsonl_path: Path,
    sandbox: SandboxRuntime,
    episode_protocol: EpisodeProtocolRuntime,
    taskset_names: tuple[str, ...],
) -> dict[str, object]:
    environment = loaded.definition
    config: dict[str, object] = {
        "rollout_input": {
            "type": "materialized_tasks",
            "path": str(input_jsonl_path),
        },
        "environment_routing_mode": "taskset",
        "tasksets": {name: {"task_input_contract": environment.episode_protocol} for name in taskset_names},
        "environment_server_routes": {name: episode_protocol.environment_server_name for name in taskset_names},
        "agent_bindings": {
            name: {
                "resources_server": {
                    "type": "resources_servers",
                    "name": binding.resources_server_name,
                }
            }
            for name, binding in episode_protocol.agent_roles.items()
        },
        ENVIRONMENT_ADAPTER_NAME: {
            "resources_servers": {
                "environment_adapter": {
                    "environment_root": str(loaded.root),
                    "runtime_image": sandbox.runtime_image,
                    "sandbox_provider": sandbox.sandbox_provider_ref,
                    "sandbox_config": sandbox.sandbox_config,
                    "trusted_environment_code": True,
                }
            }
        },
    }
    config.update(episode_protocol.environment_server_config)
    return config


__all__ = [
    "ENVIRONMENT_ADAPTER_NAME",
    "AgentRoleBinding",
    "EpisodeProtocolRuntime",
    "EnvironmentRunArtifacts",
    "SandboxRuntime",
    "compose_environment_run",
]
