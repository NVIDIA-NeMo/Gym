# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gym eval run <folder | harbor:dataset> --agent <name>``: prepare a taskset and run it.

Preparation writes, under ``results/harbor/<taskset>/``:

- ``tasks.jsonl``: one materialized row per task;
- ``run_config.yaml``: the resources server, agent and environment server blocks.

Then the ordinary end-to-end rollout collection runs with that config plus the
selected agent, environment server and model configs.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nemo_gym import component_search_roots
from nemo_gym.tasks.harbor.hub import HubRef, datasets_dir, fetch_ref, is_hub_ref
from nemo_gym.tasks.harbor.materialize import run_config, write_rows
from nemo_gym.tasks.harbor.task import HarborTask, discover_tasks


logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("results/harbor")
ENVIRONMENT_SERVER_CONFIG = "environment_servers/single_agent_turn/configs/single_agent_turn.yaml"
RESOURCES_SERVER_CONFIG = "resources_servers/harbor/configs/harbor.yaml"
SANDBOX_PROVIDER_CONFIG = "nemo_gym/sandbox/providers/{provider}/configs/{provider}.yaml"
# Harness-specific settings a Harbor run needs; keyed by the agent implementation folder.
AGENT_OVERRIDES: dict[str, dict[str, Any]] = {
    # Hermes borrows the task sandbox and drives it through its terminal toolset. vLLM-only
    # `chat_template_kwargs` are off so OpenAI-compatible model servers accept its requests;
    # override `chat_template_kwargs_enabled` in the run config when serving with vLLM.
    "hermes_agent": {
        "sandbox_provider": None,
        "enabled_toolsets": ["terminal"],
        "chat_template_kwargs_enabled": False,
    },
}


@dataclass(frozen=True)
class PreparedTaskset:
    taskset: str
    folder: Path
    tasks: list[HarborTask]
    rows_path: Path
    output_dir: Path


@dataclass(frozen=True)
class AgentSelection:
    config_path: Path
    instance_name: str
    impl_name: str


def prepare_target(
    target: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    refresh_registry: bool = False,
) -> PreparedTaskset:
    """Fetch (for hub references) and load the tasks, then write the rows file."""
    if is_hub_ref(target):
        taskset = HubRef.parse(target).name
        folder = fetch_ref(target, refresh_registry=refresh_registry)
        print(f"Fetched {target} into {folder}")
    else:
        folder = Path(target).expanduser().resolve()
        taskset = folder.name
    tasks = discover_tasks(folder)
    output_dir = Path(output_root) / taskset
    rows_path = output_dir / "tasks.jsonl"
    write_rows(tasks, taskset, rows_path)
    print(f"Materialized {len(tasks)} task(s) from {folder} into {rows_path}")
    return PreparedTaskset(taskset=taskset, folder=folder, tasks=tasks, rows_path=rows_path, output_dir=output_dir)


def resolve_agent(agent: str) -> AgentSelection:
    """Map ``--agent NAME[/FLAVOR]`` to ``responses_api_agents/<NAME>/configs/<FLAVOR>.yaml``.

    ``NAME`` may omit a trailing ``_agent`` (``hermes`` selects ``hermes_agent``).
    """
    name, _, flavor = agent.partition("/")
    candidates = [name] if name.endswith("_agent") else [name, f"{name}_agent"]
    for root in component_search_roots():
        for folder_name in candidates:
            path = root / "responses_api_agents" / folder_name / "configs" / f"{flavor or folder_name}.yaml"
            if path.is_file():
                config = yaml.safe_load(path.read_text()) or {}
                instances = [key for key, value in config.items() if isinstance(value, dict)]
                if len(instances) != 1 or "responses_api_agents" not in config[instances[0]]:
                    raise ValueError(f"{path} must define exactly one responses_api_agents instance")
                impls = list(config[instances[0]]["responses_api_agents"])
                if len(impls) != 1:
                    raise ValueError(f"{path} must define exactly one agent implementation")
                return AgentSelection(config_path=path.resolve(), instance_name=instances[0], impl_name=impls[0])
    raise ValueError(f"No agent config found for `--agent {agent}` (looked for {' or '.join(candidates)})")


def resolve_sandbox_config(provider: str) -> Path:
    relative = SANDBOX_PROVIDER_CONFIG.format(provider=provider)
    for root in component_search_roots():
        path = root / relative
        if path.is_file():
            return path.resolve()
    raise ValueError(f"No sandbox provider config found for `--sandbox {provider}` (expected {relative})")


def _config_path(relative: str) -> Path:
    for root in component_search_roots():
        path = root / relative
        if path.is_file():
            return path.resolve()
    raise ValueError(f"Missing built-in config {relative}")


def _has_override(overrides: list[str], key: str) -> bool:
    return any(token.lstrip("+").split("=", 1)[0] == key for token in overrides)


def build_run(
    prepared: PreparedTaskset,
    agent: AgentSelection,
    *,
    sandbox: str | None,
    overrides: list[str],
) -> tuple[Path, list[str]]:
    """Write ``run_config.yaml`` and return it with the Hydra overrides for collection."""
    config = run_config(
        taskset=prepared.taskset,
        folder=prepared.folder,
        tasks=prepared.tasks,
        rows_path=prepared.rows_path,
        agent_instance_name=agent.instance_name,
        agent_impl=agent.impl_name,
        agent_overrides=AGENT_OVERRIDES.get(agent.impl_name),
    )
    config_path = prepared.output_dir / "run_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    config_paths = [
        config_path.resolve(),
        _config_path(RESOURCES_SERVER_CONFIG),
        agent.config_path,
        _config_path(ENVIRONMENT_SERVER_CONFIG),
    ]
    if sandbox is not None:
        config_paths.append(resolve_sandbox_config(sandbox))
    tokens = [token for token in overrides if not token.startswith("+agent_name=")]
    tokens.append(f"+config_paths=[{','.join(str(path) for path in config_paths)}]")
    if not _has_override(tokens, "split"):
        tokens.append("+split=validation")
    if not _has_override(tokens, "output_jsonl_fpath"):
        tokens.append(f"+output_jsonl_fpath={prepared.output_dir / 'rollouts.jsonl'}")
    return config_path, tokens


def run_target(args: argparse.Namespace, overrides: list[str]) -> None:
    """Entry point for ``gym eval run <target> ...``."""
    from nemo_gym.cli.main import _merge_config_paths, dispatch

    if getattr(args, "no_serve", False):
        raise ValueError("A Harbor target starts its own servers; drop --no-serve")
    agent_name = getattr(args, "agent", None)
    if not agent_name:
        raise ValueError("A Harbor target needs `--agent <harness>` (for example `--agent hermes`)")
    prepared = prepare_target(args.target)
    agent = resolve_agent(agent_name)
    config_path, tokens = build_run(prepared, agent, sandbox=getattr(args, "sandbox", None), overrides=overrides)
    print(f"Run config written to {config_path} (datasets folder: {datasets_dir()})")
    dispatch("nemo_gym.cli.eval:e2e_rollout_collection", _merge_config_paths(tokens))
