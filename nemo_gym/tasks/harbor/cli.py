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
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nemo_gym import component_search_roots
from nemo_gym.path_utils import failures_path_for
from nemo_gym.tasks.harbor.hub import HubRef, datasets_dir, fetch_ref, is_hub_ref
from nemo_gym.tasks.harbor.materialize import run_config, write_rows
from nemo_gym.tasks.harbor.task import HarborTask, HarborTaskError, discover_tasks


logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("results/harbor")
ENVIRONMENT_SERVER_CONFIG = "environment_servers/single_agent_turn/configs/single_agent_turn.yaml"
RESOURCES_SERVER_CONFIG = "resources_servers/harbor/configs/harbor.yaml"
# The oracle needs no model, but a run needs a policy model block; this one is never called.
PLACEHOLDER_MODEL_CONFIG = "responses_api_models/openai_model/configs/openai_model.yaml"
PLACEHOLDER_MODEL_OVERRIDES = [
    "+policy_base_url=http://oracle.invalid/v1",
    "+policy_api_key=unused",
    "+policy_model_name=oracle",
]
ORACLE_AGENT = "oracle_agent"
# Harnesses that call the Gym model server from inside the task sandbox, where 127.0.0.1 is the container
# itself. The model server must advertise the node's IP (`use_absolute_ip`) for them. An agent whose config
# declares `SANDBOX_MODEL_BASE_URL_KEY` runs its harness in the sandbox too; the set covers harnesses whose
# config has no such key.
SANDBOX_MODEL_BASE_URL_KEY = "sandbox_model_base_url"
IN_SANDBOX_AGENTS = frozenset(
    {
        "hermes_agent",
        "hermes_sandboxed_agent",
        "miniswe_sandboxed_agent",
        "opencode_sandboxed_agent",
        "pi_sandboxed_agent",
        "terminus_2_sandboxed_agent",
    }
)
USE_ABSOLUTE_IP_KEY = "use_absolute_ip"
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
    # Episode sessions are process-local, so the harness must run as one worker.
    "terminus_2_sandboxed_agent": {"num_workers": 1},
}


@dataclass(frozen=True)
class PreparedTaskset:
    taskset: str
    folder: Path
    tasks: list[HarborTask]
    rows_path: Path
    output_dir: Path
    skipped: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSelection:
    config_path: Path
    instance_name: str
    impl_name: str
    # The harness calls the model server from inside the task sandbox (see ``IN_SANDBOX_AGENTS``).
    runs_in_sandbox: bool = False


def prepare_target(
    target: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    refresh_registry: bool = False,
) -> PreparedTaskset:
    """Fetch (for hub references) and load the tasks, then write the rows file.

    A task folder that does not load is skipped with a warning (and listed in
    ``PreparedTaskset.skipped``) so the rest of the dataset still prepares.
    """
    if is_hub_ref(target):
        taskset = HubRef.parse(target).name
        folder = fetch_ref(target, refresh_registry=refresh_registry)
        print(f"Fetched {target} into {folder}")
    else:
        folder = Path(target).expanduser().resolve()
        taskset = folder.name
    errors: dict[str, HarborTaskError] = {}
    tasks = discover_tasks(folder, skipped=errors)
    for task_id, error in errors.items():
        logger.warning("Skipping task %s: %s", task_id, error)
    output_dir = Path(output_root) / taskset
    rows_path = output_dir / "tasks.jsonl"
    write_rows(tasks, taskset, rows_path)
    skipped_note = f", skipped {len(errors)}" if errors else ""
    print(f"Materialized {len(tasks)} task(s) from {folder} into {rows_path}{skipped_note}")
    return PreparedTaskset(
        taskset=taskset,
        folder=folder,
        tasks=tasks,
        rows_path=rows_path,
        output_dir=output_dir,
        skipped={task_id: str(error) for task_id, error in errors.items()},
    )


def runs_in_sandbox(impl_name: str, impl_config: Any) -> bool:
    """Whether the harness behind ``impl_name`` calls the model server from inside the task sandbox."""
    declares_sandbox_url = isinstance(impl_config, dict) and SANDBOX_MODEL_BASE_URL_KEY in impl_config
    return declares_sandbox_url or impl_name in IN_SANDBOX_AGENTS


def resolve_agent(agent: str) -> AgentSelection:
    """Map ``--agent NAME[/FLAVOR]`` to ``responses_api_agents/<NAME>/configs/<FLAVOR>.yaml``.

    ``NAME`` may omit a trailing ``_agent`` or ``_sandboxed_agent`` (``simple`` selects ``simple_agent``).
    A short name that matches two harnesses, such as ``hermes`` (``hermes_agent`` and
    ``hermes_sandboxed_agent``) or ``terminus_2``, is refused with both full names, so a run never
    lands on the wrong harness by default.
    """
    name, _, flavor = agent.partition("/")
    candidates = [name] if name.endswith("_agent") else [name, f"{name}_agent", f"{name}_sandboxed_agent"]
    matches: list[Path] = []
    for root in component_search_roots():
        for folder_name in candidates:
            path = root / "responses_api_agents" / folder_name / "configs" / f"{flavor or folder_name}.yaml"
            if path.is_file() and path.resolve() not in matches:
                matches.append(path.resolve())
    if not matches:
        raise ValueError(f"No agent config found for `--agent {agent}` (looked for {' or '.join(candidates)})")
    folders = {path.parents[1].name for path in matches}
    if len(folders) > 1:
        options = " or ".join(f"`--agent {folder}`" for folder in sorted(folders))
        raise ValueError(f"`--agent {agent}` is ambiguous; use {options}")
    path = matches[0]
    config = yaml.safe_load(path.read_text()) or {}
    instances = [key for key, value in config.items() if isinstance(value, dict)]
    if len(instances) != 1 or "responses_api_agents" not in config[instances[0]]:
        raise ValueError(f"{path} must define exactly one responses_api_agents instance")
    impls_config = config[instances[0]]["responses_api_agents"]
    impls = list(impls_config)
    if len(impls) != 1:
        raise ValueError(f"{path} must define exactly one agent implementation")
    return AgentSelection(
        config_path=path,
        instance_name=instances[0],
        impl_name=impls[0],
        runs_in_sandbox=runs_in_sandbox(impls[0], impls_config[impls[0]]),
    )


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
    if agent.runs_in_sandbox and not _has_override(tokens, USE_ABSOLUTE_IP_KEY):
        tokens.append(f"+{USE_ABSOLUTE_IP_KEY}=true")
        logger.info(
            "Setting %s=true: the %s harness calls the model server from inside the sandbox, "
            "where a loopback address is the container itself",
            USE_ABSOLUTE_IP_KEY,
            agent.impl_name,
        )
    return config_path, tokens


def run_target(args: argparse.Namespace, overrides: list[str]) -> None:
    """Entry point for ``gym eval run <target> ...``."""
    from nemo_gym.cli.main import _merge_config_paths, dispatch

    if getattr(args, "no_serve", False):
        raise ValueError("A Harbor target starts its own servers; drop --no-serve")
    agent_name = getattr(args, "agent", None)
    if not agent_name:
        raise ValueError("A Harbor target needs `--agent <harness>` (for example `--agent hermes_agent`)")
    prepared = prepare_target(args.target)
    agent = resolve_agent(agent_name)
    config_path, tokens = build_run(prepared, agent, sandbox=getattr(args, "sandbox", None), overrides=overrides)
    print(f"Run config written to {config_path} (datasets folder: {datasets_dir()})")
    dispatch("nemo_gym.cli.eval:e2e_rollout_collection", _merge_config_paths(tokens))


def validate_target(args: argparse.Namespace, overrides: list[str]) -> None:
    """Entry point for ``gym dataset validate <target>``: run every task's reference solution and score it."""
    from nemo_gym.cli.main import _merge_config_paths, dispatch

    prepared = prepare_target(args.target)
    agent = resolve_agent(ORACLE_AGENT)
    unvalidated = [task.task_id for task in prepared.tasks if not task.has_solution]
    if len(unvalidated) == len(prepared.tasks):
        print(f"No task in {prepared.folder} has solution/solve.sh; nothing to validate.")
        return
    tokens = list(overrides)
    if not any(token.startswith("+config_paths=") and "responses_api_models/" in token for token in tokens):
        tokens += [f"+config_paths=[{_config_path(PLACEHOLDER_MODEL_CONFIG)}]", *PLACEHOLDER_MODEL_OVERRIDES]
    config_path, tokens = build_run(prepared, agent, sandbox=getattr(args, "sandbox", None), overrides=tokens)
    print(
        f"Run config written to {config_path}; validating {len(prepared.tasks) - len(unvalidated)} task(s) with the oracle"
    )
    dispatch("nemo_gym.cli.eval:e2e_rollout_collection", _merge_config_paths(tokens))
    rollouts = prepared.output_dir / "rollouts.jsonl"
    report = summarize_validation(rollouts, unvalidated)
    print(report.text)
    if not report.ok:
        sys.exit(1)


@dataclass(frozen=True)
class ValidationReport:
    text: str
    ok: bool


def _dig(mapping: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def _task_name(row: dict[str, Any]) -> str:
    identity = row.get("_ng_task_id") or row.get("task_id") or {}
    return str(identity.get("task_id", "?")) if isinstance(identity, dict) else str(identity)


def summarize_validation(rollouts: Path, unvalidated: list[str]) -> ValidationReport:
    """One line per task: the oracle's reward, or why there is none.

    Rollout collection stores an environment server's result as returned, so a row is the verify
    response itself (``reward``, ``mask_sample``, ``failure_kind``, ``response``) plus ``_ng_task_id``.
    Episodes that failed before verification are in the failures sidecar next to the rollouts file.
    """
    lines = ["task_id\tstatus\treward"]
    ok = True
    if rollouts.is_file():
        for raw in rollouts.read_text().splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            task_id = _task_name(row)
            oracle = _dig(row, "response", "metadata", "oracle")
            reward = row.get("reward")
            if row.get("mask_sample"):
                ok = False
                lines.append(f"{task_id}\tmasked ({row.get('failure_kind')})\t-")
            elif oracle == "unvalidated":
                lines.append(f"{task_id}\tunvalidated (no solution/)\t-")
            else:
                if reward is None or reward < 1.0:
                    ok = False
                kind = row.get("failure_kind")
                note = f" ({kind})" if kind else ""
                lines.append(f"{task_id}\toracle {oracle or 'ran'}{note}\t{reward}")
        failures = failures_path_for(rollouts)
        if failures.is_file():
            for raw in failures.read_text().splitlines():
                if not raw.strip():
                    continue
                row = json.loads(raw)
                ok = False
                message = row.get("_ng_failure_message") or row.get("_ng_failure_class") or "failed"
                lines.append(f"{_task_name(row)}\tfailed: {str(message)[:120]}\t-")
    else:
        ok = False
        lines.append(f"-\tno rollouts written at {rollouts}\t-")
    for task_id in unvalidated:
        if not any(line.startswith(f"{task_id}\t") for line in lines):
            lines.append(f"{task_id}\tunvalidated (no solution/)\t-")
    return ValidationReport(text="\n".join(lines), ok=ok)
