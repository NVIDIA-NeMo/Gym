# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gym eval run <folder | harbor:dataset> --agent <name>``: prepare a taskset and run it.

Preparation writes, under ``results/harbor/<taskset>/``:

- ``tasks.jsonl``: one materialized row per loadable task;
- ``skipped.json``: omitted task IDs and reasons (empty when all tasks load);
- ``run_config.yaml``: the agent and environment server blocks.

Then the ordinary end-to-end rollout collection runs with that config plus the
selected agent, environment server and model configs.
"""

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from nemo_gym import component_search_roots
from nemo_gym.cli.utils import exit_cleanly_on_config_error
from nemo_gym.config_types import ConfigError
from nemo_gym.path_utils import failures_path_for
from nemo_gym.tasks.harbor.digest import DIGEST_KEY
from nemo_gym.tasks.harbor.hub import HubError, datasets_dir, fetch_ref, is_hub_ref
from nemo_gym.tasks.harbor.materialize import run_config, write_rows
from nemo_gym.tasks.harbor.task import HarborTask, HarborTaskError, discover_tasks


logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("results/harbor")
ENVIRONMENT_SERVER_CONFIG = "environment_servers/harbor/configs/harbor.yaml"
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
        folder = fetch_ref(target, refresh_registry=refresh_registry)
        print(f"Fetched {target} into {folder}")
    else:
        folder = Path(target).expanduser().resolve()
    taskset = folder.name
    output_dir = Path(output_root) / taskset
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: dict[str, HarborTaskError] = {}
    try:
        tasks = discover_tasks(folder, skipped=errors)
    finally:
        # Keep the omitted IDs and reasons even when no task in the dataset loads.
        (output_dir / "skipped.json").write_text(
            json.dumps({task_id: str(error) for task_id, error in errors.items()}, indent=2) + "\n"
        )
    for task_id, error in errors.items():
        logger.warning("Skipping task %s: %s", task_id, error)
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
    # Defaults belong below explicit config files and command-line overrides.
    config["output_jsonl_fpath"] = str(prepared.output_dir / "rollouts.jsonl")
    if agent.runs_in_sandbox:
        config[USE_ABSOLUTE_IP_KEY] = True
    config_path = prepared.output_dir / "run_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    config_paths = [config_path.resolve(), agent.config_path, _config_path(ENVIRONMENT_SERVER_CONFIG)]
    if sandbox is not None:
        config_paths.append(resolve_sandbox_config(sandbox))
    # Later config files win: explicit user overlays must follow component/provider defaults.
    tokens = [f"+config_paths=[{','.join(json.dumps(str(path), ensure_ascii=False) for path in config_paths)}]"]
    tokens += [token for token in overrides if not token.startswith("+agent_name=")]
    if not _has_override(tokens, "split"):
        tokens.append("+split=validation")
    return config_path, tokens


def run_target(args: argparse.Namespace, overrides: list[str]) -> None:
    """Entry point for ``gym eval run <target> ...``."""
    from nemo_gym.cli.main import _merge_config_paths, dispatch

    if getattr(args, "no_serve", False):
        args._parser.error("A Harbor target starts its own servers; drop --no-serve")
    agent_name = getattr(args, "agent", None)
    if not agent_name:
        args._parser.error("A Harbor target needs `--agent <harness>` (for example `--agent hermes_agent`)")
    sandbox = getattr(args, "sandbox", None)
    # Resolve selectors before preparation can download a registry dataset.
    try:
        agent = resolve_agent(agent_name)
        if sandbox is not None:
            resolve_sandbox_config(sandbox)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        args._parser.error(str(exc))
    try:
        prepared = prepare_target(args.target)
    except (HarborTaskError, HubError, OSError) as exc:
        args._parser.error(str(exc))
    if prepared.skipped and not getattr(args, "skip_unloadable", False):
        args._parser.error(
            f"{len(prepared.skipped)} task(s) could not load; see {prepared.output_dir / 'skipped.json'}. "
            "Use --skip-unloadable to explicitly run only the loadable subset."
        )
    try:
        config_path, tokens = build_run(prepared, agent, sandbox=sandbox, overrides=overrides)
    except (HarborTaskError, OSError) as exc:
        args._parser.error(str(exc))
    print(f"Run config written to {config_path} (datasets folder: {datasets_dir()})")
    dispatch("nemo_gym.cli.eval:e2e_rollout_collection", _merge_config_paths(tokens))


@exit_cleanly_on_config_error
def validate_target(args: argparse.Namespace, overrides: list[str]) -> None:
    """Entry point for ``gym dataset validate <target>``: run every task's reference solution and score it."""
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationResolutionError

    from nemo_gym.cli.main import _merge_config_paths, dispatch
    from nemo_gym.global_config import get_global_config_dict, translate_interpolation_error
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper

    # Evaluation and reference validation must never share inputs, configs, or cached results.
    try:
        agent = resolve_agent(ORACLE_AGENT)
        sandbox = getattr(args, "sandbox", None)
        if sandbox is not None:
            resolve_sandbox_config(sandbox)
        prepared = prepare_target(args.target, output_root=DEFAULT_OUTPUT_ROOT / "validation")
    except (HarborTaskError, HubError, ValueError, OSError, yaml.YAMLError) as exc:
        args._parser.error(str(exc))
    unvalidated = [task.task_id for task in prepared.tasks if not task.has_solution]
    prepared = replace(prepared, tasks=[task for task in prepared.tasks if task.has_solution])
    write_rows(prepared.tasks, prepared.taskset, prepared.rows_path)
    rollouts = prepared.output_dir / "rollouts.jsonl"
    expected = []
    if prepared.tasks:
        # Oracle runs reference scripts directly; it needs no placeholder model server.
        config_path, tokens = build_run(prepared, agent, sandbox=sandbox, overrides=list(overrides))
        tokens = _merge_config_paths(tokens)
        # Resolve through the same parser and cache used by dispatch, including file overlays.
        sys.argv = [sys.argv[0], *tokens]
        config = get_global_config_dict()
        try:
            resolved = OmegaConf.to_container(config, resolve=True)
        except InterpolationResolutionError as exc:
            raise translate_interpolation_error(exc) from exc
        collection = RolloutCollectionConfig.model_validate({**resolved, "input_jsonl_fpath": str(prepared.rows_path)})
        expected = RolloutCollectionHelper()._preprocess_rows_from_config(collection)
        rollouts = Path(collection.output_jsonl_fpath)
        if collection.resume_from_cache and rollouts.is_file() and collection.materialized_jsonl_fpath.is_file():
            # The collector resumes its saved inputs, not the just-prepared task rows. Reject a stale
            # selection or digest before startup so an old reward cannot validate a changed solution.
            cached = _validation_rows(collection.materialized_jsonl_fpath)
            current_digests = {
                _validation_key(row): _dig(row, "task_input", "task_data", DIGEST_KEY) for row in expected
            }
            cached_digests = {_validation_key(row): _dig(row, "task_input", "task_data", DIGEST_KEY) for row in cached}
            if cached_digests != current_digests:
                raise ConfigError(
                    f"Cannot resume Harbor validation from {collection.materialized_jsonl_fpath}: "
                    "the selected tasks or task digests have changed. Use +resume_from_cache=false "
                    "to validate the current tasks, or choose a new +output_jsonl_fpath to keep the old results."
                )
        print(f"Run config written to {config_path}; validating {len(expected)} oracle rollout(s)")
        dispatch("nemo_gym.cli.eval:e2e_rollout_collection", tokens)
    report = summarize_validation(rollouts, unvalidated, expected=expected, skipped=prepared.skipped)
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


def _validation_key(row: dict[str, object]) -> tuple[str, str, object, object]:
    identity = row.get("_ng_task_id") or row.get("task_id") or {}
    taskset = str(identity.get("taskset", "")) if isinstance(identity, dict) else ""
    return taskset, _task_name(row), row.get("_ng_task_index"), row.get("_ng_rollout_index")


def _validation_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_validation(
    rollouts: Path,
    unvalidated: list[str],
    *,
    expected: list[dict[str, object]] | None = None,
    skipped: dict[str, str] | None = None,
) -> ValidationReport:
    """Report every selected task/repeat, reconciling results with failed attempts.

    The collector stores verify responses directly and keeps historical failed attempts in a
    sidecar. A result supersedes failures for the same task/repeat, but never another repeat.
    ``expected`` contains the collector's selected, expanded input rows (after limit/repeats).
    """
    lines = ["task_id\tstatus\treward"]
    ok = True
    selected = {_validation_key(row): row for row in expected} if expected is not None else None
    results = {_validation_key(row): row for row in _validation_rows(rollouts)}
    failures = {_validation_key(row): row for row in _validation_rows(failures_path_for(rollouts))}
    keys = selected if selected is not None else dict.fromkeys([*results, *failures])
    for key in keys:
        _, task_id, _, repeat = key
        label = f"{task_id} [repeat {repeat}]" if repeat is not None else task_id
        if task_id in unvalidated:
            continue
        row = results.get(key)
        if row is None:
            ok = False
            failure = failures.get(key)
            if failure is None:
                lines.append(f"{label}\tmissing result\t-")
            else:
                message = failure.get("_ng_failure_message") or failure.get("_ng_failure_class") or "failed"
                lines.append(f"{label}\tfailed: {str(message)[:120]}\t-")
            continue
        oracle = _dig(row, "response", "metadata", "oracle")
        reward = row.get("reward")
        if row.get("mask_sample"):
            ok = False
            lines.append(f"{label}\tmasked ({row.get('failure_kind')})\t-")
        elif oracle == "unvalidated":
            # A selected reference solution disappearing is not a successful validation.
            if selected is not None:
                ok = False
            lines.append(f"{label}\tunvalidated (no solution/)\t-")
        else:
            if oracle == "failed" or not isinstance(reward, (int, float)) or not math.isfinite(reward) or reward < 1.0:
                ok = False
            kind = row.get("failure_kind")
            note = f" ({kind})" if kind else ""
            lines.append(f"{label}\toracle {oracle or 'ran'}{note}\t{reward}")
    if selected is None and not rollouts.is_file():
        ok = False
        lines.append(f"-\tno rollouts written at {rollouts}\t-")
    for task_id in unvalidated:
        lines.append(f"{task_id}\tunvalidated (no solution/)\t-")
    for task_id, reason in (skipped or {}).items():
        ok = False
        lines.append(f"{task_id}\tskipped: {reason}\t-")
    return ValidationReport(text="\n".join(lines), ok=ok)
