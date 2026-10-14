# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gym eval run <folder | harbor:dataset> --agent <name>``: prepare a taskset and run it.

Preparation writes, under ``results/harbor/<taskset>/``:

- ``tasks.jsonl``: one materialized row per loadable task;
- ``skipped.json``: omitted task IDs and reasons (empty when all tasks load);
- ``run_config_<agent>.yaml``: the agent and environment server blocks.

Then the ordinary end-to-end rollout collection runs with that config plus the
selected agent, environment server and model configs.
"""

import argparse
import fnmatch
import json
import logging
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from nemo_gym import component_search_roots
from nemo_gym.cli.utils import exit_cleanly_on_config_error
from nemo_gym.config_types import ConfigError
from nemo_gym.path_utils import materialized_path_for
from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE, read_dataset_config
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.hub import HubError, datasets_dir, fetch_ref, is_hub_ref
from nemo_gym.tasks.harbor.image_configs import record_base_images, record_compose_images
from nemo_gym.tasks.harbor.materialize import run_config, write_rows
from nemo_gym.tasks.harbor.task import HarborTask, discover_tasks, is_task_folder
from nemo_gym.tasks.harbor.validation_report import (
    _task_digests,
    _validation_key,
    _validation_rows,
    summarize_validation,
)


if TYPE_CHECKING:
    from nemo_gym.rollout_collection import RolloutCollectionConfig

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("results/harbor")
ENVIRONMENT_SERVER_CONFIG = "environment_servers/harbor/configs/harbor.yaml"
ORACLE_AGENT = "harbor_oracle_agent"
# A harness that calls the Gym model server from inside the task sandbox, where 127.0.0.1 is the container
# itself, needs the model server to advertise the node's IP (`use_absolute_ip`). The harness declares that on
# the agent class its entrypoint runs (`calls_model_from_sandbox: ClassVar[bool] = True`), read from source
# like `ray_enabled`; an agent whose config declares `SANDBOX_MODEL_BASE_URL_KEY` routes its model calls
# through the sandbox too.
SANDBOX_MODEL_BASE_URL_KEY = "sandbox_model_base_url"
CALLS_MODEL_FROM_SANDBOX_KEY = "calls_model_from_sandbox"
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
    # Episode sessions are process-local, so the harness must run as one worker. Its shipped template
    # leaves `resources_server.name` unset for benchmark composition; here the environment server
    # drives it, so the generated instance declares none (and is not a composition source).
    "terminus_2_sandboxed_agent": {"num_workers": 1, "resources_server": None},
    "miniswe_sandboxed_agent": {"num_workers": 1},
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
    # The harness calls the model server from inside the task sandbox (see ``runs_in_sandbox``).
    runs_in_sandbox: bool = False


def _write_skipped(output_dir: Path, errors: dict[str, HarborTaskError]) -> None:
    (output_dir / "skipped.json").write_text(
        json.dumps({task_id: str(error) for task_id, error in errors.items()}, indent=2) + "\n"
    )


def prepare_target(
    target: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    refresh_registry: bool = False,
    exclude: list[str] | None = None,
    only: list[str] | None = None,
    rows_name: str = "tasks.jsonl",
) -> PreparedTaskset:
    """Fetch (for hub references) and load the tasks, then write the rows file.

    A task folder that does not load is skipped with a warning (and listed in
    ``PreparedTaskset.skipped``) so the rest of the dataset still prepares.

    Tasks matching an ``exclude`` glob, or not matching an ``only`` glob, are left out of the
    rows and listed on stdout. ``rows_name`` names the rows file inside the taskset's output folder:
    a run scopes it to its agent, so two agents prepared with different selections on one taskset
    keep their own rows.
    """
    if is_hub_ref(target):
        folder = fetch_ref(target, refresh_registry=refresh_registry)
        print(f"Fetched {target} into {folder}")
    else:
        folder = Path(target).expanduser().resolve()
    taskset = folder.name
    exclude = exclude or []

    def selected(task_id: str) -> bool:
        return selection_reason(task_id, exclude, only) is None

    # Dockerfile ENV, WORKDIR and USER resolve from the base image's configuration, so it is recorded
    # before the tasks load; a base image the registry cannot resolve stops the preparation here. Only
    # the selected tasks' images are recorded, so an image a left-out task uses cannot stop it. A target
    # that is itself a task folder always loads (discover_tasks raises for it); selection applies after.
    # A missing or unreadable target is reported by discover_tasks below, in its own words.
    images = (
        record_base_images(folder, select=None if is_task_folder(folder) else selected) if folder.is_dir() else None
    )
    if images is not None:
        print(f"Base image configurations recorded in {images}")
    output_dir = Path(output_root) / taskset
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: dict[str, HarborTaskError] = {}
    try:
        loaded = discover_tasks(folder, skipped=errors)
    finally:
        # Keep the omitted IDs and reasons even when no task in the dataset loads.
        _write_skipped(output_dir, errors)
    _check_dataset_task_keys(folder, loaded, errors)
    tasks, skipped = select_tasks(loaded, exclude, only)
    # A left-out task that did not load (its base image was not recorded) is left out, not reported as broken.
    for task_id in [task_id for task_id in errors if not selected(task_id)]:
        skipped.append((task_id, selection_reason(task_id, exclude, only)))
        del errors[task_id]
    _write_skipped(output_dir, errors)
    for task_id, error in errors.items():
        logger.warning("Skipping task %s: %s", task_id, error)
    for task_id, reason in sorted(skipped):
        print(f"Skipping {task_id}: {reason}")
    if not tasks:
        raise ValueError(f"No runnable task left in {folder}")
    images = record_compose_images(tasks, tasks[0].path.parent)
    if images is not None:
        print(f"Compose image configurations recorded in {images}")
    rows_path = output_dir / rows_name
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


def _check_dataset_task_keys(folder: Path, loaded: list[HarborTask], errors: dict[str, HarborTaskError]) -> None:
    """Refuse a ``[gym.tasks."<id>"]`` table that names no task folder: a misspelt id would otherwise apply to nothing.

    A target that is itself a task folder is prepared without its siblings, so the check is skipped for it.
    """
    if is_task_folder(folder):
        return
    try:
        dataset = read_dataset_config(folder)
    except ValueError:
        return  # already reported per task by the loader
    unknown = set(dataset.tasks) - {task.task_id for task in loaded} - set(errors)
    if unknown:
        raise ValueError(f"{folder / DATASET_FILE} [gym.tasks] names unknown tasks: {sorted(unknown)}")


def runs_in_sandbox(impl_name: str, impl_config: Any) -> bool:
    """Whether the harness behind ``impl_name`` calls the model server from inside the task sandbox.

    The harness says so itself: ``calls_model_from_sandbox = True`` on the agent class its entrypoint runs, read
    from the entrypoint source (no agent module is imported here), or ``sandbox_model_base_url`` in its config.
    """
    from nemo_gym.server_utils import entrypoint_class_flag

    config = impl_config if isinstance(impl_config, dict) else {}
    if SANDBOX_MODEL_BASE_URL_KEY in config:
        return True
    entrypoint = config.get("entrypoint") or "app.py"
    for root in component_search_roots():
        path = root / "responses_api_agents" / impl_name / entrypoint
        if path.is_file():
            return entrypoint_class_flag(path, CALLS_MODEL_FROM_SANDBOX_KEY) is True
    return False


def select_tasks(
    tasks: list[HarborTask], exclude: list[str], only: list[str] | None = None
) -> tuple[list[HarborTask], list[tuple[str, str]]]:
    """Split tasks into the ones to run and ``(task_id, reason)`` pairs left out."""
    kept: list[HarborTask] = []
    skipped: list[tuple[str, str]] = []
    for task in tasks:
        reason = selection_reason(task.task_id, exclude, only)
        if reason is None:
            kept.append(task)
        else:
            skipped.append((task.task_id, reason))
    return kept, skipped


def selection_reason(task_id: str, exclude: list[str], only: list[str] | None = None) -> str | None:
    """Why ``--only-tasks``/``--exclude-tasks`` leave ``task_id`` out, or ``None`` when they keep it."""
    if only and not any(fnmatch.fnmatchcase(task_id, pattern) for pattern in only):
        return "not in --only-tasks"
    pattern = next((pattern for pattern in exclude if fnmatch.fnmatchcase(task_id, pattern)), None)
    if pattern is not None:
        return f"excluded by --exclude-tasks {pattern!r}"
    return None


def _patterns(args: argparse.Namespace, name: str) -> list[str]:
    raw = getattr(args, name, None) or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


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
    return _override_value(overrides, key) is not None


def _override_value(overrides: list[str], key: str) -> str | None:
    for token in overrides:
        name, _, value = token.lstrip("+").partition("=")
        if name == key:
            return value
    return None


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
    # One file per agent, so a validate and an eval run on the same taskset do not overwrite each other.
    config_path = prepared.output_dir / f"run_config_{agent.instance_name}.yaml"
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


@exit_cleanly_on_config_error
def run_target(args: argparse.Namespace, overrides: list[str]) -> None:
    """Entry point for ``gym eval run <target> ...``."""
    from nemo_gym.cli.main import _merge_config_paths, dispatch
    from nemo_gym.rollout_collection import RolloutCollectionHelper

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
    if agent.impl_name == ORACLE_AGENT:
        # The oracle runs the reference solution, not a model: its run is `gym dataset validate`, whose results
        # live under results/harbor/validation/ where no model run, resumed or not, can pick them up as its own.
        validate_target(args, overrides, agent=agent)
        return
    try:
        prepared = prepare_target(
            args.target,
            exclude=_patterns(args, "exclude_tasks"),
            only=_patterns(args, "only_tasks"),
            rows_name=_rows_name(agent),
        )
    except (HarborTaskError, HubError, ValueError, OSError) as exc:
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
    tokens = _merge_config_paths(tokens)
    collection = _resolved_collection(tokens, prepared)
    cache = _resumable_cache(collection)
    if cache is not None:
        # The collector resumes its saved inputs, not the rows just prepared: a selection or digest that
        # changed since would run (or 404 on) tasks the server no longer maps. Refuse before startup.
        conflict = _resume_conflict(cache, RolloutCollectionHelper()._preprocess_rows_from_config(collection))
        if conflict:
            args._parser.error(conflict)
    print(f"Run config written to {config_path} (datasets folder: {datasets_dir()})")
    dispatch("nemo_gym.cli.eval:e2e_rollout_collection", tokens)


def _rows_name(agent: AgentSelection) -> str:
    """The rows file of one agent's run on a taskset: ``tasks_<agent>.jsonl`` next to ``run_config_<agent>.yaml``."""
    return f"tasks_{agent.instance_name}.jsonl"


def _resolved_collection(tokens: list[str], prepared: PreparedTaskset) -> "RolloutCollectionConfig":
    """The collector's effective configuration for ``tokens``: resolved through the same parser and cache
    dispatch uses, file overlays included, so what it would resume and run is read from it rather than
    guessed from the command line."""
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationResolutionError

    from nemo_gym.global_config import get_global_config_dict, translate_interpolation_error
    from nemo_gym.rollout_collection import RolloutCollectionConfig

    sys.argv = [sys.argv[0], *tokens]
    config = get_global_config_dict()
    try:
        resolved = OmegaConf.to_container(config, resolve=True)
    except InterpolationResolutionError as exc:
        raise translate_interpolation_error(exc) from exc
    return RolloutCollectionConfig.model_validate({**resolved, "input_jsonl_fpath": str(prepared.rows_path)})


def _resumable_cache(collection: "RolloutCollectionConfig") -> Path | None:
    """The saved inputs the collector would resume, or None when it starts fresh. Its own condition: the
    effective ``resume_from_cache`` (flag or config file) and both the results file and the saved inputs."""
    if not collection.resume_from_cache:
        return None
    output = Path(collection.output_jsonl_fpath)
    materialized = materialized_path_for(output)
    if not output.is_file() or not materialized.is_file():
        return None
    return materialized


def _resume_conflict(materialized: Path, expected: list[dict[str, object]]) -> str | None:
    """Why ``--resume`` cannot go ahead: the cached rows name other tasks or digests than the rows the collector
    would run now (``expected``, after ``limit``)."""

    def identity(row: dict[str, object]) -> tuple[str, str]:
        return _validation_key(row)[:2]

    cached = _task_digests(_validation_rows(materialized), identity)
    current = _task_digests(expected, identity)
    if cached == current:
        return None
    dropped = sorted(name for taskset, name in cached if (taskset, name) not in current)
    added = sorted(name for taskset, name in current if (taskset, name) not in cached)
    changed = sorted(
        name for key, digest in current.items() if key in cached and cached[key] != digest for name in [key[1]]
    )
    detail = ", ".join(
        part
        for part in (
            f"no longer selected: {', '.join(dropped)}" if dropped else "",
            f"newly selected: {', '.join(added)}" if added else "",
            f"changed digest: {', '.join(changed)}" if changed else "",
        )
        if part
    )
    return (
        f"Cannot resume from {materialized}: the task selection or task digests differ from the cached run "
        f"({detail}). Rerun with the same --only-tasks/--exclude-tasks, or pass +resume_from_cache=false to "
        f"start over, or a new +output_jsonl_fpath to keep the old results"
    )


@exit_cleanly_on_config_error
def validate_target(args: argparse.Namespace, overrides: list[str], *, agent: AgentSelection | None = None) -> None:
    """Entry point for ``gym dataset validate <target>``: run every task's reference solution and score it.

    ``gym eval run <target> --agent harbor_oracle`` lands here too, with its ``agent`` selection, so the oracle's
    results never share a folder with a model's.
    """
    from nemo_gym.cli.main import _merge_config_paths, dispatch
    from nemo_gym.rollout_collection import RolloutCollectionHelper

    # Evaluation and reference validation must never share inputs, configs, or cached results.
    try:
        agent = agent or resolve_agent(ORACLE_AGENT)
        sandbox = getattr(args, "sandbox", None)
        if sandbox is not None:
            resolve_sandbox_config(sandbox)
        prepared = prepare_target(
            args.target,
            output_root=DEFAULT_OUTPUT_ROOT / "validation",
            exclude=_patterns(args, "exclude_tasks"),
            only=_patterns(args, "only_tasks"),
            rows_name=_rows_name(agent),
        )
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
        collection = _resolved_collection(tokens, prepared)
        expected = RolloutCollectionHelper()._preprocess_rows_from_config(collection)
        rollouts = Path(collection.output_jsonl_fpath)
        cache = _resumable_cache(collection)
        if cache is not None:
            # The collector resumes its saved inputs, not the just-prepared task rows. Reject a stale
            # selection or digest before startup so an old reward cannot validate a changed solution.
            cached_digests = _task_digests(_validation_rows(cache), _validation_key)
            if cached_digests != _task_digests(expected, _validation_key):
                raise ConfigError(
                    f"Cannot resume Harbor validation from {cache}: "
                    "the selected tasks or task digests have changed. Use +resume_from_cache=false "
                    "to validate the current tasks, or choose a new +output_jsonl_fpath to keep the old results."
                )
            # Only the oracle agent stamps `response.metadata.oracle`. Cached results without it were written by
            # another run (a model evaluation on the same file); resumed, they would stand in for the oracle's.
            foreign = sorted(
                _validation_key(row)[1]
                for row in _validation_rows(rollouts)
                if ((row.get("response") or {}).get("metadata") or {}).get("oracle") is None
            )
            if foreign:
                raise ConfigError(
                    f"Cannot resume Harbor validation from {rollouts}: {len(foreign)} cached result(s) were not "
                    f"produced by the oracle agent (no oracle metadata): {', '.join(foreign[:5])}"
                    f"{', ...' if len(foreign) > 5 else ''}. Use +resume_from_cache=false to validate the current "
                    "tasks, or choose a new +output_jsonl_fpath to keep the old results."
                )
        print(f"Run config written to {config_path}; validating {len(expected)} oracle rollout(s)")
        dispatch("nemo_gym.cli.eval:e2e_rollout_collection", tokens)
    report = summarize_validation(rollouts, unvalidated, expected=expected, skipped=prepared.skipped)
    print(report.text)
    if not report.ok:
        sys.exit(1)
