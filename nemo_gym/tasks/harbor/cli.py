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
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from nemo_gym import _resolve_under_cwd_or_install, component_search_roots
from nemo_gym.cli.utils import exit_cleanly_on_config_error
from nemo_gym.config_types import ConfigError
from nemo_gym.path_utils import failures_path_for, materialized_path_for
from nemo_gym.sandbox.compose_config import unsupported_compose_fields
from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE, read_dataset_config
from nemo_gym.tasks.harbor.digest import DIGEST_KEY
from nemo_gym.tasks.harbor.hub import HubError, datasets_dir, fetch_ref, is_hub_ref
from nemo_gym.tasks.harbor.image_build import (
    BUILD_TOOLS,
    DEFAULT_PLATFORM,
    REPOSITORY_ENV,
    BuildError,
    BuildPlanEntry,
    ImageBuilder,
    build_images,
    plan_builds,
    recorded_image_mismatch,
)
from nemo_gym.tasks.harbor.image_configs import (
    COMPOSE_FILE_NAMES,
    ImageConfigError,
    compose_file,
    load_compose_document,
    record_base_images,
    record_compose_images,
    without_main_build,
    without_pull_policy,
    without_restated_main_build,
)
from nemo_gym.tasks.harbor.materialize import run_config, write_rows
from nemo_gym.tasks.harbor.task import (
    HarborTask,
    HarborTaskError,
    discover_tasks,
    is_task_folder,
    read_built_images,
    read_image_configs,
)


logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("results/harbor")
IMAGES_FILE = "images.json"
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
    # Episode sessions are process-local, so the harness must run as one worker. Its shipped template
    # leaves `resources_server.name` unset for benchmark composition; here the environment server
    # drives it, so the generated instance declares none (and is not a composition source).
    "terminus_2_sandboxed_agent": {"num_workers": 1, "resources_server": None},
    "miniswe_sandboxed_agent": {"num_workers": 1},
}


# The top-level configuration block that sets Dockerfile builds up, and the settings it takes.
BUILDS_KEY = "builds"
BUILD_SETTINGS = ("tool", "repository", "platform", "enabled")
HYDRA_QUIET_OVERRIDES = (
    "hydra.run.dir=.",
    "hydra.output_subdir=null",
    "hydra/job_logging=none",
    "hydra/hydra_logging=none",
)


@dataclass(frozen=True)
class BuildOptions:
    """How prepare treats Dockerfiles outside pull and overlay mode: the ``builds:`` configuration block.

    ``tool`` builds them (buildah, podman or docker; default: the first one on PATH that fits, see
    :func:`nemo_gym.tasks.harbor.image_build.detect_tool`). ``repository`` is the registry path built
    images are pushed under (default: the ``NEMO_GYM_BUILD_REPOSITORY`` variable; unset keeps them local
    in the Docker daemon). ``platform`` is passed to the tool (default ``linux/amd64``). With ``enabled``
    false nothing is built, and the tasks that need a build are skipped with the loader's message.
    """

    enabled: bool = True
    tool: str | None = None
    repository: str | None = None
    platform: str | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "BuildOptions":
        """The ``builds:`` block of a merged configuration dict, applied over the defaults.

        ``config`` is the merged global configuration (``env.yaml``, ``--config`` files and
        ``++builds.<setting>=<value>`` overrides) or any dict shaped like it; a missing block keeps
        every default. Unknown settings and values of the wrong type are refused.
        """
        block = (config or {}).get(BUILDS_KEY)
        if block is None:
            block = {}
        if not isinstance(block, Mapping):
            raise ValueError(f"`{BUILDS_KEY}` must be a mapping of {', '.join(BUILD_SETTINGS)}; got {block!r}")
        unknown = sorted(set(block) - set(BUILD_SETTINGS))
        if unknown:
            raise ValueError(
                f"Unknown `{BUILDS_KEY}` setting(s) {', '.join(unknown)}; expected {', '.join(BUILD_SETTINGS)}"
            )
        enabled = block.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"`{BUILDS_KEY}.enabled` must be true or false; got {enabled!r}")
        tool = _build_setting(block, "tool")
        if tool is not None and tool not in BUILD_TOOLS:
            raise ValueError(f"`{BUILDS_KEY}.tool` must be one of {', '.join(BUILD_TOOLS)}; got {tool!r}")
        return cls(
            enabled=enabled,
            tool=tool,
            repository=_build_setting(block, "repository") or os.environ.get(REPOSITORY_ENV) or None,
            platform=_build_setting(block, "platform"),
        )


def _build_setting(block: Mapping[str, Any], name: str) -> str | None:
    value = block.get(name)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"`{BUILDS_KEY}.{name}` must be a string; got {value!r}")
    return value


def build_options_from_overrides(overrides: list[str]) -> BuildOptions:
    """Read the ``builds:`` block from the configuration the command line names, before the run config exists.

    Preparation builds images before the run config can be written, so the global configuration is not
    composed yet. This merges the sources the user controls, in the global config's order (the files
    the ``+config_paths`` tokens name, from ``--config`` and the asset selectors, then ``env.yaml``, then
    the ``+``/``++`` overrides), and reads the block from the result. The full global config is still
    parsed once, at dispatch, with the run config added.
    """
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationResolutionError

    from nemo_gym.global_config import (
        CONFIG_PATHS_KEY_NAME,
        GlobalConfigDictParser,
        _load_config_yaml,
        translate_interpolation_error,
    )

    parser = GlobalConfigDictParser()
    saved_argv = sys.argv
    # A side-effect-free Hydra parse: no outputs/<date>/<time> run folder and no logging reconfiguration,
    # which the one parse at dispatch keeps as its own.
    sys.argv = [saved_argv[0], *overrides, *HYDRA_QUIET_OVERRIDES]
    try:
        cli = parser.parse_global_config_dict_from_cli()
    finally:
        sys.argv = saved_argv
    OmegaConf.set_struct(cli, False)
    dotenv_path = _resolve_under_cwd_or_install("env.yaml")
    dotenv = _load_config_yaml(dotenv_path) if dotenv_path.exists() else OmegaConf.create({})
    config_paths = OmegaConf.merge(dotenv, cli).get(CONFIG_PATHS_KEY_NAME) or []
    _, extra_configs = parser.load_extra_config_paths([str(path) for path in config_paths])
    merged = OmegaConf.merge(*extra_configs, dotenv, cli)
    block = merged.get(BUILDS_KEY)
    try:
        resolved = OmegaConf.to_container(block, resolve=True) if block is not None else None
    except InterpolationResolutionError as exc:
        raise translate_interpolation_error(exc) from exc
    return BuildOptions.from_config({BUILDS_KEY: resolved})


@dataclass(frozen=True)
class PreparedTaskset:
    taskset: str
    folder: Path
    tasks: list[HarborTask]
    rows_path: Path
    output_dir: Path
    skipped: dict[str, str] = field(default_factory=dict)
    # Every distinct image the prepared tasks start from, for pre-pulling.
    images: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentSelection:
    config_path: Path
    instance_name: str
    impl_name: str
    # The harness calls the model server from inside the task sandbox (see ``IN_SANDBOX_AGENTS``).
    runs_in_sandbox: bool = False


REFUSED_NOTE = " (the sandbox Compose adapter does not start them)"


def unsupported_compose_fields_error(task: HarborTask) -> HarborTaskError | None:
    """Why the sandbox Compose adapter would refuse ``task``'s document at seed, or ``None`` when it starts it.

    The document checked is the one the server hands the adapter (``environment_servers/harbor/app.py``,
    ``_compose_document``): interpolated and ``env_file``-inlined by the loader, ``pull_policy`` dropped from
    every service, ``main``'s ``build:`` dropped when the task runs from its recorded built image or when it
    only restates Harbor's own build of the task's Dockerfile (:func:`without_restated_main_build`; a build
    key Gym cannot apply without a build is this task's reason). A field no provider starts (``deploy``,
    ``extra_hosts``, ``platform``, a sidecar ``build:``) is then a per-task reason naming the service and the
    field, recorded in ``skipped.json`` like a task that does not load, instead of a failure of every episode
    at seed.
    """
    path = compose_file(task)
    document = load_compose_document(task)
    if path is None or document is None:
        return None
    document = without_pull_policy(document)
    if task.built_image is not None:
        document = without_main_build(document)
    else:
        document, reason = without_restated_main_build(task, document)
        if reason is not None:
            return HarborTaskError(f"{path.name}: {reason}")
    refused = unsupported_compose_fields(document)
    if not refused:
        return None
    return HarborTaskError(f"{path.name}: {'; '.join(refused)}{REFUSED_NOTE}")


def refused_before_build(plan: list[BuildPlanEntry]) -> dict[str, HarborTaskError]:
    """The field refusals of the tasks ``plan`` would build, read from their Compose files before any build runs.

    :func:`unsupported_compose_fields_error` is the authoritative check, on the loader's document once the
    tasks load. This one needs only key names, which interpolation and ``env_file`` inlining do not add, so it
    reads the raw file the way the loader's document will look: ``pull_policy`` dropped, ``main``'s ``build:``
    dropped (the task is about to be built and will run from its recorded image) and ``env_file`` set aside,
    as the loader folds it into ``environment``. A task it names is neither built nor pushed for an image no
    episode would start, and a build that would fail for it cannot stop the preparation. A file that does not
    parse is left to the loader to report in its own words.
    """
    refused: dict[str, HarborTaskError] = {}
    for entry in plan:
        path = next((entry.context / name for name in COMPOSE_FILE_NAMES if (entry.context / name).is_file()), None)
        if path is None:
            continue
        try:
            document = yaml.safe_load(path.read_text()) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        if not isinstance(document, dict):
            continue
        document = without_main_build(without_pull_policy(document))
        services = document.get("services")
        if isinstance(services, dict):
            for name, service in services.items():
                if isinstance(service, dict) and "env_file" in service:
                    services[name] = {key: value for key, value in service.items() if key != "env_file"}
        messages = unsupported_compose_fields(document)
        if messages:
            refused.update(
                dict.fromkeys(entry.tasks, HarborTaskError(f"{path.name}: {'; '.join(messages)}{REFUSED_NOTE}"))
            )
    return refused


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
    build: BuildOptions | None = None,
    rows_name: str = "tasks.jsonl",
) -> PreparedTaskset:
    """Fetch (for hub references) and load the tasks, then write the rows file.

    A task folder that does not load is skipped with a warning (and listed in
    ``PreparedTaskset.skipped``) so the rest of the dataset still prepares; so is a task whose
    image fails to build or whose Compose file does not resolve, with the build's or the
    resolver's reason.

    Tasks matching an ``exclude`` glob, or not matching an ``only`` glob, are left out of the
    rows and listed on stdout. Dockerfiles outside pull and overlay mode are built once and
    recorded before the tasks load, as ``build`` (the ``builds:`` configuration block) says; the
    default builds locally with docker, and pushes with the first tool on PATH when a repository
    is named. ``rows_name`` names the rows file inside the taskset's output folder: a run scopes
    it to its agent, so two agents prepared with different selections on one taskset keep their
    own rows.
    """
    build = build or BuildOptions()
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
    unbuilt: dict[str, str] = {}
    if folder.is_dir():
        unbuilt = _build_dockerfiles(folder, build, select=None if is_task_folder(folder) else selected)
    output_dir = Path(output_root) / taskset
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: dict[str, HarborTaskError] = {}
    try:
        loaded = discover_tasks(folder, skipped=errors)
    except HarborTaskError:
        if not unbuilt:
            # Keep the omitted IDs and reasons even when no task in the dataset loads.
            _write_skipped(output_dir, errors)
            raise
        # Nothing loaded, and the builds that failed are the reason to report below, not the loader's message.
        loaded = []
    # A task whose image failed to build is left out with the build's reason; the loader's own message for it
    # (its Dockerfile needs a build that is not recorded) says less.
    for task_id, reason in unbuilt.items():
        errors[task_id] = HarborTaskError(reason)
    loaded = [task for task in loaded if task.task_id not in unbuilt]
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
        raise _no_runnable_task(folder, errors)
    # A Compose file that does not resolve leaves its task out the same way; the others' images are still recorded.
    unresolved: dict[str, str] = {}
    images = record_compose_images(tasks, tasks[0].path.parent, skipped=unresolved)
    if unresolved:
        for task_id, reason in unresolved.items():
            errors[task_id] = HarborTaskError(reason)
            logger.warning("Skipping task %s: %s", task_id, reason)
        tasks = [task for task in tasks if task.task_id not in unresolved]
        _write_skipped(output_dir, errors)
        if not tasks:
            raise _no_runnable_task(folder, errors)
    if images is not None:
        print(f"Compose image configurations recorded in {images}")
    # The Compose adapter's field check, per task and before any sandbox exists: a field it does not start is
    # a skipped task with the field named, as an unloadable task is, not a seed failure of every episode.
    refused = {task.task_id: error for task in tasks if (error := unsupported_compose_fields_error(task))}
    if refused:
        errors.update(refused)
        _write_skipped(output_dir, errors)
        for task_id, error in refused.items():
            logger.warning("Skipping task %s: %s", task_id, error)
        tasks = [task for task in tasks if task.task_id not in refused]
        if not tasks:
            raise _no_runnable_task(folder, errors)
    rows_path = output_dir / rows_name
    write_rows(tasks, taskset, rows_path)
    skipped_note = f", skipped {len(errors)}" if errors else ""
    print(f"Materialized {len(tasks)} task(s) from {folder} into {rows_path}{skipped_note}")
    # Every distinct image a run starts, for pre-pulling onto the sandbox hosts before it: the task images, the
    # Compose sidecars' (as recorded, the pinned reference the sandbox gets) and the separate verifiers'.
    distinct = run_images(tasks, read_image_configs(tasks[0].path.parent))
    (output_dir / IMAGES_FILE).write_text(json.dumps(list(distinct), indent=2) + "\n")
    if distinct:
        print(f"Images: {len(distinct)} distinct, listed in {output_dir / IMAGES_FILE}")
        for image in distinct:
            print(f"  {image}")
    return PreparedTaskset(
        taskset=taskset,
        folder=folder,
        tasks=tasks,
        rows_path=rows_path,
        output_dir=output_dir,
        skipped={task_id: str(error) for task_id, error in errors.items()},
        images=distinct,
    )


def run_images(tasks: list[HarborTask], image_configs: Mapping[str, Any]) -> tuple[str, ...]:
    """Every distinct image a run of ``tasks`` starts, sorted: each task's image, the image of a separate
    verifier (``[verifier.environment].docker_image``) and every Compose service image, the last as the
    recorded configuration pins it (``compose-images.json``, what ``resolve_compose`` hands the sandbox),
    else as written when it has no record."""
    images: set[str] = set()
    for task in tasks:
        if task.image:
            images.add(task.image)
        verifier_environment = task.config.verifier.environment
        if (
            not task.config.is_shared_verifier
            and verifier_environment is not None
            and verifier_environment.docker_image
        ):
            images.add(verifier_environment.docker_image)
        document = load_compose_document(task) if task.needs_compose else None
        for service in ((document or {}).get("services") or {}).values():
            reference = (service or {}).get("image")
            if reference:
                record = image_configs.get(reference)
                images.add(str(record.get("image") or reference) if isinstance(record, dict) else reference)
    return tuple(sorted(images))


def _listed(reasons: Mapping[str, object]) -> str:
    """``reasons`` (task id to why it was left out) as indented lines; a reason that starts with its task id says it once."""
    return "".join(f"\n  {task_id}: {str(reason).removeprefix(f'{task_id}: ')}" for task_id, reason in reasons.items())


def _no_runnable_task(folder: Path, errors: Mapping[str, object]) -> ValueError:
    """The error of a preparation that keeps no task, listing every task left out and why."""
    return ValueError(f"No runnable task left in {folder}" + (f"; left out:{_listed(errors)}" if errors else ""))


def _build_dockerfiles(folder: Path, build: BuildOptions, *, select) -> dict[str, str]:
    """Build and record the images of the selected tasks whose Dockerfile needs a build.

    With builds disabled, the tasks that need one are left for the loader to skip with its
    message, and a recorded image that does not fit the run (another platform, or local when the run
    pushes) leaves its tasks out with the reason, as the enabled path would rebuild it. When every context
    has a usable record nothing is built and no build tool is looked for. Otherwise a missing builder stops
    the preparation, naming the tasks. A failed build does not:
    each context is built on its own, and the tasks of one that fails are returned by id with the
    reason, for the caller to leave out the way it leaves out a task that does not load. A planned task
    whose Compose file the sandbox adapter would refuse whatever the build gave (:func:`refused_before_build`)
    is returned the same way, and its context is neither built nor pushed.
    """
    plan = plan_builds(folder, select=select)
    refused = {task_id: str(error) for task_id, error in refused_before_build(plan).items()}
    plan = [entry for entry in plan if not any(task_id in refused for task_id in entry.tasks)]
    if not plan:
        return refused
    # A recorded image is reused when it fits this run (pushed when the run pushes, of the requested platform),
    # with builds enabled or not; the check is build_images' own.
    parent = folder.parent if is_task_folder(folder) else folder
    recorded, image_configs = read_built_images(parent), read_image_configs(parent)
    platform = build.platform or DEFAULT_PLATFORM
    reusable: dict[str, str] = {}
    unusable: dict[str, str] = {}
    for entry in plan:
        if entry.digest not in recorded:
            continue
        why = recorded_image_mismatch(
            recorded[entry.digest], image_configs, repository=build.repository, platform=platform
        )
        (reusable if why is None else unusable)[entry.digest] = why or recorded[entry.digest]["image"]
    if not build.enabled:
        unbuilt = [task for entry in plan if entry.digest not in recorded for task in entry.tasks[:3]]
        if unbuilt:
            print(
                f"Skipping Dockerfile builds ({BUILDS_KEY}.enabled is false); tasks that need one are left out: "
                + ", ".join(unbuilt)
            )
        # A record that does not fit the run is no better than none: its tasks are left out with the reason.
        mismatched: dict[str, str] = {}
        for entry in plan:
            if entry.digest in unusable:
                image = recorded[entry.digest]["image"]
                reason = f"recorded image {image} cannot be used: {unusable[entry.digest]}; builds are disabled"
                print(f"Skipping {', '.join(entry.tasks[:3])}: {reason}")
                mismatched.update(dict.fromkeys(entry.tasks, reason))
        return {**refused, **mismatched}
    needs_build = [entry for entry in plan if entry.digest not in reusable]
    if not needs_build:
        # Nothing to build, so no build tool is needed: a host with every image recorded prepares without one.
        for entry in plan:
            print(f"Built image recorded for {', '.join(entry.tasks[:3])}: {reusable[entry.digest]}")
        return refused
    names = ", ".join(task for entry in needs_build for task in entry.tasks[:3])
    try:
        builder = ImageBuilder(tool=build.tool, repository=build.repository, platform=build.platform)
    except BuildError as exc:
        raise ValueError(
            f"Could not build the Dockerfile image(s) for task(s) {names}: {exc} (set {BUILDS_KEY}.enabled to false "
            "to run with recorded images only)"
        ) from exc
    results = []
    failures: dict[str, str] = {}
    for entry in plan:
        # One build_images call per context, so a failure stops that context only and the rest still build.
        tasks = set(entry.tasks)
        try:
            results += build_images(folder, builder=builder, select=lambda name: name in tasks)
        except BuildError as exc:
            reason = f"Could not build the Dockerfile image(s) for task(s) {', '.join(entry.tasks[:3])}: {exc}"
            failures.update(dict.fromkeys(entry.tasks, reason))
    if results:
        pushed = sum(1 for result in results if result.pushed)
        print(
            f"Built {len(results)} image(s) with {results[0].tool}"
            + (f", pushed {pushed}" if pushed else ", kept local")
        )
    return {**refused, **failures}


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
    """Whether the harness behind ``impl_name`` calls the model server from inside the task sandbox."""
    declares_sandbox_url = isinstance(impl_config, dict) and SANDBOX_MODEL_BASE_URL_KEY in impl_config
    return declares_sandbox_url or impl_name in IN_SANDBOX_AGENTS


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
        # The `builds:` block comes from the configuration the command names; the run config does not exist yet.
        build = build_options_from_overrides(overrides)
        prepared = prepare_target(
            args.target,
            exclude=_patterns(args, "exclude_tasks"),
            only=_patterns(args, "only_tasks"),
            build=build,
            rows_name=_rows_name(agent),
        )
    except (HarborTaskError, HubError, BuildError, ImageConfigError, ConfigError, ValueError, OSError) as exc:
        args._parser.error(str(exc))
    if prepared.skipped and not getattr(args, "skip_unloadable", False):
        args._parser.error(
            f"{len(prepared.skipped)} task(s) could not load (see {prepared.output_dir / 'skipped.json'}):"
            f"{_listed(prepared.skipped)}\nUse --skip-unloadable to explicitly run only the loadable subset."
        )
    try:
        config_path, tokens = build_run(prepared, agent, sandbox=sandbox, overrides=overrides)
    except (HarborTaskError, OSError) as exc:
        args._parser.error(str(exc))
    tokens = _merge_config_paths(tokens)
    if _resume_requested(tokens):
        # The collector resumes its saved inputs, not the rows just prepared: a selection or digest that
        # changed since would run (or 404 on) tasks the server no longer maps. Refuse before startup.
        conflict = _resume_conflict(_resumed_materialized_path(tokens, prepared), prepared, agent)
        if conflict:
            args._parser.error(conflict)
    print(f"Run config written to {config_path} (datasets folder: {datasets_dir()})")
    dispatch("nemo_gym.cli.eval:e2e_rollout_collection", tokens)


def _rows_name(agent: AgentSelection) -> str:
    """The rows file of one agent's run on a taskset: ``tasks_<agent>.jsonl`` next to ``run_config_<agent>.yaml``."""
    return f"tasks_{agent.instance_name}.jsonl"


def _resume_requested(tokens: list[str]) -> bool:
    return any(
        token.lstrip("+").startswith("resume_from_cache=") and token.rsplit("=", 1)[1].strip().lower() == "true"
        for token in tokens
    )


def _resumed_materialized_path(tokens: list[str], prepared: PreparedTaskset) -> Path:
    """Where the collector would read its cached inputs from, through the same parser and cache as dispatch."""
    from omegaconf import OmegaConf

    from nemo_gym.global_config import get_global_config_dict

    sys.argv = [sys.argv[0], *tokens]
    config = get_global_config_dict()
    output = OmegaConf.select(config, "output_jsonl_fpath", throw_on_resolution_failure=False)
    return materialized_path_for(Path(str(output)) if output else prepared.output_dir / "rollouts.jsonl")


def _task_digests(rows: list[dict[str, object]], key: Any) -> dict[Any, object]:
    """``key(row) -> task digest`` for materialized rows; shared by the eval and validation resume checks."""
    return {key(row): _dig(row, "task_input", "task_data", DIGEST_KEY) for row in rows}


def _resume_conflict(materialized: Path, prepared: PreparedTaskset, agent: AgentSelection) -> str | None:
    """Why ``--resume`` cannot go ahead: the cached rows name other tasks or digests than the current selection."""
    if not materialized.is_file():
        return None
    cached = _task_digests(_validation_rows(materialized), lambda row: _validation_key(row)[:2])
    current = {(prepared.taskset, task.task_id): task.digest for task in prepared.tasks}
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
        prepared = prepare_target(
            args.target,
            output_root=DEFAULT_OUTPUT_ROOT / "validation",
            exclude=_patterns(args, "exclude_tasks"),
            only=_patterns(args, "only_tasks"),
            build=build_options_from_overrides(overrides),
            rows_name=_rows_name(agent),
        )
    except (
        HarborTaskError,
        HubError,
        BuildError,
        ImageConfigError,
        ConfigError,
        ValueError,
        OSError,
        yaml.YAMLError,
    ) as exc:
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
            cached_digests = _task_digests(_validation_rows(collection.materialized_jsonl_fpath), _validation_key)
            if cached_digests != _task_digests(expected, _validation_key):
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
