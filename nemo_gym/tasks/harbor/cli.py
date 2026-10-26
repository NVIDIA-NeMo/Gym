# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gym eval run <folder | harbor:dataset> --agent <name>``: prepare a taskset and run it.

Preparation writes, under ``results/harbor/<taskset>/``:

- ``tasks.jsonl``: one materialized row per loadable task;
- ``skipped.json``: omitted task IDs and reasons (empty when all tasks load);
- ``run_config_<agent>.yaml``: the agent and environment server blocks.

Beside the task folders it reconciles ``manifest.toml`` (:mod:`nemo_gym.tasks.harbor.manifest`) before
anything is built or pushed: a fetched dataset whose folders no longer match the manifest is refused, a
local folder gets one. The dataset digest goes into the rows and the run config.

Then the ordinary end-to-end rollout collection runs with that config plus the
selected agent, environment server and model configs.
"""

import argparse
import fnmatch
import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from nemo_gym import component_search_roots
from nemo_gym.cli.utils import exit_cleanly_on_config_error
from nemo_gym.config_types import ConfigError
from nemo_gym.path_utils import materialized_path_for
from nemo_gym.sandbox.compose_config import deployment_platform
from nemo_gym.sandbox.compose_fields import DEFAULT_PLATFORM
from nemo_gym.sandbox.config import SANDBOX_BLOCK_VOLUME_SOURCES_KEY, validate_volume_sources
from nemo_gym.tasks.harbor.build_options import (
    BuildOptions,
    _block_from_overrides,
    _merged_cli_config,
    build_options_from_overrides,
)
from nemo_gym.tasks.harbor.build_plan import (
    DEFAULT_PLATFORM,
    BuildError,
)
from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE, read_dataset_config
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.hub import HubError, datasets_dir, fetch_ref, is_hub_ref
from nemo_gym.tasks.harbor.image_configs import (
    load_compose_document,
    record_base_images,
    record_compose_images,
)
from nemo_gym.tasks.harbor.manifest import (
    LOCAL_SOURCE,
    Manifest,
    ManifestError,
    check_manifest,
    check_task_manifest,
    dataset_digest,
    dataset_toml_digest,
)
from nemo_gym.tasks.harbor.materialize import (
    ENVIRONMENT_SERVER_IMPL,
    ENVIRONMENT_SERVER_INSTANCE,
    run_config,
    write_rows,
)
from nemo_gym.tasks.harbor.prepare_builds import _build_dockerfiles
from nemo_gym.tasks.harbor.records import read_image_configs
from nemo_gym.tasks.harbor.registry import ImageConfigError
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
IMAGES_FILE = "images.json"
ENVIRONMENT_SERVER_CONFIG = "environment_servers/harbor/configs/harbor.yaml"
ORACLE_AGENT = "harbor_oracle_agent"
# A harness that calls the Gym model server from inside the task sandbox, where 127.0.0.1 is the container
# itself, needs the model server to advertise the node's IP (`use_absolute_ip`). The harness declares that on
# the agent class its entrypoint runs (`calls_model_from_sandbox: ClassVar[bool] = True`), read from source
# like `ray_enabled`; an agent whose config declares `SANDBOX_MODEL_BASE_URL_KEY` routes its model calls
# through the sandbox too. Only the model server is moved (`absolute_ip_server_types`): the head server, whose
# config endpoint returns the merged configuration, the agent and the environment server stay on loopback as
# on any other run; a deployment that needs more sets `absolute_ip_server_types` or `default_host` itself.
SANDBOX_MODEL_BASE_URL_KEY = "sandbox_model_base_url"
CALLS_MODEL_FROM_SANDBOX_KEY = "calls_model_from_sandbox"
USE_ABSOLUTE_IP_KEY = "use_absolute_ip"
ABSOLUTE_IP_SERVER_TYPES_KEY = "absolute_ip_server_types"
SANDBOX_REACHABLE_SERVER_TYPES = ("responses_api_models",)
SANDBOX_PROVIDER_CONFIG = "nemo_gym/sandbox/providers/{provider}/configs/{provider}.yaml"
# The top-level block the Harbor environment server reads its provider from by default (`sandbox_provider`).
SANDBOX_BLOCK_KEY = "sandbox"
# Where the server's settings sit in the configuration a command names: the shipped environment server instance
# every run config inherits from. `gpu_sandbox_provider` names the block GPU-declaring tasks and GPU sidecars run
# on; `sandbox_provider_options.platform` ({os, arch}) is the platform a service's `platform` must match.
HARBOR_SERVER_KEY = f"{ENVIRONMENT_SERVER_INSTANCE}.environment_servers.{ENVIRONMENT_SERVER_IMPL}"
GPU_SANDBOX_PROVIDER_KEY = f"{HARBOR_SERVER_KEY}.gpu_sandbox_provider"
SANDBOX_PROVIDER_OPTIONS_KEY = f"{HARBOR_SERVER_KEY}.sandbox_provider_options"
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


def _block_provider_name(overrides: list[str], sandbox: str | None, block_key: str) -> str | None:
    """The provider name a top-level sandbox block carries, read from its keys alone (never its values)."""
    from omegaconf import OmegaConf

    from nemo_gym.sandbox.config import SANDBOX_BLOCK_RESERVED_KEYS

    files = [resolve_sandbox_config(sandbox)] if sandbox is not None else []
    block = OmegaConf.select(_merged_cli_config(overrides, files), block_key, throw_on_resolution_failure=False)
    if block is None or not hasattr(block, "keys"):
        return None
    providers = [str(key) for key in block.keys() if key not in SANDBOX_BLOCK_RESERVED_KEYS]
    return providers[0] if len(providers) == 1 else None


def sandbox_provider_from_overrides(overrides: list[str], sandbox: str | None) -> str | None:
    """The provider name (``opensandbox``, ``docker``, ...) of the ``sandbox:`` block the command names.

    The block comes from the ``--sandbox`` provider file (when given) and the configuration the command line
    names, merged as :func:`_merged_cli_config` does; only its keys are read, never its values, so provider
    credentials are not needed at prepare. Prepare checks each Compose task's ``tmpfs``, ``ulimits``,
    ``sysctls`` and ``security_opt`` against this provider's capabilities. ``None`` when no block names a
    provider: the check then runs at seed, against the live provider.
    """
    return _block_provider_name(overrides, sandbox, SANDBOX_BLOCK_KEY)


def gpu_sandbox_provider_from_overrides(overrides: list[str], sandbox: str | None) -> str | None:
    """The provider name of the block GPU-declaring tasks and GPU sidecars run on: the environment server's
    ``gpu_sandbox_provider`` in the configuration the command names (``GPU_SANDBOX_PROVIDER_KEY``), read like
    :func:`sandbox_provider_from_overrides` reads the ``sandbox:`` block. ``None`` when no GPU block is set
    (every service is then checked against the ``sandbox:`` block) or the block does not name one provider."""
    block = _block_from_overrides(overrides, GPU_SANDBOX_PROVIDER_KEY)
    if block is None:
        return None
    if not isinstance(block, str) or not block:
        raise ValueError(f"`{GPU_SANDBOX_PROVIDER_KEY}` must name a top-level sandbox block; got {block!r}")
    return _block_provider_name(overrides, sandbox, block)


def compose_platform_from_overrides(overrides: list[str]) -> str:
    """The platform a Compose service's ``platform`` must match at prepare: the environment server's
    ``sandbox_provider_options.platform`` (``{os, arch}``) in the configuration the command names, else
    linux/amd64, exactly as the server reads it at seed (``deployment_platform``).

    Only the ``platform`` subkey is selected and resolved. The sibling options (volumes, credentials, a
    ``${results_dir}`` the server's configuration parser defaults later) are not needed at prepare and are
    not evaluated; a block that is not a mapping selects no platform and the default stands, the server's own
    configuration validation names the malformed block at dispatch."""
    platform = _block_from_overrides(overrides, f"{SANDBOX_PROVIDER_OPTIONS_KEY}.platform")
    return deployment_platform({"platform": platform} if platform is not None else None)


def volume_sources_from_overrides(overrides: list[str], sandbox: str | None) -> dict[str, str] | None:
    """The ``volume_sources`` of the ``sandbox:`` block the command names, for checking Compose volumes at prepare.

    The block comes from the ``--sandbox`` provider file (when given) and the configuration the command
    line names, merged as :func:`_block_from_overrides` does. ``None`` when no ``volume_sources`` is
    configured: prepare then reports every external volume and foreign bind mount as unmapped.
    """
    return _block_volume_sources(overrides, sandbox, SANDBOX_BLOCK_KEY)


def gpu_volume_sources_from_overrides(overrides: list[str], sandbox: str | None) -> dict[str, str] | None:
    """The ``volume_sources`` of the block GPU-declaring tasks run on, for checking their Compose volumes at prepare.

    The block is the environment server's ``gpu_sandbox_provider`` in the configuration the command names
    (``GPU_SANDBOX_PROVIDER_KEY``). ``None`` when no GPU block is set: every task is then checked against
    the ``sandbox:`` block. An empty mapping when the block is set but carries no ``volume_sources``, which
    is what the server resolves such a task against.
    """
    block = _block_from_overrides(overrides, GPU_SANDBOX_PROVIDER_KEY)
    if block is None:
        return None
    if not isinstance(block, str) or not block:
        raise ValueError(f"`{GPU_SANDBOX_PROVIDER_KEY}` must name a top-level sandbox block; got {block!r}")
    return _block_volume_sources(overrides, sandbox, block) or {}


def _block_volume_sources(overrides: list[str], sandbox: str | None, block: str) -> dict[str, str] | None:
    files = [resolve_sandbox_config(sandbox)] if sandbox is not None else []
    sources = _block_from_overrides(overrides, f"{block}.{SANDBOX_BLOCK_VOLUME_SOURCES_KEY}", config_files=files)
    if sources is None:
        return None
    return validate_volume_sources(sources, f"the `{block}:` block")


def shared_storage_from_overrides(overrides: list[str], sandbox: str | None) -> bool | None:
    """Whether the ``sandbox:`` block's provider configures shared storage, for checking Compose named volumes and
    shared logs binds at prepare: OpenSandbox's ``shared_storage.host_path`` under the block's one provider key,
    read from the ``--sandbox`` provider file and the configuration the command names. ``None`` when the block
    names no single provider (the adapter then decides at seed); otherwise whether the path is set, which is
    the one setting the server's provider requires for every such volume."""
    return _block_shared_storage(overrides, sandbox, SANDBOX_BLOCK_KEY)


def gpu_shared_storage_from_overrides(overrides: list[str], sandbox: str | None) -> bool | None:
    """:func:`shared_storage_from_overrides` for the block GPU-declaring tasks and GPU sidecars run on
    (``GPU_SANDBOX_PROVIDER_KEY``); ``None`` when no GPU block is set."""
    block = _block_from_overrides(overrides, GPU_SANDBOX_PROVIDER_KEY)
    if block is None:
        return None
    if not isinstance(block, str) or not block:
        raise ValueError(f"`{GPU_SANDBOX_PROVIDER_KEY}` must name a top-level sandbox block; got {block!r}")
    return _block_shared_storage(overrides, sandbox, block)


def _block_shared_storage(overrides: list[str], sandbox: str | None, block: str) -> bool | None:
    provider = _block_provider_name(overrides, sandbox, block)
    if provider is None:
        return None
    files = [resolve_sandbox_config(sandbox)] if sandbox is not None else []
    host_path = _block_from_overrides(overrides, f"{block}.{provider}.shared_storage.host_path", config_files=files)
    return bool(host_path)


@dataclass(frozen=True)
class PreparedTaskset:
    taskset: str
    folder: Path
    tasks: list[HarborTask]
    rows_path: Path
    output_dir: Path
    skipped: dict[str, str] = field(default_factory=dict)
    # Every distinct image a run of the prepared tasks starts, for pre-pulling (``run_images``): the task images,
    # the separate verifiers' (declared or built from tests/Dockerfile) and the Compose services' as recorded.
    images: tuple[str, ...] = ()
    # The dataset's identity (``manifest.toml``'s ``[dataset].digest``) and the digest of its
    # ``dataset.toml``, recorded in every row and in the run config.
    dataset_digest: str | None = None
    dataset_toml_digest: str | None = None


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
    build: BuildOptions | None = None,
    sandbox_provider: str | None = None,
    gpu_sandbox_provider: str | None = None,
    compose_platform: str = DEFAULT_PLATFORM,
    volume_sources: dict[str, str] | None = None,
    gpu_volume_sources: dict[str, str] | None = None,
    shared_storage: bool | None = None,
    gpu_shared_storage: bool | None = None,
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
    is named. Compose service fields are checked against ``sandbox_provider`` and ``gpu_sandbox_provider``,
    the provider names of the ``sandbox:`` and GPU blocks, which decide the ``tmpfs``, ``ulimits``, ``sysctls``
    and ``security_opt`` each service's provider can apply, and against ``compose_platform``, the deployment's (see
    ``nemo_gym.tasks.harbor.image_configs.check_task_compose_fields``). Compose ``volumes`` are checked against ``volume_sources``, the deployment's mapping of
    external volumes and foreign bind sources to shared storage (see ``nemo_gym.sandbox.config``); a task that
    declares GPUs is checked against ``gpu_volume_sources`` when given, the mapping of the block the server runs
    such tasks on; ``shared_storage`` and ``gpu_shared_storage`` say whether each block's provider has the shared
    storage named volumes and shared logs binds need, so a task that needs it is left out at prepare, naming the
    setting, when the block has none (``None``: unknown, decided at seed). ``rows_name`` names the rows file inside the taskset's output folder: a run scopes it to its
    agent, so two agents prepared with different selections on one taskset keep their own rows.

    Before any image is recorded, built or pushed, ``manifest.toml`` beside the task folders is
    reconciled with their digests (:func:`~nemo_gym.tasks.harbor.manifest.check_manifest`): a fetched
    dataset whose folders changed is refused, a local folder gets a manifest. The digests hashed for
    that check are reused for the tasks that do not load; the loader's own folder digests are checked
    against them. Selection does not enter the dataset digest; it covers every task folder, so two runs of
    different subsets still name one dataset. A task folder that cannot be hashed (a file under it escapes
    it through a link, or cannot be read) is recorded in the manifest by name without a digest and does
    not enter the dataset digest; the loader refuses it, so it is skipped like any task that does not load,
    and selection applies to it as to any other task.
    """
    build = build or BuildOptions()
    if is_hub_ref(target):
        folder = fetch_ref(target, refresh_registry=refresh_registry)
        print(f"Fetched {target} into {folder}")
    else:
        folder = Path(target).expanduser().resolve()
    taskset = folder.name
    # Dataset-level files (dataset.toml, the recorded image configurations, the built images) live at the
    # dataset root, which is where the loader and the environment server read them from for nested tasks.
    dataset_root = folder.parent if is_task_folder(folder) else folder
    exclude = exclude or []
    # The refusal of a hand-edited fetched dataset comes first, so no image of unverified content is
    # built, pushed or recorded. A single task folder has no manifest of its own (see _dataset_identity).
    digests: dict[str, str] = {}
    identity = check_manifest(folder, digests=digests) if folder.is_dir() and not is_task_folder(folder) else None

    def selected(task_id: str) -> bool:
        return selection_reason(task_id, exclude, only) is None

    # A single task folder inside a fetched dataset is held here to the manifest its dataset folder's check holds it
    # to (the nearest fetched one within that check's discovery range: at most MAX_DISCOVERY_DEPTH levels up, through
    # visible folders that are not links), before any image is recorded or built (check_task_manifest): preparing
    # one task of a fetched dataset is no way around the check its dataset folder gets. Its identity then carries
    # the fetch's source (see _dataset_identity); a task no such manifest holds is held to nothing.
    parent = check_task_manifest(folder, digests=digests) if is_task_folder(folder) else None

    # Dockerfile ENV, WORKDIR and USER resolve from the base image's configuration, so it is recorded
    # before the tasks load; a base image the registry cannot resolve stops the preparation here. Only
    # the selected tasks' images are recorded, so an image a left-out task uses cannot stop it. A target
    # that is itself a task folder always loads (discover_tasks raises for it); selection applies after.
    # A missing or unreadable target is reported by discover_tasks below, in its own words. Duplicate
    # task IDs are rejected by the folder walk itself, before any registry or docker work. Images are
    # recorded for the platform the sandboxes run, ``builds.platform`` (linux/amd64 unless set), as the
    # built ones are.
    images = (
        record_base_images(folder, select=None if is_task_folder(folder) else selected, platform=build.platform)
        if folder.is_dir()
        else None
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
    identity = _dataset_identity(folder, loaded, identity, digests, parent=parent)
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
    # A Compose file that does not resolve, a field the sandbox Compose adapter would refuse at seed, or volumes the
    # group cannot run, leaves its task out the same way, with the field named, as an unloadable task is, not as a
    # seed failure of every episode; the others' images are still recorded.
    unresolved: dict[str, str] = {}
    images = record_compose_images(
        tasks,
        dataset_root,
        skipped=unresolved,
        platform=build.platform,
        sandbox_provider=sandbox_provider,
        gpu_sandbox_provider=gpu_sandbox_provider,
        compose_platform=compose_platform,
        volume_sources=volume_sources,
        gpu_volume_sources=gpu_volume_sources,
        shared_storage=shared_storage,
        gpu_shared_storage=gpu_shared_storage,
    )
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
    rows_path = output_dir / rows_name
    write_rows(tasks, taskset, rows_path, dataset_digest=identity.digest)
    skipped_note = f", skipped {len(errors)}" if errors else ""
    print(f"Materialized {len(tasks)} task(s) from {folder} into {rows_path}{skipped_note}")
    print(f"Dataset digest {identity.digest} ({identity.source})")
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
        dataset_digest=identity.digest,
        dataset_toml_digest=identity.dataset_toml_digest,
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
        if task.verifier_image:
            # The separate verifier's sandbox starts this image: the one prepare built from tests/Dockerfile
            # (recorded under the tests/ digest, never a Compose service, so nothing below would name it) or the
            # declared one, which the check below names too (HarborTask.verifier_image covers both).
            images.add(task.verifier_image)
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


def _dataset_identity(
    folder: Path,
    loaded: list[HarborTask],
    checked: Manifest | None,
    digests: dict[str, str],
    *,
    parent: Manifest | None = None,
) -> Manifest:
    """The dataset's manifest, reconciled with the loaded tasks' folder digests.

    A folder of task folders carries ``manifest.toml``, reconciled by ``check_manifest`` before the
    builds (``checked``, with the folder ``digests`` it hashed). The loader hashed the folders that loaded
    again, with the same function (``HarborTask.folder_digest``); a digest that differs means a folder
    changed under the preparation, so the manifest is reconciled once more with the loader's digests (and
    a fetched dataset refused). The manifest records folder digests only: ``HarborTask.digest``, the one
    rows carry, also folds in the ``dataset.toml`` settings, which the manifest covers once through
    ``dataset_toml_digest``, so a settings edit changes the dataset digest but is not a task file edit.
    A single task folder names no dataset folder of its own (its parent is whatever holds it), so its
    identity is computed in memory over that one task, with the ``dataset.toml`` beside it, and nothing
    is written (``checked`` is ``None`` exactly then). When a fetched dataset's check reaches the task (its
    manifest is the nearest fetched one within discovery range, wherever under that folder the task sits),
    ``parent`` is that manifest, which ``check_task_manifest`` held the folder to before the builds: the identity
    is that same in-memory one with the fetch's source instead of ``local`` (still over the ``dataset.toml`` beside
    the task: the manifest holds the task's content, not the settings a nested task is prepared with), and a
    folder that changed under the preparation is held to the manifest once more with the loader's digest, as a
    dataset folder is.
    """
    if parent is not None:
        hashed = {task.task_id: task.folder_digest for task in loaded}
        if any(digests.get(name) != digest for name, digest in hashed.items()):
            check_task_manifest(folder, digests=hashed)
        return replace(_dataset_identity(folder, loaded, None, digests), source=parent.source)
    if checked is None:
        hashed = {task.task_id: task.folder_digest for task in loaded}
        toml_digest = dataset_toml_digest(folder.parent)
        return Manifest(
            source=LOCAL_SOURCE,
            digest=dataset_digest(hashed, toml_digest),
            dataset_toml_digest=toml_digest,
            tasks={name: {"digest": digest} for name, digest in hashed.items()},
        )
    if all(digests.get(task.task_id) == task.folder_digest for task in loaded):
        return checked
    return check_manifest(folder, digests={task.task_id: task.folder_digest for task in loaded})


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
        dataset_digest=prepared.dataset_digest,
        dataset_toml_digest=prepared.dataset_toml_digest,
    )
    # Defaults belong below explicit config files and command-line overrides.
    config["output_jsonl_fpath"] = str(prepared.output_dir / "rollouts.jsonl")
    if agent.runs_in_sandbox:
        config[USE_ABSOLUTE_IP_KEY] = True
        config[ABSOLUTE_IP_SERVER_TYPES_KEY] = list(SANDBOX_REACHABLE_SERVER_TYPES)
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
        # The `builds:` and `sandbox:` blocks come from the configuration the command names; the run config
        # does not exist yet.
        build = build_options_from_overrides(overrides)
        prepared = prepare_target(
            args.target,
            exclude=_patterns(args, "exclude_tasks"),
            only=_patterns(args, "only_tasks"),
            build=build,
            sandbox_provider=sandbox_provider_from_overrides(overrides, sandbox),
            gpu_sandbox_provider=gpu_sandbox_provider_from_overrides(overrides, sandbox),
            compose_platform=compose_platform_from_overrides(overrides),
            volume_sources=volume_sources_from_overrides(overrides, sandbox),
            gpu_volume_sources=gpu_volume_sources_from_overrides(overrides, sandbox),
            shared_storage=shared_storage_from_overrides(overrides, sandbox),
            gpu_shared_storage=gpu_shared_storage_from_overrides(overrides, sandbox),
            rows_name=_rows_name(agent),
        )
    except (
        HarborTaskError,
        HubError,
        BuildError,
        ImageConfigError,
        ManifestError,
        ConfigError,
        ValueError,
        OSError,
    ) as exc:
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

    from nemo_gym.global_config import get_global_config_dict, reset_global_config_dict, translate_interpolation_error
    from nemo_gym.rollout_collection import RolloutCollectionConfig

    sys.argv = [sys.argv[0], *tokens]
    # These tokens, not whatever this process resolved before them (the run config did not exist yet); the
    # result stays cached, so dispatch runs on it without a second parse.
    reset_global_config_dict()
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
            build=build_options_from_overrides(overrides),
            sandbox_provider=sandbox_provider_from_overrides(overrides, sandbox),
            gpu_sandbox_provider=gpu_sandbox_provider_from_overrides(overrides, sandbox),
            compose_platform=compose_platform_from_overrides(overrides),
            volume_sources=volume_sources_from_overrides(overrides, sandbox),
            gpu_volume_sources=gpu_volume_sources_from_overrides(overrides, sandbox),
            shared_storage=shared_storage_from_overrides(overrides, sandbox),
            gpu_shared_storage=gpu_shared_storage_from_overrides(overrides, sandbox),
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
    write_rows(prepared.tasks, prepared.taskset, prepared.rows_path, dataset_digest=prepared.dataset_digest)
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
