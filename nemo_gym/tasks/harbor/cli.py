# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``gym eval run <folder | harbor:dataset | entry> --agent <name>``: prepare a taskset and run it.

A target is a task folder, a folder of task folders, a ``harbor:`` reference (fetched first) or the name of a
committed entry under ``environments/harbor/`` (resolved to the folder its pull filled, never fetched here;
``gym dataset pull``, :func:`pull_dataset`, is the fetch). Preparation writes, under ``results/harbor/<taskset>/``:

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
import shlex
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError

from nemo_gym import component_search_roots
from nemo_gym.cli.utils import exit_cleanly_on_config_error
from nemo_gym.config.config_types import ConfigError
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
from nemo_gym.tasks.harbor.compose_overlay import HarborComposeOverlay
from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE, GymDatasetConfig, read_dataset_config
from nemo_gym.tasks.harbor.entries import (
    DatasetEntry,
    PullPlan,
    entry_at,
    entry_target,
    plan_pull,
    prepare_destination,
)
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.hub import (
    REGISTRY_SOURCE,
    HubError,
    HubRef,
    datasets_dir,
    declares_recorded_fetch,
    expected_folder,
    fetch_ref,
    is_hub_ref,
    resolved_folder,
)
from nemo_gym.tasks.harbor.image_configs import (
    load_compose_document,
    record_base_images,
    record_compose_images,
)
from nemo_gym.tasks.harbor.manifest import (
    LOCAL_SOURCE,
    MANIFEST_FILE,
    Manifest,
    ManifestError,
    check_manifest,
    check_task_manifest,
    dataset_digest,
    dataset_files_digest,
    dataset_toml_digest,
    gym_layer_change,
    refresh_gym_layer,
    report_rewrite,
)
from nemo_gym.tasks.harbor.materialize import (
    ENVIRONMENT_SERVER_IMPL,
    ENVIRONMENT_SERVER_INSTANCE,
    run_config,
    write_rows,
)
from nemo_gym.tasks.harbor.package_store import PACKAGE_STORE_SOURCE
from nemo_gym.tasks.harbor.parameterized import is_template_folder, render_template_folder
from nemo_gym.tasks.harbor.prepare_builds import _build_dockerfiles
from nemo_gym.tasks.harbor.records import read_image_configs
from nemo_gym.tasks.harbor.registry import (
    ImageConfigError,
    RegistryClient,
    credentials_for_image_prefixes,
    platform_pair,
)
from nemo_gym.tasks.harbor.run_summary import NO_REFERENCE_SOLUTION, output_signatures, write_run_summary
from nemo_gym.tasks.harbor.task import HarborTask, discover_tasks, is_task_folder, iter_task_folders
from nemo_gym.tasks.harbor.task_data import check_task_folders, check_tasks, load_task_data_schema
from nemo_gym.tasks.harbor.validation_report import (
    _task_digests,
    _validation_key,
    _validation_rows,
    summarize_validation,
)
from nemo_gym.utils.path_utils import materialized_path_for


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
# The job-level Compose overlay (extra Compose files, MCP entries, prompt template), read at prepare so the
# overlay's images are recorded and its merge checked with the tasks', exactly as the server reads it at seed.
COMPOSE_OVERLAY_KEY = f"{HARBOR_SERVER_KEY}.compose_overlay"
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
    credentials are not needed for this (the registry credentials prepare does read are the block's
    ``images.auth``, :func:`registry_credentials_from_overrides`). Prepare checks each Compose task's ``tmpfs``, ``ulimits``,
    ``sysctls`` and ``security_opt`` against this provider's capabilities. ``None`` when no block names a
    provider: the check then runs at seed, against the live provider.
    """
    return _block_provider_name(overrides, sandbox, SANDBOX_BLOCK_KEY)


# The sandbox provider block's image credentials (OpenSandbox's `images.auth`: image prefix to username and
# password), the one place a deployment names registry credentials; prepare reads private images with the same.
SANDBOX_IMAGES_AUTH_KEY = "images.auth"


def registry_credentials_from_overrides(overrides: list[str], sandbox: str | None) -> dict[str, tuple[str, str]]:
    """The registry credentials prepare records private images with: the ``sandbox:`` block's ``images.auth``
    (what the provider pulls those images with at seed), turned into the registry client's prefix-keyed form
    (:func:`~nemo_gym.tasks.harbor.registry.credentials_for_image_prefixes`), so each image is read with the
    credentials of its longest matching prefix, as the provider pulls it.

    Read from the ``--sandbox`` provider file and the configuration the command names, resolved against them
    (so ``${gitlab_registry_password}`` comes from ``env.yaml`` as it does at seed), like
    :func:`volume_sources_from_overrides`. Empty when no block names a provider or the block has no
    ``images.auth``: prepare then reads registries anonymously, as before. A malformed block is a
    :class:`ValueError` naming the key.
    """
    provider = sandbox_provider_from_overrides(overrides, sandbox)
    if provider is None:
        return {}
    files = [resolve_sandbox_config(sandbox)] if sandbox is not None else []
    key = f"{SANDBOX_BLOCK_KEY}.{provider}.{SANDBOX_IMAGES_AUTH_KEY}"
    return credentials_for_image_prefixes(_block_from_overrides(overrides, key, config_files=files), key)


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


def pairing_override_from_overrides(overrides: list[str]) -> bool:
    """Whether the run waives the agent admission checks: ``--allow-unsupported-pairing`` (which the CLI translates
    to ``+allow_unsupported_pairing=true``), a configuration file the command names that sets the key, or
    ``NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING``. The same reading as the collector's pairing guard at dispatch
    (:func:`~nemo_gym.config.global_config.pairing_override_enabled`), so the early admission check of ``prepare_target``
    and the guard agree on one explicit override."""
    from omegaconf import OmegaConf

    from nemo_gym.config.global_config import ALLOW_UNSUPPORTED_PAIRING_KEY_NAME, pairing_override_enabled

    block = _block_from_overrides(overrides, ALLOW_UNSUPPORTED_PAIRING_KEY_NAME)
    return pairing_override_enabled(OmegaConf.create({ALLOW_UNSUPPORTED_PAIRING_KEY_NAME: block}))


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


def compose_overlay_from_overrides(overrides: list[str]) -> HarborComposeOverlay | None:
    """The environment server's ``compose_overlay`` block in the configuration the command names
    (``COMPOSE_OVERLAY_KEY``), validated as the server validates it, or ``None`` when the block is absent or null.
    Prepare merges its files onto every task's Compose document and records their images, so the document the
    server starts is the one prepare checked; a block that does not validate is a configuration error here."""
    block = _block_from_overrides(overrides, COMPOSE_OVERLAY_KEY)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(
            f"`{COMPOSE_OVERLAY_KEY}` must be a mapping (files, mcp_servers, prompt_template); got {block!r}"
        )
    try:
        overlay = HarborComposeOverlay.model_validate(block)
    except ValidationError as exc:
        raise ValueError(f"`{COMPOSE_OVERLAY_KEY}`: {exc}") from exc
    return overlay if overlay.configured else None


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
    # the separate verifiers' (declared or built from tests/Dockerfile, the steps' own included) and the Compose
    # services' as recorded.
    images: tuple[str, ...] = ()
    # The dataset's identity (``manifest.toml``'s ``[dataset].digest``) and the digest of its
    # ``dataset.toml``, recorded in every row and in the run config.
    dataset_digest: str | None = None
    dataset_toml_digest: str | None = None
    # The ``[gym]`` table of the dataset's ``dataset.toml`` (empty without one): what the generated
    # ``datasets:`` entry and the run's defaults are built from.
    dataset: GymDatasetConfig = field(default_factory=GymDatasetConfig)
    # The dataset label every row carries as ``task_source``: the dataset folder's name, which is the taskset
    # unless the target is a single task folder inside a dataset (a parent with a ``dataset.toml`` or a
    # ``manifest.toml``, whose table and images it uses); a task folder whose parent is no dataset is labelled
    # by its own name (see ``_dataset_label``); a template's rows carry the template's name, not the ``rendered``
    # folder's. ``None`` (a hand-built taskset) labels the rows with the taskset.
    task_source: str | None = None
    # Where the dataset came from (``manifest.toml``'s ``[dataset].source``: ``harbor-registry``,
    # ``harbor-package-store``, ``git`` or ``local``), for the run summary.
    dataset_source: str | None = None


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
    render_root: Path | None = None,
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
    compose_overlay: HarborComposeOverlay | None = None,
    rows_name: str = "tasks.jsonl",
    registry_credentials: Mapping[str, tuple[str, str]] | None = None,
    agent: str | None = None,
    allow_unsupported_pairing: bool = False,
) -> PreparedTaskset:
    """Fetch (for hub references) and load the tasks, then write the rows file.

    A task folder that does not load is skipped with a warning (and listed in
    ``PreparedTaskset.skipped``) so the rest of the dataset still prepares; so is a task whose
    image fails to build or whose Compose file does not resolve, with the build's or the
    resolver's reason. A dataset that ships ``task_data.py`` is refused as a whole, with the task
    named, when a task's ``task_data.json`` does not satisfy its ``TaskData`` or when the module does
    not import (:mod:`nemo_gym.tasks.harbor.task_data`); both are checked before any image is
    recorded or built, and the loaded tasks are checked again before a row is written.

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
    setting, when the block has none (``None``: unknown, decided at seed). ``compose_overlay`` is the job-level Compose overlay (the server's ``compose_overlay`` block):
    its files are merged onto every task's Compose document, a plain image task included, and its images recorded
    with the tasks'. ``rows_name`` names the rows file inside the taskset's output folder: a run scopes it to its
    agent, so two agents prepared with different selections on one taskset keep their own rows. ``registry_credentials``
    (registry host or image prefix to username and password, :func:`registry_credentials_from_overrides`) are what the base images
    and the Compose images are read from their registries with; without them a private image stops the
    preparation with the registry's refusal.
    A template target (:func:`~nemo_gym.tasks.harbor.parameterized.is_template_folder`) is rendered into
    ``<render_root>/<taskset>/rendered/`` first, ``output_root`` when ``render_root`` is not given. The rendered
    folder is the dataset root (its ``manifest.toml``, ``compose-images.json`` and ``built-images.json`` live in it),
    so every command that prepares one template must render it into one place: ``gym dataset validate`` keeps its
    rows and results under ``results/harbor/validation/`` but renders where ``gym eval run`` does, and an image built
    for the template under one command is reused, not rebuilt and pushed, under the other.

    Before any image is recorded, built or pushed, ``manifest.toml`` beside the task folders is
    reconciled with their digests (:func:`~nemo_gym.tasks.harbor.manifest.check_manifest`): a fetched
    dataset whose folders changed is refused, a local folder gets a manifest. The digests hashed for
    that check are reused for the tasks that do not load; the loader's own folder digests are checked
    against them. Selection does not enter the dataset digest; it covers every task folder, so two runs of
    different subsets still name one dataset. A task folder that cannot be hashed (a file under it escapes
    it through a link, or cannot be read) is recorded in the manifest by name without a digest and does
    not enter the dataset digest; the loader refuses it, so it is skipped like any task that does not load,
    and selection applies to it as to any other task.

    ``agent`` is the harness the run selected (its implementation name, what ``--agent`` resolves to). It is
    checked against the dataset's ``[gym] allowed_agents`` as soon as ``dataset.toml`` is at hand, after a
    hub fetch and before any manifest check, image record, build or server start; a harness the dataset does
    not admit is refused with both named. ``None`` skips the check (the oracle's validation run, which plays
    the reference solution and not a model), and so does ``allow_unsupported_pairing``, the run's pairing override
    (``--allow-unsupported-pairing`` or ``NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING``, :func:`pairing_override_from_overrides`):
    the collector's guard waives the same list at dispatch under it, so an explicit experimental run proceeds
    without editing the dataset's declaration. An ``example`` dataset (``[gym] type``) is refused at the same
    point whatever the harness, before anything is written: that type marks committed smoke-test samples,
    which neither ``gym eval run`` nor ``gym dataset validate`` runs.
    """
    build = build or BuildOptions()
    entry: DatasetEntry | None = None
    if is_hub_ref(target):
        pulled = pull_dataset(target, refresh_registry=refresh_registry)
        print(pulled.fetched_line())
        if pulled.superseded is not None:
            print(pulled.superseded_line())
        folder, entry = pulled.folder, pulled.plan.entry
    elif (entry := entry_target(target)) is not None:
        folder = entry_folder(entry)
    else:
        folder = Path(target).expanduser().resolve()
        entry = entry_at(folder)
        if entry is not None:
            # The path names a committed entry's pull elsewhere than its folder (a moved NEMO_GYM_DATASETS_DIR, or an
            # entry committed outside the working root): the copy of the entry's Gym layer beside the tasks is kept
            # current, or refused as edited in place, exactly as the pull and the entry's name do, so the manifest
            # check below never absorbs an edit of the copy that the next pull would then overwrite with the entry's
            # file.
            prepare_destination(entry, create=False)
    taskset = folder.name
    template = folder if folder.is_dir() and is_template_folder(folder) else None
    if template is not None:
        # A parameterized dataset: the template and its rows become plain task folders under the taskset's output
        # folder, and everything below (manifest, images, loading, rows) reads those files, as the servers will.
        rendered = render_template_folder(template, Path(render_root or output_root) / taskset / "rendered")
        print(f"Rendered {len(iter_task_folders(rendered))} task(s) from {template} into {rendered}")
        folder = rendered
    # Dataset-level files (dataset.toml, the recorded image configurations, the built images) live at the
    # dataset root, which is where the loader and the environment server read them from for nested tasks.
    dataset_root = folder.parent if is_task_folder(folder) else folder
    # A dataset.toml that does not parse is reported per task by the loader below, with every task skipped;
    # the admission check needs a readable table, and the prepared taskset keeps the one it read.
    dataset = _readable_dataset_config(dataset_root)
    # Both refusals come here, where the table is first in hand: before the manifest check, the image records,
    # the builds and the rows file, so a run that cannot happen leaves nothing behind.
    if dataset is not None:
        refuse_example_dataset(dataset)
        if agent is not None and not allow_unsupported_pairing:
            refuse_disallowed_agent(dataset, agent)
    exclude = exclude or []
    # The refusal of a hand-edited fetched dataset comes first, so no image of unverified content is
    # built, pushed or recorded. A single task folder has no manifest of its own (see _dataset_identity).
    digests: dict[str, str] = {}
    try:
        identity = check_manifest(folder, digests=digests) if folder.is_dir() and not is_task_folder(folder) else None
    except ManifestError as exc:
        raise with_entry_hint(exc, entry, folder) from exc
    # The dataset's TaskData (task_data.py beside dataset.toml) is imported once here, before any image is
    # recorded or built: a module that does not import refuses the dataset now, and so does a task folder whose
    # task_data.json the model refuses (the fields need no loaded task, and a dataset refused as a whole should not
    # first resolve, build and push every image). Every loaded task's fields are checked again below, before a row
    # is written. Duplicate task IDs are rejected by the folder walk itself, as they are for the image records.
    task_data_schema = load_task_data_schema(dataset_root) if dataset_root.is_dir() else None
    if task_data_schema is not None and folder.is_dir():
        check_task_folders(task_data_schema, [folder] if is_task_folder(folder) else iter_task_folders(folder))

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
    # One registry client for the base images and the Compose images, for the platform the images are recorded for
    # and with the deployment's credentials; None keeps the recorder's own anonymous client.
    client = (
        RegistryClient(dict(registry_credentials), platform=platform_pair(build.platform))
        if registry_credentials
        else None
    )
    images = (
        record_base_images(
            folder, select=None if is_task_folder(folder) else selected, platform=build.platform, client=client
        )
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
    # A task whose fields do not satisfy the dataset's TaskData refuses the whole dataset, named: the rows would
    # carry data the dataset's own verifier cannot read. Selection does not narrow the check; the dataset is one unit.
    check_tasks(task_data_schema, loaded)
    # A Compose file that does not resolve, a field the sandbox Compose adapter would refuse at seed, or volumes the
    # group cannot run, leaves its task out the same way, with the field named, as an unloadable task is, not as a
    # seed failure of every episode; the others' images are still recorded.
    unresolved: dict[str, str] = {}
    images = record_compose_images(
        tasks,
        dataset_root,
        client=client,
        skipped=unresolved,
        platform=build.platform,
        sandbox_provider=sandbox_provider,
        gpu_sandbox_provider=gpu_sandbox_provider,
        compose_platform=compose_platform,
        volume_sources=volume_sources,
        gpu_volume_sources=gpu_volume_sources,
        shared_storage=shared_storage,
        gpu_shared_storage=gpu_shared_storage,
        overlay=compose_overlay,
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
    # A template's rows are labelled by the template (the taskset), not by the `rendered` folder they were loaded
    # from, which every parameterized dataset shares: the label is what groups and filters rows and results.
    task_source = taskset if template is not None else _dataset_label(folder, dataset_root)
    write_rows(tasks, taskset, rows_path, dataset_digest=identity.digest, task_source=task_source)
    skipped_note = f", skipped {len(errors)}" if errors else ""
    print(f"Materialized {len(tasks)} task(s) from {folder} into {rows_path}{skipped_note}")
    print(f"Dataset digest {identity.digest} ({identity.source})")
    # Every distinct image a run starts, for pre-pulling onto the sandbox hosts before it: the task images, the
    # Compose sidecars' (as recorded, the pinned reference the sandbox gets) and the separate verifiers'.
    distinct = run_images(tasks, read_image_configs(tasks[0].path.parent), overlay=compose_overlay)
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
        dataset=dataset or GymDatasetConfig(),
        task_source=task_source,
        dataset_source=identity.source,
    )


def entry_folder(entry: DatasetEntry) -> Path:
    """The folder a committed entry's target resolves to: where its pull landed (:attr:`DatasetEntry.destination`).

    Nothing is fetched here: ``gym dataset validate <entry>`` and ``gym eval run <entry>`` read what
    ``gym dataset pull <entry>`` left, and the manifest check that follows refuses a folder that no longer
    matches that pull. For an entry that declares a source, the folder must hold that pull's record: task folders
    with no ``manifest.toml``, or with one that records another source (``local``, or the other hub fetcher's) or a
    fetch of another reference than the entry declares now (its ``source`` changed since the pull), are
    refused with the pull to run, never run as a local dataset under the entry's name, since nothing then ties them
    to the declared reference (the pull of a package-store entry keeps and records the folders that still match the
    store, and re-records the reference when only that changed; the registry fetch says in its own words what to move
    aside). When the pull lands elsewhere than the entry's folder (``NEMO_GYM_DATASETS_DIR``, or an entry committed
    outside the working root) the copy of the entry's Gym layer beside the tasks is brought up to date, or refused,
    exactly as the pull would (:func:`~nemo_gym.tasks.harbor.entries.prepare_destination`), so the two commands never
    disagree about one folder. A folder without a pulled task names the pull to run (or, for an entry that declares
    no source, says the
    task folder path is the target). Tasks are looked for as the loader finds them (:func:`iter_task_folders`:
    nested under group folders too), so a layout the folder's path prepares, the entry's name prepares as well.
    """
    folder = prepare_destination(entry, create=False)
    if folder.is_dir() and iter_task_folders(folder):
        if entry.source is not None:
            _check_fetch_record(entry, folder)
        return folder
    if entry.source is None:
        raise HarborTaskError(
            f"Entry {entry.name} has no task folders in {folder} and declares no [gym] source in "
            f"{entry.folder / DATASET_FILE} to pull; pass the folder that holds its tasks instead"
        )
    raise HarborTaskError(
        f"Entry {entry.name} has no task folders in {folder} yet; run `gym dataset pull {entry.name}` "
        f"to fetch {entry.source}"
    )


def _check_fetch_record(entry: DatasetEntry, folder: Path) -> None:
    """Refuse ``folder``'s task folders as ``entry``'s pull unless its manifest records a fetch of the entry's source.

    The manifest's ``source`` says which fetcher wrote it and its ``ref`` which reference the fetch was made from
    (:func:`~nemo_gym.tasks.harbor.hub.declares_recorded_fetch` reads the declaration against it); both are checked
    offline, so an entry whose ``source`` moved to another tag since the pull names the pull instead of running the
    tasks of the tag it had. A manifest that records no reference (written before it was recorded) names the pull
    too, and the pull records it without fetching a folder that still matches.
    """
    try:
        recorded = Manifest.read(folder)
    except ManifestError:
        # Malformed, or written by a newer Gym: the manifest check that follows refuses it in its own words.
        return
    if recorded is None:
        raise HarborTaskError(
            f"Entry {entry.name}'s folder {folder} holds task folders but no fetch record ({MANIFEST_FILE}), so they "
            f"cannot be run as its pull of {entry.source}; run `gym dataset pull {entry.name}` to fetch it"
        )
    expected = PACKAGE_STORE_SOURCE if HubRef.parse(entry.source).is_package else REGISTRY_SOURCE
    if recorded.source != expected:
        raise HarborTaskError(
            f"Entry {entry.name}'s folder {folder} holds task folders whose {MANIFEST_FILE} records source "
            f"{recorded.source!r}, not a fetch of {entry.source} ({expected}); delete, or move aside, the task folders "
            f"and {MANIFEST_FILE} there and run `gym dataset pull {entry.name}`"
        )
    if recorded.ref is None or not declares_recorded_fetch(entry.source, recorded.ref):
        held = (
            f"a fetch of {recorded.ref}" if recorded.ref is not None else "a fetch whose reference it does not record"
        )
        raise HarborTaskError(
            f"Entry {entry.name}'s folder {folder} holds task folders whose {MANIFEST_FILE} records {held}, not of "
            f"{entry.source}, which {entry.folder / DATASET_FILE} declares now; run `gym dataset pull {entry.name}` to "
            f"fetch it (a folder holding another version's tasks is refused there, with what to move aside)"
        )


# How check_manifest's refusal of an interrupted fetch ends (nemo_gym.tasks.harbor.manifest).
_FETCH_AGAIN = "run the fetch again"


def with_entry_hint(exc: Exception, entry: DatasetEntry | None, folder: Path) -> Exception:
    """The fetchers' and the manifest check's refusals name the dataset folder and say to delete it, or move it aside,
    and run again. When that folder is a committed entry's own (not the hub's folder, nor the entry's pull elsewhere
    than its folder, under a moved ``NEMO_GYM_DATASETS_DIR`` or for an entry committed outside the working root,
    which hold nothing git tracks), the same refusal says what to delete there: the pulled task folders and
    ``manifest.toml``, not the README, YAML files and ``dataset.toml`` the entry commits, and that another version is
    pulled by changing ``source`` in that ``dataset.toml``. The manifest check's refusal
    of an interrupted fetch (``run the fetch again``) names the pull to run again instead, wherever the folder is:
    the fetch finishes it, and deleting what it landed would throw the partial fetch away. Returns ``exc`` itself for
    every other folder, and when the message does not name the folder."""
    if entry is None or str(folder) not in str(exc):
        return exc
    if str(exc).endswith(_FETCH_AGAIN):
        return type(exc)(f"{str(exc).removesuffix(_FETCH_AGAIN)}run `gym dataset pull {entry.name}` again")
    if folder != entry.folder:
        return exc
    return type(exc)(
        f"{exc}. {folder} is the committed entry {entry.name}'s own folder: delete, or move aside, only the pulled "
        f"task folders and {MANIFEST_FILE} in it; its README, YAML files and {DATASET_FILE} are the entry's and stay "
        f"(to pull another version, change `source` in that {DATASET_FILE})"
    )


@dataclass(frozen=True)
class PulledDataset:
    """What ``gym dataset pull`` fetched: the reference, where it landed and the manifest left there, current."""

    target: str
    plan: PullPlan
    folder: Path
    manifest: Manifest
    # The folder's manifest read the same before the fetch: nothing was downloaded, replaced or rewritten.
    unchanged: bool
    # The hub's own folder for the reference (``terminal-bench-4.0.0/``), still present beside the entry the
    # reference now lands in: a fetch from before the entry declared it, which nothing reads any more.
    superseded: Path | None = None

    @property
    def entry(self) -> str | None:
        return self.plan.entry.name if self.plan.entry is not None else None

    @property
    def task_count(self) -> int:
        return len(self.manifest.tasks)

    def fetched_line(self) -> str:
        where = f" (entry {self.entry})" if self.entry is not None else ""
        return f"Fetched {self.plan.ref} into {self.folder}{where}"

    def superseded_line(self) -> str:
        return (
            f"Note: {self.superseded} holds an earlier fetch of {self.plan.ref}; fetches of it now land in "
            f"{self.folder} and nothing reads the old folder, so it can be moved aside or deleted"
        )

    def text(self) -> str:
        """The command's report: where the dataset landed, how many tasks, its digest and the next command."""
        suffix = ", unchanged since the last pull" if self.unchanged else ""
        next_target = self.entry if self.entry is not None else shlex.quote(str(self.folder))
        lines = [
            self.fetched_line().replace("Fetched", "Pulled", 1),
            f"{self.task_count} task(s), dataset digest {self.manifest.digest} ({self.manifest.source}){suffix}",
        ]
        if self.superseded is not None:
            lines.append(self.superseded_line())
        lines.append(f"Next: gym dataset validate {next_target} --sandbox <provider>")
        return "\n".join(lines)


def pull_dataset(target: str, *, refresh_registry: bool = False) -> PulledDataset:
    """Fetch ``target`` (a ``harbor:`` reference or a committed entry's name) and report where it landed.

    The destination comes from :func:`~nemo_gym.tasks.harbor.entries.plan_pull`: the entry that declares the
    reference as its source (its own folder, or ``<datasets dir>/<entry>/`` with its ``dataset.toml`` placed first
    when ``NEMO_GYM_DATASETS_DIR`` or an entry committed outside the working root moved the destination,
    :attr:`~nemo_gym.tasks.harbor.entries.DatasetEntry.destination`), else the hub's own folder under the datasets
    dir. The fetch is the one every
    ``gym eval run harbor:...`` makes, so a stale, edited or foreign folder is refused in the same words (a refusal
    naming a committed entry's own folder says what to delete there, :func:`with_entry_hint`); nothing is prepared,
    built or run. The manifest left after the fetch gives the task count and the dataset digest; its Gym-layer
    digests are first brought up to date with the folder (:func:`~nemo_gym.tasks.harbor.manifest.refresh_gym_layer`),
    since a fetch that downloads nothing rewrites nothing and would otherwise report the digest of a ``dataset.toml``
    that changed since the last pull, which the next prepare would then rewrite. ``unchanged`` compares the manifest
    on disk before anything was touched with the one reported; a registry reference whose version the registry picks
    is resolved first (:func:`~nemo_gym.tasks.harbor.hub.resolved_folder`) so that folder is known. When the Gym
    layer's digests moved while the task record stayed (the entry's ``dataset.toml`` changed, or one written by hand
    into a hub folder), the rewrite is announced in the manifest check's words whichever step applied it: the
    refresh above, or the fetch itself, which rewrites the manifest without a word when every folder is in place
    (the package store's does). A hub folder the reference landed in before an entry declared it
    (``terminal-bench-4.0.0/`` beside ``terminal_bench_4/``) is left alone and named in the report, since nothing
    reads it any more. ``refresh_registry`` downloads the registry again instead of reading
    ``.harbor/registry.json``; it is :func:`prepare_target`'s pass-through, and deleting that cache file does the
    same for the command.
    """
    plan = plan_pull(target)
    if plan is None:
        raise HubError(
            f"{target!r} is neither a `harbor:` reference nor a committed entry under environments/harbor/; "
            "pass `harbor:<org>/<name>[@<tag> | @sha256:<digest>]`, `harbor:<dataset>[@<version>]` or an entry name"
        )
    hub_folder = expected_folder(plan.ref)
    if hub_folder is None:
        # The registry picks the version: resolve it now, against the cache the fetch reads (refreshed here when
        # asked, once), so the folder's manifest can be read before the fetch.
        hub_folder = resolved_folder(plan.ref, refresh_registry=refresh_registry)
        refresh_registry = False
    target_folder = plan.folder if plan.folder is not None else hub_folder
    before = _manifest_before(target_folder)
    # prepare_destination announces the refresh it makes itself; the fetch is measured from what it found.
    folder = prepare_destination(plan.entry) if plan.entry is not None else None
    staged = _manifest_before(target_folder)
    try:
        fetched = fetch_ref(plan.ref, folder=folder, refresh_registry=refresh_registry)
    except (HubError, ManifestError) as exc:
        raise with_entry_hint(exc, plan.entry, folder) from exc
    landed = _manifest_before(fetched)
    manifest = refresh_gym_layer(fetched)
    if manifest is None:
        raise HubError(f"{fetched} has no {MANIFEST_FILE} after the fetch; run the pull again")
    if (
        staged is not None
        and landed is not None
        and landed.text() == manifest.text()
        and staged.digest != landed.digest
        and (staged.source, staged.tasks) == (landed.source, landed.tasks)
    ):
        # The fetch rewrote the manifest with the same task record and the digest moved: only the Gym layer changed,
        # and the refresh above found nothing left to do, so it is announced here (a rewrite that only re-recorded
        # the reference moves no digest and announces nothing: the pull's report names the reference).
        report_rewrite(fetched, staged, landed, gym_layer_change(staged, landed))
    superseded = hub_folder if plan.entry is not None and hub_folder.is_dir() else None
    return PulledDataset(
        target=target,
        plan=plan,
        folder=fetched,
        manifest=manifest,
        unchanged=before is not None and before.text() == manifest.text(),
        superseded=superseded if superseded != fetched else None,
    )


def _manifest_before(folder: Path | None) -> Manifest | None:
    """The manifest in ``folder`` as it stands, or ``None``; the fetch itself reports an unreadable one."""
    if folder is None or not folder.is_dir():
        return None
    try:
        return Manifest.read(folder)
    except ManifestError:
        return None


def pull_target(args: argparse.Namespace, overrides: list[str]) -> None:
    """Entry point for ``gym dataset pull <target>``."""
    if overrides:
        args._parser.error(f"`gym dataset pull` takes no configuration overrides: {' '.join(overrides)}")
    try:
        pulled = pull_dataset(args.target)
    except (HubError, ManifestError, HarborTaskError, ValueError, OSError) as exc:
        args._parser.error(str(exc))
    print(pulled.text())


def _dataset_label(folder: Path, dataset_root: Path) -> str:
    """The dataset label the rows of ``folder`` carry as ``task_source``.

    A dataset folder target is labelled by its own name, the taskset. A single task folder target is labelled by
    its parent's name when the parent is a dataset: it holds a ``dataset.toml`` or a ``manifest.toml``, so the
    table, image records and digests the preparation uses are the parent's, and two single-task runs of one
    dataset carry one label. A task folder whose parent holds neither is a standalone task (the parent is only
    the directory that happens to hold it: a scratch folder, a user's home) and is labelled by its own name, as a
    hand-built taskset is, so two unrelated task folders prepared from one parent do not share a label.
    """
    if folder == dataset_root:
        return dataset_root.name
    if (dataset_root / DATASET_FILE).is_file() or (dataset_root / MANIFEST_FILE).is_file():
        return dataset_root.name
    return folder.name


def _readable_dataset_config(dataset_root: Path) -> GymDatasetConfig | None:
    """The dataset's ``[gym]`` table, or ``None`` when ``dataset.toml`` does not parse (the loader reports that)."""
    try:
        return read_dataset_config(dataset_root)
    except ValueError:
        return None


def refuse_example_dataset(dataset: GymDatasetConfig) -> None:
    """Refuse a dataset whose ``[gym] type`` is ``example``: that type marks committed smoke-test samples, which
    no split runs.

    ``prepare_target`` raises it as soon as ``dataset.toml`` is read, before any manifest check, image record,
    build or rows file; ``build_run`` raises it again as a backstop for a hand-built :class:`PreparedTaskset`.
    """
    if dataset.effective_type != "example":
        return
    raise HarborTaskError(
        f'{dataset.path or DATASET_FILE} [gym] type = "example" marks smoke-test samples, which '
        "`gym eval run` and `gym dataset validate` do not run; set type to benchmark, validation or train"
    )


def refuse_disallowed_agent(dataset: GymDatasetConfig, agent: str) -> None:
    """Refuse a harness the dataset's ``[gym] allowed_agents`` does not admit, naming the harness and the list.

    Raised before any fetch result is built on, any image recorded or built, or any server or sandbox started.
    """
    if dataset.admits(agent):
        return
    admitted = ", ".join(dataset.allowed_agents or ())
    raise ValueError(
        f"{dataset.path or DATASET_FILE} [gym] allowed_agents admits {admitted}, not `--agent {agent}`. "
        f"Run it with one of those harnesses, or add {agent!r} to allowed_agents in the dataset."
    )


def run_images(
    tasks: list[HarborTask], image_configs: Mapping[str, Any], *, overlay: HarborComposeOverlay | None = None
) -> tuple[str, ...]:
    """Every distinct image a run of ``tasks`` starts, sorted: each task's image, the image of a separate
    verifier (``[verifier.environment].docker_image``), the image of each step's separate verifier
    (:attr:`HarborTurn.verifier_image`) and every Compose service image (the job-level ``overlay``'s services
    included), the last as the recorded configuration pins it (``compose-images.json``, what ``resolve_compose``
    hands the sandbox), else as written when it has no record."""
    images: set[str] = set()
    for task in tasks:
        if task.image:
            images.add(task.image)
        if task.verifier_image:
            # The separate verifier's sandbox starts this image: the one prepare built from tests/Dockerfile
            # (recorded under the tests/ digest, never a Compose service, so nothing below would name it) or the
            # declared one, which the check below names too (HarborTask.verifier_image covers both).
            images.add(task.verifier_image)
        for turn in task.turns:
            # A step that verifies separately starts its own verifier image (the server runs the turn with it in
            # place of the task's, `_turn_task`): a declared [steps.verifier.environment].docker_image, the task's
            # declared one, or the tests/ build the task's own shared verifier never asked for. A shared step, or a
            # separate one that inherits the task image, has none.
            if turn.verifier_image:
                images.add(turn.verifier_image)
        verifier_environment = task.config.verifier.environment
        if (
            not task.config.is_shared_verifier
            and verifier_environment is not None
            and verifier_environment.docker_image
        ):
            images.add(verifier_environment.docker_image)
        document = load_compose_document(task, overlay=overlay)
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
        files_digest = dataset_files_digest(folder.parent)
        return Manifest(
            source=LOCAL_SOURCE,
            digest=dataset_digest(hashed, toml_digest, files_digest),
            dataset_toml_digest=toml_digest,
            tasks={name: {"digest": digest} for name, digest in hashed.items()},
            dataset_files_digest=files_digest,
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
    from nemo_gym.server_runtime.entrypoint_flags import entrypoint_class_flag

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
    """Write ``run_config.yaml`` and return it with the Hydra overrides for collection.

    The dataset's ``[gym]`` table sets the run's defaults: the split the collector runs is the entry's
    ``type`` (``+split=`` on the command line still wins), and the run config's top-level ``num_repeats``
    is ``[gym.defaults].num_repeats``, below every config file and command-line override, so
    ``+num_repeats=`` replaces it. The count lives at the run level and not in the entry: collate would
    duplicate rows by an entry count and the collector would multiply on top, and the metrics sidecar
    beside the rows records the entry, so a moving entry count would conflict run to run. The oracle's
    validation run is not an evaluation: ``validate_target`` adds ``+num_repeats=1`` unless the command
    line sets a count, so each reference solution plays once. An ``example`` dataset is refused here again
    (:func:`refuse_example_dataset`; ``prepare_target`` refused it before anything was written), so a hand-built
    taskset of smoke-test samples does not run either.

    The entry carries the dataset's ``allowed_agents``, and the collector's pairing guard checks the agent
    block's harness against it at dispatch. The oracle plays the reference solution, not a model, and is
    always admitted (``prepare_target`` checks no agent for it), so its run config carries the entry without
    the list: the list speaks of model harnesses, and the guard would otherwise refuse the validation run.
    """
    refuse_example_dataset(prepared.dataset)
    split = prepared.dataset.effective_type
    dataset = prepared.dataset
    if agent.impl_name == ORACLE_AGENT and dataset.allowed_agents is not None:
        dataset = dataset.model_copy(update={"allowed_agents": None})
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
        dataset=dataset,
    )
    # Defaults belong below explicit config files and command-line overrides.
    config["output_jsonl_fpath"] = str(prepared.output_dir / "rollouts.jsonl")
    config["num_repeats"] = prepared.dataset.defaults.num_repeats
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
        tokens.append(f"+split={split}")
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
            compose_overlay=compose_overlay_from_overrides(overrides),
            rows_name=_rows_name(agent),
            registry_credentials=registry_credentials_from_overrides(overrides, sandbox),
            agent=agent.impl_name,
            allow_unsupported_pairing=pairing_override_from_overrides(overrides),
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
    helper = RolloutCollectionHelper()
    expected: list[dict[str, Any]] | None = None
    before_dispatch = None
    cache = _resumable_cache(collection)
    if cache is not None:
        # The collector resumes its saved inputs, not the rows just prepared: a selection or digest that
        # changed since would run (or 404 on) tasks the server no longer maps. Refuse before startup.
        expected = helper._preprocess_rows_from_config(collection)
        conflict = _resume_conflict(cache, expected)
        if conflict:
            args._parser.error(conflict)
        # The collector resumes the schedule it saved (the materialized inputs: every task and repeat as that run
        # expanded them, with the input each carried), not the rows the current config expands: a resume with
        # another `num_repeats` passes the check above and dispatches no further attempts, so the summary's
        # expected attempts and input echo are the saved ones, not a missing attempt the run never owed.
        expected = _validation_rows(cache)
    else:
        # A fresh start clears its output files only once the servers are up: what sits under those names now
        # is an earlier run's, and the summary must not report it as this run's if the collector never gets there.
        before_dispatch = output_signatures(Path(collection.output_jsonl_fpath))
        # The collated split metrics an earlier run left are that run's selection; the data processor would
        # refuse this one against them. Cleared here, they are rewritten before any server starts.
        _clear_collector_metrics(Path(collection.output_jsonl_fpath), split=_collector_split())
    print(f"Run config written to {config_path} (datasets folder: {datasets_dir()})")
    try:
        dispatch("nemo_gym.cli.eval:e2e_rollout_collection", tokens)
    finally:
        _write_summary(
            Path(collection.output_jsonl_fpath),
            command="gym eval run",
            target=args.target,
            prepared=prepared,
            agent=agent,
            run_config=config_path,
            # A fresh run reads its selected rows only for the summary, after the collector has had them.
            expected=expected if expected is not None else lambda: helper._preprocess_rows_from_config(collection),
            split=_collector_split(),
            before_dispatch=before_dispatch,
        )


def _collector_split() -> str | None:
    """The split the resolved run config dispatches (``validation`` unless overridden), for the run summary."""
    from nemo_gym.config.global_config import get_global_config_dict

    split = get_global_config_dict().get("split")
    return str(split) if split else None


def _clear_collector_metrics(rollouts: Path, *, split: str | None) -> None:
    """Remove the collated split metrics an earlier run left under ``<rollouts stem>/preprocessed_datasets/``
    (``<split>_metrics.json`` and the ``<split>_metrics_conflict.json`` of a refused run), right before a fresh
    start dispatches the collector.

    ``e2e_rollout_collection`` collates the rows it dispatches there and checks the metrics it computes against
    that file, refusing a mismatch ("Found conflicting aggregate metrics") unless ``overwrite_metrics_conflicts``
    is set. That protects a committed source dataset; on this path the rows are regenerated from the task folder
    on every command and the output folder is reused per taskset, so the file is the previous selection's, and
    a user must neither see that error nor pass the flag. A resumed run keeps the file: its rows are unchanged
    (:func:`_resume_conflict`). So does a run that reuses the prepared input (``reuse_existing_data_preparation``
    with the collated ``<split>.jsonl`` present): the processor does not run, and the metrics stay with the input
    they describe. ``split`` is the resolved run config's (:func:`_collector_split`); without one nothing is known
    to clear.
    """
    from nemo_gym.config.global_config import get_global_config_dict
    from nemo_gym.tasks.harbor.run_summary import PREPROCESSED_SUBDIR

    if not split:
        return
    preprocessed = Path(rollouts).with_suffix("") / PREPROCESSED_SUBDIR
    reuses_input = bool(get_global_config_dict().get("reuse_existing_data_preparation"))
    if reuses_input and (preprocessed / f"{split}.jsonl").is_file():
        return
    metrics = preprocessed / f"{split}_metrics.json"
    for path in (metrics, metrics.with_name(f"{metrics.stem}_conflict.json")):
        path.unlink(missing_ok=True)


def _write_summary(
    rollouts: Path, *, expected: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]] | None, **kwargs: Any
) -> None:
    """Write the run summary beside ``rollouts`` (:mod:`nemo_gym.tasks.harbor.run_summary`) and say where.

    Runs after collection whatever it did, so a run that stopped early still leaves a summary of what it
    wrote; a summary that cannot be written is logged and never displaces the run's own outcome. ``expected``
    may be a callable that reads the collector's selected rows, so a run that did not need them before
    dispatch reads them here, inside the same guard. ``before_dispatch`` (in ``kwargs``) is what a fresh start
    found under its output names before dispatch, so files it never got to clear are not reported as its own.
    """
    try:
        if callable(expected):
            expected = expected()
        path = write_run_summary(rollouts, expected=expected, **kwargs)
    except Exception:
        logger.warning("Could not write the run summary beside %s", rollouts, exc_info=True)
        return
    print(f"Run summary written to {path}")


def _rows_name(agent: AgentSelection) -> str:
    """The rows file of one agent's run on a taskset: ``tasks_<agent>.jsonl`` next to ``run_config_<agent>.yaml``."""
    return f"tasks_{agent.instance_name}.jsonl"


def _resolved_collection(tokens: list[str], prepared: PreparedTaskset) -> "RolloutCollectionConfig":
    """The collector's effective configuration for ``tokens``: resolved through the same parser and cache
    dispatch uses, file overlays included, so what it would resume and run is read from it rather than
    guessed from the command line."""
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationResolutionError

    from nemo_gym.config.global_config import (
        get_global_config_dict,
        reset_global_config_dict,
        translate_interpolation_error,
    )
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


def _requested_output(tokens: list[str], default: Path) -> Path:
    """The rollouts path ``tokens`` ask for (``--output``, ``+output_jsonl_fpath=`` or a ``--config`` overlay), or
    ``default`` when they name none.

    Resolved through the same parser dispatch uses, so a quoted ``--output`` and a file overlay both count, for
    the path that dispatches nothing and so has no run config to resolve a collection from.
    """
    from nemo_gym.config.global_config import get_global_config_dict, reset_global_config_dict

    sys.argv = [sys.argv[0], *tokens]
    reset_global_config_dict()
    requested = get_global_config_dict().get("output_jsonl_fpath")
    return Path(str(requested)) if requested else default


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
            # A template renders where `gym eval run` renders it: one rendered dataset, one set of records (an image
            # built for it once), while the oracle's rows and results stay under validation/.
            render_root=DEFAULT_OUTPUT_ROOT,
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
            compose_overlay=compose_overlay_from_overrides(overrides),
            rows_name=_rows_name(agent),
            registry_credentials=registry_credentials_from_overrides(overrides, sandbox),
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
    # Left out of the run, not failed: the summary names them with the reason so the result folder explains itself.
    left_out = dict.fromkeys(unvalidated, NO_REFERENCE_SOLUTION)
    prepared = replace(prepared, tasks=[task for task in prepared.tasks if task.has_solution])
    write_rows(
        prepared.tasks,
        prepared.taskset,
        prepared.rows_path,
        dataset_digest=prepared.dataset_digest,
        task_source=prepared.task_source,
    )
    rollouts = prepared.output_dir / "rollouts.jsonl"
    expected = []
    if prepared.tasks:
        # Oracle runs reference scripts directly; it needs no placeholder model server.
        overrides = list(overrides)
        # Validation is not an evaluation: a reference solution scores the same every time, so the dataset's
        # `[gym.defaults].num_repeats` does not apply to it. An explicit `+num_repeats=` still wins.
        if not _has_override(overrides, "num_repeats"):
            overrides.append("+num_repeats=1")
        config_path, tokens = build_run(prepared, agent, sandbox=sandbox, overrides=overrides)
        tokens = _merge_config_paths(tokens)
        # Resolve through the same parser and cache used by dispatch, including file overlays.
        collection = _resolved_collection(tokens, prepared)
        expected = RolloutCollectionHelper()._preprocess_rows_from_config(collection)
        rollouts = Path(collection.output_jsonl_fpath)
        before_dispatch = None
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
            # What the collector resumes is its saved schedule (see run_target): the report and the summary
            # expect the saved attempts, with the input each carried.
            expected = _validation_rows(cache)
        else:
            # See run_target: a fresh start's files under the output names are an earlier run's until the
            # collector clears them, which happens only once the servers are up; the collated metrics of an
            # earlier selection would make the data processor refuse this one, so they go now.
            before_dispatch = output_signatures(rollouts)
            _clear_collector_metrics(rollouts, split=_collector_split())
        print(f"Run config written to {config_path}; validating {len(expected)} oracle rollout(s)")
        try:
            dispatch("nemo_gym.cli.eval:e2e_rollout_collection", tokens)
        finally:
            _write_summary(
                rollouts,
                command="gym dataset validate",
                target=args.target,
                prepared=prepared,
                agent=agent,
                run_config=config_path,
                expected=expected,
                split=_collector_split(),
                before_dispatch=before_dispatch,
                unvalidated=left_out,
            )
    else:
        # Nothing to dispatch (no selected task has a solution/): the summary still says what was selected,
        # beside the rollouts file the command line asked for (there is no run config to resolve it from).
        rollouts = _requested_output(_merge_config_paths(list(overrides)), rollouts)
        _write_summary(
            rollouts,
            command="gym dataset validate",
            target=args.target,
            prepared=prepared,
            agent=agent,
            run_config=None,
            expected=expected,
            # Nothing ran, so whatever sits under the output names is an earlier run's.
            before_dispatch=output_signatures(rollouts),
            unvalidated=left_out,
        )
    report = summarize_validation(rollouts, unvalidated, expected=expected, skipped=prepared.skipped)
    print(report.text)
    if not report.ok:
        sys.exit(1)
