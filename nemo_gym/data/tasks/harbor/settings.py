# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The settings a Harbor preparation reads from the configuration a command names, before the run config exists.

`gym eval run` and the in-process API (`nemo_gym.environment.harbor_benchmark`) prepare a Harbor target with the same
settings: the `sandbox:` block's provider, credentials and volume sources, the environment server's platform and Compose
overlay, and the pairing override. Each reader takes the command's override tokens (and the `--sandbox` provider name)
and reads only the keys it needs, so a provider block's credentials are not needed to read a sibling key.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from nemo_gym.data.tasks.harbor.build_options import (
    _block_from_overrides,
    _merged_cli_config,
    build_options_from_overrides,
)
from nemo_gym.data.tasks.harbor.compose_overlay import HarborComposeOverlay
from nemo_gym.data.tasks.harbor.materialize import ENVIRONMENT_SERVER_IMPL, ENVIRONMENT_SERVER_INSTANCE
from nemo_gym.data.tasks.harbor.prepare import resolve_sandbox_config
from nemo_gym.data.tasks.harbor.registry import credentials_for_image_prefixes
from nemo_gym.sandbox.compose_config import deployment_platform
from nemo_gym.sandbox.config import SANDBOX_BLOCK_VOLUME_SOURCES_KEY, validate_volume_sources


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
    (:func:`~nemo_gym.data.tasks.harbor.registry.credentials_for_image_prefixes`), so each image is read with the
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


def preparation_settings(overrides: list[str], sandbox: str | None) -> dict[str, Any]:
    """The keyword arguments of :func:`~nemo_gym.data.tasks.harbor.prepare.prepare_target` that the configuration a
    command names decides: the ``builds:`` block, the sandbox providers and what they can apply or mount, the
    registry credentials and the Compose overlay.

    ``gym eval run``, ``gym eval prepare``, ``gym dataset validate`` and the in-process API prepare a target with the
    same settings, so each passes these along with what is its own (selection, output folder, rows file, agent).
    """
    return {
        "build": build_options_from_overrides(overrides),
        "sandbox_provider": sandbox_provider_from_overrides(overrides, sandbox),
        "gpu_sandbox_provider": gpu_sandbox_provider_from_overrides(overrides, sandbox),
        "compose_platform": compose_platform_from_overrides(overrides),
        "volume_sources": volume_sources_from_overrides(overrides, sandbox),
        "gpu_volume_sources": gpu_volume_sources_from_overrides(overrides, sandbox),
        "shared_storage": shared_storage_from_overrides(overrides, sandbox),
        "gpu_shared_storage": gpu_shared_storage_from_overrides(overrides, sandbox),
        "compose_overlay": compose_overlay_from_overrides(overrides),
        "registry_credentials": registry_credentials_from_overrides(overrides, sandbox),
    }
