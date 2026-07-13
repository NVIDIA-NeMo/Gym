# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve a sandbox provider reference into a provider config.

An agent selects a sandbox by name (``sandbox_provider: sandbox``). The named
block lives in its own provider config file, so swapping providers is swapping a
``config_paths`` entry, not editing the agent config::

    # nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml
    sandbox:
      opensandbox:
        connection: { ... }

    # agent config
    sandbox_provider: sandbox

An inline single-key mapping (``{provider_name: {...}}``) is also accepted for
keeping everything in one file.

A block may also carry a reserved ``default_metadata`` key. Its entries are merged
into the sandbox's ``SandboxSpec.metadata`` as defaults (the agent's own
``sandbox_spec.metadata`` overrides them), so provider-identifying tags live with
the provider rather than the agent config::

    sandbox:
      opensandbox: { connection: { ... } }
      default_metadata: { sandbox-api: opensandbox-sdk }
"""

from collections.abc import Mapping
from typing import Any

from nemo_gym.sandbox.providers import SandboxResources, SandboxSpec


# Reserved keys inside a sandbox block that are not the provider config.
SANDBOX_BLOCK_DEFAULT_METADATA_KEY = "default_metadata"
SANDBOX_BLOCK_RESERVED_KEYS = frozenset({SANDBOX_BLOCK_DEFAULT_METADATA_KEY})


def _to_plain_dict(value: Any) -> Any:
    """Return a plain ``dict`` for mappings, including OmegaConf ``DictConfig``."""
    try:
        from omegaconf import DictConfig, OmegaConf
    except ImportError:  # pragma: no cover - omegaconf is a core dependency
        DictConfig = ()  # type: ignore[assignment]
        OmegaConf = None  # type: ignore[assignment]

    if OmegaConf is not None and isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _provider_keys(block: Mapping[str, Any]) -> list[str]:
    """Return the provider keys in a block (everything but reserved keys)."""
    return [key for key in block if key not in SANDBOX_BLOCK_RESERVED_KEYS]


def _candidate_sandbox_names(named_configs: Mapping[str, Any] | None) -> list[str]:
    """List top-level config keys that look like named sandbox provider blocks."""
    if not named_configs:
        return []
    candidates: list[str] = []
    for key, value in named_configs.items():
        plain = _to_plain_dict(value)
        if isinstance(plain, Mapping) and len(_provider_keys(plain)) == 1:
            candidates.append(str(key))
    return sorted(candidates)


def _resolve_block(
    sandbox_provider: str | Mapping[str, Any],
    named_configs: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    """Resolve a ``sandbox_provider`` reference or inline mapping to a plain block."""
    if isinstance(sandbox_provider, str):
        name = sandbox_provider
        if not name:
            raise ValueError("Sandbox provider reference must be a non-empty string")
        block = named_configs.get(name) if named_configs is not None else None
        if block is None:
            available = ", ".join(repr(n) for n in _candidate_sandbox_names(named_configs)) or "(none)"
            raise ValueError(
                f"Sandbox provider reference {name!r} is not defined in the merged config. "
                f"Define a top-level '{name}:' block (e.g. via "
                f"nemo_gym/sandbox/providers/<provider>/configs/<provider>.yaml) and include it in "
                f"your config_paths. Available sandbox configs: {available}"
            )
        return _to_plain_dict(block), f"reference {name!r}"
    if isinstance(sandbox_provider, Mapping):
        return _to_plain_dict(sandbox_provider), "inline sandbox_provider config"
    raise TypeError(
        "sandbox_provider must be a name reference (str) or a single-key provider mapping, "
        f"got {type(sandbox_provider).__name__}"
    )


def resolve_provider_config(
    sandbox_provider: str | Mapping[str, Any],
    named_configs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a ``sandbox_provider`` field into a single-key provider config dict.

    Args:
        sandbox_provider: Either the name of a top-level sandbox config block
            (resolved from ``named_configs``) or an inline single-key provider
            mapping of the form ``{provider_name: {...}}``.
        named_configs: Mapping of top-level config name to config block, typically
            the merged global config dict. Required when ``sandbox_provider`` is a
            name reference.

    Returns:
        A plain ``{provider_name: provider_kwargs}`` dict suitable for
        :func:`nemo_gym.sandbox.create_provider`. Reserved keys such as
        ``default_metadata`` are excluded; read them with
        :func:`resolve_provider_metadata`.

    Raises:
        TypeError: If ``sandbox_provider`` is neither a string nor a mapping.
        ValueError: If a named reference cannot be found, or if the block does not
            hold exactly one provider key.
    """
    block, source = _resolve_block(sandbox_provider, named_configs)
    if not isinstance(block, Mapping):
        raise ValueError(f"Sandbox provider config from {source} must be a mapping, got: {block!r}")

    provider_keys = _provider_keys(block)
    if len(provider_keys) != 1:
        raise ValueError(
            f"Sandbox provider config from {source} must have exactly one provider key "
            f"{{provider_name: config}}, got keys: {provider_keys!r}"
        )

    return {provider_keys[0]: block[provider_keys[0]]}


def resolve_provider_metadata(
    sandbox_provider: str | Mapping[str, Any],
    named_configs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a sandbox block's ``default_metadata``.

    These are provider-contributed defaults to merge into ``SandboxSpec.metadata``.
    Returns an empty dict when the block has no ``default_metadata`` key. See
    :func:`resolve_provider_config` for argument semantics.
    """
    block, source = _resolve_block(sandbox_provider, named_configs)
    if not isinstance(block, Mapping):
        raise ValueError(f"Sandbox provider config from {source} must be a mapping, got: {block!r}")

    metadata = block.get(SANDBOX_BLOCK_DEFAULT_METADATA_KEY) or {}
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"Sandbox '{SANDBOX_BLOCK_DEFAULT_METADATA_KEY}' from {source} must be a mapping, got: {metadata!r}"
        )
    return dict(metadata)


def sandbox_spec_from_mapping(
    spec_config: Mapping[str, Any],
    *,
    default_workdir: str | None = None,
    metadata: Mapping[str, str] | None = None,
    default_metadata: Mapping[str, str] | None = None,
) -> SandboxSpec:
    """Build a validated ``SandboxSpec`` from a plain configuration mapping.

    Provider metadata is applied as defaults, explicit ``sandbox_spec.metadata``
    overrides provider defaults, and caller metadata has final precedence.
    """

    plain_config = _to_plain_dict(spec_config)
    if not isinstance(plain_config, Mapping):
        raise TypeError(f"SandboxSpec config must be a mapping, got {type(plain_config).__name__}")
    config = dict(plain_config)
    configured_metadata = _string_mapping(config.pop("metadata", {}), "metadata")
    merged_metadata = _string_mapping(
        {**(default_metadata or {}), **configured_metadata, **(metadata or {})},
        "metadata",
    )
    image = config.pop("image", None)
    if image is not None and not isinstance(image, str):
        raise TypeError(f"SandboxSpec image must be a string or null, got {type(image).__name__}")
    ttl_s = _positive_number_or_none(config.pop("ttl_s", None), "ttl_s")
    ready_timeout_s = _positive_number_or_none(config.pop("ready_timeout_s", None), "ready_timeout_s")
    workdir = config.pop("workdir", default_workdir)
    if workdir is not None and not isinstance(workdir, str):
        raise TypeError(f"SandboxSpec workdir must be a string or null, got {type(workdir).__name__}")
    env = _string_mapping(config.pop("env", {}), "env")
    files = _string_mapping(config.pop("files", {}), "files")
    resources = SandboxResources.from_mapping(config.pop("resources", {}))
    entrypoint = config.pop("entrypoint", None)
    if entrypoint is not None and (
        not isinstance(entrypoint, (list, tuple)) or not all(isinstance(part, str) for part in entrypoint)
    ):
        raise TypeError("SandboxSpec entrypoint must be a list of strings or null")
    entrypoint = list(entrypoint) if entrypoint is not None else None
    provider_options_value = config.pop("provider_options", {})
    if not isinstance(provider_options_value, Mapping):
        raise TypeError(f"SandboxSpec provider_options must be a mapping, got {type(provider_options_value).__name__}")
    provider_options = dict(provider_options_value)
    if config:
        unknown = ", ".join(sorted(config))
        raise ValueError(f"Unknown SandboxSpec keys: {unknown}")

    return SandboxSpec(
        image=image,
        ttl_s=ttl_s,
        ready_timeout_s=ready_timeout_s,
        workdir=workdir,
        env=env,
        files=files,
        metadata=merged_metadata,
        resources=resources,
        entrypoint=entrypoint,
        provider_options=provider_options,
    )


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"SandboxSpec {field_name} must be a mapping, got {type(value).__name__}")
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise TypeError(f"SandboxSpec {field_name} keys and values must be strings")
    return dict(value)


def _positive_number_or_none(value: Any, field_name: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"SandboxSpec {field_name} must be a number or null")
    if value <= 0:
        raise ValueError(f"SandboxSpec {field_name} must be greater than zero")
    return value
