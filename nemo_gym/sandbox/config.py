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

A block may also carry a reserved ``volume_sources`` key: a mapping from a Compose
volume name (an ``external: true`` volume) or a bind-mount host path to the path, under
the provider's shared storage root, that stands in for it when a Compose group starts::

    sandbox:
      opensandbox: { shared_storage: { host_path: /mnt/shared } }
      volume_sources: { pgdata: datasets/pgdata, /var/lib/models: models }
"""

import ipaddress
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from nemo_gym.sandbox.providers.base import NETWORK_RULES, NetworkRule, SandboxResources, SandboxSpec


# Reserved keys inside a sandbox block that are not the provider config.
SANDBOX_BLOCK_DEFAULT_METADATA_KEY = "default_metadata"
SANDBOX_BLOCK_VOLUME_SOURCES_KEY = "volume_sources"
SANDBOX_BLOCK_OWNER_CONCURRENCY_KEY = "owner_concurrency"
SANDBOX_BLOCK_RESERVED_KEYS = frozenset(
    {SANDBOX_BLOCK_DEFAULT_METADATA_KEY, SANDBOX_BLOCK_VOLUME_SOURCES_KEY, SANDBOX_BLOCK_OWNER_CONCURRENCY_KEY}
)


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


class OwnerConcurrency(BaseModel):
    """How many sandboxes one owner starts, and stops, at once on a provider: a block's ``owner_concurrency``.

    Unset is unbounded, so a server that seeds hundreds of episodes at once starts all of their sandboxes at once.
    Starts and stops have separate limits, so a stop never waits behind starts.
    """

    model_config = ConfigDict(extra="forbid")

    starts: int | None = Field(default=None, ge=1)
    stops: int | None = Field(default=None, ge=1)


def resolve_provider_owner_concurrency(
    sandbox_provider: str | Mapping[str, Any],
    named_configs: Mapping[str, Any] | None = None,
) -> OwnerConcurrency:
    """Return a sandbox block's ``owner_concurrency``, unbounded when the block sets none.

    See :func:`resolve_provider_config` for argument semantics.

    Raises:
        ValueError: If the block or its ``owner_concurrency`` is invalid.
    """
    block, source = _resolve_block(sandbox_provider, named_configs)
    if not isinstance(block, Mapping):
        raise ValueError(f"Sandbox provider config from {source} must be a mapping, got: {block!r}")
    try:
        return OwnerConcurrency.model_validate(block.get(SANDBOX_BLOCK_OWNER_CONCURRENCY_KEY) or {})
    except ValidationError as error:
        raise ValueError(
            f"Sandbox '{SANDBOX_BLOCK_OWNER_CONCURRENCY_KEY}' from {source} is invalid: {error}"
        ) from error


def resolve_provider_volume_sources(
    sandbox_provider: str | Mapping[str, Any],
    named_configs: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Return a sandbox block's ``volume_sources``: Compose volume names and bind-mount host paths mapped
    to paths under the provider's shared storage root.

    Returns an empty dict when the block has no ``volume_sources`` key. Keys and values must be
    non-empty strings. See :func:`resolve_provider_config` for argument semantics.
    """
    block, source = _resolve_block(sandbox_provider, named_configs)
    if not isinstance(block, Mapping):
        raise ValueError(f"Sandbox provider config from {source} must be a mapping, got: {block!r}")

    return validate_volume_sources(block.get(SANDBOX_BLOCK_VOLUME_SOURCES_KEY), source)


def validate_volume_sources(sources: Any, source: str = "the sandbox block") -> dict[str, str]:
    """``volume_sources`` as a plain dict of non-empty strings; ``None`` is empty. ``source`` names the config
    in error messages."""
    sources = _to_plain_dict(sources) if sources is not None else {}
    if not isinstance(sources, Mapping):
        raise ValueError(
            f"Sandbox '{SANDBOX_BLOCK_VOLUME_SOURCES_KEY}' from {source} must be a mapping, got: {sources!r}"
        )
    for key, value in sources.items():
        if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
            raise ValueError(
                f"Sandbox '{SANDBOX_BLOCK_VOLUME_SOURCES_KEY}' from {source} maps volume names or host paths to "
                f"non-empty shared-storage paths, got {key!r}: {value!r}"
            )
    return dict(sources)


# provider_options entries that set a sandbox's network themselves, which a typed ``network`` rule replaces. Today
# only OpenSandbox reads one (network_policy); the other providers set their network in the provider block.
PROVIDER_NETWORK_OPTIONS = ("network_policy", "network")


# One DNS label: letters, digits and inner hyphens.
_HOST_LABEL = re.compile(r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)")


def _allowed_host(entry: str) -> str:
    """``entry`` if egress rules can match it: a host name, ``*.`` and a domain, or an IP address or CIDR block.

    A URL, a ``host:port`` or a path would never match, so the sandbox would silently have no egress to it; a loopback
    address is the sandbox itself.
    """
    try:
        network = ipaddress.ip_network(entry, strict=False)
    except ValueError:
        network = None
    if network is not None:
        if network.is_loopback or network.is_unspecified or network.prefixlen == 0:
            raise ValueError(f"allowed_hosts entry {entry!r} is a loopback or unspecified address, or every address")
        return entry
    name = entry.removeprefix("*.").removesuffix(".")
    if name.lower() == "localhost" or not all(_HOST_LABEL.fullmatch(label) for label in name.split(".")):
        raise ValueError(
            f"allowed_hosts entry {entry!r} is not a host name, '*.' and a domain, or an IP address or CIDR block; "
            "give it without scheme, port or path"
        )
    return entry


@dataclass(frozen=True)
class NetworkRuleOptions:
    """The provider options that apply one network rule, as the owner's ``OwnedSandboxes.spec`` builds them.

    ``SandboxConfig.spec`` takes them in place of the block's ``provider_options`` only for the rule they were built
    for, so no other mapping can stand in for a rule.
    """

    rule: NetworkRule
    provider_options: Mapping[str, Any]


class SandboxConfig(BaseModel):
    """The ``sandbox_config`` settings a server applies to every sandbox it starts.

    Servers that need more, such as a fixed image, subclass it with their own fields.

    ``network`` is the egress rule each sandbox starts with (see :data:`~nemo_gym.sandbox.NetworkRule`). The owner's
    ``OwnedSandboxes.spec`` turns it into the provider's own policy. ``model_server`` names the model server that
    ``model_only`` and ``model_and_tools`` allow, ``tool_servers`` the servers that ``model_and_tools`` also allows,
    and ``allowed_hosts`` further hosts any restricted rule allows, such as a dependency proxy; ``hosts_only`` allows
    only those. A block without ``network`` keeps the provider's default egress, or the ``network_policy`` its
    ``provider_options`` set, with a deprecation warning from the owner that starts it.
    """

    model_config = ConfigDict(extra="forbid")

    ttl_s: int | float | None = None
    ready_timeout_s: int | float | None = None
    resources: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    provider_options: dict[str, Any] = Field(default_factory=dict)
    network: NetworkRule | None = None
    model_server: str | None = None
    tool_servers: list[str] = Field(default_factory=list)
    allowed_hosts: list[str] = Field(default_factory=list)

    @field_validator("allowed_hosts")
    @classmethod
    def _matchable_hosts(cls, hosts: list[str]) -> list[str]:
        return [_allowed_host(host) for host in hosts]

    @model_validator(mode="after")
    def _consistent_network(self) -> "SandboxConfig":
        rule = self.network
        if rule is not None and rule not in NETWORK_RULES:
            # A subclass's own network setting, such as partner_if's tool sandbox, keeps its meaning until it moves.
            # The typed rule's fields would do nothing beside it.
            typed = [name for name in ("model_server", "tool_servers", "allowed_hosts") if getattr(self, name)]
            if typed:
                raise ValueError(
                    f"{typed} go with a network rule ({', '.join(sorted(NETWORK_RULES))}), not with this block's own "
                    f"network setting {rule!r}"
                )
            return self
        uses_model = rule in ("model_only", "model_and_tools")
        if uses_model != (self.model_server is not None):
            raise ValueError(
                f"model_server is set exactly for network model_only and model_and_tools, not for {rule!r}"
            )
        if (rule == "model_and_tools") != bool(self.tool_servers):
            raise ValueError(f"tool_servers are set exactly for network model_and_tools, not for {rule!r}")
        if rule == "hosts_only" and not self.allowed_hosts:
            raise ValueError("network hosts_only needs allowed_hosts")
        if self.allowed_hosts and rule in (None, "inherit", "no_network"):
            raise ValueError(
                f"allowed_hosts add to a restricted network rule, not to {rule!r}; use hosts_only for only those hosts"
            )
        if rule is not None:
            conflicting = [option for option in PROVIDER_NETWORK_OPTIONS if option in self.provider_options]
            if conflicting:
                raise ValueError(
                    f"network {rule!r} and provider_options {conflicting} both set this sandbox's network; set only "
                    "network"
                )
        return self

    @field_validator("resources")
    @classmethod
    def _known_resources(cls, resources: dict[str, Any]) -> dict[str, Any]:
        SandboxResources.from_mapping(resources)
        return resources

    def spec(
        self,
        sandbox_provider: str | Mapping[str, Any],
        *,
        image: str | None,
        workdir: str | None,
        metadata: Mapping[str, str] | None = None,
        files: Mapping[str, str | bytes] | None = None,
        named_configs: Mapping[str, Any] | None = None,
        network_options: NetworkRuleOptions | None = None,
    ) -> SandboxSpec:
        """Build the spec of one sandbox; ``metadata`` adds to the provider's and the config's metadata.

        ``network_options`` apply the block's ``network`` rule, such as OpenSandbox's ``network_policy`` or Docker's
        ``run_args``. Only the owner's ``OwnedSandboxes.spec`` builds them; a subclass that overrides this method
        passes them on. They replace ``provider_options``, and a restricted rule needs them.

        Raises:
            ValueError: For a restricted rule without ``network_options`` built for it, so that a block that asks
                for restricted egress never starts a sandbox with the provider's default egress.
        """
        provider_options = dict(self.provider_options)
        if self.network in NETWORK_RULES and self.network != "inherit":
            if network_options is None:
                raise ValueError(
                    f"network {self.network!r} needs the provider's policy for it: build this spec with the owner's "
                    "OwnedSandboxes.spec(), not SandboxConfig.spec() alone"
                )
            if not isinstance(network_options, NetworkRuleOptions) or network_options.rule != self.network:
                raise ValueError(
                    f"network {self.network!r} takes the NetworkRuleOptions the owner built for it, not "
                    f"{network_options!r}"
                )
            provider_options = dict(network_options.provider_options)
        return SandboxSpec(
            image=image,
            ttl_s=self.ttl_s,
            ready_timeout_s=self.ready_timeout_s,
            workdir=workdir,
            env=dict(self.env),
            files=dict(files or {}),
            metadata=resolve_provider_metadata(sandbox_provider, named_configs) | self.metadata | dict(metadata or {}),
            resources=SandboxResources.from_mapping(self.resources),
            provider_options=provider_options,
        )
