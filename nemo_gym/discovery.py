# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared component-discovery helpers: search roots, collision handling, metadata, and capability contracts.

Lives below the per-component registries (``registry.py``, ``benchmarks.py``, ``agent_registry.py``) so
they can share it without depending on each other. Reads configs only; never starts servers.
"""

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional, Tuple, TypeVar

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError

from nemo_gym import component_search_roots


_T = TypeVar("_T")

ServerGroupKey = Literal["resources_servers", "responses_api_agents", "responses_api_models"]
ComponentRole = Literal["resources_server", "agent_server", "model_server"]
SERVER_GROUP_KEYS: tuple[ServerGroupKey, ...] = (
    "resources_servers",
    "responses_api_agents",
    "responses_api_models",
)
SERVER_ROLE_BY_GROUP: dict[ServerGroupKey, ComponentRole] = {
    "resources_servers": "resources_server",
    "responses_api_agents": "agent_server",
    "responses_api_models": "model_server",
}
BASELINE_PROVIDES_BY_GROUP: dict[ServerGroupKey, Tuple[str, ...]] = {
    # Every Responses API model implements the text request/response contract.
    # Flavors declare only additional capabilities such as image input.
    "responses_api_models": ("text-model",),
}


@dataclass(frozen=True)
class CapabilityDeclaration:
    """Capabilities declared by one server implementation in one config flavor."""

    instance: str
    implementation: str
    requires: Tuple[str, ...] = ()
    provides: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfigFlavorCapabilities:
    """Typed capability declarations read from exactly one config file and role."""

    config_path: Path
    group: ServerGroupKey
    declarations: Tuple[CapabilityDeclaration, ...] = ()

    @property
    def requires(self) -> Tuple[str, ...]:
        """Ordered capabilities required by implementations in this flavor only."""

        return _ordered_unique(capability for declaration in self.declarations for capability in declaration.requires)

    @property
    def provides(self) -> Tuple[str, ...]:
        """Ordered capabilities provided by implementations in this flavor only."""

        return _ordered_unique(capability for declaration in self.declarations for capability in declaration.provides)


def _ordered_unique(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _capability_values(value: object) -> Tuple[str, ...]:
    """Best-effort capability normalization for non-executing discovery."""

    if isinstance(value, str):
        values: Iterable[object] = (value,)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return ()
    return _ordered_unique(item.strip() for item in values if isinstance(item, str) and item.strip())


def component_capability_declaration(
    *,
    instance: str,
    implementation: str,
    group: ServerGroupKey,
    server_config: object,
) -> CapabilityDeclaration:
    """Return the interface baseline plus one flavor's explicit declarations."""

    config = server_config if isinstance(server_config, (dict, DictConfig)) else {}
    return CapabilityDeclaration(
        instance=instance,
        implementation=implementation,
        requires=_capability_values(config.get("requires")),
        provides=_ordered_unique(
            (*BASELINE_PROVIDES_BY_GROUP.get(group, ()), *_capability_values(config.get("provides")))
        ),
    )


def read_config_flavor_capabilities(
    config_path: Path,
    group: ServerGroupKey,
) -> ConfigFlavorCapabilities:
    """Read capability declarations for one role from one config flavor.

    The function intentionally does not merge sibling files: two selectable
    flavors may expose different contracts even when they share an implementation
    directory. Discovery is best effort and never resolves runtime-only values.
    """

    try:
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False, throw_on_missing=False)
    except Exception:
        raw = None

    declarations: list[CapabilityDeclaration] = []
    if isinstance(raw, dict):
        for instance, top_level_value in raw.items():
            if not isinstance(top_level_value, dict):
                continue
            servers = top_level_value.get(group)
            if not isinstance(servers, dict):
                continue
            for implementation, server_config in servers.items():
                if not isinstance(server_config, dict):
                    continue
                declarations.append(
                    component_capability_declaration(
                        instance=str(instance),
                        implementation=str(implementation),
                        group=group,
                        server_config=server_config,
                    )
                )

    return ConfigFlavorCapabilities(
        config_path=config_path,
        group=group,
        declarations=tuple(declarations),
    )


def merge_by_name(per_root: Iterable[Dict[str, _T]]) -> Dict[str, _T]:
    """Merge per-root ``name -> entry`` mappings; earlier roots win on a collision (user shadows built-in),
    matching :func:`component_search_roots` precedence. Insertion order preserved.
    """
    merged: Dict[str, _T] = {}
    for entries in per_root:
        for name, entry in entries.items():
            merged.setdefault(name, entry)
    return merged


def discover_components(
    subdir: str,
    dir_scanning_fn: Callable[[Path], Dict[str, _T]],
) -> Dict[str, _T]:
    """Run ``dir_scanning_fn`` on ``subdir`` of every :func:`component_search_roots` root and merge the results.

    The shared body of ``discover_environments``/``discover_agents``/``discover_models``/
    ``discover_benchmarks``: each passes its ``<type>/`` subdir and a single-directory scan function, and
    gets user-shadows-built-in merging (via :func:`merge_by_name`) for free.
    """
    return merge_by_name(dir_scanning_fn(root / subdir) for root in component_search_roots())


# Fills unset `???`/`${...}` values during listing: they reference runtime-only values (API keys,
# endpoints) not needed to identify a component, so a placeholder lets the config still resolve.
_UNSET_VALUE_PLACEHOLDER = "__unset_for_listing__"


def _parse_no_environment_tolerating_unset_values(initial_config_dict: DictConfig) -> DictConfig:
    """`parse_no_environment` for listing: fill unset `???` and undefined `${...}` values (runtime-only
    things like API keys/endpoints) with a placeholder so the config still resolves enough to identify the
    component. Never mutates the input; errors other than those two propagate.
    """
    from nemo_gym.global_config import StaticValidationConfigParser

    working = deepcopy(initial_config_dict)  # never mutate the caller's config
    parser = StaticValidationConfigParser()

    # Fill all `???` leaves in one pass. The loop below only adds placeholder keys, so no new `???` appear.
    for path in parser.collect_missing_value_paths(working):
        OmegaConf.update(working, path, _UNSET_VALUE_PLACEHOLDER)

    # OmegaConf reports undefined `${...}` keys only one at a time (as InterpolationKeyError), so loop:
    # inject a placeholder for each reported key and retry until it resolves.
    injected: set[str] = set()
    while True:
        try:
            return parser.parse_no_environment(initial_global_config_dict=working)
        except InterpolationKeyError as e:
            # The missing key name is only in the message text — omegaconf never stores it on an attribute
            # (`e.key`/`e.full_key` point at the containing node), so a regex is the only way to read it.
            match = re.search(r"Interpolation key '([^']+)'", str(e))
            key = match.group(1) if match else None
            if not key or key in injected:
                raise  # can't identify/clear the missing key; let the caller decide (warn + skip)
            injected.add(key)
            working = OmegaConf.merge(DictConfig({key: _UNSET_VALUE_PLACEHOLDER}), working)


def resolve_config_paths_static(config_paths: Iterable[str | Path]) -> DictConfig:
    """Resolve config paths for discovery/CI without runtime integrations or a real model."""

    from nemo_gym.global_config import GlobalConfigDictParserConfig

    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        OmegaConf.create({"config_paths": [str(path) for path in config_paths]}),
    )
    return _parse_no_environment_tolerating_unset_values(initial)


def iter_server_configs(container):
    """Yield ``(group_key, server_name, server_config)`` for every server across all instances in a config.

    Walks a loaded config mapping (each top-level instance -> its ``resources_servers``/
    ``responses_api_agents``/``responses_api_models`` group -> each server). Defensive against malformed
    shapes, so it never raises. The shared primitive behind metadata reads and the inspect deep-parse.
    """
    if not isinstance(container, (dict, DictConfig)):
        return
    for instance in container.values():
        if not isinstance(instance, (dict, DictConfig)):
            continue
        for group_key in SERVER_GROUP_KEYS:
            servers = instance.get(group_key)
            if not isinstance(servers, (dict, DictConfig)):
                continue
            for server_name, server_config in servers.items():
                if isinstance(server_config, (dict, DictConfig)):
                    yield group_key, server_name, server_config


def _scan_servers_for_metadata(container) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort ``(domain, description)`` from a config mapping: the first of each found across all
    server groups. Never raises.
    """
    domain: Optional[str] = None
    description: Optional[str] = None
    for _group_key, _server_name, server_config in iter_server_configs(container):
        if domain is None and server_config.get("domain"):
            domain = str(server_config["domain"])
        if description is None and server_config.get("description"):
            description = str(server_config["description"])
    return domain, description


def read_config_metadata(config_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Shared ``(domain, description)`` reader for an environment *or* benchmark config. Two passes, because
    the two declare metadata differently:

    1. Raw (non-resolving) scan — environment configs declare it inline, and this is safe even though they
       reference servers defined elsewhere (resolving in isolation would raise).
    2. Resolving fallback for whatever's still unset — benchmark configs inherit it via
       ``config_paths``/``_inherit_from``. Tolerates unset runtime values; on failure keeps the raw result.

    Never raises: an unreadable/unresolvable config yields ``(None, None)``.
    """
    try:
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False, throw_on_missing=False)
    except Exception:
        raw = None
    domain, description = _scan_servers_for_metadata(raw)
    if domain is not None and description is not None:
        return domain, description

    try:
        from nemo_gym.global_config import POLICY_MODEL_KEY_NAME, GlobalConfigDictParserConfig

        initial_config_dict = OmegaConf.load(config_path)
        if POLICY_MODEL_KEY_NAME not in initial_config_dict:
            initial_config_dict = OmegaConf.merge(
                initial_config_dict, GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT
            )
        resolved = _parse_no_environment_tolerating_unset_values(initial_config_dict)
    except Exception:
        return domain, description

    resolved_domain, resolved_description = _scan_servers_for_metadata(resolved)
    return (
        domain if domain is not None else resolved_domain,
        description if description is not None else resolved_description,
    )
