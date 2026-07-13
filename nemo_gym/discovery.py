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
"""Shared component-discovery utilities for the ``gym list``/``gym search`` commands.

An environment and a benchmark (a benchmark is a specific kind of environment) are listed with the
same columns, read the same way. This module holds the config-reading code both listings share so the
per-component registries (``registry.py``, ``benchmarks.py``, ``agent_registry.py``) can depend on it
without depending on each other. It only reads config files — it never starts servers — and only
imports lower-level config utilities, so it sits below those registries in the import graph.
"""

import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError

from nemo_gym.global_config import (
    POLICY_MODEL_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
)


# Fills unset `???`/`${...}` values during listing: they reference runtime-only values (API keys,
# endpoints) not needed to identify a component, so a placeholder lets the config still resolve.
_UNSET_VALUE_PLACEHOLDER = "__unset_for_listing__"

# Server groups a component's `domain`/`description` may be declared on. `domain` can sit on a
# resources server (e.g. `aime24`), an agent (e.g. `tau2`), or in principle a model server.
_SERVER_GROUP_KEYS = ("resources_servers", "responses_api_agents", "responses_api_models")


def _parse_no_environment_tolerating_unset_values(initial_config_dict: DictConfig) -> DictConfig:
    """`parse_no_environment` for *listing*: fill unset `???` values and undefined `${...}` interpolations
    with a placeholder so a component referencing runtime-only values can still be identified. Never mutates
    the caller's config; errors other than those two propagate.

    `???` is filled anywhere; `${...}` only where parse forces resolution (top-level and server sections),
    not inside arbitrary non-server nested dicts — fine for listing, whose interpolations live in servers.
    """
    working = deepcopy(initial_config_dict)  # never mutate the caller's config
    parser = GlobalConfigDictParser()

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


def _scan_servers_for_metadata(container) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort ``(domain, description)`` from a config mapping, scanning every server group.

    Reads the first ``domain`` and the first ``description`` found across all server instances. Defensive
    against malformed shapes (non-mapping top level, a server group that isn't a dict) so it never raises.
    """
    domain: Optional[str] = None
    description: Optional[str] = None
    if not isinstance(container, (dict, DictConfig)):
        return None, None
    for instance in container.values():
        if not isinstance(instance, (dict, DictConfig)):
            continue
        for group_key in _SERVER_GROUP_KEYS:
            servers = instance.get(group_key)
            if not isinstance(servers, (dict, DictConfig)):
                continue
            for server_config in servers.values():
                if not isinstance(server_config, (dict, DictConfig)):
                    continue
                if domain is None and server_config.get("domain"):
                    domain = str(server_config["domain"])
                if description is None and server_config.get("description"):
                    description = str(server_config["description"])
    return domain, description


def read_config_metadata(config_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Shared listing metadata reader: ``(domain, description)`` for an environment *or* benchmark config.

    A benchmark is a specific kind of environment, so both are read the same way. Two-pass, because the two
    kinds declare metadata differently:

    1. **Raw (non-resolving) scan** of the config as written. Environment configs declare ``domain``/
       ``description`` inline, and reading them without resolution is safe even though an environment config
       references model/agent servers defined elsewhere (resolving it in isolation would raise).
    2. **Resolving fallback**, only for whatever the raw scan left unset. Benchmark configs inherit their
       metadata via ``config_paths``/``_inherit_from``, so it isn't present until the config is resolved.
       Uses the listing-tolerant parser (unset runtime values are placeholdered); on any failure — e.g. an
       environment config that references servers defined elsewhere — whatever the raw scan found is kept.

    Never raises: an unreadable or unresolvable config yields ``(None, None)``.
    """
    try:
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False, throw_on_missing=False)
    except Exception:
        raw = None
    domain, description = _scan_servers_for_metadata(raw)
    if domain is not None and description is not None:
        return domain, description

    try:
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
