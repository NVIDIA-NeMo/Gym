# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a named asset (a benchmark, environment, resources server, model type or agent type) to its config path."""

import logging
from pathlib import Path

from nemo_gym import component_search_roots
from nemo_gym.utils.text import did_you_mean


logger = logging.getLogger(__name__)


# Asset selector flag -> (parent dir, configs subdir, default config flavor). All accept `name` or `name/flavor`,
# resolving to `<parent>/<server>/[<subdir>/]<flavor>.yaml`. A None default flavor falls back to the server name.
ASSETS = {
    "benchmark": ("benchmarks", "", "config"),
    "environment": ("environments", "", "config"),
    "resources-server": ("resources_servers", "configs", None),
    "model-type": ("responses_api_models", "configs", None),
    "agent-type": ("responses_api_agents", "configs", None),
}
# Benchmarks and environments are catalog entries: their selectors resolve a name the way the catalog does.
_CATALOG_FLAGS = frozenset({"benchmark", "environment"})


def _path_configs(trees: list[Path], value: str) -> list[Path]:
    """The configs at a benchmark or environment name's path, highest-priority root first.

    This is how `--benchmark` and `--environment` found a config before names were IDs: a nested `<name>/config.yaml`
    in any search root wins over a flavor file `<folder>/<flavor>.yaml`.
    """
    server_name, _, config_flavor = value.partition("/")
    paths = [f"{value}/config.yaml"] if config_flavor else []
    paths.append(f"{server_name}/{config_flavor or 'config'}.yaml")
    for path in paths:
        found: list[Path] = []
        for tree in trees:
            candidate = tree / path
            if candidate.is_file() and candidate.resolve() not in found:
                found.append(candidate.resolve())
        if found:
            return found
    return []


def asset_config_path(flag: str, value: str) -> str:
    """Map a named asset (`name` or `name/flavor`) to its config path.

    Searches the roots from :func:`~nemo_gym.component_search_roots` (``NEMO_GYM_EXTRA_ROOTS`` +
    cwd + install root), the same helper that backs `gym list`/`gym search`, so config resolution and
    discovery agree on where components live. On a name collision across roots the highest-priority root wins
    (as in `gym list`), with a warning. ``--search-dir`` reaches here via ``NEMO_GYM_EXTRA_ROOTS`` (set
    in ``main``). Searching the install root is what lets built-ins resolve by name from an arbitrary cwd
    (e.g. a wheel install), not just inside the repo checkout.
    """
    parent, subdir, default_flavor = ASSETS[flag]
    server_name, _, config_flavor = value.partition("/")
    config_flavor = config_flavor or default_flavor or server_name
    config_dir = f"{parent}/{server_name}/{subdir}".rstrip("/")
    path = f"{config_dir}/{config_flavor}.yaml"

    roots = component_search_roots()
    matches: list[Path] = []

    if flag in _CATALOG_FLAGS:
        # Imported here so that commands that never select a benchmark or environment, such as --help, stay fast.
        from nemo_gym.environment.registry import RegistryError, _listing_for_selection

        trees = [root / parent for root in roots]
        # The catalog's first step, so that this selects what `gym list` shows: the highest-priority root that has the
        # name wins. When that root cannot list it, because two of its manifests declare the name or the catalog
        # refuses its manifest, selecting the name fails as `gym list` does. Other names still select, so a copied
        # folder that keeps its manifest's name stops only that name.
        listing = _listing_for_selection(flag, value)
        if (flag, value) in listing.refused:
            raise RegistryError(listing.refused[(flag, value)])
        listed = listing.entries.get((flag, value))
        if listed is not None:
            for config_path in (listed.config_path, *listing.shadowed.get((flag, value), ())):
                if config_path.resolve() not in matches:
                    matches.append(config_path.resolve())
        else:
            # Any other config at the name's path, such as an environment flavor file, a nested benchmark config or a
            # config that a manifest declares under another name, selects only when no root lists the name.
            matches = _path_configs(trees, value)
    else:
        for root in roots:
            candidate = root / path
            if candidate.exists():
                resolved = candidate.resolve()
                if resolved not in matches:
                    matches.append(resolved)

    if len(matches) > 1:
        shadowed = ", ".join(f"`{m}`" for m in matches[1:])
        logger.warning(
            f"`--{flag} {value}` matches multiple configs; using `{matches[0]}` from the highest-priority root "
            f"and ignoring {shadowed}. Pass `--config <path>` to select a different one."
        )
    if matches:
        return str(matches[0])

    # No match: build a "did you mean?" hint and the roots searched
    if flag in _CATALOG_FLAGS:
        from nemo_gym.environment.registry import _legacy_config_paths

        # Suggest the names the catalog lists as well as the folder/flavor aliases that remain selectable.
        config_names = {name for kind, name in listing.entries if kind == flag}
        for tree in trees:
            if not tree.is_dir():
                continue
            config_names.update(
                name for name, _ in _legacy_config_paths(tree, "benchmark" if flag == "benchmark" else "environment")
            )
            if flag == "environment":
                # Flavor files next to an environment's config select as `<folder>/<flavor>` but are not listed.
                config_names.update(
                    f"{server_name}/{candidate.stem}"
                    for candidate in (tree / server_name).glob("*.yaml")
                    if candidate.name != "manifest.yaml"
                )
        # A bare directory that only groups benchmarks (e.g. `livecodebench`) is not itself selectable, so point
        # at the config names under it; otherwise fall back to a fuzzy match across every token.
        under_dir = sorted(config_name for config_name in config_names if config_name.startswith(f"{value}/"))
        hint = f" Did you mean `{min(under_dir, key=len)}`?" if under_dir else did_you_mean(value, config_names)
        available = ", ".join(sorted(f"`{(root / parent).resolve()}`" for root in roots if (root / parent).is_dir()))
    else:
        # Suggest the closest real name across all roots: a config flavor when the server exists, else a server
        # name, reporting the full paths that were searched in each case.
        available = ", ".join(
            set(f"`{(root / config_dir).resolve()}`" for root in roots if (root / config_dir).is_dir())
        )
        typo = config_flavor
        candidates = [p.stem for root in roots for p in (root / config_dir).glob("*.yaml")]

        if len(candidates) == 0:
            available = ", ".join(set(f"`{(root / parent).resolve()}`" for root in roots if (root / parent).is_dir()))
            typo = server_name
            candidates = [
                child.name
                for root in roots
                if (root / parent).is_dir()
                for child in (root / parent).iterdir()
                if child.is_dir()
            ]
            if server_name in candidates:
                # The folder exists but has no YAML configs, so the problem is the missing config,
                # not the name. Suggesting a different server here would point the user elsewhere.
                candidates = []

        hint = did_you_mean(typo, candidates)

    raise ValueError(
        f"`--{flag} {value}` was specified which implies config `{path}`, which does not exist.{hint} "
        f"See available {flag} configs in {available}."
    )
