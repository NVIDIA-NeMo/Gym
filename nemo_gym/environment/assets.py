# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a named asset (a benchmark, environment, resources server, model type or agent type) to its config path."""

import logging
from pathlib import Path

from nemo_gym import component_search_roots
from nemo_gym.environment.catalog_renames import RENAMED_CATALOG_NAMES, renamed_catalog_name_warning
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


# Selector flags whose names are catalog names; each flag is also the catalog kind it selects.
_CATALOG_FLAGS = frozenset({"benchmark", "environment"})


def _manifest_config_path(root: Path, flag: str, value: str) -> Path | None:
    """The config that the manifest named ``value`` points at in one root, or None when the root has none.

    The catalog lists a manifest under its folder path, whether or not its config sits in that folder
    (``config_path: ../opencode.yaml``), so the selector reads the manifest the same way the catalog does. A manifest
    that fails to load is skipped, as the catalog skips it, and the config paths below decide instead.
    """
    tree_dir = root / ASSETS[flag][0]
    manifest_path = tree_dir / value / "manifest.yaml"
    if not manifest_path.is_file():
        return None
    # Imported here: the registry pulls in config parsing that the CLI does not otherwise load at startup.
    from nemo_gym.environment.manifest import ManifestError
    from nemo_gym.environment.registry import _manifest_entry

    try:
        return _manifest_entry(tree_dir, manifest_path, flag).config_path
    except ManifestError as error:
        logger.warning(f"`--{flag} {value}` ignores manifest `{manifest_path}`, which does not load: {error}")
        return None


def _config_in_root(root: Path, flag: str, value: str, path: str) -> Path | None:
    """The config ``--<flag> <value>`` selects in one root: its manifest's config, then ``<value>/config.yaml``,
    then the ``<server>/<flavor>.yaml`` path every asset flag accepts."""
    if flag in _CATALOG_FLAGS:
        if (manifest_config := _manifest_config_path(root, flag, value)) is not None:
            return manifest_config
        nested = root / ASSETS[flag][0] / value / "config.yaml"
        if "/" in value and nested.is_file():
            return nested
    candidate = root / path
    return candidate if candidate.exists() else None


def asset_config_path(flag: str, value: str) -> str:
    """Map a named asset (`name` or `name/flavor`) to its config path, following renamed catalog names.

    A renamed benchmark or environment catalog name
    (:data:`~nemo_gym.environment.catalog_renames.RENAMED_CATALOG_NAMES`) selects the config of the name that replaced
    it, with a deprecation warning, unless it still selects a config of its own: a run preset, which it keeps
    selecting without a warning.
    """
    renamed = RENAMED_CATALOG_NAMES.get(value) if flag in _CATALOG_FLAGS else None
    if renamed is None:
        return _config_path_by_name(flag, value)
    replacement = asset_config_path(flag, renamed)
    try:
        selected = _config_path_by_name(flag, value)
    except ValueError:
        selected = replacement
    if selected == replacement:
        logger.warning(renamed_catalog_name_warning(value, renamed))
    return selected


def _config_path_by_name(flag: str, value: str) -> str:
    """Map a named asset (`name` or `name/flavor`) to its config path.

    Searches the roots from :func:`~nemo_gym.component_search_roots` (``NEMO_GYM_EXTRA_ROOTS`` +
    cwd + install root), the same helper that backs `gym list`/`gym search`, so config resolution and
    discovery agree on where components live. On a name collision across roots the highest-priority root wins
    (as in `gym list`), with a warning. ``--search-dir`` reaches here via ``NEMO_GYM_EXTRA_ROOTS`` (set
    in ``main``). Searching the install root is what lets built-ins resolve by name from an arbitrary cwd
    (e.g. a wheel install), not just inside the repo checkout.

    A benchmark or environment name is first looked up as a catalog name: ``<parent>/<name>/manifest.yaml`` selects
    the config its ``config_path`` names, which may live outside that folder. Within each root the manifest comes
    first, then ``<parent>/<name>/config.yaml``, then ``<parent>/<server>/<flavor>.yaml``, matching the order in
    which `gym list` claims a name.
    """
    parent, subdir, default_flavor = ASSETS[flag]
    server_name, _, config_flavor = value.partition("/")
    config_flavor = config_flavor or default_flavor or server_name
    config_dir = f"{parent}/{server_name}/{subdir}".rstrip("/")
    path = f"{config_dir}/{config_flavor}.yaml"

    roots = component_search_roots()
    matches: list[Path] = []
    for root in roots:
        candidate = _config_in_root(root, flag, value, path)
        if candidate is not None and (resolved := candidate.resolve()) not in matches:
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
    if flag == "benchmark":
        # Benchmarks need special handling because some use non-standard config paths (arbitrary nesting), so
        # the generic one-level flavor/sibling search below can't see them.
        # Enumerate their real config names (the same values `gym list benchmarks` prints) and catalog names instead.
        from nemo_gym.environment.benchmarks import _benchmark_config_name, _benchmark_config_paths
        from nemo_gym.environment.registry import _manifest_paths

        config_names = {
            _benchmark_config_name(p.relative_to(root / parent))
            for root in roots
            for p in _benchmark_config_paths(root / parent)
        } | {
            p.parent.relative_to(root / parent).as_posix()
            for root in roots
            for p in _manifest_paths(root / parent)
            if p.parent != root / parent
        }
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
