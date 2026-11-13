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


def _path_candidates(tree: Path, value: str) -> list[Path]:
    """The configs a benchmark or environment name points to as a path under one catalog tree, in lookup order."""
    server_name, _, config_flavor = value.partition("/")
    nested = [tree / value / "config.yaml"] if "/" in value else []
    return [*nested, tree / server_name / f"{config_flavor or 'config'}.yaml"]


def _catalog_configs_in_tree(tree: Path, value: str) -> tuple[Path | None, Path | None]:
    """The config one catalog tree resolves a name to as a catalog entry, and otherwise by its old path.

    The catalog entry is the manifest that declares the name, or else a config at that path that no manifest
    claims. A config that a manifest claims under another name is listed only under that name, so its path is
    returned separately: it is used only when no root has a catalog entry, which keeps such paths working.
    """
    # Imported here so that commands that never select a benchmark or environment, such as --help, stay fast.
    from nemo_gym.environment.manifest import ManifestError, load_manifest, resolve_manifest_config_path
    from nemo_gym.environment.registry import declared_manifests

    direct = tree / value / "manifest.yaml"
    if direct.is_file():
        try:
            manifest = load_manifest(direct)
            if manifest.name == value:
                return resolve_manifest_config_path(direct, manifest), None
        except ManifestError:
            pass
    claimed: set[Path] = set()
    for manifest_path, manifest in declared_manifests(tree):
        try:
            config_path = resolve_manifest_config_path(manifest_path, manifest)
        except ManifestError:
            continue
        if manifest.name == value:
            return config_path, None
        claimed.add(config_path)
    paths = [candidate.resolve() for candidate in _path_candidates(tree, value) if candidate.is_file()]
    unclaimed = [path for path in paths if path not in claimed]
    if unclaimed:
        return unclaimed[0], None
    return None, (paths[0] if paths else None)


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
        # The catalog's rule, root by root: the highest-priority root with a catalog entry for the name wins.
        old_paths: list[Path] = []
        for root in roots:
            entry, old_path = _catalog_configs_in_tree(root / parent, value)
            if entry is not None and entry not in matches:
                matches.append(entry)
            if old_path is not None and old_path not in old_paths:
                old_paths.append(old_path)
        matches = matches or old_paths
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
        from nemo_gym.environment.registry import _legacy_config_paths, declared_manifests

        # Suggest declared IDs as well as the folder/flavor aliases that remain selectable.
        config_names: set[str] = set()
        for root in roots:
            tree = root / parent
            if not tree.is_dir():
                continue
            config_names.update(manifest.name for _, manifest in declared_manifests(tree))
            config_names.update(
                name for name, _ in _legacy_config_paths(tree, "benchmark" if flag == "benchmark" else "environment")
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
