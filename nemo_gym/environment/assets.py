# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a named asset (a benchmark, environment, resources server, model type or agent type) to its config path."""

import logging
from pathlib import Path

from nemo_gym import component_search_roots
from nemo_gym.environment.manifest import load_manifest, resolve_manifest_config_path
from nemo_gym.environment.registry import find_named_manifest
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

    if "/" in value and flag in {"benchmark", "environment"}:
        nested_matches: list[Path] = []
        for root in roots:
            candidate = root / parent / value / "config.yaml"
            resolved = candidate.resolve()
            if candidate.is_file() and resolved not in nested_matches:
                nested_matches.append(resolved)
        if nested_matches:
            if len(nested_matches) > 1:
                shadowed = ", ".join(f"`{path}`" for path in nested_matches[1:])
                logger.warning(
                    f"`--{flag} {value}` matches multiple configs; using `{nested_matches[0]}` from the "
                    f"highest-priority root and ignoring {shadowed}. Pass `--config <path>` to select a different one."
                )
            return str(nested_matches[0])

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

    # A manifest's name is its ID and need not match its folder, so a benchmark or environment that moved still
    # resolves by the name its manifest declares. A config at the path is tried first and keeps today's names working.
    if flag in {"benchmark", "environment"}:
        manifest_path = find_named_manifest(value, subdir=parent, roots=roots)
        if manifest_path is not None:
            return str(resolve_manifest_config_path(manifest_path, load_manifest(manifest_path)))

    # No match: build a "did you mean?" hint and the roots searched
    if flag == "benchmark":
        # Benchmarks need special handling because some use non-standard config paths (arbitrary nesting), so
        # the generic one-level flavor/sibling search below can't see them.
        # Enumerate their real config names (the same values `gym list benchmarks` prints) instead.
        from nemo_gym.environment.benchmarks import _benchmark_config_name, _benchmark_config_paths

        config_names = {
            _benchmark_config_name(p.relative_to(root / parent))
            for root in roots
            for p in _benchmark_config_paths(root / parent)
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
