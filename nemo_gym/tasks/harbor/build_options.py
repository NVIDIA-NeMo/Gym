# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builds table of the run configuration.

BuildOptions carries whether builds are enabled, which tool to use, the repository to push to
and the platform to build for. It is read from a Hydra config mapping and, before the run
config exists, from the command line overrides and env.yaml by a side-effect-free Hydra parse.
The module depends on the build tool list, the repository variable and the platform parser only.
"""

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.tasks.harbor.build_plan import BUILD_TOOLS, REPOSITORY_ENV
from nemo_gym.tasks.harbor.registry import ImageConfigError, platform_pair


# The top-level configuration block that sets Dockerfile builds up, and the settings it takes.
BUILDS_KEY = "builds"
BUILD_SETTINGS = ("tool", "repository", "platform", "enabled", "force")
HYDRA_QUIET_OVERRIDES = (
    "hydra.run.dir=.",
    "hydra.output_subdir=null",
    "hydra/job_logging=none",
    "hydra/hydra_logging=none",
)


@dataclass(frozen=True)
class BuildOptions:
    """How prepare treats Dockerfiles outside pull and overlay mode: the ``builds:`` configuration block.

    ``tool`` builds them (buildah, podman or docker; default: the first one on PATH that fits, see
    :func:`nemo_gym.tasks.harbor.image_build.detect_tool`). ``repository`` is the registry path built
    images are pushed under (default: the ``NEMO_GYM_BUILD_REPOSITORY`` variable; unset keeps them local
    in the Docker daemon). ``platform`` is passed to the tool (default ``linux/amd64``). With ``enabled``
    false nothing is built, and the tasks that need a build are skipped with the loader's message. With
    ``force``, a task's Dockerfile is built even when the task declares ``[environment].docker_image``, as
    Harbor's ``--force-build`` does, and the task then runs the built image.
    """

    enabled: bool = True
    tool: str | None = None
    repository: str | None = None
    platform: str | None = None
    force: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "BuildOptions":
        """The ``builds:`` block of a merged configuration dict, applied over the defaults.

        ``config`` is the merged global configuration (``env.yaml``, ``--config`` files and
        ``++builds.<setting>=<value>`` overrides) or any dict shaped like it; a missing block keeps
        every default. Unknown settings, values of the wrong type and a ``platform`` the records could not tell
        from the baseline (an amd64 micro-architecture level above it, see :func:`_check_platform`) are refused.
        """
        block = (config or {}).get(BUILDS_KEY)
        if block is None:
            block = {}
        if not isinstance(block, Mapping):
            raise ValueError(f"`{BUILDS_KEY}` must be a mapping of {', '.join(BUILD_SETTINGS)}; got {block!r}")
        unknown = sorted(set(block) - set(BUILD_SETTINGS))
        if unknown:
            raise ValueError(
                f"Unknown `{BUILDS_KEY}` setting(s) {', '.join(unknown)}; expected {', '.join(BUILD_SETTINGS)}"
            )
        flags = {name: block.get(name, default) for name, default in (("enabled", True), ("force", False))}
        for name, value in flags.items():
            if not isinstance(value, bool):
                raise ValueError(f"`{BUILDS_KEY}.{name}` must be true or false; got {value!r}")
        tool = _build_setting(block, "tool")
        if tool is not None and tool not in BUILD_TOOLS:
            raise ValueError(f"`{BUILDS_KEY}.tool` must be one of {', '.join(BUILD_TOOLS)}; got {tool!r}")
        _check_platform(block)
        return cls(
            enabled=flags["enabled"],
            tool=tool,
            repository=_build_setting(block, "repository") or os.environ.get(REPOSITORY_ENV) or None,
            platform=_build_setting(block, "platform"),
            force=flags["force"],
        )


def _build_setting(block: Mapping[str, Any], name: str) -> str | None:
    value = block.get(name)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"`{BUILDS_KEY}.{name}` must be a string; got {value!r}")
    return value


def _check_platform(block: Mapping[str, Any]) -> None:
    """Refuse a ``platform`` setting :func:`~nemo_gym.tasks.harbor.registry.platform_pair` refuses, where it enters.

    An amd64 micro-architecture level above the baseline (``linux/amd64/v3``) is another platform to docker, but
    the records keep an image's os and architecture only, so a sidecar built for it (of a task whose ``main`` is
    prebuilt, with no base image to record first) would be pushed and recorded as ``linux/amd64`` and reused by a
    later baseline run. ``ImageBuilder`` repeats the check for a :class:`BuildOptions` made in code.
    """
    platform = _build_setting(block, "platform")
    if platform is None:
        return
    try:
        platform_pair(platform)
    except ImageConfigError as exc:
        raise ValueError(f"`{BUILDS_KEY}.platform`: {exc}") from exc


def build_options_from_overrides(overrides: list[str]) -> BuildOptions:
    """The ``builds:`` block of the configuration the command line names (see :func:`_block_from_overrides`)."""
    return BuildOptions.from_config({BUILDS_KEY: _block_from_overrides(overrides, BUILDS_KEY)})


def _merged_cli_config(overrides: list[str], config_files: Sequence[Path] = ()) -> Any:
    """The configuration the command line names, merged but unresolved, before the run config exists.

    The sources the user controls, in the global config's order: ``config_files``, then the files the
    ``+config_paths`` tokens name (from ``--config`` and the asset selectors), then ``env.yaml``, then the
    ``+``/``++`` overrides. Nothing is resolved here, so a provider block's credentials are not needed to
    read a sibling key. The full global config is still parsed once, at dispatch, with the run config added.
    """
    from omegaconf import OmegaConf

    from nemo_gym.config.global_config import CONFIG_PATHS_KEY_NAME, GlobalConfigDictParser, _load_config_yaml

    parser = GlobalConfigDictParser()
    saved_argv = sys.argv
    # A side-effect-free Hydra parse: no outputs/<date>/<time> run folder and no logging reconfiguration,
    # which the one parse at dispatch keeps as its own.
    sys.argv = [saved_argv[0], *overrides, *HYDRA_QUIET_OVERRIDES]
    try:
        cli = parser.parse_global_config_dict_from_cli()
    finally:
        sys.argv = saved_argv
    OmegaConf.set_struct(cli, False)
    dotenv_path = _resolve_under_cwd_or_install("env.yaml")
    dotenv = _load_config_yaml(dotenv_path) if dotenv_path.exists() else OmegaConf.create({})
    config_paths = OmegaConf.merge(dotenv, cli).get(CONFIG_PATHS_KEY_NAME) or []
    _, extra_configs = parser.load_extra_config_paths([str(path) for path in config_paths])
    return OmegaConf.merge(*(_load_config_yaml(path) for path in config_files), *extra_configs, dotenv, cli)


def _block_from_overrides(overrides: list[str], key: str, *, config_files: Sequence[Path] = ()) -> Any:
    """The ``key`` node (dotted) of the configuration the command names, resolved; a scalar as it is, ``None`` when
    absent. Only that node's own interpolations are evaluated, so a provider block's credentials are not needed
    to read a sibling key; a node that is configured but does not resolve (``builds: ${missing_profile}``) is
    reported as a configuration error rather than read as absent. Merged as :func:`_merged_cli_config` does."""
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationResolutionError

    from nemo_gym.config.global_config import translate_interpolation_error

    merged = _merged_cli_config(overrides, config_files)
    try:
        block = OmegaConf.select(merged, key)
        if not OmegaConf.is_config(block):
            return block
        return OmegaConf.to_container(block, resolve=True)
    except InterpolationResolutionError as exc:
        raise translate_interpolation_error(exc) from exc
