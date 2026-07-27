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
"""Benchmark discovery and preparation utilities."""

import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import BenchmarkDatasetConfig
from nemo_gym.discovery import _parse_no_environment_tolerating_unset_values, discover_components
from nemo_gym.global_config import (
    POLICY_MODEL_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
)


BENCHMARKS_SUBDIR = "benchmarks"
BENCHMARKS_DIR = PARENT_DIR / BENCHMARKS_SUBDIR


class BenchmarkConfig(BaseModel):
    name: str  # this is a dataset name, not the config name (they are usually the same)
    path: Path
    agent_name: str
    num_repeats: int
    dataset: BenchmarkDatasetConfig

    @classmethod
    def from_config_path(cls, config_path: Path, *, strict: bool = True) -> "Optional[BenchmarkConfig]":
        return cls.from_initial_config_dict(
            path=config_path, initial_config_dict=OmegaConf.load(config_path), strict=strict
        )

    @classmethod
    def from_initial_config_dict(
        cls, path: Path, initial_config_dict: DictConfig, *, strict: bool = True
    ) -> "Optional[BenchmarkConfig]":
        if POLICY_MODEL_KEY_NAME not in initial_config_dict:
            initial_config_dict = OmegaConf.merge(
                initial_config_dict, GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT
            )

        # `strict=True` (default): unset `???`/`${...}` values are errors, as non-listing workflows expect.
        # `strict=False`: listing-only tolerance for those runtime-only values (see the helper's docstring).
        if strict:
            global_config_dict = GlobalConfigDictParser().parse_no_environment(
                initial_global_config_dict=initial_config_dict
            )
        else:
            global_config_dict = _parse_no_environment_tolerating_unset_values(initial_config_dict)

        datasets: List[BenchmarkDatasetConfig] = []
        candidate_agent_server_instance_names: List[str] = []
        for server_instance_name in global_config_dict:
            server_config = global_config_dict[server_instance_name]
            if not isinstance(server_config, (dict, DictConfig)) or "responses_api_agents" not in server_config:
                continue

            inner_server_config = get_first_server_config_dict(global_config_dict, server_instance_name)

            for dataset in inner_server_config.get("datasets") or []:
                if dataset["type"] != "benchmark":
                    continue

                datasets.append(BenchmarkDatasetConfig.model_validate(dataset))
                candidate_agent_server_instance_names.append(server_instance_name)

        if len(datasets) < 1:
            return

        assert len(datasets) == 1, f"Expected 1 benchmark dataset for config {path}, but found {len(datasets)}!"

        dataset = datasets[0]

        return cls(
            name=dataset.name,
            path=path,
            agent_name=candidate_agent_server_instance_names[0],
            num_repeats=dataset.num_repeats,
            dataset=dataset,
        )


def _benchmark_config_name(rel_config_path: Path) -> str:
    """The name of the benchmark config, given its path relative to ``benchmarks/``, sans ``.yaml``.

    This is the identity we key benchmarks by, so a listed benchmark is always a valid ``--benchmark`` argument.
    """
    rel = rel_config_path.with_suffix("")
    parts = rel.parts
    if len(parts) == 2 and parts[1] == "config":
        return parts[0]
    return rel.as_posix()


def _is_benchmark_config(config_path: Path) -> bool:
    """True if the config declares a `type: benchmark` dataset anywhere in its structure.

    A raw single-file parse (no `config_paths`/interpolation resolution), so it's format-agnostic and can't
    fail on includes. An unparseable file is kept (returns True) so the resolve step surfaces a diagnostic.
    """

    def declares(node: object) -> bool:
        if isinstance(node, dict):
            return node.get("type") == "benchmark" or any(declares(v) for v in node.values())
        if isinstance(node, list):
            return any(declares(v) for v in node)
        return False

    try:
        return declares(yaml.safe_load(config_path.read_text(errors="ignore")))
    except yaml.YAMLError:
        return True


def _benchmark_config_paths(benchmarks_dir: Path) -> List[Path]:
    """Sorted config paths under one dir that declare a benchmark, discovered by content.

    A config is a benchmark iff it declares a `type: benchmark` dataset, regardless of filename, so we scan
    every yaml. :func:`_is_benchmark_config` is a cheap prefilter (pay the resolve cost only on real
    candidates) that also catches non-`config.yaml` names like tau2's `configs/*.yaml`. Empty if dir missing.
    """
    if not benchmarks_dir.is_dir():
        return []
    config_paths = [benchmarks_dir / p for p in glob("**/*.yaml", root_dir=benchmarks_dir, recursive=True)]
    return sorted(p for p in config_paths if _is_benchmark_config(p))


def _discover_benchmarks_in_dir(benchmarks_dir: Path) -> Dict[str, BenchmarkConfig]:
    """Map benchmark name -> :class:`BenchmarkConfig` for every benchmark config under one dir."""
    benchmarks_dict = dict()
    for config_path in _benchmark_config_paths(benchmarks_dir):
        try:
            # Listing has no runtime context, so tolerate unset runtime-only values.
            maybe_bc = BenchmarkConfig.from_config_path(config_path, strict=False)
        except Exception as e:
            # Still unresolvable (e.g. a multi-benchmark suite) — skip with a warning rather than fail the
            # whole listing, so it isn't silently invisible.
            print(
                f"Warning: skipping benchmark config '{config_path}': could not resolve it "
                f"({type(e).__name__}: {str(e).splitlines()[0]}).",
                file=sys.stderr,
            )
            continue
        if not maybe_bc:
            continue

        benchmarks_dict[_benchmark_config_name(config_path.relative_to(benchmarks_dir))] = maybe_bc

    return benchmarks_dict


def discover_benchmarks() -> Dict[str, BenchmarkConfig]:
    """Map benchmark name -> :class:`BenchmarkConfig` for every discoverable benchmark config.

    Scans the ``benchmarks/`` subdir of every :func:`~nemo_gym.discovery.component_search_roots` root
    (``NEMO_GYM_EXTRA_ROOTS`` + cwd + built-ins), merged so user benchmarks shadow same-named built-ins.
    """
    return discover_components(BENCHMARKS_SUBDIR, _discover_benchmarks_in_dir)


# Backward-compatibility shims (CLI refactor): these symbols moved to `nemo_gym.cli.eval`.
# Re-exported lazily to avoid a circular import; accessing them emits a DeprecationWarning.
from nemo_gym.cli._compat import moved_attr_getter  # noqa: E402


__getattr__ = moved_attr_getter(
    __name__,
    {
        "list_benchmarks": "nemo_gym.cli.eval",
        "PrepareBenchmarkConfig": "nemo_gym.cli.eval",
        "prepare_benchmark": "nemo_gym.cli.eval",
    },
)
