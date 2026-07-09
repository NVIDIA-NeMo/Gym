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

import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError
from pydantic import BaseModel

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import BenchmarkDatasetConfig
from nemo_gym.global_config import (
    POLICY_MODEL_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
)


BENCHMARKS_DIR = PARENT_DIR / "benchmarks"

# Fills unset `???`/`${...}` values during listing: they reference runtime-only values (API keys,
# endpoints) not needed to identify a benchmark, so a placeholder lets the config still resolve.
_UNSET_VALUE_PLACEHOLDER = "__unset_for_listing__"


def _parse_no_environment_tolerating_unset_values(initial_config_dict: DictConfig) -> DictConfig:
    """`parse_no_environment` for *listing*: fill unset `???` values and undefined `${...}` interpolations
    with a placeholder so a benchmark referencing runtime-only values can still be identified. Never mutates
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


class BenchmarkConfig(BaseModel):
    name: str
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


def _load_benchmarks_from_config_paths(config_paths: List[Path]) -> Dict[str, BenchmarkConfig]:
    benchmarks_dict = dict()
    for config_path in config_paths:
        config_path = Path(config_path)

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

        benchmarks_dict[maybe_bc.name] = maybe_bc

    return benchmarks_dict


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
