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

import importlib
from pathlib import Path
from typing import Dict, Optional

import rich
from omegaconf import OmegaConf
from pydantic import Field

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict


BENCHMARKS_DIR = PARENT_DIR / "benchmarks"


class BenchmarkConfig:
    """Represents a discovered benchmark's configuration."""

    def __init__(self, name: str, path: Path, config_dict: dict):
        self.name = name
        self.path = path
        self.config_dict = config_dict

    @property
    def agent_name(self) -> Optional[str]:
        return self.config_dict.get("agent_name")

    @property
    def num_repeats(self) -> Optional[int]:
        return self.config_dict.get("num_repeats")


def discover_benchmarks() -> Dict[str, BenchmarkConfig]:
    """Scan the benchmarks/ directory for subdirectories containing config.yaml."""
    benchmarks = {}

    if not BENCHMARKS_DIR.exists():
        return benchmarks

    for entry in sorted(BENCHMARKS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        config_path = entry / "config.yaml"
        if not config_path.exists():
            continue

        config_dict = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        benchmarks[entry.name] = BenchmarkConfig(
            name=entry.name,
            path=entry,
            config_dict=config_dict,
        )

    return benchmarks


def get_benchmark(name: str) -> BenchmarkConfig:
    """Get a specific benchmark by name. Raises ValueError if not found."""
    benchmarks = discover_benchmarks()
    if name not in benchmarks:
        available = ", ".join(benchmarks.keys()) or "(none)"
        raise ValueError(f"Benchmark '{name}' not found. Available benchmarks: {available}")
    return benchmarks[name]


def list_benchmarks() -> None:
    """CLI command: list available benchmarks."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    benchmarks = discover_benchmarks()

    if not benchmarks:
        rich.print("[yellow]No benchmarks found.[/yellow]")
        rich.print(f"Expected benchmarks directory: {BENCHMARKS_DIR}")
        return

    rich.print(f"[bold]Available Benchmarks ({len(benchmarks)})[/bold]")
    rich.print("-" * 40)
    for name, bench in benchmarks.items():
        agent = bench.agent_name or "not specified"
        repeats = bench.num_repeats or "not specified"
        rich.print(f"  [blue]{name}[/blue]  (agent: {agent}, num_repeats: {repeats})")


class PrepareBenchmarkConfig(BaseNeMoGymCLIConfig):
    """
    Prepare benchmark data by running the benchmark's prepare.py script.

    Examples:

    ```bash
    ng_prepare_benchmark +benchmark=aime24
    ```
    """

    benchmark: str = Field(description="Name of the benchmark to prepare (e.g., 'aime24').")


def prepare_benchmark() -> None:
    """CLI command: prepare benchmark data."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    config = PrepareBenchmarkConfig.model_validate(global_config_dict)

    bench = get_benchmark(config.benchmark)
    prepare_module_path = bench.path / "prepare.py"

    if not prepare_module_path.exists():
        raise FileNotFoundError(f"No prepare.py found for benchmark '{config.benchmark}' at {prepare_module_path}")

    # Import and run the benchmark's prepare function
    module_name = f"benchmarks.{config.benchmark}.prepare"
    module = importlib.import_module(module_name)

    if not hasattr(module, "prepare"):
        raise AttributeError(f"benchmarks/{config.benchmark}/prepare.py must define a `prepare()` function")

    print(f"Preparing benchmark: {config.benchmark}")
    output_path = module.prepare()
    print(f"Benchmark data prepared at: {output_path}")
