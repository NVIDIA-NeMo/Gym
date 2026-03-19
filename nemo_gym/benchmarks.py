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
from typing import Dict, List

import rich
from omegaconf import OmegaConf
from pydantic import BaseModel

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import BaseNeMoGymCLIConfig, BenchmarkDatasetConfig
from nemo_gym.global_config import (
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
    get_global_config_dict,
)


BENCHMARKS_DIR = PARENT_DIR / "benchmarks"


class BenchmarkConfig(BaseModel):
    name: str
    path: Path
    agent_name: str
    num_repeats: str
    dataset: BenchmarkDatasetConfig

    @classmethod
    def from_initial_config_dict(self, name: str, path: Path, initial_config_dict: OmegaConf) -> "BenchmarkConfig":
        initial_config_dict = OmegaConf.merge(
            initial_config_dict, GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT
        )

        parser = GlobalConfigDictParser()
        global_config_dict = parser.parse_no_environment(initial_global_config_dict=initial_config_dict)

        datasets: List[BenchmarkDatasetConfig] = []
        candidate_agent_server_instance_names: List[str] = []
        for server_instance_name in global_config_dict:
            server_config = global_config_dict[server_instance_name]
            if "responses_api_agents" not in server_config:
                continue

            inner_server_config = get_first_server_config_dict(global_config_dict, server_instance_name)

            for dataset in inner_server_config.get("datasets") or []:
                if dataset["type"] != "benchmark":
                    continue

                datasets.append(BenchmarkDatasetConfig.model_validate(dataset))
                candidate_agent_server_instance_names.append(server_instance_name)

        assert len(datasets) == 1, f"Expected 1 benchmark dataset for {name}, but found {len(datasets)}!"

        dataset = datasets[0]

        return BenchmarkConfig(
            name=name,
            path=path,
            agent_name=candidate_agent_server_instance_names[0],
            num_repeats=dataset.num_repeats,
            dataset=dataset,
        )


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

        benchmarks[entry.name] = BenchmarkConfig.from_initial_config_dict(
            name=entry.name,
            path=entry,
            initial_config_dict=OmegaConf.load(config_path),
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

    The benchmark is identified from a config_paths entry pointing to a
    benchmarks/*/config.yaml file.

    Examples:

    ```bash
    ng_prepare_benchmark "+config_paths=[benchmarks/aime24/config.yaml]"
    ```
    """


def _find_benchmark_dirs_from_config_paths(config_paths) -> list[Path]:
    """Find benchmark directories from resolved config_paths."""
    benchmark_dirs = []
    for cp in config_paths:
        cp = Path(cp)
        if not cp.is_absolute():
            cwd_path = Path.cwd() / cp
            cp = cwd_path if cwd_path.exists() else PARENT_DIR / cp
        if BENCHMARKS_DIR in cp.parents or cp.parent == BENCHMARKS_DIR:
            benchmark_dirs.append(cp.parent)
    return benchmark_dirs


def prepare_benchmark() -> None:
    """CLI command: prepare benchmark data."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    PrepareBenchmarkConfig.model_validate(global_config_dict)

    config_paths = global_config_dict.get("config_paths") or []
    benchmark_dirs = _find_benchmark_dirs_from_config_paths(config_paths)

    if not benchmark_dirs:
        raise ValueError(
            "No benchmark config found in config_paths. "
            'Pass a benchmark config, e.g.: "+config_paths=[benchmarks/aime24/config.yaml]"'
        )

    # Validate all benchmarks before preparing any
    errors = []
    validated = []
    for bench_dir in benchmark_dirs:
        benchmark_name = bench_dir.name
        prepare_module_path = bench_dir / "prepare.py"

        if not prepare_module_path.exists():
            errors.append(f"No prepare.py found for benchmark '{benchmark_name}' at {prepare_module_path}")
            continue

        module_name = f"benchmarks.{benchmark_name}.prepare"
        module = importlib.import_module(module_name)

        if not hasattr(module, "prepare"):
            errors.append(f"benchmarks/{benchmark_name}/prepare.py must define a `prepare()` function")
            continue

        validated.append((benchmark_name, module))

    if errors:
        raise ValueError("Benchmark validation failed:\n  " + "\n  ".join(errors))

    # Prepare after all validations pass
    for benchmark_name, module in validated:
        print(f"Preparing benchmark: {benchmark_name}")
        output_path = module.prepare()
        print(f"Benchmark data prepared at: {output_path}")
