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
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rich
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field
from rich.table import Table
from tqdm.auto import tqdm

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import BaseNeMoGymCLIConfig, BenchmarkDatasetConfig
from nemo_gym.global_config import (
    POLICY_MODEL_KEY_NAME,
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
    num_repeats: int
    dataset: BenchmarkDatasetConfig

    @classmethod
    def from_config_path(cls, config_path: Path) -> "Optional[BenchmarkConfig]":
        return cls.from_initial_config_dict(path=config_path, initial_config_dict=OmegaConf.load(config_path))

    @classmethod
    def from_initial_config_dict(cls, path: Path, initial_config_dict: DictConfig) -> "Optional[BenchmarkConfig]":
        if POLICY_MODEL_KEY_NAME not in initial_config_dict:
            initial_config_dict = OmegaConf.merge(
                initial_config_dict, GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT
            )

        parser = GlobalConfigDictParser()
        global_config_dict = parser.parse_no_environment(initial_global_config_dict=initial_config_dict)

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

        maybe_bc = BenchmarkConfig.from_config_path(config_path)
        if not maybe_bc:
            continue

        benchmarks_dict[maybe_bc.name] = maybe_bc

    return benchmarks_dict


def list_benchmarks() -> None:
    """CLI command: list available benchmarks."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    assert BENCHMARKS_DIR.exists(), "Missing benchmarks directory"

    config_paths = glob("**/config.yaml", root_dir=BENCHMARKS_DIR, recursive=True)
    config_paths = [BENCHMARKS_DIR / p for p in config_paths]
    config_paths = sorted(config_paths)

    benchmarks = _load_benchmarks_from_config_paths(config_paths)

    if not benchmarks:
        rich.print("[yellow]No benchmarks found.[/yellow]")
        rich.print(f"Expected benchmarks directory: {BENCHMARKS_DIR}")
        return

    table = Table(title=f"Available benchmarks in NeMo Gym ({len(benchmarks)})")
    table.add_column("Benchmark name")
    table.add_column("Agent name")
    table.add_column("Num repeats")

    for name, bench in benchmarks.items():
        table.add_row(name, bench.agent_name, str(bench.num_repeats))

    rich.print(table)


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

    use_cached_prepared_benchmarks: bool = Field(
        default=False, description="Skip benchmark preparation if the prepared file is already present"
    )
    num_prepare_benchmark_processes: int = Field(
        default=1, description="Number of processes to parallelize benchmark preparation"
    )


def _multiprocess_benchmark_prepare_fn(args):
    benchmark_config: BenchmarkConfig
    prepare_module_path: str
    (benchmark_config, prepare_module_path) = args

    print(f"Preparing benchmark: {benchmark_config.name}")

    module = importlib.import_module(prepare_module_path)
    output_fpath = module.prepare()
    assert output_fpath.absolute() == benchmark_config.dataset.jsonl_fpath.absolute(), (
        f"Expected the actual prepared dataset output fpath to match the jsonl_fpath set in the config. Instead got {output_fpath=} jsonl_fpath={benchmark_config.dataset.jsonl_fpath}"
    )
    print(f"Benchmark data prepared at: {output_fpath}")


def prepare_benchmark() -> None:
    """CLI command: prepare benchmark data."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    prepare_benchmark_config = PrepareBenchmarkConfig.model_validate(global_config_dict)

    benchmarks_dict: Dict[str, BenchmarkConfig] = dict()
    for server_instance_name in global_config_dict:
        server_config = global_config_dict[server_instance_name]
        if not isinstance(server_config, (dict, DictConfig)) or "responses_api_agents" not in server_config:
            continue

        inner_server_config = get_first_server_config_dict(global_config_dict, server_instance_name)

        datasets: List[BenchmarkDatasetConfig] = []
        for dataset in inner_server_config.get("datasets") or []:
            if dataset["type"] != "benchmark":
                continue

            datasets.append(BenchmarkDatasetConfig.model_validate(dataset))

        if len(datasets) < 1:
            continue

        assert len(datasets) == 1, (
            f"Expected 1 benchmark dataset for `{server_instance_name}`, but found {len(datasets)}!"
        )

        dataset = datasets[0]

        benchmarks_dict[server_instance_name] = BenchmarkConfig(
            name=dataset.name,
            path=Path(""),
            agent_name=server_instance_name,
            num_repeats=dataset.num_repeats,
            dataset=dataset,
        )

    assert benchmarks_dict, (
        'No benchmark config found in config_paths. Pass a benchmark config, e.g.: "+config_paths=[benchmarks/aime24/config.yaml]"'
    )

    # Validate all benchmarks before preparing any
    prepare_script_missing: List[BenchmarkConfig] = []
    prepare_function_missing: List[BenchmarkConfig] = []

    validated: List[Tuple[BenchmarkConfig, str]] = []
    already_prepared: List[BenchmarkConfig] = []
    for benchmark_config in benchmarks_dict.values():
        prepare_script_path = benchmark_config.dataset.prepare_script
        if not prepare_script_path.exists():
            prepare_script_missing.append(benchmark_config)
            continue

        prepare_module_path = ".".join(prepare_script_path.with_suffix("").parts)
        module = importlib.import_module(prepare_module_path)
        if not hasattr(module, "prepare"):
            prepare_function_missing.append(benchmark_config)
            continue

        is_already_prepared = benchmark_config.dataset.jsonl_fpath.exists()
        if prepare_benchmark_config.use_cached_prepared_benchmarks and is_already_prepared:
            already_prepared.append(benchmark_config)
            continue

        validated.append((benchmark_config, prepare_module_path))

    if already_prepared:
        already_prepared_str = "".join(f"- {bc.name}: {bc.dataset.jsonl_fpath}\n" for bc in already_prepared)
        already_prepared_str = f"""The following benchmarks have already been prepared. Since `use_cached_prepared_benchmarks=true`, we will skip re-preparation of those benchmarks.
        {already_prepared_str}"""
        print(already_prepared_str)

    errors_to_print = ""
    if prepare_script_missing:
        prepare_script_missing_str = "".join(
            f"- {bc.name}: {bc.dataset.prepare_script}\n" for bc in prepare_script_missing
        )
        errors_to_print += f"""The following benchmarks are missing a valid prepare script:
{prepare_script_missing_str}
"""
    if prepare_function_missing:
        prepare_function_missing_str = "".join(
            f"- {bc.name}: {bc.dataset.prepare_script}\n" for bc in prepare_function_missing
        )
        errors_to_print += f"""The following benchmarks have a prepare script, but are missing the prepare function:
{prepare_function_missing_str}
"""
    if errors_to_print:
        errors_to_print = f"""Did not prepare any benchmarks due to benchmark config errors.
{errors_to_print}"""
        raise RuntimeError(errors_to_print)

    # Prepare after all validations pass
    if prepare_benchmark_config.num_prepare_benchmark_processes > 1:
        with Pool(processes=prepare_benchmark_config.num_prepare_benchmark_processes) as pool:
            results = pool.imap_unordered(_multiprocess_benchmark_prepare_fn, validated)
            list(tqdm(results, total=len(validated)))
    else:
        results = map(_multiprocess_benchmark_prepare_fn, validated)
        list(tqdm(results, total=len(validated)))
