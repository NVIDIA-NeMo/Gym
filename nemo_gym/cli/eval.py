# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import importlib
import json
from copy import deepcopy
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import orjson
import rich
from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import Field
from rich.table import Table
from tqdm.auto import tqdm

from nemo_gym.benchmarks import BENCHMARKS_DIR, BenchmarkConfig, _load_benchmarks_from_config_paths
from nemo_gym.cli.env import RunHelper
from nemo_gym.config_types import BaseNeMoGymCLIConfig, BenchmarkDatasetConfig
from nemo_gym.global_config import (
    JSON_OUTPUT_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.reward_profile import RewardProfileConfig, RewardProfiler
from nemo_gym.rollout_collection import (
    E2ERolloutCollectionConfig,
    RolloutAggregationConfig,
    RolloutAggregationHelper,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
)
from nemo_gym.train_data_utils import TrainDataProcessor


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

    if global_config_dict.get(JSON_OUTPUT_KEY_NAME, False):
        payload = [
            {"name": name, "agent_name": bench.agent_name, "num_repeats": bench.num_repeats}
            for name, bench in benchmarks.items()
        ]
        print(json.dumps(payload))
        return

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
    if prepare_function_missing:  # pragma: no cover
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
    if prepare_benchmark_config.num_prepare_benchmark_processes > 1:  # pragma: no cover
        with Pool(processes=prepare_benchmark_config.num_prepare_benchmark_processes) as pool:
            results = pool.imap_unordered(_multiprocess_benchmark_prepare_fn, validated)
            list(tqdm(results, total=len(validated)))
    else:
        results = map(_multiprocess_benchmark_prepare_fn, validated)
        list(tqdm(results, total=len(validated)))


def e2e_rollout_collection():  # pragma: no cover
    global_config_dict = get_global_config_dict()

    # Ensure we have the right config first thing
    e2e_rollout_collection_config = E2ERolloutCollectionConfig.model_validate(global_config_dict)

    # Prepare data
    data_processor_config_dict = deepcopy(global_config_dict)
    with open_dict(data_processor_config_dict):
        data_processor_config_dict["should_download"] = True
        data_processor_config_dict["mode"] = "train_preparation"

        output_fpath = Path(e2e_rollout_collection_config.output_jsonl_fpath)
        data_process_output_dir = output_fpath.parent / "preprocessed_datasets"
        data_processor_config_dict["output_dirpath"] = str(data_process_output_dir)

    input_jsonl_fpath = data_process_output_dir / f"{e2e_rollout_collection_config.split}.jsonl"
    should_skip_data_processing = (
        e2e_rollout_collection_config.reuse_existing_data_preparation and input_jsonl_fpath.exists()
    )
    if not should_skip_data_processing:
        if e2e_rollout_collection_config.reuse_existing_data_preparation:
            print(
                f"Even though the `reuse_existing_data_preparation=true` flag was set, we will still do data preparation since the final input jsonl fpath `{input_jsonl_fpath}` does not exist yet"
            )

        data_processor = TrainDataProcessor()
        data_processor.run(data_processor_config_dict)
    else:
        print(
            f"Skipping data preparation since `reuse_existing_data_preparation=true` and the final input jsonl fpath `{input_jsonl_fpath}` already exists"
        )

    # Convert to RolloutCollectionConfig
    rollout_collection_config_dict = deepcopy(global_config_dict)
    with open_dict(rollout_collection_config_dict):
        assert input_jsonl_fpath.exists(), input_jsonl_fpath
        rollout_collection_config_dict["input_jsonl_fpath"] = str(input_jsonl_fpath)

    rollout_collection_config = RolloutCollectionConfig.model_validate(
        OmegaConf.to_container(rollout_collection_config_dict)
    )

    rh = RunHelper()
    rh.start(None)

    rch = RolloutCollectionHelper()

    print(
        f"""Output artifacts:
1. Preprocessed datasets: {data_processor_config_dict["output_dirpath"]}
2. Dataset file used for rollout collection: {rollout_collection_config_dict["input_jsonl_fpath"]}
3. Rollout collection results file: {output_fpath}
"""
    )
    try:
        asyncio.run(rch.run_from_config(rollout_collection_config))
    except KeyboardInterrupt:
        pass
    finally:
        rh.shutdown()


def collect_rollouts():  # pragma: no cover
    config = RolloutCollectionConfig.model_validate(get_global_config_dict())
    rch = RolloutCollectionHelper()

    asyncio.run(rch.run_from_config(config))


def aggregate_rollouts():  # pragma: no cover
    config = RolloutAggregationConfig.model_validate(get_global_config_dict())
    rah = RolloutAggregationHelper()

    asyncio.run(rah.run_from_config(config))


def reward_profile():  # pragma: no cover
    config = RewardProfileConfig.model_validate(get_global_config_dict())

    with open(config.materialized_inputs_jsonl_fpath) as f:
        rows = list(map(orjson.loads, f))

    with open(config.rollouts_jsonl_fpath) as f:
        results = list(map(orjson.loads, f))

    # Results may be out of order.
    results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))

    rp = RewardProfiler()
    group_level_metrics, agent_level_metrics = rp.profile_from_data(
        rows, results, allow_partial_rollouts=config.allow_partial_rollouts
    )
    completion_summary = rp.profile_completion_summary(rows, results)
    reward_profiling_fpath, agent_level_metrics_fpath = rp.write_to_disk(
        group_level_metrics, agent_level_metrics, Path(config.rollouts_jsonl_fpath)
    )

    print(f"""Profiling outputs:
Reward profile completion: {completion_summary["completed_rollout_rows"]}/{completion_summary["expected_rollout_rows"]} rollout rows ({completion_summary["reward_profile_completion_pct"]:.2f}%)
Input rows: {completion_summary["total_input_rows"]} total; {completion_summary["complete_input_rows"]} complete; {completion_summary["partial_input_rows"]} partial; {completion_summary["missing_input_rows"]} without rollouts dropped from output.
Reward profiling outputs: {reward_profiling_fpath}
Agent-level metrics: {agent_level_metrics_fpath}""")
