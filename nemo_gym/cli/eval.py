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

import asyncio
import difflib
import importlib
import json
from copy import deepcopy
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import rich
from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import Field
from rich.table import Table
from tqdm.auto import tqdm

from nemo_gym.benchmarks import BENCHMARKS_DIR, BenchmarkConfig, _load_benchmarks_from_config_paths
from nemo_gym.cli.env import RunHelper
from nemo_gym.cli.utils import exit_cleanly_on_config_error, print_rich_table
from nemo_gym.config_types import BaseNeMoGymCLIConfig, BenchmarkDatasetConfig, ConfigError, ConfigPathNotFoundError
from nemo_gym.global_config import (
    JSON_OUTPUT_KEY_NAME,
    POLICY_MODEL_KEY_NAME,
    QUERY_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    GlobalConfigDictParser,
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
    loads_jsonl_line,
)
from nemo_gym.train_data_utils import TrainDataProcessor


def _fuzzy_matches(query: str, *fields: str) -> bool:
    """Whether `query` fuzzily matches any of `fields`: a substring or a close difflib match (token-aware)."""
    needle = query.lower()
    for field in fields:
        if not field:
            continue
        haystack = field.lower()
        if needle in haystack:
            return True
        tokens = haystack.replace("_", " ").replace("-", " ").split()
        if difflib.get_close_matches(needle, [haystack, *tokens], n=1, cutoff=0.70):
            return True
    return False


def _benchmark_domain(bench: BenchmarkConfig) -> str:
    """Resolve a benchmark's config to its resources server `domain` (for the domain column and `gym search`).

    `BenchmarkConfig` flattens away the resources server `domain`, so we re-resolve the config with the
    same parser `BenchmarkConfig` uses (so chained `config_paths` / `_inherit_from` are applied) and read
    the field back out.
    """
    initial_config_dict = OmegaConf.load(bench.path)
    if POLICY_MODEL_KEY_NAME not in initial_config_dict:
        initial_config_dict = OmegaConf.merge(
            initial_config_dict, GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT
        )
    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial_config_dict)

    domain = ""
    for instance_name in resolved:
        instance = resolved[instance_name]
        if not isinstance(instance, (dict, DictConfig)):
            continue

        resources_servers = instance.get("resources_servers")
        if resources_servers:
            for rs_config in resources_servers.values():
                found_domain = (rs_config or {}).get("domain")
                if found_domain:
                    domain = str(found_domain)

    return domain


def list_benchmarks() -> None:
    """CLI command: list available benchmarks, optionally filtered by a `query` (the `gym search` entry point)."""
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

    # Resolve the domain once per benchmark, for the domain column and `gym search`.
    domains = {name: _benchmark_domain(bench) for name, bench in benchmarks.items()}

    # `gym search <query>` reuses this command, narrowing the listing to fuzzy matches
    # across the benchmark name and domain.
    query = global_config_dict.get(QUERY_KEY_NAME)
    if query:
        benchmarks = {name: bench for name, bench in benchmarks.items() if _fuzzy_matches(query, name, domains[name])}

    if global_config_dict.get(JSON_OUTPUT_KEY_NAME, False):
        payload = [
            {
                "name": name,
                "agent_name": bench.agent_name,
                "domain": domains[name],
                "num_repeats": bench.num_repeats,
            }
            for name, bench in benchmarks.items()
        ]
        print(json.dumps(payload))
        return

    if not benchmarks:
        if query:
            rich.print(f"[yellow]No benchmarks match '{query}'.[/yellow]")
            return
        rich.print("[yellow]No benchmarks found.[/yellow]")
        rich.print(f"Expected benchmarks directory: {BENCHMARKS_DIR}")
        return

    title = (
        f"Benchmarks matching '{query}' ({len(benchmarks)})"
        if query
        else f"Available benchmarks in NeMo Gym ({len(benchmarks)})"
    )
    table = Table(title=title)
    table.add_column("Benchmark name")
    table.add_column("Domain")
    table.add_column("Agent name")
    table.add_column("Num repeats")

    for name, bench in benchmarks.items():
        table.add_row(name, domains[name], bench.agent_name, str(bench.num_repeats))

    print_rich_table(table)


class PrepareBenchmarkConfig(BaseNeMoGymCLIConfig):
    """
    Prepare benchmark data by running the benchmark's prepare.py script.

    The benchmark is identified from a config_paths entry pointing to a
    benchmarks/*/config.yaml file.

    Examples:

    ```bash
    gym eval prepare --benchmark aime24
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
    if output_fpath.absolute() != benchmark_config.dataset.jsonl_fpath.absolute():
        raise ConfigError(
            f"Expected the actual prepared dataset output fpath to match the jsonl_fpath set in the config. Instead got {output_fpath=} jsonl_fpath={benchmark_config.dataset.jsonl_fpath}"
        )
    print(f"Benchmark data prepared at: {output_fpath}")


@exit_cleanly_on_config_error
def prepare_benchmark() -> None:
    """CLI command: prepare benchmark data."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    prepare_benchmark_config = PrepareBenchmarkConfig.model_validate(global_config_dict)

    benchmarks_dict: Dict[str, BenchmarkConfig] = dict()
    inspected_server_instances: List[str] = []
    for server_instance_name in global_config_dict:
        server_config = global_config_dict[server_instance_name]
        if not isinstance(server_config, (dict, DictConfig)) or "responses_api_agents" not in server_config:
            continue

        inspected_server_instances.append(server_instance_name)
        inner_server_config = get_first_server_config_dict(global_config_dict, server_instance_name)

        datasets: List[BenchmarkDatasetConfig] = []
        for dataset in inner_server_config.get("datasets") or []:
            if dataset["type"] != "benchmark":
                continue

            datasets.append(BenchmarkDatasetConfig.model_validate(dataset))

        if len(datasets) < 1:
            continue

        if len(datasets) != 1:
            raise ConfigError(
                f"Expected exactly 1 benchmark dataset for server instance `{server_instance_name}`, "
                f"but found {len(datasets)}: {[d.name for d in datasets]}. "
                "A benchmark config must define a single benchmark dataset."
            )

        dataset = datasets[0]

        benchmarks_dict[server_instance_name] = BenchmarkConfig(
            name=dataset.name,
            path=Path(""),
            agent_name=server_instance_name,
            num_repeats=dataset.num_repeats,
            dataset=dataset,
        )

    if not benchmarks_dict:
        raise ConfigError(
            "No benchmark config found. "
            + (
                f"Inspected server instances {inspected_server_instances}, but none declared a `benchmark` dataset."
                if inspected_server_instances
                else "No server instances with `responses_api_agents` were found in the resolved config."
            )
            + " Pass a benchmark with `gym eval prepare --benchmark <name>` (e.g. `--benchmark aime24`)."
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
        raise ConfigError(errors_to_print)

    # Prepare after all validations pass
    if prepare_benchmark_config.num_prepare_benchmark_processes > 1:  # pragma: no cover
        with Pool(processes=prepare_benchmark_config.num_prepare_benchmark_processes) as pool:
            results = pool.imap_unordered(_multiprocess_benchmark_prepare_fn, validated)
            list(tqdm(results, total=len(validated)))
    else:
        results = map(_multiprocess_benchmark_prepare_fn, validated)
        list(tqdm(results, total=len(validated)))


@exit_cleanly_on_config_error
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

    # A benchmark can plug in a custom rollout-collection procedure via the
    # ``rollout_collection_driver`` config field (a ``module.path:function``).
    # The default path runs the built-in single-pass helper.
    driver_path = e2e_rollout_collection_config.rollout_collection_driver

    print(
        f"""Output artifacts:
1. Preprocessed datasets: {data_processor_config_dict["output_dirpath"]}
2. Dataset file used for rollout collection: {rollout_collection_config_dict["input_jsonl_fpath"]}
3. Rollout collection results file: {output_fpath}
{f"Rollout collection driver: {driver_path}" if driver_path else ""}
"""
    )
    try:
        if driver_path:
            module_name, _, fn_name = driver_path.partition(":")
            if not module_name or not fn_name:
                raise ConfigError(f"rollout_collection_driver must be 'module.path:function' (got {driver_path!r}).")
            driver_fn = getattr(importlib.import_module(module_name), fn_name)
            resolved_config = OmegaConf.to_container(global_config_dict, resolve=True)
            asyncio.run(driver_fn(rollout_collection_config, resolved_config))
        else:
            asyncio.run(rch.run_from_config(rollout_collection_config))
    except KeyboardInterrupt:
        pass
    finally:
        rh.shutdown()


@exit_cleanly_on_config_error
def collect_rollouts():  # pragma: no cover
    config = RolloutCollectionConfig.model_validate(get_global_config_dict())
    rch = RolloutCollectionHelper()

    asyncio.run(rch.run_from_config(config))


@exit_cleanly_on_config_error
def aggregate_rollouts():  # pragma: no cover
    config = RolloutAggregationConfig.model_validate(get_global_config_dict())
    rah = RolloutAggregationHelper()

    asyncio.run(rah.run_from_config(config))


@exit_cleanly_on_config_error
def reward_profile():  # pragma: no cover
    config = RewardProfileConfig.model_validate(get_global_config_dict())

    if not Path(config.materialized_inputs_jsonl_fpath).exists():
        raise ConfigPathNotFoundError(
            f"Input file not found: '{config.materialized_inputs_jsonl_fpath}' (--inputs). "
            "Check the path is spelled correctly."
        )
    if not Path(config.rollouts_jsonl_fpath).exists():
        raise ConfigPathNotFoundError(
            f"Input file not found: '{config.rollouts_jsonl_fpath}' (--rollouts). Check the path is spelled correctly."
        )

    with open(config.materialized_inputs_jsonl_fpath) as f:
        rows = [loads_jsonl_line(line, config.materialized_inputs_jsonl_fpath, i) for i, line in enumerate(f, 1)]

    with open(config.rollouts_jsonl_fpath) as f:
        results = [loads_jsonl_line(line, config.rollouts_jsonl_fpath, i) for i, line in enumerate(f, 1)]

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
