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
import importlib
import json
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import Field
from rich.table import Table
from tqdm.auto import tqdm

from nemo_gym.benchmarks import (
    BenchmarkConfig,
    discover_benchmarks,
)
from nemo_gym.cli.env import RunHelper
from nemo_gym.cli.utils import (
    exit_cleanly_on_config_error,
    exit_unknown_component,
    fuzzy_matches,
    print_no_matches,
    print_rich_table,
    render_component_inspection,
)
from nemo_gym.config_types import BaseNeMoGymCLIConfig, BenchmarkDatasetConfig, ConfigError, ConfigPathNotFoundError
from nemo_gym.discovery import read_config_metadata
from nemo_gym.environment_manifest import parse_python_callable_reference
from nemo_gym.environment_validation import validate_rollout_driver_contract
from nemo_gym.global_config import (
    COMPONENT_NAME_KEY_NAME,
    JSON_OUTPUT_KEY_NAME,
    QUERY_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
    get_global_config_dict,
)


# NOTE: `reward_profile`, `rollout_collection`, `rollout_reverification` and `train_data_utils` are imported lazily inside the run/aggregate/
# profile commands below: they pull in heavy deps (wandb, mlflow, anthropic) that the fast `list`/`search`
# commands in this module must not pay for on every invocation.


def _load_rollout_collection_driver(reference: str | None) -> Callable[..., Any] | None:
    if reference is None:
        return None
    try:
        module_name, function_name = parse_python_callable_reference(
            reference,
            field_name="rollout_collection_driver",
        )
        module = importlib.import_module(module_name)
    except (ImportError, ValueError) as error:
        raise ConfigError(f"Cannot load rollout_collection_driver {reference!r}: {error}") from error

    try:
        driver = getattr(module, function_name)
    except AttributeError as error:
        raise ConfigError(
            f"rollout_collection_driver {reference!r} does not name a function in module {module_name!r}."
        ) from error
    if not callable(driver):
        raise ConfigError(f"rollout_collection_driver {reference!r} is not callable.")
    return driver


async def _execute_rollout_collection(
    config: Any,
    global_config_dict: DictConfig,
    driver_fn: Callable[..., Any] | None,
    *,
    captured_environment: Any = None,
) -> None:
    if driver_fn is None:
        from nemo_gym.rollout_collection import RolloutCollectionHelper

        await RolloutCollectionHelper().run_from_config(config)
        return

    resolved_config = OmegaConf.to_container(global_config_dict, resolve=True)
    await driver_fn(config, resolved_config)

    from nemo_gym.trajectory_bundle import write_trajectory_bundle

    bundle_path = write_trajectory_bundle(
        rollouts_path=config.output_jsonl_fpath,
        materialized_inputs_path=config.materialized_jsonl_fpath,
        environment=captured_environment,
        identity_fields=config.trajectory_identity_fields,
    )
    print(f"Trajectory bundle: {bundle_path}")


def _inspect_benchmark(name: str, benchmarks: dict, global_config_dict) -> None:
    """Render the ``gym list benchmarks <name>`` inspect view for one benchmark."""
    bench = benchmarks.get(name)
    if bench is None:
        exit_unknown_component(name, benchmarks, "benchmark")
        return

    domain, description = read_config_metadata(bench.path)
    details = {
        "config": str(bench.path.resolve()),
        "agent": bench.agent_name,
        "num repeats": str(bench.num_repeats),
        "dataset": str(bench.dataset.jsonl_fpath),
        "prepare script": str(bench.dataset.prepare_script),
    }
    render_component_inspection(
        json_output=global_config_dict.get(JSON_OUTPUT_KEY_NAME, False),
        name=name,
        type_noun="benchmark",
        domain=domain,
        description=description,
        details=details,
        usage=f"gym eval prepare --benchmark {name}\ngym eval run --benchmark {name} --model-type vllm_model",
    )


def list_benchmarks() -> None:
    """List available benchmarks, or inspect one by name (``gym list benchmarks <name>``). Optionally filtered
    by a `query` (the `gym search` entry point).

    A benchmark is a specific kind of environment, so it shares `gym list environments`' columns (name,
    domain, description) and reads them through the same `read_config_metadata` helper. ``--search-dir``
    adds extra roots to scan on top of the cwd and built-ins.
    """
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    benchmarks = discover_benchmarks()

    name = global_config_dict.get(COMPONENT_NAME_KEY_NAME)
    if name:
        _inspect_benchmark(name, benchmarks, global_config_dict)
        return

    # Resolve domain + description once per benchmark, via the shared component-metadata reader —
    # the same one `gym list environments` uses — for the columns and `gym search`.
    metadata = {name: read_config_metadata(bench.path) for name, bench in benchmarks.items()}

    # `gym search <query>` reuses this command, narrowing the listing to fuzzy matches
    # across the benchmark config name, its dataset name, domain, and description.
    query = global_config_dict.get(QUERY_KEY_NAME)
    if query:
        benchmarks = {
            name: bench
            for name, bench in benchmarks.items()
            if fuzzy_matches(query, name, bench.name, metadata[name][0] or "", metadata[name][1] or "")
        }

    if global_config_dict.get(JSON_OUTPUT_KEY_NAME, False):
        payload = [
            {
                "name": name,
                "agent_name": bench.agent_name,
                "domain": metadata[name][0] or "",
                "num_repeats": bench.num_repeats,
                "description": metadata[name][1] or "",
            }
            for name, bench in benchmarks.items()
        ]
        print(json.dumps(payload))
        return

    if not benchmarks:
        print_no_matches("benchmarks", query)
        return

    title = (
        f"Benchmarks matching '{query}' ({len(benchmarks)})"
        if query
        else f"Available benchmarks in NeMo Gym ({len(benchmarks)})"
    )
    table = Table(title=title)
    # Shared environment columns first (name, domain, description), then benchmark-specific ones.
    table.add_column("Name")
    table.add_column("Domain")
    table.add_column("Description")
    table.add_column("Agent name")
    table.add_column("Num repeats")

    for name, bench in benchmarks.items():
        domain, description = metadata[name]
        table.add_row(name, domain or "", description or "", bench.agent_name, str(bench.num_repeats))

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
    prepare_script_args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments forwarded to the benchmark's prepare() function"
    )


def _multiprocess_benchmark_prepare_fn(args):
    benchmark_config: BenchmarkConfig
    prepare_module_path: str
    prepare_script_args: Dict[str, Any]
    (benchmark_config, prepare_module_path, prepare_script_args) = args

    print(f"Preparing benchmark: {benchmark_config.name}")

    module = importlib.import_module(prepare_module_path)
    output_fpath = module.prepare(**prepare_script_args)
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

        validated.append((benchmark_config, prepare_module_path, dict(prepare_benchmark_config.prepare_script_args)))

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
    from nemo_gym.rollout_collection import (
        E2ERolloutCollectionConfig,
        RolloutCollectionConfig,
    )
    from nemo_gym.train_data_utils import TrainDataProcessor

    global_config_dict = get_global_config_dict()
    if not global_config_dict.get("manifest_path"):
        validate_rollout_driver_contract(global_config_dict)

    # Ensure we have the right config first thing
    e2e_rollout_collection_config = E2ERolloutCollectionConfig.model_validate(global_config_dict)
    driver_path = e2e_rollout_collection_config.rollout_collection_driver
    driver_fn = _load_rollout_collection_driver(driver_path)
    driver_capture_environment = None
    if driver_fn is not None:
        from nemo_gym.trajectory_bundle import captured_environment_from_config

        driver_capture_environment = captured_environment_from_config(global_config_dict)

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
    rh.start(None, global_config_dict=global_config_dict)

    # A benchmark can plug in a custom rollout-collection procedure via the
    # ``rollout_collection_driver`` config field (a ``module.path:function``).
    # The default path runs the built-in single-pass helper.
    print(
        f"""Output artifacts:
1. Preprocessed datasets: {data_processor_config_dict["output_dirpath"]}
2. Dataset file used for rollout collection: {rollout_collection_config_dict["input_jsonl_fpath"]}
3. Rollout collection results file: {output_fpath}
{f"Rollout collection driver: {driver_path}" if driver_path else ""}
"""
    )
    try:
        asyncio.run(
            _execute_rollout_collection(
                rollout_collection_config,
                global_config_dict,
                driver_fn,
                captured_environment=driver_capture_environment,
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        rh.shutdown()


@exit_cleanly_on_config_error
def collect_rollouts():  # pragma: no cover
    from nemo_gym.rollout_collection import RolloutCollectionConfig

    global_config_dict = get_global_config_dict()
    if not global_config_dict.get("manifest_path"):
        validate_rollout_driver_contract(global_config_dict)
    config = RolloutCollectionConfig.model_validate(global_config_dict)
    driver_fn = _load_rollout_collection_driver(config.rollout_collection_driver)
    driver_capture_environment = None
    if driver_fn is not None:
        from nemo_gym.trajectory_bundle import captured_environment_from_config

        driver_capture_environment = captured_environment_from_config(global_config_dict)

    asyncio.run(
        _execute_rollout_collection(
            config,
            global_config_dict,
            driver_fn,
            captured_environment=driver_capture_environment,
        )
    )


@exit_cleanly_on_config_error
def aggregate_rollouts():  # pragma: no cover
    from nemo_gym.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper

    config = RolloutAggregationConfig.model_validate(get_global_config_dict())
    rah = RolloutAggregationHelper()

    asyncio.run(rah.run_from_config(config))


@exit_cleanly_on_config_error
def reverify_rollouts():  # pragma: no cover
    from nemo_gym.rollout_reverification import (
        RolloutReverificationConfig,
        RolloutReverificationHelper,
        reverification_fingerprint,
    )

    global_config = get_global_config_dict()
    rh = RunHelper()
    rh.start(None, global_config_dict=global_config)
    with open_dict(global_config):
        global_config["verifier_fingerprint"] = reverification_fingerprint(global_config)
    config = RolloutReverificationConfig.model_validate(global_config)
    rrh = RolloutReverificationHelper()

    asyncio.run(rrh.run_from_config(config))


@exit_cleanly_on_config_error
def reward_profile():  # pragma: no cover
    from nemo_gym.reward_profile import RewardProfileConfig, RewardProfiler
    from nemo_gym.rollout_collection import loads_jsonl_line

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
