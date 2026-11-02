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
import inspect
import json
import logging
import os
import re
import signal
import site
import subprocess
import sys
import sysconfig
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import Field
from rich.table import Table
from tqdm.auto import tqdm

from nemo_gym import PARENT_DIR
from nemo_gym.artifacts.exporters import mark_run_failed
from nemo_gym.artifacts.run_files import QUALITY_SUMMARY_FILENAME as QUALITY_SUMMARY_FILENAME
from nemo_gym.artifacts.run_files import ROLLOUT_VERDICTS_FILENAME as ROLLOUT_VERDICTS_FILENAME
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
from nemo_gym.config.config_types import (
    BaseNeMoGymCLIConfig,
    BenchmarkDatasetConfig,
    ConfigError,
    ConfigPathNotFoundError,
    DatasetConfig,
    ServerInstanceConfig,
)
from nemo_gym.config.global_config import (
    COMPONENT_NAME_KEY_NAME,
    JSON_OUTPUT_KEY_NAME,
    QUERY_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
    get_global_config_dict,
    resolve_dataset_agent,
    taskset_environment_server_name,
)
from nemo_gym.discovery import read_config_metadata
from nemo_gym.eval_plan import (
    PlanBenchmark,
    PlanGym,
    build_eval_plan,
    eval_plan_json_schema,
    write_eval_plan,
)
from nemo_gym.health.types import RunHealthConfig
from nemo_gym.orchestration.jobs import installed_gym_commit
from nemo_gym.package_info import __version__


logger = logging.getLogger(__name__)


# NOTE: `reward_profile`, `rollout_collection`, `rollout_reverification` and `train_data_utils` are imported lazily inside the run/aggregate/
# profile commands below: they pull in heavy deps (wandb, mlflow, anthropic) that the fast `list`/`search`
# commands in this module must not pay for on every invocation.


def _inspect_benchmark(name: str, benchmarks: dict, global_config_dict) -> None:
    """Render the ``gym list benchmarks <name>`` inspect view for one benchmark."""
    bench = benchmarks.get(name)
    if bench is None:
        exit_unknown_component(name, benchmarks, "benchmark")
        return

    domain, description = read_config_metadata(bench.path)
    details = {
        "config": str(bench.path.resolve()),
        "agent": bench.agent_name or "",
        "environment server": bench.environment_server or "",
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
                "environment_server": bench.environment_server,
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
        # A taskset routed to a server that fronts several agents has no single agent; name the server instead.
        agent = bench.agent_name or f"{bench.environment_server} (environment server)"
        table.add_row(name, domain or "", description or "", agent, str(bench.num_repeats))

    print_rich_table(table)


class PrepareBenchmarkConfig(BaseNeMoGymCLIConfig):
    """
    Prepare benchmark data by running the benchmark's prepare.py script.

    The benchmark is identified from a config_paths entry pointing to a
    benchmarks/*/config.yaml file.

    With `use_cached_prepared_benchmarks=true`, an existing prepared file is reused. A prepare.py whose output
    depends on its own code (for example, settings written into each row) can define
    `is_prepared_data_current(fpath: Path) -> bool`; when it returns False, the cached file is prepared again.

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

    print(f"Preparing dataset: {benchmark_config.name}")

    module = importlib.import_module(prepare_module_path)
    output_fpath = module.prepare(**prepare_script_args)
    expected_output_fpath = Path(benchmark_config.dataset.jsonl_fpath).absolute()
    if output_fpath.absolute() != expected_output_fpath:
        raise ConfigError(
            f"Expected the actual prepared dataset output fpath to match the jsonl_fpath set in the config. "
            f"Instead got {output_fpath=} jsonl_fpath={expected_output_fpath}"
        )
    print(f"Dataset prepared at: {output_fpath}")


def _install_prepare_dependencies(benchmark_config: "BenchmarkConfig") -> None:
    """Install what a benchmark's prepare script imports, before importing it.

    Gym cannot depend on every benchmark's data-prep requirements, so a benchmark
    needing something extra had to shell out to pip from inside the prepare script
    itself. Declaring it on the dataset puts it in the config instead.
    """
    dependencies = benchmark_config.dataset.prepare_dependencies
    if not dependencies:
        return
    logger.info("Installing prepare dependencies for %s: %s", benchmark_config.name, " ".join(dependencies))
    try:
        subprocess.run(["uv", "pip", "install", "--python", sys.executable, *dependencies], check=True)
        # An editable install only adds a .pth file, which `site` reads at
        # interpreter startup -- this process would not see it otherwise.
        importlib.invalidate_caches()
        site.addsitedir(sysconfig.get_paths()["purelib"])
    except FileNotFoundError as exc:
        raise ConfigError(
            f"`uv` is required to install prepare_dependencies for dataset '{benchmark_config.name}'."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ConfigError(
            f"Could not install prepare_dependencies for dataset '{benchmark_config.name}': {' '.join(dependencies)}"
        ) from exc


def _datasets_to_prepare(global_config_dict: DictConfig) -> List[BenchmarkConfig]:
    """Every dataset the resolved config prepares: benchmark datasets and other datasets with a prepare_script."""
    # A benchmark dataset may be declared by an agent block (legacy) or a resources server
    # block (decoupled layout). `resolve_dataset_agent` is the same resolver rollout dispatch
    # uses, so preparation and rollout always agree.
    datasets_to_prepare: List[BenchmarkConfig] = []
    inspected_server_instances: List[str] = []
    for server_instance_name in global_config_dict:
        server_config = global_config_dict[server_instance_name]
        if not isinstance(server_config, (dict, DictConfig)):
            continue
        is_agent = "responses_api_agents" in server_config
        if not is_agent and "resources_servers" not in server_config:
            continue

        inspected_server_instances.append(server_instance_name)
        inner_server_config = get_first_server_config_dict(global_config_dict, server_instance_name)

        datasets: List[BenchmarkDatasetConfig | DatasetConfig] = []
        for dataset in inner_server_config.get("datasets") or []:
            if dataset["type"] == "benchmark":
                datasets.append(BenchmarkDatasetConfig.model_validate(dataset))
            elif dataset.get("prepare_script"):
                datasets.append(DatasetConfig.model_validate(dataset))
            else:
                continue

        for dataset in datasets:
            try:
                agent_name = resolve_dataset_agent(
                    global_config_dict,
                    str(server_instance_name),
                    pin=getattr(dataset, "agent", None),
                    taskset=dataset.taskset,
                )
            except ConfigError as e:
                raise ConfigError(f"Dataset {dataset.name!r}: {e}") from e

            datasets_to_prepare.append(
                BenchmarkConfig(
                    name=dataset.name,
                    path=Path(""),
                    agent_name=agent_name,
                    num_repeats=dataset.num_repeats,
                    dataset=dataset,
                    environment_server=taskset_environment_server_name(global_config_dict, dataset.taskset),
                )
            )

    if not datasets_to_prepare:
        raise ConfigError(
            "No preparable dataset config found. "
            + (
                f"Inspected server instances {inspected_server_instances}, but none declared a `benchmark` dataset "
                "or another dataset with `prepare_script`."
                if inspected_server_instances
                else "No server instances with `responses_api_agents` were found in the resolved config."
            )
            + " Pass a config with `--config <config path>` or a benchmark with `--benchmark <name>`."
        )

    return datasets_to_prepare


def _prepare_datasets(datasets: Sequence[BenchmarkConfig], prepare_benchmark_config: PrepareBenchmarkConfig) -> None:
    """Run each dataset's prepare script, after validating all of them."""
    # Validate all datasets before preparing any
    prepare_script_missing: List[BenchmarkConfig] = []
    prepare_function_missing: List[BenchmarkConfig] = []

    validated: List[Tuple[BenchmarkConfig, str]] = []
    already_prepared: List[BenchmarkConfig] = []
    for benchmark_config in datasets:
        prepare_script = benchmark_config.dataset.prepare_script
        if prepare_script is None:
            raise ConfigError(f"Dataset {benchmark_config.name!r} has no prepare_script")
        prepare_script_path = Path(prepare_script)
        if not prepare_script_path.exists():
            prepare_script_missing.append(benchmark_config)
            continue

        prepare_module_path = ".".join(prepare_script_path.with_suffix("").parts)
        _install_prepare_dependencies(benchmark_config)
        module = importlib.import_module(prepare_module_path)
        if not hasattr(module, "prepare"):
            prepare_function_missing.append(benchmark_config)
            continue

        is_already_prepared = Path(benchmark_config.dataset.jsonl_fpath).exists()
        if prepare_benchmark_config.use_cached_prepared_benchmarks and is_already_prepared:
            is_current = getattr(module, "is_prepared_data_current", None)
            if callable(is_current) and not is_current(benchmark_config.dataset.jsonl_fpath):
                print(
                    f"The cached file for {benchmark_config.name} ({benchmark_config.dataset.jsonl_fpath}) "
                    "is out of date, so it will be prepared again."
                )
            else:
                already_prepared.append(benchmark_config)
                continue

        validated.append((benchmark_config, prepare_module_path, dict(prepare_benchmark_config.prepare_script_args)))

    if already_prepared:
        already_prepared_str = "".join(f"- {bc.name}: {bc.dataset.jsonl_fpath}\n" for bc in already_prepared)
        already_prepared_str = f"""The following datasets have already been prepared. Since `use_cached_prepared_benchmarks=true`, we will skip re-preparation of those datasets.
        {already_prepared_str}"""
        print(already_prepared_str)

    errors_to_print = ""
    if prepare_script_missing:
        prepare_script_missing_str = "".join(
            f"- {bc.name}: {bc.dataset.prepare_script}\n" for bc in prepare_script_missing
        )
        errors_to_print += f"""The following datasets are missing a valid prepare script:
{prepare_script_missing_str}
"""
    if prepare_function_missing:  # pragma: no cover
        prepare_function_missing_str = "".join(
            f"- {bc.name}: {bc.dataset.prepare_script}\n" for bc in prepare_function_missing
        )
        errors_to_print += f"""The following datasets have a prepare script, but are missing the prepare function:
{prepare_function_missing_str}
"""
    if errors_to_print:
        errors_to_print = f"""Did not prepare any datasets due to dataset config errors.
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
def prepare_benchmark() -> None:
    """CLI command: prepare benchmark data."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    prepare_benchmark_config = PrepareBenchmarkConfig.model_validate(global_config_dict)
    _prepare_datasets(_datasets_to_prepare(global_config_dict), prepare_benchmark_config)


class PlanBenchmarkConfig(PrepareBenchmarkConfig):
    """
    Prepare one benchmark and write its plan: every task once, with a stable id and a digest of its input.

    The plan file's format is versioned and printed by `--schema`.
    Benchmarks whose rows have no `task_id`/`problem_id`/`instance_id` are refused.

    Examples:

    ```bash
    gym eval plan --benchmark terminal_bench_2_1/terminus_2 --out plan.json
    gym eval plan --schema
    ```
    """

    plan_output_fpath: Path | None = Field(default=None, description="Where to write the plan file")
    plan_task_ids: List[str] | None = Field(default=None, description="Only plan these task ids")
    print_plan_schema: bool = Field(default=False, description="Print the plan file's JSON Schema and exit")


def _gym_relative_path(path: str) -> str:
    """A config path relative to the Gym checkout when it lies inside it, so plans match across machines."""
    try:
        return str(Path(path).resolve().relative_to(PARENT_DIR.resolve()))
    except ValueError:
        return str(path)


@exit_cleanly_on_config_error
def plan_benchmark() -> None:
    """CLI command: write a benchmark's evaluation plan."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    plan_config = PlanBenchmarkConfig.model_validate(global_config_dict)
    if plan_config.print_plan_schema:
        print(eval_plan_json_schema(), end="")
        return
    if plan_config.plan_output_fpath is None:
        raise ConfigError("Pass `--out <path>` for the plan file.")

    # Other datasets with a prepare script are not benchmarks, so they are not part of the plan.
    benchmarks = [
        dataset
        for dataset in _datasets_to_prepare(global_config_dict)
        if isinstance(dataset.dataset, BenchmarkDatasetConfig)
    ]
    if len(benchmarks) != 1:
        raise ConfigError(
            f"A plan covers exactly one benchmark, but the config declares {len(benchmarks)}: "
            f"{[benchmark.name for benchmark in benchmarks]}."
        )
    benchmark = benchmarks[0]

    _prepare_datasets([benchmark], plan_config)
    rows = [json.loads(line) for line in benchmark.dataset.jsonl_fpath.read_text().splitlines() if line.strip()]
    plan = build_eval_plan(
        rows,
        benchmark=PlanBenchmark(
            name=benchmark.name,
            config_paths=[_gym_relative_path(path) for path in global_config_dict.get("config_paths") or []],
        ),
        taskset=benchmark.dataset.taskset or benchmark.name,
        gym=PlanGym(version=__version__, commit=installed_gym_commit()),
        selected_task_ids=plan_config.plan_task_ids,
    )
    write_eval_plan(plan, plan_config.plan_output_fpath)
    print(f"Wrote the plan for {benchmark.name} ({len(plan.tasks)} tasks) to {plan_config.plan_output_fpath}")


def _validate_split_datasets_declared(split: str, server_instance_configs: Sequence[ServerInstanceConfig]) -> None:
    """Fail fast when no config declares a dataset of the requested split's type.

    Data preparation silently produces nothing for such a split, so without this check the run
    walks the entire preparation sequence (including its success banners) and only dies later
    trying to read the collated split file.
    """
    declared_lines: List[str] = []
    declared_types: set = set()
    example_fpaths: List[str] = []
    for c in server_instance_configs:
        if c.SERVER_TYPE not in ("responses_api_agents", "resources_servers"):
            continue
        for d in c.datasets or []:
            declared_types.add(d.type)
            declared_lines.append(f"- {c.name}: {d.name} (type: {d.type})")
            if d.type == "example":
                example_fpaths.append(str(d.jsonl_fpath))
    if split in declared_types:
        return

    declared_str = "\n".join(declared_lines) if declared_lines else "- (none)"
    message = (
        f"No dataset of type `{split}` is declared in this config, so `--split {split}` has nothing to run.\n"
        f"Declared datasets:\n{declared_str}"
    )
    if example_fpaths:
        example_fpaths_str = "\n".join(
            f"  gym eval run --no-serve --input {fpath} --output <out>.jsonl" for fpath in example_fpaths
        )
        message += (
            "\nExample datasets are committed smoke-test samples and are not runnable via --split. "
            "To run one, start the servers (gym env start ...) and collect against the file directly:\n"
            f"{example_fpaths_str}"
        )
    raise ConfigError(message)


def _validate_prepared_split_file_exists(input_jsonl_fpath: Path, split: str, output_dirpath: Path) -> None:
    """Explicit check (not an assert: user-facing, and must survive `python -O`)."""
    if input_jsonl_fpath.exists():
        return
    prepared = sorted(p.name for p in output_dirpath.glob("*.jsonl")) if output_dirpath.exists() else []
    raise ConfigError(
        f"Data preparation did not produce `{input_jsonl_fpath}` for split `{split}`. "
        f"Files prepared under `{output_dirpath}`: {prepared if prepared else 'none'}."
    )


@exit_cleanly_on_config_error
def e2e_rollout_collection():  # pragma: no cover
    from nemo_gym.artifacts.resolved_config import write_resolved_config
    from nemo_gym.rollout_collection import (
        E2ERolloutCollectionConfig,
        RolloutCollectionConfig,
        RolloutCollectionHelper,
    )
    from nemo_gym.train_data_utils import TrainDataProcessor

    global_config_dict = get_global_config_dict()

    # Ensure we have the right config first thing
    e2e_rollout_collection_config = E2ERolloutCollectionConfig.model_validate(global_config_dict)

    # Prepare data
    data_processor_config_dict = deepcopy(global_config_dict)
    with open_dict(data_processor_config_dict):
        data_processor_config_dict["should_download"] = True
        data_processor_config_dict["mode"] = "train_preparation"

        output_fpath = Path(e2e_rollout_collection_config.output_jsonl_fpath)
        data_process_output_dir = output_fpath.with_suffix("") / "preprocessed_datasets"
        data_processor_config_dict["output_dirpath"] = str(data_process_output_dir)

    server_instance_configs = GlobalConfigDictParser().filter_for_server_instance_configs(global_config_dict)
    _validate_split_datasets_declared(e2e_rollout_collection_config.split, server_instance_configs)

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
        _validate_prepared_split_file_exists(
            input_jsonl_fpath, e2e_rollout_collection_config.split, data_process_output_dir
        )
        rollout_collection_config_dict["input_jsonl_fpath"] = str(input_jsonl_fpath)

    rollout_collection_config = RolloutCollectionConfig.model_validate(
        OmegaConf.to_container(rollout_collection_config_dict)
    )

    write_resolved_config(global_config_dict, output_fpath)

    rh = RunHelper()
    rh.start(None)

    rch = RolloutCollectionHelper()
    rch._termination_requested = lambda: _terminated

    # A benchmark can plug in a custom rollout-collection procedure via the
    # ``rollout_collection_driver`` config field (a ``module.path:function``).
    # The default path runs the built-in single-pass helper.
    driver_path = e2e_rollout_collection_config.rollout_collection_driver
    health_check_enabled = (
        not rollout_collection_config.disable_aggregation and not rollout_collection_config.disable_health_check
    )
    # This E2E entry point prints health before its server-shutdown phase.
    # The no-serve entry point calls the collection helper directly.
    if driver_path:
        rollout_collection_config.disable_health_check = True

    print(
        f"""Output artifacts:
1. Preprocessed datasets: {data_processor_config_dict["output_dirpath"]}
2. Dataset file used for rollout collection: {rollout_collection_config_dict["input_jsonl_fpath"]}
3. Rollout collection results file: {output_fpath}
{f"Rollout collection driver: {driver_path}" if driver_path else ""}
"""
    )
    if driver_path:
        _remove_health_reports(output_fpath)
    _interrupt_on_sigterm()
    collection_completed = False
    interrupted = False
    try:
        if driver_path:
            module_name, _, fn_name = driver_path.partition(":")
            if not module_name or not fn_name:
                raise ConfigError(f"rollout_collection_driver must be 'module.path:function' (got {driver_path!r}).")
            driver_fn = getattr(importlib.import_module(module_name), fn_name)
            resolved_config = OmegaConf.to_container(global_config_dict, resolve=True)
            asyncio.run(driver_fn(rollout_collection_config, resolved_config))
        else:
            _collect_and_rejudge(rch, rollout_collection_config, global_config_dict, output_fpath)
        collection_completed = True
    except KeyboardInterrupt:
        interrupted = True
        if rollout_collection_config.require_complete:
            mark_run_failed()
    except BaseException:
        # Includes an incomplete run that check_completion rejects: the tracking run must not read as finished.
        mark_run_failed()
        raise
    finally:
        # Before the server shutdown, so a run Slurm stops writes its summary within KillWait.
        # A failed or interrupted run reports health too, so its failed and missing rollouts are counted.
        try:
            # Health deferred with aggregation is not disabled: `gym eval aggregate` reports it.
            if driver_path and not rollout_collection_config.disable_aggregation:
                from nemo_gym.rollout_health import capture_dirs_for_health

                if health_check_enabled:
                    # The driver ran with the check disabled, so a built-in collector in it recorded "disabled".
                    _remove_health_reports(output_fpath)

                # The built-in collector reports under its run lock, before shutdown.
                _health_check_collected_rollouts(
                    output_fpath,
                    rollout_collection_config,
                    None if driver_path else rch._landed_keys,
                    model_call_capture_dirs=capture_dirs_for_health(global_config_dict),
                    disabled=not health_check_enabled,
                )
        finally:
            rh.shutdown()
    _exit_if_terminated()
    try:
        _check_collection_completion(
            output_fpath,
            require_complete=rollout_collection_config.require_complete,
            completed=collection_completed,
            interrupted=interrupted,
        )
    except BaseException:
        # A custom driver can return normally while saved coverage still rejects completion.
        mark_run_failed()
        raise

    _finalize_live_exporters(output_fpath)


def _prepare_run_health_metrics(output: Path) -> dict[str, int | float]:
    """Project the existing saved-evidence health calculation under the caller's run lock.

    This is not tracking acceptance. Unknown health exports a known=0 signal, never a healthy rate.
    Scientific aggregate entries and scores remain unchanged; only the reserved run metadata is replaced.
    """
    import subprocess
    import tempfile

    from nemo_gym.artifacts.exporters.base import _artifact_version
    from nemo_gym.artifacts.exporters.metrics import flatten_aggregate_metrics, health_source_artifacts
    from nemo_gym.artifacts.run_artifacts import RunArtifacts
    from nemo_gym.orchestration.recovery import (
        COLLECT_SCRIPT,
        CollectedEvidence,
        IdentityState,
        _observed_trial_operations,
        decide_from_collected,
        failure_facts,
        fault_report,
        response_health,
        select_managed_evidence,
        trial_faults,
        trial_population,
        trial_verdict,
    )

    artifacts = RunArtifacts(output)
    version = _artifact_version(artifacts)
    health_version = health_source_artifacts(artifacts, version)
    try:
        aggregate_bytes = artifacts.aggregate_metrics_json_fpath.read_bytes()
        aggregate = json.loads(aggregate_bytes)
    except (OSError, ValueError):
        # The exporter's existing artifact validation owns missing or malformed aggregate errors.
        return {}
    if (
        not isinstance(aggregate, list)
        or not aggregate
        or any(not isinstance(entry, dict) or not isinstance(entry.get("agent_ref"), dict) for entry in aggregate)
    ):
        return {}
    health = output.parent / QUALITY_SUMMARY_FILENAME
    health_bytes = health.read_bytes() if health.is_file() else None
    try:
        summary = json.loads(health_bytes) if health_bytes is not None else None
    except ValueError:
        summary = None
    metrics: dict[str, int | float] = {
        "health/response/known": 0,
        "health/trial/known": 0,
        "health/trial/passed": 0,
        "health/grading_recovery/known": 0,
        "health/upstream_recovery/known": 0,
        "health/trial/counts_known": 0,
    }
    # The ordinary collector reads these exact conventional paths. Other layouts retain explicit UNKNOWN.
    job = output.parent.parent
    if output.name == "rollouts.jsonl" and output.parent.name == "artifacts":
        raw = subprocess.run(
            [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
        )
        collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), job)
        decision = decide_from_collected(collected, max_infra_retries=None, max_walltime_s=None, live_job=None)
        quality = collected.quality if collected.quality and collected.quality.get("source_matches") is True else None
        response, _ = response_health(decision, quality)
        if response is not None:
            counts = quality["responses"]
            metrics.update(
                {
                    "health/response/known": 1,
                    "health/response/faulted": counts["faulted"],
                    "health/response/selected": counts["selected"],
                    "health/response/failure_rate": counts["faulted"] / counts["selected"],
                }
            )
        # The trial family is evaluation health: the evaluation-side count and denominator (expected trials less
        # accepted omissions, which are reported separately) the gate judges.
        trials = trial_population(decision)
        trial, _, evaluation = trial_verdict(decision, collected)
        if evaluation is None and trials > 0:
            # Not evaluated (the run is not complete): its known faults are a lower bound.
            evaluation = fault_report(trial_faults(decision, collected), trials) | {"exact": False}
        if evaluation is not None:
            exact = evaluation["exact"]
            failed = evaluation["evaluation"]["count"]
            metrics.update(
                {
                    "health/trial/counts_known": int(exact),
                    "health/trial/failed" if exact else "health/trial/failed_lower_bound": failed,
                    "health/trial/expected": trials,
                    "health/trial/omitted": decision.counts.get(IdentityState.OMITTED, 0),
                    "health/trial/failure_rate" if exact else "health/trial/failure_rate_lower_bound": failed / trials,
                }
            )
        if trial is not None:
            metrics.update({"health/trial/known": 1, "health/trial/passed": int(trial)})
        # Response and trial failures by cause: reported facts with no limit applied, for consumers to judge.
        facts = failure_facts(decision, collected)
        if facts is not None:
            for family in ("response_failures", "trial_failures"):
                metrics[f"health/{family}/rate"] = facts[family]["rate"]
                for cause, fact in facts[family]["by_cause"].items():
                    metrics[f"health/{family}/{cause}"] = fact["count"]
            # Health metrics are numbers only: the source is the one key of its name set to 1.
            coverage = facts["metric_coverage"]
            metrics["health/metric_coverage/trials"] = coverage["trials"]
            metrics["health/metric_coverage/expected"] = coverage["expected"]
            metrics[f"health/metric_coverage/source/{coverage['source']}"] = 1
        upstream = (quality or {}).get("observed_upstream_recovery")
        upstream_keys = (
            "observed_attempt_errors",
            "observed_failed_operations",
            "recovered_attempt_errors",
            "recovered_operations",
            "operations_without_terminal_observation",
        )
        if isinstance(upstream, dict) and all(
            type(upstream.get(key)) is int and upstream[key] >= 0 for key in upstream_keys
        ):
            metrics["health/upstream_recovery/known"] = 1
            metrics.update({f"health/upstream_recovery/{key}": upstream[key] for key in upstream_keys})
        recovery = {}
        if _observed_trial_operations(decision, collected, judge_recovery=recovery, grading_failures=[]):
            metrics["health/grading_recovery/known"] = 1
            metrics.update({f"health/grading_recovery/{key}": value for key, value in recovery.items()})
    if (
        _artifact_version(artifacts) != version
        or health_source_artifacts(artifacts, version) != health_version
        or (health.read_bytes() if health.is_file() else None) != health_bytes
    ):
        raise RuntimeError("Run artifacts changed while deriving health metrics.")
    payload = {
        "source_artifacts": health_version,
        "metrics": metrics,
    }
    for entry in aggregate:
        entry.pop("_ng_run_health_metrics", None)
    aggregate[0]["_ng_run_health_metrics"] = payload
    # Refuse collisions before changing artifacts or sending any health metric to a live exporter.
    flatten_aggregate_metrics(aggregate, source_artifacts=health_version)

    def atomic_json(path, value):
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    atomic_json(artifacts.aggregate_metrics_json_fpath, aggregate)
    if (
        isinstance(summary, dict)
        and isinstance(summary.get("run"), dict)
        and isinstance(summary["run"].get("stats"), dict)
    ):
        summary["run"]["stats"]["acceptance_health"] = payload
        atomic_json(health, summary)
    return metrics


def _finalize_live_exporters(output: Path) -> None:
    """Verify final persisted coverage and health before certifying an active live exporter."""
    from nemo_gym.artifacts.exporters import get_exporters, run_failed
    from nemo_gym.artifacts.exporters.base import _artifact_version
    from nemo_gym.artifacts.run_artifacts import RunArtifacts
    from nemo_gym.rollout_health import _health_artifact_digests, _report_sources
    from nemo_gym.rollout_recovery import run_lock
    from nemo_gym.rollout_store import RolloutStore

    exporters = get_exporters()
    if not exporters or run_failed():
        return
    try:
        with run_lock(output):
            artifacts = RunArtifacts(output)
            version = _artifact_version(artifacts)
            store = RolloutStore.read(output, import_legacy=False)
            if store is None or store.coverage().get("complete") is not True:
                return
            health = output.parent / QUALITY_SUMMARY_FILENAME
            health_bytes = health.read_bytes()
            summary = json.loads(health_bytes)
            if summary["run"].get("rollouts") != _report_sources([output], output.parent):
                return
            if summary["run"]["provenance"].get("rollout_artifact_digests") != _health_artifact_digests([output]):
                return
            if any(version[key] is None for key in version):
                return
            if any(artifacts.export_done_fpath(exporter.name).exists() for exporter in exporters):
                return
            health_metrics = _prepare_run_health_metrics(output)
            version = _artifact_version(artifacts)
            health_bytes = health.read_bytes()
            for exporter in exporters:
                exporter.export_metrics(health_metrics)
                if health.read_bytes() != health_bytes or _artifact_version(artifacts) != version:
                    raise RuntimeError("Final run artifacts changed before live finalization.")
                try:
                    exporter.finalize_live(artifacts, version)
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "Live tracking completion remains unverified: %s", exc, exc_info=True
                    )
                _export_without_receipt(exporter, artifacts)
    except Exception as exc:
        # Tracking failure never changes scores or requests another model attempt.
        logging.getLogger(__name__).warning("Live tracking completion remains unverified: %s", exc, exc_info=True)


def _export_without_receipt(exporter, artifacts) -> None:
    """Export the completed run after the fact when live finalization left it without a receipt.

    A live run reopened by a later segment never certifies (`finalize_live` cannot vouch for rows an earlier segment
    sent), nor does a live delivery it could not verify. Such a run gets Gym's post-run export, as `gym eval export`
    sends it, from a fresh exporter of the same backend and config; the caller holds the run lock and has prepared the
    health metrics. An exporter that never certifies a live run keeps no receipt, so it is not exported again, and an
    existing receipt is never replaced. A failed export is logged; it never changes scores or the exit code.
    """
    from nemo_gym.artifacts.exporters.base import BaseExporter, ExportState

    if (
        type(exporter).finalize_live is BaseExporter.finalize_live
        or artifacts.export_done_fpath(exporter.name).exists()
    ):
        return
    result = type(exporter)(exporter.global_config_dict).export(artifacts)
    if result.state is ExportState.FAILED:
        logging.getLogger(__name__).warning(
            "Post-run %s export after live finalization failed; the run has no export receipt: %s",
            exporter.name,
            result.error,
        )
    elif result.state is ExportState.SUCCEEDED and result.detail.get("live_run_id") and result.detail.get("run_id"):
        # The evaluation now has two backend runs: name the run of record on both, after the receipt is written.
        exporter.link_superseding_export(result.detail["live_run_id"], result.detail["run_id"])


_terminated = False


def _interrupt_on_sigterm() -> None:
    """Make Slurm's SIGTERM (scancel, time limit, preemption) unwind like Ctrl-C, so ``finally`` blocks run.

    Python's default SIGTERM action exits without unwinding, which would skip the quality summary.
    """

    def _on_sigterm(signum: int, frame: Any) -> None:
        global _terminated
        _terminated = True
        # The run exits 143 (`_exit_if_terminated`) whatever `require_complete` says; its tracking run closes FAILED.
        mark_run_failed()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_sigterm)


def _exit_if_terminated() -> None:
    """A run stopped by SIGTERM exits as a SIGTERM kill would, so it never looks successful."""
    if _terminated:
        sys.exit(128 + signal.SIGTERM)


def _remove_health_reports(output_fpath: Path) -> None:
    from nemo_gym.rollout_health import _remove_health_reports as remove_reports

    remove_reports(output_fpath)


def _health_check_collected_rollouts(
    output_fpath: Path,
    config: Any,
    landed_keys: set | None = None,
    *,
    coverage: dict | None = None,
    model_call_capture_dirs: Sequence[Path] = (),
    disabled: bool = False,
) -> None:
    """Report collected health using the CLI's shutdown state."""
    from nemo_gym.rollout_health import _health_check_collected_rollouts as report

    report(
        output_fpath,
        config,
        landed_keys,
        coverage=coverage,
        model_call_capture_dirs=model_call_capture_dirs,
        terminated=_terminated,
        disabled=disabled,
    )


def _check_collection_completion(
    output: Path, *, require_complete: bool, completed: bool, interrupted: bool = False
) -> None:
    """Apply the same completion opt-in after successful or interrupted collection."""
    if not require_complete:
        return
    if interrupted:
        raise RuntimeError("EVAL FAILED: rollout collection interrupted; partial artifacts retained.")
    if completed:
        _check_saved_completion(output)


def _check_saved_completion(output: Path) -> None:
    """Distinguish a valid partial run from successful CLI completion."""
    from nemo_gym.rollout_records import coverage_path_for
    from nemo_gym.rollout_recovery import IncompleteEvaluationError, has_only_intentional_gaps

    coverage = coverage_path_for(output.resolve())
    if coverage.exists():
        report = json.loads(coverage.read_text())
        if report.get("complete") is False:
            if has_only_intentional_gaps(
                expected=report["expected"],
                completed=report["successful"],
                intentionally_omitted=report.get("intentionally_omitted", 0),
            ):
                return
            retryable = report.get("retryable", 1) > 0
            guidance = (
                "Resume unfinished work after resolving failures."
                if retryable
                else "No tasks are eligible for retry: inspect terminal failures, omissions and exhausted budgets."
            )
            raise IncompleteEvaluationError(
                f"{report['successful']}/{report['expected']} samples completed. "
                f"Partial artifacts retained at {output}. {guidance}",
                retryable=retryable,
                # Older coverage reports predate per-invocation progress.
                made_progress=report.get("progress_outcomes_recorded", report.get("outcomes_recorded", 1)) > 0,
            )


def _collect_and_rejudge(
    rch: Any, config: Any, global_config_dict: DictConfig | Mapping[str, Any], output: Path
) -> None:
    """Run the built-in collection, then the saved-answer re-judge pass, before a strict run is rejected.

    With ``require_complete``, ``run_from_config`` raises ``IncompleteEvaluationError`` when completed is below
    expected, and a masked judge failure is not completed: the pass runs first, and the rejection stands unless the
    saved run is complete afterwards, by the rule ``check_completion`` applies. Whether a rejected run is retried is
    then read from the settled run (``_settled_rejection``). A non-strict run and a terminated run keep the error as
    raised. A custom ``rollout_collection_driver`` runs neither this pass nor this check.

    A resumed run grades its saved answers again first, while its servers are up, and only then runs its policy
    work: a recovery that resumes a run with both re-verifies the saved answers before the missing and retryable
    rollouts run.
    """
    from nemo_gym.rollout_recovery import IncompleteEvaluationError, manifest_path_for

    if getattr(config, "resume_from_cache", False) and manifest_path_for(output).exists():
        _rejudge_saved_judge_failures(global_config_dict, output, config, aggregate=False)
    try:
        asyncio.run(rch.run_from_config(config))
    except IncompleteEvaluationError as rejected:
        if not config.require_complete or _terminated:
            raise
        _rejudge_saved_judge_failures(global_config_dict, output, config)
        if not _saved_run_complete(output):
            raise _settled_rejection(output, rejected) from None
        return
    _rejudge_saved_judge_failures(global_config_dict, output, config)


def _settled_rejection(output: Path, rejected: BaseException) -> BaseException:
    """The rejection of a strict run as its settled state decides it, after the saved-answer pass.

    Whether unfinished work remains retryable is read from the Store again, so a pass that settled rows decides the
    retry, not the state before it. The collection's own progress still counts. Unreadable evidence keeps the
    original rejection.
    """
    from nemo_gym.rollout_recovery import IncompleteEvaluationError
    from nemo_gym.rollout_store import RolloutStore

    try:
        store = RolloutStore.read(output, import_legacy=False)
        coverage = store.coverage() if store is not None else None
    except Exception:
        return rejected
    if coverage is None:
        return rejected
    retryable = coverage["retryable"] > 0
    guidance = (
        "Resume unfinished work after resolving failures."
        if retryable
        else "No tasks are eligible for retry: inspect terminal failures, omissions and exhausted budgets."
    )
    return IncompleteEvaluationError(
        f"{coverage['successful'] - coverage.get('masked_retryable', 0) + coverage['final_zeros']}/{coverage['expected']} "
        "samples completed after the "
        f"saved-answer re-verification. Partial artifacts retained at {output}. {guidance}",
        retryable=retryable,
        # exit code 1 is a retryable rejection that made no progress; any other rejection keeps progress.
        made_progress=getattr(rejected, "exit_code", 75) != 1,
        failure_kind=getattr(rejected, "failure_kind", None),
    )


def _saved_run_complete(output: Path) -> bool:
    """Whether the saved Store selection now meets ``check_completion``: unreadable evidence is not complete."""
    from nemo_gym.rollout_recovery import has_only_intentional_gaps
    from nemo_gym.rollout_store import RolloutStore

    try:
        store = RolloutStore.read(output, import_legacy=False)
        if store is None:
            return False
        coverage = store.coverage()
    except Exception:  # the original rejection stands
        return False
    completed = coverage["successful"] - coverage.get("masked_retryable", 0) + coverage["final_zeros"]
    return completed == coverage["expected"] or has_only_intentional_gaps(
        expected=coverage["expected"], completed=completed, intentionally_omitted=coverage["intentionally_omitted"]
    )


def _rejudge_saved_judge_failures(
    global_config_dict: DictConfig | Mapping[str, Any], output: Path, collection_config: Any, *, aggregate: bool = True
) -> None:
    """Re-judge the run's saved judge failures while its servers are up, then recompute its metrics in place.

    Collection calls it at the end of a run and of each resumed run (see ``rejudge_saved_judge_failures``), and at the
    start of a resumed run, with ``aggregate`` off since its collection aggregates afterwards. The saved results to
    grade again (masked by a judge or verify-transport failure, or never graded) are re-verified in the same pass. A
    pass that cannot run, or fails, leaves the failures masked as before and never fails the collected run.
    """
    from nemo_gym.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper
    from nemo_gym.rollout_reverification import rejudge_saved_judge_failures

    # The CLI passes the resolved OmegaConf config; callers that build the config themselves pass a plain mapping.
    if isinstance(global_config_dict, DictConfig):
        run_config = OmegaConf.to_container(global_config_dict, resolve=True)
    else:
        run_config = dict(global_config_dict)
    try:
        scored = asyncio.run(rejudge_saved_judge_failures(run_config, output))
    except Exception as error:  # an optional pass after the run is saved; recovery still sees the failures
        print(f"Saved judge failures were not re-judged: {type(error).__name__}: {error}", flush=True)
        return
    if not scored:
        return
    print(f"Re-judged saved judge failures: {scored} rollout(s) settled from their saved answers", flush=True)
    if collection_config.disable_aggregation or not aggregate:
        return
    try:
        aggregation = RolloutAggregationConfig.model_validate(
            run_config | {"input_glob": str(output), "output_jsonl_fpath": str(output), "merge_shards": False}
        )
        asyncio.run(RolloutAggregationHelper().run_from_config(aggregation))
    except Exception as error:  # the settled rows are saved; only the metrics file is behind them
        print(
            f"Metrics were not recomputed after the re-judgment ({type(error).__name__}: {error}); "
            f"run `gym eval aggregate +input_glob={output} +output_jsonl_fpath={output} +merge_shards=false`.",
            flush=True,
        )


class _SettledHealthSettings(RunHealthConfig):
    """The health-check settings of a run config, read for the quality summary of a run a judge-only job settled."""

    disable_health_check: bool = False
    health_check_workers: int | None = None
    health_check_ignored_checks: List[str] = Field(default_factory=list)


def _refresh_settled_source(
    run_config: DictConfig | Mapping[str, Any], output: Path, *, workdir: Path
) -> dict[str, bool]:
    """Recompute the settled run's aggregate metrics in place and rewrite its quality summary from the settled rows,
    as the run's own completion does after its saved-answer pass.

    A settled run's rollouts file holds superseded outcomes (a masked row and the grade that settled it). A health
    check that selects the Store's attempts itself reads it as it is; an earlier one reads it only through a
    selected-result projection: the Store's selected rows are written under ``workdir`` for the check, which is bound
    to the run's own artifacts, and removed after it. A step that fails is
    printed and reported in the receipt, never raised: the settlement stands, and the operator reruns the step the
    message names.
    """
    from nemo_gym.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper
    from nemo_gym.rollout_health import (
        capture_dirs_for_health,
        format_health_report,
        run_health_checks,
        write_disabled_summary,
    )
    from nemo_gym.rollout_store import RolloutStore, raw_outcomes_are_selected

    if isinstance(run_config, DictConfig):
        run_config = OmegaConf.to_container(run_config, resolve=True)
    else:
        run_config = dict(run_config)
    refreshed = {"metrics_recomputed": False, "health_summary_recomputed": False}
    try:
        aggregation = RolloutAggregationConfig.model_validate(
            run_config | {"input_glob": str(output), "output_jsonl_fpath": str(output), "merge_shards": False}
        )
        asyncio.run(RolloutAggregationHelper().run_from_config(aggregation))
        refreshed["metrics_recomputed"] = True
    except Exception as error:
        print(
            f"Metrics were not recomputed after the re-judgment ({type(error).__name__}: {error}); "
            f"run `gym eval aggregate +input_glob={output} +output_jsonl_fpath={output} +merge_shards=false`.",
            flush=True,
        )
    projection = None
    try:
        settings = _SettledHealthSettings.model_validate(run_config)
        if settings.disable_health_check:
            write_disabled_summary([output], output.parent)
            return refreshed
        store = RolloutStore.read(output, import_legacy=False)
        # A health check that selects the Store's attempts itself (NVIDIA-NeMo/Gym#3881) takes the history as it is.
        # An earlier one refuses a history with superseded outcomes and reads it only through a selected projection,
        # which it names as a parameter; drop this branch once every supported Gym selects the attempts itself.
        projected = {}
        if (
            "selected_projection" in inspect.signature(run_health_checks).parameters
            and store is not None
            and not raw_outcomes_are_selected(output, artifacts=(output,))
        ):
            workdir.mkdir(parents=True, exist_ok=True)
            projection = workdir / "selected_rollouts.jsonl"
            with projection.open("w", encoding="utf-8") as handle:
                for row in store.selected("success"):
                    handle.write(json.dumps(row) + "\n")
            projected["selected_projection"] = projection
        result = run_health_checks(
            output,
            **projected,
            output_dir=output.parent,
            workers=settings.health_check_workers,
            ignored_checks=settings.health_check_ignored_checks,
            limits=settings.run_health_limits,
            run_coverage=[store.coverage()] if store is not None else None,
            model_call_capture_dirs=capture_dirs_for_health(run_config),
        )
        print(format_health_report(result))
        refreshed["health_summary_recomputed"] = True
    except Exception as error:
        print(
            f"The quality summary was not recomputed after the re-judgment ({type(error).__name__}: {error}); "
            f"run `gym eval health-check {output.parent}`.",
            flush=True,
        )
    finally:
        if projection is not None:
            projection.unlink(missing_ok=True)
    return refreshed


def _settled_source_state(output: Path) -> tuple[dict[str, Any] | None, int | None]:
    """The settled run's coverage and how many saved answers a later pass would still grade again, for the receipt;
    None for what cannot be read."""
    from nemo_gym.rollout_reverification import due_judge_failures
    from nemo_gym.rollout_store import RolloutStore

    try:
        store = RolloutStore.read(output, import_legacy=False)
        coverage = store.coverage() if store is not None else None
    except Exception:
        coverage = None
    try:
        due = len(due_judge_failures(output))
    except Exception:
        due = None
    return coverage, due


def _reverify_with_receipt(global_config_dict: DictConfig | Mapping[str, Any], config: Any) -> None:
    """Run a re-verification; for the judge-only recovery job (``settle_receipt_fpath``), settle, refresh and write
    the receipt.

    The job grades the selected saved results again through the run's own settle path (``settle_into_source``: a
    grade replaces a saved result only when it binds the same saved answer, run and attempt, and the policy is never
    called), then recomputes the source run's aggregate metrics and quality summary from the settled rows, and ends
    with a receipt: what was selected, judged, settled and left as it was, the run's coverage, and whether the refresh
    happened. A pass that fails (the source changed while the job was queued, a verifier that cannot be reached)
    leaves a failed receipt naming the error, with the saved answers as they were, and the error is raised as before.
    """
    from nemo_gym.rollout_reverification import RolloutReverificationHelper, write_settle_receipt

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    helper = RolloutReverificationHelper()
    receipt = Path(config.settle_receipt_fpath) if config.settle_receipt_fpath else None
    source = str(config.rollouts_jsonl_fpath) if config.rollouts_jsonl_fpath else None
    job_id = os.environ.get("SLURM_JOB_ID")
    try:
        asyncio.run(helper.run_from_config(config))
    except BaseException as error:
        if receipt is not None:
            write_settle_receipt(
                receipt,
                state="failed",
                source=source,
                job_id=job_id,
                settlement=helper.settlement,
                started_at=started,
                error=f"{type(error).__name__}: {error}",
            )
        raise
    if receipt is None:
        return
    output = Path(config.rollouts_jsonl_fpath)
    refreshed = _refresh_settled_source(global_config_dict, output, workdir=receipt.parent)
    coverage, due_after = _settled_source_state(output)
    write_settle_receipt(
        receipt,
        state="completed",
        source=source,
        job_id=job_id,
        settlement=helper.settlement,
        started_at=started,
        coverage=coverage,
        due_after=due_after,
        refreshed=refreshed,
    )
    print(f"Judge-only receipt: {receipt}", flush=True)


@exit_cleanly_on_config_error
def collect_rollouts():  # pragma: no cover
    from nemo_gym.artifacts.resolved_config import write_resolved_config
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
    from nemo_gym.telemetry.setup import (
        configure_telemetry_env,
        init_telemetry,
        shutdown_telemetry,
        telemetry_config_from_global_config,
    )

    global_config_dict = get_global_config_dict()
    config = RolloutCollectionConfig.model_validate(global_config_dict)
    write_resolved_config(global_config_dict, Path(config.output_jsonl_fpath))
    rch = RolloutCollectionHelper()
    rch._termination_requested = lambda: _terminated

    # A standalone driver starts no supervisor, so it sets up its own telemetry for the `gym.job` span and driver metrics.
    configure_telemetry_env(telemetry_config_from_global_config(global_config_dict))
    init_telemetry(server_name="driver", server_type="driver")
    _interrupt_on_sigterm()
    try:
        _collect_and_rejudge(rch, config, global_config_dict, Path(config.output_jsonl_fpath))
        _check_collection_completion(
            Path(config.output_jsonl_fpath), require_complete=config.require_complete, completed=True
        )
    except KeyboardInterrupt:
        mark_run_failed()
        if not _terminated:
            raise
    except BaseException:
        mark_run_failed()
        raise
    finally:
        shutdown_telemetry()
    _exit_if_terminated()
    _finalize_live_exporters(Path(config.output_jsonl_fpath))


@exit_cleanly_on_config_error
def aggregate_rollouts():  # pragma: no cover
    from nemo_gym.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper

    global_config = get_global_config_dict()
    config = RolloutAggregationConfig.model_validate(global_config)
    rah = RolloutAggregationHelper()

    asyncio.run(rah.run_from_config(config))


def health_check_rollouts(
    run_dir: str | Path,
    *,
    rollout_file: str | Path | None = None,
    workers: int | None = None,
    ignored_checks: Sequence[str] = (),
    limits: Any = None,
    json_output: bool = False,
    model_call_capture_dir: str | Path | None = None,
):
    """Run rollout quality verification for an existing run directory.

    By default the run's persisted limits judge it again. Each limit set explicitly on ``limits`` (a
    ``RunHealthLimits``) overrides only that persisted limit, and ``None`` removes it; a limit not set keeps its
    persisted value.
    """
    from nemo_gym.rollout_health import health_check_run_dir

    return health_check_run_dir(
        run_dir,
        rollout_file=rollout_file,
        workers=workers,
        ignored_checks=ignored_checks,
        limits=limits,
        json_output=json_output,
        model_call_capture_dir=model_call_capture_dir,
    )


@exit_cleanly_on_config_error
def export_run() -> None:  # pragma: no cover
    from nemo_gym.artifacts.exporter_registry import DEFAULT_RUN_EXPORTER
    from nemo_gym.artifacts.exporters import load_run_exporter_class
    from nemo_gym.artifacts.exporters.base import ExportState
    from nemo_gym.artifacts.run_artifacts import RunArtifacts
    from nemo_gym.config.config_types import RunExportConfig

    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(skip_exporters=True)
    )
    name = global_config_dict.get("exporter") or DEFAULT_RUN_EXPORTER
    exporter = load_run_exporter_class(name).from_global_config(global_config_dict)
    from nemo_gym.rollout_recovery import run_lock

    artifacts = RunArtifacts(RunExportConfig.model_validate(global_config_dict).rollouts_jsonl_fpath)
    from nemo_gym.artifacts.exporters.base import BaseExporter

    if isinstance(exporter, BaseExporter) and artifacts.export_done_fpath(name).exists():
        # An exported run's repeat only verifies its receipt and writes nothing: it must not take the run lock, whose
        # holder record it would overwrite.
        result = exporter.export(artifacts)
    elif isinstance(exporter, BaseExporter):
        with run_lock(artifacts.rollouts_jsonl_fpath):
            # A racing exporter may have finished while this one waited for the lock: its receipt is then only
            # verified, and the health metrics it prepared are not rewritten.
            if not artifacts.export_done_fpath(name).exists():
                _prepare_run_health_metrics(artifacts.rollouts_jsonl_fpath)
            result = exporter.export(artifacts)
    else:
        result = exporter.export(artifacts)

    print(f"Export {name}: {result.state.value}")
    for key, value in result.detail.items():
        print(f"  {key}: {value}")
    if result.state is ExportState.FAILED:
        print(f"Error: {result.error}", file=sys.stderr)
        raise SystemExit(result.exit_code)


@exit_cleanly_on_config_error
def reverify_rollouts():  # pragma: no cover
    from nemo_gym.rollout_reverification import RolloutReverificationConfig

    rh = RunHelper()
    rh.start(None)
    try:
        global_config_dict = get_global_config_dict()
        config = RolloutReverificationConfig.model_validate(global_config_dict)
        # The judge-only recovery job's pass settles, refreshes the run and ends with a receipt; any other runs as before.
        _reverify_with_receipt(global_config_dict, config)
    finally:
        rh.shutdown()  # stop the servers this command started


@exit_cleanly_on_config_error
def reward_profile():  # pragma: no cover
    from nemo_gym.metrics.reward_profile import (
        RewardProfileConfig,
        RewardProfiler,
        coverage_by_agent,
        select_measured,
    )
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

    # Completeness is judged on what was actually collected, before any masking filter:
    # dropping masked pairs first would hide a genuinely missing rollout behind a set that
    # happens to align, and silently profile a partial collection as a whole one.
    rp.align_rows_and_results(rows, results, allow_partial_rollouts=config.allow_partial_rollouts)

    # Quality metrics then come from the measured subset only, the same selection the
    # aggregation path makes, so profiling the saved rollouts of a run agrees with the
    # metrics that run published. Completion accounting below still sees every row: a
    # masked rollout did run, and is not a gap in the collection.
    measured_rows, measured_results, masked, _ = select_measured(rows, results)
    group_level_metrics, agent_level_metrics, repeat_level_metrics = rp.profile_from_data(
        measured_rows, measured_results, allow_partial_rollouts=config.allow_partial_rollouts
    )

    # Each agent carries its own coverage, never the run's. An agent whose every result was
    # masked has no quality metrics at all, so it is kept as a coverage-only entry rather
    # than disappearing from the artifact.
    agent_coverage = coverage_by_agent(rows, results)
    for entry in agent_level_metrics:
        name = (entry.get("agent_ref") or {}).get("name")
        if name in agent_coverage:
            entry.update(agent_coverage.pop(name))
    for name, entry_coverage in agent_coverage.items():
        agent_level_metrics.append({"agent_ref": {"name": name}, **entry_coverage})

    completion_summary = rp.profile_completion_summary(rows, results)
    reward_profiling_fpath, agent_level_metrics_fpath, repeat_level_metrics_fpath = rp.write_to_disk(
        group_level_metrics, agent_level_metrics, repeat_level_metrics, Path(config.rollouts_jsonl_fpath)
    )

    print(f"""Profiling outputs:
Reward profile completion: {completion_summary["completed_rollout_rows"]}/{completion_summary["expected_rollout_rows"]} rollout rows ({completion_summary["reward_profile_completion_pct"]:.2f}%)
Input rows: {completion_summary["total_input_rows"]} total; {completion_summary["complete_input_rows"]} complete; {completion_summary["partial_input_rows"]} partial; {completion_summary["missing_input_rows"]} without rollouts dropped from output.
Masked from quality metrics: {len(masked)} rollout rows (kept in completion accounting above).
Reward profiling outputs: {reward_profiling_fpath}
Agent-level metrics: {agent_level_metrics_fpath}
Repeat-level metrics: {repeat_level_metrics_fpath}""")


@exit_cleanly_on_config_error
def compare() -> None:  # pragma: no cover
    from nemo_gym.comparison.report import render_key_metrics_tables, summary_lines
    from nemo_gym.comparison.runner import invoked_command, run_comparison
    from nemo_gym.comparison.schema import ComparisonConfig

    config = ComparisonConfig.model_validate(get_global_config_dict())

    result, written = run_comparison(config, invoked_command())

    for table in render_key_metrics_tables(result):
        print_rich_table(table)
    print("\n".join(summary_lines(result, written)))


def eval_status(target: str, *, json_output: bool = False) -> None:
    """Print saved progress for a run directory or a locally indexed Gym job ID.

    Submitted job artifacts must be visible on the current filesystem. This command
    reads saved progress and does not contact a scheduler or a remote host.
    """
    from nemo_gym.batch_status import BATCH_STATUS_FNAME
    from nemo_gym.orchestration.jobs import MANIFEST_NAME, SubmissionRecord, local_index_dir

    def status_files(directory: Path) -> list[Path]:
        return sorted({*directory.glob("*_batch_status.json"), *directory.glob(BATCH_STATUS_FNAME)})

    def export_facts(file: Path) -> dict[str, dict[str, Any]]:
        """Each exporter's receipt (``<stem>_<exporter>_export_done``) and live run id (``<stem>_<exporter>_run_id``)
        beside the output this status file belongs to. The legacy ``batch_status.json`` names no stem, so every
        marker in its directory counts."""
        stem = file.name.removesuffix("_batch_status.json") if file.name.endswith("_batch_status.json") else None
        facts: dict[str, dict[str, Any]] = {}
        names = sorted(entry.name for entry in file.parent.iterdir() if entry.is_file())
        for suffix, key in (("_export_done", "receipt"), ("_run_id", "run_id")):
            # Anchored on the stem with a single-token exporter name, so a shard label that is a prefix of another
            # shard's label never takes that shard's markers.
            exporter_pattern = rf"([A-Za-z0-9-]+){re.escape(suffix)}$"
            pattern = re.compile(
                rf"^{re.escape(stem)}_{exporter_pattern}" if stem is not None else rf"^(?:.*_)?{exporter_pattern}"
            )
            for name in names:
                match = pattern.match(name)
                if match is None:
                    continue
                fact = facts.setdefault(match.group(1), {"receipt": False})
                fact[key] = True if key == "receipt" else (file.parent / name).read_text().strip()
        return facts

    path = Path(target)
    if not path.is_dir() and (path.exists() or "/" in target or "\\" in target):
        raise NotADirectoryError(f"Run directory expected: {target}")
    record_path = path / MANIFEST_NAME if path.is_dir() else local_index_dir() / f"{target}.json"
    files = status_files(path) if path.is_dir() else []
    exports: dict[str, dict[str, dict[str, Any]]] = {}
    if files:
        statuses = {str(file): json.loads(file.read_text()) for file in files}
        exports = {str(file): export_facts(file) for file in files}
    elif record_path.is_file():
        record = SubmissionRecord.load(json.loads(record_path.read_text()))
        statuses = {}
        for benchmark in record.benchmarks:
            directory = Path(benchmark.job_dir) / "artifacts"
            files = status_files(directory)
            if not files:
                statuses[benchmark.benchmark] = {"status": "unavailable", "path": str(directory)}
            for file in files:
                name = benchmark.benchmark if len(files) == 1 else f"{benchmark.benchmark}/{file.name}"
                statuses[name] = json.loads(file.read_text())
                exports[name] = export_facts(file)
    else:
        raise FileNotFoundError(f"No {BATCH_STATUS_FNAME} or Gym job record found for {target}")
    if json_output:
        print(
            json.dumps(
                {
                    name: ({**status, "export": exports.get(name, {})} if "members" in status else status)
                    for name, status in statuses.items()
                }
            )
        )
        return
    for name, status in statuses.items():
        print(f"{name}:")
        if "members" not in status:
            print(f"  Status unavailable: {status['path']}")
            continue
        print(f"  updated_at: {status.get('updated_at', 'unknown')}; tracking: {status.get('tracking', 'enabled')}")
        facts = exports.get(name, {})
        # A completed export leaves a receipt; a live exporter leaves the run it logs to. Neither is proof of the
        # other, so both are reported as read.
        print(
            "  export receipt: "
            + (
                ", ".join(
                    f"{exporter} {'present' if fact.get('receipt') else 'missing'}"
                    + (f" (run id {fact['run_id']})" if fact.get("run_id") else "")
                    for exporter, fact in facts.items()
                )
                if facts
                else "missing"
            )
        )
        if status.get("tracking") == "disabled":
            print(f"  reason: {status.get('reason', 'unknown')}")
            continue
        for agent, member in status["members"].items():
            print(
                f"  {agent}: {member['completed_rollout_count']} completed, "
                f"{member['remaining_rollout_count']} remaining; "
                f"expected: {member.get('expected_rollout_count', 'unknown')}; "
                f"aggregation: {member.get('aggregation_status', 'unknown')}; "
                f"failures by class: {json.dumps(member['failures_by_class'], sort_keys=True)}; "
                f"by kind: {json.dumps(member.get('failures_by_kind', {}), sort_keys=True)}; "
                f"model-final by kind: {json.dumps(member.get('model_final_by_kind', {}), sort_keys=True)}"
            )
