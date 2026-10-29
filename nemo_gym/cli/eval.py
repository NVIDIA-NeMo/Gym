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
import argparse
import asyncio
import importlib
import json
import logging
import signal
import site
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import Field, ValidationError
from rich.table import Table
from tqdm.auto import tqdm

from nemo_gym import PARENT_DIR
from nemo_gym.artifacts.run_files import QUALITY_SUMMARY_FILENAME, ROLLOUT_VERDICTS_FILENAME
from nemo_gym.benchmarks import (
    BenchmarkConfig,
    discover_benchmarks,
)
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
from nemo_gym.data.tasks.harbor.build_options import (
    _block_from_overrides,
    _merged_cli_config,
    build_options_from_overrides,
)
from nemo_gym.data.tasks.harbor.build_plan import (
    BuildError,
)
from nemo_gym.data.tasks.harbor.compose_overlay import HarborComposeOverlay
from nemo_gym.data.tasks.harbor.errors import HarborTaskError
from nemo_gym.data.tasks.harbor.hub import (
    HubError,
    datasets_dir,
)
from nemo_gym.data.tasks.harbor.manifest import (
    ManifestError,
)
from nemo_gym.data.tasks.harbor.materialize import (
    ENVIRONMENT_SERVER_IMPL,
    ENVIRONMENT_SERVER_INSTANCE,
    write_rows,
)
from nemo_gym.data.tasks.harbor.prepare import (
    DEFAULT_OUTPUT_ROOT,
    ORACLE_AGENT,
    AgentSelection,
    PreparedTaskset,
    _has_override,
    _listed,
    build_run,
    prepare_target,
    resolve_agent,
    resolve_sandbox_config,
)
from nemo_gym.data.tasks.harbor.registry import (
    ImageConfigError,
    credentials_for_image_prefixes,
)
from nemo_gym.data.tasks.harbor.run_summary import NO_REFERENCE_SOLUTION, output_signatures, write_run_summary
from nemo_gym.data.tasks.harbor.validation_report import (
    _task_digests,
    _validation_key,
    _validation_rows,
    summarize_validation,
)
from nemo_gym.discovery import read_config_metadata
from nemo_gym.eval_plan import (
    PlanBenchmark,
    PlanGym,
    build_eval_plan,
    eval_plan_json_schema,
    write_eval_plan,
)
from nemo_gym.orchestration.jobs import installed_gym_commit
from nemo_gym.orchestration.server_supervisor import RunHelper
from nemo_gym.package_info import __version__
from nemo_gym.sandbox.compose_config import deployment_platform
from nemo_gym.sandbox.config import SANDBOX_BLOCK_VOLUME_SOURCES_KEY, validate_volume_sources
from nemo_gym.utils.path_utils import materialized_path_for


if TYPE_CHECKING:
    from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig


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
    from nemo_gym.data.dataset_preparation import TrainDataProcessor
    from nemo_gym.rollouts.rollout_collection import (
        E2ERolloutCollectionConfig,
        RolloutCollectionConfig,
        RolloutCollectionHelper,
    )

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

    # A benchmark can plug in a custom rollout-collection procedure via the
    # ``rollout_collection_driver`` config field (a ``module.path:function``).
    # The default path runs the built-in single-pass helper.
    driver_path = e2e_rollout_collection_config.rollout_collection_driver
    health_check_enabled = (
        not rollout_collection_config.disable_aggregation and not rollout_collection_config.disable_health_check
    )
    # This E2E entry point prints health only after its server-shutdown phase.
    # The no-serve entry point calls the collection helper directly.
    rollout_collection_config.disable_health_check = True

    print(
        f"""Output artifacts:
1. Preprocessed datasets: {data_processor_config_dict["output_dirpath"]}
2. Dataset file used for rollout collection: {rollout_collection_config_dict["input_jsonl_fpath"]}
3. Rollout collection results file: {output_fpath}
{f"Rollout collection driver: {driver_path}" if driver_path else ""}
"""
    )
    _remove_health_reports(output_fpath)
    _interrupt_on_sigterm()
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
        if rollout_collection_config.require_complete:
            raise RuntimeError("EVAL FAILED: rollout collection interrupted; partial artifacts retained.") from None
        pass
    finally:
        # Before the server shutdown, so a run Slurm stops writes its summary within KillWait.
        # A failed or interrupted run reports health too, so its failed and missing rollouts are counted.
        try:
            if health_check_enabled:
                from nemo_gym.rollouts.rollout_health import capture_dirs_for_health

                # A custom driver keeps no landed keys, so its stopped run gets the full check.
                _health_check_collected_rollouts(
                    output_fpath,
                    rollout_collection_config,
                    None if driver_path else rch._landed_keys,
                    model_call_capture_dirs=capture_dirs_for_health(global_config_dict),
                )
        finally:
            rh.shutdown()
    _exit_if_terminated()


_terminated = False


def _interrupt_on_sigterm() -> None:
    """Make Slurm's SIGTERM (scancel, time limit, preemption) unwind like Ctrl-C, so ``finally`` blocks run.

    Python's default SIGTERM action exits without unwinding, which would skip the quality summary.
    """

    def _on_sigterm(signum: int, frame: Any) -> None:
        global _terminated
        _terminated = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_sigterm)


def _exit_if_terminated() -> None:
    """A run stopped by SIGTERM exits as a SIGTERM kill would, so it never looks successful."""
    if _terminated:
        sys.exit(128 + signal.SIGTERM)


def _remove_health_reports(output_fpath: Path) -> None:
    # A fresh run or a resume reports its own health; a report left by an earlier job must not stand for it.
    for report in (QUALITY_SUMMARY_FILENAME, ROLLOUT_VERDICTS_FILENAME):
        (output_fpath.parent / report).unlink(missing_ok=True)


def _health_check_collected_rollouts(
    output_fpath: Path, config: Any, landed_keys: set | None = None, *, model_call_capture_dirs: Sequence[Path] = ()
) -> None:
    """Write the quality summary for whatever a run collected; a run that never started collection has none.

    After SIGTERM, Slurm kills the job once KillWait (120 s on some clusters) is over, and the full check can take
    minutes on a large run. So a stopped run with ``landed_keys`` gets the provisional summary: the run
    accounting only, without reading the rollouts.
    """
    if not output_fpath.exists():
        return
    from nemo_gym.rollouts.health.timeouts import export_run_health_timeouts
    from nemo_gym.rollouts.rollout_health import format_health_report, run_health_checks, write_provisional_summary

    if _terminated and landed_keys is not None:
        from nemo_gym.utils.path_utils import aggregation_sources_path_for

        if aggregation_sources_path_for(output_fpath).is_file():
            # A run drops the shard list with the first row it writes, so this run added none: the summary the
            # aggregation wrote from its shards still holds, and the run accounting alone would replace it wrongly.
            print("Shard-backed health was not recomputed after stopping; run `gym eval health-check` to check it.")
            return
        try:
            summary_path = write_provisional_summary(
                output_fpath,
                landed_keys,
                limits=config.run_health_limits,
                ignored_checks=config.health_check_ignored_checks,
            )
        except Exception:
            logger.exception("Writing the provisional quality summary failed; rollout artifacts are still available.")
        else:
            print(f"Provisional quality summary (run accounting only): {summary_path}")
        return
    try:
        from nemo_gym.rollouts.rollout_health import _aggregated_shards

        # An output `gym eval aggregate` merged shards into is checked from those shards, as the collector and a
        # recheck check it: its own rows have none of their failures or inventories.
        shards = _aggregated_shards(output_fpath.parent, rollout_name=output_fpath.name)
        health_result = run_health_checks(
            shards if shards is not None else output_fpath,
            output_dir=output_fpath.parent,
            workers=config.health_check_workers,
            ignored_checks=config.health_check_ignored_checks,
            limits=config.run_health_limits,
            model_call_capture_dirs=model_call_capture_dirs,
        )
    except Exception:
        logger.exception("Rollout health checks failed after collection; rollout artifacts are still available.")
    else:
        print(format_health_report(health_result))
        export_run_health_timeouts(health_result)


@exit_cleanly_on_config_error
def collect_rollouts():  # pragma: no cover
    from nemo_gym.artifacts.resolved_config import write_resolved_config
    from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
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
    health_check_enabled = not config.disable_aggregation and not config.disable_health_check
    config.disable_health_check = True  # Run once below, also when collection fails.

    # A standalone driver starts no supervisor, so it sets up its own telemetry for the `gym.job` span and driver metrics.
    configure_telemetry_env(telemetry_config_from_global_config(global_config_dict))
    init_telemetry(server_name="driver", server_type="driver")
    _remove_health_reports(Path(config.output_jsonl_fpath))
    _interrupt_on_sigterm()
    try:
        asyncio.run(rch.run_from_config(config))
    except KeyboardInterrupt:
        if not _terminated:
            raise
    finally:
        try:
            shutdown_telemetry()
        finally:
            if health_check_enabled:
                from nemo_gym.rollouts.rollout_health import capture_dirs_for_health

                _health_check_collected_rollouts(
                    Path(config.output_jsonl_fpath),
                    config,
                    rch._landed_keys,
                    model_call_capture_dirs=capture_dirs_for_health(global_config_dict),
                )
    _exit_if_terminated()


@exit_cleanly_on_config_error
def aggregate_rollouts():  # pragma: no cover
    from nemo_gym.rollouts.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper

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
    from nemo_gym.rollouts.rollout_health import health_check_run_dir

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
    result = exporter.export(RunArtifacts(RunExportConfig.model_validate(global_config_dict).rollouts_jsonl_fpath))

    print(f"Export {name}: {result.state.value}")
    for key, value in result.detail.items():
        print(f"  {key}: {value}")
    if result.state is ExportState.FAILED:
        print(f"Error: {result.error}", file=sys.stderr)
        raise SystemExit(result.exit_code)


@exit_cleanly_on_config_error
def reverify_rollouts():  # pragma: no cover
    from nemo_gym.rollout_reverification import RolloutReverificationConfig, RolloutReverificationHelper

    rh = RunHelper()
    rh.start(None)

    config = RolloutReverificationConfig.model_validate(get_global_config_dict())
    rrh = RolloutReverificationHelper()

    asyncio.run(rrh.run_from_config(config))


@exit_cleanly_on_config_error
def reward_profile():  # pragma: no cover
    from nemo_gym.metrics.reward_profile import (
        RewardProfileConfig,
        RewardProfiler,
        coverage_by_agent,
        select_measured,
    )
    from nemo_gym.rollouts.rollout_collection import loads_jsonl_line

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


# The top-level block the Harbor environment server reads its provider from by default (`sandbox_provider`).
SANDBOX_BLOCK_KEY = "sandbox"


# Where the server's settings sit in the configuration a command names: the shipped environment server instance
# every run config inherits from. `gpu_sandbox_provider` names the block GPU-declaring tasks and GPU sidecars run
# on; `sandbox_provider_options.platform` ({os, arch}) is the platform a service's `platform` must match.
HARBOR_SERVER_KEY = f"{ENVIRONMENT_SERVER_INSTANCE}.environment_servers.{ENVIRONMENT_SERVER_IMPL}"


GPU_SANDBOX_PROVIDER_KEY = f"{HARBOR_SERVER_KEY}.gpu_sandbox_provider"


SANDBOX_PROVIDER_OPTIONS_KEY = f"{HARBOR_SERVER_KEY}.sandbox_provider_options"


# The job-level Compose overlay (extra Compose files, MCP entries, prompt template), read at prepare so the
# overlay's images are recorded and its merge checked with the tasks', exactly as the server reads it at seed.
COMPOSE_OVERLAY_KEY = f"{HARBOR_SERVER_KEY}.compose_overlay"


def _block_provider_name(overrides: list[str], sandbox: str | None, block_key: str) -> str | None:
    """The provider name a top-level sandbox block carries, read from its keys alone (never its values)."""
    from omegaconf import OmegaConf

    from nemo_gym.sandbox.config import SANDBOX_BLOCK_RESERVED_KEYS

    files = [resolve_sandbox_config(sandbox)] if sandbox is not None else []
    block = OmegaConf.select(_merged_cli_config(overrides, files), block_key, throw_on_resolution_failure=False)
    if block is None or not hasattr(block, "keys"):
        return None
    providers = [str(key) for key in block.keys() if key not in SANDBOX_BLOCK_RESERVED_KEYS]
    return providers[0] if len(providers) == 1 else None


def sandbox_provider_from_overrides(overrides: list[str], sandbox: str | None) -> str | None:
    """The provider name (``opensandbox``, ``docker``, ...) of the ``sandbox:`` block the command names.

    The block comes from the ``--sandbox`` provider file (when given) and the configuration the command line
    names, merged as :func:`_merged_cli_config` does; only its keys are read, never its values, so provider
    credentials are not needed for this (the registry credentials prepare does read are the block's
    ``images.auth``, :func:`registry_credentials_from_overrides`). Prepare checks each Compose task's ``tmpfs``, ``ulimits``,
    ``sysctls`` and ``security_opt`` against this provider's capabilities. ``None`` when no block names a
    provider: the check then runs at seed, against the live provider.
    """
    return _block_provider_name(overrides, sandbox, SANDBOX_BLOCK_KEY)


# The sandbox provider block's image credentials (OpenSandbox's `images.auth`: image prefix to username and
# password), the one place a deployment names registry credentials; prepare reads private images with the same.
SANDBOX_IMAGES_AUTH_KEY = "images.auth"


def registry_credentials_from_overrides(overrides: list[str], sandbox: str | None) -> dict[str, tuple[str, str]]:
    """The registry credentials prepare records private images with: the ``sandbox:`` block's ``images.auth``
    (what the provider pulls those images with at seed), turned into the registry client's prefix-keyed form
    (:func:`~nemo_gym.data.tasks.harbor.registry.credentials_for_image_prefixes`), so each image is read with the
    credentials of its longest matching prefix, as the provider pulls it.

    Read from the ``--sandbox`` provider file and the configuration the command names, resolved against them
    (so ``${gitlab_registry_password}`` comes from ``env.yaml`` as it does at seed), like
    :func:`volume_sources_from_overrides`. Empty when no block names a provider or the block has no
    ``images.auth``: prepare then reads registries anonymously, as before. A malformed block is a
    :class:`ValueError` naming the key.
    """
    provider = sandbox_provider_from_overrides(overrides, sandbox)
    if provider is None:
        return {}
    files = [resolve_sandbox_config(sandbox)] if sandbox is not None else []
    key = f"{SANDBOX_BLOCK_KEY}.{provider}.{SANDBOX_IMAGES_AUTH_KEY}"
    return credentials_for_image_prefixes(_block_from_overrides(overrides, key, config_files=files), key)


def gpu_sandbox_provider_from_overrides(overrides: list[str], sandbox: str | None) -> str | None:
    """The provider name of the block GPU-declaring tasks and GPU sidecars run on: the environment server's
    ``gpu_sandbox_provider`` in the configuration the command names (``GPU_SANDBOX_PROVIDER_KEY``), read like
    :func:`sandbox_provider_from_overrides` reads the ``sandbox:`` block. ``None`` when no GPU block is set
    (every service is then checked against the ``sandbox:`` block) or the block does not name one provider."""
    block = _block_from_overrides(overrides, GPU_SANDBOX_PROVIDER_KEY)
    if block is None:
        return None
    if not isinstance(block, str) or not block:
        raise ValueError(f"`{GPU_SANDBOX_PROVIDER_KEY}` must name a top-level sandbox block; got {block!r}")
    return _block_provider_name(overrides, sandbox, block)


def pairing_override_from_overrides(overrides: list[str]) -> bool:
    """Whether the run waives the agent admission checks: ``--allow-unsupported-pairing`` (which the CLI translates
    to ``+allow_unsupported_pairing=true``), a configuration file the command names that sets the key, or
    ``NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING``. The same reading as the collector's pairing guard at dispatch
    (:func:`~nemo_gym.config.global_config.pairing_override_enabled`), so the early admission check of ``prepare_target``
    and the guard agree on one explicit override."""
    from omegaconf import OmegaConf

    from nemo_gym.config.global_config import ALLOW_UNSUPPORTED_PAIRING_KEY_NAME, pairing_override_enabled

    block = _block_from_overrides(overrides, ALLOW_UNSUPPORTED_PAIRING_KEY_NAME)
    return pairing_override_enabled(OmegaConf.create({ALLOW_UNSUPPORTED_PAIRING_KEY_NAME: block}))


def compose_platform_from_overrides(overrides: list[str]) -> str:
    """The platform a Compose service's ``platform`` must match at prepare: the environment server's
    ``sandbox_provider_options.platform`` (``{os, arch}``) in the configuration the command names, else
    linux/amd64, exactly as the server reads it at seed (``deployment_platform``).

    Only the ``platform`` subkey is selected and resolved. The sibling options (volumes, credentials, a
    ``${results_dir}`` the server's configuration parser defaults later) are not needed at prepare and are
    not evaluated; a block that is not a mapping selects no platform and the default stands, the server's own
    configuration validation names the malformed block at dispatch."""
    platform = _block_from_overrides(overrides, f"{SANDBOX_PROVIDER_OPTIONS_KEY}.platform")
    return deployment_platform({"platform": platform} if platform is not None else None)


def compose_overlay_from_overrides(overrides: list[str]) -> HarborComposeOverlay | None:
    """The environment server's ``compose_overlay`` block in the configuration the command names
    (``COMPOSE_OVERLAY_KEY``), validated as the server validates it, or ``None`` when the block is absent or null.
    Prepare merges its files onto every task's Compose document and records their images, so the document the
    server starts is the one prepare checked; a block that does not validate is a configuration error here."""
    block = _block_from_overrides(overrides, COMPOSE_OVERLAY_KEY)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(
            f"`{COMPOSE_OVERLAY_KEY}` must be a mapping (files, mcp_servers, prompt_template); got {block!r}"
        )
    try:
        overlay = HarborComposeOverlay.model_validate(block)
    except ValidationError as exc:
        raise ValueError(f"`{COMPOSE_OVERLAY_KEY}`: {exc}") from exc
    return overlay if overlay.configured else None


def volume_sources_from_overrides(overrides: list[str], sandbox: str | None) -> dict[str, str] | None:
    """The ``volume_sources`` of the ``sandbox:`` block the command names, for checking Compose volumes at prepare.

    The block comes from the ``--sandbox`` provider file (when given) and the configuration the command
    line names, merged as :func:`_block_from_overrides` does. ``None`` when no ``volume_sources`` is
    configured: prepare then reports every external volume and foreign bind mount as unmapped.
    """
    return _block_volume_sources(overrides, sandbox, SANDBOX_BLOCK_KEY)


def gpu_volume_sources_from_overrides(overrides: list[str], sandbox: str | None) -> dict[str, str] | None:
    """The ``volume_sources`` of the block GPU-declaring tasks run on, for checking their Compose volumes at prepare.

    The block is the environment server's ``gpu_sandbox_provider`` in the configuration the command names
    (``GPU_SANDBOX_PROVIDER_KEY``). ``None`` when no GPU block is set: every task is then checked against
    the ``sandbox:`` block. An empty mapping when the block is set but carries no ``volume_sources``, which
    is what the server resolves such a task against.
    """
    block = _block_from_overrides(overrides, GPU_SANDBOX_PROVIDER_KEY)
    if block is None:
        return None
    if not isinstance(block, str) or not block:
        raise ValueError(f"`{GPU_SANDBOX_PROVIDER_KEY}` must name a top-level sandbox block; got {block!r}")
    return _block_volume_sources(overrides, sandbox, block) or {}


def _block_volume_sources(overrides: list[str], sandbox: str | None, block: str) -> dict[str, str] | None:
    files = [resolve_sandbox_config(sandbox)] if sandbox is not None else []
    sources = _block_from_overrides(overrides, f"{block}.{SANDBOX_BLOCK_VOLUME_SOURCES_KEY}", config_files=files)
    if sources is None:
        return None
    return validate_volume_sources(sources, f"the `{block}:` block")


def shared_storage_from_overrides(overrides: list[str], sandbox: str | None) -> bool | None:
    """Whether the ``sandbox:`` block's provider configures shared storage, for checking Compose named volumes and
    shared logs binds at prepare: OpenSandbox's ``shared_storage.host_path`` under the block's one provider key,
    read from the ``--sandbox`` provider file and the configuration the command names. ``None`` when the block
    names no single provider (the adapter then decides at seed); otherwise whether the path is set, which is
    the one setting the server's provider requires for every such volume."""
    return _block_shared_storage(overrides, sandbox, SANDBOX_BLOCK_KEY)


def gpu_shared_storage_from_overrides(overrides: list[str], sandbox: str | None) -> bool | None:
    """:func:`shared_storage_from_overrides` for the block GPU-declaring tasks and GPU sidecars run on
    (``GPU_SANDBOX_PROVIDER_KEY``); ``None`` when no GPU block is set."""
    block = _block_from_overrides(overrides, GPU_SANDBOX_PROVIDER_KEY)
    if block is None:
        return None
    if not isinstance(block, str) or not block:
        raise ValueError(f"`{GPU_SANDBOX_PROVIDER_KEY}` must name a top-level sandbox block; got {block!r}")
    return _block_shared_storage(overrides, sandbox, block)


def _block_shared_storage(overrides: list[str], sandbox: str | None, block: str) -> bool | None:
    provider = _block_provider_name(overrides, sandbox, block)
    if provider is None:
        return None
    files = [resolve_sandbox_config(sandbox)] if sandbox is not None else []
    host_path = _block_from_overrides(overrides, f"{block}.{provider}.shared_storage.host_path", config_files=files)
    return bool(host_path)


def _patterns(args: argparse.Namespace, name: str) -> list[str]:
    raw = getattr(args, name, None) or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


@exit_cleanly_on_config_error
def run_target(args: argparse.Namespace, overrides: list[str]) -> None:
    """Entry point for ``gym eval run <target> ...``."""
    from nemo_gym.cli.main import _merge_config_paths, dispatch
    from nemo_gym.rollouts.rollout_collection import RolloutCollectionHelper

    if getattr(args, "no_serve", False):
        args._parser.error("A Harbor target starts its own servers; drop --no-serve")
    agent_name = getattr(args, "agent", None)
    if not agent_name:
        args._parser.error("A Harbor target needs `--agent <harness>` (for example `--agent hermes_agent`)")
    sandbox = getattr(args, "sandbox", None)
    # Resolve selectors before preparation can download a registry dataset.
    try:
        agent = resolve_agent(agent_name)
        if sandbox is not None:
            resolve_sandbox_config(sandbox)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        args._parser.error(str(exc))
    if agent.impl_name == ORACLE_AGENT:
        # The oracle runs the reference solution, not a model: its run is `gym dataset validate`, whose results
        # live under results/harbor/validation/ where no model run, resumed or not, can pick them up as its own.
        validate_target(args, overrides, agent=agent)
        return
    try:
        # The `builds:` and `sandbox:` blocks come from the configuration the command names; the run config
        # does not exist yet.
        build = build_options_from_overrides(overrides)
        prepared = prepare_target(
            args.target,
            exclude=_patterns(args, "exclude_tasks"),
            only=_patterns(args, "only_tasks"),
            build=build,
            sandbox_provider=sandbox_provider_from_overrides(overrides, sandbox),
            gpu_sandbox_provider=gpu_sandbox_provider_from_overrides(overrides, sandbox),
            compose_platform=compose_platform_from_overrides(overrides),
            volume_sources=volume_sources_from_overrides(overrides, sandbox),
            gpu_volume_sources=gpu_volume_sources_from_overrides(overrides, sandbox),
            shared_storage=shared_storage_from_overrides(overrides, sandbox),
            gpu_shared_storage=gpu_shared_storage_from_overrides(overrides, sandbox),
            compose_overlay=compose_overlay_from_overrides(overrides),
            rows_name=_rows_name(agent),
            registry_credentials=registry_credentials_from_overrides(overrides, sandbox),
            agent=agent.impl_name,
            allow_unsupported_pairing=pairing_override_from_overrides(overrides),
        )
    except (
        HarborTaskError,
        HubError,
        BuildError,
        ImageConfigError,
        ManifestError,
        ConfigError,
        ValueError,
        OSError,
    ) as exc:
        args._parser.error(str(exc))
    if prepared.skipped and not getattr(args, "skip_unloadable", False):
        args._parser.error(
            f"{len(prepared.skipped)} task(s) could not load (see {prepared.output_dir / 'skipped.json'}):"
            f"{_listed(prepared.skipped)}\nUse --skip-unloadable to explicitly run only the loadable subset."
        )
    try:
        config_path, tokens = build_run(prepared, agent, sandbox=sandbox, overrides=overrides)
    except (HarborTaskError, OSError) as exc:
        args._parser.error(str(exc))
    tokens = _merge_config_paths(tokens)
    collection = _resolved_collection(tokens, prepared)
    helper = RolloutCollectionHelper()
    expected: list[dict[str, Any]] | None = None
    before_dispatch = None
    cache = _resumable_cache(collection)
    if cache is not None:
        # The collector resumes its saved inputs, not the rows just prepared: a selection or digest that
        # changed since would run (or 404 on) tasks the server no longer maps. Refuse before startup.
        expected = helper._preprocess_rows_from_config(collection)
        conflict = _resume_conflict(cache, expected)
        if conflict:
            args._parser.error(conflict)
        # The collector resumes the schedule it saved (the materialized inputs: every task and repeat as that run
        # expanded them, with the input each carried), not the rows the current config expands: a resume with
        # another `num_repeats` passes the check above and dispatches no further attempts, so the summary's
        # expected attempts and input echo are the saved ones, not a missing attempt the run never owed.
        expected = _validation_rows(cache)
    else:
        # A fresh start clears its output files only once the servers are up: what sits under those names now
        # is an earlier run's, and the summary must not report it as this run's if the collector never gets there.
        before_dispatch = output_signatures(Path(collection.output_jsonl_fpath))
        # The collated split metrics an earlier run left are that run's selection; the data processor would
        # refuse this one against them. Cleared here, they are rewritten before any server starts.
        _clear_collector_metrics(Path(collection.output_jsonl_fpath), split=_collector_split())
    print(f"Run config written to {config_path} (datasets folder: {datasets_dir()})")
    try:
        dispatch("nemo_gym.cli.eval:e2e_rollout_collection", tokens)
    finally:
        _write_summary(
            Path(collection.output_jsonl_fpath),
            command="gym eval run",
            target=args.target,
            prepared=prepared,
            agent=agent,
            run_config=config_path,
            # A fresh run reads its selected rows only for the summary, after the collector has had them.
            expected=expected if expected is not None else lambda: helper._preprocess_rows_from_config(collection),
            split=_collector_split(),
            before_dispatch=before_dispatch,
        )


def _collector_split() -> str | None:
    """The split the resolved run config dispatches (``validation`` unless overridden), for the run summary."""
    from nemo_gym.config.global_config import get_global_config_dict

    split = get_global_config_dict().get("split")
    return str(split) if split else None


def _clear_collector_metrics(rollouts: Path, *, split: str | None) -> None:
    """Remove the collated split metrics an earlier run left under ``<rollouts stem>/preprocessed_datasets/``
    (``<split>_metrics.json`` and the ``<split>_metrics_conflict.json`` of a refused run), right before a fresh
    start dispatches the collector.

    ``e2e_rollout_collection`` collates the rows it dispatches there and checks the metrics it computes against
    that file, refusing a mismatch ("Found conflicting aggregate metrics") unless ``overwrite_metrics_conflicts``
    is set. That protects a committed source dataset; on this path the rows are regenerated from the task folder
    on every command and the output folder is reused per taskset, so the file is the previous selection's, and
    a user must neither see that error nor pass the flag. A resumed run keeps the file: its rows are unchanged
    (:func:`_resume_conflict`). So does a run that reuses the prepared input (``reuse_existing_data_preparation``
    with the collated ``<split>.jsonl`` present): the processor does not run, and the metrics stay with the input
    they describe. ``split`` is the resolved run config's (:func:`_collector_split`); without one nothing is known
    to clear.
    """
    from nemo_gym.config.global_config import get_global_config_dict
    from nemo_gym.data.tasks.harbor.run_summary import PREPROCESSED_SUBDIR

    if not split:
        return
    preprocessed = Path(rollouts).with_suffix("") / PREPROCESSED_SUBDIR
    reuses_input = bool(get_global_config_dict().get("reuse_existing_data_preparation"))
    if reuses_input and (preprocessed / f"{split}.jsonl").is_file():
        return
    metrics = preprocessed / f"{split}_metrics.json"
    for path in (metrics, metrics.with_name(f"{metrics.stem}_conflict.json")):
        path.unlink(missing_ok=True)


def _write_summary(
    rollouts: Path, *, expected: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]] | None, **kwargs: Any
) -> None:
    """Write the run summary beside ``rollouts`` (:mod:`nemo_gym.data.tasks.harbor.run_summary`) and say where.

    Runs after collection whatever it did, so a run that stopped early still leaves a summary of what it
    wrote; a summary that cannot be written is logged and never displaces the run's own outcome. ``expected``
    may be a callable that reads the collector's selected rows, so a run that did not need them before
    dispatch reads them here, inside the same guard. ``before_dispatch`` (in ``kwargs``) is what a fresh start
    found under its output names before dispatch, so files it never got to clear are not reported as its own.
    """
    try:
        if callable(expected):
            expected = expected()
        path = write_run_summary(rollouts, expected=expected, **kwargs)
    except Exception:
        logger.warning("Could not write the run summary beside %s", rollouts, exc_info=True)
        return
    print(f"Run summary written to {path}")


def _rows_name(agent: AgentSelection) -> str:
    """The rows file of one agent's run on a taskset: ``tasks_<agent>.jsonl`` next to ``run_config_<agent>.yaml``."""
    return f"tasks_{agent.instance_name}.jsonl"


def _resolved_collection(tokens: list[str], prepared: PreparedTaskset) -> "RolloutCollectionConfig":
    """The collector's effective configuration for ``tokens``: resolved through the same parser and cache
    dispatch uses, file overlays included, so what it would resume and run is read from it rather than
    guessed from the command line."""
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationResolutionError

    from nemo_gym.config.global_config import (
        get_global_config_dict,
        reset_global_config_dict,
        translate_interpolation_error,
    )
    from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig

    sys.argv = [sys.argv[0], *tokens]
    # These tokens, not whatever this process resolved before them (the run config did not exist yet); the
    # result stays cached, so dispatch runs on it without a second parse.
    reset_global_config_dict()
    config = get_global_config_dict()
    try:
        resolved = OmegaConf.to_container(config, resolve=True)
    except InterpolationResolutionError as exc:
        raise translate_interpolation_error(exc) from exc
    return RolloutCollectionConfig.model_validate({**resolved, "input_jsonl_fpath": str(prepared.rows_path)})


def _requested_output(tokens: list[str], default: Path) -> Path:
    """The rollouts path ``tokens`` ask for (``--output``, ``+output_jsonl_fpath=`` or a ``--config`` overlay), or
    ``default`` when they name none.

    Resolved through the same parser dispatch uses, so a quoted ``--output`` and a file overlay both count, for
    the path that dispatches nothing and so has no run config to resolve a collection from.
    """
    from nemo_gym.config.global_config import get_global_config_dict, reset_global_config_dict

    sys.argv = [sys.argv[0], *tokens]
    reset_global_config_dict()
    requested = get_global_config_dict().get("output_jsonl_fpath")
    return Path(str(requested)) if requested else default


def _resumable_cache(collection: "RolloutCollectionConfig") -> Path | None:
    """The saved inputs the collector would resume, or None when it starts fresh. Its own condition: the
    effective ``resume_from_cache`` (flag or config file) and both the results file and the saved inputs."""
    if not collection.resume_from_cache:
        return None
    output = Path(collection.output_jsonl_fpath)
    materialized = materialized_path_for(output)
    if not output.is_file() or not materialized.is_file():
        return None
    return materialized


def _resume_conflict(materialized: Path, expected: list[dict[str, object]]) -> str | None:
    """Why ``--resume`` cannot go ahead: the cached rows name other tasks or digests than the rows the collector
    would run now (``expected``, after ``limit``)."""

    def identity(row: dict[str, object]) -> tuple[str, str]:
        return _validation_key(row)[:2]

    cached = _task_digests(_validation_rows(materialized), identity)
    current = _task_digests(expected, identity)
    if cached == current:
        return None
    dropped = sorted(name for taskset, name in cached if (taskset, name) not in current)
    added = sorted(name for taskset, name in current if (taskset, name) not in cached)
    changed = sorted(
        name for key, digest in current.items() if key in cached and cached[key] != digest for name in [key[1]]
    )
    detail = ", ".join(
        part
        for part in (
            f"no longer selected: {', '.join(dropped)}" if dropped else "",
            f"newly selected: {', '.join(added)}" if added else "",
            f"changed digest: {', '.join(changed)}" if changed else "",
        )
        if part
    )
    return (
        f"Cannot resume from {materialized}: the task selection or task digests differ from the cached run "
        f"({detail}). Rerun with the same --only-tasks/--exclude-tasks, or pass +resume_from_cache=false to "
        f"start over, or a new +output_jsonl_fpath to keep the old results"
    )


@exit_cleanly_on_config_error
def validate_target(args: argparse.Namespace, overrides: list[str], *, agent: AgentSelection | None = None) -> None:
    """Entry point for ``gym dataset validate <target>``: run every task's reference solution and score it.

    ``gym eval run <target> --agent harbor_oracle`` lands here too, with its ``agent`` selection, so the oracle's
    results never share a folder with a model's.
    """
    from nemo_gym.cli.main import _merge_config_paths, dispatch
    from nemo_gym.rollouts.rollout_collection import RolloutCollectionHelper

    # Evaluation and reference validation must never share inputs, configs, or cached results.
    try:
        agent = agent or resolve_agent(ORACLE_AGENT)
        sandbox = getattr(args, "sandbox", None)
        if sandbox is not None:
            resolve_sandbox_config(sandbox)
        prepared = prepare_target(
            args.target,
            output_root=DEFAULT_OUTPUT_ROOT / "validation",
            # A template renders where `gym eval run` renders it: one rendered dataset, one set of records (an image
            # built for it once), while the oracle's rows and results stay under validation/.
            render_root=DEFAULT_OUTPUT_ROOT,
            exclude=_patterns(args, "exclude_tasks"),
            only=_patterns(args, "only_tasks"),
            build=build_options_from_overrides(overrides),
            sandbox_provider=sandbox_provider_from_overrides(overrides, sandbox),
            gpu_sandbox_provider=gpu_sandbox_provider_from_overrides(overrides, sandbox),
            compose_platform=compose_platform_from_overrides(overrides),
            volume_sources=volume_sources_from_overrides(overrides, sandbox),
            gpu_volume_sources=gpu_volume_sources_from_overrides(overrides, sandbox),
            shared_storage=shared_storage_from_overrides(overrides, sandbox),
            gpu_shared_storage=gpu_shared_storage_from_overrides(overrides, sandbox),
            compose_overlay=compose_overlay_from_overrides(overrides),
            rows_name=_rows_name(agent),
            registry_credentials=registry_credentials_from_overrides(overrides, sandbox),
        )
    except (
        HarborTaskError,
        HubError,
        BuildError,
        ImageConfigError,
        ConfigError,
        ValueError,
        OSError,
        yaml.YAMLError,
    ) as exc:
        args._parser.error(str(exc))
    unvalidated = [task.task_id for task in prepared.tasks if not task.has_solution]
    # Left out of the run, not failed: the summary names them with the reason so the result folder explains itself.
    left_out = dict.fromkeys(unvalidated, NO_REFERENCE_SOLUTION)
    prepared = replace(prepared, tasks=[task for task in prepared.tasks if task.has_solution])
    write_rows(
        prepared.tasks,
        prepared.taskset,
        prepared.rows_path,
        dataset_digest=prepared.dataset_digest,
        task_source=prepared.task_source,
    )
    rollouts = prepared.output_dir / "rollouts.jsonl"
    expected = []
    if prepared.tasks:
        # Oracle runs reference scripts directly; it needs no placeholder model server.
        overrides = list(overrides)
        # Validation is not an evaluation: a reference solution scores the same every time, so the dataset's
        # `[gym.defaults].num_repeats` does not apply to it. An explicit `+num_repeats=` still wins.
        if not _has_override(overrides, "num_repeats"):
            overrides.append("+num_repeats=1")
        config_path, tokens = build_run(prepared, agent, sandbox=sandbox, overrides=overrides)
        tokens = _merge_config_paths(tokens)
        # Resolve through the same parser and cache used by dispatch, including file overlays.
        collection = _resolved_collection(tokens, prepared)
        expected = RolloutCollectionHelper()._preprocess_rows_from_config(collection)
        rollouts = Path(collection.output_jsonl_fpath)
        before_dispatch = None
        cache = _resumable_cache(collection)
        if cache is not None:
            # The collector resumes its saved inputs, not the just-prepared task rows. Reject a stale
            # selection or digest before startup so an old reward cannot validate a changed solution.
            cached_digests = _task_digests(_validation_rows(cache), _validation_key)
            if cached_digests != _task_digests(expected, _validation_key):
                raise ConfigError(
                    f"Cannot resume Harbor validation from {cache}: "
                    "the selected tasks or task digests have changed. Use +resume_from_cache=false "
                    "to validate the current tasks, or choose a new +output_jsonl_fpath to keep the old results."
                )
            # Only the oracle agent stamps `response.metadata.oracle`. Cached results without it were written by
            # another run (a model evaluation on the same file); resumed, they would stand in for the oracle's.
            foreign = sorted(
                _validation_key(row)[1]
                for row in _validation_rows(rollouts)
                if ((row.get("response") or {}).get("metadata") or {}).get("oracle") is None
            )
            if foreign:
                raise ConfigError(
                    f"Cannot resume Harbor validation from {rollouts}: {len(foreign)} cached result(s) were not "
                    f"produced by the oracle agent (no oracle metadata): {', '.join(foreign[:5])}"
                    f"{', ...' if len(foreign) > 5 else ''}. Use +resume_from_cache=false to validate the current "
                    "tasks, or choose a new +output_jsonl_fpath to keep the old results."
                )
            # What the collector resumes is its saved schedule (see run_target): the report and the summary
            # expect the saved attempts, with the input each carried.
            expected = _validation_rows(cache)
        else:
            # See run_target: a fresh start's files under the output names are an earlier run's until the
            # collector clears them, which happens only once the servers are up; the collated metrics of an
            # earlier selection would make the data processor refuse this one, so they go now.
            before_dispatch = output_signatures(rollouts)
            _clear_collector_metrics(rollouts, split=_collector_split())
        print(f"Run config written to {config_path}; validating {len(expected)} oracle rollout(s)")
        try:
            dispatch("nemo_gym.cli.eval:e2e_rollout_collection", tokens)
        finally:
            _write_summary(
                rollouts,
                command="gym dataset validate",
                target=args.target,
                prepared=prepared,
                agent=agent,
                run_config=config_path,
                expected=expected,
                split=_collector_split(),
                before_dispatch=before_dispatch,
                unvalidated=left_out,
            )
    else:
        # Nothing to dispatch (no selected task has a solution/): the summary still says what was selected,
        # beside the rollouts file the command line asked for (there is no run config to resolve it from).
        rollouts = _requested_output(_merge_config_paths(list(overrides)), rollouts)
        _write_summary(
            rollouts,
            command="gym dataset validate",
            target=args.target,
            prepared=prepared,
            agent=agent,
            run_config=None,
            expected=expected,
            # Nothing ran, so whatever sits under the output names is an earlier run's.
            before_dispatch=output_signatures(rollouts),
            unvalidated=left_out,
        )
    report = summarize_validation(rollouts, unvalidated, expected=expected, skipped=prepared.skipped)
    print(report.text)
    if not report.ok:
        sys.exit(1)
