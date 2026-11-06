# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a resolved config's benchmark datasets by running their prepare scripts, and plan a benchmark.

The `gym eval prepare` and `gym eval plan` commands and callers that hold a resolved config in Python share this
code. It sits in `environment` because it builds on `BenchmarkConfig`. It is not `nemo_gym.data.dataset_preparation`,
which prepares training data.
"""

import importlib
import importlib.util
import json
import logging
import os
import site
import subprocess
import sys
import sysconfig
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Sequence

from omegaconf import DictConfig, OmegaConf
from pydantic import Field
from tqdm.auto import tqdm

from nemo_gym import PARENT_DIR, WORKING_DIR, _resolve_under_cwd_or_install
from nemo_gym.config.config_types import BaseNeMoGymCLIConfig, BenchmarkDatasetConfig, ConfigError, DatasetConfig
from nemo_gym.config.global_config import (
    get_first_server_config_dict,
    resolve_dataset_agent,
    taskset_environment_server_name,
)
from nemo_gym.environment.benchmarks import BenchmarkConfig
from nemo_gym.eval.eval_plan import EvalPlan, PlanBenchmark, PlanGym, build_eval_plan
from nemo_gym.orchestration.jobs import installed_gym_commit
from nemo_gym.package_info import __version__


logger = logging.getLogger(__name__)


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

    uv_venv_dir: Path = Field(default=WORKING_DIR, description="Writable root for isolated preparation environments")
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
    (benchmark_config, prepare_module_path, prepare_script_args, python) = args[:4]
    config = args[4] if len(args) > 4 else None

    print(f"Preparing dataset: {benchmark_config.name}")

    if python is not None:
        _run_isolated_prepare(
            python,
            prepare_module_path,
            "prepare",
            Path(benchmark_config.dataset.jsonl_fpath),
            prepare_script_args,
            config=config,
        )
        return
    from nemo_gym.cli._prepare_worker import load_preparer

    module = load_preparer(prepare_module_path)
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


def _prepare_environment(benchmark_config: "BenchmarkConfig", venv_root: Path | None = None) -> Path | None:
    """Use explicit dependency arguments as before, or sync an isolated locked project."""
    from nemo_gym.cli._locked_env import sync_environment

    dataset = benchmark_config.dataset
    if dataset.prepare_dependencies:
        # Explicit arguments retain their existing meaning, including editable installs
        # and requirements-file options. Do not silently ignore caller overrides.
        _install_prepare_dependencies(benchmark_config)
        return None
    if dataset.prepare_project is None:
        return None
    project = _resolve_under_cwd_or_install(dataset.prepare_project).resolve()
    venv = project / ".venv"
    if venv_root is not None:
        try:
            relative = project.relative_to(PARENT_DIR.resolve())
        except ValueError:
            import hashlib

            relative = Path("external") / (project.name + "-" + hashlib.sha256(str(project).encode()).hexdigest()[:12])
        venv = venv_root.absolute() / "preparation" / relative / ".venv"
    try:
        sync_environment(project, venv, sys.executable, [], [])
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise ConfigError(
            f"Could not sync locked prepare_project for dataset '{benchmark_config.name}': {exc}"
        ) from exc
    return venv / "bin/python"


def _run_isolated_prepare(
    python: Path, module: str, action: str, expected: Path, args: dict, *, config: DictConfig | dict | None = None
) -> dict:
    """Keep preparer stdout visible and pass structured results over a separate file."""
    from nemo_gym.config.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, maybe_get_global_config_dict

    # The worker script is a module of `nemo_gym.cli`, which stays where it is; locate it without importing it here.
    worker = Path(importlib.util.find_spec("nemo_gym.cli._prepare_worker").origin)
    if module.endswith(".py"):
        root = Path(module).parent
    else:
        relative = Path(*module.split(".")).with_suffix(".py")
        script = _resolve_under_cwd_or_install(relative)
        root = script.parents[len(relative.parts) - 1]
    with TemporaryDirectory(prefix="gym-prepare-") as tmp:
        result = Path(tmp) / "result.json"
        options = Path(tmp) / "arguments.json"
        if config is None:
            config = maybe_get_global_config_dict()
        serialized_config = (
            OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config or {}
        )
        serialized_args = OmegaConf.to_container(args, resolve=True) if OmegaConf.is_config(args) else args
        options.write_text(json.dumps(serialized_args, default=_prepare_json_default))
        subprocess.run(
            [
                str(python),
                str(worker),
                "--root",
                str(root.absolute()),
                "--module",
                module,
                "--action",
                action,
                "--expected",
                str(expected.absolute()),
                "--args-file",
                str(options),
                "--result",
                str(result),
            ],
            check=True,
            env=os.environ
            | {
                NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: json.dumps(serialized_config, default=_prepare_json_default),
                "NEMO_GYM_PREPARE_VENV": str(python.parent.parent),
                "VIRTUAL_ENV": str(python.parent.parent),
                "PATH": str(python.parent) + os.pathsep + os.environ.get("PATH", ""),
            },
        )
        return json.loads(result.read_text())


def _prepare_json_default(value: Any) -> Any:
    """Serialize path arguments without silently coercing unsupported values."""
    if isinstance(value, Path):
        return str(value)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    raise TypeError(f"Cannot serialize preparation value of type {type(value).__name__}")


def datasets_to_prepare(global_config_dict: DictConfig) -> List[BenchmarkConfig]:
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


def prepare_datasets(datasets: Sequence[BenchmarkConfig], prepare_benchmark_config: PrepareBenchmarkConfig) -> None:
    """Run each dataset's prepare script, after validating all of them."""
    from nemo_gym.config.global_config import maybe_get_global_config_dict

    config = maybe_get_global_config_dict()
    # Validate all datasets before preparing any
    prepare_script_missing: List[BenchmarkConfig] = []
    prepare_function_missing: List[BenchmarkConfig] = []

    validated: list[tuple[BenchmarkConfig, str, dict, Path | None, DictConfig | None]] = []
    already_prepared: List[BenchmarkConfig] = []
    for benchmark_config in datasets:
        prepare_script = benchmark_config.dataset.prepare_script
        if prepare_script is None:
            raise ConfigError(f"Dataset {benchmark_config.name!r} has no prepare_script")
        prepare_script_path = _resolve_under_cwd_or_install(prepare_script)
        if not prepare_script_path.exists():
            prepare_script_missing.append(benchmark_config)
            continue

        prepare_module_path = (
            str(prepare_script)
            if Path(prepare_script).is_absolute()
            else ".".join(Path(prepare_script).with_suffix("").parts)
        )
        python = _prepare_environment(benchmark_config, prepare_benchmark_config.uv_venv_dir)
        if python is not None:
            probe = _run_isolated_prepare(
                python,
                prepare_module_path,
                "inspect",
                Path(benchmark_config.dataset.jsonl_fpath),
                {"check_cache": prepare_benchmark_config.use_cached_prepared_benchmarks},
                config=config,
            )
            if not probe["has_prepare"]:
                prepare_function_missing.append(benchmark_config)
                continue
            if probe["cache_current"]:
                already_prepared.append(benchmark_config)
                continue
        else:
            from nemo_gym.cli._prepare_worker import load_preparer

            module = load_preparer(prepare_module_path)
            if not hasattr(module, "prepare"):
                prepare_function_missing.append(benchmark_config)
                continue
            is_already_prepared = Path(benchmark_config.dataset.jsonl_fpath).exists()
            if prepare_benchmark_config.use_cached_prepared_benchmarks and is_already_prepared:
                is_current = getattr(module, "is_prepared_data_current", None)
                if not callable(is_current) or is_current(benchmark_config.dataset.jsonl_fpath):
                    already_prepared.append(benchmark_config)
                    continue
                print(f"The cached file for {benchmark_config.name} is out of date, so it will be prepared again.")

        validated.append(
            (benchmark_config, prepare_module_path, dict(prepare_benchmark_config.prepare_script_args), python, config)
        )

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


def gym_relative_path(path: str) -> str:
    """A config path relative to the Gym checkout when it lies inside it, so plans match across machines."""
    try:
        return str(Path(path).resolve().relative_to(PARENT_DIR.resolve()))
    except ValueError:
        return str(path)


def plan_for_config(
    global_config_dict: DictConfig,
    prepare_config: PrepareBenchmarkConfig,
    task_ids: Sequence[str] | None = None,
) -> EvalPlan:
    """Prepare the one benchmark a resolved config declares and build its plan (shared by `gym eval plan`)."""
    # Other datasets with a prepare script are not benchmarks, so they are not part of the plan.
    benchmarks = [
        dataset
        for dataset in datasets_to_prepare(global_config_dict)
        if isinstance(dataset.dataset, BenchmarkDatasetConfig)
    ]
    if len(benchmarks) != 1:
        raise ConfigError(
            f"A plan covers exactly one benchmark, but the config declares {len(benchmarks)}: "
            f"{[benchmark.name for benchmark in benchmarks]}."
        )
    benchmark = benchmarks[0]

    prepare_datasets([benchmark], prepare_config)
    rows = [json.loads(line) for line in benchmark.dataset.jsonl_fpath.read_text().splitlines() if line.strip()]
    return build_eval_plan(
        rows,
        benchmark=PlanBenchmark(
            name=benchmark.name,
            config_paths=[gym_relative_path(path) for path in global_config_dict.get("config_paths") or []],
        ),
        taskset=benchmark.dataset.taskset or benchmark.name,
        gym=PlanGym(version=__version__, commit=installed_gym_commit()),
        selected_task_ids=task_ids,
    )
