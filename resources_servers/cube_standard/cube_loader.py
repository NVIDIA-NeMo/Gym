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
"""
Runtime CUBE package loading utilities.

Mirrors the pattern used in cube-standard's quick_check.py:
  pip_install → find_benchmark_class → detect_parallelization_mode → select_task_config

Also provides generate_task_jsonl() to produce NeMo Gym input JSONL from any CUBE benchmark.
"""

import importlib
import importlib.metadata
import inspect
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


def _hydrate_extra_benchmark_config(raw: dict) -> dict:
    """
    Recursively hydrate any nested dict with a '_type' key into the corresponding
    Python object. Used to construct VMBackend / ContainerBackend instances from YAML.

    Example input:
        {"vm_backend": {"_type": "cube_vm_backend.local.LocalQEMUVMBackend",
                        "cache_dir": "/tmp/cube-vms", "memory": "8G", "cpus": 4}}
    Example output:
        {"vm_backend": LocalQEMUVMBackend(cache_dir="/tmp/cube-vms", memory="8G", cpus=4)}

    Non-dict values are passed through unchanged.
    """
    result = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "_type" in value:
            type_path = value["_type"]
            kwargs = {k: v for k, v in value.items() if k != "_type"}
            # Recursively hydrate nested dicts too
            kwargs = _hydrate_extra_benchmark_config(kwargs)
            module_path, class_name = type_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            result[key] = cls(**kwargs)
        else:
            result[key] = value
    return result


def pip_install(package: str, version: Optional[str], dev_url: Optional[str]) -> None:
    """
    Idempotent pip install. Skips if the package is already installed at the right version.

    Resolution order:
    1. If already installed at the requested version — return immediately.
    2. pip install <package>==<version> (or just <package> if version is None).
    3. If PyPI install fails and dev_url is provided — pip install <dev_url>.
    4. If all installs fail — raise RuntimeError with pip output.
    """
    try:
        dist = importlib.metadata.distribution(package)
        if version is None or dist.version == version:
            logger.info("Package '%s' already installed (version=%s). Skipping.", package, dist.version)
            return
        logger.info(
            "Package '%s' installed at version %s, but %s requested. Re-installing.",
            package,
            dist.version,
            version,
        )
    except importlib.metadata.PackageNotFoundError:
        logger.info("Package '%s' not found. Installing.", package)

    pkg_spec = f"{package}=={version}" if version else package
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg_spec],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode == 0:
        logger.info("Installed '%s' from PyPI.", pkg_spec)
        return

    logger.warning("PyPI install of '%s' failed. stdout=%s stderr=%s", pkg_spec, result.stdout, result.stderr)

    if dev_url:
        logger.info("Trying dev install from: %s", dev_url)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", dev_url],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info("Installed '%s' from dev_url.", package)
            return
        logger.warning(
            "Dev install of '%s' from '%s' also failed. stdout=%s stderr=%s",
            package,
            dev_url,
            result.stdout,
            result.stderr,
        )

    raise RuntimeError(
        f"pip install failed for '{pkg_spec}'.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def find_benchmark_class(package: str) -> tuple[Optional[Type], str]:
    """
    Resolve the Benchmark class for a CUBE package.

    Resolution order:
    1. cube.benchmarks entry point matching the package name exactly.
    2. importlib.import_module(package.replace("-", "_")) → class named "Benchmark".
    3. Duck-type scan: any class in the module with a get_task_configs() method,
       whose __module__ starts with the package name.

    Returns:
        (cls, "") on success
        (None, error_message) on failure
    """
    # Strategy 1: entry points
    try:
        eps = list(importlib.metadata.entry_points(group="cube.benchmarks"))
        matched = [ep for ep in eps if ep.name == package]
        if matched:
            try:
                cls = matched[0].load()
                logger.info("Resolved '%s' via entry point: %s", package, matched[0].value)
                return cls, ""
            except Exception as e:
                return None, f"Entry point load failed: {e}"
    except Exception as e:
        logger.debug("Entry point lookup failed: %s", e)

    # Strategy 2: direct module import
    module_name = package.replace("-", "_")
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        return None, f"Could not import '{module_name}': {e}"

    # Look for a class literally named "Benchmark"
    benchmark_cls = getattr(mod, "Benchmark", None)
    if benchmark_cls is not None and inspect.isclass(benchmark_cls):
        logger.info("Resolved '%s' as module.Benchmark", package)
        return benchmark_cls, ""

    # Scan for any class named "Benchmark" in the module's namespace
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if attr is not None and inspect.isclass(attr) and attr.__name__ == "Benchmark":
            logger.info("Resolved '%s' via name scan: %s", package, attr)
            return attr, ""

    # Strategy 3: duck-type scan — any class with get_task_configs()
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if (
            attr is not None
            and inspect.isclass(attr)
            and getattr(attr, "__module__", "").startswith(module_name)
            and callable(getattr(attr, "get_task_configs", None))
        ):
            logger.info("Resolved '%s' via duck-type scan: %s", package, attr)
            return attr, ""

    return None, f"No Benchmark class found in '{module_name}'"


def detect_parallelization_mode(benchmark_class: Type) -> tuple[str, int]:
    """
    Read BenchmarkMetadata ClassVar from the benchmark class — no instantiation needed.

    Returns:
        (parallelization_mode, max_concurrent_tasks)
        parallelization_mode: "task-parallel" | "benchmark-parallel"
        max_concurrent_tasks: integer cap (9999 = effectively unlimited)
    """
    meta = getattr(benchmark_class, "benchmark_metadata", None)
    if meta is None:
        logger.warning("'%s' has no benchmark_metadata — defaulting to task-parallel.", benchmark_class.__name__)
        return "task-parallel", 9999
    mode = getattr(meta, "parallelization_mode", "task-parallel")
    max_concurrent = getattr(meta, "max_concurrent_tasks", 9999)
    return mode, max_concurrent


def select_task_config(benchmark: Any, task_id: Optional[str], seed: Optional[int]) -> Any:
    """
    Select a TaskConfig from benchmark.get_task_configs().

    task_id=None → returns the first config.
    task_id is specified → finds the matching config by task_id.
    seed is specified → returns a copy of the config with seed applied.

    Raises ValueError if no matching task is found.
    """
    configs = list(benchmark.get_task_configs())

    if not configs:
        raise ValueError(f"Benchmark '{getattr(benchmark, 'name', benchmark)}' returned no task configs.")

    if task_id is None:
        config = configs[0]
    else:
        config = next((c for c in configs if c.task_id == task_id), None)

    if config is None:
        raise ValueError(
            f"Task '{task_id}' not found in benchmark '{getattr(benchmark, 'name', benchmark)}'. "
            f"Available tasks: {[c.task_id for c in configs]}"
        )

    if seed is not None:
        config = config.model_copy(update={"seed": seed})

    return config


def generate_task_jsonl(
    cube_id: str,
    output_path: str,
    extra_benchmark_config: Optional[Dict[str, Any]] = None,
    dev_url: Optional[str] = None,
) -> None:
    """
    Generate a NeMo Gym input JSONL file from any CUBE benchmark.

    Each line in the output JSONL contains:
        {
            "task_id": "<task_id>",
            "seed": <seed_or_null>,
            "responses_create_params": {"input": []}
        }

    The empty input list is intentional — CubeAgent fills it from /seed_session.

    Args:
        cube_id: PyPI package name / entry-point key (e.g. "miniwob-cube").
        output_path: Path to write the JSONL file.
        extra_benchmark_config: Extra kwargs for the Benchmark constructor (pre-hydration).
        dev_url: Fallback git/local URL if PyPI install fails.
    """
    # Install
    pip_install(cube_id, None, dev_url)

    # Resolve class
    cls, err = find_benchmark_class(cube_id)
    if cls is None:
        raise RuntimeError(f"Cannot find Benchmark class for '{cube_id}': {err}")

    # Hydrate config
    raw_config = extra_benchmark_config or {}
    hydrated_config = _hydrate_extra_benchmark_config(raw_config)

    # Instantiate and setup
    benchmark = cls(**hydrated_config)
    benchmark.setup()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        configs = list(benchmark.get_task_configs())
        with output.open("w", encoding="utf-8") as f:
            for config in configs:
                record = {
                    "task_id": config.task_id,
                    "seed": getattr(config, "seed", None),
                    "responses_create_params": {"input": []},
                }
                f.write(json.dumps(record) + "\n")
        logger.info("Wrote %d task entries to %s", len(configs), output_path)
    finally:
        try:
            benchmark.close()
        except Exception as e:
            logger.warning("benchmark.close() raised during generate_task_jsonl: %s", e)
