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
"""Post-run exports of a submitted benchmark: which ones apply, with what settings, and whether they are valid."""

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_gym.run_exporter_registry import RUN_EXPORTER_CONFIGS


if TYPE_CHECKING:
    from nemo_gym.orchestration.api import SubmitConfig


# Set per benchmark by the orchestrator, so a submit config may not set them.
RESERVED_SETTINGS = ("rollouts_jsonl_fpath", "result_json_fpath", "exporter")
_GYM_JOB_ID_PREFIX = "gym-job-"


def _merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """`overrides` on top of `defaults`: nested settings (such as `meta`) merge, everything else is replaced."""
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def effective_exports(config: "SubmitConfig", benchmark_name: str) -> dict[str, dict[str, Any]]:
    """The exports a benchmark runs after its rollouts, by exporter name, with the submit config's own settings.

    `job.export` gives every benchmark a default; a benchmark's own `export` override it setting by setting,
    and `null` for an exporter drops it for that benchmark. A benchmark that runs a `command` has no rollouts
    file for an exporter to read, so it takes none of the job-level defaults.
    """
    benchmark = config.driver.benchmarks[benchmark_name]
    if benchmark.command is not None:
        return {}
    effective: dict[str, dict[str, Any]] = {
        name: deepcopy(settings) for name, settings in config.job.export.items() if settings is not None
    }
    for name, settings in benchmark.export.items():
        if settings is None:
            effective.pop(name, None)
        else:
            effective[name] = _merge(effective.get(name, {}), settings)
    return effective


def export_run_settings(
    name: str, settings: dict[str, Any], *, benchmark_name: str, gym_job_id: str, bench_dir: Path
) -> dict[str, Any]:
    """The full settings of one export: the submit config's own plus what only the orchestrator knows.

    Names a run after the benchmark and the submission unless the submit config names it, and points the export
    at the benchmark's rollouts and its result file.
    """
    run_name = f"{benchmark_name}-{gym_job_id}"
    full = deepcopy(settings)
    full["rollouts_jsonl_fpath"] = str(bench_dir / "artifacts" / "rollouts.jsonl")
    full["result_json_fpath"] = str(bench_dir / "exports" / f"{name}.result.json")
    if name == "mlflow":
        full.setdefault("mlflow_run_name", run_name)
    elif name == "wandb":
        full.setdefault("wandb_name", run_name)
    elif name == "atif":
        submission = gym_job_id.removeprefix(_GYM_JOB_ID_PREFIX)
        full.setdefault("session_id", f"gym-submit-{benchmark_name}-{submission}")
        full.setdefault("output_dirpath", str(bench_dir / "exports" / "atif"))
    elif name == "evalrdl":
        meta = full.setdefault("meta", {})
        # Anything but a mapping is left for the settings model to reject with its own message.
        if isinstance(meta, dict):
            meta.setdefault("benchmark", benchmark_name)
    return full


def export_install_extras(names: list[str]) -> tuple[str, ...]:
    """The extras the job's Gym install needs for in-process exporters. Isolated ones install their own SDK."""
    from nemo_gym.exporters import EXPORTER_REGISTRY

    return tuple(EXPORTER_REGISTRY[name][2] for name in names if name in EXPORTER_REGISTRY)


def validate_exports(config: "SubmitConfig") -> None:
    """Reject, at submit time, any export that could not run: raises `ValueError` naming the benchmark and why."""
    from pydantic import ValidationError

    from nemo_gym.exporters import exporter_extras, load_run_exporter_config_model
    from nemo_gym.orchestration.api import KubernetesComputeConfig

    on_kubernetes = any(isinstance(compute, KubernetesComputeConfig) for compute in config.compute.values())
    job_id = f"{_GYM_JOB_ID_PREFIX}<submission>"
    for benchmark_name, benchmark in config.driver.benchmarks.items():
        exports = effective_exports(config, benchmark_name)
        if exports and on_kubernetes:
            raise ValueError(
                f"Benchmark '{benchmark_name}' sets `export`, which is not supported on Kubernetes compute yet."
            )
        live = set(exporter_extras(benchmark.run))
        for name, settings in exports.items():
            where = f"Benchmark '{benchmark_name}' export '{name}'"
            if name not in RUN_EXPORTER_CONFIGS:
                raise ValueError(f"{where}: unknown exporter. Available exporters: {', '.join(RUN_EXPORTER_CONFIGS)}.")
            if name in live:
                raise ValueError(
                    f"{where}: the run settings already enable the live {name} exporter, which would log the same "
                    f"run twice. Use the live exporter or the export, not both."
                )
            reserved = sorted(set(settings) & set(RESERVED_SETTINGS))
            if reserved:
                raise ValueError(f"{where}: {', '.join(reserved)} are set per benchmark by the orchestrator.")
            model = load_run_exporter_config_model(name)
            unknown = sorted(set(settings) - set(model.model_fields))
            if unknown:
                known = ", ".join(sorted(set(model.model_fields) - set(RESERVED_SETTINGS)))
                raise ValueError(f"{where}: unknown settings {', '.join(unknown)}. Known settings: {known}.")
            full = export_run_settings(
                name,
                settings,
                benchmark_name=benchmark_name,
                gym_job_id=job_id,
                bench_dir=Path(config.job.output_path) / job_id / benchmark_name,
            )
            try:
                validated = model.model_validate(full)
            except ValidationError as exc:
                problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
                raise ValueError(f"{where}: invalid settings ({problems}).") from exc
            if hasattr(validated, "is_available") and not validated.is_available:
                raise ValueError(f"{where}: the settings are incomplete for {name}; see the settings of `{name}`.")
