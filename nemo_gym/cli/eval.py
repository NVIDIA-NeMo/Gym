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
import inspect
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import Field, ValidationError
from rich.table import Table

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.artifacts.exporters import mark_run_failed
from nemo_gym.artifacts.run_files import QUALITY_SUMMARY_FILENAME as QUALITY_SUMMARY_FILENAME
from nemo_gym.artifacts.run_files import ROLLOUT_VERDICTS_FILENAME as ROLLOUT_VERDICTS_FILENAME
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
    ConfigError,
    ConfigPathNotFoundError,
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
    get_global_config_dict,
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
from nemo_gym.environment.benchmark_preparation import (
    PrepareBenchmarkConfig,
    datasets_to_prepare,
    plan_for_config,
    prepare_datasets,
)
from nemo_gym.environment.benchmarks import (
    discover_benchmarks,
)
from nemo_gym.environment.discovery import read_config_metadata
from nemo_gym.eval.eval_plan import (
    eval_plan_json_schema,
    write_eval_plan,
)
from nemo_gym.rollouts.health.types import RunHealthConfig
from nemo_gym.sandbox.compose_config import deployment_platform
from nemo_gym.sandbox.config import SANDBOX_BLOCK_VOLUME_SOURCES_KEY, validate_volume_sources
from nemo_gym.server_runtime.supervisor.server_supervisor import RunHelper
from nemo_gym.utils.path_utils import materialized_path_for


if TYPE_CHECKING:
    from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig


logger = logging.getLogger(__name__)


# NOTE: `reward_profile`, `rollout_collection`, `rollout_reverification` and `dataset_preparation` are imported lazily inside the run/aggregate/
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


@exit_cleanly_on_config_error
def prepare_benchmark() -> None:
    """CLI command: prepare benchmark data."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    prepare_benchmark_config = PrepareBenchmarkConfig.model_validate(global_config_dict)
    prepare_datasets(datasets_to_prepare(global_config_dict), prepare_benchmark_config)


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

    plan = plan_for_config(global_config_dict, plan_config, plan_config.plan_task_ids)
    write_eval_plan(plan, plan_config.plan_output_fpath)
    print(f"Wrote the plan for {plan.benchmark.name} ({len(plan.tasks)} tasks) to {plan_config.plan_output_fpath}")


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
                from nemo_gym.rollouts.rollout_health import capture_dirs_for_health

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

    from nemo_gym.artifacts.exporters.base import _artifact_version
    from nemo_gym.artifacts.exporters.metrics import flatten_aggregate_metrics, health_source_artifacts
    from nemo_gym.artifacts.run_artifacts import RunArtifacts
    from nemo_gym.orchestration.recovery import (
        COLLECT_SCRIPT,
        CollectedEvidence,
        IdentityState,
        RecoveryOutcomePolicyConfig,
        _observed_trial_operations,
        decide_from_collected,
        decision_classification,
        failure_facts,
        fault_report,
        response_health,
        select_managed_evidence,
        trial_faults,
        trial_population,
        trial_verdict,
    )
    from nemo_gym.rollouts.rollout_recovery import atomic_output_file, saved_failure_classification

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
        decision = decide_from_collected(
            collected,
            max_infra_retries=None,
            max_walltime_s=None,
            live_job=None,
            outcome_policy=RecoveryOutcomePolicyConfig(classification=saved_failure_classification(output)),
        )
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
            attributions = Counter()
            kinds = trial_faults(decision, collected, attributions=attributions)
            evaluation = fault_report(
                kinds, trials, attributions=attributions, classification=decision_classification(decision)
            ) | {"exact": False}
        if evaluation is not None:
            exact = evaluation["exact"]
            failed = evaluation["evaluation"]["count"]
            metrics.update(
                {
                    "health/trial/counts_known": int(exact),
                    "health/trial/unclassified": evaluation["unknown"]["count"],
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
        # Exclusive creation under the caller's umask; a rewrite keeps the file's existing mode.
        with atomic_output_file(path) as stream:
            stream.write((json.dumps(value, indent=2) + "\n").encode("utf-8"))

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
    from nemo_gym.rollouts import RolloutStore
    from nemo_gym.rollouts.rollout_health import _health_artifact_digests, _report_sources
    from nemo_gym.rollouts.rollout_recovery import run_lock

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
    from nemo_gym.rollouts.rollout_health import _remove_health_reports as remove_reports

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
    from nemo_gym.rollouts.rollout_health import _health_check_collected_rollouts as report

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
    from nemo_gym.rollouts.rollout_records import coverage_path_for
    from nemo_gym.rollouts.rollout_recovery import (
        IncompleteEvaluationError,
        has_only_intentional_gaps,
        unsettled_masked_note,
    )

    coverage = coverage_path_for(output.resolve())
    if coverage.exists():
        report = json.loads(coverage.read_text())
        if report.get("complete") is False:
            # Settled, not merely completed: an omission never covers a masked result that is not settled. Reports
            # that predate ``settled`` have no masked buckets, so their completed count is their settled count.
            settled = report.get("settled", report["successful"])
            if has_only_intentional_gaps(
                expected=report["expected"],
                completed=settled,
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
                f"{settled}/{report['expected']} samples completed. "
                f"Partial artifacts retained at {output}. {guidance} {unsettled_masked_note(report)}".rstrip(),
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
    from nemo_gym.rollouts import IncompleteEvaluationError, manifest_path_for

    if getattr(config, "resume_from_cache", False) and manifest_path_for(output).exists():
        _rejudge_saved_judge_failures(global_config_dict, output, config, aggregate=False)
    try:
        asyncio.run(rch.run_from_config(config))
    except IncompleteEvaluationError as rejected:
        if not config.require_complete or _terminated:
            raise
        if not _rejudge_saved_judge_failures(global_config_dict, output, config, complete_only=True):
            raise _settled_rejection(output, rejected) from None
        return
    _rejudge_saved_judge_failures(global_config_dict, output, config)


def _settled_rejection(output: Path, rejected: BaseException) -> BaseException:
    """The rejection of a strict run as its settled state decides it, after the saved-answer pass.

    Whether unfinished work remains retryable is read from the Store again, so a pass that settled rows decides the
    retry, not the state before it. The collection's own progress still counts. Unreadable evidence keeps the
    original rejection.
    """
    from nemo_gym.rollouts import IncompleteEvaluationError, RolloutStore
    from nemo_gym.rollouts.rollout_recovery import unsettled_masked_note

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
        f"{coverage['settled']}/{coverage['expected']} "
        "samples completed after the "
        f"saved-answer re-verification. Partial artifacts retained at {output}. {guidance} "
        f"{unsettled_masked_note(coverage)}".rstrip(),
        retryable=retryable,
        # exit code 1 is a retryable rejection that made no progress; any other rejection keeps progress.
        made_progress=getattr(rejected, "exit_code", 75) != 1,
        failure_kind=getattr(rejected, "failure_kind", None),
    )


def _saved_run_complete(output: Path) -> bool:
    """Whether the saved Store selection now meets ``check_completion``: unreadable evidence is not complete."""
    from nemo_gym.rollouts import RolloutStore
    from nemo_gym.rollouts.rollout_recovery import has_only_intentional_gaps

    try:
        store = RolloutStore.read(output, import_legacy=False)
        if store is None:
            return False
        coverage = store.coverage()
    except Exception:  # the original rejection stands
        return False
    completed = coverage["settled"]
    return completed == coverage["expected"] or has_only_intentional_gaps(
        expected=coverage["expected"], completed=completed, intentionally_omitted=coverage["intentionally_omitted"]
    )


def _rejudge_saved_judge_failures(
    global_config_dict: DictConfig | Mapping[str, Any],
    output: Path,
    collection_config: Any,
    *,
    aggregate: bool = True,
    complete_only: bool = False,
) -> bool:
    """Re-judge the run's saved judge failures while its servers are up, then recompute its metrics in place.

    Collection calls it at the end of a run and of each resumed run (see ``rejudge_saved_judge_failures``), and at the
    start of a resumed run, with ``aggregate`` off since its collection aggregates afterwards. The saved results to
    grade again (masked by a judge or verify-transport failure, or never graded) are re-verified in the same pass. A
    pass that cannot run, or fails, leaves the failures masked as before and never fails the collected run.
    ``complete_only`` recomputes the metrics only when the settled run is complete: a strict run that was rejected as
    incomplete publishes no score; the saved run's completeness is then read once, after the pass, and returned
    (always False without ``complete_only``).
    """
    from nemo_gym.eval.rollout_reverification import rejudge_saved_judge_failures
    from nemo_gym.rollouts.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper

    # The CLI passes the resolved OmegaConf config; callers that build the config themselves pass a plain mapping.
    if isinstance(global_config_dict, DictConfig):
        run_config = OmegaConf.to_container(global_config_dict, resolve=True)
    else:
        run_config = dict(global_config_dict)
    scored = 0
    try:
        scored = asyncio.run(rejudge_saved_judge_failures(run_config, output))
    except Exception as error:  # an optional pass after the run is saved; recovery still sees the failures
        print(f"Saved judge failures were not re-judged: {type(error).__name__}: {error}", flush=True)
    # A pass that raised after saving its grades can still have completed a strict run: settled completeness,
    # not a normal return, decides whether its metrics are rebuilt.
    complete = complete_only and _saved_run_complete(output)
    if not scored and not complete:
        return complete
    if scored:
        print(f"Re-judged saved judge failures: {scored} rollout(s) settled from their saved answers", flush=True)
    if collection_config.disable_aggregation or not aggregate:
        return complete
    if complete_only and not complete:
        return complete
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
    return complete


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
    from nemo_gym.rollouts import RolloutStore, run_health_checks
    from nemo_gym.rollouts.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper
    from nemo_gym.rollouts.rollout_health import capture_dirs_for_health, format_health_report, write_disabled_summary
    from nemo_gym.rollouts.rollout_store import raw_outcomes_are_selected

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
    from nemo_gym.eval.rollout_reverification import due_judge_failures
    from nemo_gym.rollouts import RolloutStore

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
    leaves a failed receipt naming the error and counting what it settled and withdrew before it failed: a grade it
    settled stays and counts as progress, the other saved answers stay as they were, and the error is raised as before.
    """
    from nemo_gym.eval.rollout_reverification import RolloutReverificationHelper, write_settle_receipt

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
    from nemo_gym.rollouts.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper
    from nemo_gym.telemetry.setup import (
        configure_telemetry_env,
        init_telemetry,
        shutdown_telemetry,
        telemetry_config_from_global_config,
    )

    global_config = get_global_config_dict()
    config = RolloutAggregationConfig.model_validate(global_config)
    rah = RolloutAggregationHelper()

    # Like a standalone collection, a standalone aggregation sets up its own telemetry, so the merged run's summary
    # reaches OpenTelemetry as well as an experiment tracker.
    configure_telemetry_env(telemetry_config_from_global_config(global_config))
    init_telemetry(server_name="driver", server_type="driver")
    try:
        asyncio.run(rah.run_from_config(config))
    finally:
        shutdown_telemetry()


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
    from nemo_gym.rollouts.rollout_recovery import run_lock

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
    from nemo_gym.eval.rollout_reverification import RolloutReverificationConfig

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
    from nemo_gym.eval.comparison.report import render_key_metrics_tables, summary_lines
    from nemo_gym.eval.comparison.runner import invoked_command, run_comparison
    from nemo_gym.eval.comparison.schema import ComparisonConfig

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


def eval_status(
    target: str,
    *,
    json_output: bool = False,
    job_artifact_dirs: Callable[[str], dict[str, Path] | None],
) -> None:
    """Print saved progress for a run directory or, through `job_artifact_dirs`, a Gym job ID.

    `job_artifact_dirs` is required: it maps a target that is not a run directory with status files to each benchmark's
    artifacts directory, or to None when it names no submission. Artifacts must be visible on the current filesystem. This
    command reads saved progress and does not contact a scheduler or a remote host.
    """
    from nemo_gym.rollouts.batch_status import BATCH_STATUS_FNAME

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
    files = status_files(path) if path.is_dir() else []
    exports: dict[str, dict[str, dict[str, Any]]] = {}
    if files:
        statuses = {str(file): json.loads(file.read_text()) for file in files}
        exports = {str(file): export_facts(file) for file in files}
    else:
        artifact_dirs = job_artifact_dirs(target)
        if artifact_dirs is None:
            raise FileNotFoundError(f"No {BATCH_STATUS_FNAME} or Gym job record found for {target}")
        statuses = {}
        for benchmark, directory in artifact_dirs.items():
            files = status_files(directory)
            if not files:
                statuses[benchmark] = {"status": "unavailable", "path": str(directory)}
            for file in files:
                name = benchmark if len(files) == 1 else f"{benchmark}/{file.name}"
                statuses[name] = json.loads(file.read_text())
                exports[name] = export_facts(file)
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


@exit_cleanly_on_config_error
def compare_suite() -> None:
    """Compare baseline and candidate runs across a configured benchmark and model suite."""
    from nemo_gym.environment.comparison import resolve_benchmark_reward_metric
    from nemo_gym.eval.comparison.suite.report import render_suite_summary, write_suite_reports
    from nemo_gym.eval.comparison.suite.runner import build_suite_comparison, invoked_command
    from nemo_gym.eval.comparison.suite.schema import SuiteComparisonConfig

    config_dict = get_global_config_dict()
    if not config_dict.get("suite_fpath") and not (config_dict.get("help") or config_dict.get("h")):
        raise ConfigError("gym eval compare-suite requires --suite PATH.")
    config = SuiteComparisonConfig.model_validate(config_dict)
    result = build_suite_comparison(config, invoked_command(), reward_metric_resolver=resolve_benchmark_reward_metric)
    output = Path(config.output_dirpath) if config.output_dirpath else result.suite_fpath.parent
    written = write_suite_reports(result, output, config.report_format)
    print(render_suite_summary(result))
    for path in written:
        print(f"Wrote: {path}")


__getattr__ = moved_names_getattr(__name__)
