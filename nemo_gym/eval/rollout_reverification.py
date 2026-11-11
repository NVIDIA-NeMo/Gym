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
import hashlib
import json
import math
import os
import re
import time
import uuid
import warnings
from asyncio import Future, Semaphore
from collections import Counter, defaultdict
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union

import orjson
from omegaconf import DictConfig
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.artifacts.exporters import export_metrics, export_rollouts, get_exporters
from nemo_gym.artifacts.exporters.metrics import flatten_aggregate_metrics
from nemo_gym.artifacts.rollout_rows import NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY, _rollout_for_export
from nemo_gym.client.http_client import (
    get_response_json,
    is_global_aiohttp_client_request_debug_enabled,
    raise_for_status,
)
from nemo_gym.client.server_client import ServerClient, setup_server_client
from nemo_gym.config.config_types import (
    BaseNeMoGymCLIConfig,
    ConfigError,
    RemovedRolloutOptionsMixin,
    UploadRolloutsConfigMixin,
)
from nemo_gym.config.global_config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ENVIRONMENT_SERVER_STAMP_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
    rollout_agent_label,
    rollout_run_key,
    rollout_run_labels,
)
from nemo_gym.core import VERIFIER_ERROR
from nemo_gym.core.failure_transport import FailureHeader, FailureReportRow
from nemo_gym.rollouts.rollout_collection import (
    NG_RESULT_TYPE_KEY,
    _coverage_report,
    _get_max_rollout_attempts,
    _planned_repeat_inventory,
    _rollout_request_debug_summary,
    is_omitted_rollout_row,
    is_terminal_failure,
    matches_failure_class,
)
from nemo_gym.rollouts.rollout_records import (
    REJUDGE_OPERATION_ID_KEY,
    REJUDGED_FAILURE_ID_KEY,
    RUN_ID_KEY,
    coverage_path_for,
    journal_path_for,
    logical_rollout_id,
    materialized_path_for,
    rejudge_predecessor_id,
    resolve_rollout_owner,
    resolve_rollout_path,
)
from nemo_gym.rollouts.rollout_recovery import RunManifest, manifest_path_for, run_lock
from nemo_gym.rollouts.rollout_store import RolloutStore, raw_outcomes_are_selected
from nemo_gym.server_runtime.base_resources_server import AggregateMetrics, AggregateMetricsRequest, ReverifyMode
from nemo_gym.trajectory.atif_json import strict_json_loads
from nemo_gym.trajectory.atif_reverification import (
    AtifProjectionError,
    index_materialized_inputs,
    load_atif_manifest,
    project_atif_manifest_entries,
)
from nemo_gym.utils.path_utils import (
    aggregate_metrics_path_for,
    aggregation_sources_path_for,
    failures_path_for,
    resubmit_offset_path_for,
)
from nemo_gym.utils.tqdm import tqdm


# Todo after merging branch `edobrowolska/judge_failures_v2`: replace this by importing from judge.py
JUDGE_FAILED_FAILURE_CLASS = "judge_failed"
ATIF_PROVENANCE_KEY = "_ng_atif_provenance"
_SOURCE_RUN_ID_KEY = "_ng_reverification_source_run_id"
_SOURCE_ANSWER_KEY = "_ng_reverification_source_answer"
ATIF_NO_PERSIST_FAILURE_CLASS = "kill_shaped"
_CONFIG_BOOL_ADAPTER = TypeAdapter(bool)

# Printed at the start of a `--judge-failed-only` run.
_RECOVERY_TWO_SOURCES_WARNING = (
    "WARNING: judge-failed-only recovery merges rewards from TWO sources — the successful rollouts in "
    "--rollouts were scored by the ORIGINAL run's verifier/judge, while the recovered rows are scored by the "
    "CURRENT verification config. If any crucial parameter (judge model, judge prompt/params, verifier config, "
    "etc.) changed between the original run and this recovery, the merged rewards are inconsistent. Ensure the "
    "verification config matches the original run."
)


class RolloutReverificationConfig(RemovedRolloutOptionsMixin, UploadRolloutsConfigMixin, BaseNeMoGymCLIConfig):
    input_format: Literal["gym", "atif"] = Field(
        default="gym",
        description=(
            "Input format: native Gym rollout JSONL, or a manifest for the initial "
            "Relay-exported ATIF v1.7 text-only/stateless subset."
        ),
    )
    materialized_inputs_jsonl_fpath: str = Field(
        description="The file path of the materialized inputs as output by `gym eval run`."
    )
    rollouts_jsonl_fpath: Optional[str] = Field(
        default=None, description="The file path of the rollouts to re-verify, as output by `gym eval run`."
    )
    atif_manifest_jsonl_fpath: Optional[str] = Field(
        default=None,
        description=(
            "A JSONL manifest explicitly mapping each ATIF trajectory path to a materialized Gym task and rollout."
        ),
    )
    output_jsonl_fpath: str = Field(description="The output data jsonl file path with recomputed rewards.")
    force: bool = Field(
        default=False,
        description=(
            "Re-verify even against servers whose reverify_mode is UNSUPPORTED (rewards may be "
            "incorrect); output filenames are prefixed with `unsafe_`."
        ),
    )
    disable_aggregation: bool = Field(
        default=False,
        description=(
            "Skip the post-reverification aggregate-metrics computation and file write. "
            "Used when sharding rollouts across multiple jobs that will be aggregated together "
            "afterward by `gym eval aggregate`."
        ),
    )
    num_samples_in_parallel: Optional[int] = Field(
        default=None, ge=1, description="Maximum number of samples to re-verify in parallel (omit for unbounded)."
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of examples to re-verify (omit for no limit). When combined with resume_from_cache, already-completed rows within the limit count against it, so fewer (or zero) rows may actually be re-verified.",
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "If the output file already exists, delete it and start fresh. "
            "By default, an existing output file raises an error to prevent accidental appending or overwriting. "
            "Ignored when resume_from_cache=true (the existing file is intentionally reused)."
        ),
    )
    resume_from_cache: bool = Field(
        default=False,
        description=(
            "Resume reverification from a partially-completed output file. "
            "Rows already present in the output file (or flagged terminal/maxed-out in the failures sidecar) "
            "are skipped; only the remaining rows are re-verified and appended."
        ),
    )
    judge_failed_only: bool = Field(
        default=False,
        description=(
            "Failure-recovery mode: carry the SUCCESSFUL rollouts (rollouts_jsonl_fpath) through unchanged "
            "and re-verify ONLY the run's previously judge-failed rollouts, read from the failures sidecar "
            "auto-derived next to rollouts_jsonl_fpath (`<rollouts_stem>_failures.jsonl`)."
        ),
    )
    append: bool = Field(
        default=False,
        description=(
            "Recovery-only: append the re-verified judge-failure results to an EXISTING output file instead of "
            "seeding the successful rollouts into a fresh output. Use to add the recovered rows directly onto a "
            "file that already holds the successes (e.g. point --output at the run's rollouts file). The output "
            "is opened in append mode (never cleared) and already-present keys are skipped, so re-running is "
            "idempotent. Only valid together with judge_failed_only=true, and mutually exclusive with overwrite."
        ),
    )
    settle_into_source: bool = Field(
        default=False,
        description=(
            "Recovery-only: after re-judging into the separate output, settle each re-scored judge failure in the "
            "manifest-backed source run under its run lock. The scored result keeps the failure's attempt identity and "
            "saved answer, so no model attempt is added; the failure and its debit stay. A rollout that fails to judge "
            "again stays a failure. Only valid together with judge_failed_only=true."
        ),
    )
    settle_source_snapshot_json: str | None = None
    withdraw_unresolved_admissions: bool = Field(
        default=False,
        description=(
            "Recovery-only: with settle_into_source, withdraw each admitted re-judgment that has no saved outcome "
            "before binding, so its judge failure is judged again. Only a dispatcher that ruled out a live job of the "
            "run sets it: the pass that admitted them died."
        ),
    )
    skip_spent_judge_failures: bool = Field(
        default=False,
        description=(
            "Recovery-only: with settle_into_source, leave out a judge failure whose saved failure attempts are "
            "spent, instead of refusing the whole re-judgment. That failure keeps its masked row."
        ),
    )
    settle_receipt_fpath: Optional[str] = Field(
        default=None,
        description=(
            "Recovery-only: with settle_into_source, where the judge-only recovery job writes its receipt. After the "
            "settlement the source run's aggregate metrics and quality summary are recomputed from the settled rows, "
            "and the receipt records what was selected, judged, settled and left as it was; a pass that fails leaves "
            "a failed receipt naming the error."
        ),
    )

    retry_terminal_timeouts: bool = Field(
        default=False,
        description=(
            "With resume_from_cache, retry failures-sidecar rows of class `timeout_exceeded` even when they are stamped `_ng_failure_terminal`, as rollout "
            "collection's option of the same name does. Off (default): a sidecar row is terminal iff it is "
            "stamped `_ng_failure_terminal`."
        ),
    )

    @model_validator(mode="after")
    def _validate_append(self) -> "RolloutReverificationConfig":
        if self.append and not self.judge_failed_only:
            raise ValueError("`append` is only valid together with `judge_failed_only` (pass --judge-failed-only).")
        if self.settle_into_source and not self.judge_failed_only:
            raise ValueError("`settle_into_source` is only valid together with `judge_failed_only`.")
        if self.settle_source_snapshot_json is not None:
            if not self.settle_into_source:
                raise ValueError("A source snapshot requires settle_into_source.")
            _parse_settle_source_snapshot(self.settle_source_snapshot_json)
        if self.skip_spent_judge_failures and not self.settle_into_source:
            raise ValueError("`skip_spent_judge_failures` is only valid together with `settle_into_source`.")
        if self.withdraw_unresolved_admissions and not self.settle_into_source:
            raise ValueError("`withdraw_unresolved_admissions` is only valid together with `settle_into_source`.")
        if self.settle_receipt_fpath is not None and not self.settle_into_source:
            raise ValueError("`settle_receipt_fpath` is only valid together with `settle_into_source`.")
        if self.append and self.overwrite:
            raise ValueError("`append` and `overwrite` are mutually exclusive: one appends, the other clears.")
        if self.input_format == "gym":
            if not self.rollouts_jsonl_fpath:
                raise ValueError("native Gym reverification requires `rollouts_jsonl_fpath` (pass --rollouts).")
            if self.atif_manifest_jsonl_fpath is not None:
                raise ValueError("`atif_manifest_jsonl_fpath` is only valid with input_format=atif.")
        else:
            if not self.atif_manifest_jsonl_fpath:
                raise ValueError("ATIF reverification requires `atif_manifest_jsonl_fpath` (pass --atif-manifest).")
            if self.rollouts_jsonl_fpath is not None:
                raise ValueError("`rollouts_jsonl_fpath` cannot be combined with input_format=atif.")
            if self.judge_failed_only or self.append:
                raise ValueError("ATIF reverification does not support judge-failure recovery or append mode.")
            if self.force:
                raise ValueError("ATIF reverification requires a stateless verifier; --force is not supported.")
            if self.resume_from_cache:
                raise ValueError(
                    "ATIF reverification does not support --resume until cache keys include source hashes."
                )
        return self


@dataclass
class InputRolloutPair:
    input: Dict[str, Any]  # from materialized inputs
    rollout: Dict[str, Any]  # from rollouts


@dataclass
class OutputPaths:
    output: Path
    failures: Path


@dataclass
class CacheKeysByStatus:
    successful_keys: set[tuple[int, int]]
    terminal_keys: set[tuple[int, int]]
    maxed_out_keys: set[tuple[int, int]]


# ---------------------------------------------------------------------------
# Agent-name → resources-server-name routing helpers
# Used by RolloutReverificationHelper to resolve which resources server to call
# for each rollout row, given a Hydra global config dict.
# ---------------------------------------------------------------------------


def _agent_to_rs_mapping_from_agent_blocks(
    global_config_dict: Union[Dict[str, Any], "DictConfig"],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw_name, block in global_config_dict.items():
        name = str(raw_name)
        if isinstance(block, (dict, DictConfig)) and "responses_api_agents" in block:
            impl = next(iter(block["responses_api_agents"].values()))
            rs = (impl.get("resources_server") or {}).get("name")
            if rs:
                mapping[name] = rs
    return mapping


def _agent_to_rs_mapping_from_resources_only_config(
    global_config_dict: Union[Dict[str, Any], "DictConfig"],
) -> Dict[str, str]:
    # The rollout rows still carry agent names that were never started, so fall back to the
    # single resources server for EVERY requested key — loudly, since this also absorbs typos
    # and stale agent names that would otherwise be routing errors.
    resources_server_names = [
        str(name)
        for name, block in global_config_dict.items()
        if isinstance(block, (dict, DictConfig)) and "resources_servers" in block
    ]
    if len(resources_server_names) == 1:
        only = resources_server_names[0]
        warnings.warn(
            f"reverify: config has no agent blocks; routing EVERY rollout agent name to the only "
            f"resources server {only!r}. Mismatched or stale agent names cannot be detected in "
            "this mode.",
            stacklevel=2,
        )
        return defaultdict(lambda: only)  # any key → the one resources server instance
    if not resources_server_names:
        raise ConfigError("reverify: no resources server found in the config.")
    raise ConfigError(
        f"reverify: multiple resources servers {resources_server_names} and no agent blocks to "
        "route by. Use a config with agent blocks."
    )


def _rs_for_row(
    row: Dict[str, Any],
    agent_to_rs: Dict[str, str],
    global_config_dict: Union[Dict[str, Any], "DictConfig"],
) -> str:
    """The resources server that verifies this row.

    A task_source naming a resources server is authoritative (it is the declaring instance the
    dataset was stamped with — no agent indirection needed). Otherwise fall back to the rollout's
    agent_ref via the config's agent->rs edges.
    """
    ts = row.get(TASK_SOURCE_KEY_NAME)
    if ts is not None:
        block = global_config_dict.get(ts)
        if isinstance(block, (dict, DictConfig)) and "resources_servers" in block:
            return str(ts)
    server = row.get(ENVIRONMENT_SERVER_STAMP_KEY_NAME)
    server_block = global_config_dict.get(server) if isinstance(server, str) else None
    if isinstance(server_block, (dict, DictConfig)):
        for environment_server in (server_block.get("environment_servers") or {}).values():
            resources_ref = environment_server.get("resources_server") if environment_server else None
            if resources_ref and resources_ref.get("name"):
                return str(resources_ref["name"])
    agent_name = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
    try:
        return agent_to_rs[agent_name]
    except KeyError:
        raise ConfigError(
            f"reverify: cannot find a resources server for row (agent_ref.name={agent_name!r}, "
            f"task_source={ts!r}). Known agents: {sorted(agent_to_rs)}."
        ) from None


def _build_agent_to_resources_server_mapping(
    global_config_dict: Union[Dict[str, Any], "DictConfig"],
) -> Dict[str, str]:
    mapping = _agent_to_rs_mapping_from_agent_blocks(global_config_dict)
    if mapping:
        return mapping
    return _agent_to_rs_mapping_from_resources_only_config(global_config_dict)


def _selected_atif_resources_server_routes(
    global_config_dict: Union[Dict[str, Any], "DictConfig"],
    payloads: List[Dict[str, Any]],
) -> List[str]:
    """Resolve each ATIF row through the same routing policy used by ``/verify``."""

    configured_mapping = _build_agent_to_resources_server_mapping(global_config_dict)
    return [_rs_for_row(row, configured_mapping, global_config_dict) for row in payloads]


def _resources_server_exposes_tools_over_mcp(
    global_config_dict: Union[Dict[str, Any], "DictConfig"],
    resources_server_name: str,
) -> bool:
    """Read the selected resources server's MCP exposure flag from Gym config."""

    block = global_config_dict.get(resources_server_name)
    if not isinstance(block, (dict, DictConfig)):
        raise ConfigError(f"reverify: resources server {resources_server_name!r} is missing from the config.")
    implementations = block.get("resources_servers")
    if not isinstance(implementations, (dict, DictConfig)) or len(implementations) != 1:
        raise ConfigError(
            f"reverify: resources server {resources_server_name!r} must contain exactly one resources_servers entry."
        )
    implementation = next(iter(implementations.values()))
    if not isinstance(implementation, (dict, DictConfig)):
        raise ConfigError(f"reverify: resources server {resources_server_name!r} has an invalid config entry.")
    exposes_tools = implementation.get("expose_tools_over_mcp", False)
    try:
        return _CONFIG_BOOL_ADAPTER.validate_python(exposes_tools)
    except ValidationError as exc:
        raise ConfigError(
            f"reverify: resources server {resources_server_name!r} has an invalid expose_tools_over_mcp value."
        ) from exc


def _response_has_function_calls(row: Dict[str, Any]) -> bool:
    response = row.get("response")
    output = response.get("output") if isinstance(response, dict) else None
    return isinstance(output, list) and any(
        isinstance(item, dict) and item.get("type") == "function_call" for item in output
    )


# ---------------------------------------------------------------------------
# Function used to summarize the debug information for a failed verification
# ---------------------------------------------------------------------------
def _rollout_verify_debug_summary(row: Dict[str, Any], resources_server_name: str) -> Dict[str, Any]:
    summary = {
        TASK_INDEX_KEY_NAME: row.get(TASK_INDEX_KEY_NAME),
        ROLLOUT_INDEX_KEY_NAME: row.get(ROLLOUT_INDEX_KEY_NAME),
        "agent_name": rollout_agent_label(row),
        "resources_server_name": resources_server_name,
    }
    return {k: v for k, v in summary.items() if v is not None}


# ---------------------------------------------------------------------------
# Functions used to deal with the cache - partially completed output file and
# the failures sidecar file
# ---------------------------------------------------------------------------


def _parse_output_line(line: bytes) -> Dict[str, Any]:
    result_str = line.strip()
    if not result_str:
        return {}
    return orjson.loads(result_str)


def _parse_output_line_key(line: bytes) -> tuple[int, int] | None:
    result = _parse_output_line(line)
    task_idx = result.get(TASK_INDEX_KEY_NAME)
    rollout_idx = result.get(ROLLOUT_INDEX_KEY_NAME)
    if task_idx is None or rollout_idx is None:
        return None
    return task_idx, rollout_idx


def _load_cache_keys_by_status(
    output_fpaths: OutputPaths, *, retry_terminal_timeouts: bool = False
) -> CacheKeysByStatus:
    if not (output_fpaths.output.exists() or output_fpaths.failures.exists()):
        print("Skipping resume_from_cache because cache paths don't exist!")
        return CacheKeysByStatus(
            successful_keys=set(),
            terminal_keys=set(),
            maxed_out_keys=set(),
        )
    # Successes (and any legacy '-failed' rows written by pre-fix Gym
    # builds) live in the main jsonl. They short-circuit dispatch.
    successful_keys: set[tuple[int, int]] = set()
    if output_fpaths.output.exists():
        with output_fpaths.output.open("rb") as f:
            successful_keys = {key for line in f if (key := _parse_output_line_key(line)) is not None}

    # Sidecar: one row per non-kill_shaped failure attempt. Count attempts
    # per key + flag terminal rows so chain-hop 2 retries the right ones.
    attempts_by_key: Counter = Counter()
    terminal_keys: set = set()
    if output_fpaths.failures.exists():
        with output_fpaths.failures.open("rb") as f:
            for line in f:
                fr = _parse_output_line(line)
                if not fr:
                    continue
                if TASK_INDEX_KEY_NAME not in fr or ROLLOUT_INDEX_KEY_NAME not in fr:
                    continue
                k = (fr[TASK_INDEX_KEY_NAME], fr[ROLLOUT_INDEX_KEY_NAME])
                attempts_by_key[k] += 1
                if is_terminal_failure(fr, retry_terminal_timeouts=retry_terminal_timeouts):
                    terminal_keys.add(k)

    max_attempts = _get_max_rollout_attempts()
    maxed_out_keys = {k for k, n in attempts_by_key.items() if n >= max_attempts}
    return CacheKeysByStatus(
        successful_keys=successful_keys,
        terminal_keys=terminal_keys,
        maxed_out_keys=maxed_out_keys,
    )


def _drop_cache_from_payloads(payloads: List[Dict], cache: CacheKeysByStatus) -> Iterator[Dict]:
    for payload in payloads:
        key = (payload[TASK_INDEX_KEY_NAME], payload[ROLLOUT_INDEX_KEY_NAME])
        if key in cache.successful_keys:
            continue
        if key in cache.terminal_keys:
            continue
        if key in cache.maxed_out_keys:
            continue
        yield payload


def summarize_cache_usage(cache: CacheKeysByStatus, all_payloads: List[Dict], filtered_payloads: List[Dict]) -> None:
    print(
        f"""Resumed from cache. Found:
- {len(all_payloads)} total rows to be re-verified
- {len(cache.successful_keys)} rows already done (in main jsonl)
- {len(cache.terminal_keys)} sidecar-terminal → not retried
- {len(cache.maxed_out_keys)} hit max_attempts → not retried
- {len(filtered_payloads)} rows that still need to be run"""
    )


# ---------------------------------------------------------------------------
# Judge-failure recovery (`--judge-failed-only`): re-verify only the previously
# judge-failed rollouts from the failures sidecar and merge them back with the
# already-successful rollouts (seeded into the output) so the aggregate metrics
# cover the union — identical to a clean run, with no inference re-run.
# ---------------------------------------------------------------------------


def _is_judge_failure(row: Dict[str, Any]) -> bool:
    """Whether a failures-sidecar row is a judge failure, the class `--judge-failed-only` recovers.

    A ``record_omitted_rollouts`` row records a rollout that produced no result, so there is nothing to judge.
    """
    return matches_failure_class(row, {JUDGE_FAILED_FAILURE_CLASS}) and not is_omitted_rollout_row(row)


def _recovery_rollout_predicate(
    skip_keys: Optional[set[tuple[Any, Any]]] = None,
) -> Callable[[Dict[str, Any]], bool]:
    """Build the row filter for `--judge-failed-only` recovery.

    Keeps only judge-class failures, skips any whose `(task, rollout)` key is in `skip_keys` (the keys
    already present in the output: seeded successes + rows a prior recovery already appended) so a
    key-in-both is never re-verified/duplicated and re-runs are idempotent — this dedup is done here,
    independent of the resume/cache machinery — and dedups on the key so a sidecar with multiple failure
    attempts for one key re-verifies it exactly once.
    """
    skip_keys = skip_keys or set()
    seen: set[tuple[Any, Any]] = set()

    def predicate(row: Dict[str, Any]) -> bool:
        if not _is_judge_failure(row):
            return False
        key = (row.get(TASK_INDEX_KEY_NAME), row.get(ROLLOUT_INDEX_KEY_NAME))
        if key in skip_keys or key in seen:
            return False
        seen.add(key)
        return True

    return predicate


def _reject_multistage_recovery_source(rollouts_jsonl_fpath: Path) -> None:
    """Refuse `--judge-failed-only` on multi-stage rows before any file is written.

    Multi-stage rows carry ``stage_index``. Recovering them outside the multi-stage collection
    would lose stage identity and the adaptive reference set. Recovery reads the failures sidecar.
    """
    fpath = failures_path_for(rollouts_jsonl_fpath)
    if not fpath.exists():
        return
    with fpath.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            row = orjson.loads(line)
            # A rollout that produced no result has nothing to recover, so its row cannot lose stage identity.
            if "stage_index" in row and not is_omitted_rollout_row(row):
                raise ConfigError(
                    "--judge-failed-only does not support multi-stage rows; resume the multi-stage rollout "
                    "collection so stage identity and adaptive references are preserved"
                )


def _project_reverification_row(row: Dict) -> Dict:
    """Retain source provenance without claiming ownership of its recovery history."""
    result = dict(row)
    if RUN_ID_KEY in result:
        result[_SOURCE_RUN_ID_KEY] = result.pop(RUN_ID_KEY)
    return result


def _seed_output_with_successes(
    successes_fpath: Path, output_fpath: Path, exclude: frozenset[tuple[int, int]] = frozenset()
) -> set[tuple[int, int]]:
    """Copy the already-successful rollout rows into the output file so the final aggregate covers
    successes + recovered rows, and return the set of (task, rollout) keys ALREADY PRESENT in the output
    afterward. Rows whose key is in ``exclude`` (saved results that are re-verified) are not seeded."""
    present: set[tuple[int, int]] = set()
    if output_fpath.exists():
        with output_fpath.open("rb") as f:
            present = {key for line in f if (key := _parse_output_line_key(line)) is not None}
    seeded = 0
    source_store = RolloutStore.read(successes_fpath, import_legacy=False)
    with successes_fpath.open("rb") as src, output_fpath.open("ab") as dst:
        selected = (
            (
                orjson.dumps(_project_reverification_row(record.read())) + b"\n"
                for record in source_store.selected_records("success").values()
            )
            if source_store is not None
            else src
        )
        for line in selected:
            if not line.strip():
                continue
            key = _parse_output_line_key(line)
            if key is not None:
                if key in present or key in exclude:
                    continue
                present.add(key)
            if seeded == 0:
                # Seeded rows are not the merged ones an aggregation's shard list names.
                aggregation_sources_path_for(output_fpath).unlink(missing_ok=True)
            dst.write(line if line.endswith(b"\n") else line + b"\n")
            seeded += 1
    print(f"Recovery mode: seeded {seeded} successful rollout(s) into {output_fpath}")
    return present


# ---------------------------------------------------------------------------
# Functions used by the main RolloutReverificationHelper in the reverification process:
# Yielding InputRolloutPair objects to be re-verified
# Preparing the payloads from them (by skipping rows that are already in the cache and some formatting)
# And running the verification requests in parallel
# ---------------------------------------------------------------------------


def _yield_inputs_and_rollouts_paired(
    materialized_inputs_jsonl_fpath: Path,
    rollouts_jsonl_fpath: Path,
    limit: Optional[int] = None,
    rollout_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    *,
    selected_rollouts: Optional[Iterable[Dict]] = None,
) -> "Iterator[InputRolloutPair]":
    inputs_by_key = {}
    with open(materialized_inputs_jsonl_fpath) as m_f:
        for line in m_f:
            r = orjson.loads(line)
            inputs_by_key[(r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])] = r
    # `limit` bounds the number of pairs actually YIELDED (post-predicate)
    n_yielded = 0
    with nullcontext(selected_rollouts) if selected_rollouts is not None else open(rollouts_jsonl_fpath) as r_f:
        rows = iter(tqdm(r_f, desc="Reading rollouts"))
        while limit is None or n_yielded < limit:
            try:
                line = next(rows)
            except StopIteration:
                break
            rollout_row = line if selected_rollouts is not None else orjson.loads(line)
            if rollout_predicate is not None and not rollout_predicate(rollout_row):
                continue
            input_row = inputs_by_key.get((rollout_row[TASK_INDEX_KEY_NAME], rollout_row[ROLLOUT_INDEX_KEY_NAME]))
            if input_row is None:
                raise ConfigError(f"No matching materialized input row found for rollout row {rollout_row}")
            yield InputRolloutPair(input=input_row, rollout=rollout_row)
            n_yielded += 1


def _rollout_response(rollout: Dict[str, Any]) -> Any:
    """Return the response a rollout's verifier scored, or explain which result type lacks one."""
    # Native judge failures keep the generated answer outside the failure contract.
    # Diagnostic _ng_failure_response is deliberately not an eligible saved answer.
    if (
        "response" not in rollout
        and _is_judge_failure(rollout)
        and isinstance(rollout.get("_ng_failure_partial_response"), dict)
    ):
        return rollout["_ng_failure_partial_response"]
    if "response" not in rollout:
        result_type = rollout.get(NG_RESULT_TYPE_KEY, "unknown")
        raise ConfigError(
            f"reverify: rollout (task {rollout.get(TASK_INDEX_KEY_NAME)}, rollout {rollout.get(ROLLOUT_INDEX_KEY_NAME)}) "
            f"of result type {result_type!r} has no `response`, which reverification needs"
        )
    return rollout["response"]


# Resources server implementations whose /verify grades a saved workspace patch given back to it
# (``saved_model_patch``) on a fresh sandbox. Every other server reads the patch from the agent session at verify
# time and would grade an empty one, so a saved patch is never sent to it.
_SAVED_PATCH_REPLAY_SERVERS = frozenset({"deepswe"})


def check_saved_patch_replay(
    row: Dict[str, Any], resources_server_name: str, global_config_dict: Union[Dict[str, Any], "DictConfig"]
) -> None:
    """Refuse to send a saved patch (``saved_model_patch``) to a resources server that does not replay one."""
    if "saved_model_patch" not in row:
        return
    block = global_config_dict.get(resources_server_name)
    implementations = (
        list((block.get("resources_servers") or {}).keys()) if isinstance(block, (dict, DictConfig)) else []
    )
    if len(implementations) != 1 or implementations[0] not in _SAVED_PATCH_REPLAY_SERVERS:
        raise ConfigError(
            f"reverify: saved model_patch replay is unsupported by resources server {resources_server_name!r} "
            f"({implementations}); retain the original patch and failure"
        )


# Collection's stamps naming the Environment Server and its result type: kept on the re-verified row, never sent
# to /verify.
_IDENTITY_KEYS = (ENVIRONMENT_SERVER_STAMP_KEY_NAME, NG_RESULT_TYPE_KEY)


def _build_verify_payload(pair: InputRolloutPair) -> Dict:
    saved_patch_failure = _is_judge_failure(pair.rollout) or (
        pair.rollout.get("failure_kind") == VERIFIER_ERROR and pair.rollout.get("evaluation_completed") is False
    )
    saved_patch = pair.rollout.get("model_patch")
    replay_patch = False
    if saved_patch_failure and isinstance(saved_patch, str):
        # Only the patch the verifier graded is graded again, never another patch (the task's reference patch, say)
        # and never a new one from the agent: it must hash to the saved digest, and it goes to /verify as itself.
        digest = pair.rollout.get("model_patch_sha256")
        if (
            _is_judge_failure(pair.rollout)
            or not isinstance(digest, str)
            or (hashlib.sha256(saved_patch.encode("utf-8")).hexdigest() != digest)
        ):
            raise ConfigError(
                "reverify: saved model_patch replay needs the verifier's own patch with its saved sha256; retain the "
                "original patch and failure"
            )
        replay_patch = True
    response = _rollout_response(pair.rollout)
    if "opencode_evidence" in response or "opencode_tool_reward" in pair.rollout:
        raise ConfigError(
            "reverify: OpenCode environment results require replaying their final-answer scoring view and "
            "answer-dependent tool rewards. Direct resources-server reverification does not support that "
            "contract; refusing to replace the saved reward with a different scoring calculation."
        )
    task_input = pair.input.get("task_input")
    if not isinstance(task_input, dict):
        payload = pair.input | {"response": response}
    else:
        # Accept generic flat task input and the historical single-agent task_data container.
        row_keys = {k: v for k, v in pair.input.items() if k not in ("task_id", "task_input")}
        fields = dict(task_input)
        task_data = fields.pop("task_data", {})
        if not isinstance(task_data, dict):
            raise ConfigError("reverify: task_input.task_data must be an object")
        for key in fields.keys() & task_data.keys():
            if fields[key] != task_data[key]:
                raise ConfigError(f"reverify: conflicting task field {key!r} inside and outside task_data")
        payload = row_keys | task_data | fields | {"response": response}
    # File-backed verifiers need the artifact path produced by the rollout, and
    # adaptive comparison needs the exact reference subset used for that row.
    # Preserve only verifier inputs, not rewards or failure bookkeeping. SciCode's /verify grades the accumulated
    # per-sub-step solutions an agent sent, which a masked row keeps (the response holds only the last step), with the
    # usage accounting that travels through the verifier.
    for key in ("deliverables_dir", "reference_ids", "solutions", "token_usage_version", "step_usage"):
        if key in pair.rollout:
            payload[key] = pair.rollout[key]
    if replay_patch:
        payload["saved_model_patch"] = saved_patch
        payload["saved_model_patch_sha256"] = pair.rollout["model_patch_sha256"]
    # The Environment Server that ran the rollout routes the row and groups its result as in the run; the
    # materialized input never carries it. It is not sent to /verify (see _IDENTITY_KEYS).
    for key in (*_IDENTITY_KEYS, _SOURCE_RUN_ID_KEY, _SOURCE_ANSWER_KEY):
        if key in pair.rollout:
            payload[key] = pair.rollout[key]
    return payload


def _rescoring_source_answer(row: Dict, run_id: str) -> dict[str, str | int]:
    """Bind a re-score to its saved answer, independently of changes to the source grade."""
    return {
        "run_id": run_id,
        "rollout_id": logical_rollout_id(row),
        "attempt": row.get(ATTEMPT_INDEX_KEY_NAME, 0),
        "task_index": row[TASK_INDEX_KEY_NAME],
        "rollout_index": row[ROLLOUT_INDEX_KEY_NAME],
        "answer_sha256": hashlib.sha256(orjson.dumps(_rollout_response(row), option=orjson.OPT_SORT_KEYS)).hexdigest(),
    }


def _selected_rescoring_rows(store: RolloutStore) -> Iterator[Dict]:
    """Read answers on demand so a bounded re-score does not materialize the whole run."""
    for record in store.selected_records("success").values():
        row = record.read()
        yield row | {
            _SOURCE_RUN_ID_KEY: store.manifest.run_id,
            _SOURCE_ANSWER_KEY: _rescoring_source_answer(row, store.manifest.run_id),
        }


def _validate_rescoring_cache(store: RolloutStore, output_fpaths: OutputPaths) -> None:
    """Refuse stale successes and failure-budget entries before cache filtering or any dispatch."""
    selected = store.selected_records("success")
    # Repeated cached failures share an answer: retain its small binding, never its trajectory.
    bindings: dict[tuple[str, int], dict[str, str | int]] = {}
    for path in (output_fpaths.output, output_fpaths.failures):
        if not path.exists():
            continue
        with path.open("rb") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                message = (
                    f"Cached re-scoring row in {path}:{line_number} is not bound to the currently selected "
                    "source answer. Choose a fresh output path; saved artifacts were not changed."
                )
                try:
                    cached = orjson.loads(line)
                except orjson.JSONDecodeError as error:
                    raise ConfigError(message) from error
                binding = cached.get(_SOURCE_ANSWER_KEY) if isinstance(cached, dict) else None
                if not isinstance(binding, dict):
                    raise ConfigError(message)
                identity, attempt = binding.get("rollout_id"), binding.get("attempt")
                if not isinstance(identity, str) or type(attempt) is not int:
                    raise ConfigError(message)
                key = (identity, attempt)
                record = selected.get(key)
                if record is None:
                    raise ConfigError(message)
                if key not in bindings:
                    bindings[key] = _rescoring_source_answer(record.read(), store.manifest.run_id)
                expected = bindings[key]
                if (
                    binding != expected
                    or cached.get(_SOURCE_RUN_ID_KEY) != store.manifest.run_id
                    or cached.get(TASK_INDEX_KEY_NAME) != expected["task_index"]
                    or cached.get(ROLLOUT_INDEX_KEY_NAME) != expected["rollout_index"]
                ):
                    raise ConfigError(message)


def _prepare_payloads(
    materialized_inputs_jsonl_fpath: Path,
    rollouts_jsonl_fpath: Path,
    output_fpaths: OutputPaths,
    resume_from_cache: bool,
    limit: Optional[int] = None,
    rollout_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    retry_terminal_timeouts: bool = False,
    *,
    selected_rollouts: Optional[Iterable[Dict]] = None,
) -> List[Dict]:
    all_payloads = [
        _build_verify_payload(pair)
        for pair in _yield_inputs_and_rollouts_paired(
            materialized_inputs_jsonl_fpath,
            rollouts_jsonl_fpath,
            limit=limit,
            rollout_predicate=rollout_predicate,
            **({"selected_rollouts": selected_rollouts} if selected_rollouts is not None else {}),
        )
    ]
    if resume_from_cache:
        cache = _load_cache_keys_by_status(output_fpaths, retry_terminal_timeouts=retry_terminal_timeouts)
        payloads = list(_drop_cache_from_payloads(all_payloads, cache))
        summarize_cache_usage(cache, all_payloads, payloads)
        prepared_payloads = payloads
    else:
        prepared_payloads = all_payloads
    if not prepared_payloads:
        print("WARNING: Nothing to be re-verified.")
    return prepared_payloads


def _prepare_atif_payloads(
    materialized_inputs_jsonl_fpath: Path,
    atif_manifest_jsonl_fpath: Path,
    limit: Optional[int] = None,
) -> List[Dict]:
    """Build verifier payloads from an explicit ATIF-to-materialized-task manifest."""

    materialized_rows: list[Dict[str, Any]] = []
    try:
        materialized_inputs = materialized_inputs_jsonl_fpath.open("rb")
    except OSError as exc:
        raise AtifProjectionError(
            f"could not read materialized inputs {materialized_inputs_jsonl_fpath}: {exc}"
        ) from exc
    with materialized_inputs:
        for line_number, line in enumerate(materialized_inputs, start=1):
            if not line.strip():
                continue
            try:
                row = strict_json_loads(line)
            except ValueError as exc:
                raise AtifProjectionError(
                    f"invalid materialized input row {line_number} in {materialized_inputs_jsonl_fpath}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise AtifProjectionError(
                    f"materialized input row {line_number} in {materialized_inputs_jsonl_fpath} is not an object"
                )
            materialized_rows.append(row)

    entries = load_atif_manifest(atif_manifest_jsonl_fpath)
    if limit is not None:
        entries = entries[:limit]
    projected = project_atif_manifest_entries(
        entries,
        index_materialized_inputs(materialized_rows),
        manifest_directory=atif_manifest_jsonl_fpath.parent,
    )
    return [
        item.payload
        | {
            ATIF_PROVENANCE_KEY: {
                "trajectory_id": item.trajectory_id,
                "session_id": item.session_id,
                "source_sha256": item.source_sha256,
                "schema_version": item.schema_version,
                "projection_status": item.projection_status,
            }
        }
        for item in projected
    ]


def _run_verification_payloads(
    payloads: List[Dict],
    semaphore: Semaphore | nullcontext[None] | None = None,
) -> Iterator[Future]:  # pragma: no cover
    semaphore = semaphore or nullcontext[None]()
    server_client = setup_server_client()
    agent_to_rs = _build_agent_to_resources_server_mapping(server_client.global_config_dict)

    async def _post_subroutine(row: Dict) -> Tuple[Dict, Dict]:
        async with semaphore:
            rs_name = _rs_for_row(row, agent_to_rs, server_client.global_config_dict)
            check_saved_patch_replay(row, rs_name, server_client.global_config_dict)
            request_row = {
                key: value
                for key, value in row.items()
                if key not in (ATIF_PROVENANCE_KEY, _SOURCE_RUN_ID_KEY, _SOURCE_ANSWER_KEY)
                and key not in _IDENTITY_KEYS
            }
            # A /verify request that may have been delivered is never resent: the verifier may already have graded it
            # (and called its judge). A refused connection is still retried. A lost outcome fails this run; with
            # settle_into_source the outcomes already answered are settled and its admission is withdrawn, so a
            # later pass grades it again (see _settle_what_a_failed_pass_judged).
            res = await server_client.post(
                server_name=rs_name, url_path="/verify", json=request_row, _resend_after_disconnect=False
            )
            try:
                await raise_for_status(
                    res
                )  # this code works similarly to the rollout collection code, so *_failures.jsonl is empty now
            # IMO we need another task to unify dealing with failed cases (and writing to the *_failures.jsonl file if needed)
            except Exception:
                if is_global_aiohttp_client_request_debug_enabled():
                    print(
                        "[rollout_reverification] /verify failed "
                        f"status={getattr(res, 'status', None)} "
                        f"row={json.dumps(_rollout_verify_debug_summary(row, rs_name), sort_keys=True)}",
                        flush=True,
                    )
                raise
            result = await get_response_json(res)
            # As collection saves a dispatch reply's report: failures this /verify observed, such as a judge attempt
            # a retry recovered from, are kept beside the result; a failed result keeps its own failure_id.
            report = FailureHeader.decode(getattr(res, "headers", None))
            if report is not None:
                result = FailureReportRow.merged_into(result, report)
            return row, result

    return tqdm.as_completed(
        map(_post_subroutine, payloads),
        desc="Collecting reverification results",
        miniters=10,
        total=len(payloads),
        maxinterval=60,
    )


# ---------------------------------------------------------------------------
# Reverify mode helpers
# Used by RolloutReverificationHelper to decide if reverification is safe
# ---------------------------------------------------------------------------


def _get_rs_names(agent_to_rs: Dict[str, str]) -> List[str]:
    # Resources-only configs return a keyless defaultdict whose .values() is empty until a key is
    # accessed; materialise the single RS explicitly so it isn't silently skipped by the safety check.
    rs_names = set(agent_to_rs.values())
    if not rs_names and isinstance(agent_to_rs, defaultdict) and agent_to_rs.default_factory is not None:
        rs_names = {agent_to_rs.default_factory()}
    return list(rs_names)


async def _check_reverify_mode(server_client: "ServerClient", agent_to_rs: Dict[str, str]) -> List[str]:
    """Query GET /reverify_mode on each unique resource server referenced by agent_to_rs.

    Returns a sorted list of RS names that reported ReverifyMode.UNSUPPORTED or ReverifyMode.UNKNOWN.
    """
    unsupported: List[str] = []
    for rs_name in _get_rs_names(agent_to_rs):
        res = await server_client.get(server_name=rs_name, url_path="/reverify_mode")
        await raise_for_status(res)
        mode = ReverifyMode(await get_response_json(res))
        if mode in (ReverifyMode.UNSUPPORTED, ReverifyMode.UNKNOWN):
            unsupported.append(rs_name)
    return sorted(unsupported)


async def _guard_reverify_mode(config: RolloutReverificationConfig) -> Optional[str]:
    """Check reverify_mode for every RS in the config before reverification starts.

    Returns a warning string when the user runs full re-verification (not judge-failed-only) and force=True and at least one RS is UNSUPPORTED or UNKNOWN (caller must
    print it and apply the unsafe_ output prefix).
    Raises ConfigError when force=False and at least one RS is UNSUPPORTED or UNKNOWN.
    Returns None when all RS are STATELESS.
    """
    if config.judge_failed_only:
        return None
    server_client = setup_server_client()
    agent_to_rs = _build_agent_to_resources_server_mapping(server_client.global_config_dict)
    non_stateless_rs = await _check_reverify_mode(server_client, agent_to_rs)
    if not non_stateless_rs:
        return None
    if not config.force:
        raise ConfigError(
            f"Resource server(s) {non_stateless_rs} have reverify_mode=UNSUPPORTED or UNKNOWN. "
            "Rewards computed by reverification may be incorrect. "
            "Pass ++force=true to override (output will be prefixed with 'unsafe_')."
        )
    return (
        f"WARNING: resource server(s) {non_stateless_rs} have reverify_mode=UNSUPPORTED or UNKNOWN. "
        "Rewards computed by reverification may be incorrect. "
        "Output is prefixed with 'unsafe_'."
    )


async def _guard_atif_preflight(payloads: List[Dict[str, Any]]) -> None:
    """Validate selected routes before ATIF reverification touches output paths."""

    server_client = setup_server_client()
    selected_routes = _selected_atif_resources_server_routes(
        server_client.global_config_dict,
        payloads,
    )
    selected_mapping = {str(index): route for index, route in enumerate(selected_routes)}
    non_stateless_rs = await _check_reverify_mode(server_client, selected_mapping)
    if non_stateless_rs:
        raise ConfigError(
            f"ATIF reverification requires stateless verifiers; resource server(s) {non_stateless_rs} "
            "reported reverify_mode=UNSUPPORTED or UNKNOWN."
        )
    for row, resources_server_name in zip(payloads, selected_routes, strict=True):
        if _response_has_function_calls(row) and _resources_server_exposes_tools_over_mcp(
            server_client.global_config_dict, resources_server_name
        ):
            raise AtifProjectionError(
                "Relay ATIF tool calls cannot be reverified against MCP-exposed resources server "
                f"{resources_server_name!r}: ATIF proves call/result correlation but does not carry Gym's "
                "canonical (server_name, tool_name) provenance. Use a non-MCP stateless verifier or a "
                "text-only trajectory."
            )


# ---------------------------------------------------------------------------
# Function used to compute the aggregate metrics after the reverification process
# Very similar to the rollout collection code, but we need to send the request to
# resources servers instead of the agent server, since the second one might not be started
# ---------------------------------------------------------------------------
async def _call_aggregate_metrics(
    results: List[Dict],
    rows: List[Dict],
    output_fpath: Path,
    *,
    materialized_fpaths: Optional[List[Path]] = None,
) -> Optional[Path]:
    """Call /aggregate_metrics on each resource server after rollouts complete.

    Writes a single _aggregate_metrics.json with one entry per agent (same shape
    as the old _agent_metrics.json). Returns the file path.
    """
    if not results:
        return None

    server_client = setup_server_client()
    agent_to_rs = _build_agent_to_resources_server_mapping(server_client.global_config_dict)
    # Group results per (agent, resources server), routing each row with the SAME resolver used
    # for /verify (_rs_for_row: task_source authoritative, agent_ref via config edges as the
    # fallback). Routing aggregation independently by the agent's configured server allowed a
    # remapped row to be verified by one server and aggregated by another.
    agent_results: Dict[Tuple[str, str], List[Dict]] = {}
    labels = rollout_run_labels(rows)
    inventory = await asyncio.to_thread(
        _planned_repeat_inventory,
        materialized_fpaths or [],
        lambda row: (rollout_run_key(row), _rs_for_row(row, agent_to_rs, server_client.global_config_dict)),
    )
    for row, result in zip(rows, results):
        key = rollout_run_key(row)
        if not key:
            continue
        agent_name = labels[key]
        rs_name = _rs_for_row(row, agent_to_rs, server_client.global_config_dict)
        agent_results.setdefault((agent_name, rs_name), []).append(result)

    async def _fetch_agent_metrics(agent_name: str, rs_name: str, agent_result_list: List[Dict]) -> Dict:
        # Preserve usage and truncation metadata: policy token-limit failures score zero, not as judge failures.
        stripped = []
        for r in agent_result_list:
            entry = {
                k: v for k, v in r.items() if k not in ("response", "responses_create_params", ATIF_PROVENANCE_KEY)
            }
            response = r.get("response") or {}
            response_metadata = {
                key: response[key] for key in ("usage", "incomplete_details") if response.get(key) is not None
            }
            context_marker = (response.get("metadata") or {}).get("nemo_gym_context_window_exceeded")
            if context_marker is not None:
                response_metadata["metadata"] = {"nemo_gym_context_window_exceeded": context_marker}
            if response_metadata:
                entry["response"] = response_metadata
            stripped.append(entry)

        run_keys = {rollout_run_key(row) for row in rows if labels.get(rollout_run_key(row)) == agent_name}
        expected = None
        if inventory is not None:
            expected = {}
            for run_key in run_keys:
                for task, indices in inventory.get((run_key, rs_name), {}).items():
                    expected[task] = sorted(set(expected.get(task, [])) | set(indices))
        agg_request = AggregateMetricsRequest(verify_responses=stripped, expected_rollout_indices=expected)
        agg_response = await server_client.post(
            server_name=rs_name,
            url_path="/aggregate_metrics",
            json=agg_request,
        )

        await raise_for_status(agg_response)
        agg_result = AggregateMetrics.model_validate(await get_response_json(agg_response))

        agent_entry = {
            AGENT_REF_KEY_NAME: {"name": agent_name},
            "agent_metrics": agg_result.agent_metrics,
            "key_metrics": agg_result.key_metrics,
            "group_level_metrics": agg_result.group_level_metrics,
            "repeat_level_metrics": agg_result.repeat_level_metrics,
        }
        return agent_entry

    all_agent_metrics: List[Dict] = []
    tasks = [
        _fetch_agent_metrics(name, rs_name, results_list) for (name, rs_name), results_list in agent_results.items()
    ]
    for coro in asyncio.as_completed(tasks):
        agent_entry = await coro
        all_agent_metrics.append(agent_entry)

        agent_name = agent_entry[AGENT_REF_KEY_NAME]["name"]
        key_metrics = agent_entry.get("key_metrics", {})
        print(f"\nKey metrics for {agent_name}:\n" + json.dumps(key_metrics, indent=4))

    metrics_to_log = flatten_aggregate_metrics(all_agent_metrics)
    export_metrics(metrics_to_log)

    # Write single file with all agents
    metrics_fpath = aggregate_metrics_path_for(output_fpath)
    metrics_fpath.write_bytes(orjson.dumps(all_agent_metrics, option=orjson.OPT_INDENT_2))

    return metrics_fpath


# ---------------------------------------------------------------------------
# Function used to name, initialize or clean up the output paths for the reverification process
# ---------------------------------------------------------------------------


def _prepare_output_fpaths(
    output_name_prefix: str,
    output_jsonl_fpath: str,
    resume_from_cache: bool,
    overwrite: bool,
    append: bool,
    *,
    protected_sources: tuple[Path, ...] = (),
) -> OutputPaths:
    output_fpath = Path(output_jsonl_fpath)
    output_fpath = output_fpath.with_name(output_name_prefix + output_fpath.name)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    # Inspect both spellings, but retain the destination for atomic publication:
    # replacing an output symlink must not replace its target.
    for target in {output_fpath, resolve_rollout_owner(output_fpath)}:
        if manifest_path_for(target).exists() or journal_path_for(target).exists():
            if overwrite:
                raise ConfigError(
                    "Cannot overwrite a manifest-backed run with reverification; choose a new output path."
                )
            raise ConfigError(
                "Manifest-backed reverification is a follow-up for in-place output; choose a separate output."
            )
    # Preserve an unambiguous legacy alias layout on append/resume. New outputs
    # put companions beside the target; fresh overwrite replaces the alias itself.
    # Manifest/journal-backed writes remain rejected above, through either spelling.
    replacing_alias = overwrite and not (append or resume_from_cache) and output_fpath.is_symlink()
    destination = (
        output_fpath.absolute()
        if replacing_alias
        else resolve_rollout_path(output_fpath, read_only=append or resume_from_cache)
    )
    failures_fpath = failures_path_for(destination)
    for written in (output_fpath, failures_fpath, aggregate_metrics_path_for(destination)):
        for source in protected_sources:
            if written.resolve() == source.resolve() or (
                written.exists() and source.exists() and written.samefile(source)
            ):
                raise ConfigError(
                    "Reverification output must not overwrite source rollout artifacts or its recovery files. "
                    "Choose a separate output path. Saved artifacts were not changed."
                )
    if not (append or resume_from_cache):
        # A fresh run must not silently clobber a prior run's rollouts: delete only when the user
        # explicitly opts in via overwrite, otherwise refuse. resume_from_cache and append reuse the file.
        for fpath in (output_fpath, failures_fpath):
            if not fpath.exists():
                continue
            if overwrite:
                fpath.unlink()
                print(f"Deleted existing output file: '{fpath}'")
            else:
                raise ConfigError(
                    f"Output file already exists: '{fpath}'. Pass --overwrite to delete it and start fresh, "
                    "or --resume to continue from it."
                )
        # Nor the shard list of the merged rows it deleted. A run that appends drops it when it starts writing.
        aggregation_sources_path_for(destination).unlink(missing_ok=True)
    return OutputPaths(output=destination, failures=failures_fpath)


def _start_over_without_cache(config: RolloutReverificationConfig, output_fpaths: OutputPaths) -> None:
    """A resume that finds neither the output nor its failures sidecar starts the output anew, so the exporters,
    which reopened the run recorded next to it, start a new run too. Append recovery continues its output."""
    if config.resume_from_cache and not config.append:
        if not (output_fpaths.output.exists() or output_fpaths.failures.exists()):
            for exporter in get_exporters():
                exporter.start_over()


def _source_identity_stamps(
    fpaths: Tuple[Path, ...],
) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], set[Tuple[int, int]]]:
    """Collection's identity stamps of each source rollout by (task, rollout), and the keys of every source rollout.

    Rollouts without stamps have a key but no stamps entry.
    """
    stamps_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    source_keys: set[Tuple[int, int]] = set()
    for fpath in dict.fromkeys(fpaths):
        if not fpath.exists():
            continue
        with fpath.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                row = orjson.loads(line)
                task, rollout = row.get(TASK_INDEX_KEY_NAME), row.get(ROLLOUT_INDEX_KEY_NAME)
                if task is None or rollout is None:
                    continue
                source_keys.add((task, rollout))
                stamps = {key: row[key] for key in _IDENTITY_KEYS if key in row}
                if stamps:
                    stamps_by_key.setdefault((task, rollout), stamps)
    return stamps_by_key, source_keys


def _load_reverified_results(
    output_fpath: Path, identity_sources: Tuple[Path, ...] = ()
) -> Tuple[List[Dict], List[Dict]]:
    """Load the full main jsonl (cached + newly re-verified successes), sorted by (task, rollout).

    Returns ``(results, rows)``: ``results`` are the parsed rows — the source of truth used for both
    the W&B rollouts export and the aggregate-metrics payload; ``rows`` is a minimal
    ``{agent_ref, task_source}`` projection used only to route each result to its resources server
    (with the same resolver as /verify). Read once and reused for both so the file is never read twice.

    A row cached by a reverification that did not keep collection's identity stamps gets the stamps its source
    rollout (same task and rollout index, from ``identity_sources``) carries, so a resumed output groups as a fresh
    one does. Only missing stamps are filled, in memory; the output file is not rewritten. A row whose source
    rollout has no stamp groups as that rollout does. If an unstamped row has no source rollout at all while other
    rows are stamped, its identity cannot be established and aggregating would split one run, so this raises
    ``ConfigError`` before anything is exported or aggregated; the output file keeps every row.
    """
    with output_fpath.open("rb") as f:
        results = [orjson.loads(line) for line in f if line.strip()]
    results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
    if identity_sources and any(key not in r for r in results for key in _IDENTITY_KEYS):
        stamps_by_key, source_keys = _source_identity_stamps(identity_sources)
        for r in results:
            for key, value in stamps_by_key.get((r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]), {}).items():
                r.setdefault(key, value)
        unresolved = [
            (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])
            for r in results
            if ENVIRONMENT_SERVER_STAMP_KEY_NAME not in r
            and (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]) not in source_keys
        ]
        if unresolved and any(ENVIRONMENT_SERVER_STAMP_KEY_NAME in r for r in results):
            raise ConfigError(
                f"{len(unresolved)} cached row(s) in {output_fpath} (first (task, rollout): {unresolved[:3]}) have no "
                "Environment Server stamp and no source rollout to take it from, while the other rows carry one; "
                "aggregating them would split one run into two groups. Nothing was exported or aggregated, and the "
                "output keeps every row. Re-run with resume_from_cache=false and a fresh output against the source "
                "rollouts that produced them, or restore that rollouts file and resume again."
            )
    rows = [
        {k: r[k] for k in (AGENT_REF_KEY_NAME, TASK_SOURCE_KEY_NAME, ENVIRONMENT_SERVER_STAMP_KEY_NAME) if k in r}
        for r in results
    ]
    return results, rows


def _parse_settle_source_snapshot(raw: str) -> dict[str, str | None]:
    try:
        expected = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ConfigError("Invalid rejudge source snapshot JSON.") from error
    required = {
        "rollouts.jsonl",
        "rollouts_failures.jsonl",
        "rollouts_manifest.json",
        "rollouts_materialized_inputs.jsonl",
        "rollouts_resolved_config.yaml",
        "rollouts_attempts.jsonl",
    }
    if not isinstance(expected, dict) or not required.issubset(expected):
        raise ConfigError("Incomplete rejudge source snapshot.")
    if any(
        not isinstance(name, str)
        or Path(name).name != name
        or name in {".", ".."}
        or (digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)))
        for name, digest in expected.items()
    ):
        raise ConfigError("Invalid rejudge source snapshot entry.")
    if any(expected[name] is None for name in required - {"rollouts_attempts.jsonl"}):
        raise ConfigError("Required rejudge source artifact was not observed.")
    return expected


def validate_settle_source_snapshot(source: Path, raw: str) -> None:
    """Recheck the queued artifact snapshot under the caller's existing run lock, before judging."""
    for name, expected in _parse_settle_source_snapshot(raw).items():
        path = source.parent / name
        try:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        except FileNotFoundError:
            actual = None
        if actual != expected:
            raise ConfigError(f"Rejudge source artifact {name} changed while queued; nothing was judged.")


def _reverifiable_saved_result(row: Dict[str, Any]) -> bool:
    """Whether a selected saved result is graded again from its saved answer (``reverification_cause``)."""
    from nemo_gym.orchestration.recovery import reverification_cause, saved_row_evidence

    return reverification_cause(saved_row_evidence(row)) is not None


def bind_judge_failures(
    source: Path, *, skip_spent: bool = False, withdraw_unresolved: bool = False
) -> Dict[str, Any]:
    """Snapshot the source run's selected judge failures and provenance before any judge call; the caller holds run_lock.

    A failed re-judgment spends the run's failure budget (``attempt_counts`` against the saved
    ``max_rollout_attempts``); a rollout that has spent it, or a run without a saved cap, is refused before judging.
    With ``skip_spent`` a rollout that has spent it is left out of the binding, and the others are bound. An admitted
    re-judgment with no saved outcome is refused, unless ``withdraw_unresolved``: the caller ruled out a live job of
    the run, so the pass that admitted it died, and it is withdrawn and its row bound again.

    The selected saved results whose grading failed for a judge or verify-transport cause, or was never recorded
    (``_reverifiable_saved_result``), are bound too, under ``saved_results``: a failed re-verification of one saves
    nothing, so it spends no failure attempt.
    """
    store = RolloutStore.open_for_rejudgment(source)
    for identity, pending in dict(store.manifest.pending_rejudge).items():
        if identity in store._state.unresolved_dispatches and withdraw_unresolved:
            store.withdraw_rejudgment(identity, pending.operation_id, unsaved_judge_failure=True)
            continue
        if identity in store._state.unresolved_dispatches:
            raise ConfigError(
                f"Rollout {identity!r} has an admitted re-judgment ({pending.operation_id}) with no saved outcome; "
                "whether its judge ran is unknown. Nothing was judged."
            )
        store.close_rejudgment(identity, pending.operation_id)  # its outcome was saved before the close
    cap = store.manifest.max_rollout_attempts
    if cap is None:
        raise ConfigError(f"{source} has no saved failure cap; judge-only settlement needs one. Nothing was judged.")
    spent = set()
    for row in store.failures():
        identity = logical_rollout_id(row)
        if row.get("_ng_failure_class") == "judge_failed" and store._state.failure_count(identity) >= cap:
            if not skip_spent:
                raise ConfigError(
                    f"Rollout {identity!r} has spent its {cap} saved failure attempts; nothing was judged."
                )
            spent.add(identity)
    saved_results = {
        logical_rollout_id(row): row for row in store.selected("success") if _reverifiable_saved_result(row)
    }
    return {
        "manifest": store.manifest.model_dump_json(),
        "materialized": materialized_path_for(store.output).read_bytes(),
        "failures": {
            logical_rollout_id(row): row
            for row in store.failures()
            if row.get("_ng_failure_class") == "judge_failed" and logical_rollout_id(row) not in spent
        }
        | saved_results,
        "saved_results": set(saved_results),
    }


# Settings of the run's own collection that the in-place re-judgment of its judge failures replaces.
_COLLECTION_ONLY_KEYS = (
    "limit",
    "output_jsonl_fpath",
    "materialized_inputs_jsonl_fpath",
    "resume_from_cache",
    "resubmit",
    "require_complete",
)


async def _drop_unsupported_saved_results(binding: Dict[str, Any]) -> None:
    """Leave out the bound saved results whose verifier is not stateless: only a ``STATELESS`` verifier grades a saved
    answer again. Judge failures stay bound as before."""
    saved = binding["saved_results"]
    binding["unsupported"] = []  # the saved results left out, for the pass's settlement record
    if not saved:
        return
    server_client = setup_server_client()
    agent_to_rs = _build_agent_to_resources_server_mapping(server_client.global_config_dict)
    routes = {
        identity: _rs_for_row(binding["failures"][identity], agent_to_rs, server_client.global_config_dict)
        for identity in saved
    }
    unsupported = set(await _check_reverify_mode(server_client, dict(routes)))
    skipped = sorted(identity for identity, name in routes.items() if name in unsupported)
    if skipped:
        print(
            f"{len(skipped)} saved result(s) are not re-verified because their resources server(s) "
            f"{sorted(unsupported)} report reverify_mode UNSUPPORTED or UNKNOWN; they stay as they were."
        )
    for identity in skipped:
        saved.discard(identity)
        binding["failures"].pop(identity)
    binding["unsupported"] = skipped


def due_judge_failures(output: Path) -> List[str]:
    """The rollouts of a managed run whose selected judge failure, or saved result to grade again
    (``_reverifiable_saved_result``), a re-judgment can still settle.

    A judge failure whose saved failure attempts are spent, a run without a manifest and a run without a saved
    failure cap have none.
    """
    store = RolloutStore.read(output, import_legacy=False)
    if store is None or store.manifest.legacy_import or store.manifest.max_rollout_attempts is None:
        return []
    cap = store.manifest.max_rollout_attempts
    return sorted(
        {
            logical_rollout_id(row)
            for row in store.failures()
            if _is_judge_failure(row) and store._state.failure_count(logical_rollout_id(row)) < cap
        }
        | {logical_rollout_id(row) for row in store.selected("success") if _reverifiable_saved_result(row)}
    )


def _settled_count(output: Path) -> int:
    """How many selected results of the run a re-judgment settled."""
    store = RolloutStore.read(output, import_legacy=False)
    return sum(REJUDGE_OPERATION_ID_KEY in row for row in store.selected("success")) if store is not None else 0


async def rejudge_saved_judge_failures(run_config: Dict[str, Any], output: Path) -> int:
    """Re-judge the saved judge failures of the run ``output`` in place; return how many rollouts it settled.

    The saved results whose grading failed for a judge or verify-transport cause, or was never recorded, are graded
    again from their saved answers too (``bind_judge_failures``); only stateless verifiers grade them.

    A run's driver calls it at the end of collection, and so again at the end of each resumed collection, while the
    run's servers are up. Only the saved answers go to /verify again: no policy call, no new model attempt. It is the
    ``gym eval reverify --judge-failed-only --settle-into-source`` path: each re-judgment is admitted in the run
    manifest before its judge call and spends one of the rollout's saved failure attempts, so an interrupted pass is
    not judged twice, and a rollout that has spent them keeps its masked row. The judged rows are kept under
    ``reverification/<pass>/`` beside the run, and each settled row names its admission and the failure it settles.
    A run with no due judge failure returns 0 and judges nothing.
    """
    if not due_judge_failures(output):
        return 0
    settled_before = _settled_count(output)
    rejudge_pass = f"auto-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    config = RolloutReverificationConfig.model_validate(
        {key: value for key, value in run_config.items() if key not in _COLLECTION_ONLY_KEYS}
        | {
            "rollouts_jsonl_fpath": str(output),
            "materialized_inputs_jsonl_fpath": str(materialized_path_for(output)),
            "output_jsonl_fpath": str(output.parent / "reverification" / rejudge_pass / output.name),
            "judge_failed_only": True,
            "settle_into_source": True,
            "skip_spent_judge_failures": True,
            "disable_aggregation": True,
        }
    )
    await RolloutReverificationHelper().run_from_config(config)
    return _settled_count(output) - settled_before


def settle_rejudgments(source: Path, judged: List[Dict], binding: Dict[str, Any]) -> int:
    """Settle the bound judge failures with the results judged in this run; returns how many were added.

    The caller holds the run lock taken for the binding, so no other writer ran in between. The source must still match
    the binding. A judged identity that the binding does not hold, a changed manifest or materialized input, or a
    failure that is no longer the selected outcome is refused. A bound saved result replaces its grade only with a
    result that binds the same saved answer and attempt (``is_rejudgment_of``); one whose re-verification answered
    without a grade keeps its saved row, and its admission is withdrawn.

    Each settlement is recorded in ``binding["settled_operations"]`` (this pass's admission operation ids) as soon as
    it is fsynced, and each completed withdrawal in ``binding["withdrawn_operations"]``, so a pass that raises
    afterwards, for example on the Store's exit-time coverage write, still counts them, each once. Settling again
    after such an error (the failed pass's exit callback) skips what this pass already settled or withdrew, closes a
    settled admission that is still open, and expects only the pass's other admissions to be open.
    """
    store = RolloutStore.open_for_rejudgment(source)
    settled_operations = binding.setdefault("settled_operations", set())
    withdrawn_operations = binding.setdefault("withdrawn_operations", set())
    done = settled_operations | withdrawn_operations
    admitted = binding.get("admitted", {})
    open_now = store.manifest.pending_rejudge
    # A settlement whose admission was not closed before an error: its outcome is saved, so it is closed below.
    unclosed = {
        identity: pending
        for identity, pending in open_now.items()
        if admitted.get(identity) == pending and pending.operation_id in settled_operations
    }
    own = {identity: pending for identity, pending in open_now.items() if identity not in unclosed}
    # Must equal this pass's admissions that it has not settled or withdrawn yet, exactly.
    expected = {identity: pending for identity, pending in admitted.items() if pending.operation_id not in done}
    if (
        store.manifest.model_copy(update={"pending_rejudge": {}}).model_dump_json()
        != RunManifest.model_validate_json(binding["manifest"])
        .model_copy(update={"pending_rejudge": {}})
        .model_dump_json()
        or own != expected
        or materialized_path_for(store.output).read_bytes() != binding["materialized"]
    ):
        raise ConfigError(f"{source} changed between re-judgment and settlement; nothing was settled.")
    planned = []
    identities = [logical_rollout_id(result) for result in judged]
    if len(identities) != len(set(identities)):
        raise ConfigError(
            f"Re-judged results name a rollout more than once; nothing was settled: {sorted(identities)}"
        )
    saved_results = binding.get("saved_results", set())
    withdrawn = []
    for result in judged:
        identity = logical_rollout_id(result)
        failure = binding["failures"].get(identity)
        if failure is None:
            raise ConfigError(f"Re-judged rollout {identity!r} is not a bound judge failure of {source}.")
        if binding["admitted"][identity].operation_id in done:
            continue  # this pass settled or withdrew it before an error
        if _ungraded_reverification(result) and (
            identity in saved_results or result.get(NG_FAILURE_CLASS_KEY) != "judge_failed"
        ):
            # The verifier answered without a grade or a judge failure: the bound row stays as it was, and nothing
            # is saved. A judge failure that failed to judge again is saved below as its re-judgment's outcome.
            withdrawn.append((identity, binding["admitted"][identity].operation_id))
            continue
        # /verify returns task and rollout indices only; the bound failure supplies its run and attempt identity.
        for name in (RUN_ID_KEY, ATTEMPT_INDEX_KEY_NAME):
            if name in result and result[name] != failure.get(name):
                raise ConfigError(f"Re-judged rollout {identity!r} has {name}={result[name]!r}, not the bound value.")
        planned.append(
            (
                failure,
                result
                | {
                    RUN_ID_KEY: failure.get(RUN_ID_KEY),
                    ATTEMPT_INDEX_KEY_NAME: failure.get(ATTEMPT_INDEX_KEY_NAME, 0),
                    REJUDGED_FAILURE_ID_KEY: rejudge_predecessor_id(failure),
                    "_ng_rejudge_operation_id": binding["admitted"][identity].operation_id,
                    "_ng_rejudge_admitted_at": binding["admitted"][identity].admitted_at,
                    # In-process monotonic interval of the whole admitted batch (admission to settlement). It is the
                    # batch's interval, not this rollout's share: never sum it over outcomes.
                    "_ng_rejudge_batch_interval_s": time.monotonic() - binding["admitted_monotonic"],
                },
            )
        )
    # Every row is validated before the first write, so a bad row commits nothing. Each write is one fsynced
    # outcome; after an interruption a rerun skips saved settlements (check_rejudgment returns None).
    for failure, result in planned:
        store.check_rejudgment(failure, result)
    settled = 0
    with store:
        for identity, pending in unclosed.items():
            store.close_rejudgment(identity, pending.operation_id)
        for failure, result in planned:
            operation_id = result["_ng_rejudge_operation_id"]
            if store.record_rejudgment(failure, result):
                settled += 1
                settled_operations.add(operation_id)  # fsynced: it counts even if a later write raises
            store.close_rejudgment(logical_rollout_id(result), operation_id)
        for identity, operation_id in withdrawn:
            store.withdraw_rejudgment(identity, operation_id, unsaved_judge_failure=identity not in saved_results)
            withdrawn_operations.add(operation_id)
    if withdrawn:
        print(f"{len(withdrawn)} re-verified result(s) were not graded again and stay as they were")
    return settled


def _validate_rescored_completion(result: object) -> None:
    """Keep a malformed verifier reply out of a managed source's separate scoring output."""
    if not isinstance(result, dict):
        raise ConfigError("The verifier must return a JSON object.")
    if result.get(NG_FAILURE_CLASS_KEY) is not None or result.get(NG_NO_PERSIST_KEY):
        return
    try:
        scored = type(result.get("reward")) in (int, float) and isfinite(result["reward"])
    except OverflowError:
        scored = False
    if not isinstance(result.get("response"), dict) or not scored:
        raise ConfigError("A completed verifier reply requires a response object and a finite numeric reward.")


def _ungraded_reverification(result: Dict[str, Any]) -> bool:
    """Whether a re-verification answered without a usable grade: a failure, a mask or no finite reward."""
    reward = result.get("reward")
    return (
        result.get(NG_FAILURE_CLASS_KEY) is not None
        or bool(result.get("mask_sample"))
        or result.get("evaluation_completed") is False
        or type(reward) not in (int, float)
        or not math.isfinite(reward)
    )


def _settle_what_a_failed_pass_judged(
    source: Path,
    identities: List[str],
    judged: List[Dict],
    binding: Dict[str, Any],
    record: Callable[[Dict[str, int]], None],
) -> Callable[..., bool]:
    """An exit callback for a pass that raised in-process after admitting its re-judgments, run under the run lock.

    The outcomes the pass already received are settled as usual, then each admission still without a saved outcome
    is withdrawn: its row stays selected, so a later pass grades it again and recovery decides REVERIFY instead of
    stopping on an outcome it cannot know. At worst a /verify call that did reach its server is made again for the
    same saved answer. An interrupted or killed pass leaves its admissions for recovery as before. ``record`` gets
    the pass's settlement counts, so its failed receipt says what it settled and withdrew.
    """

    def settle(exc_type, exc, traceback) -> bool:
        if exc_type is None or not issubclass(exc_type, Exception):
            return False
        if judged:
            try:
                settle_rejudgments(source, judged, binding)
            except Exception as error:  # what cannot be settled is withdrawn below; a settled row is never touched
                print(f"Re-judged results of a failed pass were not settled: {type(error).__name__}: {error}")
        withdrawn = 0
        withdrawn_operations = binding.setdefault("withdrawn_operations", set())
        try:
            store = RolloutStore.open_for_rejudgment(source)
            for identity in identities:
                pending = store.manifest.pending_rejudge.get(identity)
                if pending is None:
                    continue
                try:
                    store.withdraw_rejudgment(identity, pending.operation_id, unsaved_judge_failure=True)
                except ConfigError:  # an outcome was saved for it: the next binding closes its admission
                    continue
                withdrawn_operations.add(pending.operation_id)
                withdrawn += 1
        except Exception as error:  # the pass error stands; recovery sees the admissions as before
            print(f"Re-judgment admissions were not withdrawn: {type(error).__name__}: {error}", flush=True)
        if withdrawn:
            print(f"Withdrew {withdrawn} re-judgment admission(s) with no outcome; their rows stay as they were")
        unsupported = binding.get("unsupported", [])
        record(
            {
                "selected": len(binding["failures"]) + len(unsupported),
                "unsupported": len(unsupported),
                "judged": len(judged),
                # Every settlement this pass fsynced, including those saved before the error.
                "settled": len(binding.get("settled_operations", ())),
                # Withdrawals that happened, by admission, each once: not the ones settlement only planned.
                "withdrawn": len(withdrawn_operations),
            }
        )
        return False

    return settle


SETTLE_RECEIPT_VERSION = 1
# The counts a settle_into_source pass records (``RolloutReverificationHelper.settlement``) and its receipt carries.
_SETTLEMENT_COUNTS = ("selected", "judged", "settled", "withdrawn", "unsupported")


def write_settle_receipt(
    path: Path,
    *,
    state: str,
    source: Optional[str],
    job_id: Optional[str],
    settlement: Optional[Dict[str, Any]],
    started_at: str,
    coverage: Optional[Dict[str, Any]] = None,
    due_after: Optional[int] = None,
    refreshed: Optional[Dict[str, bool]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Write the judge-only recovery job's receipt at ``path`` (``settle_receipt_fpath``) and return it.

    ``state`` is ``completed`` or ``failed``. The counts are the pass's settlement record (zeros when the pass ended
    before it settled anything): the saved answers selected for grading again, those judged, those settled with a
    grade bound to the same saved answer, run and attempt, those re-verified without a grade and left as they were,
    and those not re-verified because their verifier is not stateless. ``coverage`` is the settled run's
    ``RolloutStore.coverage()`` (its counts only), ``due_after`` how many saved answers a later pass would still grade
    again, and ``refreshed`` whether the run's aggregate metrics and quality summary were recomputed. The receipt is
    written whole and moved into place, so a reader never sees a partial one.
    """
    counts = {name: int((settlement or {}).get(name) or 0) for name in _SETTLEMENT_COUNTS}
    receipt: Dict[str, Any] = {
        "version": SETTLE_RECEIPT_VERSION,
        "state": state,
        "job_id": job_id,
        "source": source,
        "started_at": started_at,
        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **counts,
        "due_after": due_after,
        "coverage": {k: v for k, v in coverage.items() if type(v) is int} if coverage else None,
        "metrics_recomputed": bool((refreshed or {}).get("metrics_recomputed")),
        "health_summary_recomputed": bool((refreshed or {}).get("health_summary_recomputed")),
        "error": error,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return receipt


def read_settle_receipt(path: Path) -> Dict[str, Any]:
    """The receipt at ``path``, as ``write_settle_receipt`` wrote it."""
    return json.loads(Path(path).read_text())


class RolloutReverificationHelper(BaseModel):
    # What the last ``settle_into_source`` pass did, in the counts ``write_settle_receipt`` records; None before a
    # pass settled, or when the pass does not settle into its source.
    settlement: Optional[Dict[str, Any]] = None

    async def run_from_config(self, config: RolloutReverificationConfig) -> List[Dict]:
        # With settle_into_source the source run lock is held from binding through judging to settlement, so a second
        # re-judgment of the same run is refused (non-blocking flock) instead of judging against the same budget.
        with ExitStack() as settle_lock:
            return await self._run_from_config(config, settle_lock)

    async def _run_from_config(self, config: RolloutReverificationConfig, settle_lock: ExitStack) -> List[Dict]:
        # A separate output may reuse selected source answers. Updating a managed
        # run in place still needs the history-aware recovery follow-up.
        for name in (config.output_jsonl_fpath,):
            if name is not None:
                path = _resolve_under_cwd_or_install(name)
                path = resolve_rollout_owner(path)
                if manifest_path_for(path).exists() or journal_path_for(path).exists():
                    raise ConfigError(
                        "Manifest-backed reverification is a follow-up for in-place output. "
                        "Re-score ordinary results into a separate output, or use a selected-result projection."
                    )
        # settle_into_source judges only the Store-selected judge failures (bound-row predicate below), so a history
        # whose superseded rows are earlier judge failures of the same attempt may be re-judged again.
        if (
            config.rollouts_jsonl_fpath is not None
            and config.judge_failed_only
            and not config.settle_into_source
            and not raw_outcomes_are_selected(_resolve_under_cwd_or_install(config.rollouts_jsonl_fpath))
        ):
            raise ConfigError(
                "Reverification of histories with superseded outcomes is a follow-up to evaluation resume. "
                "Use a selected-result projection in a separate output. Saved artifacts were not changed."
            )
        # Check every actual destination after applying the optional unsafe_
        # prefix, before output preparation can unlink or append any source.
        force_warning: Optional[str] = None
        output_name_prefix = ""
        if config.input_format != "atif":
            force_warning = await _guard_reverify_mode(config)
            if force_warning:
                print(force_warning)
                output_name_prefix = "unsafe_"

        materialized_inputs_jsonl_fpath = _resolve_under_cwd_or_install(config.materialized_inputs_jsonl_fpath)
        identity_sources: Tuple[Path, ...] = ()
        if config.input_format == "atif":
            assert config.atif_manifest_jsonl_fpath is not None
            atif_manifest_jsonl_fpath = _resolve_under_cwd_or_install(config.atif_manifest_jsonl_fpath)
            payloads_to_reverify = _prepare_atif_payloads(
                materialized_inputs_jsonl_fpath,
                atif_manifest_jsonl_fpath,
                config.limit,
            )
            await _guard_atif_preflight(payloads_to_reverify)
            output_fpaths = _prepare_output_fpaths(
                output_name_prefix,
                config.output_jsonl_fpath,
                config.resume_from_cache,
                config.overwrite,
                config.append,
                protected_sources=(materialized_inputs_jsonl_fpath, atif_manifest_jsonl_fpath),
            )
        else:
            assert config.rollouts_jsonl_fpath is not None
            source_path = _resolve_under_cwd_or_install(config.rollouts_jsonl_fpath)
            rollouts_jsonl_fpath = resolve_rollout_path(source_path, read_only=True)
            protected_sources = {materialized_inputs_jsonl_fpath}
            owner = resolve_rollout_owner(source_path)
            if manifest_path_for(owner).exists() or journal_path_for(owner).exists():
                for source in (source_path, rollouts_jsonl_fpath, owner):
                    protected_sources.update(
                        (
                            source,
                            failures_path_for(source),
                            materialized_path_for(source),
                            manifest_path_for(source),
                            journal_path_for(source),
                            coverage_path_for(source),
                            resubmit_offset_path_for(source),
                            aggregate_metrics_path_for(source),
                            source.with_name(source.stem + "_run.lock"),
                        )
                    )
            if config.judge_failed_only:
                _reject_multistage_recovery_source(rollouts_jsonl_fpath)
            output_fpaths = _prepare_output_fpaths(
                output_name_prefix,
                config.output_jsonl_fpath,
                config.resume_from_cache,
                config.overwrite,
                config.append,
                protected_sources=tuple(protected_sources),
            )
            _start_over_without_cache(config, output_fpaths)
            reverify_source_fpath = rollouts_jsonl_fpath
            rollout_predicate = None
            source_store = (
                RolloutStore.read(rollouts_jsonl_fpath, import_legacy=False) if not config.judge_failed_only else None
            )
            if source_store is not None:
                _validate_rescoring_cache(source_store, output_fpaths)

            if config.judge_failed_only:
                print(_RECOVERY_TWO_SOURCES_WARNING)
                if config.settle_into_source:
                    settle_lock.enter_context(run_lock(rollouts_jsonl_fpath))
                    if config.settle_source_snapshot_json is not None:
                        validate_settle_source_snapshot(rollouts_jsonl_fpath, config.settle_source_snapshot_json)
                    settle_binding = bind_judge_failures(
                        rollouts_jsonl_fpath,
                        skip_spent=config.skip_spent_judge_failures,
                        withdraw_unresolved=config.withdraw_unresolved_admissions,
                    )
                    await _drop_unsupported_saved_results(settle_binding)
                reverify_source_fpath = failures_path_for(rollouts_jsonl_fpath)
                saved_results = settle_binding["saved_results"] if config.settle_into_source else set()
                # Seed the successes and dedup so the re-verification doesn't judge successes again. A saved result
                # that is re-verified is not seeded: its re-verified row takes its place.
                skip_keys = _seed_output_with_successes(
                    rollouts_jsonl_fpath,
                    output_fpaths.output,
                    exclude=frozenset(
                        (settle_binding["failures"][identity][TASK_INDEX_KEY_NAME],)
                        + (settle_binding["failures"][identity][ROLLOUT_INDEX_KEY_NAME],)
                        for identity in saved_results
                    ),
                )
                rollout_predicate = _recovery_rollout_predicate(skip_keys)
                if config.settle_into_source:
                    # Judge exactly the bound selected failure rows, never a superseded row of the same attempt.
                    bound_rows = settle_binding["failures"]
                    recovery_predicate = rollout_predicate

                    def rollout_predicate(row: Dict[str, Any]) -> bool:
                        return bound_rows.get(logical_rollout_id(row)) == row and recovery_predicate(row)

            identity_sources = (rollouts_jsonl_fpath, reverify_source_fpath)

            payloads_to_reverify = _prepare_payloads(
                materialized_inputs_jsonl_fpath,
                reverify_source_fpath,
                output_fpaths,
                config.resume_from_cache,
                config.limit,
                rollout_predicate=rollout_predicate,
                retry_terminal_timeouts=config.retry_terminal_timeouts,
                **({"selected_rollouts": _selected_rescoring_rows(source_store)} if source_store is not None else {}),
            )
            if config.settle_into_source and settle_binding["saved_results"]:
                # The bound saved results are read from the run's results, exactly as bound.
                bound_results = {
                    identity: settle_binding["failures"][identity] for identity in settle_binding["saved_results"]
                }
                payloads_to_reverify += _prepare_payloads(
                    materialized_inputs_jsonl_fpath,
                    rollouts_jsonl_fpath,
                    output_fpaths,
                    config.resume_from_cache,
                    rollout_predicate=lambda row: bound_results.get(logical_rollout_id(row)) == row,
                )

        if config.settle_into_source:
            for row in payloads_to_reverify:
                bound = settle_binding["failures"].get(logical_rollout_id(row))
                if bound is None or row.get("response") != bound.get("response"):
                    raise ConfigError(
                        f"Judge input {logical_rollout_id(row)!r} is not a selected judge failure of "
                        f"{rollouts_jsonl_fpath} with the same saved answer; nothing was judged."
                    )

        judged_results: List[Dict] = []  # results and judge failures of this run, kept apart from the seeded ones
        if config.settle_into_source:
            # Durable admissions just before the judge calls; the run lock is held until settlement.
            identities = [logical_rollout_id(row) for row in payloads_to_reverify]
            if len(identities) != len(set(identities)):
                raise ConfigError("Judge inputs name a rollout more than once; nothing was judged.")
            settle_binding["admitted_monotonic"] = time.monotonic()
            # Runs before the run lock is released, so a pass that raises from here on settles what it judged and
            # withdraws what it admitted with no outcome.
            settle_lock.push(
                _settle_what_a_failed_pass_judged(
                    rollouts_jsonl_fpath,
                    identities,
                    judged_results,
                    settle_binding,
                    lambda counts: setattr(self, "settlement", counts),
                )
            )
            admit_store = RolloutStore.open_for_rejudgment(rollouts_jsonl_fpath)
            settle_binding["admitted"] = {
                logical_rollout_id(row): admit_store.admit_rejudgment(
                    settle_binding["failures"][logical_rollout_id(row)]
                )
                for row in payloads_to_reverify
            }

        semaphore = nullcontext()
        if config.num_samples_in_parallel is not None:
            print(f"Verifying with {config.num_samples_in_parallel} concurrent requests")
            semaphore = Semaphore(config.num_samples_in_parallel)

        pcts_to_print = [20, 40, 60, 80, 90, 95, 98, 99, 100]
        run_labels = rollout_run_labels(payloads_to_reverify)
        counts_left = Counter(run_labels.get(rollout_run_key(r)) for r in payloads_to_reverify)
        results_file = output_fpaths.output.open("ab")
        failures_file = output_fpaths.failures.open("ab")
        failure_counts: Counter = Counter()
        completed = 0  # number of rows re-verified this run (for progress reporting)
        sources_dropped = False
        try:
            for future in _run_verification_payloads(payloads_to_reverify, semaphore=semaphore):
                row, result = await future
                if _SOURCE_RUN_ID_KEY in row:
                    _validate_rescored_completion(result)

                result[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
                result[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]
                for key in (AGENT_REF_KEY_NAME, *_IDENTITY_KEYS):
                    if key in row:
                        result[key] = row[key]
                # Keep task_source alongside agent_ref: aggregation routes with the same resolver
                # as /verify (task_source authoritative), so it must survive into the output file.
                if TASK_SOURCE_KEY_NAME in row:
                    result[TASK_SOURCE_KEY_NAME] = row[TASK_SOURCE_KEY_NAME]
                if SKILLS_REF_KEY_NAME in row:
                    result[SKILLS_REF_KEY_NAME] = row[SKILLS_REF_KEY_NAME]
                for key in (ATIF_PROVENANCE_KEY, _SOURCE_RUN_ID_KEY, _SOURCE_ANSWER_KEY):
                    if key in row:
                        result[key] = row[key]

                no_persist = bool(result.get(NG_NO_PERSIST_KEY))
                failure_class = result.get(NG_FAILURE_CLASS_KEY)

                serialized = orjson.dumps(result)

                # ATIF writes kill-shaped diagnostics to its sidecar too.
                if not sources_dropped and (not no_persist or config.input_format == "atif"):
                    # The rows this run adds are not the merged ones an aggregation's shard list names. The list goes with the
                    # first row the run writes, so a run that writes none leaves it as it was.
                    aggregation_sources_path_for(output_fpaths.output).unlink(missing_ok=True)
                    sources_dropped = True
                if no_persist and config.input_format != "atif":
                    # kill_shaped: don't write anywhere. Set-difference on resume
                    # naturally re-dispatches; per-task timeout bounds wallclock.
                    pass
                elif no_persist or failure_class is not None:
                    # Ordinary failures go to the sidecar. ATIF also persists
                    # kill-shaped diagnostics because that mode cannot resume.
                    # The aggregator reads only the main jsonl, so neither path
                    # contaminates the score.
                    failure_class = failure_class or ATIF_NO_PERSIST_FAILURE_CLASS
                    result[NG_FAILURE_CLASS_KEY] = failure_class
                    serialized = orjson.dumps(result)
                    failure_counts[failure_class] += 1
                    # Every dropped rollout says so as it happens, as in rollout collection.
                    detail = str(result.get("_ng_failure_message") or result.get("error") or "")[:200]
                    tqdm.write(
                        "🚨 [rollout_reverification] rollout dropped from the score: "
                        f"row={json.dumps(_rollout_request_debug_summary(row), sort_keys=True)} "
                        f"class={failure_class} error={detail}"
                    )
                    failures_file.write(serialized + b"\n")
                    failures_file.flush()
                    if failure_class == "judge_failed" or (
                        config.settle_into_source and logical_rollout_id(row) in settle_binding["failures"]
                    ):
                        judged_results.append(result)
                else:
                    # Success → main jsonl.
                    results_file.write(serialized + b"\n")
                    results_file.flush()
                    judged_results.append(result)

                label = run_labels.get(rollout_run_key(row))
                counts_left[label] -= 1
                if counts_left[label] <= 0:
                    counts_left.pop(label)

                completed += 1
                current_pct = 100 * completed / len(payloads_to_reverify)
                if pcts_to_print and current_pct >= pcts_to_print[0]:
                    while pcts_to_print and current_pct >= pcts_to_print[0]:
                        pcts_to_print.pop(0)

                    top_left = counts_left.most_common(5)  # Fix to top 3 for now.
                    if top_left:
                        top_left_str = "\n".join(f"{i + 1}. {k}: {v}" for i, (k, v) in enumerate(top_left))
                        # Use tqdm.write here so we can print properly with tqdm being used.
                        tqdm.write(f"Examples left:\n{top_left_str}")
        finally:
            results_file.close()
            failures_file.close()

        # Read the full main jsonl (cached + newly re-verified successes) ONCE — the source of truth,
        # reused for both the rollouts export and aggregate metrics so the file is never re-read.
        results, agg_rows = _load_reverified_results(output_fpaths.output, identity_sources)

        if config.upload_rollouts and get_exporters():  # pragma: no cover
            print("Uploading rollouts. This may take a few minutes if your data is large.")
            export_rollouts([_rollout_for_export(r) for r in results])

        # Compute and write aggregate metrics via /aggregate_metrics on each agent server
        if config.disable_aggregation:
            print(
                "Skipping aggregate-metrics computation because disable_aggregation=True. "
                "Run `gym eval aggregate` after all shards finish to compute the global metrics."
            )
            aggregate_metrics_fpath = None
        else:
            print("Computing aggregate metrics")
            aggregate_metrics_fpath = await _call_aggregate_metrics(
                results,
                agg_rows,
                output_fpaths.output,
                materialized_fpaths=[Path(config.materialized_inputs_jsonl_fpath)],
            )

        expected_rollouts = len(results) + sum(failure_counts.values())
        coverage = _coverage_report(expected_rollouts, len(results), failure_counts, output_fpaths.failures)
        if get_exporters():  # pragma: no cover
            export_metrics(
                {
                    "coverage/expected": expected_rollouts,
                    "coverage/scored": len(results),
                    "coverage/missing": sum(failure_counts.values()),
                }
            )

        print(f"""Finished rollout collection! View results at:
        Re-verified rollouts: {output_fpaths.output}
        Aggregate metrics: {aggregate_metrics_fpath}{coverage}""")
        if force_warning:
            print(force_warning)

        if config.settle_into_source:
            settled = settle_rejudgments(rollouts_jsonl_fpath, judged_results, settle_binding)
            print(f"Settled {settled} re-judged judge failure(s) into {rollouts_jsonl_fpath}")
            unsupported = settle_binding.get("unsupported", [])
            self.settlement = {
                "selected": len(settle_binding["failures"]) + len(unsupported),
                "unsupported": len(unsupported),
                "judged": len(payloads_to_reverify),
                "settled": len(settle_binding["settled_operations"]),
                "withdrawn": len(settle_binding["withdrawn_operations"]),
            }

        # The full main jsonl (cached + newly re-verified successes), sorted by (task, rollout).
        return results
