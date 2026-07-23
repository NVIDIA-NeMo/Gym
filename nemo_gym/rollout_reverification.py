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
import json
from asyncio import Future, Semaphore
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import orjson
from omegaconf import DictConfig
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm
from wandb import Table

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest, ReverifyMode
from nemo_gym.config_types import BaseNeMoGymCLIConfig, ConfigError
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    get_wandb_run,
)
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_collection import (
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    NG_TERMINAL_KEY,
    _get_max_rollout_attempts,
)
from nemo_gym.server_utils import (
    ServerClient,
    get_response_json,
    is_global_aiohttp_client_request_debug_enabled,
    raise_for_status,
    setup_server_client,
)


class RolloutReverificationConfig(BaseNeMoGymCLIConfig):
    materialized_inputs_jsonl_fpath: str = Field(
        description="The file path of the materialized inputs as output by `gym eval run`."
    )
    rollouts_jsonl_fpath: str = Field(
        description="The file path of the rollouts to re-verify, as output by `gym eval run`."
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
    upload_rollouts_to_wandb: bool = Field(
        default=True,
        description="Upload the rollouts to W&B. Sometimes this should be off because the rollouts are massive. Default: True",
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
    global_conflict_dict: Union[Dict[str, Any], "DictConfig"],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw_name, block in global_conflict_dict.items():
        name = str(raw_name)
        if isinstance(block, (dict, DictConfig)) and "responses_api_agents" in block:
            impl = next(iter(block["responses_api_agents"].values()))
            rs = (impl.get("resources_server") or {}).get("name")
            if rs:
                mapping[name] = rs
    return mapping


def _agent_to_rs_mapping_from_resources_only_config(
    global_conflict_dict: Union[Dict[str, Any], "DictConfig"],
) -> Dict[str, str]:
    # The rollout rows still carry agent names that were never started, so fall back to the
    # single resources server for EVERY requested key.
    resources_server_names = [
        str(name)
        for name, block in global_conflict_dict.items()
        if isinstance(block, (dict, DictConfig)) and "resources_servers" in block
    ]
    if len(resources_server_names) == 1:
        only = resources_server_names[0]
        return defaultdict(lambda: only)  # any key → the one resources server instance
    if not resources_server_names:
        raise ConfigError("reverify: no resources server found in the config.")
    raise ConfigError(
        f"reverify: multiple resources servers {resources_server_names} and no agent blocks to "
        "route by. Use a config with agent blocks."
    )


def _build_agent_to_resources_server_mapping(
    global_conflict_dict: Union[Dict[str, Any], "DictConfig"],
) -> Dict[str, str]:
    mapping = _agent_to_rs_mapping_from_agent_blocks(global_conflict_dict)
    if mapping:
        return mapping
    return _agent_to_rs_mapping_from_resources_only_config(global_conflict_dict)


# ---------------------------------------------------------------------------
# Function used to summarize the debug information for a failed verification
# ---------------------------------------------------------------------------
def _rollout_verify_debug_summary(row: Dict[str, Any], resources_server_name: str) -> Dict[str, Any]:
    agent_ref = row.get(AGENT_REF_KEY_NAME) or {}
    summary = {
        TASK_INDEX_KEY_NAME: row.get(TASK_INDEX_KEY_NAME),
        ROLLOUT_INDEX_KEY_NAME: row.get(ROLLOUT_INDEX_KEY_NAME),
        "agent_name": agent_ref.get("name") if isinstance(agent_ref, dict) else None,
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


# to do: OutputPaths to OutputFPaths
def _load_cache_keys_by_status(output_fpaths: OutputPaths) -> CacheKeysByStatus:
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
                if fr.get(NG_TERMINAL_KEY):
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
- {len(cache.terminal_keys)} sidecar-terminal (timeout_exceeded / skipped) → not retried
- {len(cache.maxed_out_keys)} hit max_attempts → not retried
- {len(filtered_payloads)} rows that still need to be run"""
    )


# ---------------------------------------------------------------------------
# Functions used by the main RolloutReverificationHelper in the reverification process:
# Yielding InputRolloutPair objects to be re-verified
# Preparing the payloads from them (by skipping rows that are already in the cache and some formatting)
# And running the verification requests in parallel
# ---------------------------------------------------------------------------


def _yield_inputs_and_rollouts_paired(
    materialized_inputs_jsonl_fpath: Path, rollouts_jsonl_fpath: Path, limit: Optional[int] = None
) -> "Iterator[InputRolloutPair]":
    inputs_by_key = {}
    with open(materialized_inputs_jsonl_fpath) as m_f:
        for line in m_f:
            r = orjson.loads(line)
            inputs_by_key[(r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])] = r
    with open(rollouts_jsonl_fpath) as r_f:
        for i, line in tqdm(enumerate(r_f), desc="Reading rollouts"):  # never holds the whole file
            if limit is not None and i >= limit:
                break
            rollout_row = orjson.loads(line)
            input_row = inputs_by_key.get((rollout_row[TASK_INDEX_KEY_NAME], rollout_row[ROLLOUT_INDEX_KEY_NAME]))
            if input_row is None:
                raise ConfigError(f"No matching materialized input row found for rollout row {rollout_row}")
            yield InputRolloutPair(input=input_row, rollout=rollout_row)


def _build_verify_payload(pair: InputRolloutPair) -> Dict:
    return pair.input | {"response": pair.rollout["response"]}


def _prepare_payloads(
    materialized_inputs_jsonl_fpath: Path,
    rollouts_jsonl_fpath: Path,
    output_fpaths: OutputPaths,
    resume_from_cache: bool,
    limit: Optional[int] = None,
) -> List[Dict]:
    all_payloads = [
        _build_verify_payload(pair)
        for pair in _yield_inputs_and_rollouts_paired(
            materialized_inputs_jsonl_fpath, rollouts_jsonl_fpath, limit=limit
        )
    ]
    if resume_from_cache:
        cache = _load_cache_keys_by_status(output_fpaths)
        payloads = list(_drop_cache_from_payloads(all_payloads, cache))
        summarize_cache_usage(cache, all_payloads, payloads)
        return payloads
    else:
        return all_payloads


def _run_verification_payloads(
    payloads: List[Dict],
    semaphore: Semaphore | nullcontext[None] | None = None,
) -> Iterator[Future]:  # pragma: no cover
    semaphore = semaphore or nullcontext[None]()
    server_client = setup_server_client()
    agent_to_rs = _build_agent_to_resources_server_mapping(server_client.global_config_dict)

    async def _post_subroutine(row: Dict) -> Tuple[Dict, Dict]:
        async with semaphore:
            rs_name = agent_to_rs[row[AGENT_REF_KEY_NAME]["name"]]
            res = await server_client.post(server_name=rs_name, url_path="/verify", json=row)
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
            return row, await get_response_json(res)

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


async def _check_reverify_mode(server_client: "ServerClient", agent_to_rs: Dict[str, str]) -> List[str]:
    """Query GET /reverify_mode on each unique resource server referenced by agent_to_rs.

    Returns a sorted list of RS names that reported ReverifyMode.UNSUPPORTED or ReverifyMode.UNKNOWN.
    """
    unsupported: List[str] = []
    for rs_name in set(agent_to_rs.values()):
        res = await server_client.get(server_name=rs_name, url_path="/reverify_mode")
        await raise_for_status(res)
        mode = ReverifyMode(await get_response_json(res))
        if mode in (ReverifyMode.UNSUPPORTED, ReverifyMode.UNKNOWN):
            unsupported.append(rs_name)
    return sorted(unsupported)


async def _guard_reverify_mode(config: RolloutReverificationConfig) -> Optional[str]:
    """Check reverify_mode for every RS in the config before reverification starts.

    Returns a warning string when force=True and at least one RS is UNSUPPORTED or UNKNOWN (caller must
    print it and apply the unsafe_ output prefix).
    Raises ConfigError when force=False and at least one RS is UNSUPPORTED or UNKNOWN.
    Returns None when all RS are STATELESS.
    """
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


# ---------------------------------------------------------------------------
# Function used to compute the aggregate metrics after the reverification process
# Very similar to the rollout collection code, but we need to send the request to
# resources servers instead of the agent server, since the second one might not be started
# ---------------------------------------------------------------------------
async def _call_aggregate_metrics(
    results: List[Dict],
    rows: List[Dict],
    output_fpath: Path,
) -> Optional[Path]:
    """Call /aggregate_metrics on each resource server after rollouts complete.

    Writes a single _aggregate_metrics.json with one entry per agent (same shape
    as the old _agent_metrics.json). Returns the file path.
    """
    if not results:
        return None

    server_client = setup_server_client()
    agent_to_rs = _build_agent_to_resources_server_mapping(server_client.global_config_dict)
    # Group results by agent name
    agent_results: Dict[str, List[Dict]] = {}
    for row, result in zip(rows, results):
        agent_name = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
        if not agent_name:
            continue
        agent_results.setdefault(agent_name, []).append(result)

    async def _fetch_agent_metrics(agent_name: str, agent_result_list: List[Dict]) -> Dict:
        # Strip heavyweight fields before sending, but preserve response.usage
        stripped = []
        for r in agent_result_list:
            entry = {k: v for k, v in r.items() if k not in ("response", "responses_create_params")}
            usage = (r.get("response") or {}).get("usage")
            if usage:
                entry["response"] = {"usage": usage}
            stripped.append(entry)

        agg_request = AggregateMetricsRequest(verify_responses=stripped)
        agg_response = await server_client.post(
            server_name=agent_to_rs[agent_name],
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
        }
        return agent_entry

    all_agent_metrics: List[Dict] = []
    tasks = [_fetch_agent_metrics(name, results_list) for name, results_list in agent_results.items()]
    for coro in asyncio.as_completed(tasks):
        agent_entry = await coro
        all_agent_metrics.append(agent_entry)

        agent_name = agent_entry[AGENT_REF_KEY_NAME]["name"]
        key_metrics = agent_entry.get("key_metrics", {})
        print(f"\nKey metrics for {agent_name}:\n" + json.dumps(key_metrics, indent=4))

    primitive_types = (bool, int, float, str, type(None))
    metrics_to_log = dict()
    for agent_entry in all_agent_metrics:
        agent_name = agent_entry[AGENT_REF_KEY_NAME]["name"]
        metrics_to_log.update(
            {f"{agent_name}/{k}": v for k, v in agent_entry["agent_metrics"].items() if isinstance(v, primitive_types)}
        )
        metrics_to_log.update(
            {
                f"key_metrics/{agent_name}/{k}": v
                for k, v in agent_entry["key_metrics"].items()
                if isinstance(v, primitive_types)
            }
        )
    if get_wandb_run():  # pragma: no cover
        get_wandb_run().log(metrics_to_log)

    # Write single file with all agents
    metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
    metrics_fpath.write_bytes(orjson.dumps(all_agent_metrics, option=orjson.OPT_INDENT_2))

    return metrics_fpath


# ---------------------------------------------------------------------------
# Function used to name, initialize or clean up the output paths for the reverification process
# ---------------------------------------------------------------------------


def _prepare_output_fpaths(
    output_name_prefix: str, output_jsonl_fpath: str, resume_from_cache: bool, overwrite: bool
) -> OutputPaths:
    output_fpath = Path(output_jsonl_fpath)
    output_fpath = output_fpath.with_name(output_name_prefix + output_fpath.name)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    failures_fpath = failures_path_for(output_fpath)
    if not resume_from_cache:
        # A fresh run must not silently clobber a prior run's rollouts: delete only when the user
        # explicitly opts in via overwrite, otherwise refuse. resume_from_cache reuses the file.
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
    return OutputPaths(output=output_fpath, failures=failures_fpath)


def _load_reverified_results(output_fpath: Path) -> Tuple[List[Dict], List[Dict]]:
    """Load the full main jsonl (cached + newly re-verified successes), sorted by (task, rollout).

    Returns ``(results, rows)``: ``results`` are the parsed rows — the source of truth used for both
    the W&B rollouts export and the aggregate-metrics payload; ``rows`` is a minimal ``{AGENT_REF}``
    projection used only to route each result to its resources server. Read once and reused for both
    so the file is never read twice.
    """
    with output_fpath.open("rb") as f:
        results = [orjson.loads(line) for line in f if line.strip()]
    results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
    rows = [{AGENT_REF_KEY_NAME: r.get(AGENT_REF_KEY_NAME)} for r in results]
    return results, rows


class RolloutReverificationHelper(BaseModel):
    async def run_from_config(self, config: RolloutReverificationConfig) -> List[Dict]:
        force_warning = await _guard_reverify_mode(config)
        if force_warning:
            print(force_warning)
            output_name_prefix = "unsafe_"
        else:
            output_name_prefix = ""

        output_fpaths = _prepare_output_fpaths(
            output_name_prefix, config.output_jsonl_fpath, config.resume_from_cache, config.overwrite
        )
        materialized_inputs_jsonl_fpath = _resolve_under_cwd_or_install(config.materialized_inputs_jsonl_fpath)
        rollouts_jsonl_fpath = _resolve_under_cwd_or_install(
            config.rollouts_jsonl_fpath
        )  # rollouts are inputs for the verification

        payloads_to_reverify = _prepare_payloads(
            materialized_inputs_jsonl_fpath,
            rollouts_jsonl_fpath,
            output_fpaths,
            config.resume_from_cache,
            config.limit,
        )
        semaphore = nullcontext()
        if config.num_samples_in_parallel is not None:
            print(f"Verifying with {config.num_samples_in_parallel} concurrent requests")
            semaphore = Semaphore(config.num_samples_in_parallel)

        pcts_to_print = [20, 40, 60, 80, 90, 95, 98, 99, 100]
        counts_left = Counter(r[AGENT_REF_KEY_NAME]["name"] for r in payloads_to_reverify)
        results_file = output_fpaths.output.open("ab")
        failures_file = output_fpaths.failures.open("ab")
        completed = 0  # number of rows re-verified this run (for progress reporting)
        try:
            for future in _run_verification_payloads(payloads_to_reverify, semaphore=semaphore):
                row, result = await future

                result[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
                result[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]
                result[AGENT_REF_KEY_NAME] = row[AGENT_REF_KEY_NAME]
                if SKILLS_REF_KEY_NAME in row:
                    result[SKILLS_REF_KEY_NAME] = row[SKILLS_REF_KEY_NAME]

                no_persist = bool(result.get(NG_NO_PERSIST_KEY))
                failure_class = result.get(NG_FAILURE_CLASS_KEY)

                serialized = orjson.dumps(result)

                if no_persist:
                    # kill_shaped: don't write anywhere. Set-difference on resume
                    # naturally re-dispatches; per-task timeout bounds wallclock.
                    pass
                elif failure_class is not None:
                    # Non-kill_shaped failure → sidecar. The aggregator only reads
                    # the main jsonl, so this keeps win-rate uncontaminated.
                    failures_file.write(serialized + b"\n")
                    failures_file.flush()
                else:
                    # Success → main jsonl.
                    results_file.write(serialized + b"\n")
                    results_file.flush()

                counts_left[row[AGENT_REF_KEY_NAME]["name"]] -= 1
                if counts_left[row[AGENT_REF_KEY_NAME]["name"]] <= 0:
                    counts_left.pop(row[AGENT_REF_KEY_NAME]["name"])

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
        # reused for both the W&B rollouts export and aggregate metrics so the file is never re-read.
        results, agg_rows = _load_reverified_results(output_fpaths.output)

        if config.upload_rollouts_to_wandb and get_wandb_run():  # pragma: no cover
            print("Uploading rollouts to W&B. This may take a few minutes if your data is large.")
            get_wandb_run().log({"Rollouts": Table(data=[[orjson.dumps(r)] for r in results], columns=["Rollout"])})

        # Compute and write aggregate metrics via /aggregate_metrics on each agent server
        if config.disable_aggregation:
            print(
                "Skipping aggregate-metrics computation because disable_aggregation=True. "
                "Run `gym eval aggregate` after all shards finish to compute the global metrics."
            )
            aggregate_metrics_fpath = None
        else:
            print("Computing aggregate metrics")
            aggregate_metrics_fpath = await _call_aggregate_metrics(results, agg_rows, output_fpaths.output)

        print(f"""Finished rollout collection! View results at:
        Re-verified rollouts: {output_fpaths.output}
        Aggregate metrics: {aggregate_metrics_fpath}""")
        if force_warning:
            print(force_warning)

        # The full main jsonl (cached + newly re-verified successes), sorted by (task, rollout).
        return results
