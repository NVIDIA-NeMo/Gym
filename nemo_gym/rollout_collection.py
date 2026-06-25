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
import glob as glob_module
import json
import os
from asyncio import Future, Semaphore
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import orjson
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm
from wandb import Table

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.config_types import BaseNeMoGymCLIConfig, BaseServerConfig
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    RESPONSES_CREATE_PARAMS_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    get_wandb_run,
)
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    get_response_json,
    is_global_aiohttp_client_request_debug_enabled,
    is_global_aiohttp_client_setup,
    raise_for_status,
    set_global_aiohttp_client,
)


# ---------------------------------------------------------------------------
# Failure-routing sentinels (set by agent servers, read by the dispatcher).
#
# Background:
#   The historical contract was "every dispatched task produces one row in
#   the main rollouts jsonl, succeeded or failed." That contract broke
#   resume-after-walltime: synthetic ``-failed`` rows written during a
#   SIGTERM grace window look identical to real successes to the dedup in
#   ``_load_from_cache`` (which keys only on (task_index, rollout_index)),
#   so chain-hop 2 thinks failed tasks are done and never retries them.
#
# New contract:
#   - Successes go to the main jsonl (``output_jsonl_fpath``).
#   - Failures go to a sidecar (``<output_stem>_failures.jsonl``), one row
#     per attempt, with ``_ng_failure_class`` set.
#   - ``kill_shaped`` failures (Slurm SIGTERM, Ray actor died, OOM, ...) go
#     NOWHERE: the absence of a row is the canonical signal. Resume's
#     set-difference re-dispatches them naturally; per-task timeout bounds
#     the chain-hop wallclock.
#   - On resume, ``_load_from_cache`` reads BOTH files: main jsonl tells
#     it what's permanently done, sidecar tells it how many attempts each
#     non-success has consumed (capped at NEMO_GYM_MAX_ROLLOUT_ATTEMPTS,
#     default 3). Rows flagged ``_ng_failure_terminal=True`` are never
#     retried regardless of attempt count.
# ---------------------------------------------------------------------------

NG_FAILURE_CLASS_KEY = "_ng_failure_class"
NG_NO_PERSIST_KEY = "_ng_no_persist"
NG_TERMINAL_KEY = "_ng_failure_terminal"

_DEFAULT_MAX_ROLLOUT_ATTEMPTS = 3


def _get_max_rollout_attempts() -> int:
    """Read ``NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`` (positive int) or default to 3."""
    raw = os.environ.get("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS")
    if raw is None or raw == "":
        return _DEFAULT_MAX_ROLLOUT_ATTEMPTS
    try:
        n = int(raw)
        if n < 1:
            raise ValueError(f"must be >= 1, got {n}")
        return n
    except (TypeError, ValueError) as e:
        print(
            f"WARNING: could not parse NEMO_GYM_MAX_ROLLOUT_ATTEMPTS={raw!r} ({e}); "
            f"falling back to default {_DEFAULT_MAX_ROLLOUT_ATTEMPTS}.",
            flush=True,
        )
        return _DEFAULT_MAX_ROLLOUT_ATTEMPTS


def _failures_path_for(output_fpath: Path) -> Path:
    """Sidecar path used by the dispatcher and ``_load_from_cache``."""
    return output_fpath.with_name(output_fpath.stem + "_failures.jsonl")


class SharedRolloutCollectionConfig(BaseNeMoGymCLIConfig):
    output_jsonl_fpath: str = Field(description="The output data jsonl file path.")
    num_samples_in_parallel: Optional[int] = Field(
        default=None, description="Limit the number of concurrent samples running at once."
    )
    responses_create_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Overrides for the responses_create_params e.g. temperature, max_output_tokens, etc.",
    )
    upload_rollouts_to_wandb: bool = Field(
        default=True,
        description="Upload the rollouts to W&B. Sometimes this should be off because the rollouts are massive. Default: True",
    )
    disable_aggregation: bool = Field(
        default=False,
        description=(
            "Skip the post-rollout aggregate-metrics computation and file write. "
            "Used when sharding rollouts across multiple jobs that will be aggregated together "
            "afterward by `gym eval aggregate`."
        ),
    )


class E2ERolloutCollectionConfig(SharedRolloutCollectionConfig):
    """
    Spin up all necessary servers and perform a batch of rollout collection using each dataset inside the provided configs.

    Examples:

    ```bash
    gym eval run \
        +output_jsonl_fpath=weather_rollouts.jsonl \
        +num_samples_in_parallel=10
    ```
    """

    split: Union[Literal["train"], Literal["validation"], Literal["benchmark"]]
    reuse_existing_data_preparation: bool = False


class RolloutCollectionConfig(SharedRolloutCollectionConfig):
    """
    Perform a batch of rollout collection.

    Examples:

    ```bash
    gym eval run --no-serve \
        +agent_name=example_single_tool_call_simple_agent \
        +input_jsonl_fpath=weather_query.jsonl \
        +output_jsonl_fpath=weather_rollouts.jsonl \
        +limit=100 \
        +num_repeats=4 \
        +num_samples_in_parallel=10
    ```
    """

    agent_name: Optional[str] = Field(
        default=None,
        description="The agent to collect rollouts from. If not specified, uses agent_ref from each data row.",
    )
    input_jsonl_fpath: str = Field(
        description="The input data source to use to collect rollouts, in the form of a file path to a jsonl file."
    )
    limit: Optional[int] = Field(
        default=None, description="Maximum number of examples to load and take from the input dataset."
    )
    num_repeats: Optional[int] = Field(
        default=None,
        description="The number of times to repeat each example to run. Useful if you want to calculate mean@k e.g. mean@4 or mean@16.",
    )
    num_repeats_add_seed: bool = Field(
        default=False,
        description='When num_repeats > 1, pass a per-rollout "seed" via metadata.extra_body (honored by vLLM model servers).',
    )
    resume_from_cache: bool = Field(
        default=False,
        description="If the same command is run multiple times, check the materialized inputs and current outputs and remove the inputs that have already been run",
    )
    prompt_config: Optional[str] = Field(
        default=None,
        description="Path to a prompt YAML file. Builds responses_create_params.input from the template at rollout time. Mutually exclusive with pre-populated responses_create_params.input in the JSONL data.",
    )

    @property
    def materialized_jsonl_fpath(self) -> Path:
        output_fpath = Path(self.output_jsonl_fpath)
        return output_fpath.with_stem(output_fpath.stem + "_materialized_inputs").with_suffix(".jsonl")


def _rollout_request_debug_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    agent_ref = row.get(AGENT_REF_KEY_NAME) or {}
    summary = {
        TASK_INDEX_KEY_NAME: row.get(TASK_INDEX_KEY_NAME),
        ROLLOUT_INDEX_KEY_NAME: row.get(ROLLOUT_INDEX_KEY_NAME),
        "agent_name": agent_ref.get("name") if isinstance(agent_ref, dict) else None,
    }
    return {k: v for k, v in summary.items() if v is not None}


class RolloutCollectionHelper(BaseModel):
    def _preprocess_rows_from_config(self, config: RolloutCollectionConfig) -> List[Dict]:
        range_iterator = repeat(0)
        if config.limit:
            range_iterator = range(config.limit)
            print(f"Limiting the number of rows to {config.limit}")

        if config.num_repeats_add_seed:
            print(
                "Adding unique `seed` values to each input via metadata.extra_body (only honored by vLLM model servers)"
            )

        if config.agent_name:
            print(f"Using `{config.agent_name}` for rows that do not already have an agent ref")

        if config.responses_create_params:
            print(f"Overriding responses_create_params fields with {config.responses_create_params}")
            responses_create_params_overrides = OmegaConf.to_container(
                OmegaConf.create(config.responses_create_params), resolve=True
            )
        else:
            responses_create_params_overrides = dict()

        num_repeats = config.num_repeats or 1
        if num_repeats:
            print(f"Repeating rows {num_repeats} times (in a pattern of abc to aabbcc)!")

        # Load prompt config if specified
        prompt_cfg = None
        if config.prompt_config:
            prompt_cfg = load_prompt_config(config.prompt_config)
            print(f"Using prompt config: {config.prompt_config}")

        _input_path = Path(config.input_jsonl_fpath)
        if not _input_path.is_absolute():
            _cwd_path = Path.cwd() / _input_path
            _input_path = _cwd_path if _cwd_path.exists() else PARENT_DIR / _input_path
        with open(_input_path) as input_file:
            rows_iterator: Iterator[str] = tqdm(input_file, desc="Reading rows")
            rows_iterator: Iterator[tuple[int, str]] = zip(range_iterator, rows_iterator)
            raw_rows = [(row_idx, row_str, orjson.loads(row_str)) for row_idx, row_str in rows_iterator]

        # Validate and apply prompt config before per-row processing
        if prompt_cfg is not None:
            validate_prompt_compatibility([row for _, _, row in raw_rows], prompt_cfg)
            raw_rows = [(idx, s, apply_prompt_to_row(row, prompt_cfg)) for idx, s, row in raw_rows]

        # For gym eval profile to match rollouts to tasks
        row_to_task_idx: Dict[str, int] = dict()
        task_idx_to_rollout_idx: Dict[int, int] = Counter()
        row_idxs_missing_agent_ref: List[int] = []
        rows: List[Dict] = []
        for row_idx, row_str, row in raw_rows:
            # Resolve agent name
            if config.agent_name:
                row.setdefault(AGENT_REF_KEY_NAME, {"name": config.agent_name})
            elif not row.get(AGENT_REF_KEY_NAME, dict()).get("name"):
                row_idxs_missing_agent_ref.append(row_idx)

            # Responses create params
            row[RESPONSES_CREATE_PARAMS_KEY_NAME] = (
                row[RESPONSES_CREATE_PARAMS_KEY_NAME] | responses_create_params_overrides
            )

            # Resolve task index. Honor a caller-provided value when present (e.g. when an
            # upstream slicer has stamped a globally-stable index across chunks so that
            # subsequent /aggregate_metrics groupby unions chunks correctly); otherwise dedupe
            # identical input rows to the same task index as before.
            if TASK_INDEX_KEY_NAME not in row:
                row[TASK_INDEX_KEY_NAME] = row_to_task_idx.setdefault(row_str, len(row_to_task_idx))

            for _ in range(num_repeats):
                row = deepcopy(row)

                # Resolve rollout index
                row[ROLLOUT_INDEX_KEY_NAME] = task_idx_to_rollout_idx[row[TASK_INDEX_KEY_NAME]]
                task_idx_to_rollout_idx[row[TASK_INDEX_KEY_NAME]] += 1

                if config.num_repeats_add_seed:
                    metadata = row[RESPONSES_CREATE_PARAMS_KEY_NAME].setdefault("metadata", {})
                    extra_body = json.loads(metadata.get("extra_body", "{}"))
                    extra_body["seed"] = row[ROLLOUT_INDEX_KEY_NAME]
                    metadata["extra_body"] = json.dumps(extra_body)

                rows.append(row)

        if row_idxs_missing_agent_ref:
            raise ValueError(
                f"No agent specified for rows {row_idxs_missing_agent_ref}. Either provide +agent_name config or include agent_ref in data."
            )

        return rows

    def _load_from_cache(
        self, config: RolloutCollectionConfig
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[List[str]]]:
        with config.materialized_jsonl_fpath.open() as f:
            original_input_rows = list(map(orjson.loads, f))
        with Path(config.output_jsonl_fpath).open("rb") as f:
            result_strs = [[line.strip()] for line in f]
        results = [orjson.loads(p[0]) for p in result_strs]

        get_key = lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])

        # Successes (and any legacy '-failed' rows written by pre-fix Gym
        # builds) live in the main jsonl. They short-circuit dispatch.
        successes_seen = set(map(get_key, results))

        # Sidecar: one row per non-kill_shaped failure attempt. Count attempts
        # per key + flag terminal rows so chain-hop 2 retries the right ones.
        failures_fpath = _failures_path_for(Path(config.output_jsonl_fpath))
        attempts_by_key: Counter = Counter()
        terminal_keys: set = set()
        if failures_fpath.exists():
            with failures_fpath.open("rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    fr = orjson.loads(line)
                    if TASK_INDEX_KEY_NAME not in fr or ROLLOUT_INDEX_KEY_NAME not in fr:
                        continue
                    k = (fr[TASK_INDEX_KEY_NAME], fr[ROLLOUT_INDEX_KEY_NAME])
                    attempts_by_key[k] += 1
                    if fr.get(NG_TERMINAL_KEY):
                        terminal_keys.add(k)

        max_attempts = _get_max_rollout_attempts()
        maxed_out = {k for k, n in attempts_by_key.items() if n >= max_attempts}
        gated = successes_seen | terminal_keys | maxed_out

        input_rows = [row for row in original_input_rows if get_key(row) not in gated]

        key_to_row = dict(zip(map(get_key, original_input_rows), original_input_rows))
        rows = [key_to_row[get_key(result)] for result in results]

        print(
            f"""Resumed from cache. Found:
- {len(original_input_rows)} original input rows
- {len(rows)} rows already done (in main jsonl)
- {sum(attempts_by_key.values())} prior failure attempts ({len(attempts_by_key)} unique tasks) in sidecar
- {len(terminal_keys)} sidecar-terminal (timeout_exceeded / skipped) → not retried
- {len(maxed_out)} hit max_attempts={max_attempts} → not retried
- {len(input_rows)} rows that still need to be run"""
        )

        return input_rows, rows, results, result_strs

    async def run_from_config(self, config: RolloutCollectionConfig) -> Tuple[List[Dict]]:
        output_fpath = Path(config.output_jsonl_fpath)

        if config.resume_from_cache and config.materialized_jsonl_fpath.exists() and output_fpath.exists():
            (
                input_rows,
                rows,
                results,
                result_strs,
            ) = self._load_from_cache(config)
        else:
            if config.resume_from_cache:
                if not output_fpath.exists():
                    print(f"Skipping resume_from_cache because output_fpath {output_fpath} doesn't exist!")
                if not config.materialized_jsonl_fpath.exists():
                    print(
                        f"Skipping resume_from_cache because materialized_jsonl_fpath {config.materialized_jsonl_fpath} doesn't exist!"
                    )
            else:
                print("Clearing output fpath since `resume_from_cache=False`!")

            rows: List[Dict] = []
            results: List[Dict] = []
            result_strs: List[List[str]] = []

            input_rows = self._preprocess_rows_from_config(config)
            # Returned rows are sorted by (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])

            with config.materialized_jsonl_fpath.open("wb") as f:
                for row in input_rows:
                    f.write(orjson.dumps(row) + b"\n")

            output_fpath.unlink(missing_ok=True)

        semaphore = nullcontext()
        if config.num_samples_in_parallel:
            print(f"Querying with {config.num_samples_in_parallel} concurrent requests")
            semaphore = Semaphore(config.num_samples_in_parallel)

        output_fpath.parent.mkdir(exist_ok=True, parents=True)
        failures_fpath = _failures_path_for(output_fpath)

        pcts_to_print = [20, 40, 60, 80, 90, 95, 98, 99, 100]
        counts_left = Counter(r[AGENT_REF_KEY_NAME]["name"] for r in input_rows)
        results_file = output_fpath.open("ab")
        failures_file = failures_fpath.open("ab")
        for future in self.run_examples(input_rows, semaphore=semaphore):
            row, result = await future

            result[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
            result[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]
            result[AGENT_REF_KEY_NAME] = row[AGENT_REF_KEY_NAME]

            no_persist = bool(result.get(NG_NO_PERSIST_KEY))
            failure_class = result.get(NG_FAILURE_CLASS_KEY)

            rows.append(row)
            results.append(result)
            serialized = orjson.dumps(result)
            result_strs.append([serialized])

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

            current_pct = 100 * len(results) / len(input_rows)
            if pcts_to_print and current_pct >= pcts_to_print[0]:
                while pcts_to_print and current_pct >= pcts_to_print[0]:
                    pcts_to_print.pop(0)

                top_left = counts_left.most_common(5)  # Fix to top 3 for now.
                if top_left:
                    top_left_str = "\n".join(f"{i + 1}. {k}: {v}" for i, (k, v) in enumerate(top_left))
                    # Use tqdm.write here so we can print properly with tqdm being used.
                    tqdm.write(f"Examples left:\n{top_left_str}")

        results_file.close()
        failures_file.close()

        if config.upload_rollouts_to_wandb and get_wandb_run():  # pragma: no cover
            print("Uploading rollouts to W&B. This may take a few minutes if your data is large.")
            get_wandb_run().log({"Rollouts": Table(data=result_strs, columns=["Rollout"])})
        del result_strs

        print("Sorting results to ensure consistent ordering")
        rows.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
        results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))

        # Compute and write aggregate metrics via /aggregate_metrics on each agent server
        if config.disable_aggregation:
            print(
                "Skipping aggregate-metrics computation because disable_aggregation=True. "
                "Run `gym eval aggregate` after all shards finish to compute the global metrics."
            )
            aggregate_metrics_fpath = None
        else:
            print("Computing aggregate metrics")
            aggregate_metrics_fpath = await self._call_aggregate_metrics(results, rows, output_fpath)

        print(f"""Finished rollout collection! View results at:
Fully materialized inputs: {config.materialized_jsonl_fpath}
Rollouts: {output_fpath}
Aggregate metrics: {aggregate_metrics_fpath}""")

        return results

    async def _call_aggregate_metrics(
        self,
        results: List[Dict],
        rows: List[Dict],
        output_fpath: Path,
    ) -> Optional[Path]:
        """Call /aggregate_metrics on each agent server after rollouts complete.

        Writes a single _aggregate_metrics.json with one entry per agent (same shape
        as the old _agent_metrics.json). Returns the file path.
        """
        if not results:
            return None

        # Group results by agent name
        agent_results: Dict[str, List[Dict]] = {}
        for row, result in zip(rows, results):
            agent_name = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
            if not agent_name:
                continue
            agent_results.setdefault(agent_name, []).append(result)

        server_client = self.setup_server_client()

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
                server_name=agent_name,
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
                {
                    f"{agent_name}/{k}": v
                    for k, v in agent_entry["agent_metrics"].items()
                    if isinstance(v, primitive_types)
                }
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

    def run_examples(
        self,
        examples: List[Dict],
        head_server_config: Optional[BaseServerConfig] = None,
        semaphore: Optional[Semaphore] = None,
    ) -> Iterator[Future]:  # pragma: no cover
        """
        We provide this function as a lower level interface for running rollout collection.
        """
        server_client = self.setup_server_client(head_server_config)
        semaphore = semaphore or nullcontext()

        async def _post_subroutine(row: Dict) -> Tuple[Dict, Dict]:
            async with semaphore:
                res = await server_client.post(server_name=row["agent_ref"]["name"], url_path="/run", json=row)
                try:
                    await raise_for_status(res)
                except Exception:
                    if is_global_aiohttp_client_request_debug_enabled():
                        print(
                            "[rollout_collection] /run failed "
                            f"status={getattr(res, 'status', None)} "
                            f"row={json.dumps(_rollout_request_debug_summary(row), sort_keys=True)}",
                            flush=True,
                        )
                    raise
                return row, await get_response_json(res)

        return tqdm.as_completed(
            map(_post_subroutine, examples),
            desc="Collecting rollouts",
            miniters=10,
            total=len(examples),
            maxinterval=60,
        )

    def setup_server_client(
        self, head_server_config: Optional[BaseServerConfig] = None
    ) -> ServerClient:  # pragma: no cover
        server_client = ServerClient.load_from_global_config(head_server_config)

        # We set this rollout global aiohttp client to use the same max connections as the underlying head server global config.
        if not is_global_aiohttp_client_setup():
            set_global_aiohttp_client(
                cfg=GlobalAIOHTTPAsyncClientConfig.model_validate(server_client.global_config_dict)
            )

        return server_client


class RolloutAggregationConfig(BaseNeMoGymCLIConfig):
    """
    Aggregate metrics across rollout shards produced by `gym eval run --no-serve +disable_aggregation=true`.

    Reads every JSONL file matching `input_glob`, computes aggregate metrics by POSTing to each
    agent server's `/aggregate_metrics` endpoint over the global union of records, and writes a
    single `<output_jsonl_fpath stem>_aggregate_metrics.json` next to the rollouts. By default
    also concatenates all shards into `output_jsonl_fpath`.

    Examples:

    ```bash
    gym eval aggregate \
        "+config_paths=[benchmarks/aime24/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
        +input_glob='results/rollouts-rs*-chunk*.jsonl' \
        +output_jsonl_fpath=results/rollouts.jsonl
    ```
    """

    input_glob: str = Field(
        description=(
            "Glob pattern or comma-separated list of glob patterns matching the rollout shards "
            "to aggregate (e.g. 'results/rollouts-rs*-chunk*.jsonl' or "
            "'results/run1/rollouts.jsonl,results/run2/rollouts.jsonl'). Whitespace around "
            "commas is stripped. Duplicate matches across patterns are deduplicated."
        )
    )
    output_jsonl_fpath: str = Field(
        description=(
            "Path used to derive the aggregate-metrics output location "
            "('<stem>_aggregate_metrics.json' next to this path) and, when merge_shards=True, "
            "the merged-rollouts file."
        ),
    )
    merge_shards: bool = Field(
        default=True,
        description="Concatenate the matched shard JSONLs into output_jsonl_fpath alongside the metrics file.",
    )


def _expand_input_glob(input_glob: str) -> List[str]:
    """Expand a glob-or-comma-separated-globs string into a sorted, deduplicated list of paths.

    Examples:
      'results/rollouts.jsonl' -> ['results/rollouts.jsonl'] (if it exists)
      'a/*.jsonl, b/*.jsonl'   -> matches of both patterns, deduplicated
    """
    patterns = [p.strip() for p in input_glob.split(",") if p.strip()]
    seen: Dict[str, None] = {}  # preserve insertion order while deduping
    for pattern in patterns:
        for path in sorted(glob_module.glob(pattern)):
            seen.setdefault(path, None)
    return list(seen)


class RolloutAggregationHelper(BaseModel):
    async def run_from_config(self, config: RolloutAggregationConfig) -> Optional[Path]:
        input_paths = _expand_input_glob(config.input_glob)
        if not input_paths:
            raise FileNotFoundError(f"No shards matched input_glob={config.input_glob!r}")
        print(f"Aggregating {len(input_paths)} shard(s):")
        for p in input_paths:
            print(f"  - {p}")

        results: List[Dict] = []
        for shard_path in input_paths:
            with open(shard_path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    results.append(orjson.loads(line))
        print(f"Loaded {len(results)} rollout record(s) from {len(input_paths)} shard(s)")

        # Sort for deterministic aggregation ordering (matches run_from_config's post-collection sort)
        results.sort(key=lambda r: (r.get(TASK_INDEX_KEY_NAME), r.get(ROLLOUT_INDEX_KEY_NAME)))

        output_fpath = Path(config.output_jsonl_fpath)
        output_fpath.parent.mkdir(parents=True, exist_ok=True)

        if config.merge_shards:
            print(f"Merging shards into {output_fpath}")
            with output_fpath.open("wb") as out:
                for r in results:
                    out.write(orjson.dumps(r) + b"\n")

        # `_call_aggregate_metrics` only inspects each row's AGENT_REF_KEY_NAME, which results already carry.
        helper = RolloutCollectionHelper()
        aggregate_metrics_fpath = await helper._call_aggregate_metrics(results, results, output_fpath)

        print(f"""Finished rollout aggregation! View results at:
Merged rollouts: {output_fpath if config.merge_shards else "<not merged>"}
Aggregate metrics: {aggregate_metrics_fpath}""")

        return aggregate_metrics_fpath


# Backward-compatibility shims (CLI refactor): these CLI entry points moved to `nemo_gym.cli.eval`.
# Re-exported lazily to avoid a circular import; accessing them emits a DeprecationWarning.
from nemo_gym.cli._compat import moved_attr_getter  # noqa: E402


__getattr__ = moved_attr_getter(
    __name__,
    {
        "collect_rollouts": "nemo_gym.cli.eval",
        "aggregate_rollouts": "nemo_gym.cli.eval",
    },
)
