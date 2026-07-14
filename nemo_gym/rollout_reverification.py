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
from asyncio import Future, Semaphore
from collections import Counter
from contextlib import nullcontext
from itertools import repeat
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import orjson
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm
from wandb import Table

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.config_types import BaseNeMoGymCLIConfig, BaseServerConfig, ConfigError, ConfigPathNotFoundError
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    get_wandb_run,
)
from nemo_gym.rollout_collection import _failures_path_for
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    get_response_json,
    is_global_aiohttp_client_request_debug_enabled,
    is_global_aiohttp_client_setup,
    raise_for_status,
    set_global_aiohttp_client,
)


NG_FAILURE_CLASS_KEY = "_ng_failure_class"
NG_NO_PERSIST_KEY = "_ng_no_persist"
NG_TERMINAL_KEY = "_ng_failure_terminal"


class RolloutReverificationConfig(BaseNeMoGymCLIConfig):
    # to do - we provide description 2 times here - once in the config main.py and once in the field
    # to do can we use Path already here?
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
    judge_failed_only: bool = Field(
        default=False,
        description=(
            "Only re-verify rollouts whose judge call previously failed; successful rows are copied through unchanged."
        ),
    )
    disable_aggregation: bool = Field(
        default=False,
        description=(
            "Skip the post-rollout aggregate-metrics computation and file write. "
            "Used when sharding rollouts across multiple jobs that will be aggregated together "
            "afterward by `gym eval aggregate`."
        ),
    )
    limit: Optional[int] = Field(default=None, description="Maximum number of examples to re-verify.")


# to do add validation if I decide to do it here
def _resolve_input_path(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    if not input_path.is_absolute():
        _cwd_path = Path.cwd() / input_path
        _input_path = _cwd_path if _cwd_path.exists() else PARENT_DIR / input_path
    if not _input_path.is_file():
        raise ConfigPathNotFoundError(
            f"Given input file not found: '{input_path}'. Check it is spelled correctly and exists."
        )
    return _input_path


class RolloutReverificationHelper(BaseModel):
    def _validate_input_paths(self, config: RolloutReverificationConfig) -> None:
        materialized_inputs_jsonl_fpath = Path(config.materialized_inputs_jsonl_fpath)
        rollouts_jsonl_fpath = Path(config.rollouts_jsonl_fpath)
        if not materialized_inputs_jsonl_fpath.exists():
            raise ConfigPathNotFoundError(
                f"Materialized inputs JSONL file {materialized_inputs_jsonl_fpath} does not exist!"
            )
        if not rollouts_jsonl_fpath.exists():
            raise ConfigPathNotFoundError(f"Rollouts JSONL file {rollouts_jsonl_fpath} does not exist!")

    def _yield_inputs_and_rollouts_paired(
        self, materialized_inputs_jsonl_fpath: Path, rollouts_jsonl_fpath: Path
    ) -> Iterator[Tuple[Dict, Dict]]:
        inputs_by_key = {}
        with open(materialized_inputs_jsonl_fpath) as m_f:
            for line in m_f:
                r = orjson.loads(line)
                inputs_by_key[(r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])] = r
        # to do - change to use namedtuple
        # to do - tqdm or other progress bar already used in Gym
        with open(rollouts_jsonl_fpath) as r_f:
            for line in r_f:  # never holds the whole file
                rollout_row = orjson.loads(line)
                input_row = inputs_by_key.get((rollout_row[TASK_INDEX_KEY_NAME], rollout_row[ROLLOUT_INDEX_KEY_NAME]))
                if input_row is None:
                    raise ConfigError(f"No matching materialized input row found for rollout row {rollout_row}")
                yield (input_row, rollout_row)

    def _count_file_lines(self, fpath: Path) -> int:
        with open(fpath) as f:
            return sum(1 for _ in f)

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

    def _run_examples(
        self,
        examples: List[Dict],
        head_server_config: Optional[BaseServerConfig] = None,
        semaphore: Optional[Semaphore] = None,
    ) -> Iterator[Future]:  # pragma: no cover
        """
        We provide this function as a lower level interface for running rollout collection.
        """
        resource_server_client = self.setup_resource_server_client(head_server_config)
        semaphore = semaphore or nullcontext()

        async def _post_subroutine(row: Dict) -> Tuple[Dict, Dict]:
            async with semaphore:
                res = await resource_server_client.post(
                    server_name=row["agent_ref"]["name"], url_path="/verify", json=row
                )
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

    def _prepare_rows_from_config(self, config: RolloutReverificationConfig) -> List[Dict]:
        range_iterator = repeat(0)
        if config.limit:
            range_iterator = range(config.limit)
            print(f"Limiting the number of rows to {config.limit}")
        # to do - what about agent name?

        materialized_inputs_jsonl_fpath = _resolve_input_path(config.materialized_inputs_jsonl_fpath)
        rollouts_jsonl_fpath = _resolve_input_path(config.rollouts_jsonl_fpath)
        # to do - add validation for output_fpath and failures_fpath
        output_fpath = Path(config.output_jsonl_fpath)
        failures_fpath = _failures_path_for(output_fpath)

        for input_row, rollout_row in self._yield_inputs_and_rollouts_paired(
            materialized_inputs_jsonl_fpath, rollouts_jsonl_fpath
        ):
            payload = input_row | {"response": rollout_row["response"]}

        pcts_to_print = [20, 40, 60, 80, 90, 95, 98, 99, 100]
        counts_left = self._count_file_lines(
            rollouts_jsonl_fpath
        )  # to do Assume only one agent for now, so counts_left = Counter(r[AGENT_REF_KEY_NAME]["name"] for r in input_rows)
        results_file = output_fpath.open("ab")
        failures_file = failures_fpath.open("ab")
        for future in self._run_examples(input_rows, semaphore=semaphore):
            row, result = await future

            result[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
            result[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]
            result[AGENT_REF_KEY_NAME] = row[AGENT_REF_KEY_NAME]
            if SKILLS_REF_KEY_NAME in row:
                result[SKILLS_REF_KEY_NAME] = row[SKILLS_REF_KEY_NAME]

            no_persist = bool(result.get(NG_NO_PERSIST_KEY))
            failure_class = result.get(NG_FAILURE_CLASS_KEY)

            rows.append(row)
            results.append(result)
            serialized = orjson.dumps(result)
            result_strs.append([serialized])

    async def run_from_config(self, config: RolloutReverificationConfig) -> Tuple[List[Dict]]:
        materialized_inputs_jsonl_fpath = Path(config.materialized_inputs_jsonl_fpath)
        rollouts_jsonl_fpath = Path(config.rollouts_jsonl_fpath)
        output_fpath = Path(config.output_jsonl_fpath)

        self._validate_input_paths(config)

        rows_to_reverify = self._prepare_rows_from_config(config)
        # semaphore = nullcontext()  # to do add semaphore if I decide to do it here

        from nemo_gym.rollout_collection import _failures_path_for

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
            if SKILLS_REF_KEY_NAME in row:
                result[SKILLS_REF_KEY_NAME] = row[SKILLS_REF_KEY_NAME]

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


def loads_jsonl_line(raw, fpath, line_no: int):
    """Parse one JSONL line, raising a clean `ConfigError` (naming file + line) on malformed JSON."""
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise ConfigError(f"Malformed JSON in '{fpath}' at line {line_no}: {e}") from e


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
            raise ConfigPathNotFoundError(f"No shards matched input_glob={config.input_glob!r}")
        print(f"Aggregating {len(input_paths)} shard(s):")
        for p in input_paths:
            print(f"  - {p}")

        results: List[Dict] = []
        for shard_path in input_paths:
            with open(shard_path, "rb") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    results.append(loads_jsonl_line(line, shard_path, line_no))
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
