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
import urllib.request
from asyncio import Future, Semaphore
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import orjson
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator
from tqdm.asyncio import tqdm
from wandb import Table

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
    get_global_config_dict,
    get_response_json,
    is_global_aiohttp_client_setup,
    raise_for_status,
    set_global_aiohttp_client,
)


class SharedRolloutCollectionConfig(BaseNeMoGymCLIConfig):
    output_jsonl_fpath: str = Field(description="The output data jsonl file path.")
    num_samples_in_parallel: Optional[int] = Field(
        default=None,
        description="Limit the number of concurrent samples running at once. Set to null/None for unlimited.",
    )
    responses_create_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Overrides for the responses_create_params e.g. temperature, max_output_tokens, etc.",
    )


class E2ERolloutCollectionConfig(SharedRolloutCollectionConfig):
    """
    Spin up all necessary servers and perform a batch of rollout collection using each dataset inside the provided configs.

    Examples:

    ```bash
    ng_collect_rollouts \
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
    # From a local JSONL file:
    ng_collect_rollouts \
        +agent_name=example_single_tool_call_simple_agent \
        +input_jsonl_fpath=weather_query.jsonl \
        +output_jsonl_fpath=weather_rollouts.jsonl \
        +limit=100 \
        +num_repeats=4 \
        +num_samples_in_parallel=10

    # From a gym URL (fetches all tasks):
    ng_collect_rollouts \
        +agent_name=browser_openai_agent \
        +input_gym_url=https://your-gym-url.com \
        +output_jsonl_fpath=rollouts.jsonl \
        +num_repeats=5

    # From a gym URL (specific tasks):
    ng_collect_rollouts \
        +agent_name=browser_openai_agent \
        +input_gym_url=https://your-gym-url.com \
        "+input_gym_task_id=[TASK-ID-001,TASK-ID-002]" \
        +output_jsonl_fpath=rollouts.jsonl \
        +num_repeats=3
    ```
    """

    agent_name: Optional[str] = Field(
        default=None,
        description="The agent to collect rollouts from. If not specified, uses agent_ref from each data row.",
    )
    input_jsonl_fpath: Optional[str] = Field(
        default=None,
        description="The input data source to use to collect rollouts, in the form of a file path to a jsonl file.",
    )
    input_gym_url: Optional[str] = Field(
        default=None,
        description="Base URL of a gym to fetch tasks from via /api/v1/get_expected_state. Alternative to input_jsonl_fpath.",
    )
    input_gym_task_id: Optional[List[str]] = Field(
        default=None,
        description="If set with input_gym_url, only run specific task ID(s). Use list syntax for multiple: [ID1,ID2]. Errors if any task ID is not found.",
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
        description='When num_repeats > 1, add a "seed" parameter on the Responses create params.',
    )
    resume_from_cache: bool = Field(
        default=False,
        description="If the same command is run multiple times, check the materialized inputs and current outputs and remove the inputs that have already been run",
    )
    prompt_config: Optional[str] = Field(
        default=None,
        description="Path to a prompt YAML file. Builds responses_create_params.input from the template at rollout time. Mutually exclusive with pre-populated responses_create_params.input in the JSONL data.",
    )

    @model_validator(mode="after")
    def validate_input_source(self):
        if not self.input_jsonl_fpath and not self.input_gym_url:
            raise ValueError("Either input_jsonl_fpath or input_gym_url must be provided")
        if self.input_jsonl_fpath and self.input_gym_url:
            raise ValueError("Cannot provide both input_jsonl_fpath and input_gym_url — use one or the other")
        if self.input_gym_task_id and not self.input_gym_url:
            raise ValueError("input_gym_task_id requires input_gym_url to be set")
        return self

    @property
    def materialized_jsonl_fpath(self) -> Path:
        output_fpath = Path(self.output_jsonl_fpath)
        return output_fpath.with_stem(output_fpath.stem + "_materialized_inputs").with_suffix(".jsonl")


def _fetch_gym_tasks(gym_url: str, task_ids: Optional[List[str]] = None) -> List[str]:
    """Fetch tasks from a gym's /api/v1/get_expected_state endpoint.

    Returns a list of JSON strings in the same format as JSONL input lines,
    each containing responses_create_params and verifier_metadata.
    """
    endpoint = f"{gym_url.rstrip('/')}/api/v1/get_expected_state"
    print(f"Fetching tasks from {endpoint} ...")

    req = urllib.request.Request(endpoint, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = orjson.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise ValueError(f"Failed to fetch tasks from {endpoint}: HTTP {e.code} — {e.reason}") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Could not connect to gym at {endpoint}: {e.reason}") from e

    verifiers = payload.get("verifiers", {})
    if not verifiers:
        raise ValueError(f"No verifiers found in response from {endpoint}")

    if task_ids:
        missing_ids = [tid for tid in task_ids if tid not in verifiers]
        if missing_ids:
            available = ", ".join(sorted(verifiers.keys())[:20])
            raise ValueError(
                f"Task ID(s) not found in gym at {gym_url}: {', '.join(missing_ids)}. "
                f"Available task IDs ({len(verifiers)} total): {available}"
            )
        verifiers = {tid: verifiers[tid] for tid in task_ids}

    rows: List[str] = []
    for tid, details in verifiers.items():
        prompt = ""
        if isinstance(details, dict):
            prompt = details.get("task_statement") or details.get("prompt", "")

        start_url = details.get("start_url", gym_url) if isinstance(details, dict) else gym_url

        viewport = details.get("viewport_size") if isinstance(details, dict) else None
        if isinstance(viewport, list) and len(viewport) == 2:
            viewport = {"width": viewport[0], "height": viewport[1]}
        elif not isinstance(viewport, dict):
            viewport = {"width": 1280, "height": 720}

        row = {
            "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
            "verifier_metadata": {
                "task_id": tid,
                "gym_url": gym_url,
                "start_url": start_url,
                "viewport": viewport,
            },
        }
        rows.append(orjson.dumps(row).decode())

    print(f"Fetched {len(rows)} task(s) from gym")
    return rows


class RolloutCollectionHelper(BaseModel):
    def _preprocess_rows_from_config(self, config: RolloutCollectionConfig) -> List[Dict]:
        range_iterator = repeat(0)
        if config.limit:
            range_iterator = range(config.limit)
            print(f"Limiting the number of rows to {config.limit}")

        if config.num_repeats_add_seed:
            print("Adding unique `seed` values to each input")

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

        prompt_cfg = None
        if config.prompt_config:
            prompt_cfg = load_prompt_config(config.prompt_config)
            print(f"Using prompt config: {config.prompt_config}")

        if config.input_gym_url:
            raw_lines = _fetch_gym_tasks(config.input_gym_url, config.input_gym_task_id)
            rows_iterator: Iterator[str] = tqdm(raw_lines, desc="Reading rows from gym")
            rows_iterator: Iterator[tuple[int, str]] = zip(range_iterator, rows_iterator)
            raw_rows = [(row_idx, row_str, orjson.loads(row_str)) for row_idx, row_str in rows_iterator]
        else:
            with open(config.input_jsonl_fpath) as input_file:
                rows_iterator: Iterator[str] = tqdm(input_file, desc="Reading rows")
                rows_iterator: Iterator[tuple[int, str]] = zip(range_iterator, rows_iterator)
                raw_rows = [(row_idx, row_str, orjson.loads(row_str)) for row_idx, row_str in rows_iterator]

        if prompt_cfg is not None:
            validate_prompt_compatibility([row for _, _, row in raw_rows], prompt_cfg)
            raw_rows = [(idx, s, apply_prompt_to_row(row, prompt_cfg)) for idx, s, row in raw_rows]

        # For ng_reward_profile to match rollouts to tasks
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

            # Resolve task index
            row[TASK_INDEX_KEY_NAME] = row_to_task_idx.setdefault(row_str, len(row_to_task_idx))

            for _ in range(num_repeats):
                row = deepcopy(row)

                # Resolve rollout index
                row[ROLLOUT_INDEX_KEY_NAME] = task_idx_to_rollout_idx[row[TASK_INDEX_KEY_NAME]]
                task_idx_to_rollout_idx[row[TASK_INDEX_KEY_NAME]] += 1

                if config.num_repeats_add_seed:
                    row[RESPONSES_CREATE_PARAMS_KEY_NAME]["seed"] = row[ROLLOUT_INDEX_KEY_NAME]

                rows.append(row)

        if row_idxs_missing_agent_ref:
            raise ValueError(
                f"No agent specified for rows {row_idxs_missing_agent_ref}. Either provide +agent_name config or include agent_ref in data."
            )

        return rows

    def _load_from_cache(self, config: RolloutCollectionConfig) -> Tuple[List[Dict], List[Dict]]:
        """Load cache and return (remaining_input_rows, already_completed_rows).

        Deserializes each cached result line to extract index keys (task_index,
        rollout_index) for matching against the original input rows.  Full result
        dicts are not retained — only the key set is kept in memory.
        """
        with config.materialized_jsonl_fpath.open() as f:
            original_input_rows = list(map(orjson.loads, f))

        get_key = lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])

        seen_keys: set = set()
        with Path(config.output_jsonl_fpath).open("rb") as f:
            for line in f:
                result_key = orjson.loads(line)
                seen_keys.add((result_key[TASK_INDEX_KEY_NAME], result_key[ROLLOUT_INDEX_KEY_NAME]))

        input_rows = [row for row in original_input_rows if get_key(row) not in seen_keys]
        rows = [row for row in original_input_rows if get_key(row) in seen_keys]

        print(
            f"""Resumed from cache. Found:
- {len(original_input_rows)} original input rows
- {len(rows)} rows that have already been run
- {len(input_rows)} rows that still need to be run"""
        )

        return input_rows, rows

    async def run_from_config(self, config: RolloutCollectionConfig) -> Tuple[List[Dict]]:
        output_fpath = Path(config.output_jsonl_fpath)

        if config.resume_from_cache and config.materialized_jsonl_fpath.exists() and output_fpath.exists():
            input_rows, rows = self._load_from_cache(config)
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

        pcts_to_print = [20, 40, 60, 80, 90, 95, 98, 99, 100]
        counts_left = Counter(r[AGENT_REF_KEY_NAME]["name"] for r in input_rows)
        results_file = output_fpath.open("ab")
        # Track how many new results we've written in this run (rows from cache are already in rows).
        initial_rows_len = len(rows)
        for future in self.run_examples(input_rows, semaphore=semaphore):
            row, result = await future

            result[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
            result[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]
            result[AGENT_REF_KEY_NAME] = row[AGENT_REF_KEY_NAME]

            # Write immediately and release the result dict so GC can reclaim it.
            # For CUA rollouts each result holds the full trajectory with all screenshots,
            # so accumulating them in-memory would exhaust RAM over large collection runs.
            result_bytes = orjson.dumps(result)
            results_file.write(result_bytes + b"\n")
            results_file.flush()

            rows.append(row)

            counts_left[row[AGENT_REF_KEY_NAME]["name"]] -= 1
            if counts_left[row[AGENT_REF_KEY_NAME]["name"]] <= 0:
                counts_left.pop(row[AGENT_REF_KEY_NAME]["name"])

            newly_completed = len(rows) - initial_rows_len
            current_pct = 100 * newly_completed / len(input_rows)
            if pcts_to_print and current_pct >= pcts_to_print[0]:
                while pcts_to_print and current_pct >= pcts_to_print[0]:
                    pcts_to_print.pop(0)

                top_left = counts_left.most_common(5)  # Fix to top 3 for now.
                if top_left:
                    top_left_str = "\n".join(f"{i + 1}. {k}: {v}" for i, (k, v) in enumerate(top_left))
                    print(f"Examples left:\n{top_left_str}")

        results_file.close()

        # Re-read all results from disk (covers both cached results from resume path and
        # newly written results). Avoids holding large result dicts in memory during collection.
        with output_fpath.open("rb") as f:
            results = [orjson.loads(line) for line in f]

        if get_wandb_run():  # pragma: no cover
            print("Uploading rollouts to W&B. This may take a few minutes if your data is large.")
            result_strs = [[orjson.dumps(r)] for r in results]
            get_wandb_run().log({"Rollouts": Table(data=result_strs, columns=["Rollout"])})

        print("Sorting results to ensure consistent ordering")
        rows.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
        results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))

        # Compute and write aggregate metrics via /aggregate_metrics on each agent server
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
                await raise_for_status(res)
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


def collect_rollouts():  # pragma: no cover
    config = RolloutCollectionConfig.model_validate(get_global_config_dict())
    rch = RolloutCollectionHelper()

    asyncio.run(rch.run_from_config(config))
