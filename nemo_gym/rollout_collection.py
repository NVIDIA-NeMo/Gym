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
from asyncio import Future, Semaphore
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import orjson
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm
from wandb import Table

from nemo_gym.config_types import BaseNeMoGymCLIConfig, BaseServerConfig
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    RESPONSES_CREATE_PARAMS_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    get_wandb_run,
)
from nemo_gym.reward_profile import RewardProfiler
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
        default=None, description="Limit the number of concurrent samples running at once."
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

    split: Union[Literal["train"], Literal["validation"]]


class RolloutCollectionConfig(SharedRolloutCollectionConfig):
    """
    Perform a batch of rollout collection.

    Examples:

    ```bash
    ng_collect_rollouts \
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
        description='When num_repeats > 1, add a "seed" parameter on the Responses create params.',
    )
    resume_from_cache: bool = Field(
        default=False,
        description="If the same command is run multiple times, check the materialized inputs and current outputs and remove the inputs that have already been run",
    )

    @property
    def materialized_jsonl_fpath(self) -> Path:
        output_fpath = Path(self.output_jsonl_fpath)
        return output_fpath.with_stem(output_fpath.stem + "_materialized_inputs").with_suffix(".jsonl")


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

        num_repeats = config.num_repeats or 1
        if num_repeats:
            print(f"Repeating rows {num_repeats} times (in a pattern of abc to aabbcc)!")

        input_file = open(config.input_jsonl_fpath)
        rows_iterator: Iterator[str] = input_file
        rows_iterator: Iterator[str] = tqdm(rows_iterator, desc="Reading rows")
        rows_iterator: Iterator[tuple[int, str]] = zip(range_iterator, rows_iterator)

        # For ng_reward_profile to match rollouts to tasks
        row_to_task_idx: Dict[str, int] = dict()
        task_idx_to_rollout_idx: Dict[int, int] = Counter()
        row_idxs_missing_agent_ref: List[int] = []
        rows: List[Dict] = []
        for row_idx, row_str in rows_iterator:
            row = orjson.loads(row_str)

            # Resolve agent name
            if config.agent_name:
                row.setdefault(AGENT_REF_KEY_NAME, {"name": config.agent_name})
            elif not row.get(AGENT_REF_KEY_NAME, dict()).get("name"):
                row_idxs_missing_agent_ref.append(row_idx)

            # Responses create params
            row[RESPONSES_CREATE_PARAMS_KEY_NAME] = (
                row[RESPONSES_CREATE_PARAMS_KEY_NAME] | config.responses_create_params
            )

            # Resolve task index
            row[TASK_INDEX_KEY_NAME] = row_to_task_idx.setdefault(row_str, len(row_to_task_idx))

            for repeat_idx in range(num_repeats):
                row = deepcopy(row)

                # Resolve rollout index
                row[ROLLOUT_INDEX_KEY_NAME] = task_idx_to_rollout_idx[row[TASK_INDEX_KEY_NAME]]
                task_idx_to_rollout_idx[row[TASK_INDEX_KEY_NAME]] += 1

                if config.num_repeats_add_seed:
                    row[RESPONSES_CREATE_PARAMS_KEY_NAME]["seed"] = row[ROLLOUT_INDEX_KEY_NAME]

                rows.append(row)

        input_file.close()

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

        seen_rows = set(map(get_key, results))
        input_rows = [row for row in original_input_rows if get_key(row) not in seen_rows]

        key_to_row = dict(zip(map(get_key, original_input_rows), original_input_rows))
        rows = [key_to_row[get_key(result)] for result in results]

        print(
            f"""Resumed from cache. Found:
- {len(original_input_rows)} original input rows
- {len(rows)} rows that have already been run
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
            rows: List[Dict] = []
            results: List[Dict] = []
            result_strs: List[List[str]] = []

            input_rows = self._preprocess_rows_from_config(config)
            # Returned rows are sorted by (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])

            with config.materialized_jsonl_fpath.open("wb") as f:
                for row in input_rows:
                    f.write(orjson.dumps(row) + b"\n")

            print("Clearing output fpath since `resume_from_cache=False`!")
            output_fpath.unlink(missing_ok=True)

        semaphore = nullcontext()
        if config.num_samples_in_parallel:
            print(f"Querying with {config.num_samples_in_parallel} concurrent requests")
            semaphore = Semaphore(config.num_samples_in_parallel)

        output_fpath.parent.mkdir(exist_ok=True, parents=True)

        results_file = output_fpath.open("ab")
        for future in self.run_examples(input_rows, semaphore=semaphore):
            row, result = await future

            result[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
            result[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]

            rows.append(row)
            results.append(result)
            result_strs.append([orjson.dumps(result)])
            results_file.write(result_strs[-1][0] + b"\n")
        results_file.close()

        if get_wandb_run():  # pragma: no cover
            get_wandb_run().log({"Rollouts": Table(data=result_strs, columns=["Rollout"])})
        del result_strs

        # Sort to ensure consistent ordering
        rows.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
        results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))

        rp = RewardProfiler()
        group_level_metrics, agent_level_metrics = rp.profile_from_data(rows, results)
        reward_profiling_fpath, agent_level_metrics_fpath = rp.write_to_disk(
            group_level_metrics, agent_level_metrics, output_fpath
        )

        if get_wandb_run():  # pragma: no cover
            agent_level_metrics_to_log = dict()
            for agent_metrics in agent_level_metrics:
                agent_name = agent_metrics[AGENT_REF_KEY_NAME]["name"]
                for key, value in agent_metrics.items():
                    agent_level_metrics_to_log[f"{agent_name}/{key}"] = value

                agent_level_metrics_to_log.pop(f"{agent_name}/{AGENT_REF_KEY_NAME}")

            get_wandb_run().log(agent_level_metrics_to_log)

        print(f"""Finished rollout collection! View results at:
Fully materialized inputs: {config.materialized_jsonl_fpath}
Rollouts: {output_fpath}
Reward profiling outputs: {reward_profiling_fpath}
Agent-level metrics: {agent_level_metrics_fpath}""")

        return results

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
            map(_post_subroutine, examples), desc="Collecting rollouts", miniters=10, total=len(examples)
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
