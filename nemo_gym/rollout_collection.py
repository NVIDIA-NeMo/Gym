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
import logging
from asyncio import Future, Semaphore
from collections import Counter, defaultdict
from contextlib import nullcontext
from itertools import repeat
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from nemo_gym.config_types import BaseNeMoGymCLIConfig, BaseServerConfig
from nemo_gym.global_config import TASK_INDEX_KEY_NAME
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    get_global_config_dict,
    get_response_json,
    is_global_aiohttp_client_setup,
    raise_for_status,
    set_global_aiohttp_client,
)


if TYPE_CHECKING:
    from resources_servers.genrm_compare.comparison_strategies import ComparisonStrategy

logger = logging.getLogger(__name__)


class RolloutCollectionConfig(BaseNeMoGymCLIConfig):
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
    output_jsonl_fpath: str = Field(description="The output data jsonl file path.")
    limit: Optional[int] = Field(
        default=None, description="Maximum number of examples to load and take from the input dataset."
    )
    num_repeats: Optional[int] = Field(
        default=None,
        description="The number of times to repeat each example to run. Useful if you want to calculate mean@k e.g. mean@4 or mean@16.",
    )
    num_samples_in_parallel: Optional[int] = Field(
        default=None, description="Limit the number of concurrent samples running at once."
    )
    responses_create_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Overrides for the responses_create_params e.g. temperature, max_output_tokens, etc.",
    )


class RolloutCollectionHelper(BaseModel):  # pragma: no cover
    def _load_comparison_strategy_from_global_config(self) -> Optional["ComparisonStrategy"]:
        """Load comparison strategy from global config.

        This automatically detects if a comparison strategy is configured
        in the global config and loads it. No parameters needed - everything
        comes from the global config.

        Expected global config structure:
            comparison_strategy:
                type: GenRMStrategy  # Strategy type
                agent_names: [genrm_simple_agent]
                genrm_compare_server_name: genrm_compare
                policy_model_server_name: policy_model
                num_generations_per_prompt: 16

        Returns:
            ComparisonStrategy if configured, None otherwise
        """
        try:
            global_config = get_global_config_dict()
            strategy_config = global_config.get("comparison_strategy")

            if not strategy_config:
                return None

            strategy_type = strategy_config.get("type")

            if strategy_type == "GenRMStrategy":
                from resources_servers.genrm_compare.comparison_strategies import (
                    GenRMStrategy,
                    GenRMStrategyConfig,
                )

                return GenRMStrategy(
                    GenRMStrategyConfig(
                        agent_names=strategy_config.get("agent_names", ["genrm_simple_agent"]),
                        genrm_compare_server_name=strategy_config.get("genrm_compare_server_name", "genrm_compare"),
                        policy_model_server_name=strategy_config.get("policy_model_server_name", "policy_model"),
                        num_generations_per_prompt=strategy_config.get("num_generations_per_prompt", 16),
                    )
                )
            else:
                logger.warning(f"Unknown comparison strategy type: {strategy_type}")
                return None

        except Exception as e:
            logger.warning(f"Failed to load comparison strategy from global config: {e}")
            return None

    async def run_from_config(self, config: RolloutCollectionConfig):
        range_iterator = repeat(0)
        if config.limit:
            range_iterator = range(config.limit)
            print(f"Limiting the number of rows to {config.limit}!")

        with open(config.input_jsonl_fpath) as input_dataset:
            rows = [row for _, row in zip(range_iterator, map(json.loads, input_dataset))]
        print(f"Found {len(rows)} rows!")

        if config.num_repeats:
            previous_length = len(rows)
            expanded = []
            for task_idx, row in enumerate(rows):
                for _ in range(config.num_repeats):
                    expanded.append({**row, TASK_INDEX_KEY_NAME: task_idx})
            rows = expanded
            print(f"Repeating rows (in a pattern of abc to aabbcc) from {previous_length} to {len(rows)}!")

        semaphore = nullcontext()
        if config.num_samples_in_parallel:
            print(f"Querying with {config.num_samples_in_parallel} concurrent requests")
            semaphore = Semaphore(config.num_samples_in_parallel)

        server_client = self.setup_server_client()

        tqdm_miniters = 10
        print(
            f"The tqdm progress bar will only update every {tqdm_miniters} samples that finish to ensure that you are not being spammed."
        )

        if config.responses_create_params:
            print(f"Overriding responses_create_params fields with {config.responses_create_params}")

        # Validate all rows have an agent specified (either via config or agent_ref in data)
        if not config.agent_name:
            missing_agent_indices = [idx for idx, row in enumerate(rows) if not row.get("agent_ref", {}).get("name")]
            if missing_agent_indices:
                raise ValueError(
                    f"No agent specified for rows {missing_agent_indices}. Either provide +agent_name config or include agent_ref in data."
                )

        metrics = Counter()
        Path(config.output_jsonl_fpath).parent.mkdir(exist_ok=True, parents=True)
        with open(config.output_jsonl_fpath, "a") as f:

            async def _post_coroutine(row: dict) -> None:
                row["responses_create_params"] = row["responses_create_params"] | config.responses_create_params
                # Use config.agent_name if specified, otherwise use agent_ref from the row
                agent_name = config.agent_name or row.get("agent_ref", {}).get("name")
                async with semaphore:
                    response = await server_client.post(server_name=agent_name, url_path="/run", json=row)
                    await raise_for_status(response)
                    result = await get_response_json(response)
                    metrics.update({k: v for k, v in result.items() if isinstance(v, (int, float))})
                    # For ng_profile to match rollouts to tasks
                    if TASK_INDEX_KEY_NAME in row:
                        result[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
                    f.write(json.dumps(result) + "\n")

            await tqdm.gather(*map(_post_coroutine, rows), desc="Collecting rollouts", miniters=tqdm_miniters)

        avg_metrics = {k: v / len(rows) for k, v in metrics.items()}
        avg_metrics.setdefault("reward", 0.0)
        print(json.dumps(avg_metrics, indent=4))

    def run_examples(
        self,
        examples: List[Dict],
        head_server_config: Optional[BaseServerConfig] = None,
    ) -> Iterator[Future]:
        """
        Run rollout collection.

        Automatically detects if a comparison strategy is configured in the global
        config and routes examples accordingly. No explicit strategy parameter needed -
        everything is determined by the global config and the examples' agent_ref.

        When a comparison strategy is configured, samples matching strategy.agent_names
        are processed with generation-only + buffering + comparison, while other
        samples go through the standard agent /run path. Both run in parallel.

        Args:
            examples: List of example dicts to process (must have agent_ref.name)
            head_server_config: Optional server configuration

        Returns:
            Iterator of futures for async completion

        Note:
            The comparison strategy is loaded from global config at:
                comparison_strategy:
                    type: GenRMStrategy
                    agent_names: [...]
                    genrm_compare_server_name: ...
                    policy_model_server_name: ...
                    num_generations_per_prompt: ...
        """
        server_client = self.setup_server_client(head_server_config)

        # Automatically load comparison strategy from global config
        comparison_strategy = self._load_comparison_strategy_from_global_config()

        if comparison_strategy:
            logger.info(f"Using comparison strategy: {comparison_strategy.__class__.__name__}")
            return self._run_with_comparison_strategy(examples, server_client, comparison_strategy)
        else:
            return self._run_standard(examples, server_client)

    def _run_standard(self, examples: List[Dict], server_client: ServerClient) -> Iterator[Future]:
        """Standard rollout collection - each sample through its agent.

        Args:
            examples: List of example dicts
            server_client: Configured server client

        Returns:
            Iterator of futures
        """

        async def _post_subroutine(row: Dict) -> Tuple[Dict, Dict]:
            res = await server_client.post(server_name=row["agent_ref"]["name"], url_path="/run", json=row)
            await raise_for_status(res)
            return row, await get_response_json(res)

        return tqdm.as_completed(
            map(_post_subroutine, examples), desc="Collecting rollouts", miniters=10, total=len(examples)
        )

    def _run_with_comparison_strategy(
        self,
        examples: List[Dict],
        server_client: ServerClient,
        strategy: "ComparisonStrategy",
    ) -> Iterator[Future]:
        """Run with comparison strategy - strategy samples get generation + compare, others get /run.

        This method implements the GRPO workflow where:
        1. Generate N responses per prompt using policy model
        2. Buffer responses by prompt_key (including principle)
        3. When N responses collected, call strategy.compare()
        4. Attach rewards to results

        Args:
            examples: List of example dicts
            server_client: Configured server client
            strategy: Comparison strategy to use

        Returns:
            Iterator of futures
        """
        from resources_servers.genrm_compare.comparison_strategies import (
            extract_conversation_history,
            generate_response,
            get_prompt_key,
        )

        strategy_agent_names = set(strategy.agent_names)
        strategy_samples = []
        standard_samples = []

        for idx, example in enumerate(examples):
            agent_ref = example.get("agent_ref", {})
            agent_name = agent_ref.get("name", "") if isinstance(agent_ref, dict) else ""
            if agent_name in strategy_agent_names:
                strategy_samples.append((idx, example))
            else:
                standard_samples.append((idx, example))

        logger.info(f"Comparison strategy: {len(strategy_samples)} samples, Standard: {len(standard_samples)} samples")

        async def _run_all() -> List[Dict]:
            results = [None] * len(examples)

            async def process_standard():
                async def _do(idx: int, ex: Dict):
                    ex_copy = ex.copy()
                    agent_name = ex_copy.pop("agent_ref")["name"]
                    res = await server_client.post(server_name=agent_name, url_path="/run", json=ex_copy)
                    await raise_for_status(res)
                    results[idx] = await res.json()

                if standard_samples:
                    await asyncio.gather(*[_do(idx, ex) for idx, ex in standard_samples])

            async def process_strategy():
                if not strategy_samples:
                    return
                num_gens = strategy.num_generations_per_prompt
                policy_model = strategy.policy_model_server_name
                prompt_buffers: Dict[str, List[tuple]] = defaultdict(list)
                compare_tasks: List[asyncio.Task] = []
                compared: Set[str] = set()
                lock = asyncio.Lock()

                async def on_gen_complete(idx: int, example: Dict, gen_result: Dict):
                    prompt_key = get_prompt_key(example)
                    async with lock:
                        prompt_buffers[prompt_key].append((idx, example, gen_result))
                        if len(prompt_buffers[prompt_key]) == num_gens and prompt_key not in compared:
                            compared.add(prompt_key)
                            group = prompt_buffers[prompt_key]
                            task = asyncio.create_task(_compare_group(prompt_key, group))
                            compare_tasks.append(task)

                async def _compare_group(prompt_key: str, group: List[tuple]):
                    first_example = group[0][1]
                    conv_history = extract_conversation_history(first_example)
                    # Extract principle from example data for principle-based GenRM
                    principle = first_example.get("principle")

                    # Debug log: show whether GenRM is using principle-based judging
                    if principle:
                        logger.info(f"[GenRM] Judging with PRINCIPLE (len={len(principle)}): {principle[:100]}...")
                    else:
                        logger.info("[GenRM] Judging WITHOUT principle")

                    # Pass raw Response API objects - text extraction happens in genrm_compare
                    response_objs = [gr for _, _, gr in group]
                    rewards, genrm_metrics = await strategy.compare(
                        conv_history, response_objs, server_client, principle=principle
                    )

                    for i, (idx, _, gen_result) in enumerate(group):
                        # Include GenRM metrics in each result so they flow back to NeMo-RL
                        results[idx] = {
                            "response": gen_result,
                            "reward": rewards[i],
                            **{f"genrm_{k}": v for k, v in genrm_metrics.items()},
                        }

                async def gen_and_notify(idx: int, example: Dict):
                    gen_result = await generate_response(example, server_client, policy_model)
                    await on_gen_complete(idx, example, gen_result)

                await asyncio.gather(*[gen_and_notify(idx, ex) for idx, ex in strategy_samples])
                if compare_tasks:
                    await asyncio.gather(*compare_tasks)

            await asyncio.gather(process_standard(), process_strategy())
            return results

        main_future = asyncio.ensure_future(_run_all())

        async def _get_at(idx: int) -> Dict:
            results = await main_future
            return results[idx]

        futures = [asyncio.ensure_future(_get_at(i)) for i in range(len(examples))]
        return tqdm.as_completed(futures, desc="Collecting rollouts", miniters=10, total=len(examples))

    def setup_server_client(self, head_server_config: Optional[BaseServerConfig] = None) -> ServerClient:
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
