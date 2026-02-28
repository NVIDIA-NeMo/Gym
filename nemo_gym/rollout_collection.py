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
import asyncio
import json
from asyncio import Future, Semaphore
from collections import Counter
from contextlib import nullcontext
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from nemo_gym.config_types import BaseNeMoGymCLIConfig, BaseServerConfig
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    get_global_config_dict,
    is_global_aiohttp_client_setup,
    raise_for_status,
    set_global_aiohttp_client,
)


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

    agent_name: str = Field(description="The agent to collect rollouts from.")
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
    # pragma: no cover
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
            rows = list(chain.from_iterable(repeat(row, config.num_repeats) for row in rows))
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

        metrics = Counter()
        skipped_count = 0
        successful_count = 0

        output_path = Path(config.output_jsonl_fpath)
        errors_fpath = output_path.parent / "errors.json"
        schema_errors: List[Dict[str, Any]] = []

        with open(config.output_jsonl_fpath, "ab") as f:

            async def _post_coroutine(row: dict) -> None:
                nonlocal skipped_count, successful_count

                row_id = row.get("id", "unknown")

                # Safe merge even if either side is None/missing
                row_params = row.get("responses_create_params") or {}
                cfg_params = config.responses_create_params or {}
                row["responses_create_params"] = {**row_params, **cfg_params}

                async with semaphore:
                    response = await server_client.post(server_name=config.agent_name, url_path="/run", json=row)
                    await raise_for_status(response)
                    result = await response.json()

                    validation_results = result.get("validation_results", [])
                    schema_validation_errors = [
                        vr
                        for vr in validation_results
                        if vr.get("instruction") == "schema_validation" and vr.get("status") == "Failed"
                    ]
                    language_compatibility_errors = [
                        vr
                        for vr in validation_results
                        if vr.get("instruction") == "language_compatibility"
                        and vr.get("status") in ("Failed", "Skipped")
                    ]
                    skip_errors = schema_validation_errors or language_compatibility_errors

                    if skip_errors:
                        skipped_count += 1
                        schema_errors.append({"id": row_id, "errors": [err.get("message") for err in skip_errors]})
                        return

                    successful_count += 1

                    line = json.dumps(result, ensure_ascii=False, default=str) + "\n"
                    f.write(line.encode("utf-8"))

                    metrics.update({k: v for k, v in result.items() if isinstance(v, (int, float))})

            await tqdm.gather(*map(_post_coroutine, rows), desc="Collecting rollouts", miniters=tqdm_miniters)

        if schema_errors:
            err_str = json.dumps(schema_errors, indent=2, ensure_ascii=False, default=str)
            with open(errors_fpath, "wb") as ef:
                ef.write(err_str.encode("utf-8"))
            print(f"\n{skipped_count} rollout(s) skipped due to validation errors. See {errors_fpath}")

        print(f"\n{successful_count} rollout(s) completed successfully.")

        if successful_count > 0:
            avg_metrics = {k: v / successful_count for k, v in metrics.items()}
        else:
            avg_metrics = {}
        avg_metrics.setdefault("reward", 0.0)
        print(json.dumps(avg_metrics, indent=4, ensure_ascii=False, default=str))

    def run_examples(
        self, examples: List[Dict], head_server_config: Optional[BaseServerConfig] = None
    ) -> Iterator[Future]:
        """
        We provide this function as a lower level interface for running rollout collection.
        """
        server_client = self.setup_server_client(head_server_config)

        async def _post_subroutine(row: Dict) -> Tuple[Dict, Dict]:
            res = await server_client.post(server_name=row["agent_ref"]["name"], url_path="/run", json=row)
            await raise_for_status(res)
            return row, await res.json()

        return tqdm.as_completed(
            map(_post_subroutine, examples), desc="Collecting rollouts", miniters=10, total=len(examples)
        )

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
