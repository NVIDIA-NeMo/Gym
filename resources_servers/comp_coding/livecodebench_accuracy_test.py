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

"""
We use the livecodebench verification logic directly so we don't need to re-implement all the code parsing, test case run, etc ourselves.
The train data we use is fundamentally different from livecodebench however.

Prepare the verification data
```bash
python resources_servers/comp_coding/livecodebench_accuracy_test_prep.py
```

Run the comp coding server via:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/comp_coding/configs/comp_coding.yaml"
ng_run "+config_paths=[${config_paths}]"
```
"""

import json
from asyncio import Semaphore, as_completed, run
from typing import Optional

from tqdm.auto import tqdm

from nemo_gym.server_utils import ServerClient, raise_for_status


async def _single_post(semaphore: Semaphore, server_client: ServerClient, agent_name: str, url_path: str, f) -> dict:
    async with semaphore:
        row = json.loads(next(f))

        # From LCB defaults in lcb_runner/runner/oai_runner.py and the cli args defaults
        row["responses_create_params"] = row["responses_create_params"] | {
            "temperature": 0.2,
            # Changed from max_tokens to max_output_tokens for Responses.
            "max_output_tokens": 2000,
            "top_p": 0.95,
            # Unused in Responses
            # "frequency_penalty": 0,
            # "presence_penalty": 0,
        }

        response = await server_client.post(
            agent_name,
            url_path=url_path,
            json=row,
        )
        await raise_for_status(response)
        result = await response.json()

        expected_reward = row["reward"]
        actual_reward = result["reward"]

        mismatch_str = ""
        if expected_reward != actual_reward:
            mismatch_str = " | MISMATCH!!!!!"
            print(f"Expected reward: {expected_reward} | Actual reward: {actual_reward}{mismatch_str}")

        return result


async def _test_accuracy_helper(
    output_fpath: str, agent_name: str, url_path: str, num_concurrent_calls: Optional[int] = None
) -> None:
    server_client = ServerClient.load_from_global_config()

    if not num_concurrent_calls:
        num_concurrent_calls = server_client.global_config_dict["comp_coding"]["resources_servers"]["comp_coding"][
            "num_processes"
        ]
    semaphore = Semaphore(num_concurrent_calls)
    limit = None

    input_fpath = "resources_servers/comp_coding/data/livecodebench_v5_2024-07-01_2025-02-01_validation.jsonl"
    with open(input_fpath) as f:
        num_rows = sum(1 for _ in tqdm(f, desc="Reading num rows"))

    if limit:
        num_rows = min(num_rows, limit)

    with open(input_fpath) as f:
        tasks = []
        for _ in range(num_rows):
            task = _single_post(semaphore, server_client, agent_name, url_path, f)
            tasks.append(task)

        total_reward = 0.0
        pbar = tqdm(total=len(tasks), desc=f"Verifying (reward={total_reward:.3f})")
        with open(output_fpath, "w") as f:
            for future in as_completed(tasks):
                result = await future
                total_reward += result["reward"]
                f.write(json.dumps(result) + "\n")

                pbar.update()
                pbar.set_description(f"Verifying (reward={total_reward / pbar.n:.3f})")

        print(f"Average reward: {total_reward / num_rows:.3f}")


async def verifier_accuracy_test():
    return await _test_accuracy_helper(
        output_fpath="resources_servers/comp_coding/data/livecodebench_verify_accuracy_results.jsonl",
        agent_name="comp_coding",
        url_path="/verify",
    )


async def e2e_accuracy_test():
    return await _test_accuracy_helper(
        output_fpath="resources_servers/comp_coding/data/livecodebench_e2e_accuracy_results.jsonl",
        agent_name="comp_coding_simple_agent",
        url_path="/run",
        num_concurrent_calls=16,
    )


if __name__ == "__main__":
    run(verifier_accuracy_test())
    # run(e2e_accuracy_test())
