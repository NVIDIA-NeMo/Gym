# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
We use the livecodebench verification logic directly so we don't need to re-implement all the code parsing, test case run, etc ourselves.
The train data we use is fundamentally different from livecodebench however.

Download the verification data (produced by resources_servers/comp_coding/livecodebench_accuracy_test_prep.py):
```bash
ng_upload_dataset_to_gitlab \
    +dataset_name=livecodebench \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/comp_coding/data/livecodebench_v5_2024-07-01_2025-02-01_validation.jsonl
```

Run the comp coding server via:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/comp_coding/configs/comp_coding.yaml"
ng_run "+config_paths=[${config_paths}]"
```
"""

import json
from asyncio import Semaphore, run

from tqdm.auto import tqdm

from nemo_gym.server_utils import ServerClient


async def _single_post(semaphore: Semaphore, server_client: ServerClient, f) -> dict:
    async with semaphore:
        row = json.loads(next(f))
        response = await server_client.post(
            "comp_coding",
            url_path="/verify",
            json=row,
        )
        result = await response.json()

        expected_reward = row["reward"]
        actual_reward = result["reward"]

        mismatch_str = ""
        if expected_reward != actual_reward:
            mismatch_str = " | MISMATCH!!!!!"
            print(f"Expected reward: {expected_reward} | Actual reward: {actual_reward}{mismatch_str}")

        return result


async def test_verifier_accuracy():
    server_client = ServerClient.load_from_global_config()
    semaphore = Semaphore(
        server_client.global_config_dict["comp_coding"]["resources_servers"]["comp_coding"]["num_processes"]
    )
    limit = None

    input_fpath = "resources_servers/comp_coding/data/livecodebench_v5_2024-07-01_2025-02-01_validation.jsonl"
    with open(input_fpath) as f:
        num_rows = sum(1 for _ in tqdm(f, desc="Reading num rows"))

    if limit:
        num_rows = min(num_rows, limit)

    with open(input_fpath) as f:
        tasks = []
        for _ in range(num_rows):
            task = _single_post(semaphore, server_client, f)
            tasks.append(task)

        with open("resources_servers/comp_coding/data/livecodebench_verify_accuracy_results.jsonl", "w") as f:
            for future in tqdm.as_completed(tasks, desc="Verifying"):
                result = await future
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    run(test_verifier_accuracy())
