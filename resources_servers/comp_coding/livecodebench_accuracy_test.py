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
from asyncio import run

from tqdm.auto import tqdm

from nemo_gym.server_utils import ServerClient


async def main():
    server_client = ServerClient.load_from_global_config()
    limit = 10

    with open("resources_servers/comp_coding/data/livecodebench_v5_2024-07-01_2025-02-01_validation.jsonl") as f:
        expected_total_reward = 0.0
        tasks = []
        for _, row in zip(range(limit), f):
            row = json.loads(row)
            task = server_client.post(
                "comp_coding",
                url_path="/verify",
                json=row,
            )
            tasks.append(task)

            expected_total_reward += row["reward"]

        actual_total_reward = 0.0
        with open("resources_servers/comp_coding/data/livecodebench_verify_accuracy_results.jsonl", "w") as f:
            for future in tqdm.as_completed(tasks, desc="Verifying"):
                response = await future
                result = await response.json()
                f.write(json.dumps(result) + "\n")

                actual_total_reward += result["reward"]

        expected_average_reward = expected_total_reward / len(tasks)
        actual_average_reward = actual_total_reward / len(tasks)
        print(f"""Expected average reward: {expected_average_reward:.3f}
Actual average reward: {actual_average_reward:.3f}""")


if __name__ == "__main__":
    run(main())
