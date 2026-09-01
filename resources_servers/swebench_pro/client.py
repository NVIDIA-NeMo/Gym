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

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import ServerClient


EMPTY_RESPONSE = {
    "output": [],
    "id": "",
    "created_at": 0,
    "model": "",
    "object": "response",
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}


async def main() -> None:
    config = get_global_config_dict()
    with open(config["benchmark_jsonl"], encoding="utf-8") as benchmark:
        example = json.loads(next(benchmark))
    example |= {"responses_create_params": {"input": []}, "response": EMPTY_RESPONSE}

    client = ServerClient.load_from_global_config()
    result = await client.post(
        server_name="swebench_pro_resources_server",
        url_path="/verify",
        json=example,
    )
    print(json.dumps(await result.json(), indent=4))


if __name__ == "__main__":
    asyncio.run(main())
