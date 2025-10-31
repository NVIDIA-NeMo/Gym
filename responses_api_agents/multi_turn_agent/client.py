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
import asyncio
from asyncio import run

from nemo_gym.server_utils import ServerClient


async def process_single(task):
    result = await task
    output = (await result.json())["output"]
    return output


async def run_multi_turn_loop():
    server_client = ServerClient.load_from_global_config()
    task = server_client.post(
        server_name="tales",
        url_path="/seed_session",
        json={},
    )
    async with await task as result:
        output = await result.json()
        print("output", output)
    await asyncio.sleep(0.5)
    print("post async output")
    print(output["observation"])
    session_id = output["session_id"]
    print("\n\n\n")
    walkthrough = output["info"]["extra.walkthrough"]
    # In an actual training loop, you would be querying a model for the action. Here, we just use the actions from the walkthrough to make sure its actually working
    for action in walkthrough:
        task = server_client.post(
            server_name="tales",
            url_path="/execute_command",
            json={"command": action, "session_id": session_id},
        )
        async with await task as result:
            output = await result.json()
            print("output", output)
        await asyncio.sleep(0.5)


run(run_multi_turn_loop())
