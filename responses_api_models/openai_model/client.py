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
from asyncio import run

from nemo_gym.server_utils import ServerClient


server_client = ServerClient.load_from_global_config()


async def main():
    task_1 = await server_client.post(
        server_name="policy_model",
        url_path="/v1/responses",
        json={"input": "hello"},
    )
    task_2 = await server_client.post(
        server_name="policy_model",
        url_path="/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    print(task_1.json())
    print(task_2.json())


if __name__ == "__main__":
    run(main())
