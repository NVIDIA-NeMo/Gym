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

import json
from asyncio import run

from nemo_gym.server_utils import ServerClient


async def test_text_game_agent():
    server_client = ServerClient.load_from_global_config()

    # Run a complete game
    task = server_client.post(
        server_name="simple_game_agent",
        url_path="/run",
        json={
            "responses_create_params": {
                "input": [{"role": "user", "content": "Let's play Sudoku!"}],
                "tools": [],
            },
            "clues": 30,
            "scale": 9,
        },
    )

    result = await task

    print("=== RAW RESPONSE DEBUG ===")
    print(f"Status Code: {result.status_code}")
    print(f"Headers: {dict(result.headers)}")
    print(f"Raw Content (as bytes): {result.content}")
    print(f"Raw Content (as text): {result.text}")
    print(f"Content Length: {len(result.content)}")
    print("=========================")

    print("Game Result:")
    print(json.dumps(result.json(), indent=2))


if __name__ == "__main__":
    run(test_text_game_agent())
