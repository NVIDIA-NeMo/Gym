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

import tqdm

from nemo_gym.server_utils import ServerClient


SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " When stuck, try using the `help` command to see what commands are available."
)


async def run_multi_turn_loop(frameworks, task_no: int, split: str, max_steps: int, seed: int, output_file: str):
    for framework in frameworks:
        # Seed message history with system prompt:
        message_history = [{"role": "developer", "content": SYSTEM_PROMPT}]
        reward_history = []
        server_client = ServerClient.load_from_global_config()
        reset_task = await server_client.post(
            server_name="tales",
            url_path="/seed_session",
            json={
                "framework": framework,
                "task_no": task_no,
                "max_episode_steps": max_steps,
                "seed": seed,
                "split": split,
            },
        )
        output = await reset_task.json()

        obs = output["observation"]
        message_history.append({"role": "user", "content": obs})
        _ = output["info"]
        session_id = output["session_id"]
        # Use tdqm to show progress bar
        current_score = 0
        done = False
        for _ in tqdm.tqdm(range(25)):
            query_model = await server_client.post(
                server_name="policy_model",
                url_path="/v1/chat/completions",
                json={
                    "messages": message_history,
                },
            )
            response = await query_model.json()
            print(response)

            action = response["choices"][0]["message"]["content"]

            print("Attempting action:", action)

            action_task = await server_client.post(
                server_name="tales",
                url_path="/execute_command",
                json={"command": action, "session_id": session_id},
            )
            output = await action_task.json()
            print("Full output:", output)
            observation = output["observation"]
            score = output["score"]
            done = output["done"]
            _ = output["info"]

            if score != current_score:
                reward_history.append(score - current_score)
                current_score = score
            else:
                reward_history.append(0)

            message_history.append({"role": "assistant", "content": action})
            message_history.append({"role": "user", "content": observation})

            if done:
                break


if __name__ == "__main__":
    run(
        run_multi_turn_loop(
            frameworks=["textworld"],
            task_no=0,
            split="test",
            max_steps=25,
            seed=0,
            output_file="resources_servers/tales/data/gpt4o_examples/",
        )
    )
