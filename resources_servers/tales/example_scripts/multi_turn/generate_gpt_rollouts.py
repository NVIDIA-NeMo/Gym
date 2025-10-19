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
import os
from asyncio import run
from pathlib import Path

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
        message_history_full_response = [{"role": "developer", "content": SYSTEM_PROMPT}]
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

            message_history.append({"role": "developer", "content": action})
            message_history_full_response.append({"role": "developer", "content": json.dumps(response)})

            message_history.append({"role": "user", "content": observation})
            message_history_full_response.append({"role": "user", "content": observation})

            if done:
                break

        corrected_message_history = [message_history[0]]
        for turn in message_history[1:]:
            if turn["role"] == "developer":
                corrected_message_history.append({"role": "assistant", "content": turn["content"]})
            else:
                corrected_message_history.append(turn)

        corrected_message_history_full_response = [message_history_full_response[0]]
        for turn in message_history_full_response[1:]:
            if turn["role"] == "developer":
                corrected_message_history_full_response.append({"role": "assistant", "content": turn["content"]})
            else:
                corrected_message_history_full_response.append(turn)

        # Add the score to the conversation turns of the llm:
        for i, message in enumerate(corrected_message_history[1:]):
            if message["role"] == "assistant":
                reward_at_turn = reward_history[i // 2]
                message["reward"] = reward_at_turn

        for i, message in enumerate(corrected_message_history_full_response[1:]):
            if message["role"] == "assistant":
                reward_at_turn = reward_history[i // 2]
                message["reward"] = reward_at_turn

        # Dump to json
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        # Check if the parent directory exists, if not create it
        if not os.path.exists(Path(output_file)):
            os.makedirs(Path(output_file))

        # 'cleaned up' message history:
        with open(output_file + framework + "_clean.jsonl", "w") as f:
            for message in corrected_message_history:
                f.write(json.dumps(message) + "\n")

        # Full response message history:
        with open(output_file + framework + "_full.jsonl", "w") as f:
            for message in corrected_message_history_full_response:
                f.write(json.dumps(message) + "\n")

        print(f"\nGenerated rollout of length {len(message_history) // 2} turns for framework {framework}.")


if __name__ == "__main__":
    run(
        run_multi_turn_loop(
            frameworks=["textworld", "textworld_express", "alfworld", "scienceworld", "jericho"],
            task_no=0,
            split="test",
            max_steps=25,
            seed=0,
            output_file="resources_servers/tales/data/gpt4o_examples/",
        )
    )
