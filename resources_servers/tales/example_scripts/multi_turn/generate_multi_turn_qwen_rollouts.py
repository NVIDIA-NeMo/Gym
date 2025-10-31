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

import requests
import tqdm

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient


SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " When stuck, try using the `help` command to see what commands are available."
)

PORT = "http://0.0.0.0:8000/v1/responses"


def scrub_action(original_str):
    # This function just scrubs the input for anything that might cause the json to error out (bad characters) or actions that might crash the backend.
    # For now, we just remove any non-ascii characters and limit the length to 100 characters.
    # Remove non-ascii characters
    cleaned_str = "".join(char for char in original_str if 32 <= ord(char) <= 126)
    # Limit length to 100 characters
    if len(cleaned_str) > 100:
        cleaned_str = cleaned_str[:100]

    special_chars = [
        "\\",
        "/",
        "<",
        ">",
        "|",
        "*",
        "?",
        '"',
        "'",
        "`",
        "$",
        "#",
        "&",
        ";",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "%",
        "@",
        "+",
        "=",
        "-",
        ":",
        ",",
        ".",
        "!",
        "^",
        "action",
    ]
    for char in special_chars:
        cleaned_str = cleaned_str.replace(char, "")
        try:
            # Test if it's valid UTF-8
            _ = cleaned_str.encode("utf-8")
        except UnicodeEncodeError:
            # Convert to bytes first
            print("Encoded error tripped, original string: ", cleaned_str)
            cleaned_str = cleaned_str.encode("latin-1")
    return cleaned_str


async def run_multi_turn_loop(frameworks, tasks, split, max_steps, seeds, output_file: str):
    for framework in frameworks:
        for task_no in tasks:
            for seed in seeds:
                print(f"Running framework {framework}, task {task_no}, seed {seed}")
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
                    request_body = NeMoGymResponseCreateParamsNonStreaming(
                        input=message_history,
                        tools=[],
                    )
                    response = requests.post(PORT, json=request_body.model_dump(exclude_unset=True, mode="json"))

                    # Attempt to extract action from response
                    raw_generation = response.json()["output"][0]["content"][0]["text"]
                    action = (
                        raw_generation.split("</think>")[-1]
                        .strip()
                        .split("<action>")[-1]
                        .strip()
                        .split("</action>")[0]
                        .strip()
                    )

                    # We scrub the command to make sure it won't cause some sort of injection attack/crash the backend.
                    action = scrub_action(action)

                    action_task = await server_client.post(
                        server_name="tales",
                        url_path="/execute_command",
                        json={"command": action, "session_id": session_id},
                    )
                    output = await action_task.json()
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
                    message_history_full_response.append({"role": "developer", "content": response.json()})

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
                        corrected_message_history_full_response.append(
                            {"role": "assistant", "content": turn["content"]}
                        )
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
                output_directory = output_file + f"{framework}_task{task_no}_seed{seed}/"
                if not os.path.exists(Path(output_directory)):
                    os.makedirs(Path(output_directory))

                # 'cleaned up' message history:
                with open(output_directory + framework + "_clean.jsonl", "w") as f:
                    for message in corrected_message_history:
                        f.write(json.dumps(message) + "\n")

                # # Full response message history:
                # with open(output_directory + framework + "_full.jsonl", "w") as f:
                #     for message in corrected_message_history_full_response:
                #         f.write(json.dumps(message) + "\n")

                print(f"\nGenerated rollout of length {len(message_history) // 2} turns for framework {framework}.")


if __name__ == "__main__":
    try:
        request_body = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Hi how are you?"}],
            tools=[],
        )
        response = requests.post(PORT, json=request_body.model_dump(exclude_unset=True, mode="json"))
    except Exception as e:
        print("Error querying vllm actor model: Please make sure host is correct.")
        print("Exception:", e)
        response = ""

    if response:
        if response.status_code == 200:
            run(
                run_multi_turn_loop(
                    frameworks=["textworld", "textworld_express", "alfworld", "scienceworld", "jericho"],
                    tasks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    split="test",
                    max_steps=25,
                    seeds=[11, 22, 33, 44, 55, 66, 77, 88, 99, 1010],
                    output_file="resources_servers/tales/data/multi_turn_rollouts/",
                )
            )

            # run(run_multi_turn_loop(frameworks=['textworld'], tasks=[1,2,3], split="test", max_steps=25, seeds=[11,22],
            #                         output_file="resources_servers/tales/data/multi_turn_rollouts/"))
        elif response.status_code == 404:
            print("Error querying vllm actor model: Please make sure the model server is running.")
        else:
            print(f"Error querying vllm actor model: {response.text}")
