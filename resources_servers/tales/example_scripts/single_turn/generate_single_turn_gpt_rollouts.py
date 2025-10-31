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

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient


SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " When stuck, try using the `help` command to see what commands are available."
)

base_response_create_params = NeMoGymResponseCreateParamsNonStreaming(
    input=[
        {
            "role": "developer",
            "content": (
                "You are playing a text-based game and your goal is to finish it with the highest score."
                " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
                " When stuck, try using the `help` command to see what commands are available."
            ),
        },
    ],
)


async def run_single_turn_loop(
    frameworks, tasks, split: str, max_steps: int, seed: int, prompts: int, output_file: str
):
    for framework in frameworks:
        for task in tasks:
            # Seed message history with system prompt:
            server_client = ServerClient.load_from_global_config()
            reset_task = await server_client.post(
                server_name="tales",
                url_path="/seed_session",
                json={
                    "framework": framework,
                    "task_no": task,
                    "max_episode_steps": max_steps,
                    "seed": seed,
                    "split": split,
                },
            )
            output = await reset_task.json()

            obs = output["observation"]
            _ = output["info"]
            session_id = output["session_id"]
            # Use tdqm to show progress bar
            done = False
            examples = []
            prompt_response_pairs = []
            prompt_response_pairs_full_response = []
            current_history = "The following are the past observations and actions:\n"
            old_obs = obs
            for _ in tqdm.tqdm(range(25)):
                history_prompt = (
                    current_history + f"The current observation is:\nObservation: {obs}\nWhat is your action?\n"
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": history_prompt},
                ]
                query_model = await server_client.post(
                    server_name="policy_model",
                    url_path="/v1/chat/completions",
                    json={
                        "messages": messages,
                    },
                )
                response = await query_model.json()

                prompt_response_pairs.append((history_prompt, response["choices"][0]["message"]["content"]))
                prompt_response_pairs_full_response.append((messages, response))

                action = response["choices"][0]["message"]["content"]

                action_task = await server_client.post(
                    server_name="tales",
                    url_path="/execute_command",
                    json={"command": action, "session_id": session_id},
                )
                output = await action_task.json()
                obs = output["observation"]
                _ = output["score"]
                done = output["done"]
                _ = output["info"]

                current_history += f"Observation: {old_obs}\nAction: {action}\n"
                old_obs = obs

                examples.append(
                    {
                        "responses_create_params": {
                            "input": [
                                {"content": SYSTEM_PROMPT, "role": "developer"},
                                {"content": history_prompt, "role": "user"},
                            ]
                        }
                    }
                )

                if done:
                    break

            # Dump to json
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            # Check if the parent directory exists, if not create it
            if not os.path.exists(Path(output_file)):
                os.makedirs(Path(output_file))

            # Grab 'prompts' number of prompt-response pairs from the generated rollout
            clean_message_examples = []
            selected_examples = []
            for i in range(prompts):
                idx = i % len(prompt_response_pairs)
                history_prompt, response_content = prompt_response_pairs[idx]
                clean_message_examples.append({"input": history_prompt, "output": response_content})

                selected_examples.append(history_prompt)

            # 'cleaned up' message history:
            with open(output_file + "gpt4o_single_turn_examples/examples_clean.jsonl", "w") as f:
                for message in clean_message_examples:
                    f.write(json.dumps(message) + "\n")

            example_strs = []
            for history_prompt in selected_examples:
                example = base_response_create_params.model_copy(
                    update={
                        "input": base_response_create_params.input
                        + [NeMoGymEasyInputMessage(role="user", content=history_prompt)]
                    }
                )
                example_strs.append(
                    json.dumps({"responses_create_params": example.model_dump(exclude_unset=True)}) + "\n"
                )

            with open("resources_servers/tales/data/example_test.jsonl", "w") as f:
                f.writelines(example_strs)

            print(f"Saved {prompts} examples to {output_file}")


if __name__ == "__main__":
    run(
        run_single_turn_loop(
            frameworks=["textworld"],
            tasks=[0],
            split="test",
            max_steps=25,
            seed=0,
            prompts=5,
            output_file="resources_servers/tales/data/",
        )
    )
