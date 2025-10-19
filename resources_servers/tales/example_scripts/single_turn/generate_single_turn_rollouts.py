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

import numpy as np
import requests

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient


SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. <action>get lamp</action> (without the <action> tags)."
    " When stuck, try using the `help` command to see what commands are available."
)

PORT = "http://0.0.0.0:8000/v1/responses"


async def run_single_turn_loop(
    frameworks=["textworld", "textworld_express", "alfworld", "scienceworld"],
    split="test",
    max_steps=25,
    output_file="resources_servers/tales/data/single_turn_rollouts/qwen3_30B_A3B",
):
    examples = []
    for framework in frameworks:
        tasks_per_game = {"textworld": 10, "textworld_express": 16, "alfworld": 12, "scienceworld": 30}
        for i in range(tasks_per_game[framework]):
            print(f"Running framework {framework}, task {i}")
            # Start message history with system prompt:
            message_history = [{"role": "developer", "content": SYSTEM_PROMPT}]
            server_client = ServerClient.load_from_global_config()

            # Reset and load the environment:
            reset_task = await server_client.post(
                server_name="tales",
                url_path="/seed_session",
                json={
                    "framework": framework,
                    "task_no": i,
                    "max_episode_steps": max_steps,
                    "seed": 0,
                    "split": split,
                },
            )
            output = await reset_task.json()
            obs = output["observation"]
            message_history.append({"role": "user", "content": obs})
            _ = output["info"]

            # Get the walkthrough for the task:
            walkthrough = output["info"]["extra.walkthrough"]
            session_id = output["session_id"]

            current_score = 0
            walkthrough_history = []
            old_obs = obs
            # Run through the actions in the walkthrough to get the gold trajectory
            for action in walkthrough:
                action_task = await server_client.post(
                    server_name="tales",
                    url_path="/execute_command",
                    json={"command": action, "session_id": session_id},
                )
                output = await action_task.json()
                observation = output["observation"]
                score = output["score"]
                if framework == "textworld":
                    # TextWorld reports total score but we need the delta (reward)
                    reward = score - current_score
                    current_score = score
                else:
                    # Other environments report reward directly
                    reward = score
                _ = output["info"]
                walkthrough_history.append((old_obs, action, score))
                old_obs = observation

            past_acts = []
            # We condense the past action and history to a single input:
            current_history = "/no_think The following are the past observations and actions:\n"
            for obs, act, score in walkthrough_history:
                # When there is a reward, we feed the LLM the history and see if it will predict the action correctly:
                if score != 0 and len(past_acts) < 50:
                    # Generate 16 queries to see what the action distribution looks like
                    all_responses = []
                    all_responses_full = []
                    for j in range(17):
                        history_prompt = (
                            current_history
                            + f"The current observation is:\nObservation: {obs}\nWhat is your action?\n"
                        )
                        request_body = NeMoGymResponseCreateParamsNonStreaming(
                            input=[
                                {"role": "developer", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": history_prompt},
                            ],
                            tools=[],
                        )
                        response = requests.post(PORT, json=request_body.model_dump(exclude_unset=True, mode="json"))

                        # Attempt to extract action from response
                        raw_generation = response.json()["output"][0]["content"][0]["text"]
                        all_responses_full.append(response.json())
                        action = (
                            raw_generation.split("</think>")[-1]
                            .strip()
                            .split("<action>")[-1]
                            .strip()
                            .split("</action>")[0]
                            .strip()
                        )
                        if j == 0:
                            print("Input to LLM:")
                            print(history_prompt)
                            print("LLM Response:")
                            print(raw_generation)
                            print(f"Extracted Action: {action}")

                        all_responses.append(action)

                    # Group the responses into only the unique actions
                    uniq_acts, uniq_counts = np.unique(all_responses, return_counts=True)
                    print("All responses:", all_responses)
                    act_reward_dict = {}
                    for u_act in uniq_acts:
                        # For each unique action, fast-forward the environment to the step and see if it results in a reward:

                        # Fast-Forward the environment to the step before this one:
                        # Reset and load the environment:
                        reset_task = await server_client.post(
                            server_name="tales",
                            url_path="/seed_session",
                            json={
                                "framework": framework,
                                "task_no": i,
                                "max_episode_steps": max_steps,
                                "seed": 0,
                                "split": split,
                            },
                        )
                        output = await reset_task.json()
                        print("Past acts to fast-forward:", past_acts)
                        for p_act in past_acts:
                            action_task = await server_client.post(
                                server_name="tales",
                                url_path="/execute_command",
                                json={"command": p_act, "session_id": session_id},
                            )
                            output = await action_task.json()
                            observation = output["observation"]
                            score = output["score"]
                            if framework == "textworld":
                                # TextWorld reports total score but we need the delta (reward)
                                reward = score - current_score
                                current_score = score
                            else:
                                # Other environments report reward directly
                                reward = score
                            _ = output["info"]

                        # Pass the action to the fast-forwarded environment and see if its accepted:
                        action_task = await server_client.post(
                            server_name="tales",
                            url_path="/execute_command",
                            json={"command": p_act, "session_id": session_id},
                        )
                        output = await action_task.json()
                        observation = output["observation"]
                        score = output["score"]
                        if framework == "textworld":
                            # TextWorld reports total score but we need the delta (reward)
                            reward = score - current_score
                            current_score = score
                        else:
                            # Other environments report reward directly
                            reward = score
                        act_reward_dict[u_act] = reward

                    # Print out the total percent of actions that were actually correct.
                    total_correct = 0
                    for k, u_act in enumerate(uniq_acts):
                        if u_act == act:
                            total_correct += uniq_counts[k]
                    print(f"Action '{act}' was predicted {total_correct} out of 16 times.")
                    examples.append(
                        {
                            "framework": framework,
                            "task_no": i,
                            "current_obs_prompt": history_prompt,
                            "correct_action": act,
                            "predicted_actions": all_responses,
                            "action_rewards": act_reward_dict,
                            "all_responses_full": all_responses_full,
                            "percent_correct": total_correct / 16.0,
                        }
                    )
                    print(f"{len(examples)} examples collected so far.\n\n")

                # Add the last obs and act to the current history:
                current_history += f"Observation: {obs}\nAction: <action>{act}</action>\n"
                past_acts.append(act)

            # Dump to json
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            # Check if the parent directory exists, if not create it
            output_directory = output_file
            if not os.path.exists(Path(output_directory)):
                os.makedirs(Path(output_directory))

            # 'cleaned up' message history:
            with open(output_directory + "/examples_clean.jsonl", "w") as f:
                for example in examples:
                    filtered_example = {}
                    for key in [
                        "framework",
                        "task_no",
                        "current_obs_prompt",
                        "correct_action",
                        "predicted_actions",
                        "action_rewards",
                        "percent_correct",
                    ]:
                        filtered_example[key] = example[key]

                    f.write(json.dumps(filtered_example) + "\n")

            # Full response message history:
            with open(output_directory + "/examples_full.jsonl", "w") as f:
                for example in examples:
                    f.write(json.dumps(example) + "\n")

            print(f"Exported {len(examples)} examples to {output_directory}")


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
            run(run_single_turn_loop())
        elif response.status_code == 404:
            print("Error querying vllm actor model: Please make sure the model server is running.")
        else:
            print(f"Error querying vllm actor model: {response.text}")
