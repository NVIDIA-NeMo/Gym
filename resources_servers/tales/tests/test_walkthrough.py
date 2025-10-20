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


async def test_walkthrough(
    frameworks=["textworld", "textworld_express", "alfworld", "scienceworld"],
    split="test",
):
    server_client = ServerClient.load_from_global_config()
    for framework in frameworks:
        server_client = ServerClient.load_from_global_config()

        # Reset and load the environment:
        # Use seed env to get info about the framework and available tasks:
        seed_env = await server_client.post(
            server_name="tales",
            url_path="/seed_session",
            json={
                "framework": framework,
                "task_no": 0,
                "max_episode_steps": 1000,
                "seed": 0,
                "split": split,
            },
        )
        output = await seed_env.json()
        num_tasks = output["available_tasks"]
        for i in range(num_tasks):
            print(f"Running framework {framework}, task {i}")

            reset_task = await server_client.post(
                server_name="tales",
                url_path="/seed_session",
                json={
                    "framework": framework,
                    "task_no": i,
                    "max_episode_steps": 1000,
                    "seed": 0,
                    "split": split,
                },
            )
            output = await reset_task.json()
            _ = output["observation"]
            info = output["info"]
            max_score = info["max_score"]

            # Get the walkthrough for the task:
            walkthrough = output["info"]["extra.walkthrough"]
            session_id = output["session_id"]

            # Run through the actions in the walkthrough to get the gold trajectory
            print(f"Walkthrough has {len(walkthrough)} actions")
            check_won = True
            if len(walkthrough) > 100:
                check_won = False

            steps = 0
            total_score = 0
            for action in walkthrough:
                action_task = await server_client.post(
                    server_name="tales",
                    url_path="/execute_command",
                    json={"command": action, "session_id": session_id},
                )
                output = await action_task.json()
                _ = output["observation"]
                score = output["score"]
                if framework == "textworld":
                    if total_score != score:
                        total_score = score
                else:
                    total_score += score

                _ = output["done"]
                info = output["info"]
                if steps >= 100 and not check_won:
                    # For long walkthroughs, only play first 100 steps and report score fraction
                    total_score = float(total_score)
                    score_fraction = total_score / max_score if max_score > 0 else 0
                    print(f"Walkthrough exceeded 100 steps. Score fraction after 100 steps: {score_fraction:.2f}")
                    break
                steps += 1

            # Assert that following the walkthrough results in info["won"] = True if we are checking for win
            if check_won:
                if "won" in info:
                    assert info.get("won", False) is True, f"Failed to win the game for {framework} task {i}"
                else:
                    # If 'won' isn't in info, we need to just ensure we reached max score
                    assert total_score == max_score, f"Failed to reach max score for {framework} task {i}"


if __name__ == "__main__":
    run(test_walkthrough())
