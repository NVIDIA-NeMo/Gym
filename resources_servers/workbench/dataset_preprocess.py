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

# Run as `python resources_servers/workbench/dataset_preprocess.py --split test``
import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pandas as pd
from datasets import load_dataset

from resources_servers.workbench.utils import get_tools


@dataclass
class Sample:
    create_params: str
    reference: Any


class WorkbenchEnvironmentSamples:
    @dataclass
    class WorkbenchSample(Sample):
        category: str
        environment_name: str

    HARDCODED_CURRENT_TIME = pd.to_datetime("2023-11-30T23:59:00")

    REASONING_PROMPT = """## Reasoning:
Below is a reasoning template to guide your thinking process as you solve the problem. Make sure the reasoning steps in your thought process before </think> strictly follow the template and are in the same order as the template. You should label each step and not skip any steps in the template.

### Reasoning Template:
1. Disambiguate the user's goal and intentions. Explicitly list all possibilities of the user's intentions and reason about each of their plausibilities.
2. Analyze the current state of the system. List all variables that can be or have been changed, and internalize the relevance or each variable to the user's goal.
3. Consider the possible actions that can be taken. Consider if more information is needed, or if direct action can be taken.
4. Create a plan for tool use to interact with the system, and/or formulate a chat response to be presented to the user.
5. Execute based on the previous reasoning."""

    SYS_PROMPT = (
        f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, {HARDCODED_CURRENT_TIME.date()} "
        f"and the current time is {HARDCODED_CURRENT_TIME.time()}. Remember the current date and time when answering queries. "
        "Meetings must not start before 9am or end after 6pm."
    )

    SYS_PROMPT_REASONING = (
        f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, {HARDCODED_CURRENT_TIME.date()} "
        f"and the current time is {HARDCODED_CURRENT_TIME.time()}. Remember the current date and time when answering queries. "
        "Meetings must not start before 9am or end after 6pm."
        f"\n\n{REASONING_PROMPT}"
    )
    toolkits = [
        "email",
        "calendar",
        "analytics",
        "project_management",
        "customer_relationship_manager",
    ]

    def __init__(self, prompt_to_use: str = "SYS_PROMPT"):
        self.tool_env = get_tools(self.toolkits)

        # Set the correct system prompt based on the argument
        selected_prompt = getattr(self, prompt_to_use)

        self.base_create_params = dict(
            input=[{"role": "system", "content": selected_prompt}],
            tools=self.tool_env.get("schemas"),
            parallel_tool_calls=False,
            temperature=1.0,
        )

    def get_samples(self, split):
        dataset = load_dataset("Nexusflow/250319-workbench-fulleval", split=split)

        processed_samples = []

        for d in dataset:
            # convert into create params
            create_params = deepcopy(self.base_create_params)
            create_params["input"].append(
                {
                    "role": "user",
                    "content": d["problem"],
                }
            )

            ground_truth = json.loads(d["solution"])  # json loads ground truths/solutions

            processed_samples.append(
                self.WorkbenchSample(
                    create_params=create_params,
                    reference=ground_truth,
                    category=d["category"],
                    environment_name=d["environment_name"],
                )
            )
        return processed_samples


def main():
    parser = argparse.ArgumentParser(description="Generate samples for Workbench environment.")
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["default", "reasoning"],
        default="default",
        help="Choose the system prompt to use: 'default' or 'reasoning'. Default is 'default'.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Kind of data being created - train or test",
    )
    args = parser.parse_args()

    # Determine the system prompt based on the argument
    prompt_to_use = "SYS_PROMPT" if args.prompt == "default" else "SYS_PROMPT_REASONING"

    benchmark = WorkbenchEnvironmentSamples(prompt_to_use=prompt_to_use)
    processed_samples = benchmark.get_samples(split=args.split)

    if not processed_samples:
        print("No samples were generated. Exiting.")
        return

    output_filename = f"resources_servers/workbench/data/{args.split}.jsonl"

    with open(output_filename, "w") as f:
        for i, sample in enumerate(processed_samples):
            record = {
                "id": i,
                "responses_create_params": sample.create_params,
                "ground_truth": sample.reference,
                "category": sample.category,
                "environment_name": sample.environment_name,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Successfully created {output_filename}.")


if __name__ == "__main__":
    main()
