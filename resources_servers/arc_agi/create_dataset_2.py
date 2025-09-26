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
from pathlib import Path

def format_grid(grid):
    return '\n'.join([' '.join(map(str, row)) for row in grid])

def create_arc_v2_prompt(task_data, task_id):
    prompt = f"You are solving ARC-AGI-2 task {task_id}.\n\n"
    prompt += "Here are the training examples that demonstrate the pattern:\n\n"

    for i, example in enumerate(task_data["train"]):
        prompt += f"Example {i+1}:\n"
        prompt += "Input:\n"
        prompt += format_grid(example["input"])
        prompt += "\n\nOutput:\n"
        prompt += format_grid(example["output"])
        prompt += "\n\n"

    test_input = task_data["test"][0]["input"]
    prompt += "Now solve this test case following the same pattern:\n"
    prompt += "Test Input:\n"
    prompt += format_grid(test_input)
    prompt += "\n\nProvide your solution as a 2D array inside \\boxed{} in this exact format: \\boxed{[[row1],[row2],...]}"
    prompt += "\nFor example: \\boxed{[[1,2,3],[4,5,6],[7,8,9]]}"
    # prompt += "\nNote: You have 2 trials to get the correct answer."

    return prompt

def create_datasets():
    training_dir = Path("../../ARC-AGI-2/data/training")
    evaluation_dir = Path("../../ARC-AGI-2/data/evaluation")

    Path("data").mkdir(exist_ok=True)

    training_dataset = []
    print(f"Processing {len(list(training_dir.glob('*.json')))} training tasks...")

    for task_file in sorted(training_dir.glob("*.json")):
        task_id = task_file.stem

        with open(task_file) as f:
            task_data = json.load(f)

        prompt = create_arc_v2_prompt(task_data, task_id)

        expected_output = task_data["test"][0]["output"]
        test_input = task_data["test"][0]["input"]

        entry = {
            "responses_create_params": {
                "input": [{"role": "user", "content": prompt}]
            },
            "train": task_data["train"],
            "test_input": test_input,
            "expected_output": expected_output,
            "task_id": task_id
        }

        training_dataset.append(entry)

    training_output_file = Path("data/arc_agi_2_training.jsonl")
    with open(training_output_file, 'w') as f:
        for entry in training_dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"Created training dataset with {len(training_dataset)} tasks at {training_output_file}")

    evaluation_dataset = []
    print(f"Processing {len(list(evaluation_dir.glob('*.json')))} evaluation tasks...")

    for task_file in sorted(evaluation_dir.glob("*.json")):
        task_id = task_file.stem

        with open(task_file) as f:
            task_data = json.load(f)

        prompt = create_arc_v2_prompt(task_data, task_id)

        expected_output = task_data["test"][0]["output"]
        test_input = task_data["test"][0]["input"]

        entry = {
            "responses_create_params": {
                "input": [{"role": "user", "content": prompt}]
            },
            "train": task_data["train"],
            "test_input": test_input,
            "expected_output": expected_output,
            "task_id": task_id
        }

        evaluation_dataset.append(entry)

    evaluation_output_file = Path("data/arc_agi_2_evaluation.jsonl")
    with open(evaluation_output_file, 'w') as f:
        for entry in evaluation_dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"Created evaluation dataset with {len(evaluation_dataset)} tasks at {evaluation_output_file}")

    example_output_file = Path("data/example_2.jsonl")
    with open(example_output_file, 'w') as f:
        for entry in evaluation_dataset[:5]:
            f.write(json.dumps(entry) + '\n')

    print(f"Created example dataset with 5 tasks at {example_output_file}")

if __name__ == "__main__":
    create_datasets()