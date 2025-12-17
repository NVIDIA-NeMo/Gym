# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path

PROMPT_TEMPLATE = """Solve the following math problem step by step. Put your answer inside \\boxed{{}}.

{problem}

Remember to put your answer inside \\boxed{{}}."""


def prepare_aime25_data():
    repo_root = Path(__file__).parent.parent.parent
    input_file = repo_root / "aime25.jsonl"
    output_dir = Path(__file__).parent / "data"

    output_dir.mkdir(exist_ok=True)

    print(f"Reading from {input_file}")
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples")

    processed_samples = []
    for sample in samples:
        problem = sample['problem']
        expected_answer = sample['expected_answer']
        sample_id = sample['id']

        prompt = PROMPT_TEMPLATE.format(problem=problem)
        processed_sample = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "problem": problem,
            "expected_answer": expected_answer,
            "id": sample_id
        }
        processed_samples.append(processed_sample)

    # using all samples for both train and validation at the moment
    train_samples = processed_samples
    validation_samples = processed_samples
    example_samples = processed_samples[:5]

    train_file = output_dir / "train.jsonl"
    print(f"Writing {len(train_samples)} training samples to {train_file}")
    with open(train_file, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')

    validation_file = output_dir / "validation.jsonl"
    print(f"Writing {len(validation_samples)} validation samples to {validation_file}")
    with open(validation_file, 'w') as f:
        for sample in validation_samples:
            f.write(json.dumps(sample) + '\n')

    example_file = output_dir / "example.jsonl"
    print(f"Writing {len(example_samples)} example samples to {example_file}")
    with open(example_file, 'w') as f:
        for sample in example_samples:
            f.write(json.dumps(sample) + '\n')

    print("Data preparation complete!")

if __name__ == "__main__":
    prepare_aime25_data()
