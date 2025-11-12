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

from datasets import load_dataset


val_ds = load_dataset("HuggingFaceH4/aime_2024", split="train")

rows = []
for example in val_ds:
    row = {
        "responses_create_params": {
            "input": [
                {
                    "role": "system",
                    "content": "Your task is to solve a math problem.  Make sure to put the answer (and only the answer) inside \\boxed{}.",
                },
                {
                    "role": "user",
                    "content": example["problem"],
                },
            ]
        },
        "question": example["problem"],
        "expected_answer": example["answer"],
    }
    rows.append(json.dumps(row) + "\n")


with open("data/aime24_validation.jsonl", "w") as f:
    f.writelines(rows)
