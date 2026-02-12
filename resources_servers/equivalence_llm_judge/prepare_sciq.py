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

from datasets import load_dataset


ds = load_dataset("allenai/sciq", split="validation")

rows = []
for example in ds:
    # Build a simple prompt: prepend instruction string, no options included
    prefix = "Answer the following question. Make sure to put the final answer (and only the final answer) inside \\boxed{}.\n\n"
    user_content = prefix + example["question"]

    row = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
        },
        "question": example["question"],
        "expected_answer": example["correct_answer"],
    }
    rows.append(json.dumps(row) + "\n")


os.makedirs("data", exist_ok=True)
with open("data/sciq_validation.jsonl", "w") as f:
    f.writelines(rows)
