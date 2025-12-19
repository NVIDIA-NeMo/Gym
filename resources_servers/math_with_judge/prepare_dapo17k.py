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


train_ds = load_dataset("YouJiacheng/DAPO-Math-17k-dedup", split="train")

rows = []
for example in train_ds:
    row = {
        "responses_create_params": {"input": example["prompt"]},
        "question": example["prompt"][0]["content"],
        "expected_answer": example["reward_model"]["ground_truth"],
    }
    rows.append(json.dumps(row) + "\n")


with open("data/dapo17k_train.jsonl", "w") as f:
    f.writelines(rows)
