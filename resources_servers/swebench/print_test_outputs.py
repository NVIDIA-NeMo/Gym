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
from argparse import ArgumentParser

import orjson
from tqdm.auto import tqdm


parser = ArgumentParser()
parser.add_argument("--rollout-jsonl", type=str, required=True)
parser.add_argument("--instance-id", type=str, required=True)
args = parser.parse_args()

instance_id_to_row = dict()
with open("benchmarks/swebench/data/swebench_multilingual_benchmark.jsonl") as f:
    for line in f:
        row = orjson.loads(line)
        instance_id_to_row[row["instance_id"]] = row


num = 0
with open(args.rollout_jsonl) as f_in:
    for line in tqdm(f_in):
        row = orjson.loads(line)
        if row["instance_id"] != args.instance_id:
            continue

        print(f"Verification time taken: {row['patch_verification_time_taken']}")

        if row["test_output"].strip():
            to_write = f"Reward: {row['reward']}\n\nTest output:\n{row['test_output']}"
            # to_write = f"Reward: {row['reward']}\n\nTest output:\n{row['test_output']}\n\nModel patch:\n{row['model_patch']}\n\nOpencode stdout:\n{row['opencode_run_stdout']}"
        else:
            to_write = f"Reward: {row['reward']}\n\nOpencode stdout:\n{row['opencode_run_stdout']}"

        sample = instance_id_to_row[args.instance_id]
        sample["patch"] = row["model_patch"]
        with open("temp.jsonl", "wb") as f:
            f.write(orjson.dumps(instance_id_to_row[args.instance_id]) + b"\n")

        with open(f"temp_{num}.log", "w") as f_out:
            f_out.write(to_write)
        num += 1
