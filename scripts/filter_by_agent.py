# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


parser = ArgumentParser()
parser.add_argument("--input-jsonl", type=str, required=True)
parser.add_argument("--output-jsonl", type=str, default="temp.jsonl")
parser.add_argument("--agent", type=str, required=True)
args = parser.parse_args()

print(f"Writing results to {args.output_jsonl}")
with open(args.input_jsonl) as f_in, open(args.output_jsonl, "w") as f_out:
    for line in f_in:
        row = orjson.loads(line)
        if row.get("agent_ref", dict()).get("name") != args.agent:
            continue

        f_out.write(line)
