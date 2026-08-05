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

import json
from pathlib import Path

from datasets import load_dataset


def prepare(hf_path: str, output_fpath: Path) -> Path:
    ds = load_dataset(hf_path, split="test")

    with output_fpath.open("w", encoding="utf-8") as fout:
        for row in ds:
            row = row | {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": row["problem_statement"],
                        }
                    ],
                },
                "subset": "verified",
                "split": "test",
            }
            fout.write(json.dumps(row) + "\n")

    print(f"Wrote {len(ds)} problems to {output_fpath}")
    return output_fpath
