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

"""
We use the livecodebench verification logic directly so we don't need to re-implement all the code parsing, test case run, etc ourselves.
The train data we use is fundamentally different from livecodebench however.

Reproduce the accuracy test setting used to test the accuracy of our integration:
```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench
cd LiveCodeBench
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
# Downgrade datasets to match the poetry.lock version.
uv pip install datasets==2.18.0

HF_HOME=.cache \
OPENAI_KEY={your OpenAI API key} \
python -m lcb_runner.runner.main \
    --model gpt-4o-2024-05-13 \
    --scenario codegeneration \
    --evaluate \
    --continue_existing_with_eval \
    --num_process_evaluate 4 \
    --release_version release_v5 \
    --start_date 2024-07-01 \
    --end_date 2025-02-01
```

This is the expected output:
```bash
Downloading builder script: 5.01kB [00:00, 5.57MB/s]
Downloading readme: 3.39kB [00:00, 23.3MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.25G/1.25G [00:11<00:00, 107MB/s]
Downloading data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 713M/713M [00:06<00:00, 107MB/s]
Downloading data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 623M/623M [00:05<00:00, 107MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.20G/1.20G [00:11<00:00, 105MB/s]
Downloading data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 558M/558M [00:05<00:00, 107MB/s]
Downloading data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134M/134M [00:01<00:00, 107MB/s]
Generating test split: 880 examples [00:10, 86.65 examples/s]
Loaded 322 problems
 15%|██████████████████████████████▉                                                                                                                                                                            | 49/322 [04:52<23:04,  5.07s/it]
```

Download the test data using:
```bash
ng_download_dataset_from_gitlab \
    +dataset_name=livecodebench \
    +version=0.0.1 \
    +artifact_fpath=gpt-4o-2024-05-13_Scenario.codegeneration_10_0.2_eval_all.json \
    +output_fpath=resources_servers/comp_coding/data/gpt-4o-2024-05-13_Scenario.codegeneration_10_0.2_eval_all.json

ng_download_dataset_from_gitlab \
    +dataset_name=livecodebench \
    +version=0.0.1 \
    +artifact_fpath=livecodebench_prompts.jsonl \
    +output_fpath=resources_servers/comp_coding/data/livecodebench_prompts.jsonl
```
"""

import json

from tqdm.auto import tqdm


if __name__ == "__main__":
    with open(
        "resources_servers/comp_coding/data/gpt-4o-2024-05-13_Scenario.codegeneration_10_0.2_eval_all.json"
    ) as f:
        eval_outputs = json.load(f)

    with open("resources_servers/comp_coding/data/livecodebench_prompts.jsonl") as f:
        eval_inputs = list(map(json.loads, f))

    with open("resources_servers/comp_coding/data/livecodebench_v5_2024-07-01_2025-02-01_validation.jsonl", "w") as f:
        for eval_input, eval_output in tqdm(zip(eval_inputs, eval_outputs), desc="Examples"):
            assert eval_input["verifier_metadata"]["problem_id"] == eval_output["question_id"]

            for model_output, grade in zip(eval_output["output_list"], eval_output["graded_list"]):
                row = {
                    **eval_input,
                    "reward": 1.0 if grade else 0.0,
                    "response": {
                        "id": "resp_68af8f6e4e2c8194836b47e02e9a36ea0964a4187a9b56c6",
                        "created_at": 1756335982.0,
                        "error": None,
                        "incomplete_details": None,
                        "instructions": None,
                        "metadata": {},
                        "model": "gpt-4.1-2025-04-14",
                        "object": "response",
                        "output": [
                            {
                                "id": "msg_68af8f6ed0e08194b4404bed09bc81190964a4187a9b56c6",
                                "content": [
                                    {
                                        "annotations": [],
                                        "text": model_output,
                                        "type": "output_text",
                                        "logprobs": [],
                                    }
                                ],
                                "role": "assistant",
                                "status": "completed",
                                "type": "message",
                            }
                        ],
                        "parallel_tool_calls": True,
                        "temperature": 1.0,
                        "tool_choice": "auto",
                        "tools": [],
                        "top_p": 1.0,
                        "background": False,
                        "max_output_tokens": None,
                        "max_tool_calls": None,
                        "previous_response_id": None,
                        "prompt": None,
                        "reasoning": {"effort": None, "generate_summary": None, "summary": None},
                        "service_tier": "default",
                        "status": "completed",
                        "text": {"format": {"type": "text"}, "verbosity": "medium"},
                        "top_logprobs": 0,
                        "truncation": "disabled",
                        "usage": {
                            "input_tokens": 539,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 617,
                            "output_tokens_details": {"reasoning_tokens": 0},
                            "total_tokens": 1156,
                        },
                        "user": None,
                        "prompt_cache_key": None,
                        "safety_identifier": None,
                        "store": True,
                    },
                }

                f.write(json.dumps(row) + "\n")
