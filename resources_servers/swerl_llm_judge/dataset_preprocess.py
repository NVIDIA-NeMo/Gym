# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import os
import json
from typing import Any, Iterable, Optional

from resources_servers.swerl_llm_judge.utils import (
    extract_filenames,
    create_instance_obj,
)

from resources_servers.swerl_llm_judge.prompts import *

def write_jsonl(rows: Iterable[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_row(
    problem_statement: str,
    choices: dict[str, str],
    answer_letter: str,
    *,
    code_context: dict[str, str] = None,
    instance_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_split: Optional[str] = None,
    grading_mode: str = "strict",
    metadata: Optional[dict[str, Any]] = None,
    prompt: Optional[str] = None,
) -> dict:
    """Build a dataset row shaped as a `SWEJudgeRunRequest`.

    Produces a dict that validates as `SWEJudgeRunRequest` used by the SWE judge server:
    - responses_create_params: OpenAI-style request with the prompt text only (no metadata)
    - metadata (top-level, optional): arbitrary dict; pass-through only
    - options (top-level): list of dicts mapping a single letter to option patch, e.g. `[{"A": "patch_A"}, {"B": "patch_B"}]`
    - expected_answer (top-level): single uppercase letter identifying the correct choice
    - grading_mode (top-level): selector for the verifier parsing rules (either `"lenient"` or `"strict"`. strict mode only allows one letter in the solution block.)
    - instance_id (top-level, optional): passthrough identifier for the underlying instance
    - dataset_name (top-level, optional): passthrough identifier for the dataset
    - dataset_split (top-level, optional): passthrough identifier for the dataset split
    """
    if not choices:
        raise ValueError("choices must be a non-empty list")

    options_list = [{letter: patch} for letter, patch in choices.items()]
    if prompt is None:
        if code_context is None:
            relevant_files = set()
            for patch in choices.values():
                relevant_files.update(extract_filenames(patch))
                
            instance_obj = create_instance_obj(instance_id, dataset_name, dataset_split, repo_playground='./repo_playground')
            code_context = {file: '\n'.join(instance_obj.python_files[file]['text']) for file in relevant_files}
        prompt_list = [META_JUDGE_SOLUTION_PREMISE,
                    "<issue>",
                    problem_statement,
                    "</issue>",
                    "<relevant_files>",
                    "\n".join(f"[start of {file}]\n{code}\n[end of {file}]" for file, code in code_context.items()),
                    "</relevant_files>",
                    "<choices>",
                    "\n".join(f"{letter}: {patch}" for letter, patch in choices.items()),
                    "</choices>",
                    ]
        prompt = '\n'.join(prompt_list)

    row: dict = {
        # Required by BaseRunRequest
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        },
        # Fields from SWEJudgeRunRequest used for grading
        "options": options_list,
        "expected_answer": answer_letter.strip().upper(),
        "grading_mode": grading_mode,
        "instance_id": instance_id,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
    }

    row["metadata"] = metadata if metadata is not None else {}

    return row


if __name__ == "__main__":  # pragma: no cover
    # Minimal example demonstrating how to build and write a tiny dataset.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(cur_dir, "data"), exist_ok=True)
    
    instance_id = 'astropy__astropy-12907'
    dataset_name = 'princeton-nlp/SWE-bench_Verified'
    dataset_split = 'test'
    from datasets import load_dataset
    dataset = load_dataset(dataset_name, split=dataset_split)
    example = dataset.filter(lambda x: x['instance_id'] == instance_id)[0]
    problem_statement = example['problem_statement']
    choices = {'A': example['patch'], 'B': example['test_patch']}
    rows = [
        build_row(
            problem_statement=problem_statement,
            choices=choices,
            answer_letter="A",
            instance_id=instance_id,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        )
    ]
    
    write_jsonl(rows, os.path.join(cur_dir, "data/swerl_llm_judge_example.jsonl"))
