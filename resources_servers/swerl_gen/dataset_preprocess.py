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
from datasets import load_dataset

from resources_servers.swerl_gen.utils import (
    extract_filenames,
    get_content,
)

from resources_servers.swerl_gen.prompts import *

MODIFY_SCRIPT_COMMANDS = {
    'pandas': ('python -m pip install', 'delete'),
    'dask': ('rm -rf ~/.config/dask', 'add'),
    'dvc': ('rm -rf ~/.config/dvc', 'add'),
}

def write_jsonl(rows: Iterable[dict], out_path: str) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
def read_jsonl(in_path: str) -> list[dict]:
    rows = {}
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[f"{row['instance']['instance_id']}-{row['mode']}"] = row
    return rows
            
def get_singularity_image_path(instance_id, singularity_base_dir, dataset_name: str) -> str:
    """Get the singularity image path for the given instance."""
    if dataset_name == 'princeton-nlp/SWE-bench_Verified' or dataset_name == 'nebius/SWE-rebench':
        docker_instance_id = instance_id.replace('__', '_1776_')
    elif dataset_name == 'SWE-Gym/SWE-Gym':
        docker_instance_id = instance_id.replace('__', '_s_')
    else:
        raise ValueError(f"Invalid source: {dataset_name}")
    return f'{singularity_base_dir}sweb.eval.x86_64.{docker_instance_id}.sif'


def build_row(
    instance: dict[str, Any],
    *,
    eval_script_dir: str,
    image_dir: str,
    prompt_type: str,
    dataset_name: Optional[str] = None,
    dataset_split: Optional[str] = None,
    prompt: Optional[str] = None,
    relevant_file_contents: Optional[dict] = None,
    generate_image_path: Optional[bool] = True,
    repo_playground: Optional[str] = './repo_playground',
) -> dict:
    """Build a dataset row shaped as a `SWEGenRunRequest`.

    Produces a dict that validates as `SWEGenRunRequest` used by the SWE gen server:
    - responses_create_params: OpenAI-style request with the prompt text only (no metadata)
    - metadata: dictionary with keys: relevant_file_contents, remove_repo_name, image
    - instance: dictionary with keys: instance_id, repo, setup_script, test_script, regression_script, PASS_TO_PASS, FAIL_TO_PASS, patch
    - dataset_name (top-level, optional): passthrough identifier for the dataset
    - dataset_split (top-level, optional): passthrough identifier for the dataset split
    """
    if not instance:
        raise ValueError("instance must be a non-empty dictionary")

    instance_id = instance.get("instance_id")
    patch = instance.get("patch")
    problem_statement = instance.get("problem_statement")
    
    if not instance_id:
        raise ValueError("instance must have an instance_id key")
    if not patch:
        raise ValueError("instance must have a patch key")
    if not problem_statement:
        raise ValueError("instance must have a problem_statement key")

    def _script_path(suffix: str) -> str:
        return os.path.join(eval_script_dir, f"{instance_id}{suffix}")

    def _load_script(path: str, kind: str) -> str:
        if not os.path.exists(path):
            raise ValueError(
                f"{kind} script not found at {path}. "
                "Run gen_eval_scripts.py to generate the scripts."
            )
        with open(path, "r", encoding="utf-8") as f:
            script = f.read()
        assert script.startswith("#!/bin/bash"), f"{kind} script at {path} must start with #!/bin/bash"
        return script
    
    def _modify_script_delete_command(script: str, command: str) -> str:
        lines = script.splitlines()
        new_lines = []
        for line in lines:
            if command in line:
                continue
            new_lines.append(line)
        script = "\n".join(new_lines)
        return script
    
    def _modify_script(script: str, new_command: str) -> str:
        if new_command in script:
            return script
        lines = script.splitlines()
        if lines and lines[0].startswith('#!') and 'bash' in lines[0]:
            lines = [lines[0], new_command] + lines[1:]
        else:
            lines = [new_command] + lines
        script = "\n".join(lines)
        return script

    regression_script_path = _script_path("_regression.sh")
    setup_script_path = _script_path(".sh")
    test_script_path = _script_path("_test.sh")

    if generate_image_path:
        image_path = get_singularity_image_path(instance_id, image_dir, dataset_name)
    else:
        image_path = image_dir
    if not os.path.exists(image_path):
        print(
            f"Warning: Singularity image not found at {image_path}. "
            "Cannot run the instance on this server."
        )
        return None

    instance["regression_script"] = _load_script(regression_script_path, "Regression")
    instance["setup_script"] = _load_script(setup_script_path, "Setup")
    instance["test_script"] = _load_script(test_script_path, "Test")
    for repo in MODIFY_SCRIPT_COMMANDS:
        if repo in instance_id:
            command_to_modify, action = MODIFY_SCRIPT_COMMANDS[repo]
            if action == 'delete':
                instance["regression_script"] = _modify_script_delete_command(instance["regression_script"], command_to_modify)
                instance["setup_script"] = _modify_script_delete_command(instance["setup_script"], command_to_modify)
            elif action == 'add':
                instance["regression_script"] = _modify_script(instance["regression_script"], command_to_modify)
                instance["setup_script"] = _modify_script(instance["setup_script"], command_to_modify)
            break

    if prompt is None:
        relevant_python_files = sorted(extract_filenames(patch))
        print("relevant_python_files", relevant_python_files)
        topn_content, relevant_file_contents, num_tokens = get_content(instance, relevant_python_files, repo_playground=repo_playground, dataset_name=dataset_name, dataset_split=dataset_split)
        if not topn_content:
            print(f"Topn content is not found for instance {instance['instance_id']}, skipping...")
            return None
        
        if prompt_type == 'eval':
            prompt = PATCH_GEN_PROMPT.format(problem_statement=problem_statement, content=topn_content)
        elif prompt_type == 'repro-gen':
            prompt = PREMISE_TEST_GEN_PROMPT + TEST_GEN_PROMPT.format(problem_statement=problem_statement, content=topn_content)
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
    else:
        assert relevant_file_contents is not None, "relevant_file_contents must be provided if prompt is not None"
        relevant_file_contents = json.loads(relevant_file_contents)

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
        "metadata": {
            "relevant_file_contents": json.dumps(relevant_file_contents),
            "image": image_path,
            "remove_repo_name": False,
            "num_tokens": num_tokens,
        },
        "instance": instance,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "mode": prompt_type,
    }

    return row

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='SWE-Gym/SWE-Gym')
    parser.add_argument('--dataset_split', type=str, default='train')
    parser.add_argument('--out_path', type=str, default='data/train.jsonl')
    parser.add_argument('--eval_script_dir', type=str, default='eval_scripts')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing the singularity images')
    parser.add_argument('--repo_playground', type=str, default='./repo_playground')
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    # Minimal example demonstrating how to build and write a tiny dataset.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(cur_dir, "data"), exist_ok=True)
    
    args = parse_args()
    dataset_name = args.dataset_name
    dataset_split = args.dataset_split
    eval_script_dir = os.path.join(cur_dir, args.eval_script_dir)
    out_path = os.path.join(cur_dir, args.out_path)
    rows = {}
    if os.path.exists(out_path):
        rows = read_jsonl(out_path)
    
    dataset = load_dataset(dataset_name, split=dataset_split)
    for example in dataset:
        for prompt_type in ['eval', 'repro-gen']:
            if f"{example['instance_id']}-{prompt_type}" in rows:
                continue
            row = build_row(
                    instance=example,
                    eval_script_dir=eval_script_dir,
                    image_dir=args.image_dir,
                    prompt_type=prompt_type,
                    dataset_name=dataset_name,
                    dataset_split=dataset_split,
                )
            if row is not None:
                write_jsonl([row], out_path)
    
