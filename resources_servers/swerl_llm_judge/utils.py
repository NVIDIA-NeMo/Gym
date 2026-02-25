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
import ast
import os
import pickle
import re
import shutil
import subprocess
import uuid
from time import sleep
from typing import Any, Dict, Union, cast

import tiktoken
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


def extract_filenames(text):
    # Regular expression to match the 'diff --git' line and capture python filenames
    diff_pattern = re.compile(r"^diff --git (?:a/|b/)?(.+?) (?:a/|b/)?(.+?)$", re.MULTILINE)
    matches = diff_pattern.findall(text)
    filenames = list(set([match[1] for match in matches if "/dev/null" not in match[1]]))

    return filenames


_DATASET_CACHE = {}


def get_instance(instance_id, dataset_name, dataset_split):
    if dataset_name not in _DATASET_CACHE:
        _DATASET_CACHE[dataset_name] = load_dataset(dataset_name, split=dataset_split)
    dataset = _DATASET_CACHE[dataset_name]
    instance = dataset.filter(lambda x: x["instance_id"] == instance_id)
    return instance[0]


def repo_to_folder_name(repo_name):
    return repo_name.split("/")[-1]


def get_repo_path(repo_name, repo_playground):
    return os.path.join(repo_playground, repo_to_folder_name(repo_name))


def checkout_commit(repo_name, repo_playground, commit_id, reset=False):
    """Checkout the specified commit in the given local git repository.
    :param repo_name: Name of he repository
    :param repo_playground: Base path to the local git repository
    :param commit_id: Commit ID to checkout
    :return: None
    """
    try:
        # Change directory to the provided repository path and checkout the specified commit
        repo_path = get_repo_path(repo_name, repo_playground)
        print(f"Checking out commit {commit_id} in repository at {repo_path}...")
        if reset:
            subprocess.run(f"cd {repo_path} && git stash && git reset --hard && git clean -fd", shell=True, check=True)
        subprocess.run(["git", "-C", repo_path, "checkout", commit_id], check=True)
        print("Commit checked out successfully.")
        return True
    except Exception:
        print("An error occurred while checking out the commit")
        return False


def clone_repo(repo_name, repo_playground):
    """
    Taken from AGENTLESS repository
    """
    if os.path.exists(get_repo_path(repo_name, repo_playground)):
        print(f"Repository {get_repo_path(repo_name, repo_playground)} already exists.")
        return True
    try:
        print(
            f"Cloning repository from https://github.com/{repo_name}.git to {get_repo_path(repo_name, repo_playground)}..."
        )
        subprocess.run(
            [
                "git",
                "clone",
                f"https://github.com/{repo_name}.git",
                f"{repo_playground}/{repo_to_folder_name(repo_name)}",
            ],
            check=True,
        )
        print("Repository cloned successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


def create_repo_instance(instance, repo_playground, create_tmp=False, reset=False):
    """
    Clones the repo and checkout to the base commit of the instance
    args:
        instance: the instance object from the database
        repo_playground: root project directory to save the repo

    returns:
        path to the repo
    """
    repo_name = instance["repo"]
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4())) if create_tmp else repo_playground
    repo_path = os.path.join(repo_playground, repo_to_folder_name(repo_name))
    return_status = clone_repo(repo_name=repo_name, repo_playground=repo_playground)
    if not return_status:
        ## could not clone the repo
        return ""
    return_status = checkout_commit(
        repo_name=repo_name, repo_playground=repo_playground, commit_id=instance["base_commit"], reset=reset
    )
    if not return_status:
        ## could not checkout commit
        return ""

    return repo_path


def parse_python_file(file_path, file_content=None):
    """Taken from AGENTLESS repository
    Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return {}, {}, ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return {}, {}, ""

    class_info = {}
    function_names = {}
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = {}
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods[n.name] = {
                        "name": n.name,
                        "start_line": n.lineno,
                        "end_line": n.end_lineno,
                        "text": file_content.splitlines()[n.lineno - 1 : n.end_lineno],
                    }

                    class_methods.add(n.name)
            class_info[node.name] = {
                "name": node.name,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "text": file_content.splitlines()[node.lineno - 1 : node.end_lineno],
                "methods": methods,
            }
        elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
            if node.name not in class_methods:
                function_names[node.name] = {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[node.lineno - 1 : node.end_lineno],
                }

    return class_info, function_names, file_content.splitlines()


TOKEN_COUNT_MAP: dict[tuple[str, str], int] = {}
_TOKENIZER = None
TOKENIZER_MODEL = cast(str, os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-8B"))
assert TOKENIZER_MODEL is not None
TOKENIZER_TYPE = os.getenv("TOKENIZER_TYPE", "hf")
assert TOKENIZER_TYPE in ["hf", "tiktoken"], f"Invalid TOKENIZER_TYPE: {TOKENIZER_TYPE}"


def get_tokenizer() -> PreTrainedTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    return _TOKENIZER


def count_tokens(messages_or_prompt: list[dict] | str) -> int:
    """Count tokens for the specified tokenizer."""
    if TOKENIZER_TYPE == "hf":
        return count_hf_tokens(messages_or_prompt)
    return count_tiktoken_tokens(messages_or_prompt)


def count_hf_tokens(messages_or_prompt: list[dict] | str) -> int:
    """Count tokens for HF tokenizer."""
    tokenizer = get_tokenizer()
    if isinstance(messages_or_prompt, str):
        return len(tokenizer.encode(messages_or_prompt))
    return len(tokenizer.apply_chat_template(messages_or_prompt, add_generation_prompt=True))


def count_tiktoken_tokens(messages: list[dict] | str) -> int:
    """Returns the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(TOKENIZER_MODEL)
    if isinstance(messages, str):
        return len(encoding.encode(messages))
    num_tokens = sum(len(encoding.encode(message["content"])) for message in messages)
    return num_tokens


def cache_token_count(instance_id: str, file_name: str, content: str) -> int:
    key = (instance_id, file_name)
    if key in TOKEN_COUNT_MAP:
        return TOKEN_COUNT_MAP[key]
    tokens = count_tokens(content)
    TOKEN_COUNT_MAP[key] = tokens
    return tokens


def construct_topn_file_context(
    instance_id: str,
    target_files: list[str],
    file_contents: dict[str, str],
    max_input_tokens: int,
):
    num_tokens = 0
    all_contents = list[str]()
    for target_file in target_files:
        content = file_contents[target_file]
        content = f"[start of {target_file}]\n{content}\n[end of {target_file}]"
        num_new_tokens = cache_token_count(instance_id, target_file, content)
        if num_tokens + num_new_tokens > max_input_tokens:
            print(
                f"Skipping {target_file} as it is exceeding the max input tokens: {num_tokens + num_new_tokens} > {max_input_tokens}"
            )
            continue
        num_tokens += num_new_tokens
        all_contents.append(content)

    if len(all_contents) == 0 and len(target_files) > 0:
        return f"[start of {target_files[0]}]\n{file_contents[target_files[0]]}\n[end of {target_files[0]}]"
    return "\n\n".join(all_contents)


def get_content(item, target_files, repo_playground: str, dataset_name: str, dataset_split: str):
    """
    Get the code content of the target files.
    """

    instance_id = item["instance_id"]
    instance_obj = create_instance_obj(
        item, repo_playground=repo_playground, dataset_name=dataset_name, dataset_split=dataset_split
    )
    target_file_contents = {
        file: "\n".join(instance_obj.python_files[file]["text"])
        for file in instance_obj.python_files
        if file in target_files
    }
    all_existing_files = list(target_file_contents.keys())
    flag = False
    for target_file in target_files:
        if target_file not in all_existing_files:
            flag = True
            break
    if flag:
        # If any of the found files are not in the repo_file_contents_dict, return None
        print("Some target files are not found in the repo, skipping...")
        return None, None

    topn_content = construct_topn_file_context(
        instance_id,
        target_files,
        target_file_contents,
        max_input_tokens=28000,
    )
    return topn_content, target_file_contents


def create_structure(directory_path):
    """Taken from AGENTLESS repository and modified slightly
    Create the structure of the repository directory by parsing Python files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
    structure = {}

    for root, _, files in os.walk(directory_path):
        relative_root = os.path.relpath(root, os.path.dirname(directory_path))
        relative_root_wo_dir = "/".join(relative_root.split(os.sep)[1:]) if relative_root != "." else ""
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                    "relative_path": f"{relative_root_wo_dir}/{file_name}" if relative_root_wo_dir else file_name,
                }
            elif os.path.basename(file_name).lower().startswith("readme"):
                try:
                    with open(os.path.join(root, file_name), "r") as f:
                        content = f.read().splitlines()
                except Exception:
                    content = "[BINARY FILE]"
                curr_struct[file_name] = {
                    "text": content,
                    "relative_path": f"{relative_root_wo_dir}/{file_name}" if relative_root_wo_dir else file_name,
                }
            else:
                curr_struct[file_name] = {}

    return structure


def get_python_files(python_files, structure):
    for k, v in structure.items():
        if k.endswith(".py"):
            python_files[v["relative_path"]] = v
        elif len(v) > 0:
            if type(v) is dict:
                get_python_files(python_files, v)


def get_readme_files(readmes, structure):
    for k, v in structure.items():
        if k.split("/")[-1].lower().startswith("readme"):
            readmes[v["relative_path"]] = v
        elif not k.endswith(".py") and len(v) > 0:
            if type(v) is dict:
                get_readme_files(readmes, v)


class InstanceObj(object):
    def __init__(
        self,
        instance_id: Union[str, Dict[str, Any]],
        repo_playground: str,
        dataset_name: str,
        dataset_split: str,
        create_tmp: bool = False,
        reset: bool = False,
    ):
        if isinstance(instance_id, str):
            self.instance = get_instance(instance_id, dataset_name, dataset_split)
            self.instance_id = instance_id
        elif isinstance(instance_id, dict):
            self.instance = instance_id
            assert "instance_id" in instance_id, "Expected a dictionary with instance_id as a key"
            self.instance_id = instance_id["instance_id"]
        else:
            raise ValueError(
                "Expected either a string showing the instance_id or a "
                + f"dictionary representing the instance, but got {type(instance_id)}."
            )

        repo_playground = os.path.abspath(repo_playground)
        repo_base_path = os.path.join(repo_playground, self.instance_id) if not reset else repo_playground
        self.repo_path = os.path.join(repo_base_path, repo_to_folder_name(self.instance["repo"]))
        structure_info = os.path.join(repo_playground, "repo_info", f"{self.instance_id}.pickle")
        os.makedirs(os.path.dirname(structure_info), exist_ok=True)
        print(f"structure_info: {structure_info}")
        print(f"repo_base_path: {repo_base_path}")

        get_info = True
        if os.path.exists(structure_info):
            try:
                with open(structure_info, "rb") as f:
                    repo_info = pickle.load(f)
                self.structure = repo_info["structure"]
                self.python_files = repo_info["python_files"]
                get_info = False
            except Exception as e:
                print(f"Error loading structure info: {e}")
                get_info = True
        if get_info:
            if reset or not os.path.exists(self.repo_path):
                self.repo_path = ""
                while self.repo_path == "":  ## continue until repo is successfully cloned
                    self.repo_path = create_repo_instance(
                        self.instance, repo_base_path, create_tmp=create_tmp, reset=reset
                    )
                    sleep(5)

            if self.repo_path != "":
                self.structure = create_structure(self.repo_path)
                self.python_files = {}
                get_python_files(self.python_files, self.structure)
                repo_info = {"structure": self.structure, "python_files": self.python_files}
                with open(structure_info, "wb") as f:
                    pickle.dump(repo_info, f, pickle.HIGHEST_PROTOCOL)

    def add_readme(self):
        self.readmes = {}
        get_readme_files(self.readmes, self.structure)

    def del_repo(self):
        if os.path.isdir(os.path.dirname(self.repo_path)):
            shutil.rmtree(os.path.dirname(self.repo_path))


def create_instance_obj(instance_id, dataset_name, dataset_split, repo_playground, reset=False):
    return InstanceObj(instance_id, repo_playground, dataset_name, dataset_split, reset=reset)
