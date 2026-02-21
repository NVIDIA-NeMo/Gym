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


## Install SWE-bench. For example, if you want to generate eval scripts for SWE-Gym, you can run:
## git clone https://github.com/SWE-Gym/SWE-Bench-Fork
## cd SWE-Bench-Fork
## pip install -e .
import argparse
import os
import stat
from pathlib import Path

from datasets import load_dataset
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.test_spec.test_spec import (
    make_test_spec,
)
from tqdm import tqdm


def generate_regression_setup_script(test_spec) -> str:
    """Return the regression setup script content for a given test spec."""
    lines: list[str] = ["#!/bin/bash"]
    for line in test_spec.eval_script_list:
        stripped = line.strip()
        if (
            stripped == "git status"
            or stripped == "git show"
            or stripped.startswith("git diff")
            or stripped.startswith("git config")
        ):
            continue

        if stripped.startswith("git apply"):
            break

        lines.append(line)
    return "\n".join(lines)


def generate_setup_script(test_spec) -> str:
    """Return the full setup script content (including the git apply line)."""
    lines: list[str] = ["#!/bin/bash"]
    for line in test_spec.eval_script_list:
        stripped = line.strip()
        if (
            stripped == "git status"
            or stripped == "git show"
            or stripped.startswith("git diff")
            or stripped.startswith("git config")
        ):
            continue

        lines.append(line)
        if stripped.startswith("git apply"):
            break

    return "\n".join(lines)


def generate_test_script(test_spec) -> str:
    """Return the test script content (no git commands, no init/pip install)."""
    lines: list[str] = ["#!/bin/bash"]
    for line in test_spec.eval_script_list:
        stripped = line.strip()
        if stripped.startswith("git"):
            continue
        if stripped == "make init":
            continue
        if "pip install" in line:
            continue
        if line not in lines:
            lines.append(line)
    return "\n".join(lines)


def write_executable_script(path: Path, content: str) -> None:
    """Write `content` to `path` and mark it as executable."""
    path.write_text(content, encoding="utf-8")
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def process_dataset(dataset_name: str, dataset_split: str, output_dir: Path, image_dir: str) -> None:
    """Load the dataset and generate regression/setup/test scripts for each instance."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split=dataset_split)
    for instance in tqdm(dataset, desc="Generating eval scripts"):
        if not instance["image_name"]:
            continue
        image_path = instance["image_name"] if "image_name" in instance else instance["instance_id"]
        if not os.path.exists(f"{image_dir}/{image_path}.sif"):
            continue
        instance_id = instance["instance_id"]
        test_spec = make_test_spec(SWEbenchInstance(**instance))

        regression_content = generate_regression_setup_script(test_spec)
        write_executable_script(output_dir / f"{instance_id}_regression.sh", regression_content)

        setup_content = generate_setup_script(test_spec)
        write_executable_script(output_dir / f"{instance_id}.sh", setup_content)

        test_content = generate_test_script(test_spec)
        write_executable_script(output_dir / f"{instance_id}_test.sh", test_content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SWE-Gym eval setup and test scripts from SWEbench instances.",
    )

    default_output_dir = Path(__file__).resolve().parent / "eval_scripts"

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SWE-Gym/SWE-Gym",
        help="Hugging Face dataset name to load (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help="Directory where generated shell scripts will be written (default: %(default)s).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    process_dataset(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
