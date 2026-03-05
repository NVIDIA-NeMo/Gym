#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
from pathlib import Path

from resources_servers.newton_bench.newton_bench_utils.prompt_utils import get_physics_prompt
from resources_servers.newton_bench.newton_bench_utils.schemas import MODULE_REQUEST_CLASSES_MAPPING, get_tool_schema


def generate_record(
    record_id: int,
    module_name: str,
    difficulty: str,
    system: str,
    noise_level: float,
    law_version: str,
    is_code_assisted: bool = True,
):
    task_prompt = get_physics_prompt(
        module_name=module_name, system=system, is_code_assisted=is_code_assisted, noise_level=noise_level
    )

    tools = [get_tool_schema(module_name)]

    if is_code_assisted:
        tools.append(
            {
                "type": "function",
                "name": "execute_python",
                "description": "Execute Python code for mathematical reasoning, hypothesis testing, and data analysis. Pre-imported libraries: numpy (as np), pandas (as pd), scipy, and math.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        )

    record = {
        "id": record_id,
        "module_name": module_name,
        "difficulty": difficulty,
        "system": system,
        "noise_level": noise_level,
        "law_version": law_version,
        "responses_create_params": {
            "input": [
                {"role": "system", "content": task_prompt},
                {
                    "role": "user",
                    "content": f"Begin your scientific discovery process for the {module_name}. Design experiments, analyze data, and discover the underlying law.",
                },
            ],
            "tools": tools,
            "parallel_tool_calls": True,
        },
    }

    return record


def write_dataset(configs, output_path):
    """Writes a list of configurations to a JSONL dataset file."""
    print(f"Generating dataset: {output_path}")
    with open(output_path, "w") as f:
        for idx, config in enumerate(configs):
            record = generate_record(idx, **config)
            f.write(json.dumps(record) + "\n")
    print(f"Generated {len(configs)} records")


def generate_example_configs(is_code_assisted):
    """Generates configuration list for example records."""
    base_example_configs = [
        {"difficulty": "easy", "system": "vanilla_equation", "noise_level": 0.0},
        {"difficulty": "medium", "system": "vanilla_equation", "noise_level": 0.0},
        {"difficulty": "hard", "system": "vanilla_equation", "noise_level": 0.0},
        {"difficulty": "easy", "system": "simple_system", "noise_level": 0.0},
        {"difficulty": "easy", "system": "complex_system", "noise_level": 0.0},
    ]
    configs = []
    for base_config in base_example_configs:
        configs.append(
            {"module_name": "m0_gravity", **base_config, "law_version": "v0", "is_code_assisted": is_code_assisted}
        )
    return configs


def generate_train_configs(target_modules, diff_list, sys_list, noise_list, is_code_assisted):
    """Generates configuration list for training records by crossing parameters."""
    configs = []
    for module_name in target_modules:
        for difficulty in diff_list:
            for system in sys_list:
                for noise_level in noise_list:
                    for law_version in ["v0", "v1", "v2"]:
                        configs.append(
                            {
                                "module_name": module_name,
                                "difficulty": difficulty,
                                "system": system,
                                "noise_level": noise_level,
                                "law_version": law_version,
                                "is_code_assisted": is_code_assisted,
                            }
                        )
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code-assisted",
        action="store_true",
        default=False,
        help="Include Python code execution support (default: False)",
    )
    parser.add_argument(
        "-m",
        "--modules",
        type=str,
        help="Comma-separated list of modules to generate (default: all)",
    )
    parser.add_argument(
        "-d",
        "--difficulties",
        type=str,
        default="easy,medium,hard",
        help="Comma-separated difficulties (default: easy,medium,hard)",
    )
    parser.add_argument(
        "-s",
        "--systems",
        type=str,
        default="vanilla_equation,simple_system,complex_system",
        help="Comma-separated systems (default: vanilla_equation,simple_system,complex_system)",
    )
    parser.add_argument(
        "-n",
        "--noise-levels",
        type=str,
        default="0.0,0.0001,0.001,0.01,0.1",
        help="Comma-separated noise levels (default: 0.0,0.0001,0.001,0.01,0.1)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Specify the output filename (overrides default naming logic)",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        default=False,
        help="Generate example.jsonl instead of train.jsonl",
    )
    args = parser.parse_args()

    is_code_assisted = args.code_assisted

    if args.modules:
        target_modules = [m.strip() for m in args.modules.split(",")]
    else:
        target_modules = list(MODULE_REQUEST_CLASSES_MAPPING.keys())

    diff_list = [d.strip() for d in args.difficulties.split(",")]
    sys_list = [s.strip() for s in args.systems.split(",")]
    noise_list = [float(n.strip()) for n in args.noise_levels.split(",")]

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    if args.example:
        print(f"Generating example dataset with code_assisted={is_code_assisted}")
        configs = generate_example_configs(is_code_assisted)
        out_path = output_dir / (args.output_name if args.output_name else "example.jsonl")
        write_dataset(configs, out_path)
    else:
        print(f"Generating train dataset with code_assisted={is_code_assisted} and target modules={target_modules}")
        configs = generate_train_configs(target_modules, diff_list, sys_list, noise_list, is_code_assisted)
        out_path = output_dir / (args.output_name if args.output_name else "train.jsonl")
        write_dataset(configs, out_path)

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
