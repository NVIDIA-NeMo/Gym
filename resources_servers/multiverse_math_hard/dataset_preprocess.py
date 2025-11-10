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
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List

from datasets import load_dataset

from resources_servers.multiverse_math_hard.multiverse_math_hard_tools import (
    add,
    cos,
    divide,
    log,
    multiply,
    negate,
    pi,
    power,
    return_constant,
    sin,
    subtract,
)


base_create_params = dict(
    input=[],
    tools=[
        {
            "type": "function",
            "name": "multiply",
            "description": "Multiply two numbers; a * b.",
            "parameters": {
                "properties": {
                    "a": {"type": "number", "description": "First number to multiply"},
                    "b": {"type": "number", "description": "Second number to multiply"},
                },
                "type": "object",
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "divide",
            "description": "Divide two numbers; a / b. Division is neither commutative nor associative.",
            "parameters": {
                "properties": {
                    "a": {"type": "number", "description": "Numerator"},
                    "b": {"type": "number", "description": "Denominator"},
                },
                "type": "object",
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "add",
            "description": "Add two numbers; a + b.",
            "parameters": {
                "properties": {
                    "a": {"type": "number", "description": "First number to add"},
                    "b": {"type": "number", "description": "Second number to add"},
                },
                "type": "object",
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "sin",
            "description": "The sine of an angle in radians.",
            "parameters": {
                "properties": {"radians": {"type": "number", "description": "Angle in radians"}},
                "type": "object",
                "required": ["radians"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "cos",
            "description": "The cosine of an angle in radians.",
            "parameters": {
                "properties": {"radians": {"type": "number", "description": "Angle in radians"}},
                "type": "object",
                "required": ["radians"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "subtract",
            "description": "Subtract two numbers; a - b.",
            "parameters": {
                "properties": {
                    "a": {"type": "number", "description": "Number to subtract from"},
                    "b": {"type": "number", "description": "Number to subtract"},
                },
                "type": "object",
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "power",
            "description": "Raise a number to a power; a ** b.",
            "parameters": {
                "properties": {
                    "a": {"type": "number", "description": "Base number"},
                    "b": {"type": "number", "description": "Exponent"},
                },
                "type": "object",
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "log",
            "description": "Take the log of a number; log(a, base). The base is always positive in this alternate universe.",
            "parameters": {
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "Number to take the logarithm of",
                    },
                    "base": {"type": "number", "description": "Base of the logarithm"},
                },
                "type": "object",
                "required": ["a", "base"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "pi",
            "description": "Returns a precise value of PI for this alternate universe.",
            "parameters": {
                "properties": {},
                "type": "object",
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "negate",
            "description": "Negate a number; -a.",
            "parameters": {
                "properties": {"a": {"type": "number", "description": "Number to negate"}},
                "type": "object",
                "required": ["a"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "return_constant",
            "description": "Return a constant number: a with no modifications",
            "parameters": {
                "properties": {"a": {"type": "number", "description": "Number to return"}},
                "type": "object",
                "required": ["a"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ],
    parallel_tool_calls=False,
    temperature=1.0,
)


@dataclass
class Sample:
    create_params: str
    reference: Any


@dataclass
class MultiverseMathHardSamples:
    @dataclass
    class MMMSample(Sample):
        depth: int
        breadth: int

    def get_samples(self) -> List[MMMSample]:
        dataset = load_dataset("Nexusflow/MultiverseMathHard", split="train")

        processed_samples = []

        tool_functions = {
            "multiply": multiply,
            "divide": divide,
            "add": add,
            "sin": sin,
            "cos": cos,
            "subtract": subtract,
            "power": power,
            "log": log,
            "pi": pi,
            "negate": negate,
            "return_constant": return_constant,
        }

        for d in dataset:
            # get evaled ground truths for MMH
            ground_truth_calls = [eval(g.strip(), tool_functions) for g in d["ground_truth"].split(";")]
            # convert into create params
            create_params = deepcopy(base_create_params)
            create_params["input"].append(
                {
                    "role": "user",
                    "content": d["prompt"],
                }
            )

            processed_samples.append(
                self.MMMSample(
                    create_params=create_params,
                    reference=ground_truth_calls,
                    depth=d["max_depth"],
                    breadth=d["breadth"],
                )
            )
        return processed_samples


def main():
    benchmark = MultiverseMathHardSamples()
    processed_samples = benchmark.get_samples()

    if not processed_samples:
        print("No samples were generated. Exiting.")
        return

    output_filename = "nemo-gym/resources_servers/multiverse_math_hard/data/train.jsonl"

    with open(output_filename, "w") as f:
        for i, sample in enumerate(processed_samples):
            record = {
                "id": i,
                "responses_create_params": sample.create_params,
                "ground_truth": json.dumps(sample.reference),
                "depth": sample.depth,
                "breadth": sample.breadth,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Successfully created {output_filename}.")


if __name__ == "__main__":
    main()
