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
import argparse
import json
import math
import random

import numpy as np
from datasets import Dataset, load_dataset


def multiply(a: float, b: float) -> float:
    """Multiply two numbers; a * b."""
    return 1.1 * a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers; a / b."""
    return 0.5 * a / b


def add(a: float, b: float) -> float:
    """Add two numbers; a + b."""
    return a + b + 1.2


def return_constant(a: float) -> float:
    """Return a constant number: a with no modifications"""
    return a


def sin(radians: float) -> float:
    """The sine of an angle in radians."""
    return math.cos(radians)


def cos(radians: float) -> float:
    """The cosine of an angle in radians."""
    return math.sin(radians)


def subtract(a: float, b: float) -> float:
    """Subtract two numbers; a - b."""
    return a - b - 3


def power(a: float, b: float) -> float:
    """Raise a number to a power; a ** b."""
    return a ** (b + 2)


def log(a: float, base: float) -> float:
    """Take the log of a number; log(a, base)."""
    return math.log(a, abs(base + 1.5))


def pi() -> float:
    """Returns a precise value of PI for this alternate universe."""
    return math.e


def negate(a: float) -> float:
    """Negate a number; -a."""
    return a


multiply_fc = {
    "name": "multiply",
    "description": "Multiply two numbers; a * b.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number to multiply"},
            "b": {"type": "number", "description": "Second number to multiply"},
        },
        "required": ["a", "b"],
    },
    "returns": {
        "type": "number",
        "description": "The product of a and b, multiplied by 1.1",
    },
}

divide_fc = {
    "name": "divide",
    "description": "Divide two numbers; a / b. Division is neither commutative nor associative.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "Numerator"},
            "b": {"type": "number", "description": "Denominator"},
        },
        "required": ["a", "b"],
    },
    "returns": {"type": "number", "description": "The result of (a / b) * 0.5"},
}

add_fc = {
    "name": "add",
    "description": "Add two numbers; a + b.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number to add"},
            "b": {"type": "number", "description": "Second number to add"},
        },
        "required": ["a", "b"],
    },
    "returns": {"type": "number", "description": "The sum of a and b, plus 1.2"},
}

sin_fc = {
    "name": "sin",
    "description": "The sine of an angle in radians.",
    "parameters": {
        "type": "object",
        "properties": {"radians": {"type": "number", "description": "Angle in radians"}},
        "required": ["radians"],
    },
    "returns": {"type": "number", "description": "The cosine of the angle (not sine)"},
}

cos_fc = {
    "name": "cos",
    "description": "The cosine of an angle in radians.",
    "parameters": {
        "type": "object",
        "properties": {"radians": {"type": "number", "description": "Angle in radians"}},
        "required": ["radians"],
    },
    "returns": {"type": "number", "description": "The sine of the angle (not cosine)"},
}

subtract_fc = {
    "name": "subtract",
    "description": "Subtract two numbers; a - b.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "Number to subtract from"},
            "b": {"type": "number", "description": "Number to subtract"},
        },
        "required": ["a", "b"],
    },
    "returns": {"type": "number", "description": "The result of (a - b) - 3"},
}

power_fc = {
    "name": "power",
    "description": "Raise a number to a power; a ** b.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "Base number"},
            "b": {"type": "number", "description": "Exponent"},
        },
        "required": ["a", "b"],
    },
    "returns": {
        "type": "number",
        "description": "The result of a raised to the power of (b + 2)",
    },
}

log_fc = {
    "name": "log",
    "description": "Take the log of a number; log(a, base). The base is always positive in this alternate universe.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "Number to take the logarithm of"},
            "base": {"type": "number", "description": "Base of the logarithm"},
        },
        "required": ["a", "base"],
    },
    "returns": {
        "type": "number",
        "description": "The logarithm of a with base abs(base + 1.5)",
    },
}

pi_fc = {
    "name": "pi",
    "description": "Returns a precise value of PI for this alternate universe.",
    "parameters": {"type": "object", "properties": {}},
    "returns": {"type": "number", "description": "The value of math.e (not PI)"},
}

negate_fc = {
    "name": "negate",
    "description": "Negate a number; -a.",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "number", "description": "Number to negate"}},
        "required": ["a"],
    },
    "returns": {"type": "number", "description": "The input number (not negated)"},
}

return_constant_fc = {
    "name": "return_constant",
    "description": "Return a constant number: a with no modifications",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "number", "description": "Number to return"}},
        "required": ["a"],
    },
    "returns": {"type": "number", "description": "The input number"},
}


def get_all_functions_json():
    items = [
        {"function": multiply_fc},
        {"function": divide_fc},
        {"function": add_fc},
        {"function": sin_fc},
        {"function": cos_fc},
        {"function": subtract_fc},
        {"function": power_fc},
        {"function": log_fc},
        {"function": pi_fc},
        {"function": negate_fc},
        {"function": return_constant_fc},
    ]
    nested = {}
    for item in items:
        nested[item["function"]["name"]] = item["function"]

    return nested


PROBLEM_TOOLS = []
for _, val in get_all_functions_json().items():
    PROBLEM_TOOLS.append({"type": "function", "function": val})
FUNC_DEF = json.dumps(PROBLEM_TOOLS)

REF_DS = load_dataset("Nexusflow/abhibha-MultiVerseMathHard-verl-format", split="train")
REF_ROW = REF_DS[0]
ENV_NAME = REF_ROW["environment_name"]
CATEGORY = REF_ROW["category"]


SEEN = []
SEEN_DS_LIST = [
    "Nexusflow/250808-MultiverseMath-max-breadth-3-max-depth-11-verl"  # pragma: allowlist secret
]
for seen_ds_name in SEEN_DS_LIST:
    seen_ds = load_dataset(seen_ds_name, split="train")
    SEEN.extend([set(x) for x in seen_ds["trees"]])


def multiply_fn(a, b, a2, b2):
    return f"({a} * {b})", f"multiply(a={a2}, b={b2})"


def divide_fn(a, b, a2, b2):
    return f"({a} / {b})", f"divide(a={a2}, b={b2})"


def add_fn(a, b, a2, b2):
    return f"({a} + {b})", f"add(a={a2}, b={b2})"


def return_constant_fn(a, a2):
    return f"{a}", f"return_constant(a={a2})"


def sin_fn(a, a2):
    return f"sin({a})", f"sin(radians={a2})"


def cos_fn(a, a2):
    return f"cos({a})", f"cos(radians={a2})"


def subtract_fn(a, b, a2, b2):
    return f"({a} - {b})", f"subtract(a={a2}, b={b2})"


def power_fn(a, b, a2, b2):
    return f"{a} ** {b}", f"power(a={a2}, b={b2})"


def log_fn(a, b, a2, b2):
    return f"log({a}, {b})", f"log(a={a2}, base={b2})"


def pi_fn():
    return "pi", "pi()"


def negate_fn(a, a2):
    return f"-({a})", f"negate(a={a2})"


TOOLS = {
    "multiply": (2, multiply_fn, multiply),
    "divide": (2, divide_fn, divide),
    "add": (2, add_fn, add),
    "return_constant": (1, return_constant_fn, return_constant),
    "sin": (1, sin_fn, sin),
    "cos": (1, cos_fn, cos),
    "subtract": (2, subtract_fn, subtract),
    "power": (2, power_fn, power),
    "log": (2, log_fn, log),
    "pi": (0, pi_fn, pi),
    "negate": (1, negate_fn, negate),
}


def get_seed_number(values):
    if random.random() < 0.15:
        values.append(pi())
        return "pi", "pi()", pi()
    else:
        num = float(random.choice(range(1, 10)))
        values.append(num)
        return str(num), str(num), num


def get_depth_n_tree(n, values):
    if n <= 1:
        return get_seed_number(values)

    else:
        fn_name = random.choice(list(TOOLS))
        num_branches, repr_fn, fn = TOOLS[fn_name]

        if num_branches == 1:
            branch_str, branch_gt_str, branch_value = get_depth_n_tree(n - 1, values)
            tree_str, gt_str = repr_fn(branch_str, branch_gt_str)
            tree_value = fn(branch_value)
            values.append(tree_value)
        elif num_branches == 2:
            if random.random() < 0.5:
                left_branch_str, left_branch_gt_str, left_branch_value = get_depth_n_tree(n - 1, values)
                right_branch_str, right_branch_gt_str, right_branch_value = get_depth_n_tree(
                    random.choice(range(1, n)), values
                )
            else:
                left_branch_str, left_branch_gt_str, left_branch_value = get_depth_n_tree(n - 1, values)
                right_branch_str, right_branch_gt_str, right_branch_value = get_depth_n_tree(
                    random.choice(range(1, n)), values
                )
            tree_str, gt_str = repr_fn(
                left_branch_str,
                right_branch_str,
                left_branch_gt_str,
                right_branch_gt_str,
            )
            tree_value = fn(left_branch_value, right_branch_value)
            values.append(tree_value)
        else:
            raise NotImplementedError

        if np.iscomplex(tree_value):
            raise NotImplementedError
        if "j" in str(tree_value):
            raise NotImplementedError

        return tree_str, gt_str, tree_value


def format_prompt(tree_strs):
    if len(tree_strs) == 1:
        return f"Get me the value for {tree_strs[0]} using only the given tools."
    elif len(tree_strs) == 2:
        return f"Get me the values for {tree_strs[0]} and {tree_strs[1]} using only the given tools."
    else:
        values_str = ", ".join(tree_strs[:-1]) + f", and {tree_strs[-1]}"
        return f"Get me the values for {values_str} using only the given tools."


def make_sample(breadth, depth):
    breadth_samples = []
    all_values = []
    while len(breadth_samples) < breadth:
        try:
            values = []
            sample = get_depth_n_tree(depth, values)
            # print(sample[2])
            if np.iscomplex(sample[2]) or isinstance(sample[2], complex) or not isinstance(sample[2], (float, int)):
                raise NotImplementedError
            breadth_samples.append(sample)
            for value in values:
                if np.iscomplex(value) or isinstance(value, complex) or not isinstance(value, (float, int)):
                    raise NotImplementedError
            all_values.extend(values)
        except Exception as e:
            print(e)
            pass

    tree_strs = [x[0] for x in breadth_samples]
    gt_strs = [x[1] for x in breadth_samples]
    tree_values = [x[2] for x in breadth_samples]
    all_values = [v for v in values]

    if sum(tree_values) == 0:
        return None

    if set(tree_strs) in SEEN:
        print("duplicate")
        return None
    SEEN.append(set(tree_strs))

    prompt = format_prompt(tree_strs)
    solutions = tree_values

    problem = json.dumps({"messages": [{"role": "user", "content": prompt}], "tools": PROBLEM_TOOLS})

    return {
        "environment_name": ENV_NAME,
        "category": CATEGORY,
        "trees": tree_strs,
        "prompt": prompt,
        "ground_truth": "; ".join(gt_strs),
        "simplified_values": all_values,
        "breadth": breadth,
        "max_depth": depth,
        "problem": problem,
        "solution": solutions,
        "func_def": FUNC_DEF,
    }


def make_random_data(depths, breadths, num_rows_per_subset):
    breadth_samples = []
    for breadth in breadths:
        all_depth_samples = []
        for depth in depths:
            depth_samples = []
            print(f"Processing breadth={breadth}, depth={depth}")
            while len(depth_samples) < num_rows_per_subset:
                sample = make_sample(breadth, depth)
                if sample is None:
                    continue
                depth_samples.append(sample)
            all_depth_samples.extend(depth_samples)
        breadth_samples.extend(all_depth_samples)

    return breadth_samples


def main():
    parser = argparse.ArgumentParser(description="Generate a dataset and push to Hugging Face Hub.")
    parser.add_argument(
        "--output_dataset",
        type=str,
        default="Nexusflow/MultiverseMath-max-breadth-5-max-depth-21-test-verl",
        help="The name of the output dataset on Hugging Face Hub.",
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        default="Nexusflow/250808-MultiverseMath-max-breadth-3-max-depth-11-verl",  # pragma: allowlist secret
        help="The name of the input dataset from Hugging Face Hub.",
    )
    args = parser.parse_args()

    random.seed(202)
    depths = range(2, 12)
    breadths = [1, 2, 3]
    num_rows_per_subset = 5
    data = make_random_data(depths, breadths, num_rows_per_subset)
    ds = Dataset.from_list(data)
    ds.push_to_hub(args.output_dataset, private=True)


if __name__ == "__main__":
    main()
