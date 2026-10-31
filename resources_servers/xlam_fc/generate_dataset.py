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

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm


SYSTEM_PROMPT = """You are a helpful AI assistant with access to various functions. When you need to use a function to answer a user's request, call the appropriate function with the correct arguments. You can call multiple functions if needed to fully address the user's query."""


# Bare xlam type names and their JSON-schema types. Compound annotations (``List[int]``, ``int, optional``,
# ``Tuple[float, float]``) are parsed by ``parameter_schema``; an annotation it cannot type declares no type at
# all rather than claiming ``string``: the dataset's expected_answers keep native JSON values, and the verifier
# compares them strictly, so a wrongly declared ``string`` makes a schema-following call unwinnable (NVBug 6900453).
_SIMPLE_TYPES = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "tuple": "array",
    "set": "array",
    "frozenset": "array",
    "sequence": "array",
    "dict": "object",
    "object": "object",
    "mapping": "object",
}
_ARRAY_GENERICS = {"list", "tuple", "set", "frozenset", "sequence", "iterable"}
_OBJECT_GENERICS = {"dict", "mapping"}


def _split_top_level(text: str, separator: str = ",") -> List[str]:
    """Split ``text`` on ``separator`` outside brackets: ``"Tuple[int, str], optional"`` -> two parts."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def parameter_schema(param_type: str) -> Dict[str, Any]:
    """The JSON-schema fragment (``type`` and, for arrays, ``items``) for an xlam parameter type string.

    xlam writes Python-style annotations with free-text qualifiers: ``"int, optional"``, ``"str, default='x'"``,
    ``"List[int]"``, ``"Tuple[float, float]"``, ``"Optional[int]"``, ``"Union[int, str]"``. Qualifiers after the
    first top-level comma are dropped; ``Optional[X]`` is ``X``; array generics get ``items`` when every member
    types the same way; a ``Union`` is typed only when all members agree. Anything else (``Callable[...]``,
    ``Any``, an empty string) returns ``{}`` so the parameter stays untyped instead of being declared a string.
    """
    parts = _split_top_level((param_type or "").strip())
    if not parts:  # empty, or only separators and spaces (","): nothing to type
        return {}
    head = parts[0]
    if head.lower().endswith(" or none"):
        head = head[: -len(" or none")].strip()
    match = re.fullmatch(r"([A-Za-z_]+)\s*(?:\[(.*)\])?", head, flags=re.DOTALL)
    if match is None:
        return {}
    name, inner = match.group(1).lower(), match.group(2)
    if inner is not None:
        members = [parameter_schema(member) for member in _split_top_level(inner)]
        if name == "optional":
            return members[0] if len(members) == 1 else {}
        if name == "union":
            return members[0] if members and all(member == members[0] for member in members) else {}
        if name in _ARRAY_GENERICS:
            schema: Dict[str, Any] = {"type": "array"}
            if members and "type" in members[0] and all(member == members[0] for member in members):
                schema["items"] = members[0]
            return schema
        if name in _OBJECT_GENERICS:
            return {"type": "object"}
        return {}
    json_type = _SIMPLE_TYPES.get(name)
    return {"type": json_type} if json_type else {}


def convert_parameter_type(param_type: str) -> Optional[str]:
    """The top-level JSON-schema type for an xlam type string, or None when it cannot be typed."""
    return parameter_schema(param_type).get("type")


def convert_tool_to_openai_format(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a tool from the dataset format to OpenAI function calling format.

    Dataset format example:
    {
        "name": "tool_name",
        "description": "Tool description",
        "parameters": {
            "param_name": {
                "description": "param description",
                "type": "str",
                "default": "value"
            },
            "ids": {"description": "ids", "type": "List[int]"}
        }
    }

    OpenAI format (types follow the xlam annotation, see ``parameter_schema``):
    {
        "type": "function",
        "name": "tool_name",
        "description": "Tool description",
        "parameters": {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "param description"
                },
                "ids": {"type": "array", "items": {"type": "integer"}, "description": "ids"}
            },
            "required": ["ids"],
            "additionalProperties": False
        },
        "strict": False
    }
    """
    openai_tool = {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": False,
    }

    if "parameters" in tool and isinstance(tool["parameters"], dict):
        for param_name, param_info in tool["parameters"].items():
            if isinstance(param_info, dict):
                prop = {
                    **parameter_schema(param_info.get("type", "")),
                    "description": param_info.get("description", ""),
                }

                if "default" in param_info:
                    prop["default"] = param_info["default"]

                if "enum" in param_info:
                    prop["enum"] = param_info["enum"]

                openai_tool["parameters"]["properties"][param_name] = prop

                if "default" not in param_info:
                    openai_tool["parameters"]["required"].append(param_name)

    return openai_tool


def parse_expected_answers(answers_str: str) -> List[Dict[str, Any]]:
    try:
        answers = json.loads(answers_str)
        if not isinstance(answers, list):
            answers = [answers]
        return answers
    except json.JSONDecodeError as e:
        print(f"Error parsing answers: {answers_str}")
        print(f"Error: {e}")
        return []


def parse_tools(tools_str: str) -> List[Dict[str, Any]]:
    try:
        tools = json.loads(tools_str)
        if not isinstance(tools, list):
            tools = [tools]

        openai_tools = []
        for tool in tools:
            openai_tool = convert_tool_to_openai_format(tool)
            openai_tools.append(openai_tool)

        return openai_tools
    except json.JSONDecodeError as e:
        print(f"Error parsing tools: {tools_str}")
        print(f"Error: {e}")
        return []


def generate_dataset(output_dir: str = "resources_servers/xlam_fc/data", valid_size: int = 1000):
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    print(f"Loaded {len(dataset)} examples")

    print(f"Splitting dataset: {valid_size} for validation, rest for training...")
    split_dataset = dataset.train_test_split(test_size=valid_size, seed=42)
    train_dataset = split_dataset["train"]
    valid_dataset = split_dataset["test"]

    print(f"Train: {len(train_dataset)} examples")
    print(f"Valid: {len(valid_dataset)} examples")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.jsonl"
    valid_path = output_path / "valid.jsonl"

    print("\nProcessing training set...")
    train_processed, train_skipped = process_and_write_split(train_dataset, train_path)

    print("\nProcessing validation set...")
    valid_processed, valid_skipped = process_and_write_split(valid_dataset, valid_path)

    print("\nDataset generation complete!")
    print(f"Train - Processed: {train_processed}, Skipped: {train_skipped}")
    print(f"Valid - Processed: {valid_processed}, Skipped: {valid_skipped}")
    print(f"Output directory: {output_dir}")


def process_and_write_split(dataset, output_path: Path) -> tuple[int, int]:
    processed_count = 0
    skipped_count = 0

    with open(output_path, "w") as f:
        for example in tqdm(dataset):
            tools = parse_tools(example["tools"])
            expected_answers = parse_expected_answers(example["answers"])

            if not tools or not expected_answers:
                skipped_count += 1
                continue

            responses_create_params = {
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["query"]},
                ],
                "tools": tools,
            }

            record = {
                "id": example["id"],
                "responses_create_params": responses_create_params,
                "expected_answers": expected_answers,
            }

            f.write(json.dumps(record) + "\n")
            processed_count += 1

    return processed_count, skipped_count


def generate_example_file(output_dir: str = "resources_servers/xlam_fc/data", num_examples: int = 5):
    dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    example_path = output_path / "example.jsonl"

    print(f"Generating {num_examples} examples...")
    processed_count = 0
    skipped_count = 0

    with open(example_path, "w") as f:
        for i, example in enumerate(dataset):
            if processed_count >= num_examples:
                break

            tools = parse_tools(example["tools"])
            expected_answers = parse_expected_answers(example["answers"])

            if not tools or not expected_answers:
                skipped_count += 1
                continue

            responses_create_params = {
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["query"]},
                ],
                "tools": tools,
            }

            record = {
                "id": processed_count,
                "responses_create_params": responses_create_params,
                "expected_answers": expected_answers,
            }

            f.write(json.dumps(record) + "\n")
            processed_count += 1

    print(f"Generated {processed_count} examples in {example_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        generate_example_file()
    else:
        generate_dataset()
