import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm

SYSTEM_PROMPT = """You are a helpful AI assistant with access to various functions. When you need to use a function to answer a user's request, call the appropriate function with the correct arguments. You can call multiple functions if needed to fully address the user's query."""

def convert_parameter_type(param_type: str) -> str:
    type_mapping = {
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
        "dict": "object",
        "object": "object",
    }
    return type_mapping.get(param_type.lower(), "string")


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
            }
        }
    }

    OpenAI format:
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
                }
            },
            "required": ["param_name"],
            "additionalProperties": False
        },
        "strict": True
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
                    "type": convert_parameter_type(param_info.get("type", "string")),
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


def generate_dataset(output_path: str = "resources_servers/xlam_fc/data/train.jsonl"):
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    print(f"Loaded {len(dataset)} examples")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    print("Processing and writing dataset...")
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

    print(f"\nDataset generation complete!")
    print(f"Processed: {processed_count} examples")
    print(f"Skipped: {skipped_count} examples")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    generate_dataset()