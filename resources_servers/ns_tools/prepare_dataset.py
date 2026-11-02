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
Prepare a dataset for use with the ns_tools NeMo Gym resources server.

This script transforms a source dataset (e.g., comp-math-24-25/test.txt) into the
JSONL format required by nemo-gym, using NeMo Skills prompt configs (copied to prompts/)
and NeMo Skills tool schemas.

Usage:
    python prepare_dataset.py \
        --input /path/to/source.jsonl \
        --output /path/to/output.jsonl \
        --prompt_config generic/math \
        --tools nemo_skills.mcp.servers.python_tool::DirectPythonTool \
        --verifier_type math_with_judge

Example:
    python prepare_dataset.py \
        --input ~/nemo_skills/dataset/comp-math-24-25/test.txt \
        --output data/compmath_test.jsonl \
        --prompt_config generic/math \
        --tools nemo_skills.mcp.servers.python_tool::DirectPythonTool
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from nemo_skills.mcp.tool_manager import ToolManager

from nemo_gym.prompt import PromptConfig, load_prompt_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Prompt config names such as generic/math resolve to copies of the NeMo Skills YAMLs here, because the pinned
# nemo-skills-tools package ships none.
PROMPT_CONFIG_DIR = Path(__file__).parent / "prompts"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for ns_tools NeMo Gym resources server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input JSONL file (e.g., comp-math-24-25/test.txt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--prompt_config",
        default="generic/math",
        help="Prompt config name under prompts/ (e.g., generic/math), or a path to a prompt YAML file",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"],
        help="List of tool module specs to include (e.g., nemo_skills.mcp.servers.python_tool::DirectPythonTool)",
    )
    parser.add_argument(
        "--verifier_type",
        default=None,
        help="Verifier type to use (e.g., math_with_judge). If not set, uses default from config.",
    )
    parser.add_argument(
        "--agent_ref",
        default="ns_tools_simple_agent",
        help="Agent reference name for routing (default: ns_tools_simple_agent)",
    )
    parser.add_argument(
        "--problem_field",
        default="problem",
        help="Field name in source data containing the problem text",
    )
    parser.add_argument(
        "--answer_field",
        default="expected_answer",
        help="Field name in source data containing the expected answer",
    )
    parser.add_argument(
        "--id_field",
        default="id",
        help="Field name in source data containing the sample ID",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )
    parser.add_argument(
        "--sandbox_host",
        default="localhost",
        help="Sandbox host for code execution tools",
    )
    parser.add_argument(
        "--sandbox_port",
        type=int,
        default=6000,
        help="Sandbox port for code execution tools",
    )
    return parser.parse_args()


def format_tools_for_responses_api(raw_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format raw tool list for OpenAI responses API format."""
    formatted = []
    for t in raw_tools:
        input_schema = t.get("input_schema", {})
        # Remove title fields that aren't needed for the model
        input_schema.pop("title", None)
        for prop in input_schema.get("properties", {}).values():
            prop.pop("title", None)

        formatted.append(
            {
                "type": "function",
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": input_schema,
                "strict": True,
            }
        )
    return formatted


async def get_tool_schemas(
    tool_specs: list[str],
    sandbox_host: str,
    sandbox_port: int,
) -> list[dict[str, Any]]:
    """
    Initialize ToolManager and get formatted tool schemas.
    """
    # Provide sandbox config for tools that need it
    context = {
        "sandbox": {
            "host": sandbox_host,
            "port": sandbox_port,
        }
    }

    tool_manager = ToolManager(
        module_specs=tool_specs,
        overrides={},
        context=context,
    )

    # Get raw tool list
    raw_tools = await tool_manager.list_all_tools(use_cache=False)

    # Format for responses API endpoint
    formatted_tools = format_tools_for_responses_api(raw_tools)

    # Shutdown tool manager
    await tool_manager.shutdown()

    return formatted_tools


def load_prompt(prompt_config: str) -> PromptConfig:
    """
    Load a prompt config by name from prompts/ (e.g., generic/math), or from a path ending in .yaml.
    """
    if not prompt_config.endswith(".yaml"):
        prompt_config = str(PROMPT_CONFIG_DIR / f"{prompt_config}.yaml")
    return load_prompt_config(prompt_config)


def build_input_messages(problem: str, prompt: PromptConfig) -> list[dict[str, str]]:
    """
    Fill the prompt config's system and user templates for one problem.
    """
    # Few-shot examples are not included, so {examples} is empty.
    values = {"problem": problem, "examples": ""}
    input_messages = []
    if prompt.system:
        input_messages.append({"role": "system", "content": prompt.system.format(**values)})
    input_messages.append({"role": "user", "content": prompt.user.format(**values)})
    return input_messages


def process_sample(
    sample: dict[str, Any],
    idx: int,
    tool_schemas: list[dict[str, Any]],
    prompt: PromptConfig,
    problem_field: str,
    answer_field: str,
    id_field: str,
    verifier_type: str | None,
    agent_ref: str | None,
) -> dict[str, Any]:
    """
    Process a single sample into the nemo-gym format.
    """
    # Extract fields
    sample_id = sample.get(id_field, idx)
    problem = sample.get(problem_field, "")
    expected_answer = sample.get(answer_field, "")

    if not problem:
        logger.warning(f"Sample {sample_id} has no problem text")

    # Build the input messages from the prompt config
    input_messages = build_input_messages(problem, prompt)

    # Build the output entry
    output = {
        "id": sample_id,
        "question": problem,
        "expected_answer": expected_answer,
        "agent_ref": {"type": "responses_api_agents", "name": agent_ref},
        "responses_create_params": {
            "input": input_messages,
        },
    }

    # Add tools if available
    if tool_schemas:
        output["responses_create_params"]["tools"] = tool_schemas

    # Add verifier type if specified
    if verifier_type:
        output["verifier_type"] = verifier_type

    # Preserve additional fields from source
    preserved_fields = ["subset_for_metrics", "reference_solution", "level", "label"]
    for field in preserved_fields:
        if field in sample:
            output[field] = sample[field]

    return output


async def main():
    args = parse_args()

    # Validate input file
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Loaded before any output is written, so a missing prompt config fails the run instead of leaving raw problems.
    prompt = load_prompt(args.prompt_config)
    # Fill it once too, so a template error (unknown placeholder, stray brace) fails before the output is truncated.
    build_input_messages("", prompt)

    # Create output directory if needed
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Prompt config: {args.prompt_config}")
    logger.info(f"Tools: {args.tools}")
    if args.verifier_type:
        logger.info(f"Verifier type: {args.verifier_type}")

    # Get tool schemas
    logger.info("Loading tool schemas...")
    tool_schemas = await get_tool_schemas(
        tool_specs=args.tools,
        sandbox_host=args.sandbox_host,
        sandbox_port=args.sandbox_port,
    )
    logger.info(f"Loaded {len(tool_schemas)} tools: {[t.get('name') for t in tool_schemas]}")

    if prompt.system:
        logger.info(f"System prompt: {prompt.system[:100]}...")
    else:
        logger.info("No system prompt in config")

    # Process input file
    logger.info("Processing samples...")
    samples_processed = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for idx, line in enumerate(fin):
            if args.limit and idx >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {idx}: invalid JSON: {e}")
                continue

            output = process_sample(
                sample=sample,
                idx=idx,
                tool_schemas=tool_schemas,
                prompt=prompt,
                problem_field=args.problem_field,
                answer_field=args.answer_field,
                id_field=args.id_field,
                verifier_type=args.verifier_type,
                agent_ref=args.agent_ref,
            )

            fout.write(json.dumps(output) + "\n")
            samples_processed += 1

    logger.info(f"Processed {samples_processed} samples -> {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
