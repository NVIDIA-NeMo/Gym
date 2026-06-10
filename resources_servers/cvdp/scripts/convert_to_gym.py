#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Converts CVDP local_export prompts to NeMo-Gym format.
#
# Input mode 1: prompts.jsonl produced by CVDP's local_export mode, with fields:
#   {id, prompt, system, user}
#
# Also requires the original CVDP dataset for verifier_metadata (harness_files,
# target_files), which the resource server (app.py) needs to run the Docker harness.
#
# Input mode 2: raw CVDP agentic dataset, with fields:
#   {id, system_message, prompt, context, patch, harness, categories}
#
# Usage:
#   python convert_to_gym.py \
#       --prompts  prompts.jsonl \
#       --dataset  cvdp_dataset.jsonl \
#       --output   data/train.jsonl
#
# Or, for raw agentic datasets:
#   python convert_to_gym.py \
#       --dataset  cvdp_v1.1.0_agentic_code_generation_no_commercial.jsonl \
#       --output   data/train.jsonl
#
# By default, raw agentic conversion emits verifier-friendly prompts that ask
# for final RTL/code content. Use --raw-prompt-style original to expose the raw
# dataset system_message to the agent instead.

import argparse
import json
from pathlib import Path


# Code comprehension categories use subjective scoring, not docker-compose harness.
# They have no target_files or harness_files — instead they carry a reference answer.
CODE_COMPREHENSION_CATEGORIES = [6, 8, 9, 10]

BASE_SYSTEM_CONTEXT = """You are a helpful assistance.
Consider that you have a folder structure like the following:

    - rtl/*   : Contains files which are RTL code.
    - verif/* : Contains files which are used to verify the correctness of the RTL code.
    - docs/*  : Contains files used to document the project, like Block Guides, RTL Plans and Verification Plans.

When generating files, return the file name in the correct place at the folder structure.
"""

CATEGORY_GUIDANCE = {
    2: "You are solving an 'RTL Code Completion' problem. To solve this problem correctly, you should only respond with the RTL code generated according to the requirements.",
    3: "You are solving a 'Specification to RTL Translation' problem. To solve this problem correctly, you should only respond with the RTL code translated from the specification.",
    4: "You are solving an 'RTL Code Modification' problem. To solve this problem correctly, you should only respond with the modified RTL code according to the requirements.",
    5: "You are solving a 'Specification to RTL Translation: Module Instantiation and Component Reuse' problem. To solve this problem correctly, you should only respond with the RTL code translated from the specification and with proper module instantiation and component reuse.",
    6: "You are solving an 'RTL Correspondence' problem. To solve this problem correctly, you should only respond with a verbatim quote from the specification (if the context is RTL) that corresponds to the RTL code snippet or verbatim RTL source code (if the context is specification) that corresponds to the specification snippet.",
    7: "You are solving an 'RTL Lint Improvement or Power-Performance Optimization' problem. To solve this problem correctly, you should only respond with improved RTL code to address lint issues or optimize for power/performance.",
    8: "You are solving a 'Testbench Correspondence' problem. To solve this problem correctly, you should only respond with the verbatim testbench code that corresponds to the test plan snippet (if the context is testbench code) or verbatim test plan that corresponds to the testbench code snippet (if the context is testbench code).",
    9: "You are solving a 'Question & Answer on RTL' problem. To solve this problem correctly, you should only respond with a detailed answer to the question about RTL.",
    10: "You are solving a 'Question & Answer on Testbench' problem. To solve this problem correctly, you should only respond with a detailed answer to the question about the testbench.",
    12: "You are solving a 'Test Plan to Testbench Stimulus Generation' problem. To solve this problem correctly, you should only respond with the testbench stimulus code generated based on the test plan specification.",
    13: "You are solving a 'Test Plan to Testbench Checker Generation' problem. To solve this problem correctly, you should only respond with the testbench checker code generated based on the test plan specification.",
    14: "You are solving a 'Test Plan to Assertions Generation' problem. To solve this problem correctly, you should only respond with the assertions for the testbench based on the test plan specification.",
    16: "You are solving an 'RTL Debugging and Bug Fixing' problem. To solve this problem correctly, you should only respond with the RTL code that is debugged and fixed to address the bug.",
}

MULTI_FILE_SCHEMA = ['{ "code": [{ "<name>" : "<code>"}] }', '{ "response": "<response>" }']


def _get_category_num(entry: dict) -> int | None:
    """Extract the numeric category from the categories list (e.g. 'cid010' -> 10)."""
    categories = entry.get("categories", [])
    if categories and isinstance(categories[0], str) and categories[0].startswith("cid"):
        return int(categories[0][3:])
    return None


def _get_target_files(entry: dict) -> list[str]:
    """Target files from transformed output.context or raw agentic patch keys."""
    output_context = (entry.get("output") or {}).get("context") or {}
    if output_context:
        return list(output_context.keys())

    patch = entry.get("patch") or {}
    if isinstance(patch, dict):
        return list(patch.keys())

    return []


def _get_harness_files(entry: dict) -> dict[str, str | None]:
    """Docker-compose + test scripts — passed as-is, matching CVDP.

    Transformed rows store these under harness.files. Raw agentic rows store
    the file map directly under harness.
    """
    harness = entry.get("harness") or {}
    if not isinstance(harness, dict):
        return {}
    if "files" in harness and isinstance(harness["files"], dict):
        return harness["files"]
    return harness


def _get_context_files(entry: dict) -> dict[str, str]:
    """Companion RTL files from input.context that the model doesn't generate
    but are needed for compilation (e.g. floor_to_seven_segment.sv).

    Returns input.context files that are NOT in output.context (i.e. not
    target files the model is asked to produce)."""
    input_context = (entry.get("input") or {}).get("context") or {}
    if not input_context:
        input_context = entry.get("context") or {}
    target_keys = set(_get_target_files(entry))
    return {k: v for k, v in input_context.items() if k not in target_keys and v}


def _get_subjective_reference(entry: dict) -> str | None:
    """Reference answer for code-comprehension categories — from output.response."""
    return (entry.get("output") or {}).get("response")


def _create_system_prompt(entry: dict, target_files: list[str]) -> str:
    cat_num = _get_category_num(entry)
    system = BASE_SYSTEM_CONTEXT
    if cat_num in CATEGORY_GUIDANCE:
        system += f"\n{CATEGORY_GUIDANCE[cat_num]}\n"

    if len(target_files) > 1:
        system += "\nProvide the response in one of the following JSON schemas: \n"
        system += "\nor\n".join(MULTI_FILE_SCHEMA)
        system += (
            "\nThe response should be in JSON format, including double-quotes around keys and values, "
            "and proper escaping of quotes within values, and escaping of newlines."
        )

    return system


def _create_user_prompt(entry: dict, target_files: list[str]) -> str:
    parts: list[str] = []
    context_files = entry.get("context") or (entry.get("input") or {}).get("context") or {}
    for filepath, content in context_files.items():
        parts.append(f"\nConsider the following content for the file {filepath}:\n```\n{content}\n```")

    prompt = entry.get("prompt") or (entry.get("input") or {}).get("prompt") or ""
    if prompt:
        parts.append(f"\nProvide me one answer for this request: {prompt}")

    if len(target_files) == 1:
        parts.append(
            "\nPlease provide your response as plain text without any JSON formatting. "
            f"Your response will be saved directly to: {target_files[0]}."
        )
    elif target_files:
        parts.append(f"\nName the files as: {target_files}.")

    return "\n".join(parts)


def _make_verifier_metadata(raw: dict, task_id: str, target_files: list[str]) -> dict:
    context_files = _get_context_files(raw)
    verifier_metadata = {
        "task_id": task_id,
        "categories": raw.get("categories", []),
        "difficulty": raw.get("difficulty", ""),
        "target_files": target_files,
        "harness_files": _get_harness_files(raw),
    }
    if context_files:
        verifier_metadata["context_files"] = context_files
    return verifier_metadata


def _convert_row(prompt_row: dict, raw: dict, task_id: str, raw_prompt_style: str) -> dict | None:
    cat_num = _get_category_num(raw)
    is_comprehension = cat_num is not None and cat_num in CODE_COMPREHENSION_CATEGORIES

    target_files = _get_target_files(raw)
    if not target_files and not is_comprehension:
        return None

    verifier_metadata = _make_verifier_metadata(raw, task_id, target_files)

    if is_comprehension:
        ref = _get_subjective_reference(raw)
        if not ref:
            print(f"WARNING: no output.response for comprehension task {task_id}, skipping")
            return None
        verifier_metadata["subjective_reference"] = ref

    if "system" in prompt_row and "user" in prompt_row:
        system = prompt_row["system"]
        user = prompt_row["user"]
    elif raw_prompt_style == "original":
        system = prompt_row.get("system_message") or ""
        user = _create_user_prompt(prompt_row, target_files)
    else:
        system = _create_system_prompt(prompt_row, target_files)
        user = _create_user_prompt(prompt_row, target_files)

    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        },
        "verifier_metadata": verifier_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CVDP data to NeMo-Gym format")
    parser.add_argument(
        "--prompts",
        help="prompts.jsonl from CVDP local_export mode. Omit for direct raw agentic conversion.",
    )
    parser.add_argument(
        "--raw-prompt-style",
        choices=["verifier_friendly", "original"],
        default="verifier_friendly",
        help=(
            "Only used when --prompts is omitted. verifier_friendly asks for final code in the "
            "format the CVDP verifier parses; original uses the dataset system_message verbatim."
        ),
    )
    parser.add_argument("--dataset", required=True, help="CVDP dataset JSONL (for verifier_metadata)")
    parser.add_argument("--output", required=True, help="Output NeMo-Gym JSONL")
    args = parser.parse_args()

    # Index dataset by id for verifier_metadata lookup.
    dataset: dict[str, dict] = {}
    with open(args.dataset) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                dataset[entry["id"]] = entry

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with open(args.output, "w") as fout:
        if args.prompts:
            row_iter = open(args.prompts)
        else:
            row_iter = open(args.dataset)

        with row_iter as fin:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = row["id"]

                if task_id not in dataset:
                    skipped += 1
                    continue

                raw = dataset[task_id]
                gym_row = _convert_row(row, raw, task_id, args.raw_prompt_style)
                if gym_row is None:
                    skipped += 1
                    continue

                fout.write(json.dumps(gym_row) + "\n")
                written += 1

    print(f"Wrote {written} entries to {args.output}")
    if skipped:
        print(f"Skipped {skipped} entries (no dataset match, no target files, or missing reference)")


if __name__ == "__main__":
    main()
