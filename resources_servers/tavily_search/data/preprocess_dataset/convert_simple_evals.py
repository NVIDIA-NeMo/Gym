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
import copy
import json
import os
import random
from pathlib import Path


# Add parent directory to path so we can import from app.py

random.seed(42)


# SFT_DATA_FILE_PATH="/lustre/fsw/portfolios/llmservice/users/rgala/repos/research-scratch/11_12_search/data/12_14_postprocessed_exa_search_tool_sft_1395samples_with_think_and_tools.jsonl"
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Final Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

TOOLS = [
    {
        "type": "function",
        "name": "web_search",
        "description": "Search the web for a query and return up to 10 search results with <link, summary> for each result.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The term to search for"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": False,
    }
]

# ---------- New tool definitions ----------

FIND_IN_PAGE_TOOL = {
    "type": "function",
    "name": "find_in_page",
    "description": (
        "Extract and search content from a specific URL. "
        "Uses the query to find and return the most relevant content chunks from the page."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the page to extract content from",
            },
            "query": {
                "type": "string",
                "description": "The search query to find relevant content on the page",
            },
        },
        "required": ["url", "query"],
        "additionalProperties": False,
    },
    "strict": False,
}

SCROLL_PAGE_TOOL = {
    "type": "function",
    "name": "scroll_page",
    "description": (
        "Retrieve the content of a web page and return a specific section of it as words. "
        "Use start_index and n to paginate through the page content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the page to retrieve content from",
            },
            "start_index": {
                "type": "integer",
                "description": "The word index to start reading from (default: 0)",
            },
            "n": {
                "type": "integer",
                "description": "The number of words to return (default: 2000)",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    "strict": False,
}

NEW_TOOLS = [FIND_IN_PAGE_TOOL, SCROLL_PAGE_TOOL]
ALL_TOOLS = TOOLS + NEW_TOOLS

# ---------- System prompt for new tools ----------

SYSTEM_PROMPT = """
Please think step by step and reason about the problem.

## Steps to approach Queries:
1. Identify Core Dependencies: Break the riddle into individual logical constraints (e.g., dates, amounts, specific entities, geographic markers).
2. Establish Hierarchical Priorities: Determine which constraints are "Hard" (must be an exact match) and which are "Soft" (potentially subject to translation or paraphrasing).
3. Identify Anchor Values: Identify "High-Specificity Strings": Isolate technical jargon or precise dates and use them in quotation marks to force exact-match results. If a prompt uses specific terminology, prioritize those specific phrases in the query.
4. Backtracking and Pivoting: If a lead requires more than a few queries without a high-confidence match, discard the geographic or character-based assumption and re-evaluate the next most likely candidate.
5. Avoid "Popularity Bias": Do not prioritize famous entities or frequent news topics unless they satisfy every logical constraint. Focus on the most appropriate match, regardless of how obscure it may be.
6.Verification Against All Constraints: Before finalization, cross-reference the proposed answer against every single data point in the prompt to ensure zero contradictions.


##Tools:

You are encouraged to use the tools provided to you to solve the problem, to make sure you can get to the right answer. You must only issue one tool call at a time. Once you are done issuing calls, then return your final answer

- web_search(query): Search the web for a query and return up to 10 search results with <link, summary> for each result.
- find_in_page(url, query): Extract and search content from a specific URL. Given a URL and a query, it returns the most relevant content chunks from that page. Use this when you find a promising search result and want to look for specific information on that page.
- scroll_page(url, start_index=0, n=2000): Retrieve the full content of a web page and return a window of n words starting at start_index. Use this to read through a page's content. You can paginate by increasing start_index. The response includes total_words so you know how much content is available.
"""


def read_jsonl_file(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def update_system_prompt(input_messages: list) -> list:
    """Replace/append the system message with the new SYSTEM_PROMPT."""
    for msg in input_messages:
        if msg.get("role") == "system":
            msg["content"] = SYSTEM_PROMPT.strip()
            break
    else:
        # No system message found -- prepend one
        input_messages.insert(
            0,
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
            },
        )
    return input_messages


def add_tools_to_dataset(input_path: Path, output_path: Path, skip_system_prompt: bool = False) -> None:
    """Read input JSONL, append new tools to each record, write output JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    flag = True
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            # Get the existing tools list (or create one if missing)
            rcp = record.setdefault("responses_create_params", {})
            tools = rcp.setdefault("tools", [])

            # Only add tools that aren't already present (by name)
            existing_names = {t["name"] for t in tools}
            for tool in NEW_TOOLS:
                if tool["name"] not in existing_names:
                    tools.append(tool)

            # Update the system prompt with new-tool guidance
            if not skip_system_prompt:
                input_msgs = rcp.setdefault("input", [])
                update_system_prompt(input_msgs)

            f_out.write(json.dumps(record) + "\n")

            if flag:
                print(json.dumps(record, indent=4))
                flag = False
            count += 1

    print(f"Processed {count} records")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    if skip_system_prompt:
        print("  System prompt update: SKIPPED")
    else:
        print("  System prompt update: appended new-tool guidance")


def map_sft_sample_to_rl_sample(sft_sample):
    def set_question_in_query_template(messages):
        assert len(messages) >= 2
        question = messages[1]["content"]
        question_in_query_template = QUERY_TEMPLATE.format(Question=question)
        messages[1]["content"] = question_in_query_template
        return messages

    def strip_messages_to_no_asst(messages):
        assert len(messages) >= 2
        assert messages[0]["role"] == "system" or messages[0]["role"] == "developer"
        assert messages[1]["role"] == "user"
        messages[0]["role"] = "developer"
        messages = set_question_in_query_template(messages)
        return copy.deepcopy(messages[:2])

    def get_question(messages):
        assert len(messages) >= 2
        assert messages[1]["role"] == "user"
        return messages[1]["content"]

    # def reformat_tools(tools):
    # new_tools = []
    # for tool in tools:
    #     new_tools.append({
    #         "type": "function",
    #         **tool["function"]
    #     })
    #     new_tools[-1]["parameters"]["required"] = new_tools[-1]["parameters"].get("required", ["query"])
    #     new_tools[-1]["parameters"]["additionalProperties"] = False
    #     new_tools[-1]["strict"] = False #got 422 error without this.
    # return new_tools

    messages_with_system_prompt_and_question = strip_messages_to_no_asst(sft_sample["messages"])
    question = get_question(messages_with_system_prompt_and_question)

    messages_with_system_prompt_and_question = update_system_prompt(messages_with_system_prompt_and_question)

    responses_create_params = {"input": messages_with_system_prompt_and_question, "tools": ALL_TOOLS}

    return {
        "responses_create_params": responses_create_params,
        "ground_truth": sft_sample["metadata"]["final_answer_entity"],
        "question": question,
    }


def map_benchmark_sample_to_rl_sample(benchmark_sample):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": QUERY_TEMPLATE.format(Question=benchmark_sample["problem"])},
    ]

    responses_create_params = {"input": messages, "tools": ALL_TOOLS}

    return {
        "responses_create_params": responses_create_params,
        "ground_truth": benchmark_sample["answer"],
        "question": benchmark_sample["problem"],
    }


def test_final_sample(sample):
    required_keys = ["responses_create_params", "ground_truth", "question"]
    for key in required_keys:
        if key not in sample:
            raise ValueError(f"Key {key} not found in sample")
    return sample


def write_benchmark_samples(
    rl_samples, output_data_folder, completed_test_set_file_name, n_100_file_name, n_30_file_name
):
    random.shuffle(rl_samples)
    with open(os.path.join(output_data_folder, completed_test_set_file_name), "w") as file:
        for rl_sample in rl_samples:
            file.write(json.dumps(rl_sample) + "\n")
    with open(os.path.join(output_data_folder, n_100_file_name), "w") as file:
        for rl_sample in rl_samples[:100]:
            file.write(json.dumps(rl_sample) + "\n")
    with open(os.path.join(output_data_folder, n_30_file_name), "w") as file:
        for rl_sample in rl_samples[:30]:
            file.write(json.dumps(rl_sample) + "\n")


def write_train_validation_samples(rl_samples, output_data_folder, train_file_name, validation_file_name):
    random.shuffle(rl_samples)
    with open(os.path.join(output_data_folder, train_file_name), "w") as file:
        for rl_sample in rl_samples[:-100]:
            file.write(json.dumps(rl_sample) + "\n")
    with open(os.path.join(output_data_folder, validation_file_name), "w") as file:
        for rl_sample in rl_samples[-100:]:
            file.write(json.dumps(rl_sample) + "\n")


if __name__ == "__main__":
    INPUT_DIR = Path(
        "/lustre/fsw/portfolios/llmservice/users/rgala/repos/abhibha-browsecomp"
        "/nemo-gym/resources_servers/tavily_search/data/benchmark/browsecomp"
    )
    OUTPUT_DIR = Path(
        "/lustre/fsw/portfolios/llmservice/users/rgala/repos/browsecomp-integration-finsih"
        "/resources_servers/tavily_search/data/benchmark/browsecomp"
    )

    SOURCE_FILES = [
        "browsecomp_test_set.jsonl",
        "browsecomp_n_10.jsonl",
        "browsecomp_n_100.jsonl",
        "browsecomp_n_200.jsonl",
        "browsecomp_n_400.jsonl",
        "browsecomp_n_600.jsonl",
        "browsecomp_n_800.jsonl",
        "browsecomp_n_1000.jsonl",
        "browsecomp_n_1200.jsonl",
    ]

    for filename in SOURCE_FILES:
        input_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / filename
        print(f"\n{'=' * 60}")
        print(f"Processing: {filename}")
        print(f"{'=' * 60}")
        add_tools_to_dataset(input_path, output_path)
