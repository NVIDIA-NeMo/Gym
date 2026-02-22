#!/usr/bin/env python3
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
"""Convert questions.jsonl to test format with tool definitions."""

import argparse
import json


PROMPT = """You are a financial agent. You are given a question and you need to answer it using the tools provided.
You will not be able to interact with the user or ask clarifications, you must answer the question only based on the information provided.

You should answer all questions as if the current date is February 23, 2026.

You will have access to a data storage system. You can use this system to store parsed contents of SEC filings retrieved from the web.
You can then use the retrieve_information tool to answer questions or gather information from the stored documents using LLM-based prompts.
This data storage system is designed to help you avoid context window issues.

When you have the final answer, you should call the `submit_final_result` tool with it. Your submission will not be processed unless you call this tool.

You should include any necessary step-by-step reasoning, justification, calculations, or explanation in your answer. You will be evaluated both on the accuracy of the final answer, and the correctness of the supporting logic.

When possible, please provide any calculated answers to at least two decimal places (e.g. 18.78% rather than 19%). Please do not round intermediate steps in any calculations - you should only round your final answer.

At the end of your answer, you should provide your sources in a dictionary with the following format:
{{
    "sources": [
        {{
            "url": "https://example.com",
            "name": "Name of the source"
        }},
        ...
    ]
}}

Question:
"""

SEC_TOOLS = [
    {
        "type": "function",
        "name": "sec_filing_search",
        "description": "Search SEC EDGAR for company filings by stock ticker symbol. Returns at most 30 filing metadata entries (sorted by filing date, most recent first), including filing_url, form type, and report_date. It does not contain the full text of the filing.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'NVDA')"},
                "form_types": {
                    "type": "array",
                    "description": "(optional) Limits search to specific EDGAR form types (e.g., ['8-K'], ['DEF 14A', '10-K']). Default: ['10-K', '10-Q', 'DEF 14A']. Override to search other form types.",
                    "items": {"type": "string"},
                },
            },
            "required": ["ticker"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "download_and_parse_filing",
        "description": "This tool is used to download and parse an SEC filing and save it to the agent's data storage system. The tool will retrieve the filing from the URL provided, parse it from HTML to plain text, and save it to the data storage under the key provided. You can use the retrieve_information tool to later retrieve information about the stored filing.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The filing URL from sec_filing_search results"},
                "key": {
                    "type": "string",
                    "description": "The key to use when saving the result in the conversation's data storage.",
                },
            },
            "required": ["url", "key"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "retrieve_information",
        "description": "This tool allows you to retrieve data from previously saved documents from the agent's data storage system, by applying an LLM prompt to the stored document.\n\nTo use the tool, you will need to provide a prompt. This prompt will include both the query to be sent to the LLM, as well as the keys of files you have previously saved to the data storage system.\n\nThe {{key_name}} will be replaced with the full text of the document stored under that key before the query is sent.\n\nIMPORTANT: Your prompt MUST include at least one key from the data storage using this exact format: {{key_name}}.\nIf you don't use this exact format with double braces, the tool will fail to retrieve the information.\n\nYou can also optionally only pass *a portion* of each document to the LLM, rather than the entire document. This can be used to avoid token limit errors or improve efficiency.\nTo do so, use the input_character_ranges parameter to specify which portions of documents to extract.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt that will be passed to the LLM. You MUST include at least one data storage key in the format {{key_name}}. The content stored under each key will replace the {{key_name}} placeholder.",
                },
                "input_character_ranges": {
                    "type": "array",
                    "description": "An optional list of character range specifications for extracting only portions of documents. Each object should have 'key' (the document key), 'start' (start character index, inclusive), and 'end' (end character index, exclusive). By default, the full document is used.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "The document key from data storage"},
                            "start": {"type": "integer", "description": "The starting character index (inclusive)"},
                            "end": {"type": "integer", "description": "The ending character index (exclusive)"},
                        },
                        "required": ["key", "start", "end"],
                    },
                },
            },
            "required": ["prompt"],
        },
        "strict": False,
    },
]

SUBMIT_TOOL = {
    "type": "function",
    "name": "submit_final_result",
    "description": "Submits the final answer to the user. You should include your final answer, as well as any necessary reasoning, justification, calculations, and explanation. Finally, you should provide any sources used to answer the question.\n\nYou MUST use this tool to submit your final result. The user will not see your response if you do not use this tool to submit.\nYou will not be able to continue working after this tool is called; the conversation will be ended.",
    "parameters": {
        "type": "object",
        "properties": {"final_result": {"type": "string", "description": "The final result to submit to the agent"}},
        "required": ["final_result"],
    },
    "strict": False,
}

WEB_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": "Search the public internet for information. Returns up to 10 results, each containing a url, title, and content excerpt.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query"}},
        "required": ["query"],
    },
    "strict": False,
}


def convert(input_file, output_file, include_web_search):
    """Convert questions to test format."""
    tools = SEC_TOOLS + [SUBMIT_TOOL]
    if include_web_search:
        tools.append(WEB_TOOL)

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            question = data["question"]
            expected = data["expected_answer"]

            output = {
                "question": question,
                "expected_answer": expected,
                "responses_create_params": {
                    "input": [{"role": "user", "content": PROMPT + question, "type": "message"}],
                    "tools": tools,
                    "max_output_tokens": 32768,
                    "temperature": 0.6,
                    "parallel_tool_calls": False,
                },
            }

            f_out.write(json.dumps(output) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert questions.jsonl to test format")
    parser.add_argument("--input", "-i", required=True, help="Input questions JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output test JSONL file")
    parser.add_argument("--include-web-search", "-w", action="store_true", help="Include web search tool")
    args = parser.parse_args()

    convert(args.input, args.output, args.include_web_search)
    print(f"Converted {args.input} -> {args.output}")
    if args.include_web_search:
        print("Web search tool included")
    else:
        print("Web search tool not included")
