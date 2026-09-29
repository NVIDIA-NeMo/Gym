# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Prepare the Vals AI Finance Agent benchmark (public 50-question set).

Downloads public.csv from the Vals AI GitHub repository and converts each
question to Gym benchmark format with responses_create_params and tool
definitions matching the public finance_sec_search environment.

Reference:
    - Dataset: https://github.com/vals-ai/finance-agent
"""

import csv
import io
import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "finance_sec_search_benchmark.jsonl"
OUTPUT_FPATH_WEB = DATA_DIR / "finance_sec_search_benchmark_web_search.jsonl"

CSV_URL = "https://raw.githubusercontent.com/vals-ai/finance-agent/main/data/public.csv"

# ---------------------------------------------------------------------------
# Prompt — matches the public finance_sec_search convert_questions.py
# ---------------------------------------------------------------------------
PROMPT = """You are a financial agent. You are given a question and you need to answer it using the tools provided.
You will not be able to interact with the user or ask clarifications, you must answer the question only based on the information provided.

You should answer all questions as if the current date is February 23, 2026.

You will have access to a data storage system. You can use this system to store parsed contents of HTML pages retrieved from the web.
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

# ---------------------------------------------------------------------------
# Tool definitions — public finance_sec_search tools only
# (from resources_servers/finance_sec_search/scripts/convert_questions.py)
# ---------------------------------------------------------------------------
SEC_FILING_SEARCH_TOOL = {
    "type": "function",
    "name": "sec_filing_search",
    "description": "Search SEC EDGAR for company filings by stock ticker symbol. Returns filing metadata entries (sorted by filing date, most recent first), including filing_url, form type, and report_date. It does not contain the full text of the filing. Use form_types, start_date, and end_date to narrow results.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'NVDA')"},
            "form_types": {
                "type": "array",
                "description": "(optional) Limits search to specific EDGAR form types (e.g., ['10-K'], ['10-Q', '8-K']). Default: all form types.",
                "items": {"type": "string"},
            },
            "start_date": {
                "type": "string",
                "description": "(optional) Filter filings on or after this date (YYYY-MM-DD)",
            },
            "end_date": {
                "type": "string",
                "description": "(optional) Filter filings on or before this date (YYYY-MM-DD)",
            },
        },
        "required": ["ticker"],
    },
    "strict": False,
}

PARSE_HTML_TOOL = {
    "type": "function",
    "name": "parse_html_page",
    "description": "This tool is used to parse the contents of an HTML page and save it to the agent's data storage system. The tool will retrieve the HTML page from the URL provided, then parse it from HTML to plain text. Finally, it will save it to the agent's data storage system under the key provided. You can use the retrieve_information tool to later retrieve information about the stored page.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL of the HTML page to parse"},
            "key": {
                "type": "string",
                "description": "The key to use when saving the result in the conversation's data storage.",
            },
        },
        "required": ["url", "key"],
    },
    "strict": False,
}

RETRIEVE_INFORMATION_TOOL = {
    "type": "function",
    "name": "retrieve_information",
    "description": 'This tool allows you to retrieve data from previously saved documents from the agent\'s data storage system, by applying an LLM prompt to the stored document.\n\nTo use the tool, you will need to provide a prompt. This prompt will include both the query to be sent to the LLM, as well as the keys of files you have previously saved to the data storage system.\n\nFor example, if you want to analyze data stored under the key "financial_report", your prompt should look like the following:\n"Analyze the following financial report and extract the revenue figures: {{financial_report}}"\n\nThe {{key_name}} will be replaced with the full text of the document stored under that key before the query is sent.\n\nIMPORTANT: Your prompt MUST include at least one key from the data storage using this exact format: {{key_name}}. If you don\'t use this exact format with double braces, the tool will fail to retrieve the information.\n\nYou can also optionally only pass *a portion* of each document to the LLM, rather than the entire document. This can be used to avoid token limit errors or improve efficiency. To do so, use the input_character_ranges parameter to specify which portions of documents to extract. For example, if "financial_report" contains "Annual Report 2023" and you specify:  [{"key": "financial_report", "start": 1, "end": 6}], then only "nnual" will be inserted into the prompt (characters 1 through 5, as end is exclusive).',
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The prompt that will be passed to the LLM. You MUST include at least one data storage key in the format {{key_name}} - for example: 'Summarize this 10-K filing: {{company_10k}}'. The content stored under each key will replace the {{key_name}} placeholder.",
            },
            "input_character_ranges": {
                "type": "array",
                "description": "An optional list of character range specifications for extracting only portions of documents. Each object should have 'key' (the document key), 'start' (start character index, inclusive), and 'end' (end character index, exclusive). By default, the full document is used if this parameter is not provided or if a key is not included in the list.",
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
}

SUBMIT_TOOL = {
    "type": "function",
    "name": "submit_final_result",
    "description": "Submits the final answer to the user. You should include your final answer, as well as any necessary "
    "reasoning, justification, calculations, and explanation. Finally, you should provide any sources used to answer the question. "
    "You MUST use this tool to submit your final result. The user will not see your response if you do not use this tool to submit. "
    "You will not be able to continue working after this tool is called; the conversation will be ended.",
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
    "description": "Search the public internet for information. Each result will contain a url, a title, and one excerpt taken directly from the page.",
    "parameters": {
        "type": "object",
        "properties": {
            "search_query": {
                "type": "string",
                "description": "The query to search for",
            },
            "start_date": {
                "type": "string",
                "description": "(optional) The start date for the search range in the format YYYY-MM-DD",
            },
            "end_date": {
                "type": "string",
                "description": "(optional) The end date for the search range in the format YYYY-MM-DD",
            },
            "number_of_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return.",
                "maximum": 20,
                "minimum": 1,
                "default": 10,
            },
        },
        "required": ["search_query"],
    },
    "strict": False,
}

TOOLS = [RETRIEVE_INFORMATION_TOOL, PARSE_HTML_TOOL, SEC_FILING_SEARCH_TOOL, SUBMIT_TOOL]


def _convert_row(row: dict, include_web_search: bool = False) -> dict:
    """Convert a single CSV row to Gym benchmark format with tools and prompt."""
    question = row["Question"]
    expected_answer = row["Answer"]

    tools = [WEB_TOOL] + TOOLS if include_web_search else TOOLS

    record = {
        "question": question,
        "expected_answer": expected_answer,
        "responses_create_params": {
            "input": [{"role": "user", "content": PROMPT + question, "type": "message"}],
            "tools": tools,
        },
    }

    if "Question Type" in row and row["Question Type"]:
        record["question_type"] = row["Question Type"]
    if "Expert time (mins)" in row and row["Expert time (mins)"]:
        record["expert_time_mins"] = row["Expert time (mins)"]
    if "Rubric" in row and row["Rubric"]:
        record["rubric"] = row["Rubric"]

    return record


def prepare(include_web_search: bool = False) -> Path:
    """Download Vals AI public.csv and convert to Gym benchmark JSONL.

    Args:
        include_web_search: Include the web_search tool in tool definitions.
            Default False (gym eval prepare calls prepare() with no args).
            Pass True or use --include-web-search on the CLI.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_fpath = OUTPUT_FPATH_WEB if include_web_search else OUTPUT_FPATH

    print(f"Downloading public.csv from {CSV_URL} ...")
    with urllib.request.urlopen(CSV_URL) as resp:
        csv_text = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    print(f"Downloaded {len(rows)} questions (web_search={'enabled' if include_web_search else 'disabled'})")

    count = 0
    with open(output_fpath, "w", encoding="utf-8") as f:
        for row in rows:
            record = _convert_row(row, include_web_search=include_web_search)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} benchmark samples to {output_fpath}")
    return output_fpath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Vals AI finance_sec_search benchmark")
    parser.add_argument(
        "--include-web-search",
        action="store_true",
        help="Include web_search tool in tool definitions",
    )
    args = parser.parse_args()
    prepare(include_web_search=args.include_web_search)
