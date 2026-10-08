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

EDGAR_SEARCH_TOOL = {
    "type": "function",
    "name": "edgar_search",
    "description": "Search the EDGAR Database through the SEC API. You should provide a search query. You can also optionally provide a start date, an end date, a page number, top N results, a list of form types, and/or a list of CIKs. The results are returned as a list of dictionaries, each containing the metadata for a filing. It does not contain the full text of the filing.",
    "parameters": {
        "type": "object",
        "properties": {
            "search_query": {
                "type": "string",
                "description": 'The case-insensitive search-term or phrase to search the contents of fillings and their attachments. This can be a single word, phrase, or combination of words and phrases. Supported search features include wildcards (*), Boolean operators (OR, NOT), and exact phrase matching by enclosing phrases in quotation marks ("exact phrase"). By default, all terms are joined by an implicit AND operator.',
            },
            "form_types": {
                "type": "array",
                "description": "(optional) Limits search to specific EDGAR form types (e.g., ['8-K', '10-Q']) list of strings. Default: all form types",
                "items": {"type": "string"},
            },
            "ciks": {
                "type": "array",
                "description": '(optional) Filters results to filings from specified CIKs, type list of strings. Leading zeros are optional but may be included. Example: [ "0001811414", "1318605" ]. Default: all CIKs',
                "items": {"type": "string"},
            },
            "start_date": {
                "type": "string",
                "description": "(optional) Start date for the search range in yyyy-mm-dd format. If the value is a date that is later than 2025-04-07, it will be set to 2025-04-07.",
                "default": "1900-01-01",
            },
            "end_date": {
                "type": "string",
                "description": "(optional) End date for the search range, in the same format as startDate. If the value is a date that is later than 2025-04-07, it will be set to 2025-04-07.",
                "default": "2025-04-07",
            },
            "page": {
                "type": "integer",
                "description": "(optional) Used for pagination. Each page contains up to 100 matching filings. Increase the page number to retrieve the next set of 100 filings. Example: 3 retrieves the third page. Default: 1",
                "default": 1,
            },
            "top_n_results": {
                "type": "integer",
                "description": "(optional) Return only the first N results out of 100 from the page. If not provided, all 100 results will be returned. E.g. if page is 2, and number_of_results is 10, you will receive results 100 to 110.",
                "maximum": 100,
                "default": 100,
            },
        },
        "required": ["search_query"],
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


SEARCH_TOOLS = {
    "sec_filing_search": SEC_FILING_SEARCH_TOOL,
    "edgar_search": EDGAR_SEARCH_TOOL,
}


def convert_entry(
    data: dict,
    include_web_search: bool = False,
    search_tool: str = "sec_filing_search",
) -> dict:
    """Convert a single question entry to test format with tool definitions.

    Args:
        data: Dict with "question" and "expected_answer" keys.
        include_web_search: Whether to include the web search tool.
        search_tool: Which filing-search tool to expose. One of
            "sec_filing_search" (ticker-based metadata lookup) or
            "edgar_search" (full-text search).

    Returns:
        Converted dict with responses_create_params and tools.
    """
    if search_tool not in SEARCH_TOOLS:
        raise ValueError(f"search_tool must be one of {sorted(SEARCH_TOOLS)}, got '{search_tool}'")

    question = data.get("question") or data.get("problem", "")

    tools = [RETRIEVE_INFORMATION_TOOL, PARSE_HTML_TOOL, SEARCH_TOOLS[search_tool]]
    if include_web_search:
        tools = [WEB_TOOL] + tools
    tools.append(SUBMIT_TOOL)

    return {
        **data,
        "question": question,
        "responses_create_params": {
            "input": [{"role": "user", "content": PROMPT + question, "type": "message"}],
            "tools": tools,
            "parallel_tool_calls": True,
            "metadata": {
                "chat_template_kwargs": '{"enable_thinking": true}',
            },
        },
    }


def convert_file(input_file, output_file, include_web_search=False, search_tool="sec_filing_search"):
    """Convert a questions JSONL file to test format."""
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            output = convert_entry(data, include_web_search, search_tool)
            f_out.write(json.dumps(output) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert questions.jsonl to test format")
    parser.add_argument("--input", "-i", required=True, help="Input questions JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output test JSONL file")
    parser.add_argument("--include-web-search", "-w", action="store_true", help="Include web_search tool")
    parser.add_argument(
        "--search-tool",
        "-s",
        choices=sorted(SEARCH_TOOLS),
        default="sec_filing_search",
        help="Which filing-search tool to expose (default: sec_filing_search)",
    )
    args = parser.parse_args()

    convert_file(args.input, args.output, args.include_web_search, args.search_tool)
    print(f"Converted {args.input} -> {args.output}")
    sample = convert_entry({"question": "", "expected_answer": ""}, args.include_web_search, args.search_tool)
    tools_list = [t["name"] for t in sample["responses_create_params"]["tools"]]
    print(f"Tools: {', '.join(tools_list)}")
