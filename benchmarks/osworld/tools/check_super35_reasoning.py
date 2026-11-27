# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression checks using the real vLLM 0.32.0 parser and local tokenizer.

Run inside the serving environment, not Gym's dependency environment. No model
weights, model generation, desktop actions, or GPU computation are required.
The JSON report retains every fixture, token ID, and parsed result.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path


VALID_TOOL = (
    "<tool_call><function=exec_py><parameter=code_block>pyautogui.click(0.5, 0.5)</parameter></function></tool_call>"
)
ORIGINAL_REASONING = (
    'The search for "volume" showed VOD results, which is not what I need. '
    'Let me clear the search and try searching for "color" or "style" to find '
    "appearance-related settings. Let me clear the search box first."
)
ORIGINAL_CONTENT = (
    "\nExecute the desktop action.\n</think>\nExecute the desktop action.\n"
    "<tool_call>\n<function=exec_py>\n<parameter=code_block>\n"
    "pyautogui.tripleClick(0.372, 0.231)\n</parameter>\n</function>\n</tool_call>"
)
INVALID_PYTHON = "<tool_call><function=exec_py><parameter=code_block>print([1,2)</parameter></function></tool_call>"
INVALID_JSON = '<tool_call>{"name":"exec_py","arguments":{"code_block":[1,2}</tool_call>'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path, help="Local pinned checkpoint/tokenizer directory")
    parser.add_argument(
        "--output", required=True, type=Path, help="New JSON report; refuses to overwrite an old report"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; keep earlier evidence and choose a new path")

    import vllm
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser import ParserManager
    from vllm.reasoning import ReasoningParserManager

    if vllm.__version__.split("+")[0] != "0.32.0":
        raise RuntimeError("These regression expectations are pinned to vLLM 0.32.0")
    ReasoningParserManager.import_reasoning_parser("super35_reasoning")
    registered = ReasoningParserManager.get_reasoning_parser("nemotron_v3_lossless_nonstream")
    ReasoningParserManager.import_reasoning_parser("super35_reasoning")
    assert ReasoningParserManager.get_reasoning_parser("nemotron_v3_lossless_nonstream") is registered
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    native_class = ParserManager.get_parser(tool_parser_name=None, reasoning_parser_name="nemotron_v3")
    candidate_class = ParserManager.get_parser(
        tool_parser_name=None, reasoning_parser_name="nemotron_v3_lossless_nonstream"
    )
    request = ChatCompletionRequest(
        model="nemotron-3.5-super-vl",
        messages=[{"role": "user", "content": "Parser regression fixture"}],
        tool_choice="none",
        skip_special_tokens=False,
        chat_template_kwargs={"enable_thinking": True},
        stream=False,
    )
    # Expected fields are explicit, including malformed text that must survive.
    cases = [
        (
            "original_r5_call12",
            ORIGINAL_REASONING + "\n</think>" + ORIGINAL_CONTENT,
            ORIGINAL_REASONING,
            ORIGINAL_CONTENT,
        ),
        ("ordinary", "reasoning</think>\n" + VALID_TOOL, "reasoning", "\n" + VALID_TOOL),
        (
            "reasoning_trailing_whitespace",
            "reasoning \n\t</think>\n" + VALID_TOOL + "\n",
            "reasoning",
            "\n" + VALID_TOOL + "\n",
        ),
        ("no_boundary", "unfinished reasoning " + VALID_TOOL, "unfinished reasoning " + VALID_TOOL, None),
        (
            "missing_tool_end",
            "reasoning</think>" + VALID_TOOL.removesuffix("</tool_call>"),
            "reasoning",
            VALID_TOOL.removesuffix("</tool_call>"),
        ),
        ("invalid_python", "reasoning</think>" + INVALID_PYTHON, "reasoning", INVALID_PYTHON),
        ("invalid_json", "reasoning</think>" + INVALID_JSON, "reasoning", INVALID_JSON),
        ("repeated_end", "reasoning</think>text</think>" + VALID_TOOL, "reasoning", "text</think>" + VALID_TOOL),
        (
            "repeated_start",
            "reasoning</think><think>literal</think>" + VALID_TOOL,
            "reasoning",
            "<think>literal</think>" + VALID_TOOL,
        ),
        ("empty_content", "reasoning</think>", "reasoning", ""),
        ("empty_reasoning", "</think>" + VALID_TOOL, None, VALID_TOOL),
        ("opening_marker_in_generation", "<think>reasoning</think>" + VALID_TOOL, "<think>reasoning", VALID_TOOL),
    ]
    rows = []
    for name, original, reasoning, content in cases:
        token_ids = tokenizer.encode(original, add_special_tokens=False)
        native = native_class(tokenizer, chat_template_kwargs={"enable_thinking": True})
        candidate = candidate_class(tokenizer, chat_template_kwargs={"enable_thinking": True})
        native_result = native.parse(original, request, enable_auto_tools=False, model_output_token_ids=token_ids)
        result = candidate.parse(original, request, enable_auto_tools=False, model_output_token_ids=token_ids)
        assert result == (reasoning, content, []), (name, result)
        assert candidate.count_reasoning_tokens(token_ids) == native.count_reasoning_tokens(token_ids), name
        if name in {
            "ordinary",
            "reasoning_trailing_whitespace",
            "missing_tool_end",
            "invalid_python",
            "invalid_json",
            "empty_reasoning",
        }:
            assert result == native_result, name
        if name in {"original_r5_call12", "repeated_end", "repeated_start"}:
            assert native_result != result, f"Native boundary corruption no longer reproduces: {name}"
        rows.append(
            {
                "case": name,
                "original": original,
                "token_ids": token_ids,
                "native": native_result,
                "candidate": result,
                "expected": [reasoning, content, []],
                "reasoning_tokens": candidate.count_reasoning_tokens(token_ids),
            }
        )

    rejected = []
    for name, updates in [
        ("streaming_request", {"stream": True}),
        ("thinking_disabled", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("force_nonempty_content", {"chat_template_kwargs": {"force_nonempty_content": True}}),
    ]:
        try:
            registered(tokenizer).extract_reasoning(
                "reasoning</think>" + VALID_TOOL, request.model_copy(update=updates)
            )
        except ValueError as error:
            rejected.append({"case": name, "error": str(error)})
        else:
            raise AssertionError(f"Unsupported request was accepted: {name}")
    try:
        registered(tokenizer).extract_reasoning_streaming()
    except ValueError as error:
        rejected.append({"case": "streaming_entrypoint", "error": str(error)})
    else:
        raise AssertionError("Streaming entrypoint was accepted")

    plugin = Path(inspect.getfile(registered))
    report = {
        "state": "passed",
        "vllm_version": vllm.__version__,
        "plugin_sha256": hashlib.sha256(plugin.read_bytes()).hexdigest(),
        "repeat_module_import": "same registered class",
        "cases": rows,
        "unsupported_requests": rejected,
        "model_outputs_repaired": False,
        "desktop_actions_executed": 0,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"state": "passed", "cases": len(rows), "unsupported_requests": len(rejected)}))


if __name__ == "__main__":
    main()
