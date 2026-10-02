# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parser equivalence checks; requires vLLM 0.29.0, but no model weights or GPU.

Set INKLING_TOKENIZER_PATH to also run against an installed Inkling tokenizer.
"""

import json
import os
import runpy
import string
from collections import Counter
from pathlib import Path

import pytest


vllm = pytest.importorskip("vllm")
if vllm.__version__.split("+", 1)[0] != "0.29.0":
    pytest.skip("These plugins target vLLM 0.29.0", allow_module_level=True)

from tokenizers import Tokenizer, decoders, models  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402
from vllm.parser.inkling import INKLING_SPECIAL_TOKENS  # noqa: E402
from vllm.reasoning import ReasoningParserManager  # noqa: E402
from vllm.tokenizers import get_tokenizer  # noqa: E402
from vllm.tool_parsers import ToolParserManager  # noqa: E402


PLUGIN_DIR = Path(__file__).resolve().parents[1]
TOOL_PLUGIN = PLUGIN_DIR / "inkling_complete_tool_parser.py"
REASONING_PLUGIN = PLUGIN_DIR / "inkling_complete_reasoning_parser.py"
THINKING = "<|message_model|><|content_thinking|>Inspect first.<|end_message|>"


def tool_block(name: str, args: dict) -> str:
    return (
        "<|message_model|><|content_invoke_tool_json|>"
        + json.dumps({"name": name, "args": args}, ensure_ascii=False)
        + "<|end_message|>"
    )


CASES = {
    "plain": "<|message_model|><|content_text|>Finished.<|end_message|>",
    "thinking": THINKING,
    "empty_args": tool_block("bash", {}),
    "nested_escaped": tool_block("bash", {"command": 'printf "café ☃\\n"', "data": {"args": [None, True, 2.5]}}),
    "multiple_tools": tool_block("bash", {"command": "pwd"}) + tool_block("write", {"content": "hello\n"}),
    "schema_coercion": tool_block("bash", {"command": "true", "count": "2", "enabled": "true"}),
    "mixed": THINKING
    + tool_block("bash", {"command": "pwd"})
    + "<|message_model|><|content_text|>Done.<|end_message|>",
    "truncated_string": '<|message_model|><|content_invoke_tool_json|>{"name":"bash","args":{"command":"echo',
    "truncated_object": '<|message_model|><|content_invoke_tool_json|>{"name":"bash","args":{"command":"echo ok"}',
    "invalid_args": '<|message_model|><|content_invoke_tool_json|>{"name":"bash","args":42}<|end_message|>',
    "no_args": '<|message_model|><|content_invoke_tool_json|>{"name":"bash"}<|end_message|>',
    "large_args": THINKING + tool_block("bash", {"command": "printf 'fixture data\\n'\n" * 128}),
}


def normalize(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items() if key != "id"}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


@pytest.fixture(scope="module", autouse=True)
def register_plugins():
    ToolParserManager.import_tool_parser(str(TOOL_PLUGIN))
    ReasoningParserManager.import_reasoning_parser(str(REASONING_PLUGIN))


@pytest.fixture(
    scope="module", params=["portable", "inkling"] if os.getenv("INKLING_TOKENIZER_PATH") else ["portable"]
)
def tokenizer(request):
    if request.param == "inkling":
        return get_tokenizer(os.environ["INKLING_TOKENIZER_PATH"], tokenizer_mode="inkling", trust_remote_code=True)
    # Character-level BPE with the real delimiter strings keeps fixtures portable.
    pieces = ["<unk>", *sorted(INKLING_SPECIAL_TOKENS), *sorted(set(string.printable + "é☃"))]
    backend = Tokenizer(
        models.BPE(vocab={piece: index for index, piece in enumerate(pieces)}, merges=[], unk_token="<unk>")
    )
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", additional_special_tokens=sorted(INKLING_SPECIAL_TOKENS)
    )


@pytest.fixture
def request_body():
    return ChatCompletionRequest(
        model="fixture",
        messages=[{"role": "user", "content": "test"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "count": {"type": "integer"},
                            "enabled": {"type": "boolean"},
                        },
                    },
                },
            }
            for name in ("bash", "write")
        ],
    )


@pytest.mark.parametrize("case", CASES)
def test_complete_tool_outputs_match_stock(tokenizer, request_body, case):
    outputs = []
    for name in ("inkling", "inkling_complete_fast"):
        parser = ToolParserManager.get_tool_parser(name)(tokenizer, tools=request_body.tools)
        outputs.append(normalize(parser.extract_tool_calls(CASES[case], request_body)))
        assert parser._parser_engine._stream_arg_deltas is True
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("case", CASES)
def test_reasoning_counts_match_stock(tokenizer, case):
    ids = tokenizer.encode(CASES[case], add_special_tokens=False)
    counts = []
    for name in ("inkling", "inkling_count_fast"):
        parser = ReasoningParserManager.get_reasoning_parser(name)(tokenizer)
        assert parser.count_reasoning_tokens([]) == 0
        counts.append(parser.count_reasoning_tokens(ids))
        assert parser._counting_parser_engine._stream_arg_deltas is True
    assert counts[0] == counts[1]
    if case in ("thinking", "mixed", "large_args"):
        assert counts[0] > 0


@pytest.mark.parametrize("chunk_size", [1, 3, 17])
def test_tool_streaming_after_complete_parse_matches_stock(tokenizer, request_body, chunk_size):
    text = CASES["multiple_tools"]
    ids = tokenizer.encode(text, add_special_tokens=False)
    streams = []
    for name in ("inkling", "inkling_complete_fast"):
        parser = ToolParserManager.get_tool_parser(name)(tokenizer, tools=request_body.tools)
        parser.extract_tool_calls(CASES["nested_escaped"], request_body)
        previous_ids, previous_text, deltas = [], "", []
        for end in range(chunk_size, len(ids) + chunk_size, chunk_size):
            current_ids = ids[:end]
            current_text = tokenizer.decode(current_ids, skip_special_tokens=False)
            assert current_text.startswith(previous_text)
            delta = parser.extract_tool_calls_streaming(
                previous_text,
                current_text,
                current_text[len(previous_text) :],
                previous_ids,
                current_ids,
                current_ids[len(previous_ids) :],
                request_body.model_copy(update={"stream": True}),
            )
            deltas.append(normalize(delta))
            previous_ids, previous_text = current_ids, current_text
        deltas.append(normalize(parser.finish_streaming()))
        assert any(delta and delta.get("tool_calls") for delta in deltas)
        streams.append(deltas)
    assert streams[0] == streams[1]


def test_reasoning_streaming_and_reuse_match_stock(tokenizer):
    ids = tokenizer.encode(CASES["mixed"], add_special_tokens=False)
    results = []
    for name in ("inkling", "inkling_count_fast"):
        parser = ReasoningParserManager.get_reasoning_parser(name)(tokenizer)
        parser.count_reasoning_tokens(ids)
        previous_ids, previous_text, deltas = [], "", []
        for end in range(3, len(ids) + 3, 3):
            current_ids = ids[:end]
            current_text = tokenizer.decode(current_ids, skip_special_tokens=False)
            assert current_text.startswith(previous_text)
            delta = parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                current_text[len(previous_text) :],
                previous_ids,
                current_ids,
                current_ids[len(previous_ids) :],
            )
            deltas.append(normalize(delta))
            previous_ids, previous_text = current_ids, current_text
        assert parser._streaming_count_valid
        assert any(delta and delta.get("reasoning") for delta in deltas)
        results.append((deltas, parser.count_reasoning_tokens(ids)))
    assert results[0] == results[1]


@pytest.mark.parametrize("kind", ["tool", "reasoning"])
def test_complete_parsing_omits_partial_conversion(tokenizer, request_body, monkeypatch, kind):
    ids = tokenizer.encode(CASES["large_args"], add_special_tokens=False)
    measurements = []
    for fast in (False, True):
        if kind == "tool":
            parser = ToolParserManager.get_tool_parser("inkling_complete_fast" if fast else "inkling")(
                tokenizer, tools=request_body.tools
            )
            engine = parser._parser_engine
        else:
            parser = ReasoningParserManager.get_reasoning_parser("inkling_count_fast" if fast else "inkling")(
                tokenizer
            )
            parser._counting_parser_engine = parser._parser_engine_cls(tokenizer, **parser._parser_engine_kwargs)
            engine = parser._counting_parser_engine
        counts = Counter()
        converter = engine._arg_converter

        def counted(raw_args, partial):
            counts["partial" if partial else "complete"] += 1
            return converter(raw_args, partial)

        monkeypatch.setattr(engine, "_arg_converter", counted)
        output = (
            parser.extract_tool_calls(CASES["large_args"], request_body)
            if kind == "tool"
            else parser.count_reasoning_tokens(ids)
        )
        measurements.append((normalize(output), counts))
    assert measurements[0][0] == measurements[1][0]
    assert measurements[0][1]["partial"] > 0
    assert measurements[1][1]["partial"] == 0
    assert measurements[1][1]["complete"] > 0


@pytest.mark.parametrize("kind", ["tool", "reasoning"])
@pytest.mark.parametrize("previous", [False, True])
def test_conversion_flag_restored_after_failure(tokenizer, request_body, monkeypatch, kind, previous):
    if kind == "tool":
        parser = ToolParserManager.get_tool_parser("inkling_complete_fast")(tokenizer, tools=request_body.tools)
        engine = parser._parser_engine
        method = "extract_tool_calls_from_content"
    else:
        parser = ReasoningParserManager.get_reasoning_parser("inkling_count_fast")(tokenizer)
        parser._counting_parser_engine = parser._parser_engine_cls(tokenizer, **parser._parser_engine_kwargs)
        engine = parser._counting_parser_engine
        method = "_single_pass_parse"
    engine._stream_arg_deltas = previous

    def fail(*args, **kwargs):
        assert engine._stream_arg_deltas is False
        raise ValueError("fixture parse failure")

    monkeypatch.setattr(engine, method, fail)
    with pytest.raises(ValueError, match="fixture parse failure"):
        if kind == "tool":
            parser.extract_tool_calls(CASES["plain"], request_body)
        else:
            parser.count_reasoning_tokens(tokenizer.encode(THINKING, add_special_tokens=False))
    assert engine._stream_arg_deltas is previous


@pytest.mark.parametrize("path", [TOOL_PLUGIN, REASONING_PLUGIN])
@pytest.mark.parametrize("version", ["0.29.1", "0.30.0", "0.29.0rc1"])
def test_unvalidated_versions_rejected(monkeypatch, path, version):
    monkeypatch.setattr(vllm, "__version__", version)
    with pytest.raises(RuntimeError, match="validated vLLM 0.29.0"):
        runpy.run_path(str(path))
