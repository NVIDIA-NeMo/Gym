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
"""Benchmark the Responses <-> Chat Completions conversion on synthetic conversations.

Measures request conversion, response conversion, and the CPU work of one model-server proxy hop
(request parse, conversion, outbound dump, backend reply validation, reply conversion, reply
serialization) for conversations with tools, image parts, reasoning and token metadata. Network
and engine time are excluded.

To compare two checkouts, run this script once per checkout with ``PYTHONPATH`` pointing at it and
alternate the runs. ``--dump-outputs`` writes the normalized converted outputs so two checkouts can
be compared for identical results:

    PYTHONPATH=<checkout> python scripts/benchmark_chat_completions_conversion.py --json out.json
"""

import argparse
import gc
import json
import re
import statistics
import time
import tracemalloc
from typing import Any, Callable, Dict, List

from nemo_gym.core import NeMoGymChatCompletion, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.models import ChatCompletionsConverter


IMAGE_URL = "data:image/png;base64," + "iVBORw0KGgo" * 64
TOOL_NAMES = [f"tool_{i}" for i in range(8)]


def _tokens(start: int, count: int) -> List[int]:
    return list(range(start, start + count))


def _token_metadata(turn: int, *, prompt_per_turn: int, prompt_cap: int, generation: int) -> Dict[str, Any]:
    prompt = min(prompt_per_turn * (turn + 1), prompt_cap)
    return {
        "prompt_token_ids": _tokens(1000, prompt),
        "generation_token_ids": _tokens(5000, generation),
        "generation_log_probs": [-0.125] * generation,
    }


def build_request(turns: int, *, token_metadata: bool) -> Dict[str, Any]:
    """A Responses request replaying ``turns`` completed agent turns plus a new user message."""
    tools = [
        {
            "type": "function",
            "name": name,
            "description": f"Tool {name} does something useful.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            "strict": True,
        }
        for name in TOOL_NAMES
    ]
    tools.append({"type": "custom", "name": "apply_patch", "description": "Apply a patch."})

    items: List[Dict[str, Any]] = [{"role": "developer", "content": "Follow the repository conventions."}]
    for turn in range(turns):
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": f"Turn {turn}: please inspect the file."}]
        if turn % 4 == 0:
            content.append({"type": "input_image", "image_url": IMAGE_URL, "detail": "auto"})
        items.append({"type": "message", "role": "user", "content": content})

        tokens = _token_metadata(turn, prompt_per_turn=64, prompt_cap=8192, generation=128) if token_metadata else {}
        items.append(
            {
                "type": "reasoning",
                "id": f"rs_{turn}",
                "summary": [{"type": "summary_text", "text": "I should call a tool. " * 8}],
            }
        )
        call_id = f"call_{turn}"
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{turn}",
                "call_id": call_id,
                "name": TOOL_NAMES[turn % len(TOOL_NAMES)],
                "arguments": json.dumps({"path": f"src/file_{turn}.py", "count": turn}),
                "status": "completed",
                **tokens,
            }
        )
        output: Any = "line\n" * 40
        if turn % 8 == 3:
            output = [{"type": "input_text", "text": "line\n" * 40}]
        items.append({"type": "function_call_output", "call_id": call_id, "output": output})
        if turn % 8 == 5:
            patch_id = f"patch_{turn}"
            items.append(
                {"type": "custom_tool_call", "call_id": patch_id, "name": "apply_patch", "input": "*** Begin Patch"}
                | tokens
            )
            items.append({"type": "custom_tool_call_output", "call_id": patch_id, "output": "applied"})
        items.append(
            {
                "type": "message",
                "id": f"msg_{turn}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Done with this step. " * 6, "annotations": []}],
                **tokens,
            }
        )
    items.append({"type": "message", "role": "user", "content": "Now summarize what changed."})

    return {
        "model": "policy",
        "input": items,
        "instructions": "You are a careful coding agent.",
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "max_output_tokens": 4096,
        "temperature": 1.0,
        "top_p": 1.0,
        "reasoning": {"effort": "high"},
        "metadata": {"task": "bench"},
    }


def build_chat_completion(turns: int, *, token_metadata: bool) -> Dict[str, Any]:
    """A backend chat completion with reasoning, a tool call, logprobs and token metadata."""
    generation = 2048
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "<think>" + "Let me think about the change. " * 40 + "</think>" + "Here is the summary. " * 20,
        "tool_calls": [
            {
                "id": "call_new",
                "type": "function",
                "function": {"name": TOOL_NAMES[0], "arguments": json.dumps({"path": "src/main.py"})},
            }
        ],
    }
    if token_metadata:
        message.update(
            prompt_token_ids=_tokens(1000, min(64 * (turns + 1), 32768)),
            generation_token_ids=_tokens(5000, generation),
            generation_log_probs=[-0.25] * generation,
        )
    return {
        "id": "chatcmpl-bench",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "policy",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": message,
                "logprobs": {
                    "content": [
                        {
                            "token": "x",
                            "logprob": -0.5,
                            "bytes": None,
                            "top_logprobs": [{"token": "x", "logprob": -0.5}],
                        }
                    ]
                    * 64
                },
            }
        ],
        "usage": {
            "prompt_tokens": 64 * (turns + 1),
            "completion_tokens": generation,
            "total_tokens": 64 * (turns + 1) + generation,
            "prompt_tokens_details": {"cached_tokens": 32},
            "completion_tokens_details": {"reasoning_tokens": 512},
        },
    }


_GENERATED_ID = re.compile(r"\b(resp|msg|rs)_[0-9a-f]{32}\b")


def normalize(value: Any) -> Any:
    """JSON-ready value with converter-minted UUIDs replaced, for differential comparison.

    Key order is kept, so two dumps compare equal only when the outputs match field for field and in order.
    """
    text = json.dumps(value, default=str)
    return json.loads(_GENERATED_ID.sub(r"\1_<generated>", text))


def _measure(fn: Callable[[], Any], *, iterations: int) -> Dict[str, float]:
    for _ in range(min(3, iterations)):
        fn()
    gc.collect()
    wall, cpu = [], []
    for _ in range(iterations):
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        fn()
        cpu.append(time.process_time() - start_cpu)
        wall.append(time.perf_counter() - start_wall)
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "cpu_ms_mean": 1000 * statistics.fmean(cpu),
        "cpu_ms_min": 1000 * min(cpu),
        "wall_ms_p50": 1000 * statistics.median(wall),
        "wall_ms_max": 1000 * max(wall),
        "peak_alloc_kib": peak / 1024,
    }


def run_case(turns: int, *, token_metadata: bool, iterations: int, dump: Dict[str, Any]) -> Dict[str, Any]:
    converter = ChatCompletionsConverter(
        return_token_id_information=token_metadata, custom_tool_strategy="function_bridge"
    )
    request_dict = build_request(turns, token_metadata=token_metadata)
    request_json = json.dumps(request_dict)
    request = NeMoGymResponseCreateParamsNonStreaming.model_validate(request_dict)
    chat_completion_dict = build_chat_completion(turns, token_metadata=token_metadata)
    chat_completion_json = json.dumps(chat_completion_dict)
    chat_completion = NeMoGymChatCompletion.model_validate(chat_completion_dict)

    request_before = request.model_dump()
    chat_completion_before = chat_completion.model_dump()

    def request_conversion() -> Any:
        return converter.responses_to_chat_completion_create_params(request)

    def response_conversion() -> Any:
        return converter.chat_completion_to_response(request, chat_completion)

    def proxy_hop() -> bytes:
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate_json(request_json)
        chat = converter.responses_to_chat_completion_create_params(body)
        json.dumps(chat.model_dump(exclude_unset=True))
        backend = NeMoGymChatCompletion.model_validate_json(chat_completion_json)
        return converter.chat_completion_to_response(body, backend).model_dump_json().encode()

    key = f"turns={turns},tokens={token_metadata}"
    dump[key] = {
        "chat_request": normalize(request_conversion().model_dump(exclude_unset=True)),
        "response": normalize(response_conversion().model_dump()),
        # Fields-set differences only show in an exclude_unset dump, which request replay uses.
        "response_exclude_unset": normalize(response_conversion().model_dump(exclude_unset=True)),
        "proxy_reply": normalize(json.loads(proxy_hop())),
    }
    assert request.model_dump() == request_before, "request conversion mutated its source"
    assert chat_completion.model_dump() == chat_completion_before, "response conversion mutated its source"

    return {
        "case": key,
        "request_conversion": _measure(request_conversion, iterations=iterations),
        "response_conversion": _measure(response_conversion, iterations=iterations),
        "proxy_hop": _measure(proxy_hop, iterations=iterations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--turns", type=int, nargs="+", default=[1, 32, 512])
    parser.add_argument("--iterations", type=int, default=0, help="Per case; 0 scales with conversation size.")
    parser.add_argument("--json", help="Write the measurements to this file.")
    parser.add_argument("--dump-outputs", help="Write the normalized converted outputs to this file.")
    args = parser.parse_args()

    results, dump = [], {}
    for turns in args.turns:
        iterations = args.iterations or max(20, 4000 // max(turns, 1))
        for token_metadata in (False, True):
            result = run_case(turns, token_metadata=token_metadata, iterations=iterations, dump=dump)
            results.append(result)
            for phase in ("request_conversion", "response_conversion", "proxy_hop"):
                m = result[phase]
                print(
                    f"{result['case']:<24} {phase:<20} cpu mean {m['cpu_ms_mean']:9.3f} ms  "
                    f"min {m['cpu_ms_min']:9.3f} ms  wall p50 {m['wall_ms_p50']:9.3f} ms  max {m['wall_ms_max']:9.3f} ms  "
                    f"peak {m['peak_alloc_kib']:10.1f} KiB"
                )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
    if args.dump_outputs:
        with open(args.dump_outputs, "w") as f:
            json.dump(dump, f)


if __name__ == "__main__":
    main()
