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
from copy import deepcopy

from responses_api_agents.opencode_agent.token_tracking import (
    attach_parent_tokens,
    chain_output_items,
    main_chain,
    record_call,
)


def image(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def reply(content: str, prompt: list[int], gen: list[int], tool_ids: tuple[str, ...] = ()) -> dict:
    message = {
        "role": "assistant",
        "content": content,
        "prompt_token_ids": prompt,
        "generation_token_ids": gen,
        "generation_log_probs": [-0.5] * len(gen),
    }
    if tool_ids:
        message["tool_calls"] = [
            {"id": i, "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
            for i in tool_ids
        ]
    return message


def echoed(message: dict) -> dict:
    """The assistant message as OpenCode sends it back: no token fields, reasoning stripped."""
    out = {k: v for k, v in message.items() if k in ("role", "tool_calls")}
    out["content"] = message["content"].split("</think>")[-1]
    return out


def test_multi_turn_chain_reattaches_tokens_and_emits_tool_images() -> None:
    calls = []
    user = {"role": "user", "content": [image("data:input"), {"type": "text", "text": "q"}]}

    messages = [user]
    parent, new_images = attach_parent_tokens(messages, calls)
    assert (parent, new_images) == (None, ["data:input"])
    first = reply("<think>look</think>Cropping.", [1, 2, 3], [4, 5, 11], ("call_a",))
    assert record_call(calls, parent, new_images, first)
    assert "prompt_token_ids" not in first  # OpenCode never sees the token fields

    messages = [
        user,
        echoed(first),
        {"role": "tool", "tool_call_id": "call_a", "content": "ok"},
        {"role": "user", "content": [image("data:crop")]},
    ]
    sent = deepcopy(messages)
    parent, new_images = attach_parent_tokens(sent, calls)
    assert (parent, new_images) == (0, ["data:crop"])
    assert sent[1]["prompt_token_ids"] == [1, 2, 3]
    assert sent[1]["generation_token_ids"] == [4, 5, 11]
    assert record_call(calls, parent, new_images, reply("\\boxed{7}", [1, 2, 3, 4, 5, 11, 6, 7], [8, 11]))

    # A side call (e.g. a session title) is its own root and is not in the main chain.
    side = [{"role": "user", "content": "title?"}]
    side_parent, side_images = attach_parent_tokens(side, calls)
    assert record_call(calls, side_parent, side_images, reply("A title", [9], [10, 11]))

    assert [c.depth for c in main_chain(calls)] == [0, 1]
    items = chain_output_items(calls, num_input_images=1)
    kinds = [(getattr(i, "type", None), getattr(i, "role", None)) for i in items]
    assert kinds == [
        ("message", "assistant"),
        ("function_call", None),
        ("message", "user"),
        ("message", "assistant"),
    ]
    assert items[0].content[0].text == "Cropping."
    assert items[0].generation_token_ids == [4, 5, 11]
    assert items[2].content[0]["image_url"] == "data:crop"
    assert items[3].prompt_token_ids[: len(items[0].prompt_token_ids) + 3] == [1, 2, 3, 4, 5, 11]
    assert items[3].content[0].text == "\\boxed{7}"


def test_reply_without_tokens_is_not_recorded() -> None:
    calls = []
    message = {"role": "assistant", "content": ""}
    assert not record_call(calls, None, [], message)
    assert calls == [] and chain_output_items(calls, 1) == []


def test_unmatched_history_starts_a_new_root() -> None:
    calls = []
    assert record_call(calls, None, [], reply("hi", [1], [2, 11], ("call_a",)))
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]}]
    assert attach_parent_tokens(messages, calls) == (None, [])
    assert "prompt_token_ids" not in messages[1]


def test_topk_is_kept_on_output_but_not_sent_back() -> None:
    calls = []
    first = reply("Cropping.", [1, 2], [3, 11], ("call_a",))
    first["generation_topk_token_ids"] = [[3, 4], [11, 5]]
    first["generation_topk_log_probs"] = [[-0.1, -2.5], [-0.01, -5.0]]
    assert record_call(calls, None, [], first)
    assert "generation_topk_token_ids" not in first

    messages = [{"role": "user", "content": "q"}, echoed(first)]
    assert attach_parent_tokens(messages, calls)[0] == 0
    assert "generation_topk_token_ids" not in messages[1]
    assert messages[1]["generation_token_ids"] == [3, 11]

    (item, _) = chain_output_items(calls, num_input_images=0)
    assert item.generation_topk_token_ids == [[3, 4], [11, 5]]
    assert item.generation_topk_log_probs == [[-0.1, -2.5], [-0.01, -5.0]]
