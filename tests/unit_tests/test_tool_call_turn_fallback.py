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
"""Tool argument rewrites may not hide other changes to the captured model turn."""

from copy import deepcopy

import pytest

from nemo_gym.token_id_capture import stamp_lineage
from nemo_gym.token_id_capture.lineage import RolloutLineage, stamp_continuation
from nemo_gym.token_id_capture.records import ParentResolutionStatus, TokenEntry
from nemo_gym.token_id_capture.terminal import resolve_terminal


def _tool(call_id: str, name: str, arguments: str) -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _message(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _served() -> list[dict]:
    return [
        _message("I will inspect it."),
        _tool("t1", "Bash", '{"command":"cd /testbed && ls"}'),
        _tool("t2", "Edit", '{"path":"a"}'),
    ]


def _rewritten() -> list[dict]:
    return [
        _message("I will inspect it."),
        _tool("t1", "Bash", '{"command":"ls"}'),
        _tool("t2", "Edit", '{"path":"a","replace_all":false}'),
    ]


def _mutate(kind: str) -> list[dict]:
    turn = _rewritten()
    if kind == "extra_tool":
        turn.append(_tool("unrecorded", "Bash", "{}"))
    elif kind == "missing_tool":
        turn.pop()
    elif kind == "reordered_tools":
        turn[1:] = reversed(turn[1:])
    elif kind == "changed_name":
        turn[1]["name"] = "Delete"
    elif kind == "changed_namespace":
        turn[1]["namespace"] = "other"
    elif kind == "changed_text":
        turn[0] = _message("The task is complete.")
    elif kind == "missing_text":
        turn.pop(0)
    elif kind == "extra_final_message":
        turn.append(_message("An uncaptured final answer."))
    elif kind == "extra_empty_message":
        turn.append(_message(""))
    elif kind == "duplicate_tool_id":
        turn.append(deepcopy(turn[1]))
    elif kind == "missing_tool_id":
        turn[2]["call_id"] = ""
    return turn


def _entry(output: list[dict]) -> TokenEntry:
    entry = TokenEntry(
        rollout_id="r",
        model_call_id="served",
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.1],
        output_items=output,
        response_id="served-response",
    )
    stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
    return stamp_continuation(entry, [{"role": "user", "content": "fix it"}])


@pytest.mark.parametrize(
    "kind",
    [
        "extra_tool",
        "missing_tool",
        "reordered_tools",
        "changed_name",
        "changed_namespace",
        "changed_text",
        "missing_text",
        "extra_final_message",
        "extra_empty_message",
        "duplicate_tool_id",
        "missing_tool_id",
    ],
)
@pytest.mark.parametrize("index_path", ["record", "add_entry", "terminal"])
def test_tool_fallback_rejects_unwitnessed_turn_changes(kind: str, index_path: str) -> None:
    ctx = [{"role": "user", "content": "fix it"}]
    echoed = _mutate(kind)
    if index_path == "terminal":
        result = resolve_terminal([_entry(_served())], {"id": "harness", "output": ctx + echoed})
        assert result.model_call_id is None
    else:
        lineage = RolloutLineage()
        if index_path == "record":
            lineage.record("served", ctx + _served(), [1, 2, 3], "digest", context_len=len(ctx))
        else:
            lineage.add_entry(_entry(_served()))
        assert lineage.resolve(ctx + echoed).status == ParentResolutionStatus.UNRESOLVED


@pytest.mark.parametrize("dialect", ["responses", "chat", "anthropic"])
@pytest.mark.parametrize("index_path", ["record", "add_entry", "terminal"])
def test_tool_fallback_keeps_text_and_complete_identity_across_dialects(dialect: str, index_path: str) -> None:
    ctx = [{"role": "user", "content": "fix it"}]
    echoed = _rewritten()
    if dialect == "chat":
        echoed = [
            {
                "role": "assistant",
                "content": echoed[0]["content"],
                "tool_calls": [
                    {
                        "id": t["call_id"],
                        "type": "function",
                        "function": {"name": t["name"], "arguments": t["arguments"]},
                    }
                    for t in echoed[1:]
                ],
            }
        ]
    elif dialect == "anthropic":
        import json

        echoed = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": echoed[0]["content"]}]
                + [
                    {"type": "tool_use", "id": t["call_id"], "name": t["name"], "input": json.loads(t["arguments"])}
                    for t in echoed[1:]
                ],
            }
        ]
    if index_path == "terminal":
        result = resolve_terminal([_entry(_served())], {"id": "harness", "output": ctx + echoed})
        assert result.model_call_id == "served"
        assert result.method == "content_tool_ids"
    else:
        lineage = RolloutLineage()
        if index_path == "record":
            lineage.record("served", ctx + _served(), [1, 2, 3], "digest", context_len=len(ctx))
        else:
            lineage.add_entry(_entry(_served()))
        result = lineage.resolve(ctx + echoed)
        assert result.status == ParentResolutionStatus.RESOLVED
        assert result.match.model_call_id == "served"
