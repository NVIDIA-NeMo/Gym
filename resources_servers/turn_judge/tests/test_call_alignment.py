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
from resources_servers.turn_judge.call_alignment import call_alignment, step_output_tokens


def step(think=None, visible=None, tools=(), raw_text=None):
    parts = [{"type": "step-start"}]
    if raw_text is not None:
        parts.append({"type": "text", "text": raw_text})
    elif think is not None or visible is not None:
        parts.append(
            {"type": "text", "text": (f"<think>{think}</think>" if think is not None else "") + (visible or "")}
        )
    for t in tools:
        parts.append({"type": "tool", "callID": t, "tool": "bash", "state": {"input": {}, "output": "ok"}})
    parts.append({"type": "step-finish"})
    return parts


def export(*steps):
    return {
        "messages": [
            {"info": {"role": "user"}, "parts": [{"type": "text", "text": "task"}]},
            {"info": {"role": "assistant"}, "parts": [p for s in steps for p in s]},
        ]
    }


def items(*calls):
    """response.output as Gym builds it: per step its reasoning, visible message, then tool call + output."""
    out = []
    for think, visible, tools in calls:
        if think:
            out.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": think}]})
        if visible:
            out.append({"type": "message", "content": [{"type": "output_text", "text": visible}]})
        for t in tools:
            out.append({"type": "function_call", "call_id": t, "name": "bash", "arguments": "{}"})
            out.append({"type": "function_call_output", "call_id": t, "output": "ok"})
    return out


def test_aligned_session():
    e = export(step("plan A", tools=["c1"]), step("look", tools=["c2", "c3"]), step("done", "All set."))
    o = items(("plan A", None, ["c1"]), ("look", None, ["c2", "c3"]), ("done", "All set.", []))
    assert call_alignment(e, o) == (True, 0, "ok")


def test_trailing_empty_calls_are_counted():
    e = export(
        step("plan", tools=["c1"]), step("fix", tools=["c2"]), step(raw_text=""), step(raw_text="<think></think>")
    )
    o = items(("plan", None, ["c1"]), ("fix", None, ["c2"]))
    assert call_alignment(e, o) == (True, 2, "ok")


def test_call_without_reasoning_is_not_aligned():
    # Step 1 has no think: Gym merges its items into call 0, so call k != step k from there on.
    e = export(step("plan", tools=["c1"]), step(visible="", tools=["c2"]), step("next", tools=["c3"]))
    o = items(("plan", None, ["c1"]), (None, None, ["c2"]), ("next", None, ["c3"]))
    ok, k, why = call_alignment(e, o)
    assert not ok and k == 0


def test_empty_step_in_the_middle_is_not_aligned():
    e = export(step("plan", tools=["c1"]), step(raw_text=""), step("next", tools=["c2"]))
    o = items(("plan", None, ["c1"]), ("next", None, ["c2"]))
    assert call_alignment(e, o)[:2] == (False, 0)


def test_compensating_errors_are_caught():
    # One step without reasoning and one with two think blocks keep the counts equal, but the calls are shifted.
    e = export(
        step("a", tools=["c1"]),
        step(visible="", tools=["c2"]),
        step(raw_text="<think>b</think>x<think>c</think>", tools=["c3"]),
    )
    o = [
        {"type": "reasoning", "summary": [{"text": "a"}]},
        {"type": "function_call", "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1"},
        {"type": "function_call", "call_id": "c2"},
        {"type": "function_call_output", "call_id": "c2"},
        {"type": "reasoning", "summary": [{"text": "b"}]},
        {"type": "reasoning", "summary": [{"text": "c"}]},
        {"type": "function_call", "call_id": "c3"},
        {"type": "function_call_output", "call_id": "c3"},
    ]
    assert call_alignment(e, o)[:2] == (False, 0)


def test_reasoning_from_another_step_is_caught():
    e = export(step("a", tools=["c1"]), step("b", tools=["c2"]))
    o = items(("b", None, ["c1"]), ("a", None, ["c2"]))  # texts swapped
    assert call_alignment(e, o)[:2] == (False, 0)


def test_tool_call_in_the_wrong_call_is_caught():
    e = export(step("a", tools=["c1"]), step("b", tools=["c2"]))
    o = items(("a", None, ["c1", "c2"]), ("b", None, []))
    assert call_alignment(e, o)[:2] == (False, 0)


def test_first_call_may_start_without_reasoning():
    e = export(step(visible="Hi.", tools=["c1"]), step("then", tools=["c2"]))
    o = items((None, "Hi.", ["c1"]), ("then", None, ["c2"]))
    assert call_alignment(e, o) == (True, 0, "ok")


def test_unproven_inputs():
    assert call_alignment(None, items(("a", None, []))) == (False, 0, "no_export")
    assert call_alignment(export(step("a")), [])[:2] == (False, 0)
    bad = {"messages": [{"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "x"}]}]}  # no step-start
    assert call_alignment(bad, items(("a", None, [])))[:2] == (False, 0)


def test_step_output_tokens():
    def msg(tokens, steps=1, finished=True):
        info = {"role": "assistant"} if tokens is None else {"role": "assistant", "tokens": tokens}
        parts = [{"type": "step-start"}, {"type": "step-finish"}] * steps
        if not finished:
            parts = [p for p in parts if p["type"] != "step-finish"]
        return {"info": info, "parts": parts}

    user = {"info": {"role": "user"}, "parts": [{"type": "text", "text": "task"}]}
    out = lambda n, r=0: {"output": n, "reasoning": r}
    assert step_output_tokens({"messages": [user, msg(out(120)), msg(out(7)), msg(out(1))]}) == [120, 7, 1]
    # Reasoning reported apart counts: the row's run is their sum.
    assert step_output_tokens({"messages": [user, msg(out(265, 27)), msg({"output": 4})]}) == [292, 4]
    # A step that never finished (session ended during its tool call): usage not recorded.
    assert step_output_tokens({"messages": [user, msg(out(5)), msg(out(0), finished=False)]}) == [5, None]
    # An empty assistant message is no model call.
    assert step_output_tokens({"messages": [user, msg(out(5)), {"info": {"role": "assistant"}, "parts": []}]}) == [5]
    # Not attributable per step: the trainer applies no label.
    assert step_output_tokens({"messages": [user, msg(out(120), steps=2)]}) is None
    assert step_output_tokens({"messages": [user, msg(None)]}) is None
    assert step_output_tokens({"messages": [user, msg({"output": "12"})]}) is None
    assert step_output_tokens(None) is None
