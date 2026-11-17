# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The judge grades the instruction actually supplied, not a reconstructed requirement."""

from types import SimpleNamespace

import pytest

from resources_servers.agentic_if.if_grading.configuration import load_config
from resources_servers.agentic_if.if_grading.contracts import InstructionContext, ResolvedCheck, SessionEvidence
from resources_servers.agentic_if.if_grading.engine import grade
from resources_servers.agentic_if.if_grading.instructions import instruction_parts
from resources_servers.agentic_if.if_grading.judge import Judge


def response(text='{"why":"Kept the instruction.","answer":"yes"}'):
    return SimpleNamespace(output_text=text)


@pytest.mark.parametrize("at,scope", [("task_message", {}), ("user_turn", {"activation": "continuation_start"})])
def test_engine_uses_the_captured_carrier_of_the_instruction(at, scope):
    trace = SessionEvidence(
        status="complete",
        steps=[
            {"id": "u", "session_id": "s", "role": "user", "text": "Fix it. End with FINISH."},
            {"id": "a", "session_id": "s", "role": "assistant", "text": "FINISH"},
        ],
        continuation_start=0 if at == "user_turn" else None,
    )
    check = ResolvedCheck(
        id="c",
        kind="judge",
        rubric="Did it follow the instruction?",
        parameters={"context": "message"},
        scope=scope | {"places": ["final"]},
        instruction={"text": "End with FINISH.", "at": at, "carrier": "stale authoring text"},
    )
    calls = []
    result = grade([check], trace, judge=Judge(lambda body: calls.append(body) or response()))
    assert result.if_all == 1
    assert "Fix it. [[End with FINISH.]]" in calls[0]["input"][0]["content"]
    assert "stale authoring text" not in calls[0]["input"][0]["content"]


def judged_prompt(steps, instruction, events=()):
    """The judge's question for a message rule with this instruction, graded on these steps."""
    trace = SessionEvidence(
        status="complete",
        steps=[{"id": str(index), "session_id": "s", **step} for index, step in enumerate(steps)],
        events=list(events),
    )
    scope = {"places": ["final"]} | ({"activation": "event", "event_id": events[0]["id"]} if events else {})
    check = ResolvedCheck(
        id="c",
        kind="judge",
        rubric="Did it follow the instruction?",
        parameters={"context": "message"},
        scope=scope,
        instruction=instruction,
    )
    calls = []
    assert grade([check], trace, judge=Judge(lambda body: calls.append(body) or response())).if_all == 1
    return calls[0]["input"][0]["content"]


@pytest.mark.parametrize(
    "system,carrier",
    [
        (
            "You are a coding agent.\nEnd with FINISH.\nBe brief.",
            "You are a coding agent.\n[[End with FINISH.]]\nBe brief.",
        ),
        # a system prompt without the instruction is no carrier of it: the authored one stands
        ("You are a coding agent.", "Authored: [[End with FINISH.]]"),
    ],
)
def test_a_system_prompt_instruction_is_shown_in_the_system_prompt_that_has_it(system, carrier):
    prompt = judged_prompt(
        [
            {"role": "system", "text": system},
            {"role": "user", "text": "Fix it."},
            {"role": "assistant", "text": "FINISH"},
        ],
        {"text": "End with FINISH.", "at": "system_prompt", "carrier": "Authored: End with FINISH."},
    )
    assert "WHERE IT WAS: THE ASSISTANT'S SYSTEM PROMPT, with the text around it" in prompt
    assert f"<<<\n{carrier}\n>>>" in prompt


def test_a_tool_output_instruction_is_read_from_the_result_its_event_fired_on():
    call = {"call_id": "c", "invocation_id": "s", "tool": "bash", "model_tool": "bash", "args": {}, "model_args": {}}
    prompt = judged_prompt(
        [
            {"role": "user", "text": "Fix it."},
            {"role": "assistant", "calls": [call]},
            {"role": "tool", "tool": "bash", "call_id": "c", "invocation_id": "s", "output": "Failed. Do not retry."},
            {"role": "assistant", "text": "The command failed; I did not retry it."},
        ],
        {"text": "Do not retry.", "at": "tool_output"},
        events=[{"id": "notice", "step_index": 2}],
    )
    assert "WHERE IT WAS: A TOOL RESULT THE ASSISTANT GOT DURING THE SESSION" in prompt
    assert "<<<\nFailed. [[Do not retry.]]\n>>>" in prompt


CONFIG = load_config()
WEIGH = CONFIG["instruction"]["weigh"]


@pytest.mark.parametrize(
    "text,at,carrier,where,weigh",
    [
        # the carrier is the instruction itself (apart from spaces): nothing else to show
        ("End with FINISH.", "task_message", "  End with FINISH.\n", "the instruction above is the whole of it", ""),
        (
            "End with FINISH.",
            "task_message",
            "",
            "WHERE IT WAS: THE USER'S TASK MESSAGE (the first message of the session).\n",
            "",
        ),
        (
            "End with FINISH.",
            "task_message",
            None,
            "WHERE IT WAS: THE USER'S TASK MESSAGE (the first message of the session).\n",
            "",
        ),
        (
            "End with FINISH.",
            "user_turn",
            "Please finish each reply with FINISH.",
            "(the instruction is not in it word for word):\n<<<\nPlease finish each reply with FINISH.\n>>>",
            WEIGH,
        ),
        # an instruction with spaces around it is found by its words
        (
            " End with FINISH. ",
            "tool:read",
            "Reads a file. End with FINISH.",
            "WHERE IT WAS: THE DESCRIPTION OF THE TOOL read, with the text around it "
            "(the instruction is marked [[ ]]):\n<<<\nReads a file. [[End with FINISH.]]\n>>>",
            WEIGH,
        ),
    ],
    ids=["whole", "empty_carrier", "no_carrier", "unmarked", "stripped_tool_description"],
)
def test_instruction_parts_mark_the_instruction_in_its_carrier(text, at, carrier, where, weigh):
    parts = instruction_parts(CONFIG, InstructionContext(text=text, at=at, carrier=carrier))
    assert parts["instruction"] == text
    assert where in parts["where"]
    assert parts["weigh"] == weigh


def test_a_long_carrier_is_shown_as_a_window_cut_at_line_breaks():
    lines = [f"Line {n:03d}: " + "x" * 89 for n in range(60)]
    carrier = "\n".join(lines[:30] + ["End with FINISH."] + lines[30:])
    parts = instruction_parts(CONFIG, InstructionContext(text="End with FINISH.", at="system_prompt", carrier=carrier))
    around = parts["where"].split("<<<\n", 1)[1].split("\n>>>", 1)[0]
    assert around.startswith("[...]\nLine 022: ")
    assert f"\n{lines[29]}\n[[End with FINISH.]]\n{lines[30]}\n" in around
    assert around.endswith(f"\n{lines[33]}\n[...]")
    # an unmarked carrier over whole_chars is cut at its end
    unmarked = instruction_parts(CONFIG, InstructionContext(text="Never push.", at="user_turn", carrier="y" * 3100))
    assert unmarked["where"].endswith("<<<\n" + "y" * 3000 + " [...]\n>>>\n")
