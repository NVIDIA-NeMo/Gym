# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the judge and the simulator are shown: truncation, items, head-and-tail histories, the corrected turn, the
agent's system prompt and the tool definitions."""

import json
from typing import Any

import pytest

from resources_servers.agentic_turn_simulation.render import (
    ELISION,
    JUDGE_RENDER,
    SIMULATOR_RENDER,
    SIMULATOR_SYSTEM_PROMPT_CHARS,
    RenderConfig,
    _head_tail,
    agent_system_prompt,
    format_tools,
    render_conversation,
    render_item,
    render_system_prompt,
    truncate,
)


OPEN = "<rejected_turn>  <!-- the turn the user is correcting -->\n"
CLOSE = "\n</rejected_turn>"


def _msg(role: str, text: Any) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def _body(rendered: str) -> str:
    """The text between <conversation> and </conversation>."""
    return rendered.split("<conversation>\n", 1)[1].split("\n</conversation>", 1)[0]


class TestTruncate:
    def test_short_text_and_disabled_limit_are_untouched(self) -> None:
        assert truncate("abc", 3) == "abc"
        assert truncate("x" * 50, 0) == "x" * 50

    def test_keeps_head_and_tail_around_the_elision_marker(self) -> None:
        text = "H" * 3 + "m" * 2000 + "T" * 4
        out = truncate(text, 7)
        assert out == "HHH" + "\n[... 2,000 characters elided ...]\n" + "TTTT"

    def test_odd_limit_gives_tail_the_extra_character(self) -> None:
        assert truncate("abcdefghij", 5) == "ab\n[... 5 characters elided ...]\nhij"


class TestRenderItem:
    cfg = RenderConfig()

    def test_user_and_assistant_messages(self) -> None:
        assert render_item(_msg("user", "hi"), self.cfg) == '<message from="User">\nhi\n</message>'
        assert render_item(_msg("assistant", "yo"), self.cfg) == '<message from="Agent">\nyo\n</message>'

    def test_role_only_item_is_a_message(self) -> None:
        assert render_item({"role": "user", "content": "hi"}, self.cfg) == '<message from="User">\nhi\n</message>'

    def test_content_parts_are_joined(self) -> None:
        parts = [{"type": "output_text", "text": "a"}, "junk", {"type": "input_image"}, {"text": None}, {"text": "b"}]
        assert render_item(_msg("assistant", parts), self.cfg) == '<message from="Agent">\nab\n</message>'

    @pytest.mark.parametrize(
        "item",
        [
            _msg("user", "   "),
            _msg("assistant", None),
            _msg("system", "you are an agent"),
            _msg("developer", "rules"),
            {"type": "reasoning", "summary": []},
            {"type": "something_else"},
            {"content": "no role, no type"},
        ],
    )
    def test_skipped_items(self, item: dict[str, Any]) -> None:
        assert render_item(item, self.cfg) is None

    def test_function_call_arguments_are_truncated(self) -> None:
        cfg = RenderConfig(tool_args_max_chars=4)
        item = {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "abcdefgh"}
        assert render_item(item, cfg) == (
            '<tool_call id="c1" name="bash">\nab\n[... 4 characters elided ...]\ngh\n</tool_call>'
        )
        assert render_item({"type": "function_call", "call_id": "c", "name": "x"}, cfg).endswith(">\n\n</tool_call>")

    def test_function_call_output_truncated_and_non_string_dumped(self) -> None:
        cfg = RenderConfig(tool_output_max_chars=4)
        out = render_item({"type": "function_call_output", "call_id": "c1", "output": "abcdefgh"}, cfg)
        assert out == '<tool_result id="c1">\nab\n[... 4 characters elided ...]\ngh\n</tool_result>'
        listed = render_item({"type": "function_call_output", "call_id": "c2", "output": [{"t": 1}]}, RenderConfig())
        assert listed == '<tool_result id="c2">\n' + json.dumps([{"t": 1}]) + "\n</tool_result>"


class TestHeadTail:
    cfg = RenderConfig(head_chars=10, tail_chars=10)

    def test_under_budget_is_unchanged(self) -> None:
        blocks = ["aaaa", "bbbb", "cccc"]
        assert _head_tail(blocks, self.cfg) == blocks

    def test_keeps_whole_blocks_from_both_ends(self) -> None:
        blocks = ["a" * 6, "b" * 3, "c" * 6, "d" * 6, "e" * 6, "f" * 4]
        assert _head_tail(blocks, self.cfg) == [
            "a" * 6,
            "b" * 3,
            "[... 2 blocks omitted from the middle of the conversation ...]",
            "e" * 6,
            "f" * 4,
        ]

    def test_no_marker_when_nothing_is_omitted(self) -> None:
        # The separators push the joined text over budget, but every block still fits in head or tail.
        blocks = ["a" * 5, "b" * 5, "c" * 5, "d" * 5]
        assert _head_tail(blocks, self.cfg) == blocks

    def test_oversized_first_block_leaves_head_empty(self) -> None:
        blocks = ["a" * 30, "b" * 4]
        assert _head_tail(blocks, self.cfg) == [
            "[... 1 blocks omitted from the middle of the conversation ...]",
            "b" * 4,
        ]


class TestRenderConversation:
    history = [
        _msg("user", "make a site"),  # 0
        _msg("assistant", "done"),  # 1
        {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{}"},  # 2
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},  # 3
        _msg("user", "that is wrong"),  # 4
    ]

    def test_layout(self) -> None:
        step = [_msg("assistant", "fixed")]
        out = render_conversation(self.history[:1], step, RenderConfig())
        assert out == (
            '<conversation>\n<message from="User">\nmake a site\n</message>\n</conversation>\n\n'
            '<step_under_evaluation>\n<message from="Agent">\nfixed\n</message>\n</step_under_evaluation>'
        )

    def test_empty_step(self) -> None:
        out = render_conversation(self.history, [{"type": "reasoning"}], JUDGE_RENDER)
        assert "<step_under_evaluation>\n[the agent produced nothing]\n</step_under_evaluation>" in out

    def test_rejected_turn_wraps_the_corrected_span(self) -> None:
        body = _body(render_conversation(self.history, [], JUDGE_RENDER, [1, 4]))
        blocks = body.split("\n\n")
        assert blocks[0] == '<message from="User">\nmake a site\n</message>'
        assert blocks[1].startswith(OPEN + '<message from="Agent">')
        assert blocks[3].endswith("</tool_result>" + CLOSE)
        assert blocks[4] == '<message from="User">\nthat is wrong\n</message>'
        assert body.count("<rejected_turn>") == 1 and body.count("</rejected_turn>") == 1

    def test_span_starting_on_a_skipped_item_opens_at_the_next_rendered_one(self) -> None:
        history = [_msg("user", "u"), {"type": "reasoning"}, _msg("assistant", "a"), _msg("user", "no")]
        body = _body(render_conversation(history, [], JUDGE_RENDER, [1, 3]))
        assert OPEN + '<message from="Agent">\na\n</message>' + CLOSE in body
        assert body.endswith('<message from="User">\nno\n</message>')

    def test_span_running_to_the_end_on_skipped_items_is_still_closed(self) -> None:
        history = [_msg("user", "u"), _msg("assistant", "a"), {"type": "reasoning"}]
        body = _body(render_conversation(history, [], JUDGE_RENDER, [1, 3]))
        assert body.endswith(OPEN + '<message from="Agent">\na\n</message>' + CLOSE)

    def test_span_ending_on_a_skipped_item_closes_before_the_correction(self) -> None:
        history = [_msg("user", "u"), _msg("assistant", "a"), _msg("assistant", ""), _msg("user", "no, redo it")]
        body = _body(render_conversation(history, [], JUDGE_RENDER, [1, 3]))
        assert body.endswith('</message>\n</rejected_turn>\n\n<message from="User">\nno, redo it\n</message>')

    @pytest.mark.parametrize(
        "cfg, span",
        [(RenderConfig(mark_corrected_turn=False), [1, 4]), (JUDGE_RENDER, None), (JUDGE_RENDER, [])],
    )
    def test_no_marking(self, cfg: RenderConfig, span) -> None:
        assert "rejected_turn" not in render_conversation(self.history, [], cfg, span)

    def test_conversation_cap_applies_to_history_only(self) -> None:
        cfg = RenderConfig(max_conversation_chars=40)
        history = [_msg("user", "x" * 500)]
        step = [_msg("assistant", "y" * 300)]
        out = render_conversation(history, step, cfg)
        assert "characters elided" in _body(out) and len(_body(out)) < 100
        assert "y" * 300 in out

    def test_head_tail_mode_omits_the_middle(self) -> None:
        cfg = RenderConfig(prefix_mode="head_tail", head_chars=60, tail_chars=60)
        history = [_msg("user", f"turn {i} " + "z" * 20) for i in range(10)]
        body = _body(render_conversation(history, [], cfg))
        assert "turn 0 " in body and "turn 9 " in body and "turn 5 " not in body
        assert "blocks omitted from the middle of the conversation" in body

    def test_render_presets(self) -> None:
        assert (JUDGE_RENDER.prefix_mode, JUDGE_RENDER.max_conversation_chars) == ("full", 200_000)
        assert (JUDGE_RENDER.tool_args_max_chars, JUDGE_RENDER.tool_output_max_chars) == (2_000, 2_000)
        assert JUDGE_RENDER.mark_corrected_turn is True
        assert (SIMULATOR_RENDER.prefix_mode, SIMULATOR_RENDER.head_chars, SIMULATOR_RENDER.tail_chars) == (
            "head_tail",
            20_000,
            100_000,
        )
        assert (SIMULATOR_RENDER.tool_args_max_chars, SIMULATOR_RENDER.tool_output_max_chars) == (6_000, 4_000)
        assert SIMULATOR_RENDER.mark_corrected_turn is False and SIMULATOR_RENDER.max_conversation_chars == 0


class TestFormatTools:
    @pytest.mark.parametrize("tools", [None, []])
    def test_no_tools(self, tools) -> None:
        assert format_tools(tools) == "<tools>\n(tool definitions omitted)\n</tools>"

    def test_tools_are_listed_with_their_schemas(self) -> None:
        tools = [
            {"name": "bash", "description": "Run a command.", "parameters": {"type": "object"}},
            {"name": "ask_user"},
        ]
        assert format_tools(tools) == (
            '<tools>\n<tool name="bash">\nRun a command.\nparameters: {"type": "object"}\n</tool>\n'
            '<tool name="ask_user">\n\nparameters: {}\n</tool>\n</tools>'
        )


class TestAgentSystemPrompt:
    def test_system_and_developer_messages_in_order(self) -> None:
        items = [
            _msg("system", "You are a coding agent."),
            _msg("user", "hi"),
            _msg("developer", [{"type": "input_text", "text": "Workspace: /srv/app"}]),
            _msg("assistant", "hello"),
            _msg("system", "   "),
        ]
        assert agent_system_prompt(items) == "You are a coding agent.\n\nWorkspace: /srv/app"

    def test_none(self) -> None:
        assert agent_system_prompt([_msg("user", "hi")]) == ""
        assert render_system_prompt("") == "" and render_system_prompt(" \n") == ""

    def test_block_is_whole_unless_capped(self) -> None:
        text = "a" * 50 + "b" * 50
        assert render_system_prompt(text) == f"<agent_system_prompt>\n{text}\n</agent_system_prompt>\n\n"
        capped = render_system_prompt(text, 40)
        assert capped == f"<agent_system_prompt>\n{truncate(text, 40)}\n</agent_system_prompt>\n\n"
        assert capped.startswith("<agent_system_prompt>\n" + "a" * 20) and ("b" * 20 + "\n</agent") in capped
        assert ELISION.format(n=60) in capped

    def test_simulator_cap(self) -> None:
        assert SIMULATOR_SYSTEM_PROMPT_CHARS == 60_000
