# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cline's XML tool calls parsed from synthetic reply texts, and the "no tools used" message the agent sends for a
reply without one."""

from resources_servers.agentic_turn_simulation.cline_xml import NO_TOOLS_USED, calls, no_tools_used


class TestCalls:
    def test_a_call_with_parameters(self) -> None:
        text = "I'll read it.\n\n<read_file>\n<path>src/app.ts</path>\n</read_file>"
        (call,) = calls(text)
        assert (call.name, call.params) == ("read_file", {"path": "src/app.ts"})
        assert text[call.start : call.end] == "<read_file>\n<path>src/app.ts</path>\n</read_file>"

    def test_every_call_in_order(self) -> None:
        text = "<list_files><path>.</path></list_files>\nthen\n<read_file><path>a.py</path></read_file>"
        assert [(c.name, c.params) for c in calls(text)] == [
            ("list_files", {"path": "."}),
            ("read_file", {"path": "a.py"}),
        ]

    def test_a_tool_tag_inside_a_calls_body_belongs_to_the_body(self) -> None:
        text = "<write_to_file>\n<path>doc.md</path>\n<content>Use <read_file> to read.</content>\n</write_to_file>"
        (call,) = calls(text)
        assert call.params["content"] == "Use <read_file> to read."

    def test_multiline_values_keep_inner_newlines_and_lose_the_outer_ones(self) -> None:
        text = "<write_to_file>\n<path>notes.md</path>\n<content>\n# Title\n\nbody\n</content>\n</write_to_file>"
        (call,) = calls(text)
        assert call.params == {"path": "notes.md", "content": "# Title\n\nbody"}

    def test_an_unclosed_call_runs_to_the_end_of_the_text(self) -> None:
        (call,) = calls("<execute_command>\n<command>npm test</command>\n<requires_approval>false")
        assert call.name == "execute_command"
        assert call.params == {"command": "npm test", "requires_approval": "false"}

    def test_unknown_tags_and_plain_text_are_not_calls(self) -> None:
        assert calls("Use <div> and <thinking>no tools</thinking> here.") == []
        assert calls("") == [] and calls(None) == []

    def test_a_reply_tool_is_a_call_too(self) -> None:
        (call,) = calls("<attempt_completion>\n<result>\nAll tests pass.\n</result>\n</attempt_completion>")
        assert (call.name, call.params) == ("attempt_completion", {"result": "All tests pass."})


class TestNoToolsUsed:
    def test_the_error_then_the_latest_environment_details_from_user_messages(self) -> None:
        old = "<environment_details>\n# Current Mode\nPLAN MODE\n</environment_details>"
        new = "<environment_details>\n# Current Mode\nACT MODE\n</environment_details>"
        messages = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "<task>x</task>"}, {"type": "input_text", "text": old}],
            },
            {"type": "message", "role": "assistant", "content": "<environment_details>not this</environment_details>"},
            {"type": "message", "role": "user", "content": "[read_file for 'a'] Result:\n1 | x\n" + new},
            {"type": "function_call_output", "call_id": "c", "output": "x"},
        ]
        assert no_tools_used(messages) == [NO_TOOLS_USED, new]
        assert no_tools_used(messages[:2]) == [NO_TOOLS_USED, old]
        assert no_tools_used([]) == [NO_TOOLS_USED]
