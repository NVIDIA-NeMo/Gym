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

"""Wire tests for the base-model server.

A base model has no chat template, so the transcript IS the interface. The
separator shape is load-bearing: the stop markers are "\\nUser:" / "\\nAssistant:",
so if rendering ever emits a blank line between turns (or drops the leading
newline) the stop list stops matching turn boundaries and the model runs on into
a hallucinated conversation.
"""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BM = _load("base_model_app", Path(__file__).parent.parent / "app.py")


class TestRenderTranscript:
    def test_roles_are_labelled(self):
        rendered = BM.render_transcript(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "U1"},
                {"role": "assistant", "content": "A1"},
            ]
        )
        assert rendered == "\nSystem: SYS\nUser: U1\nAssistant: A1"

    def test_exactly_one_newline_precedes_each_marker(self):
        # Blank lines between turns would stop the stop-markers from matching.
        rendered = BM.render_transcript([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])
        assert "\n\n" not in rendered
        for marker in ("\nUser:", "\nAssistant:"):
            assert marker in rendered

    def test_stop_markers_align_with_rendered_boundaries(self):
        rendered = BM.render_transcript(
            [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}, {"role": "tool", "content": "t"}]
        )
        for stop in ("\nUser:", "\nAssistant:", "\nTool Result:"):
            assert stop in rendered, f"{stop!r} does not match the rendered transcript"

    def test_tool_calls_render_as_tool_call_turn(self):
        rendered = BM.render_transcript([{"role": "assistant", "tool_calls": [{"id": "1"}]}])
        assert rendered.startswith("\nAssistant (tool call): ")
        assert '"id": "1"' in rendered

    def test_unknown_role_is_skipped(self):
        assert BM.render_transcript([{"role": "developer", "content": "x"}]) == ""

    def test_observation_content_is_verbatim(self):
        observation = "<returncode>0</returncode>\n<output>\nhi\n</output>"
        assert BM.render_transcript([{"role": "user", "content": observation}]) == f"\nUser: {observation}"


class TestMessageText:
    def test_plain_string(self):
        assert BM.message_text("hello") == "hello"

    def test_responses_api_content_parts(self):
        assert BM.message_text([{"type": "output_text", "text": "a"}, {"type": "output_text", "text": "b"}]) == "ab"

    def test_none_is_empty(self):
        assert BM.message_text(None) == ""


class TestWireConstants:
    def test_prefill_opens_the_action_fence(self):
        # Pre-opening the envelope is what stops the model echoing the steer's
        # literal "<your command>" placeholder back as its action.
        assert BM.DEFAULT_PREFILL == "```mswea_bash_command\n"

    def test_primer_opens_an_assistant_turn(self):
        assert BM.DEFAULT_PRIMER == "\nAssistant: "

    def test_steer_names_the_repo_root(self):
        # The single most costly misconfiguration in this harness: a steer that
        # names the wrong root yields a perfect command against a missing path.
        assert "/testbed" in BM.MINI_BASH_STEER

    def test_stop_list_entries_are_turn_boundaries(self):
        assert all(entry.startswith("\n") for entry in BM.DEFAULT_STOP)
        assert "\nUser:" in BM.DEFAULT_STOP and "\nAssistant:" in BM.DEFAULT_STOP

    def test_backticks_stop_list_fits_the_openai_cap(self):
        # Servers that enforce the cap 400 every request with a fifth entry.
        assert len(BM.WIRE_DEFAULTS["backticks"]["stop"]) <= 4

    def test_every_dropped_marker_is_absent_from_a_backticks_transcript(self):
        # Only a message with tool_calls renders the tool-call marker, and the
        # backticks wire never sends one: the marker cannot end a turn there.
        transcript = BM.render_transcript(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "THOUGHT\n```mswea_bash_command\nls\n```"},
                {"role": "user", "content": "<returncode>0</returncode>"},
            ]
        )
        assert "\nAssistant (tool call):" not in transcript
        assert "\nAssistant (tool call):" not in BM.DEFAULT_STOP


class TestParseBashToolCall:
    """The function-calling parser. Its one hard rule: a generation cut inside a
    string is a truncated command and must FAIL, never be completed."""

    NESTED = '[{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{\\"command\\": \\"ls -la\\"}"}}]'

    def test_balanced_nested_call_with_string_arguments(self):
        assert BM.parse_bash_tool_calls(self.NESTED) == [{"name": "bash", "arguments": {"command": "ls -la"}}]

    def test_flat_shape_with_object_arguments(self):
        raw = '[{"name": "bash", "arguments": {"command": "pwd"}}]'
        assert BM.parse_bash_tool_calls(raw)[0]["arguments"] == {"command": "pwd"}

    def test_unbalanced_outer_brackets_recovered_when_strings_terminated(self):
        # The call is complete; only the closing "}}]" is missing.
        raw = '[{"function": {"name": "bash", "arguments": "{\\"command\\": \\"ls\\"}"'
        assert BM.parse_bash_tool_calls(raw)[0]["arguments"] == {"command": "ls"}

    def test_truncated_inside_command_string_fails(self):
        # Cut mid-heredoc: completing it would hang the shell or corrupt a file.
        raw = '[{"function": {"name": "bash", "arguments": "{\\"command\\": \\"cat > f <<\'EOF\'\\nline'
        assert BM.parse_bash_tool_calls(raw) == []

    def test_hallucinated_next_turn_is_ignored(self):
        raw = self.NESTED + '\nTool Result: {"returncode": 0}\nAssistant (tool call): [{'
        assert BM.parse_bash_tool_calls(raw)[0]["arguments"] == {"command": "ls -la"}

    def test_other_tool_name_is_rejected(self):
        assert BM.parse_bash_tool_calls('[{"name": "python", "arguments": {"command": "x"}}]') == []

    @pytest.mark.parametrize("name", [None, 0, 1, True, [], ["bash"], {}, {"name": "bash"}])
    @pytest.mark.parametrize("nested", [False, True])
    def test_non_string_names_are_rejected(self, name: object, nested: bool) -> None:
        call = {"name": name, "arguments": {"command": "pwd"}}
        malformed = {"function": call} if nested else call
        assert BM.parse_bash_tool_calls(json.dumps([malformed])) == []
        valid = {"name": "bash", "arguments": {"command": "ls"}}
        assert BM.parse_bash_tool_calls(json.dumps([malformed, valid])) == [valid]

    def test_empty_command_is_rejected(self):
        assert BM.parse_bash_tool_calls('[{"name": "bash", "arguments": {"command": "  "}}]') == []

    def test_raw_newline_inside_string_tolerated(self):
        # strict=False: a literal newline where JSON wants "\n" is admitted, not repaired.
        raw = '[{"name": "bash", "arguments": {"command": "echo a\necho b"}}]'
        assert BM.parse_bash_tool_calls(raw)[0]["arguments"]["command"] == "echo a\necho b"

    def test_every_bash_call_is_kept_in_order(self):
        raw = '[{"name": "bash", "arguments": {"command": "one"}}, {"name": "bash", "arguments": {"command": "two"}}]'
        assert [c["arguments"]["command"] for c in BM.parse_bash_tool_calls(raw)] == ["one", "two"]

    def test_calls_beyond_the_cap_are_dropped(self):
        # The reference caps a base model's calls per turn at 3.
        raw = "[" + ", ".join(f'{{"name": "bash", "arguments": {{"command": "c{i}"}}}}' for i in range(5)) + "]"
        assert [c["arguments"]["command"] for c in BM.parse_bash_tool_calls(raw)] == ["c0", "c1", "c2"]
        assert len(BM.parse_bash_tool_calls(raw, max_calls=5)) == 5

    def test_no_array_is_none(self):
        assert BM.parse_bash_tool_calls("I think the fix is to edit foo.py") == []


class TestCloseJsonPrefix:
    def test_cuts_at_first_complete_value(self):
        assert BM.close_json_prefix("[1, 2] trailing [3]") == "[1, 2]"

    def test_brackets_inside_strings_do_not_count(self):
        assert BM.close_json_prefix('["a]b", "c{d"]') == '["a]b", "c{d"]'

    def test_escaped_quote_does_not_end_string(self):
        assert BM.close_json_prefix('["a\\"b"') == '["a\\"b"]'

    def test_closes_open_brackets_in_order(self):
        assert BM.close_json_prefix('[{"a": [1') == '[{"a": [1]}]'

    def test_unterminated_string_is_none(self):
        assert BM.close_json_prefix('[{"a": "unfinish') is None

    def test_mismatched_bracket_is_none(self):
        assert BM.close_json_prefix("[}") is None


class TestClientSideStop:
    """A server that misses a stop string must not hand the harness invented turns."""

    def _complete(self, wire, text, finish_reason):
        defaults = BM.WIRE_DEFAULTS[wire]
        server = BM.BaseModelServer.__new__(BM.BaseModelServer)
        cfg = dict(
            wire=wire, tool_definitions_json=None, extra_body={}, openai_model="m", max_tokens=1024, temperature=0.6
        )
        object.__setattr__(server, "config", type("C", (), cfg | dict(top_p=None, debug=False))())
        server._steer, server._primer, server._prefill = defaults["steer"], defaults["primer"], defaults["prefill"]
        server._stop = list(defaults["stop"])
        server._semaphore = asyncio.Semaphore(1)

        async def create_completion(**kwargs):
            return {"choices": [{"text": text, "finish_reason": finish_reason}], "usage": {}}

        server._client = SimpleNamespace(create_completion=create_completion)
        generated, _usage, finish = asyncio.run(server._complete([{"role": "user", "content": "task"}], {}))
        return generated.removeprefix(defaults["prefill"]), finish

    def test_leaked_role_marker_ends_the_turn(self):
        # Seen from a server that ignored "\nUser:" after a fence: a whole invented exchange.
        leaked = "ls\n```\nUser: [Execution results]\nAssistant: Here are the files\n```mswea_bash_command\ncat x\n```"
        assert self._complete("backticks", leaked, "length") == ("ls\n```", "stop")

    def test_earliest_stop_wins(self):
        assert self._complete("backticks", "a\nAssistant: b\nUser: c", "length") == ("a", "stop")

    def test_clean_completion_is_untouched(self):
        assert self._complete("backticks", "ls\n```", "stop") == ("ls\n```", "stop")
        assert self._complete("backticks", "cat big_file", "length") == ("cat big_file", "length")

    def test_function_calling_sends_no_stops_so_nothing_is_cut(self):
        # Tool-call JSON routinely contains "\nUser:"-like text; that wire has no stop list.
        call = 'function": {"name": "bash", "arguments": "{\\"command\\": \\"ls\\"}"}}]\nUser: invented'
        assert "\nUser:" in call
        assert self._complete("function_calling", call, "stop") == (call, "stop")


class TestFunctionCallingWire:
    TOOLS = '[{"type": "function", "function": {"name": "bash"}}]'

    def _server(self, **overrides):
        # Build the prompt without constructing the web server: mirror
        # model_post_init's wire resolution on a bare object.
        cfg = dict(wire="function_calling", tool_definitions_json=self.TOOLS) | overrides
        defaults = BM.WIRE_DEFAULTS[cfg["wire"]]
        server = BM.BaseModelServer.__new__(BM.BaseModelServer)
        object.__setattr__(server, "config", type("C", (), cfg)())
        server._steer, server._primer, server._prefill = defaults["steer"], defaults["primer"], defaults["prefill"]
        server._stop = list(defaults["stop"])
        return server

    def test_stop_list_is_off_on_function_calling(self):
        # INVERTS vs backticks: tool-call JSON routinely contains "\nUser:"-like text.
        assert BM.WIRE_DEFAULTS["function_calling"]["stop"] == []
        assert BM.WIRE_DEFAULTS["backticks"]["stop"] == BM.DEFAULT_STOP

    def test_prompt_order_is_header_transcript_steer_primer_prefill(self):
        prompt = self._server().build_prompt([{"role": "user", "content": "task"}])
        assert prompt == (
            f"\nTool definitions:\n{self.TOOLS}\n\n"
            "\nUser: task" + BM.BASH_TOOL_STEER + "\nAssistant (tool call): " + '[{"'
        )

    def test_steer_names_app_root_and_bash_tool(self):
        assert "/app" in BM.BASH_TOOL_STEER and "`bash`" in BM.BASH_TOOL_STEER
        assert "/testbed" not in BM.BASH_TOOL_STEER

    def test_assistant_tool_call_turn_renders_calls_only(self):
        calls = [{"id": "c", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
        rendered = BM.render_transcript([{"role": "assistant", "content": "ignored prose", "tool_calls": calls}])
        assert rendered.startswith("\nAssistant (tool call): [")
        assert "ignored prose" not in rendered

    def test_tool_calls_render_in_litellm_key_order(self):
        # The reference bridge rendered calls as mini re-sent them, i.e. after
        # litellm's re-serialization -- not in the order the trace stores.
        calls = [{"id": "c", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]
        assert BM.render_transcript([{"role": "assistant", "tool_calls": calls}]) == (
            '\nAssistant (tool call): [{"function": {"arguments": "{\\"command\\": \\"ls\\"}", "name": "bash"}, '
            '"id": "c", "type": "function"}]'
        )
