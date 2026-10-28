# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the simulator LLM is shown of a call, how its replies are read, and the failure marker."""

import json
from typing import Any

import pytest

from resources_servers.agentic_turn_simulation.harnesses.sandbox import SandboxWorkspace
from resources_servers.agentic_turn_simulation.simulator import (
    SIMULATOR_FAILED,
    holds_simulator_failure,
    parse_json_text_reply,
    parse_text_reply,
    simulation_history,
    simulator_failure,
    simulator_user_message,
)


def _call(name: str, call_id: str = "c1", **args: Any) -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args)}


def _out(call_id: str, output: Any) -> dict[str, Any]:
    if not isinstance(output, str):
        output = json.dumps(output)
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _user(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": text}


def _assistant(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": text}


class TestSimulatorFailure:
    def test_simulator_failure_is_marked_and_bounded(self) -> None:
        out = json.loads(simulator_failure("x" * 500))
        assert out["status"] == "error"
        assert out["message"].startswith(SIMULATOR_FAILED + ": ")
        assert out["message"] == f"{SIMULATOR_FAILED}: {'x' * 120}]"

    @pytest.mark.parametrize(
        "text",
        [
            simulator_failure("HTTP 503: busy"),
            simulator_failure('a "quoted"\nreason'),
            f"[read_file for 'a.py'] Result:\n{simulator_failure('timeout')}",
            f"{simulator_failure('HTTP 500: oops')}\n[read_file for 'b.py'] Result:\nprint(1)",
            f"{simulator_failure('x')[:-1]}\n{simulator_failure('y')}",
        ],
        ids=["whole_result", "escaped_reason", "cline_framed", "cline_joined_unframed", "after_a_broken_copy"],
    )
    def test_a_complete_failure_output_is_found(self, text: str) -> None:
        assert holds_simulator_failure(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            f"{SIMULATOR_FAILED}: HTTP 503]",
            f'12: SIMULATOR_FAILED = "{SIMULATOR_FAILED}"',
            simulator_failure("x")[:-1],
            json.dumps({"status": "error", "message": f"{SIMULATOR_FAILED}: x]", "code": 1}),
            json.dumps({"status": "error", "message": f"{SIMULATOR_FAILED} x]"}),
            json.dumps({"status": "error", "message": f"{SIMULATOR_FAILED}: x"}),
        ],
        ids=["empty", "phrase_only", "phrase_in_code", "cut_off", "extra_key", "no_reason", "unclosed_message"],
    )
    def test_the_phrase_or_another_object_is_not_a_failure(self, text: str) -> None:
        assert not holds_simulator_failure(text)


class TestSimulationHistory:
    def test_first_step_single_call(self) -> None:
        msg, call = _assistant("let me look"), _call("bash", "c1", command="ls")
        assert simulation_history([msg, call], "c1") == [msg]

    def test_parallel_calls_see_the_step_message_and_earlier_sibling_outputs(self) -> None:
        m1, c1, o1 = _assistant("step 1"), _call("bash", "c1", command="ls"), _out("c1", {"status": "completed"})
        m2 = _assistant("step 2")
        c2a, c2b, c2c = (_call("bash", f"c2{x}", command=x) for x in "abc")
        o2a = _out("c2a", {"stdout": "A"})
        o2b = _out("c2b", {"stdout": "B"})
        turn = [m1, c1, o1, m2, c2a, c2b, c2c, o2a, o2b]
        before = simulation_history(turn, "c2c")
        # earlier steps whole, then this step's message (not its calls), then the outputs already produced
        assert before == [m1, c1, o1, m2, o2a, o2b]

    def test_first_of_parallel_calls(self) -> None:
        m, ca, cb = _assistant("x"), _call("bash", "a"), _call("bash", "b")
        assert simulation_history([m, ca, cb], "a") == [m]

    def test_a_repeated_call_id_is_the_latest_call(self) -> None:
        # Parsers that take ids from the model's text (functions.bash:0) can repeat one across steps.
        c1, o1 = _call("bash", "functions.bash:0", command="ls"), _out("functions.bash:0", "a.py")
        m2, c2 = _assistant("now reading"), _call("bash", "functions.bash:0", command="cat a.py")
        assert simulation_history([c1, o1, m2, c2], "functions.bash:0") == [c1, o1, m2]

    def test_a_turn_without_the_call_is_all_history(self) -> None:
        turn = [_assistant("x"), _out("zz", {})]
        before = simulation_history(turn, "zz")
        assert before == turn and before is not turn


class TestSimulatorUserMessage:
    def test_tags_are_renamed_and_workspace_files_attached(self) -> None:
        ws = SandboxWorkspace()
        ws.files["/home/user/app.py"] = "print('hi')"
        text = simulator_user_message(_call("bash", command="python /home/user/app.py"), [_user("run it")], ws)
        assert "<step_under_evaluation>" not in text
        assert text.index("<tool_call_to_simulate>") < text.index("python /home/user/app.py")
        assert '<message from="User">\nrun it\n</message>' in text
        assert text.endswith(
            "<workspace_files>\n<file path=\"/home/user/app.py\">\nprint('hi')\n</file>\n</workspace_files>"
        )

    def test_no_workspace_section_without_relevant_files(self) -> None:
        text = simulator_user_message(_call("web_search", query="weather"), [_user("hi")], SandboxWorkspace())
        assert "<workspace_files>" not in text and text.endswith("</tool_call_to_simulate>")


# --- replies ------------------------------------------------------------------------------------------------


class TestParseTextReply:
    def test_verbatim_with_leading_spaces_kept(self) -> None:
        text = "     1\tdef parse(s):\n     2\t    return int(s or 0)"
        assert parse_text_reply({"content": text}) == text
        assert parse_text_reply({"content": "\n" + text + "\n\n"}) == text

    def test_json_output_is_not_reserialised(self) -> None:
        raw = '{"content":"snow","total_lines":1}'
        assert parse_text_reply({"content": raw}) == raw

    def test_inline_reasoning_is_dropped(self) -> None:
        assert parse_text_reply({"content": "<think>it exists</think>\n\nok: 3 files"}) == "ok: 3 files"

    @pytest.mark.parametrize(
        "message, finish",
        [
            ({"content": "partial outp"}, "length"),
            ({"content": ""}, "stop"),
            ({"content": None, "reasoning_content": "the output is: hello"}, "stop"),
            ({"content": "<think>only thinking</think>\n  "}, "stop"),
        ],
        ids=["cut-off", "empty", "reasoning-channel-only", "think-only"],
    )
    def test_unusable(self, message: dict[str, Any], finish: str) -> None:
        assert parse_text_reply(message, finish) is None


class TestJsonTextReplies:
    @pytest.mark.parametrize(
        "content, expected",
        [
            ('{"output": "ok", "exit_code": 0, "error": null}', '{"output": "ok", "exit_code": 0, "error": null}'),
            ('```json\n{"total_count": 0}\n```', '{"total_count": 0}'),
            (
                '{"total_count": 3}\n\n[Hint: Results truncated. Use offset=50 to see more.]',
                '{"total_count": 3}\n\n[Hint: Results truncated. Use offset=50 to see more.]',
            ),
            (
                '{"bytes_written": 3, "dirs_created": true}\nThe file was written.',
                '{"bytes_written": 3, "dirs_created": true}',
            ),
            (
                "Tool 'browse' does not exist. Available tools: clarify, terminal",
                "Tool 'browse' does not exist. Available tools: clarify, terminal",
            ),
            ('{"output": "unfinished', None),
        ],
        ids=["object", "fenced", "with-hint", "commentary-dropped", "plain-text", "unfinished"],
    )
    def test_parse(self, content: str, expected: str | None) -> None:
        assert parse_json_text_reply({"content": content}) == expected
