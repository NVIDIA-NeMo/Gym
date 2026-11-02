# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cline's harness on synthetic transcripts: the transcript as tool items, result framing, the tool documentation
taken from the system prompt, simulation of a reply's calls, and verification."""

import copy
import json
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import MASK_SAMPLE_FIELD
from resources_servers.agentic_turn_simulation.app import (
    AgenticTurnSimulationRunRequest,
    AgenticTurnSimulationVerifyRequest,
    SimulateToolCallRequest,
)
from resources_servers.agentic_turn_simulation.harnesses import cline, select_harness
from resources_servers.agentic_turn_simulation.harnesses.cline import CLINE
from resources_servers.agentic_turn_simulation.judge import load_prompt
from resources_servers.agentic_turn_simulation.tests.support import VERDICT, chat, make_server, req, script


ENV = "<environment_details>\n# Visual Studio Code Visible Files\nsrc/app.ts\n\n# Current Mode\nACT MODE\n</environment_details>"
SYSTEM = """You are Cline, a synthetic coding assistant.

TOOL USE

# Tools

## execute_command
Description: Run a CLI command on the user's machine.
Parameters:
- command: (required) The command to run.
Usage:
<execute_command>
<command>Your command here</command>
</execute_command>

## read_file
Description: Read a file. Lines come back labelled `1 | `.
Parameters:
- path: (required) The file path.

## attempt_completion
Description: Present the result of the task to the user.
Parameters:
- result: (required) The result.

# Tool Use Examples

Nothing here."""
COMPACT_SYSTEM = """## TOOLS

**execute_command** — Run terminal commands.
Params: command, requires_approval.
*Example:*
<execute_command>
<command>npm run build</command>
</execute_command>

**read_file** — Read file.
Params: path.

## RULES

Be brief."""


def _parts(*texts: str) -> list[dict[str, Any]]:
    return [{"type": "input_text", "text": t} for t in texts]


def _user(*texts: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": _parts(*texts)}


def _assistant(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": text}


READ_CALL = "I'll check the entry point.\n\n<read_file>\n<path>src/app.ts</path>\n</read_file>"
READ_RESULT = (
    "[read_file for 'src/app.ts'] Result:\n1 | export const port = 3000;\n\n(File has 1 lines total.)\n" + ENV
)
HISTORY = [
    {"type": "message", "role": "system", "content": _parts(SYSTEM)},
    _user("<task>\nWhat port does the app use?\n</task>", ENV),
    _assistant(READ_CALL),
    _user(READ_RESULT),
    _assistant("<attempt_completion>\n<result>\nIt listens on port 3000.\n</result>\n</attempt_completion>"),
    _user(
        "[attempt_completion] Result: Done",
        "The user has provided feedback on the results. Consider their input to continue the task, and then attempt completion again.",
        "<feedback>\nMake it 8080.\n</feedback>",
        ENV,
    ),
]
METADATA = {"harness": "cline", "protocol": "cline_xml", "record_kind": "interactive", "source": "synthetic_cline"}


class TestAdapter:
    def test_calls_results_and_replies(self) -> None:
        items = CLINE.to_tool_items(HISTORY)
        kinds = [(x.get("type"), x.get("role") or x.get("name")) for x in items]
        assert kinds == [
            ("message", "system"),
            ("message", "user"),
            ("message", "assistant"),
            ("function_call", "read_file"),
            ("function_call_output", None),
            ("message", "assistant"),
            ("message", "user"),
        ]
        assert (
            items[1]["content"] == "<task>\nWhat port does the app use?\n</task>\n" + ENV
        )  # parts on their own lines
        assert items[2]["content"] == "I'll check the entry point."
        assert json.loads(items[3]["arguments"]) == {"path": "src/app.ts"}
        assert items[4] == {"type": "function_call_output", "call_id": items[3]["call_id"], "output": READ_RESULT}
        assert items[5]["content"] == "It listens on port 3000."  # attempt_completion is what the agent said
        assert items[6]["content"].startswith("[attempt_completion] Result: Done\nThe user has provided feedback")

    def test_several_calls_share_one_result_message(self) -> None:
        reply = _assistant(
            "<list_files>\n<path>src</path>\n</list_files>\nAlso:\n<read_file>\n<path>a.ts</path>\n</read_file>"
        )
        result = _user(
            "[list_files for 'src'] Result:\na.ts\n[read_file for 'a.ts'] Result:\n1 | x\n\n(File has 1 lines total.)\n"
            + ENV
        )
        items = CLINE.to_tool_items([reply, result])
        assert [x["type"] for x in items] == [
            "function_call",
            "message",
            "function_call",
            "function_call_output",
            "function_call_output",
        ]
        first, second = items[3]["output"], items[4]["output"]
        assert first == "[list_files for 'src'] Result:\na.ts"
        assert second.startswith("[read_file for 'a.ts'] Result:") and second.endswith(ENV)
        assert items[3]["call_id"] == items[0]["call_id"] and items[4]["call_id"] == items[2]["call_id"]

    def test_feedback_before_the_header_stays_with_the_result(self) -> None:
        reply = _assistant("<execute_command>\n<command>rm -rf build</command>\n</execute_command>")
        result = _user(
            "The user provided the following feedback:\n<feedback>\nnot that\n</feedback>\n[execute_command for 'rm -rf build'] Result:\nThe user denied this operation.\n"
            + ENV
        )
        items = CLINE.to_tool_items([reply, result])
        assert items[1]["type"] == "function_call_output" and items[1]["output"].startswith("The user provided")

    def test_a_reply_without_calls_and_unanswered_messages_pass_through(self) -> None:
        nudge = _user("[ERROR] You did not use a tool in your previous response! Please retry with a tool use.")
        items = CLINE.to_tool_items([_assistant("Thinking out loud."), nudge])
        assert [x["type"] for x in items] == ["message", "message"] and items[1]["role"] == "user"


class TestFraming:
    @pytest.mark.parametrize(
        "name, params, header",
        [
            ("read_file", {"path": "src/a.ts"}, "[read_file for 'src/a.ts'] Result:"),
            (
                "execute_command",
                {"command": "npm test", "requires_approval": "false"},
                "[execute_command for 'npm test'] Result:",
            ),
            (
                "search_files",
                {"path": ".", "regex": "TODO", "file_pattern": "*.ts"},
                "[search_files for 'TODO' in '*.ts'] Result:",
            ),
            ("search_files", {"path": ".", "regex": "TODO"}, "[search_files for 'TODO'] Result:"),
            ("use_skill", {"skill_name": "deploy"}, '[use_skill for "deploy"] Result:'),
            ("load_mcp_documentation", {}, "[load_mcp_documentation] Result:"),
        ],
    )
    def test_headers(self, name: str, params: dict[str, str], header: str) -> None:
        assert cline.result_header(name, params) == header

    def test_environment_details_close_only_the_last_result(self) -> None:
        call = {"name": "read_file", "arguments": json.dumps({"path": "a.ts"})}
        body = "1 | x\n\n(File has 1 lines total.)\n" + ENV
        assert CLINE.frame(call, body, more_follow=False) == "[read_file for 'a.ts'] Result:\n" + body
        assert (
            CLINE.frame(call, body, more_follow=True)
            == "[read_file for 'a.ts'] Result:\n1 | x\n\n(File has 1 lines total.)"
        )


class TestToolDocs:
    def test_full_prompt_sections(self) -> None:
        docs = cline.tool_docs(HISTORY)
        assert [d["name"] for d in docs] == ["execute_command", "read_file", "attempt_completion"]
        assert (
            docs[0]["description"].startswith("Description: Run a CLI command")
            and "<execute_command>" in docs[0]["description"]
        )
        assert docs[2]["description"].endswith("- result: (required) The result.")

    def test_compact_prompt_entries(self) -> None:
        docs = cline.tool_docs([{"type": "message", "role": "system", "content": COMPACT_SYSTEM}])
        assert [d["name"] for d in docs] == ["execute_command", "read_file"]
        assert (
            docs[0]["description"].startswith("Run terminal commands.")
            and docs[1]["description"] == "Read file.\nParams: path."
        )


# --- the server ---------------------------------------------------------------------------------------------


def _server(**overrides: Any):
    return make_server(tool_simulator_max_attempts=1, **overrides)


def _rcp() -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(input=copy.deepcopy(HISTORY), tools=[])


async def _seed(server):
    """(what /harness then tells the agent, the session)."""
    await server.seed_session(
        req(), AgenticTurnSimulationRunRequest(responses_create_params=_rcp(), verifier_metadata=METADATA)
    )
    return await server.describe_harness(req()), server._sessions["s1"]


def _turn_call(k: int, name: str, params: dict[str, str], turn: list[dict[str, Any]]) -> SimulateToolCallRequest:
    return SimulateToolCallRequest(call_id=f"xml_1_{k}", name=name, arguments=json.dumps(params), turn=turn)


def _response(items: list[dict[str, Any]], stop_reason: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "r",
            "created_at": 0.0,
            "model": "p",
            "object": "response",
            "output": items,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "metadata": {"stop_reason": stop_reason, "steps": "2"},
        }
    )


def _output_message(text: str) -> dict[str, Any]:
    return {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class TestServer:
    def test_selection(self) -> None:
        assert select_harness(METADATA) is CLINE and CLINE.protocol == "cline_xml"

    async def test_seed_selects_the_protocol_and_learns_the_files(self) -> None:
        reply, state = await _seed(_server())
        assert (reply.stop_tools, reply.protocol) == (
            ["ask_followup_question", "attempt_completion", "plan_mode_respond"],
            "cline_xml",
        )
        assert state.workspace.files == {"src/app.ts": "export const port = 3000;"}
        tools = state.simulator_preamble.split("</agent_system_prompt>\n\n", 1)[1]
        assert re.findall(r'<tool name="(\w+)">', tools) == ["execute_command", "read_file", "attempt_completion"]
        assert tools.startswith("<tools>\n") and tools.endswith("</tools>\n\n")
        assert any(x.get("type") == "function_call" for x in state.prefix)

    async def test_a_replys_calls_are_simulated_in_order_with_cline_framing(self) -> None:
        server = _server()
        await _seed(server)
        sent = script(server, [chat("a.ts\nb.ts\n" + ENV), chat("Command executed.\nOutput:\nok\n" + ENV)])
        reply = {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Two things.\n<list_files>\n<path>src</path>\n</list_files>\n<execute_command>\n<command>npm test</command>\n<requires_approval>false</requires_approval>\n</execute_command>",
                }
            ],
        }

        first = await server.simulate_tool_call(req(), _turn_call(1, "list_files", {"path": "src"}, [reply]))
        assert (first.source, first.output) == (
            "llm",
            "[list_files for 'src'] Result:\na.ts\nb.ts",
        )  # more results follow
        partial = {"type": "message", "role": "user", "content": _parts(first.output)}
        second = await server.simulate_tool_call(
            req(),
            _turn_call(2, "execute_command", {"command": "npm test", "requires_approval": "false"}, [reply, partial]),
        )
        assert second.output == "[execute_command for 'npm test'] Result:\nCommand executed.\nOutput:\nok\n" + ENV

        system, user = (m["content"] for m in sent[1]["json"]["messages"])
        assert system == load_prompt("tool-simulator-cline.md")
        assert user.startswith("<agent_system_prompt>\nYou are Cline, a synthetic coding assistant.")
        prompt, user = user.split("\n</agent_system_prompt>\n\n", 1)
        assert "# Tool Use Examples" in prompt  # the whole system prompt, before the tools and the conversation
        assert user.startswith('<tools>\n<tool name="execute_command">\nDescription: Run a CLI command')
        before, call = user.split("<tool_call_to_simulate>")
        assert '<tool_call id="' in call and 'name="execute_command"' in call and "npm test" in call
        assert "[list_files for 'src'] Result:" in before and "Two things." in before  # the sibling's result
        assert '<message from="User">\n[list_files' not in user

    async def test_a_simulator_failure_is_framed_and_masks_the_rollout(self) -> None:
        server = _server()
        await _seed(server)
        script(server, [chat("")])
        reply = _assistant("<read_file>\n<path>missing.ts</path>\n</read_file>")
        out = await server.simulate_tool_call(req(), _turn_call(1, "read_file", {"path": "missing.ts"}, [reply]))
        assert out.source == "failed" and out.output.startswith("[read_file for 'missing.ts'] Result:\n{")
        items = [
            _output_message(reply["content"]),
            {"type": "message", "role": "user", "content": _parts(out.output)},
            _output_message("Hmm."),
        ]
        verified = await server.verify(
            req(),
            AgenticTurnSimulationVerifyRequest(
                responses_create_params=_rcp(), response=_response(items, "final_message"), verifier_metadata=METADATA
            ),
        )
        assert getattr(verified, MASK_SAMPLE_FIELD) is True and verified.failure_kind.endswith("simulator_failed")

    async def test_an_unframed_simulator_failure_masks_the_rollout(self) -> None:
        # A lost session's failure comes back unframed, so the scored turn keeps it as a user message.
        server = _server()
        lost = await server.simulate_tool_call(req("gone"), _turn_call(1, "read_file", {"path": "a.ts"}, []))
        assert lost.source == "failed" and not lost.output.startswith("[read_file")
        items = [
            _output_message("<read_file>\n<path>a.ts</path>\n</read_file>"),
            {"type": "message", "role": "user", "content": _parts(lost.output)},
            _output_message("<attempt_completion>\n<result>\nDone.\n</result>\n</attempt_completion>"),
        ]
        out = await server.verify(
            req(),
            AgenticTurnSimulationVerifyRequest(
                responses_create_params=_rcp(),
                response=_response(items, "attempt_completion"),
                verifier_metadata=METADATA,
            ),
        )
        assert getattr(out, MASK_SAMPLE_FIELD) is True and out.failure_kind.endswith("simulator_failed")
        assert out.failure_reason == lost.output
        server.server_client.post.assert_not_called()

    async def test_the_judge_sees_tool_use_not_user_messages(self) -> None:
        server = _server()
        sent = script(server, [chat(json.dumps(VERDICT))])
        result = (
            "[execute_command for 'sed -i s/3000/8080/ src/app.ts'] Result:\nCommand failed with exit code 2.\nOutput:\nsed: no such file\n"
            + ENV
        )
        items = [
            _output_message(
                "<execute_command>\n<command>sed -i s/3000/8080/ src/app.ts</command>\n</execute_command>"
            ),
            {"type": "message", "role": "user", "content": _parts(result)},
            _output_message(
                "<attempt_completion>\n<result>\nChanged the port to 8080.\n</result>\n</attempt_completion>"
            ),
        ]
        out = await server.verify(
            req(),
            AgenticTurnSimulationVerifyRequest(
                responses_create_params=_rcp(),
                response=_response(items, "attempt_completion"),
                verifier_metadata=METADATA,
            ),
        )
        assert out.reward == 1.0 and out.code_gate is None and out.shell_failures == 1 and out.harness == "cline_xml"
        assert (out.ended_on_stop_tool, out.executed_calls) == ("attempt_completion", 1)
        system, user = (m["content"] for m in sent[0]["json"]["messages"])
        turn = user.split("<step_under_evaluation>")[1]
        assert '<tool_call id="' in turn and 'name="execute_command"' in turn and "<tool_result" in turn
        assert 'from="User"' not in turn and turn.rstrip().endswith(
            '<message from="Agent">\nChanged the port to 8080.\n</message>\n</step_under_evaluation>'
        )
        conversation = user.split("<step_under_evaluation>")[0]
        assert '<tool_call id="' in conversation and 'name="read_file"' in conversation
        assert '<message from="User">\n<task>' in conversation
        assert "The agent is Cline" in system and '<tool name="read_file">' in system

    async def test_only_the_calls_before_a_question_ran(self) -> None:
        # Cline runs a reply's calls in order and stops at the question: the calls after it never ran.
        server = _server()
        script(server, [chat(json.dumps(VERDICT))])
        reply = _output_message(
            "<read_file>\n<path>src/app.ts</path>\n</read_file>\n"
            "<ask_followup_question>\n<question>Which port?</question>\n</ask_followup_question>\n"
            "<list_files>\n<path>src</path>\n</list_files>"
        )
        result = "[read_file for 'src/app.ts'] Result:\n1 | export const port = 3000;\n\n(File has 1 lines total.)\n"
        items = [reply, {"type": "message", "role": "user", "content": _parts(result + ENV)}]
        out = await server.verify(
            req(),
            AgenticTurnSimulationVerifyRequest(
                responses_create_params=_rcp(),
                response=_response(items, "ask_followup_question"),
                verifier_metadata=METADATA,
            ),
        )
        assert (out.ended_on_stop_tool, out.executed_calls) == ("ask_followup_question", 1)

    async def test_calls_are_never_argument_gated(self) -> None:
        # Cline's tools are documented in prose, not schemas: a call without a documented required parameter
        # (requires_approval) or with an undocumented one goes to the judge.
        assert CLINE.gates_missing_arguments is False
        server = _server()
        sent = script(server, [chat(json.dumps(VERDICT))])
        result = "[execute_command for 'ls'] Result:\napp.ts\n" + ENV
        items = [
            _output_message("<execute_command>\n<command>ls</command>\n<shell>zsh</shell>\n</execute_command>"),
            {"type": "message", "role": "user", "content": _parts(result)},
            _output_message("<attempt_completion>\n<result>\nListed it.\n</result>\n</attempt_completion>"),
        ]
        out = await server.verify(
            req(),
            AgenticTurnSimulationVerifyRequest(
                responses_create_params=_rcp(),
                response=_response(items, "attempt_completion"),
                verifier_metadata=METADATA,
            ),
        )
        assert (out.code_gate, out.reward, out.harness, len(sent)) == (None, 1.0, "cline_xml", 1)

    async def test_a_reference_turn_ending_on_a_question_is_judged_without_the_answer(self) -> None:
        server = _server(score_source="reference_turn")
        sent = script(server, [chat(json.dumps(VERDICT))])
        reference = [
            _assistant("<ask_followup_question>\n<question>Which port?</question>\n</ask_followup_question>"),
            _user("[ask_followup_question for 'Which port?'] Result:\n<answer>\n8080\n</answer>\n" + ENV),
        ]
        out = await server.verify(
            req(),
            AgenticTurnSimulationVerifyRequest(
                responses_create_params=_rcp(),
                response=_response([], "none"),
                verifier_metadata={**METADATA, "reference_turn": reference},
            ),
        )
        assert not getattr(out, MASK_SAMPLE_FIELD, False) and out.reward == 1.0
        turn = sent[0]["json"]["messages"][1]["content"].split("<step_under_evaluation>")[1]
        assert 'name="ask_followup_question"' in turn and "8080" not in turn

    def test_the_harness_endpoint_names_the_protocol(self) -> None:
        client = TestClient(_server().setup_webserver())
        seed = AgenticTurnSimulationRunRequest(responses_create_params=_rcp(), verifier_metadata=METADATA)
        assert client.post("/seed_session", json=seed.model_dump(mode="json", exclude_none=True)).json() == {}
        assert client.post("/harness").json() == {
            "stop_tools": ["ask_followup_question", "attempt_completion", "plan_mode_respond"],
            "protocol": "cline_xml",
        }
