# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness selection and the registry, and the generic harness end to end on hand-written rows shaped like a
Claude Code and a Hermes trace but from harnesses without a module of their own (synthetic data),
including the row's system prompt shown to the simulator and the judge. Model servers are scripted."""

import copy
import json
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import MASK_SAMPLE_FIELD
from resources_servers.agentic_turn_simulation import harnesses
from resources_servers.agentic_turn_simulation.app import (
    AgenticTurnSimulationRunRequest,
    AgenticTurnSimulationVerifyRequest,
    SimulateToolCallRequest,
)
from resources_servers.agentic_turn_simulation.harnesses import GENERIC, HARNESSES, select_harness
from resources_servers.agentic_turn_simulation.harnesses.sandbox import KNOWN_TOOLS, SANDBOX
from resources_servers.agentic_turn_simulation.items import UNPARSEABLE_ARGS_KEY
from resources_servers.agentic_turn_simulation.judge import build_system_prompt, load_prompt
from resources_servers.agentic_turn_simulation.render import (
    JUDGE_RENDER,
    SIMULATOR_SYSTEM_PROMPT_CHARS,
    render_conversation,
    render_system_prompt,
)
from resources_servers.agentic_turn_simulation.tests.support import (
    VERDICT,
    chat,
    function_call,
    function_output,
    harness_row,
    make_server,
    req,
    script,
    seed_row,
    sim,
    text_message,
)
from resources_servers.agentic_turn_simulation.workspace import FileToolWorkspace


SUBAGENT_NOTE = load_prompt("judge-requester-subagent.md")
# Sandbox harness rows name it (rows without a harness run under generic).
SANDBOX_METADATA = {"harness": "sandbox"}


def _tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": f"The {name} tool.",
        "parameters": {"type": "object"},
        "strict": False,
    }


def _reasoning(text: str, item_id: str = "rs_1") -> dict[str, Any]:
    return {"type": "reasoning", "id": item_id, "summary": [], "content": [{"type": "reasoning_text", "text": text}]}


# --- synthetic rows -----------------------------------------------------------------------------------------


CLAUDE_CODE_TOOLS = [_tool(n) for n in ("Bash", "Read", "Write", "Edit", "AskUserQuestion")]
CLAUDE_CODE_ROW = {
    "input": [
        text_message(
            "system", "You are a coding agent.\n<env>\nWorking directory: /Users/sam/proj\nPlatform: darwin\n</env>"
        ),
        text_message("user", "Why does test_parse fail?"),
        _reasoning("PRIOR_TURN_REASONING"),
        function_call("Read", "t1", {"file_path": "/Users/sam/proj/parse.py"}),
        function_output("t1", "     1\tdef parse(s):\n     2\t    return int(s)\n"),
        text_message("assistant", "parse() calls int() on the raw string, so a blank line raises ValueError."),
        text_message("user", "<system-reminder>The todo list is empty.</system-reminder>\nFix it so blanks give 0."),
    ],
    "metadata": {
        "harness": "goose",
        "harness_version": "2.1.220",
        "protocol": "native",
        "record_kind": "interactive",
        "source": "synthetic_claude_code",
        "trace_id": "synthetic-cc-1",
        "os": "macos",
        "categories": ["code_debugging"],
    },
}


HERMES_TOOLS = [_tool(n) for n in ("terminal", "read_file", "write_file", "patch", "clarify")]
HERMES_ROW = {
    "input": [
        text_message("system", "You are Hermes, a personal assistant on the user's computer."),
        text_message("user", "Save a haiku about rain to /home/user/notes/haiku.txt"),
        function_call("write_file", "h1", {"path": "/home/user/notes/haiku.txt", "content": "old pond"}),
        function_output("h1", json.dumps({"bytes_written": 8, "dirs_created": True})),
        text_message("assistant", "Saved to /home/user/notes/haiku.txt."),
        text_message("user", "Now make it about snow and read it back."),
    ],
    "metadata": {
        "harness": "aider",
        "harness_version": "0.20.5",
        "protocol": "native",
        "record_kind": "interactive",
        "source": "synthetic_hermes",
        "trace_id": "synthetic-hermes-1",
        "os": "linux",
    },
}
HERMES_SUBAGENT_ROW = {
    "input": [
        text_message("system", "You are a Hermes subagent. Complete the delegated task and report back."),
        text_message("user", "List the failing tests in /home/user/app and report their names."),
    ],
    "metadata": dict(HERMES_ROW["metadata"], record_kind="subagent", trace_id="synthetic-hermes-2"),
}


def _rcp(row: dict[str, Any], tools: list[dict[str, Any]]) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(input=copy.deepcopy(row["input"]), tools=tools)


async def _seed(server, row, tools, *, session_id: str = "s1", metadata: dict[str, Any] | None = None):
    """(what /harness then tells the agent, the session)."""
    metadata = row["metadata"] if metadata is None else metadata
    await server.seed_session(
        req(session_id),
        AgenticTurnSimulationRunRequest(responses_create_params=_rcp(row, tools), verifier_metadata=metadata),
    )
    return await server.describe_harness(req(session_id)), server._sessions[session_id]


def _response(items: list[dict[str, Any]], stop_reason: str = "final_message") -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "resp",
            "created_at": 0.0,
            "model": "policy",
            "object": "response",
            "output": items,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "metadata": {"stop_reason": stop_reason, "steps": "2"},
        }
    )


def _final(text: str) -> dict[str, Any]:
    return {
        "id": "m_final",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


async def _verify(server, row, tools, items, *, metadata=None, session_id: str = "s1"):
    metadata = row["metadata"] if metadata is None else metadata
    body = AgenticTurnSimulationVerifyRequest(
        responses_create_params=_rcp(row, tools), response=_response(items), verifier_metadata=metadata
    )
    return await server.verify(req(session_id), body)


# --- selection and the registry -----------------------------------------------------------------------------


class TestSelectHarness:
    @pytest.mark.parametrize(
        "metadata, expected",
        [
            (None, "generic"),
            ({}, "generic"),
            ({"categories": ["web_dev"]}, "generic"),
            ({"harness": "sandbox"}, "sandbox"),
            ({"harness": "oh_my_opencode", "protocol": "native"}, "opencode"),
            ({"harness": "generic"}, "generic"),
            ({"harness": "goose"}, "generic"),
            ({"harness": "claude_code"}, "claude_code"),
            ({"harness": "hermes"}, "hermes"),
            ({"harness": "openclaw"}, "openclaw"),
            ({"harness": "cline", "protocol": "cline_xml"}, "cline_xml"),
            ({"harness": "anything", "protocol": "cline_xml"}, "cline_xml"),
            ({"harness": "cline", "protocol": "native"}, "generic"),
            ({"harness": ["not", "a", "name"]}, "generic"),
        ],
    )
    def test_selection(self, metadata: dict[str, Any] | None, expected: str) -> None:
        assert select_harness(metadata).name == expected

    def test_an_unknown_harness_is_reported_once(self, caplog: pytest.LogCaptureFixture) -> None:
        harnesses._reported_unknown.discard('"aider"')
        with caplog.at_level(logging.WARNING, logger=harnesses.__name__):
            for _ in range(3):
                assert select_harness({"harness": "aider"}) is GENERIC
        assert [r.getMessage() for r in caplog.records] == ['unknown harness="aider"; using generic']

    def test_a_missing_harness_is_the_default_and_not_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=harnesses.__name__):
            assert select_harness(None) is GENERIC and select_harness({"categories": ["web_dev"]}) is GENERIC
        assert caplog.records == []


class TestSandboxHarness:
    """Rows naming the sandbox harness run under it, whatever else their metadata says."""

    METADATA = {"harness": "sandbox", "harness_version": None, "protocol": "native", "source": "synthetic"}

    async def test_seed_and_simulation_follow_the_sandbox_profile(self) -> None:
        server = make_server()
        tools = [_tool(n) for n in sorted(KNOWN_TOOLS)]
        reply, state = await _seed(server, HERMES_ROW, tools, metadata=self.METADATA)
        assert state.harness is SANDBOX and (reply.stop_tools, reply.protocol) == (["ask_user"], "native")
        write = await server.simulate_tool_call(
            req(), sim("write_file", "w1", {"path": "site/index.html", "content": "<h1>Hi</h1>"})
        )
        read = await server.simulate_tool_call(req(), sim("read_file", "r1", {"path": "/home/user/site/index.html"}))
        assert (write.source, read.source) == ("exact", "exact")
        assert json.loads(write.output)["status"] == "success" and json.loads(read.output)["content"] == "<h1>Hi</h1>"
        assert state.workspace.files == {"/home/user/site/index.html": "<h1>Hi</h1>"}

    async def test_verify_uses_the_sandbox_judge_frame(self) -> None:
        server = make_server()
        sent = script(server, [chat(json.dumps(VERDICT))])
        tools = [_tool(n) for n in sorted(KNOWN_TOOLS)]
        out = await _verify(server, HERMES_ROW, tools, [_final("Done.")], metadata=self.METADATA)
        assert sent[0]["json"]["messages"][0]["content"] == build_system_prompt(
            tools, template="judge-system-sandbox.md"
        )
        assert out.harness == "sandbox" and out.shell_failures == 0


class TestRegistry:
    def test_generic_answers_nothing_in_code(self) -> None:
        assert GENERIC.files is None and GENERIC.new_workspace({}) is None
        write = function_call("write_file", "w1", {"path": "a.txt", "content": "x"})
        assert GENERIC.exact_answer(write, history=[], workspace=None, tool_names=frozenset({"write_file"})) is None
        assert (GENERIC.reply_format, GENERIC.default_tools, GENERIC.shell_failures) == ("text", frozenset(), None)
        assert {"AskUserQuestion", "clarify", "question", "ask_user"} <= set(GENERIC.stop_tools)

    @pytest.mark.parametrize("harness", list(HARNESSES.values()), ids=list(HARNESSES))
    def test_prompts_load(self, harness) -> None:
        assert load_prompt(harness.simulator_prompt) and load_prompt(harness.judge_prompt)
        requester = harness.requester({"record_kind": "subagent"})
        assert requester.note_file is None or load_prompt(requester.note_file)

    def test_generic_prompts_assume_no_sandbox(self) -> None:
        for name in ("tool-simulator-generic.md", "judge-system-generic.md", "judge-requester-subagent.md"):
            text = load_prompt(name)
            for sandbox_only in ("Debian", "/home/user", "hosted sandbox", "Linux sandbox"):
                assert sandbox_only not in text, (name, sandbox_only)


HARNESS_PROMPTS = ["claude-code", "hermes", "openclaw", "opencode", "kilo-code", "cline"]


class TestFileToolHarnesses:
    @pytest.mark.parametrize("prompt", HARNESS_PROMPTS)
    def test_prompts_exist_and_assume_no_sandbox(self, prompt: str) -> None:
        for name in (f"tool-simulator-{prompt}.md", f"judge-environment-{prompt}.md"):
            text = load_prompt(name)
            for sandbox_only in ("Debian", "/home/user", "hosted sandbox", "Linux sandbox"):
                assert sandbox_only not in text, (name, sandbox_only)
        assert "<tools>" in load_prompt(f"tool-simulator-{prompt}.md")

    @pytest.mark.parametrize(
        "harness, stop_tools",
        [
            ("claude_code", ["AskUserQuestion", "ExitPlanMode"]),
            ("hermes", ["clarify"]),
            ("openclaw", ["ask_user"]),
            ("opencode", ["question", "plan_exit"]),
            ("oh_my_opencode", ["question", "plan_exit"]),
            ("kilo_code", ["question", "plan_exit", "ask_followup_question", "attempt_completion"]),
        ],
    )
    async def test_seed_names_the_harness_stop_tools(self, harness: str, stop_tools: list[str]) -> None:
        reply, state = await seed_row(
            make_server(), harness_row(harness, "1.0.0", ["read"], [text_message("user", "hi")])
        )
        assert (reply.stop_tools, reply.protocol) == (stop_tools, "native")
        assert state.harness is select_harness({"harness": harness}) and isinstance(state.workspace, FileToolWorkspace)

    @pytest.mark.parametrize(
        "prompt, windows_shell",
        [
            ("opencode", "On Windows it is PowerShell"),
            ("kilo-code", "On Windows it is PowerShell"),
            ("claude-code", "On Windows, `Bash` runs Git Bash"),
        ],
    )
    def test_prompts_say_which_shell_runs_commands_on_windows(self, prompt: str, windows_shell: str) -> None:
        # On Windows, opencode's and Kilo Code's bash tool run PowerShell (POSIX-only tools fail with
        # PowerShell's "not recognized" error); Claude Code's Bash runs Git Bash and its PowerShell tool PowerShell.
        text = load_prompt(f"tool-simulator-{prompt}.md")
        assert windows_shell in text and "is not recognized as the name of a cmdlet" in text

    def test_every_profile_names_its_prompts(self) -> None:
        for harness in HARNESSES.values():
            assert load_prompt(harness.simulator_prompt) and load_prompt(harness.judge_prompt)
            if harness.judge_environment:
                assert load_prompt(harness.judge_environment)


# --- /seed_session ------------------------------------------------------------------------------------------


class TestSeed:
    async def test_sandbox_row_stops_on_ask_user(self) -> None:
        server = make_server()
        reply, state = await _seed(server, HERMES_ROW, HERMES_TOOLS, metadata=SANDBOX_METADATA)
        assert reply.stop_tools == ["ask_user"] and state.harness is SANDBOX

    async def test_generic_row_gets_the_profile_stop_tools_and_keeps_history_as_is(self) -> None:
        server = make_server()
        reply, state = await _seed(server, HERMES_ROW, HERMES_TOOLS)
        assert reply.stop_tools == list(GENERIC.stop_tools) and state.harness is GENERIC
        assert state.tool_names == frozenset(t["name"] for t in HERMES_TOOLS)
        assert "/home/user/notes" in json.dumps(state.prefix)
        assert state.workspace is None  # nothing follows the files of a harness without file tools

    async def test_generic_row_without_tools_allows_none(self) -> None:
        _, state = await _seed(make_server(), HERMES_ROW, [])
        assert state.tool_names == frozenset()

    def test_seed_and_harness_over_http(self) -> None:
        client = TestClient(make_server().setup_webserver())
        seed = AgenticTurnSimulationRunRequest(
            responses_create_params=_rcp(CLAUDE_CODE_ROW, CLAUDE_CODE_TOOLS),
            verifier_metadata=CLAUDE_CODE_ROW["metadata"],
        )
        reply = client.post("/seed_session", json=seed.model_dump(mode="json", exclude_none=True))
        assert reply.status_code == 200 and reply.json() == {}
        assert client.post("/harness").json() == {"stop_tools": list(GENERIC.stop_tools), "protocol": "native"}


# --- /simulate_tool_call ------------------------------------------------------------------------------------


class TestGenericSimulation:
    async def test_claude_code_read_is_simulated_as_plain_text(self) -> None:
        server = make_server()
        _, state = await _seed(server, CLAUDE_CODE_ROW, CLAUDE_CODE_TOOLS)
        numbered = "     1\tdef parse(s):\n     2\t    return int(s or 0)"
        sent = script(server, [chat("<think>The edit is in the history.</think>\n" + numbered + "\n")])
        turn = [text_message("assistant", "Let me look at the file again.")]
        out = await server.simulate_tool_call(
            req(), sim("Read", "c1", {"file_path": "/Users/sam/proj/parse.py"}, turn)
        )

        assert (out.source, out.output) == ("llm", numbered) and state.answered_by == {"llm": 1}
        (request,) = sent
        system, user = (m["content"] for m in request["json"]["messages"])
        assert system == load_prompt("tool-simulator-generic.md")
        assert "/Users/sam/proj/parse.py" in user and "<system-reminder>The todo list is empty." in user
        assert '<tool_call id="c1" name="Read">' in user.split("<tool_call_to_simulate>")[1]
        assert "Let me look at the file again." in user.split("<tool_call_to_simulate>")[0]
        assert "PRIOR_TURN_REASONING" not in user and "<workspace_files>" not in user

    async def test_hermes_tools_named_like_the_sandbox_tools_go_to_the_llm(self) -> None:
        write = {"path": "/home/user/notes/haiku.txt", "content": "first snow"}
        read = {"path": "/home/user/notes/haiku.txt"}

        # The same calls in a sandbox session are answered in code: write_file, then read_file from the workspace.
        sandbox = make_server()
        await _seed(sandbox, HERMES_ROW, HERMES_TOOLS, metadata=SANDBOX_METADATA)
        sandbox_write = await sandbox.simulate_tool_call(req(), sim("write_file", "w1", write))
        sandbox_read = await sandbox.simulate_tool_call(req(), sim("read_file", "r1", read))
        assert (sandbox_write.source, sandbox_read.source) == ("exact", "exact")
        assert json.loads(sandbox_read.output)["content"] == "first snow"

        server = make_server()
        _, state = await _seed(server, HERMES_ROW, HERMES_TOOLS)
        written = json.dumps({"bytes_written": 10, "dirs_created": False})
        content = '{"content": "first snow", "total_lines": 1, "file_size": 10}'
        sent = script(server, [chat(written), chat(content)])
        out_write = await server.simulate_tool_call(req(), sim("write_file", "w1", write))
        out_read = await server.simulate_tool_call(req(), sim("read_file", "r1", read))
        assert [(o.source, o.output) for o in (out_write, out_read)] == [("llm", written), ("llm", content)]
        assert len(sent) == 2 and state.answered_by == {"llm": 2} and state.workspace is None
        user = sent[1]["json"]["messages"][1]["content"]
        assert "/home/user/notes/haiku.txt" in user

    async def test_simulator_text_is_returned_verbatim(self) -> None:
        server = make_server()
        await _seed(server, HERMES_ROW, HERMES_TOOLS)
        script(server, [chat("cat: /home/alex/x: No such file or directory\n[exit code 1]")])
        out = await server.simulate_tool_call(req(), sim("terminal", "t1", {"command": "cat /home/alex/x"}))
        assert out.output == "cat: /home/alex/x: No such file or directory\n[exit code 1]"

    @pytest.mark.parametrize(
        "name, arguments", [("terminal", "{not json"), ("browse", '{"url": "x"}')], ids=["malformed", "unknown-tool"]
    )
    async def test_calls_the_sandbox_answers_with_its_harness_error_go_to_the_llm(
        self, name: str, arguments: str
    ) -> None:
        server = make_server()
        await _seed(server, HERMES_ROW, HERMES_TOOLS)
        script(server, [chat("Error: invalid tool call")])
        out = await server.simulate_tool_call(
            req(), SimulateToolCallRequest(call_id="x1", name=name, arguments=arguments)
        )
        assert (out.source, out.output) == ("llm", "Error: invalid tool call")

    async def test_cut_off_or_empty_text_replies_are_retried(self, sleep: AsyncMock) -> None:
        server = make_server()
        await _seed(server, HERMES_ROW, HERMES_TOOLS)
        replies = [
            chat("partial outp", finish="length"),
            chat(""),
            chat(None, reasoning_content="I think the output is: hi"),
            chat("hi"),
        ]
        sent = script(server, replies)
        out = await server.simulate_tool_call(req(), sim("terminal", "t1", {"command": "echo hi"}))
        assert (out.source, out.output) == ("llm", "hi")
        assert [s["json"]["max_tokens"] for s in sent] == [1000, 2000, 4000, 8000]


# --- /verify ------------------------------------------------------------------------------------------------


def _policy_turn() -> list[dict[str, Any]]:
    return [
        {"type": "reasoning", "id": "r", "summary": [{"type": "summary_text", "text": "POLICY_REASONING"}]},
        function_call(
            "Edit",
            "p1",
            {"file_path": "/Users/sam/proj/parse.py", "old_string": "int(s)", "new_string": "int(s or 0)"},
        ),
        function_output("p1", "The file /Users/sam/proj/parse.py has been updated."),
        _final("Blank lines now parse as 0."),
    ]


class TestSystemPrompt:
    def test_every_profile_but_sandbox_shows_it(self) -> None:
        assert {name for name, p in HARNESSES.items() if not p.show_system_prompt} == {"sandbox"}

    async def test_the_judge_sees_it_whole_and_the_simulator_its_head_and_tail(self) -> None:
        server = make_server()
        long = "HEAD " + "x" * 100_000 + " TAIL"
        row = copy.deepcopy(CLAUDE_CODE_ROW)
        row["input"][0] = text_message("system", long)
        _, state = await _seed(server, row, CLAUDE_CODE_TOOLS)
        assert state.simulator_preamble == render_system_prompt(long, SIMULATOR_SYSTEM_PROMPT_CHARS)
        assert state.simulator_preamble.startswith("<agent_system_prompt>\nHEAD ")
        assert (
            " TAIL\n</agent_system_prompt>\n\n" in state.simulator_preamble and len(state.simulator_preamble) < 61_000
        )

        sent = script(server, [chat(json.dumps(VERDICT))])
        await _verify(server, row, CLAUDE_CODE_TOOLS, _policy_turn())
        _, user = (m["content"] for m in sent[0]["json"]["messages"])
        assert user.startswith(f"<agent_system_prompt>\n{long}\n</agent_system_prompt>\n\n<conversation>\n")

    async def test_instructions_come_before_the_system_messages(self) -> None:
        server = make_server()
        sent = script(server, [chat(json.dumps(VERDICT))])
        rcp = NeMoGymResponseCreateParamsNonStreaming(
            input=copy.deepcopy(CLAUDE_CODE_ROW["input"]), tools=CLAUDE_CODE_TOOLS, instructions="Be terse."
        )
        body = AgenticTurnSimulationVerifyRequest(
            responses_create_params=rcp,
            response=_response(_policy_turn()),
            verifier_metadata=CLAUDE_CODE_ROW["metadata"],
        )
        await server.verify(req(), body)
        _, user = (m["content"] for m in sent[0]["json"]["messages"])
        assert user.startswith("<agent_system_prompt>\nBe terse.\n\nYou are a coding agent.\n<env>")

    async def test_a_row_without_one_gets_no_block(self) -> None:
        server = make_server()
        row = copy.deepcopy(CLAUDE_CODE_ROW)
        row["input"] = row["input"][1:]
        _, state = await _seed(server, row, CLAUDE_CODE_TOOLS)
        assert state.simulator_preamble == ""
        sent = script(server, [chat(json.dumps(VERDICT))])
        await _verify(server, row, CLAUDE_CODE_TOOLS, _policy_turn())
        _, user = (m["content"] for m in sent[0]["json"]["messages"])
        assert user.startswith("<conversation>\n")

    async def test_sandbox_shows_neither_the_simulator_nor_the_judge(self) -> None:
        server = make_server()
        _, state = await _seed(server, CLAUDE_CODE_ROW, CLAUDE_CODE_TOOLS, metadata=SANDBOX_METADATA)
        assert state.harness is SANDBOX and state.simulator_preamble == ""


class TestSystemPromptOverride:
    """show_agent_system_prompt forces the block on or off for every harness; unset, each harness decides."""

    async def test_forced_on_for_sandbox_rows(self) -> None:
        server = make_server(show_agent_system_prompt=True)
        tools = [_tool(n) for n in sorted(KNOWN_TOOLS)]
        _, state = await _seed(server, CLAUDE_CODE_ROW, tools, metadata=SANDBOX_METADATA)
        assert state.harness is SANDBOX and state.simulator_preamble.startswith(
            "<agent_system_prompt>\nYou are a coding agent."
        )
        assert state.simulator_preamble.endswith("</agent_system_prompt>\n\n")  # and no tool definitions
        note = load_prompt("tool-simulator-system-prompt-note.md")
        assert server._simulator_prompts["sandbox"] == load_prompt("tool-simulator-sandbox.md") + "\n\n" + note + "\n"
        assert server._simulator_prompts["claude_code"] == load_prompt("tool-simulator-claude-code.md")

        sent = script(server, [chat(json.dumps(VERDICT))])
        await _verify(server, CLAUDE_CODE_ROW, tools, [_final("Done.")], metadata=SANDBOX_METADATA)
        system, user = (m["content"] for m in sent[0]["json"]["messages"])
        assert system == build_system_prompt(
            tools,
            template="judge-system-sandbox.md",
            system_prompt_note=load_prompt("judge-system-prompt-note.md"),
        )
        assert user.startswith("<agent_system_prompt>\nYou are a coding agent.\n<env>")
        assert "Linux sandbox rooted at /home/user" in system  # still the sandbox frame

    async def test_forced_off_for_harness_rows(self) -> None:
        server = make_server(show_agent_system_prompt=False)
        _, state = await _seed(server, CLAUDE_CODE_ROW, CLAUDE_CODE_TOOLS)
        assert state.simulator_preamble == ""
        sent = script(server, [chat(json.dumps(VERDICT))])
        await _verify(server, CLAUDE_CODE_ROW, CLAUDE_CODE_TOOLS, _policy_turn())
        system, user = (m["content"] for m in sent[0]["json"]["messages"])
        assert user.startswith("<conversation>\n") and "agent_system_prompt" not in system

    async def test_the_default_note_states_grounding_only(self) -> None:
        server = make_server()
        sent = script(server, [chat(json.dumps(VERDICT))])
        await _verify(server, CLAUDE_CODE_ROW, CLAUDE_CODE_TOOLS, _policy_turn())
        system = sent[0]["json"]["messages"][0]["content"]
        assert load_prompt("judge-system-prompt-note.md").strip() in system
        assert "does not make a request any less questionable" not in system
        assert "the user's later instruction wins" not in system

    async def test_the_note_file_is_configurable(self) -> None:
        server = make_server(judge_system_prompt_note_file="tool-simulator-system-prompt-note.md")
        sent = script(server, [chat(json.dumps(VERDICT))])
        await _verify(server, CLAUDE_CODE_ROW, CLAUDE_CODE_TOOLS, _policy_turn())
        system = sent[0]["json"]["messages"][0]["content"]
        assert load_prompt("tool-simulator-system-prompt-note.md").strip() in system
        assert "does not make a request any less questionable" not in system

    def test_unset_leaves_every_profile_its_default(self) -> None:
        server = make_server()
        assert server._simulator_prompts["sandbox"] == load_prompt("tool-simulator-sandbox.md")
        assert all(server._shows_system_prompt(p) is p.show_system_prompt for p in HARNESSES.values())


class TestGenericVerify:
    async def test_interactive_row_is_judged_in_the_generic_frame(self) -> None:
        server = make_server()
        _, state = await _seed(server, CLAUDE_CODE_ROW, CLAUDE_CODE_TOOLS)
        state.answered_by.update({"llm": 1})
        sent = script(server, [chat(json.dumps(VERDICT))])
        out = await _verify(server, CLAUDE_CODE_ROW, CLAUDE_CODE_TOOLS, _policy_turn())

        assert out.reward == 1.0 and out.harness == "generic" and out.shell_failures is None
        assert out.simulated_calls == {"llm": 1} and out.categories == ["code_debugging"]
        assert out.verifier_metadata["harness"] == "goose"  # the requested harness stays on record
        system, user = (m["content"] for m in sent[0]["json"]["messages"])
        assert system == build_system_prompt(
            CLAUDE_CODE_TOOLS,
            template="judge-system-generic.md",
            system_prompt_note=load_prompt("judge-system-prompt-note.md"),
        )
        assert '- <message from="User"> ... </message>' in system and SUBAGENT_NOTE not in system
        assert "agent harness" in system and "Linux sandbox" not in system and "$" + "requester" not in system
        assert "Check the agent's claims against the evidence." in system  # the same rubrics as the sandbox harness
        assert '<message from="User">\nWhy does test_parse fail?' in user and "/Users/sam/proj" in user
        assert "PRIOR_TURN_REASONING" not in user and "POLICY_REASONING" not in user
        assert "Blank lines now parse as 0." in user.split("<step_under_evaluation>")[1]
        env = "You are a coding agent.\n<env>\nWorking directory: /Users/sam/proj\nPlatform: darwin\n</env>"
        assert user.startswith(f"<agent_system_prompt>\n{env}\n</agent_system_prompt>\n\n<conversation>\n")
        assert user.count("Working directory: /Users/sam/proj") == 1  # not repeated in the conversation
        assert "between <agent_system_prompt> and" in system and "grounded, not invented" in system

    async def test_subagent_row_is_framed_as_a_parent_agents_request(self) -> None:
        server = make_server()
        sent = script(server, [chat(json.dumps(VERDICT))])
        items = [_final("Failing: test_parse_blank, test_parse_unicode.")]
        out = await _verify(server, HERMES_SUBAGENT_ROW, HERMES_TOOLS, items)

        assert out.reward == 1.0
        system, user = (m["content"] for m in sent[0]["json"]["messages"])
        assert '<message from="Parent agent">\nList the failing tests' in user and 'from="User"' not in user
        assert '- <message from="Parent agent"> ... </message>' in system
        assert system.count(SUBAGENT_NOTE) == 1 and "\n\n" + SUBAGENT_NOTE + "\n\n" in system

    async def test_sandbox_rows_ignore_record_kind(self) -> None:
        server = make_server()
        sent = script(server, [chat(json.dumps(VERDICT))])
        tools = [_tool(n) for n in sorted(KNOWN_TOOLS)]
        out = await _verify(
            server,
            HERMES_SUBAGENT_ROW,
            tools,
            [_final("Done.")],
            metadata={**SANDBOX_METADATA, "record_kind": "subagent"},
        )
        system, user = (m["content"] for m in sent[0]["json"]["messages"])
        assert system == build_system_prompt(tools, template="judge-system-sandbox.md")
        expected = render_conversation(HERMES_SUBAGENT_ROW["input"], [_final("Done.")], JUDGE_RENDER)
        assert user == expected and '<message from="User">' in user
        assert "agent_system_prompt" not in user and "agent_system_prompt" not in system
        assert out.harness == "sandbox" and out.shell_failures == 0

    @pytest.mark.parametrize(
        "arguments", [json.dumps({UNPARSEABLE_ARGS_KEY: "ls -"}), json.dumps({"__malformedToolArguments": "ls -"})]
    )
    async def test_argument_markers_are_malformed_in_a_generic_policy_turn(self, arguments: str) -> None:
        server = make_server()
        items = [function_call("terminal", "p1", arguments), function_output("p1", "error"), _final("Hm.")]
        out = await _verify(server, HERMES_ROW, HERMES_TOOLS, items)
        assert (out.code_gate, out.reward) == ("malformed_arguments", 0.0)
        server.server_client.post.assert_not_called()

    async def test_unknown_tool_is_gated(self) -> None:
        out = await _verify(
            make_server(), HERMES_ROW, HERMES_TOOLS, [function_call("Bash", "p1", {"command": "ls"}), _final("ok")]
        )
        assert out.code_gate == "nonexistent_tool"

    async def test_masking_is_the_same_for_every_profile(self) -> None:
        body = AgenticTurnSimulationVerifyRequest(
            responses_create_params=_rcp(HERMES_ROW, HERMES_TOOLS),
            response=_response([_final("...")], stop_reason="time_cap"),
            verifier_metadata=HERMES_ROW["metadata"],
        )
        out = await make_server().verify(req(), body)
        assert getattr(out, MASK_SAMPLE_FIELD) is True
        assert out.failure_kind == "agentic_turn_simulation:time_cap"
