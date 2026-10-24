# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code on synthetic rows: its file-tool results in its own format, its shell failures, and the harness
end to end (exact answers, what the simulator is shown, the judge's frame). Model servers are scripted."""

import json
from typing import Any

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.agentic_turn_simulation.app import AgenticTurnSimulationVerifyRequest
from resources_servers.agentic_turn_simulation.harnesses.claude_code import (
    CC_SUFFIX,
    CLAUDE_CODE_FILES,
    claude_code_shell_failures,
)
from resources_servers.agentic_turn_simulation.judge import build_system_prompt, load_prompt
from resources_servers.agentic_turn_simulation.render import format_tools
from resources_servers.agentic_turn_simulation.tests.support import (
    VERDICT,
    answer,
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
    workspace_after,
)


CC_FILE = "def total(xs):\n    return sum(xs)\n"
CC_READ = "1\tdef total(xs):\n2\t    return sum(xs)\n3\t"
CC_PATH = "/Users/ana/shop/cart.py"


class TestClaudeCodeFormat:
    def test_read_numbers_lines_with_a_tab_and_learns_them_back(self) -> None:
        ws = workspace_after(CLAUDE_CODE_FILES, "2.1.220", ("Read", {"file_path": CC_PATH}, CC_READ))
        assert ws.files == {CC_PATH: CC_FILE}
        assert (
            answer(CLAUDE_CODE_FILES, ws, "Read", {"file_path": CC_PATH, "offset": 2}) == "2\t    return sum(xs)\n3\t"
        )
        assert answer(CLAUDE_CODE_FILES, ws, "Read", {"file_path": CC_PATH, "offset": 0, "limit": 1}) == (
            "0\tdef total(xs):"
        )
        assert answer(CLAUDE_CODE_FILES, ws, "Read", {"file_path": CC_PATH, "limit": 1}) == "1\tdef total(xs):"

    def test_reads_it_answers_its_own_way_go_to_the_llm(self) -> None:
        ws = workspace_after(CLAUDE_CODE_FILES, "2.1.220", ("Read", {"file_path": CC_PATH}, CC_READ))
        assert answer(CLAUDE_CODE_FILES, ws, "Read", {"file_path": CC_PATH}) is None  # a re-read: "Wasted call"
        assert answer(CLAUDE_CODE_FILES, ws, "Read", {"file_path": CC_PATH, "offset": 9}) is None  # past the end
        assert answer(CLAUDE_CODE_FILES, ws, "Read", {"file_path": CC_PATH, "pages": "1"}) is None
        assert answer(CLAUDE_CODE_FILES, ws, "Read", {"file_path": CC_PATH, "offset": "2"}) is None

    def test_trailing_reminders_are_not_file_content(self) -> None:
        reminder = "\n\n<system-reminder>\nThe todo list is empty.\n</system-reminder>"
        ws = workspace_after(CLAUDE_CODE_FILES, None, ("Read", {"file_path": CC_PATH}, CC_READ + reminder))
        assert ws.files == {CC_PATH: CC_FILE}

    def test_empty_file(self) -> None:
        ws = workspace_after(
            CLAUDE_CODE_FILES,
            "2.1.220",
            ("Write", {"file_path": "/tmp/e.txt", "content": ""}, "File created successfully at: /tmp/e.txt"),
        )
        assert answer(CLAUDE_CODE_FILES, ws, "Read", {"file_path": "/tmp/e.txt"}) == (
            "<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>"
        )

    def test_write_answers_an_overwrite_of_a_known_file_only(self) -> None:
        ws = workspace_after(CLAUDE_CODE_FILES, "2.1.220", ("Read", {"file_path": CC_PATH}, CC_READ))
        out = answer(CLAUDE_CODE_FILES, ws, "Write", {"file_path": CC_PATH, "content": "x = 1\n"})
        assert out == f"The file {CC_PATH} has been updated successfully.{CC_SUFFIX}"
        assert ws.files[CC_PATH] == "x = 1\n"
        # A file it has not read may exist (refused) or not (created): the LLM decides.
        assert answer(CLAUDE_CODE_FILES, ws, "Write", {"file_path": "/Users/ana/new.py", "content": "y"}) is None

    def test_the_confirmation_note_follows_the_sessions_results_then_the_version(self) -> None:
        old = workspace_after(CLAUDE_CODE_FILES, "2.1.122", ("Read", {"file_path": CC_PATH}, CC_READ))
        assert answer(CLAUDE_CODE_FILES, old, "Write", {"file_path": CC_PATH, "content": "z"}) == (
            f"The file {CC_PATH} has been updated successfully."
        )
        shown = workspace_after(
            CLAUDE_CODE_FILES,
            "2.1.220",
            ("Read", {"file_path": CC_PATH}, CC_READ),
            (
                "Write",
                {"file_path": "/Users/ana/b.py", "content": "b"},
                "File created successfully at: /Users/ana/b.py",
            ),
        )
        assert answer(CLAUDE_CODE_FILES, shown, "Write", {"file_path": CC_PATH, "content": "z"}).endswith(
            "successfully."
        )
        unknown = workspace_after(CLAUDE_CODE_FILES, None, ("Read", {"file_path": CC_PATH}, CC_READ))
        assert answer(CLAUDE_CODE_FILES, unknown, "Write", {"file_path": CC_PATH, "content": "z"}) is None

    @pytest.mark.parametrize(
        "args, output, after",
        [
            (
                {"old_string": "sum(xs)", "new_string": "sum(xs or [])"},
                f"The file {CC_PATH} has been updated successfully.{CC_SUFFIX}",
                "def total(xs):\n    return sum(xs or [])\n",
            ),
            (
                {"old_string": "xs", "new_string": "items", "replace_all": True},
                f"The file {CC_PATH} has been updated. All occurrences were successfully replaced.{CC_SUFFIX}",
                "def total(items):\n    return sum(items)\n",
            ),
            (
                {"old_string": "xs", "new_string": "items"},
                "<tool_use_error>Found 2 matches of the string to replace, but replace_all is false. To replace all "
                "occurrences, set replace_all to true. To replace only one occurrence, please provide more context "
                "to uniquely identify the instance.\nString: xs</tool_use_error>",
                CC_FILE,
            ),
            (
                {"old_string": "mean(xs)", "new_string": "sum(xs)"},
                "<tool_use_error>String to replace not found in file.\nString: mean(xs)</tool_use_error>",
                CC_FILE,
            ),
            (
                {"old_string": "xs", "new_string": "xs"},
                "<tool_use_error>No changes to make: old_string and new_string are exactly the same.</tool_use_error>",
                CC_FILE,
            ),
            (
                {"old_string": "    return sum(xs)", "new_string": ""},
                f"The file {CC_PATH} has been updated successfully.{CC_SUFFIX}",
                "def total(xs):\n",
            ),
        ],
        ids=["once", "replace-all", "ambiguous", "not-found", "no-change", "delete-takes-its-line-break"],
    )
    def test_edit(self, args: dict[str, Any], output: str, after: str) -> None:
        ws = workspace_after(CLAUDE_CODE_FILES, "2.1.220", ("Read", {"file_path": CC_PATH}, CC_READ))
        assert answer(CLAUDE_CODE_FILES, ws, "Edit", {"file_path": CC_PATH, **args}) == output
        assert ws.files[CC_PATH] == after

    def test_a_non_ascii_miss_goes_to_the_llm(self) -> None:
        ws = workspace_after(CLAUDE_CODE_FILES, "2.1.220", ("Read", {"file_path": CC_PATH}, CC_READ))
        assert (
            answer(CLAUDE_CODE_FILES, ws, "Edit", {"file_path": CC_PATH, "old_string": "é", "new_string": "e"}) is None
        )

    def test_windows_paths(self) -> None:
        path = "C:\\Users\\Ana\\shop\\cart.py"
        ws = workspace_after(CLAUDE_CODE_FILES, "2.1.226", ("Read", {"file_path": path}, CC_READ))
        assert answer(CLAUDE_CODE_FILES, ws, "Read", {"file_path": "c:/users/ana/shop/cart.py", "limit": 1}) == (
            "1\tdef total(xs):"
        )
        out = answer(
            CLAUDE_CODE_FILES,
            ws,
            "Edit",
            {"file_path": "C:/Users/Ana/shop/cart.py", "old_string": "sum", "new_string": "max"},
        )
        assert out == f"The file C:/Users/Ana/shop/cart.py has been updated successfully.{CC_SUFFIX}"
        # A PowerShell command that names the file makes its content unknown again.
        ws.observe(
            function_call(
                "PowerShell", "p1", {"command": "Set-Content -Path C:\\Users\\Ana\\shop\\cart.py -Value ''"}
            ),
            "",
        )
        assert ws.files["c:\\users\\ana\\shop\\cart.py"] is None

    def test_bash_read_backs_of_a_known_file(self) -> None:
        ws = workspace_after(CLAUDE_CODE_FILES, "2.1.220", ("Read", {"file_path": CC_PATH}, CC_READ))
        bash = lambda **args: answer(CLAUDE_CODE_FILES, ws, "Bash", args)  # noqa: E731
        assert bash(command=f"cat {CC_PATH}") == CC_FILE.rstrip("\n")  # trailing line breaks trimmed
        assert bash(command=f"head -n 1 {CC_PATH}", description="First line", timeout=5000) == "def total(xs):"
        assert bash(command=f"sed -n '2p' {CC_PATH}") == "    return sum(xs)"  # leading spaces kept
        assert bash(command=f"sed -n '9p' {CC_PATH}") == "(Bash completed with no output)"
        assert bash(command=f"head -n 0 {CC_PATH}") is None  # BSD head rejects the count
        assert bash(command=f"cat {CC_PATH}", run_in_background=True) is None
        assert bash(command=f"wc -l {CC_PATH}") is None
        assert bash(command="cat shop/cart.py") is None  # its working directory is not known
        assert answer(CLAUDE_CODE_FILES, ws, "PowerShell", {"command": f"cat {CC_PATH}"}) is None

    def test_bash_output_whose_trimming_is_unsure_goes_to_the_llm(self) -> None:
        ws = workspace_after(
            CLAUDE_CODE_FILES,
            "2.1.220",
            ("Write", {"file_path": "/tmp/a.txt", "content": "\nx\n"}, "File created successfully at: /tmp/a.txt"),
            ("Write", {"file_path": "/tmp/b.txt", "content": "x  \n"}, "File created successfully at: /tmp/b.txt"),
            ("Write", {"file_path": "/tmp/c.txt", "content": "x\r\n"}, "File created successfully at: /tmp/c.txt"),
        )
        for name in "abc":
            assert answer(CLAUDE_CODE_FILES, ws, "Bash", {"command": f"cat /tmp/{name}.txt"}) is None

    def test_shell_failures(self) -> None:
        items = [
            function_call("Bash", "b1", {"command": "pytest"}),
            function_output("b1", "Exit code 1\nFAILED test_cart.py::test_total"),
            function_call("Bash", "b2", {"command": "ls"}),
            function_output("b2", "cart.py"),
            function_call("PowerShell", "b3", {"command": "npm run dev"}),
            function_output(
                "b3",
                "Command did not complete within its 120s timeout and was moved to the background (ID: a1b2c3d4e).",
            ),
            function_call("Bash", "b4", {"command": "true"}),
            function_output("b4", "(Bash completed with no output)"),
        ]
        assert claude_code_shell_failures(items) == 2


CC_ROW = harness_row(
    "claude_code",
    "2.1.220",
    ["Bash", "Read", "Write", "Edit", "AskUserQuestion", "ExitPlanMode"],
    [
        text_message("user", "Why does total() break on None?"),
        function_call("Read", "r1", {"file_path": CC_PATH}),
        function_output("r1", CC_READ),
        text_message("assistant", "sum(None) raises a TypeError."),
        text_message("user", "Fix it."),
    ],
)


class TestClaudeCodeHarness:
    async def test_known_file_ops_are_exact_and_the_rest_goes_to_the_llm_with_the_tool_definitions(self) -> None:
        server = make_server()
        _, state = await seed_row(server, CC_ROW)
        assert state.workspace.files == {CC_PATH: CC_FILE} and state.workspace.version == (2, 1, 220)

        edit = await server.simulate_tool_call(
            req(), sim("Edit", "e1", {"file_path": CC_PATH, "old_string": "sum(xs)", "new_string": "sum(xs or [])"})
        )
        assert (edit.source, edit.output) == ("exact", f"The file {CC_PATH} has been updated successfully.{CC_SUFFIX}")

        sent = script(server, [chat("<think>tests pass</think>\n1 passed in 0.01s\n")])
        bash = await server.simulate_tool_call(req(), sim("Bash", "b1", {"command": "pytest -q"}))
        assert (bash.source, bash.output) == ("llm", "1 passed in 0.01s")
        system, user = (m["content"] for m in sent[0]["json"]["messages"])
        assert system == load_prompt("tool-simulator-claude-code.md")
        tools = [dict(t) for t in CC_ROW[0].tools]
        prompt = "<agent_system_prompt>\nYou are a synthetic claude_code agent.\n</agent_system_prompt>\n\n"
        assert user.startswith(prompt + format_tools(tools) + "\n\n<conversation>\n")
        assert '<tool name="Bash">' in user and "Synthetic description of Bash." in user
        assert user.index("<tools>") < user.index("<conversation>") < user.index("<tool_call_to_simulate>")
        assert "You are a synthetic" not in user[len(prompt) :]  # the system message is not in the conversation
        assert state.answered_by == {"exact": 1, "llm": 1}

    async def test_an_llm_read_teaches_the_workspace(self) -> None:
        server = make_server()
        _, state = await seed_row(server, CC_ROW)
        script(server, [chat("1\tA = 1\n2\t")])
        out = await server.simulate_tool_call(req(), sim("Read", "r2", {"file_path": "/Users/ana/shop/const.py"}))
        assert out.source == "llm" and state.workspace.files["/Users/ana/shop/const.py"] == "A = 1\n"
        edit = await server.simulate_tool_call(
            req(), sim("Edit", "e2", {"file_path": "/Users/ana/shop/const.py", "old_string": "1", "new_string": "2"})
        )
        assert edit.source == "exact"

    async def test_simulator_sees_known_file_contents(self) -> None:
        server = make_server()
        await seed_row(server, CC_ROW)
        sent = script(server, [chat("6")])
        await server.simulate_tool_call(req(), sim("Bash", "b1", {"command": f"python {CC_PATH}"}))
        user = sent[0]["json"]["messages"][1]["content"]
        assert f'<workspace_files>\n<file path="{CC_PATH}">\n{CC_FILE}\n</file>\n</workspace_files>' in user

    async def test_verify_uses_the_harness_frame_and_counts_shell_failures(self) -> None:
        server = make_server()
        sent = script(server, [chat(json.dumps(VERDICT))])
        rcp, metadata = CC_ROW
        items = [
            function_call("Bash", "b1", {"command": "pytest"}),
            function_output("b1", "Exit code 1\n1 failed"),
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Still failing.", "annotations": []}],
            },
        ]
        response = NeMoGymResponse.model_validate(
            {
                "id": "r",
                "created_at": 0.0,
                "model": "p",
                "object": "response",
                "output": items,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "metadata": {"stop_reason": "final_message", "steps": "2"},
            }
        )
        out = await server.verify(
            req(),
            AgenticTurnSimulationVerifyRequest(
                responses_create_params=rcp, response=response, verifier_metadata=metadata
            ),
        )
        assert out.harness == "claude_code" and out.shell_failures == 1 and out.reward == 1.0
        system = sent[0]["json"]["messages"][0]["content"]
        tools = [dict(t) for t in rcp.tools]
        expected = build_system_prompt(
            tools,
            template="judge-system-harness.md",
            environment=load_prompt("judge-environment-claude-code.md"),
            system_prompt_note=load_prompt("judge-system-prompt-note.md"),
        )
        assert system == expected and system.startswith(
            "You are evaluating what an AI agent did at one point in a real conversation.\n\nThe agent is Claude Code"
        )
        assert "$environment" not in system and "Check the agent's claims against the evidence." in system
