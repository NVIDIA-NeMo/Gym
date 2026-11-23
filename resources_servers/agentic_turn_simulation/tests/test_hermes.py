# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes on synthetic rows: its file-tool results in its own JSON, its shell failures, and its JSON replies."""

import json
from typing import Any

from resources_servers.agentic_turn_simulation.harnesses.hermes import HERMES_FILES, hermes_shell_failures
from resources_servers.agentic_turn_simulation.tests.support import (
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


HM_PATH = "/home/sam/notes/todo.md"
HM_FILE = "- milk\n- eggs\n"


def _hermes_read(content: str, total: int, size: int, **extra: Any) -> str:
    return json.dumps(
        {
            "content": content,
            "total_lines": total,
            "file_size": size,
            "truncated": False,
            **extra,
            "is_binary": False,
            "is_image": False,
        },
        ensure_ascii=False,
    )


class TestHermesFormat:
    def test_read_padded_up_to_0_15_and_learned_back(self) -> None:
        recorded = _hermes_read("     1|- milk\n     2|- eggs\n     3|", 2, 14)
        ws = workspace_after(HERMES_FILES, "0.8.0", ("read_file", {"path": HM_PATH}, recorded))
        assert ws.files == {HM_PATH: HM_FILE}
        out = json.loads(answer(HERMES_FILES, ws, "read_file", {"path": HM_PATH, "offset": 2}))
        assert out == {
            "content": "     2|- eggs\n     3|",
            "total_lines": 2,
            "file_size": 14,
            "truncated": False,
            "is_binary": False,
            "is_image": False,
        }

    def test_read_unpadded_from_0_17_with_a_page_hint(self) -> None:
        ws = workspace_after(
            HERMES_FILES,
            "0.18.1",
            ("write_file", {"path": HM_PATH, "content": HM_FILE}, '{"bytes_written": 14, "dirs_created": true}'),
        )
        out = answer(HERMES_FILES, ws, "read_file", {"path": HM_PATH, "limit": 1})
        assert out == (
            '{"content": "1|- milk\\n2|", "total_lines": 2, "file_size": 14, "truncated": true, '
            '"hint": "Use offset=2 to continue reading (showing 1-1 of 2 lines)", "is_binary": false, "is_image": false}'
        )

    def test_a_file_without_a_final_line_break_and_non_ascii_text(self) -> None:
        ws = workspace_after(
            HERMES_FILES,
            "0.20.0",
            ("write_file", {"path": HM_PATH, "content": "café\nthé"}, '{"bytes_written": 10, "dirs_created": true}'),
        )
        out = json.loads(answer(HERMES_FILES, ws, "read_file", {"path": HM_PATH}))
        assert (out["content"], out["total_lines"], out["file_size"], out["truncated"]) == (
            "1|café\n2|thé",
            1,
            10,
            False,
        )

    def test_read_needs_the_version_or_an_earlier_read(self) -> None:
        ws = workspace_after(
            HERMES_FILES,
            None,
            ("write_file", {"path": HM_PATH, "content": HM_FILE}, '{"bytes_written": 14, "dirs_created": true}'),
        )
        assert answer(HERMES_FILES, ws, "read_file", {"path": HM_PATH}) is None

    def test_write_up_to_0_11_only(self) -> None:
        old = workspace_after(HERMES_FILES, "0.9.0")
        assert answer(HERMES_FILES, old, "write_file", {"path": "~/n/é.txt", "content": "née"}) == (
            '{"bytes_written": 4, "dirs_created": true}'
        )
        assert answer(HERMES_FILES, old, "write_file", {"path": "/home/sam/.ssh/config", "content": "x"}) is None
        assert (
            answer(
                HERMES_FILES, workspace_after(HERMES_FILES, "0.17.0"), "write_file", {"path": HM_PATH, "content": "x"}
            )
            is None
        )

    def test_patch_of_an_unlinted_file_with_a_unique_match(self) -> None:
        content = "".join(f"line {i}\n" for i in range(1, 11))
        ws = workspace_after(
            HERMES_FILES,
            "0.11.0",
            ("write_file", {"path": HM_PATH, "content": content}, '{"bytes_written": 70, "dirs_created": true}'),
        )
        out = json.loads(
            answer(HERMES_FILES, ws, "patch", {"path": HM_PATH, "old_string": "line 2\n", "new_string": "line two\n"})
        )
        assert out["success"] is True and out["files_modified"] == [HM_PATH]
        assert out["lint"] == {"status": "skipped", "message": "No linter for .md files"}
        assert out["diff"].startswith(
            f"--- a/{HM_PATH}\n+++ b/{HM_PATH}\n@@ -1,5 +1,5 @@\n line 1\n-line 2\n+line two\n"
        )
        assert ws.files[HM_PATH].startswith("line 1\nline two\nline 3\n")

    def test_patch_results_it_cannot_know(self) -> None:
        content = "".join(f"line {i}\n" for i in range(1, 11))
        ws = workspace_after(
            HERMES_FILES,
            "0.11.0",
            ("write_file", {"path": "/srv/app.py", "content": content}, '{"bytes_written": 70, "dirs_created": true}'),
        )
        assert (
            answer(HERMES_FILES, ws, "patch", {"path": "/srv/app.py", "old_string": "line 2", "new_string": "x"})
            is None
        )  # linted
        assert ws.files["/srv/app.py"] == content  # the LLM's answer decides whether the edit happened
        ws2 = workspace_after(
            HERMES_FILES,
            "0.11.0",
            ("write_file", {"path": HM_PATH, "content": content}, '{"bytes_written": 70, "dirs_created": true}'),
        )
        assert (
            answer(HERMES_FILES, ws2, "patch", {"path": HM_PATH, "old_string": "line 10", "new_string": "x"}) is None
        )  # near the end
        assert (
            answer(HERMES_FILES, ws2, "patch", {"path": HM_PATH, "old_string": "line  2", "new_string": "x"}) is None
        )  # fuzzy
        assert json.loads(
            answer(HERMES_FILES, ws2, "patch", {"path": HM_PATH, "old_string": "line", "new_string": "row"})
        ) == {
            "success": False,
            "error": "Found 10 matches for old_string. Provide more context to make it unique, or use replace_all=True.",
        }

    def test_terminal_read_backs_of_a_known_file(self) -> None:
        recorded = _hermes_read("1|- milk\n2|- eggs\n3|", 2, 14)
        ws = workspace_after(HERMES_FILES, "0.20.5", ("read_file", {"path": HM_PATH}, recorded))
        out = answer(HERMES_FILES, ws, "terminal", {"command": f"cat {HM_PATH}", "timeout": 30})
        assert out == '{"output": "- milk\\n- eggs", "exit_code": 0, "error": null}'
        assert json.loads(answer(HERMES_FILES, ws, "terminal", {"command": f"tail -n 1 {HM_PATH}"})) == {
            "output": "- eggs",
            "exit_code": 0,
            "error": None,
        }
        assert answer(HERMES_FILES, ws, "terminal", {"command": f"cat {HM_PATH}", "background": True}) is None
        assert answer(HERMES_FILES, ws, "terminal", {"command": f"wc -l {HM_PATH}"}) is None

    def test_terminal_read_backs_stop_once_command_screening_shows(self) -> None:
        recorded = _hermes_read("1|- milk\n2|- eggs\n3|", 2, 14)
        flagged = json.dumps(
            {"output": "", "exit_code": 0, "error": None, "approval": "Command was flagged and auto-approved."}
        )
        ws = workspace_after(
            HERMES_FILES,
            "0.20.5",
            ("read_file", {"path": HM_PATH}, recorded),
            ("terminal", {"command": "rm -rf /tmp/cache"}, flagged),
        )
        assert answer(HERMES_FILES, ws, "terminal", {"command": f"cat {HM_PATH}"}) is None

    def test_shell_failures_from_the_exit_code(self) -> None:
        items = [
            function_call("terminal", "t1", {"command": "make"}),
            function_output("t1", '{"output": "make: *** No rule", "exit_code": 2, "error": null}'),
            function_call("terminal", "t2", {"command": "ls"}),
            function_output("t2", '{"output": "a", "exit_code": 0, "error": null}'),
            function_call("terminal", "t3", {"command": "x"}),
            function_output("t3", "[Tool output pruned to save context]"),
        ]
        assert hermes_shell_failures(items) == 1


class TestHermesHarness:
    async def test_hermes_replies_are_parsed_as_json(self) -> None:
        server = make_server()
        await seed_row(
            server, harness_row("hermes", "0.8.0", ["terminal", "read_file"], [text_message("user", "Run it.")])
        )
        script(server, [chat('```json\n{"output": "hi", "exit_code": 0, "error": null}\n```')])
        out = await server.simulate_tool_call(req(), sim("terminal", "t1", {"command": "echo hi"}))
        assert out.output == '{"output": "hi", "exit_code": 0, "error": null}'
