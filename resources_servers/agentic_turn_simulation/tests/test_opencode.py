# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""opencode's and Kilo Code's file-tool results in their shared format, and Kilo Code's shell failures."""

from resources_servers.agentic_turn_simulation.harnesses.opencode import (
    OPENCODE,
    OPENCODE_FILES,
    kilo_code_shell_failures,
)
from resources_servers.agentic_turn_simulation.tests.support import (
    answer,
    function_call,
    function_output,
    workspace_after,
)


OC_FILE_PATH = "/home/lee/app/NOTES.md"


class TestOpencodeFormat:
    def test_read_tags_numbered_lines_and_the_end_notice(self) -> None:
        ws = workspace_after(
            OPENCODE_FILES,
            "1.18.23",
            ("write", {"filePath": OC_FILE_PATH, "content": "# Notes\n\nship\n"}, "Wrote file successfully."),
        )
        assert answer(OPENCODE_FILES, ws, "read", {"filePath": OC_FILE_PATH}) == (
            f"<path>{OC_FILE_PATH}</path>\n<type>file</type>\n<content>\n1: # Notes\n2: \n3: ship\n\n"
            "(End of file - total 3 lines)\n</content>"
        )
        assert answer(OPENCODE_FILES, ws, "read", {"filePath": OC_FILE_PATH, "offset": 2, "limit": 1}) == (
            f"<path>{OC_FILE_PATH}</path>\n<type>file</type>\n<content>\n2: \n\n"
            "(Showing lines 2-2 of 3. Use offset=3 to continue.)\n</content>"
        )

    def test_read_is_learned_back(self) -> None:
        recorded = f"<path>{OC_FILE_PATH}</path>\n<type>file</type>\n<content>\n1: a\n2: b\n\n(End of file - total 2 lines)\n</content>"
        ws = workspace_after(OPENCODE_FILES, "1.18.23", ("read", {"filePath": OC_FILE_PATH}, recorded))
        assert ws.files == {OC_FILE_PATH: "a\nb"}

    def test_reads_whose_echoed_path_or_shape_it_cannot_know_go_to_the_llm(self) -> None:
        win = "C:\\Users\\Lee\\app\\NOTES.md"
        ws = workspace_after(
            OPENCODE_FILES,
            "1.18.23",
            ("write", {"filePath": win, "content": "x\n"}, "Wrote file successfully."),
            ("write", {"filePath": "/home/lee/app/empty.md", "content": ""}, "Wrote file successfully."),
            ("write", {"filePath": "/home/lee/app/long.md", "content": "y" * 2001}, "Wrote file successfully."),
        )
        for path in (win, "/home/lee/app/empty.md", "/home/lee/app/long.md"):
            assert answer(OPENCODE_FILES, ws, "read", {"filePath": path}) is None

    def test_write_and_edit_of_files_no_language_server_checks(self) -> None:
        ws = workspace_after(
            OPENCODE_FILES,
            "1.18.23",
            ("write", {"filePath": OC_FILE_PATH, "content": "a b a"}, "Wrote file successfully."),
        )
        assert (
            answer(OPENCODE_FILES, ws, "write", {"filePath": "/home/lee/app/todo.txt", "content": "z"})
            == "Wrote file successfully."
        )
        assert answer(OPENCODE_FILES, ws, "write", {"filePath": "/home/lee/app/main.ts", "content": "z"}) is None
        edit = {"filePath": OC_FILE_PATH, "oldString": "b", "newString": "c"}
        assert (
            answer(OPENCODE_FILES, ws, "edit", edit) == "Edit applied successfully."
            and ws.files[OC_FILE_PATH] == "a c a"
        )
        assert (
            answer(OPENCODE_FILES, ws, "edit", {**edit, "oldString": "a"}) is None
        )  # two matches: its fuzzy matchers
        assert (
            answer(OPENCODE_FILES, ws, "edit", {**edit, "oldString": "a", "replaceAll": True})
            == "Edit applied successfully."
        )
        assert ws.files[OC_FILE_PATH] == "c c c"
        assert answer(OPENCODE_FILES, ws, "edit", {**edit, "oldString": "c", "newString": "c"}) == (
            "No changes to apply: oldString and newString are identical."
        )

    def test_a_session_with_permission_rules_leaves_writes_to_the_llm(self) -> None:
        refused = "The user has specified a rule which prevents you from using this specific tool call."
        ws = workspace_after(OPENCODE_FILES, "1.18.23", ("write", {"filePath": "/etc/x.md", "content": "x"}, refused))
        assert answer(OPENCODE_FILES, ws, "write", {"filePath": OC_FILE_PATH, "content": "z"}) is None

    def test_bash_read_backs_print_the_output_as_is(self) -> None:
        ws = workspace_after(
            OPENCODE_FILES,
            "1.18.23",
            ("write", {"filePath": OC_FILE_PATH, "content": "# Notes\n\nship\n"}, "Wrote file successfully."),
        )
        bash = lambda tool="bash", **args: answer(OPENCODE_FILES, ws, tool, args)  # noqa: E731
        assert bash(command=f"cat {OC_FILE_PATH}", description="Show notes") == "# Notes\n\nship\n"
        assert bash(command=f"tail -n 1 {OC_FILE_PATH}", workdir="/home/lee") == "ship\n"
        assert bash(command=f"head -n 0 {OC_FILE_PATH}") is None  # no output: its note varies by version
        assert bash(tool="execute_command", command=f"cat {OC_FILE_PATH}") is None  # the legacy extension's shell

    def test_a_read_back_of_a_file_known_from_a_read_stops_short_of_its_end(self) -> None:
        recorded = f"<path>{OC_FILE_PATH}</path>\n<type>file</type>\n<content>\n1: a\n2: b\n\n(End of file - total 2 lines)\n</content>"
        ws = workspace_after(OPENCODE_FILES, "1.18.23", ("read", {"filePath": OC_FILE_PATH}, recorded))
        # The read never shows whether the file ends with a line break.
        assert answer(OPENCODE_FILES, ws, "bash", {"command": f"cat {OC_FILE_PATH}"}) is None
        assert answer(OPENCODE_FILES, ws, "bash", {"command": f"head -n 1 {OC_FILE_PATH}"}) == "a\n"

    def test_kilo_shell_failures_only_where_the_shell_reports_exit_codes(self) -> None:
        ok = "Command executed in terminal  within working directory '/w'. Exit code: 0\nOutput:\nfine"
        bad = "Command executed in terminal  within working directory '/w'. Exit code: 2\nOutput:\nusage"
        items = [
            function_call("execute_command", "x1", {"command": "a"}),
            function_output("x1", ok),
            function_call("execute_command", "x2", {"command": "b"}),
            function_output("x2", bad),
        ]
        assert kilo_code_shell_failures(items, ["execute_command", "read_file"]) == 1
        assert kilo_code_shell_failures([], ["bash", "read"]) is None
        assert OPENCODE.shell_failures is None
