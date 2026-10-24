# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The path-agnostic workspace with a toy format: path keys, what the workspace learns and forgets, and when a
call is answered in code. Each harness's real format is tested in its own module's tests."""

import json
from typing import Any, Mapping

import pytest

from resources_servers.agentic_turn_simulation.workspace import (
    FileFormat,
    FileTool,
    FileToolSpec,
    FileToolWorkspace,
    ReadBack,
    path_key,
)


class ToyFormat(FileFormat):
    """READ:<content> / WROTE / EDITED, and errors as ERR:..."""

    dedupes_reads = True

    def read(self, ws, path: str, content: str, args: Mapping[str, Any]) -> str | None:
        return f"READ:{content}"

    def read_back(self, ws, args: Mapping[str, Any], output: str) -> str | None:
        return output[5:] if output.startswith("READ:") else None

    def write(self, ws, path: str, content: str, prior: str, args: Mapping[str, Any]) -> str | None:
        return f"WROTE {path} ({prior})"

    def write_succeeded(self, ws, output: str) -> bool:
        return output.startswith("WROTE")

    def edit(
        self, ws, path: str, before: str, args: Mapping[str, Any], tool: FileTool
    ) -> tuple[str | None, str | None]:
        old, new = args.get(tool.old), args.get(tool.new)
        if before.count(old) != 1:
            return "ERR:not found", None
        return f"EDITED {path}", before.replace(old, new)

    def edit_succeeded(self, ws, output: str) -> bool:
        return output.startswith("EDITED")

    def shell(self, ws, name: str, args: Mapping[str, Any], stdout: str) -> str | None:
        return f"SHELL:{stdout}" if name == "Bash" else None


class HiddenEndFormat(ToyFormat):
    """ToyFormat whose read results do not show the file's final line break."""

    read_shows_final_line_break = False


SPEC = FileToolSpec(
    format=ToyFormat(),
    file_tools={
        "Read": FileTool("read", "file_path", offset="offset", limit="limit"),
        "Write": FileTool("write", "file_path"),
        "Edit": FileTool("edit", "file_path"),
    },
    shell_tools={"Bash": "command"},
    opaque_tools=frozenset({"Agent"}),
    readonly_tools=frozenset({"WebSearch", "Grep"}),
)


TOOLS = frozenset({"Read", "Write", "Edit", "Bash", "Agent", "WebSearch", "Grep", "NotebookEdit"})


def _fc(name: str, args: Any, call_id: str = "c") -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args)}


def _out(call_id: str, output: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _ws(*pairs: tuple[dict[str, Any], str]) -> FileToolWorkspace:
    ws = FileToolWorkspace(SPEC)
    items = []
    for k, (call, output) in enumerate(pairs):
        call = dict(call, call_id=f"h{k}")
        items += [call, _out(f"h{k}", output)]
    ws.replay(items)
    return ws


WIN = "C:\\Users\\Dana\\shop\\cart.py"


class TestPathKey:
    @pytest.mark.parametrize(
        "path, key",
        [
            (WIN, "c:\\users\\dana\\shop\\cart.py"),
            ("c:/users/DANA/shop/./cart.py", "c:\\users\\dana\\shop\\cart.py"),
            ("C:\\Users\\Dana\\shop\\..\\shop\\cart.py", "c:\\users\\dana\\shop\\cart.py"),
            ("\\\\fileserver\\team\\plan.docx", "\\\\fileserver\\team\\plan.docx"),
            ("/Users/dana/shop/./cart.py", "/Users/dana/shop/cart.py"),
            ("/home/dana/Shop/cart.py", "/home/dana/Shop/cart.py"),
            ("~/shop/cart.py", "~/shop/cart.py"),
            ("./shop/cart.py ", "shop/cart.py"),
            ("", None),
            (None, None),
            (["a"], None),
        ],
    )
    def test_keys(self, path: Any, key: str | None) -> None:
        assert path_key(path) == key

    def test_posix_paths_stay_case_sensitive(self) -> None:
        assert path_key("/home/dana/A.txt") != path_key("/home/dana/a.txt")


class TestLearning:
    def test_reads_writes_and_edits_in_the_history(self) -> None:
        ws = _ws(
            (_fc("Read", {"file_path": "/srv/app/a.py"}), "READ:x = 1\n"),
            (_fc("Write", {"file_path": "/srv/app/b.py", "content": "y = 2\n"}), "WROTE /srv/app/b.py"),
            (_fc("Edit", {"file_path": "/srv/app/a.py", "old_string": "1", "new_string": "3"}), "EDITED"),
            (_fc("Edit", {"file_path": "/srv/app/c.py", "old_string": "q", "new_string": "r"}), "EDITED"),
            (_fc("Read", {"file_path": "/srv/app/d.py"}), "ERR:no such file"),
            (_fc("Write", {"file_path": "/srv/app/e.py", "content": "z"}), "ERR:denied"),
        )
        assert ws.files == {"/srv/app/a.py": "x = 3\n", "/srv/app/b.py": "y = 2\n", "/srv/app/c.py": None}

    def test_windows_paths_match_across_spellings(self) -> None:
        ws = _ws((_fc("Write", {"file_path": WIN, "content": "total = 0"}), "WROTE"))
        out = ws.answer(_fc("Read", {"file_path": "c:/users/dana/shop/cart.py"}), TOOLS)
        assert out == "READ:total = 0"

    @pytest.mark.parametrize(
        "call, forgotten",
        [
            (_fc("Bash", {"command": "sed -i 's/1/2/' /srv/app/a.py"}), {"/srv/app/a.py"}),
            (_fc("Bash", {"command": "cat /srv/app/a.py /srv/app/b.py"}), set()),
            (_fc("Bash", {"command": "npx prettier --write ."}), {"/srv/app/a.py", "/srv/app/b.py"}),
            (_fc("Bash", {"command": "npx prettier --write src"}), {"/srv/app/a.py", "/srv/app/b.py"}),
            (_fc("Bash", {"command": "git checkout main"}), {"/srv/app/a.py", "/srv/app/b.py"}),
            (_fc("Bash", {"command": "git checkout ."}), {"/srv/app/a.py", "/srv/app/b.py"}),
            (_fc("Bash", {"command": "Remove-Item b.py"}), {"/srv/app/b.py"}),
            (_fc("Agent", {"prompt": "tidy up the repo"}), {"/srv/app/a.py", "/srv/app/b.py"}),
            (_fc("NotebookEdit", {"notebook_path": "/srv/app/a.py"}), {"/srv/app/a.py"}),
            (_fc("Grep", {"pattern": "x", "path": "/srv/app/a.py"}), set()),
            (_fc("WebSearch", {"query": "b.py"}), set()),
        ],
        ids=[
            "sed-i",
            "cat",
            "formatter",
            "formatter-on-a-directory",
            "git",
            "git-checkout-all",
            "powershell",
            "subagent",
            "unknown-tool",
            "readonly",
            "readonly-2",
        ],
    )
    def test_what_other_tools_make_it_forget(self, call: dict[str, Any], forgotten: set) -> None:
        ws = _ws(
            (_fc("Write", {"file_path": "/srv/app/a.py", "content": "a"}), "WROTE"),
            (_fc("Write", {"file_path": "/srv/app/b.py", "content": "b"}), "WROTE"),
            (call, "ok"),
        )
        assert {k for k, v in ws.files.items() if v is None} == forgotten

    def test_the_row_version_is_parsed(self) -> None:
        assert FileToolWorkspace(SPEC, "2.1.220").at_least(2, 1, 128) is True
        assert FileToolWorkspace(SPEC, "0.8.0").at_least(0, 11) is False
        assert FileToolWorkspace(SPEC).at_least(1) is None

    def test_relevant_shows_known_files_the_call_names(self) -> None:
        ws = _ws((_fc("Write", {"file_path": WIN, "content": "total = 0"}), "WROTE"))
        shown = ws.shown_files(_fc("Bash", {"command": "python C:/Users/Dana/shop/cart.py"}))
        assert shown == f'<file path="{WIN}">\ntotal = 0\n</file>'
        assert ws.shown_files(_fc("Bash", {"command": "ls"})) == ""


class TestExactAnswers:
    def test_known_read_and_edit_are_exact_and_update_the_workspace(self) -> None:
        ws = _ws((_fc("Write", {"file_path": "/srv/a.py", "content": "v = 1"}), "WROTE"))
        assert ws.answer(_fc("Read", {"file_path": "/srv/a.py"}), TOOLS) == "READ:v = 1"
        assert ws.answer(_fc("Edit", {"file_path": "/srv/a.py", "old_string": "1", "new_string": "2"}), TOOLS) == (
            "EDITED /srv/a.py"
        )
        assert ws.answer(_fc("Read", {"file_path": "/srv/a.py"}), TOOLS) == "READ:v = 2"
        failed = ws.answer(_fc("Edit", {"file_path": "/srv/a.py", "old_string": "9", "new_string": "0"}), TOOLS)
        assert failed == "ERR:not found" and ws.files["/srv/a.py"] == "v = 2"

    @pytest.mark.parametrize(
        "call",
        [
            _fc("Read", {"file_path": "/srv/unknown.py"}),
            _fc("Edit", {"file_path": "/srv/unknown.py", "old_string": "a", "new_string": "b"}),
            _fc("Read", {"path": "/srv/a.py"}),
            _fc("Bash", {"command": "ls"}),
            {"type": "function_call", "call_id": "c", "name": "Read", "arguments": "{not json"},
        ],
        ids=["unknown-read", "unknown-edit", "wrong-argument", "not-a-file-tool", "malformed"],
    )
    def test_everything_else_goes_to_the_llm(self, call: dict[str, Any]) -> None:
        ws = _ws((_fc("Read", {"file_path": "/srv/a.py"}), "READ:v = 1"))
        assert ws.answer(call, TOOLS) is None

    def test_a_tool_the_row_does_not_list_goes_to_the_llm(self) -> None:
        ws = _ws((_fc("Read", {"file_path": "/srv/a.py"}), "READ:v = 1"))
        assert ws.answer(_fc("Read", {"file_path": "/srv/a.py"}), frozenset({"Bash"})) is None

    def test_after_a_mutating_command_the_file_is_unknown_again(self) -> None:
        ws = _ws(
            (_fc("Read", {"file_path": "/srv/a.py"}), "READ:v = 1"),
            (_fc("Bash", {"command": "echo 'v = 5' > /srv/a.py"}), ""),
        )
        assert ws.answer(_fc("Read", {"file_path": "/srv/a.py"}), TOOLS) is None

    def test_a_write_is_remembered_and_told_what_the_workspace_knew(self) -> None:
        ws = _ws(
            (_fc("Write", {"file_path": "/srv/k.py", "content": "k"}), "WROTE"),
            (_fc("Write", {"file_path": "/srv/s.py", "content": "s"}), "WROTE"),
            (_fc("Bash", {"command": "sed -i 's/s/t/' /srv/s.py"}), ""),
        )
        outs = [ws.answer(_fc("Write", {"file_path": f"/srv/{n}.py", "content": "new"}), TOOLS) for n in "ksn"]
        assert outs == ["WROTE /srv/k.py (known)", "WROTE /srv/s.py (stale)", "WROTE /srv/n.py (unseen)"]
        assert ws.files == {"/srv/k.py": "new", "/srv/s.py": "new", "/srv/n.py": "new"}

    def test_a_repeated_read_of_an_unchanged_range_goes_to_the_llm(self) -> None:
        ws = _ws((_fc("Read", {"file_path": "/srv/a.py"}), "READ:v = 1"))
        assert ws.answer(_fc("Read", {"file_path": "/srv/a.py"}), TOOLS) is None
        assert ws.answer(_fc("Read", {"file_path": "/srv/a.py", "offset": 1}), TOOLS) == "READ:v = 1"
        assert ws.answer(_fc("Read", {"file_path": "/srv/a.py", "offset": 1}), TOOLS) is None
        ws.answer(_fc("Edit", {"file_path": "/srv/a.py", "old_string": "1", "new_string": "2"}), TOOLS)
        assert ws.answer(_fc("Read", {"file_path": "/srv/a.py"}), TOOLS) == "READ:v = 2"


class TestReadBack:
    CONTENT = "one\ntwo\nthree\n"

    @pytest.mark.parametrize(
        "command, stdout",
        [
            ("cat /srv/a.txt", "one\ntwo\nthree\n"),
            ('cat "/srv/a.txt"', "one\ntwo\nthree\n"),
            ("head /srv/a.txt", "one\ntwo\nthree\n"),
            ("head -n 2 /srv/a.txt", "one\ntwo\n"),
            ("head -n2 /srv/a.txt", "one\ntwo\n"),
            ("head -2 /srv/a.txt", "one\ntwo\n"),
            ("head -n 0 /srv/a.txt", ""),
            ("tail -n 1 /srv/a.txt", "three\n"),
            ("tail -1 /srv/a.txt", "three\n"),
            ("tail -n 5 /srv/a.txt", "one\ntwo\nthree\n"),
            ("tail -n +2 /srv/a.txt", "two\nthree\n"),
            ("tail -n 0 /srv/a.txt", ""),
            ("sed -n '2,3p' /srv/a.txt", "two\nthree\n"),
            ("sed -n 2p /srv/a.txt", "two\n"),
            ("sed -n '3,1p' /srv/a.txt", "three\n"),
            ("sed -n '7,9p' /srv/a.txt", ""),
            ("wc -l /srv/a.txt", "3 /srv/a.txt\n"),
            ("wc -c /srv/a.txt", "14 /srv/a.txt\n"),
            ("  cat   /srv/a.txt ", "one\ntwo\nthree\n"),
        ],
    )
    def test_what_each_command_prints(self, command: str, stdout: str) -> None:
        read_back = ReadBack.parse(command)
        assert read_back is not None and read_back.path == "/srv/a.txt"
        assert read_back.stdout(self.CONTENT) == stdout

    def test_a_missing_final_line_break_and_bytes(self) -> None:
        assert ReadBack.parse("tail -n 1 notes.md").stdout("a\nb") == "b"
        assert ReadBack.parse("wc -l notes.md").stdout("a\nb") == "1 notes.md\n"
        assert ReadBack.parse("wc -c notes.md").stdout("é\n") == "3 notes.md\n"
        assert ReadBack.parse("cat ~/notes.md").path == "~/notes.md"

    @pytest.mark.parametrize(
        "command",
        [
            "cat /srv/a.txt | head -n 2",
            "cat /srv/*.txt",
            "cat /srv/a.txt > /srv/b.txt",
            "cat /srv/a.txt /srv/b.txt",
            "cat -n /srv/a.txt",
            "cat -- /srv/a.txt",
            "cat $HOME/a.txt",
            "cat 'my notes.txt'",
            "cd /srv && cat a.txt",
            "cat a.txt; rm a.txt",
            "LC_ALL=C cat a.txt",
            "sudo cat a.txt",
            "head -n -2 a.txt",
            "head -n +2 a.txt",
            "head -c 10 a.txt",
            "tail -f a.txt",
            "sed -n '0,2p' a.txt",
            "sed -n '/x/p' a.txt",
            "sed -i 's/a/b/' a.txt",
            "sed '2p' a.txt",
            "wc -w a.txt",
            "wc -l a.txt b.txt",
            "cat C:\\work\\a.txt",
            "cat 'unterminated",
            "cat",
            "",
            None,
        ],
    )
    def test_anything_else_is_not_a_read_back(self, command: Any) -> None:
        assert ReadBack.parse(command) is None

    @pytest.mark.parametrize(
        "command, reaches",
        [
            ("cat a", True),
            ("wc -c a", True),
            ("head -n 2 a", False),
            ("head -n 3 a", True),
            ("tail -n 1 a", True),
            ("sed -n '1,2p' a", False),
            ("sed -n '2,9p' a", True),
        ],
    )
    def test_whether_the_output_reaches_the_end(self, command: str, reaches: bool) -> None:
        assert ReadBack.parse(command).reaches_end(self.CONTENT) is reaches
        assert ReadBack.parse(command).reaches_end("") is True


class TestShellReadBacks:
    def _ws(self, spec: FileToolSpec = SPEC) -> FileToolWorkspace:
        ws = FileToolWorkspace(spec)
        ws.replay([_fc("Write", {"file_path": "/srv/a.txt", "content": "one\ntwo\n"}, "h"), _out("h", "WROTE")])
        return ws

    def test_a_read_back_of_a_known_file_is_answered_in_the_harness_format(self) -> None:
        ws = self._ws()
        assert ws.answer(_fc("Bash", {"command": "cat /srv/a.txt"}), TOOLS) == "SHELL:one\ntwo\n"
        assert ws.answer(_fc("Bash", {"command": "sed -n 2p /srv/./a.txt"}), TOOLS) == "SHELL:two\n"
        assert ws.files == {"/srv/a.txt": "one\ntwo\n"}  # nothing changes

    @pytest.mark.parametrize(
        "command",
        [
            "cat srv/a.txt",  # relative: the working directory is unknown
            "cat ~/a.txt",
            "wc -l /srv/a.txt",  # GNU and BSD wc print differently
            "head -n 0 /srv/a.txt",  # BSD head rejects a count of 0
            "head -0 /srv/a.txt",
            "cat /srv/unknown.txt",
            "cat /srv/a.txt | head -n 1",
            "grep one /srv/a.txt",
        ],
    )
    def test_other_commands_go_to_the_llm(self, command: str) -> None:
        assert self._ws().answer(_fc("Bash", {"command": command}), TOOLS) is None

    def test_a_shell_tool_the_row_does_not_list_or_the_format_declines_goes_to_the_llm(self) -> None:
        ws = self._ws()
        assert ws.answer(_fc("Bash", {"command": "cat /srv/a.txt"}), frozenset({"Read"})) is None
        spec = FileToolSpec(format=ToyFormat(), file_tools=SPEC.file_tools, shell_tools={"Shell": "cmd"})
        ws = self._ws(spec)
        assert ws.answer(_fc("Shell", {"cmd": "cat /srv/a.txt"}), frozenset({"Shell"})) is None

    def test_a_sed_range_ending_on_a_last_line_without_a_break_goes_to_the_llm(self) -> None:
        ws = FileToolWorkspace(SPEC)
        ws.replay([_fc("Write", {"file_path": "/srv/b.txt", "content": "one\ntwo"}, "h"), _out("h", "WROTE")])
        assert ws.answer(_fc("Bash", {"command": "sed -n 2p /srv/b.txt"}), TOOLS) is None
        assert ws.answer(_fc("Bash", {"command": "sed -n 1p /srv/b.txt"}), TOOLS) == "SHELL:one\n"
        assert ws.answer(_fc("Bash", {"command": "tail -n 1 /srv/b.txt"}), TOOLS) == "SHELL:two"

    @pytest.mark.parametrize(
        "command",
        [
            "npx prettier --write src",
            "git checkout .",
            "rm /srv/a.txt",
            # in-place editors, however the option is spelled
            "sed -E -i 's/one/1/' /srv/a.txt",
            "sed -ri 's/one/1/' /srv/a.txt",
            "sed --in-place 's/one/1/' /srv/a.txt",
            "sed -e 's/one/1/' -i /srv/a.txt",
            "perl -0pi -e 's/one/1/' /srv/a.txt",
            "awk -i inplace '{print}' /srv/a.txt",
            # downloads and output options
            "curl -o /srv/a.txt https://example.com/a",
            "wget -O /srv/a.txt https://example.com/a",
            "sort -o /srv/a.txt /srv/a.txt",
            "dos2unix /srv/a.txt",
            # package managers rewrite the manifest without naming it
            "npm install left-pad",
            "npm i -D left-pad",
            "pnpm add left-pad",
            "yarn add left-pad",
            "npm pkg set name=x",
            "npm version patch",
            # formatters and fixers as subcommands or package scripts
            "npm run format",
            "npm run lint:fix",
            "go fmt ./...",
            "cargo fix --allow-dirty",
        ],
    )
    def test_after_a_mutating_command_a_read_back_goes_to_the_llm(self, command: str) -> None:
        ws = self._ws()
        ws.observe(_fc("Bash", {"command": command}), "")
        assert ws.answer(_fc("Bash", {"command": "cat /srv/a.txt"}), TOOLS) is None

    @pytest.mark.parametrize(
        "command",
        [
            "sed -n '1,2p' /srv/a.txt",
            "perl -ne 'print' /srv/a.txt",
            "sort /srv/a.txt",
            "grep -i one /srv/a.txt",
            "curl -fsSL https://example.com/a.txt",
            "npm test",
            "npm run build",
            "go test ./...",
            "git log --oneline",
            # installing what the manifest lists leaves the manifest as it is
            "npm install",
            "npm ci",
            "cd /srv && npm install --no-audit 2>&1 | tail -5",
            "pnpm install && npm test",
        ],
    )
    def test_after_a_read_only_command_a_read_back_is_still_answered(self, command: str) -> None:
        ws = self._ws()
        ws.observe(_fc("Bash", {"command": command}), "")
        assert ws.answer(_fc("Bash", {"command": "cat /srv/a.txt"}), TOOLS) == "SHELL:one\ntwo\n"

    def test_a_mutating_command_forgets_a_file_with_a_short_basename(self) -> None:
        ws = FileToolWorkspace(SPEC)
        for k, name in enumerate(["a.c", "b.c"]):
            ws.replay(
                [_fc("Write", {"file_path": f"/srv/app/{name}", "content": "x\n"}, f"h{k}"), _out(f"h{k}", "WROTE")]
            )
        ws.observe(_fc("Bash", {"command": "rm -f a.c"}), "")
        assert ws.files == {"/srv/app/a.c": None, "/srv/app/b.c": "x\n"}
        # Showing the simulator files still needs a longer basename.
        assert ws.shown_files(_fc("Bash", {"command": "cat b.c"})) == ""

    def test_content_from_a_read_that_hides_the_final_line_break(self) -> None:
        spec = FileToolSpec(format=HiddenEndFormat(), file_tools=SPEC.file_tools, shell_tools=SPEC.shell_tools)
        ws = FileToolWorkspace(spec)
        ws.replay([_fc("Read", {"file_path": "/srv/r.txt"}, "h"), _out("h", "READ:one\ntwo")])
        # The file may end with a line break the read did not show: only output short of the end is answered.
        assert ws.answer(_fc("Bash", {"command": "cat /srv/r.txt"}), TOOLS) is None
        assert ws.answer(_fc("Bash", {"command": "head -n 1 /srv/r.txt"}), TOOLS) == "SHELL:one\n"
        # An edit keeps the doubt; a write settles it.
        ws.answer(_fc("Edit", {"file_path": "/srv/r.txt", "old_string": "two", "new_string": "2"}), TOOLS)
        assert ws.answer(_fc("Bash", {"command": "tail -n 1 /srv/r.txt"}), TOOLS) is None
        ws.answer(_fc("Write", {"file_path": "/srv/r.txt", "content": "one\n2"}), TOOLS)
        assert ws.answer(_fc("Bash", {"command": "tail -n 1 /srv/r.txt"}), TOOLS) == "SHELL:2"
