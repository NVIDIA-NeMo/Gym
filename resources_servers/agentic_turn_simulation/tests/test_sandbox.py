# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sandbox harness: its workspace model, the answers it computes in code, its JSON replies and its shell
failures."""

import json
from typing import Any

import pytest

from resources_servers.agentic_turn_simulation.harnesses.sandbox import (
    BASH_STREAM_CAP,
    GENERATION_LIMIT,
    HARNESS_ERROR,
    IMAGE_LIMIT_MESSAGE,
    IMAGE_READ,
    KNOWN_TOOLS,
    READ_CAP,
    SANDBOX,
    SPEECH_LIMIT_MESSAGE,
    SandboxWorkspace,
    file_hash,
    fuzzy_replace,
    image_read,
    last_json_object,
    norm_path,
    parse_json_reply,
    sandbox_shell_failures,
)
from resources_servers.agentic_turn_simulation.items import MALFORMED_ARGS_KEY
from resources_servers.agentic_turn_simulation.tests.support import raw_call, raw_output
from resources_servers.agentic_turn_simulation.workspace import SHOWN_FILE_CHARS, SHOWN_FILES


def _call(name: str, call_id: str = "c1", **args: Any) -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args)}


def _out(call_id: str, output: Any) -> dict[str, Any]:
    if not isinstance(output, str):
        output = json.dumps(output)
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _user(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": text}


def _run(call: dict[str, Any], ws: SandboxWorkspace, tool_names: frozenset[str] = KNOWN_TOOLS) -> dict[str, Any]:
    out = ws.answer(call, tool_names)
    assert out is not None
    return json.loads(out)


class TestSandboxSettings:
    def test_sandbox_profile_settings(self) -> None:
        assert (SANDBOX.simulator_prompt, SANDBOX.judge_prompt) == (
            "tool-simulator-sandbox.md",
            "judge-system-sandbox.md",
        )
        assert (SANDBOX.stop_tools, SANDBOX.default_tools) == (("ask_user",), KNOWN_TOOLS)
        ws = SANDBOX.new_workspace({})
        assert isinstance(ws, SandboxWorkspace) and SANDBOX.exact_answer(
            _call("write_file", path="a", content="x"), history=[], workspace=ws, tool_names=KNOWN_TOOLS
        )
        assert SANDBOX.shell_failures is sandbox_shell_failures
        assert SANDBOX.requester({"record_kind": "subagent"}).label == "User"
        assert SANDBOX.malformed_argument_keys == (MALFORMED_ARGS_KEY,)


# --- small helpers ------------------------------------------------------------------------------------------


class TestHelpers:
    def test_file_hash_is_43_char_base64url_and_stable(self) -> None:
        h = file_hash("abc")
        assert len(h) == 43 and "=" not in h and "+" not in h and "/" not in h
        assert h == file_hash("abc") != file_hash("abd")
        assert file_hash(None) == file_hash("")

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("~", "/home/user"),
            ("~/app/x.py", "/home/user/app/x.py"),
            ("app/x.py", "/home/user/app/x.py"),
            ("  app/x.py  ", "/home/user/app/x.py"),
            ("/home/alice/x.py", "/home/alice/x.py"),  # an ordinary path: nothing is rewritten
            ("/home/user/a/../b.py", "/home/user/b.py"),
            ("/etc/hosts", "/etc/hosts"),
            (None, "/home/user"),
        ],
    )
    def test_norm_path(self, raw, expected: str) -> None:
        assert norm_path(raw) == expected


class TestFuzzyReplace:
    def test_exact_match_replaces_first_occurrence_only(self) -> None:
        assert fuzzy_replace("a b a b", "a", "X") == "X b a b"

    def test_whitespace_and_indentation_tolerant(self) -> None:
        text = "def f():\n        return 1\n"
        assert fuzzy_replace(text, "def f():\n    return 1", "def g(): pass") == "def g(): pass\n"

    @pytest.mark.parametrize("old", ["", "   \n "])
    def test_empty_or_blank_old_text_is_no_match(self, old: str) -> None:
        assert fuzzy_replace("some text", old, "new") is None

    def test_missing_text_is_no_match(self) -> None:
        assert fuzzy_replace("alpha beta", "gamma", "x") is None


# --- workspace ----------------------------------------------------------------------------------------------


class TestWorkspaceReplay:
    def test_learns_writes_reads_and_edits(self) -> None:
        ws = SandboxWorkspace()
        ws.replay(
            [
                _user("build it"),
                _call("write_file", "w", path="app.py", content="print(1)\n"),
                _out("w", {"hash": "h", "status": "success"}),
                _call("edit_file", "e", path="/home/user/app.py", old_text="print(1)", new_text="print(2)"),
                _out("e", {"status": "success"}),
                _call("read_file", "r", path="~/notes.md"),
                _out("r", {"content": "hello", "kind": "text", "status": "success"}),
            ]
        )
        assert ws.files == {"/home/user/app.py": "print(2)\n", "/home/user/notes.md": "hello"}

    def test_ignores_failures_and_partial_reads(self) -> None:
        ws = SandboxWorkspace()
        ws.replay(
            [
                _call("write_file", "w", path="a.txt", content="x"),
                _out("w", {"status": "error", "message": "Something went wrong."}),
                _call("read_file", "r1", path="big.txt"),
                _out("r1", {"content": "partial", "kind": "text", "status": "success", "truncated": True}),
                _call("read_file", "r2", path="pic.png"),
                _out("r2", {"content": "...", "kind": "image", "status": "success"}),
                _call("read_file", "r3", path="raw.txt"),
                _out("r3", "not json at all"),
                _out("orphan", {"status": "success", "content": "x"}),  # no matching call
            ]
        )
        assert ws.files == {}

    def test_edit_of_unknown_or_unmatched_file(self) -> None:
        ws = SandboxWorkspace()
        ws.files["/home/user/known.txt"] = "alpha"
        ws.replay(
            [
                _call("edit_file", "e1", path="unknown.txt", old_text="a", new_text="b"),
                _out("e1", {"status": "success"}),
                _call("edit_file", "e2", path="known.txt", old_text="zzz", new_text="b"),
                _out("e2", {"status": "success"}),
            ]
        )
        # The harness said the edit worked but our copy has no such text: we no longer know the contents.
        assert ws.files == {"/home/user/known.txt": None}

    def test_generation_counters_reset_at_each_user_message(self) -> None:
        ws = SandboxWorkspace()
        items = [_user("make art")]
        for i in range(3):
            items += [
                _call("generate_image", f"i{i}", prompt="p", file_path=f"img{i}.png"),
                _out(f"i{i}", {"status": "success"}),
            ]
        items += [_call("generate_speech", "s", text="hi", file_path="a.mp3"), _out("s", {"status": "success"})]
        ws.replay(items)
        assert (ws.images, ws.speech) == (3, 1)
        assert ws.files["/home/user/img0.png"] is None and ws.files["/home/user/a.mp3"] is None
        ws.replay([_user("again")])
        assert (ws.images, ws.speech) == (0, 0)

    def test_message_without_type_counts_as_user_message(self) -> None:
        ws = SandboxWorkspace()
        ws.images = 4
        ws.replay([{"role": "user", "content": "hi"}])
        assert ws.images == 0

    def test_generation_without_file_path_only_counts(self) -> None:
        ws = SandboxWorkspace()
        ws.observe(_call("generate_image", prompt="p"), {"status": "completed"})
        assert ws.images == 1 and ws.files == {}

    def test_bash_that_mutates_a_known_file_forgets_it(self) -> None:
        ws = SandboxWorkspace()
        ws.files.update({"/home/user/app.py": "x", "/home/user/other.py": "y"})
        ws.replay([_call("bash", "b", command="sed -i 's/x/z/' /home/user/app.py"), _out("b", {})])
        assert ws.files == {"/home/user/app.py": None, "/home/user/other.py": "y"}

    def test_observe_accepts_dict_output_and_none(self) -> None:
        ws = SandboxWorkspace()
        ws.observe(_call("write_file", path="d.txt", content="c"), {"status": "success"})
        ws.observe(_call("write_file", path="e.txt", content="c"), None)
        assert ws.files == {"/home/user/d.txt": "c"}


class TestForgetMutated:
    @pytest.mark.parametrize(
        "cmd",
        [
            "echo hi > app.py",
            "rm app.py",
            "mv /home/user/app.py /tmp/",
            "cp new.py app.py",
            "git checkout app.py",
            'python3 -c \'open("app.py","w")\'',
            "perl -pi -e 's/a/b/' app.py",
            "cat x | tee app.py",
            "sed -E -i 's/a/b/' app.py",
            "sed --in-place 's/a/b/' app.py",
            "curl -fsSLo app.py https://example.com/x",
            "dos2unix app.py",
        ],
    )
    def test_mutating_commands_naming_the_file(self, cmd: str) -> None:
        ws = SandboxWorkspace()
        ws.files["/home/user/app.py"] = "x"
        ws.forget_mutated(cmd)
        assert ws.files["/home/user/app.py"] is None

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat app.py",
            "sed -n 2p app.py",
            "sort app.py",
            "ls -la",
            "rm other.txt",
            "del other.txt",
            "npm test",
            None,
            42,
        ],
    )
    def test_reads_or_other_files_keep_contents(self, cmd: Any) -> None:
        ws = SandboxWorkspace()
        ws.files["/home/user/app.py"] = "x"
        ws.forget_mutated(cmd)
        assert ws.files["/home/user/app.py"] == "x"

    @pytest.mark.parametrize(
        "cmd",
        [
            "npx prettier --write src",
            "git checkout .",
            "cd ~/site && npx eslint --fix .",
            "black .",
            "git stash",
            "npm install left-pad",
            "npm run format",
        ],
    )
    def test_formatters_and_version_control_forget_every_known_file(self, cmd: str) -> None:
        ws = SandboxWorkspace()
        ws.files.update({"/home/user/src/app.js": "a", "/home/user/README.md": "r", "/home/user/out.png": None})
        ws.replay([_call("bash", "b", command=cmd), _out("b", {"exit_code": 0, "status": "completed"})])
        assert ws.files == {"/home/user/src/app.js": None, "/home/user/README.md": None, "/home/user/out.png": None}
        assert ws.answer(_call("read_file", path="src/app.js"), KNOWN_TOOLS) is None
        assert ws.answer(_call("bash", command="cat src/app.js"), KNOWN_TOOLS) is None

    def test_windows_mutating_commands_forget_the_files_they_name(self) -> None:
        ws = SandboxWorkspace()
        ws.files.update({"/home/user/app.py": "x", "/home/user/other.py": "y"})
        ws.forget_mutated("Remove-Item app.py")
        assert ws.files == {"/home/user/app.py": None, "/home/user/other.py": "y"}


class TestRelevant:
    def test_full_path_and_basename_matches(self) -> None:
        ws = SandboxWorkspace()
        ws.files.update(
            {"/home/user/src/main.py": "MAIN /home/user", "/home/user/cfg.yaml": "CFG", "/home/user/a.c": "A"}
        )
        out = ws.shown_files(_call("bash", command="python /home/user/src/main.py && cat cfg.yaml a.c"))
        assert '<file path="/home/user/src/main.py">\nMAIN /home/user\n</file>' in out
        assert '<file path="/home/user/cfg.yaml">' in out
        assert out.count("<file path=") == 2  # "a.c" is too short a name to match on its own

    def test_basename_must_be_a_whole_name(self) -> None:
        ws = SandboxWorkspace()
        ws.files["/home/user/main.py"] = "M"
        assert ws.shown_files(_call("bash", command="cat main.pyc")) == ""
        assert ws.shown_files(_call("bash", command="cat my_main.py")) == ""
        assert "M" in ws.shown_files(_call("bash", command="cat ./main.py"))

    def test_unknown_contents_are_skipped(self) -> None:
        ws = SandboxWorkspace()
        ws.files["/home/user/img.png"] = None
        assert ws.shown_files(_call("read_file", path="img.png")) == ""

    def test_node_commands_pull_in_package_json(self) -> None:
        ws = SandboxWorkspace()
        ws.files.update({"/home/user/app/package.json": '{"name": "x"}', "/home/user/app/index.js": "js"})
        assert "package.json" in ws.shown_files(_call("bash", command="cd app && npm run build"))
        assert ws.shown_files(_call("bash", command="cd app && make")) == ""
        assert ws.shown_files(_call("read_file", path="npm.txt")) == ""  # only bash commands

    def test_limits_files_and_characters(self) -> None:
        ws = SandboxWorkspace()
        for i in range(SHOWN_FILES + 2):
            ws.files[f"/home/user/file{i}.txt"] = "z" * (SHOWN_FILE_CHARS + 40)
        cmd = " ".join(f"cat file{i}.txt;" for i in range(SHOWN_FILES + 2))
        out = ws.shown_files(_call("bash", command=cmd))
        assert out.count("<file path=") == SHOWN_FILES
        assert "z" * SHOWN_FILE_CHARS + "\n[... 40 more characters ...]\n</file>" in out


# --- answers in code ----------------------------------------------------------------------------------------


class TestImageRead:
    def test_image_mentioned_in_history_is_returned_exactly(self) -> None:
        history = [_user("here is /home/user/uploads/Chart.PNG")]
        out = image_read(_call("read_file", path="/home/user/uploads/Chart.PNG"), history)
        assert json.loads(out) == IMAGE_READ

    def test_image_never_mentioned_goes_to_the_llm(self) -> None:
        assert image_read(_call("read_file", path="ghost.png"), [_user("hello")]) is None

    def test_the_call_itself_does_not_count_as_a_mention(self) -> None:
        call = _call("read_file", path="ghost.png")
        assert image_read(call, [_user("hello"), call]) is None

    @pytest.mark.parametrize(
        "call",
        [
            _call("read_file", path="notes.txt"),
            _call("bash", command="cat x.png"),
            {"type": "function_call", "name": "read_file", "arguments": "{bad"},
            _call("read_file", path=5),
        ],
    )
    def test_not_an_image_read(self, call: dict[str, Any]) -> None:
        assert image_read(call, [_user("x.png notes.txt 5")]) is None


class TestDeterministic:
    @pytest.mark.parametrize(
        "call",
        [
            _call("teleport", target="mars"),
            {"type": "function_call", "name": "bash", "arguments": "{not json"},
            {"type": "function_call", "name": "bash", "arguments": '["ls"]'},
        ],
    )
    def test_unknown_tool_or_bad_arguments_get_the_harness_error(self, call: dict[str, Any]) -> None:
        assert _run(call, SandboxWorkspace()) == HARNESS_ERROR

    def test_known_tools_can_be_restricted(self) -> None:
        call = _call("write_file", path="a", content="x")
        assert _run(call, SandboxWorkspace(), frozenset({"bash"})) == HARNESS_ERROR

    @pytest.mark.parametrize(
        "call",
        [
            _call("bash", command="ls"),
            _call("web_search", query="q"),
            _call("fetch_page", url="https://x"),
            _call("image_search", query="q"),
            _call("ask_user", question="?"),
            _call("read_file", path="unknown.txt"),
            _call("edit_file", path="unknown.txt", old_text="a", new_text="b"),
            _call("write_file", content="no path"),
        ],
    )
    def test_unknowable_results_go_to_the_llm(self, call: dict[str, Any]) -> None:
        assert SandboxWorkspace().answer(call, KNOWN_TOOLS) is None

    def test_write_then_read_then_edit(self) -> None:
        ws = SandboxWorkspace()
        assert _run(_call("write_file", path="~/a.txt", content="one\ntwo\n"), ws) == {
            "hash": file_hash("one\ntwo\n"),
            "status": "success",
        }
        assert ws.files["/home/user/a.txt"] == "one\ntwo\n"
        assert _run(_call("read_file", path="/home/user/a.txt"), ws) == {
            "content": "one\ntwo\n",
            "kind": "text",
            "lines": 2,
            "size": 8,
            "status": "success",
        }
        assert _run(_call("edit_file", path="a.txt", old_text="two", new_text="2"), ws) == {
            "hash": file_hash("one\n2\n"),
            "message": "Edited /home/user/a.txt.",
            "status": "success",
        }
        assert ws.files["/home/user/a.txt"] == "one\n2\n"

    @pytest.mark.parametrize("content, lines, size", [("", 0, 0), ("a\nb", 2, 3), ("é\n", 1, 3)])
    def test_read_line_count_and_byte_size(self, content: str, lines: int, size: int) -> None:
        ws = SandboxWorkspace()
        ws.files["/home/user/f"] = content
        out = _run(_call("read_file", path="f"), ws)
        assert (out["lines"], out["size"]) == (lines, size)
        assert "truncated" not in out

    def test_read_is_capped(self) -> None:
        ws = SandboxWorkspace()
        ws.files["/home/user/big"] = "x" * (READ_CAP + 5)
        out = _run(_call("read_file", path="big"), ws)
        assert len(out["content"]) == READ_CAP and out["truncated"] is True and out["size"] == READ_CAP + 5

    def test_edit_without_match_is_context_not_found(self) -> None:
        ws = SandboxWorkspace()
        ws.files["/home/user/a"] = "alpha"
        out = _run(_call("edit_file", path="a", old_text="beta", new_text="x"), ws)
        assert out == {"message": "Context not found. Read the file to verify the text exists.", "status": "error"}
        assert ws.files["/home/user/a"] == "alpha"

    def test_present_file_echoes_the_path(self) -> None:
        assert _run(_call("present_file", path="out/report.pdf"), SandboxWorkspace()) == {
            "path": "out/report.pdf",
            "status": "success",
        }

    def test_generate_image_and_its_limit(self) -> None:
        ws = SandboxWorkspace()
        first = _run(_call("generate_image", prompt="cat", file_path="img/cat.png"), ws)
        assert first == {"file_path": "img/cat.png", "hash": file_hash("img/cat.png" + "cat"), "status": "success"}
        assert ws.files["/home/user/img/cat.png"] is None
        for _ in range(GENERATION_LIMIT - 1):
            assert _run(_call("generate_image", prompt="p", file_path="x.png"), ws)["status"] == "success"
        assert ws.images == GENERATION_LIMIT
        assert _run(_call("generate_image", prompt="p", file_path="y.png"), ws) == {
            "message": IMAGE_LIMIT_MESSAGE,
            "status": "error",
        }
        assert ws.images == GENERATION_LIMIT and "/home/user/y.png" not in ws.files

    def test_generate_speech_and_its_limit(self) -> None:
        ws = SandboxWorkspace()
        out = _run(_call("generate_speech", text="bonjour", file_path="a.mp3", language="fr"), ws)
        assert out["status"] == "success" and out["file_path"] == "a.mp3"
        assert out["hash"] == file_hash("a.mp3bonjour")
        assert out["attribution"]["language"] == "fr" and out["attribution"]["provider"] == "google"
        assert _run(_call("generate_speech", text="hi", file_path="b.mp3"), ws)["attribution"]["language"] == "en"
        ws.speech = GENERATION_LIMIT
        assert _run(_call("generate_speech", text="hi", file_path="c.mp3"), ws) == {
            "message": SPEECH_LIMIT_MESSAGE,
            "status": "error",
        }

    @pytest.mark.parametrize(
        "call",
        [
            _call("write_file", path="data.json", content={"k": 1}),
            _call("generate_image", prompt="p", file_path=7),
            _call("generate_image", prompt=None, file_path="a.png"),
            _call("generate_speech", text=None, file_path="a.mp3"),
        ],
        ids=["write-object-content", "image-numeric-path", "image-null-prompt", "speech-null-text"],
    )
    def test_wrongly_typed_arguments_do_not_crash(self, call: dict[str, Any]) -> None:
        out = SandboxWorkspace().answer(call, KNOWN_TOOLS)
        assert out is None or isinstance(json.loads(out), dict)


# --- replies, shell failures and code-gate excuses ----------------------------------------------------------


class TestReplies:
    def test_content_is_parsed_and_its_paths_kept(self) -> None:
        msg = {"content": 'ok {"status": "completed", "stdout": "/home/user/a"}'}
        assert SANDBOX.parse_reply(msg, None) == '{"status": "completed", "stdout": "/home/user/a"}'


class TestLastJsonObject:
    def test_plain_and_fenced(self) -> None:
        assert last_json_object('{"status": "ok"}') == {"status": "ok"}
        assert last_json_object('```json\n{"a": 1}\n```') == {"a": 1}
        assert last_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_inline_thinking_then_the_answer(self) -> None:
        text = 'First {"status": "draft"} then I decide: {"status": "completed", "exit_code": 0}'
        assert last_json_object(text) == {"status": "completed", "exit_code": 0}

    def test_only_tool_shaped_objects_count_after_prose(self) -> None:
        assert last_json_object('thinking {"note": 1} and {"results": []} then {"other": 2}') == {"results": []}
        assert last_json_object('thinking {"note": 1}') is None

    @pytest.mark.parametrize("text", [None, "", "no json here", "[1, 2]", "{broken"])
    def test_nothing_usable(self, text) -> None:
        assert last_json_object(text) is None

    def test_nested_object_does_not_replace_the_top_level_one(self) -> None:
        text = 'Let me answer. {"status": "success", "meta": {"content": "inner"}}'
        assert last_json_object(text) == {"status": "success", "meta": {"content": "inner"}}


class TestParseJsonReply:
    def test_keeps_non_ascii(self) -> None:
        assert parse_json_reply({"content": '{"content": "日本"}'}) == '{"content": "日本"}'

    @pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
    def test_empty_content_falls_back_to_the_reasoning_channel(self, key: str) -> None:
        msg = {"content": "  ", key: 'thinking... {"status": "success", "content": "x"}'}
        assert json.loads(parse_json_reply(msg)) == {"status": "success", "content": "x"}

    def test_no_fallback_when_content_has_text(self) -> None:
        msg = {"content": "I cannot do that", "reasoning_content": '{"status": "success"}'}
        assert parse_json_reply(msg) is None

    def test_nothing_anywhere(self) -> None:
        assert parse_json_reply({"content": None}) is None
        assert parse_json_reply({}) is None

    def test_inline_think_is_not_the_answer(self) -> None:
        # A draft object in the reasoning must not win over the answer after </think>.
        msg = {
            "content": '<think>draft {"status": "error", "content": "x"}</think>\n{"status": "completed", "stdout": "ok"}'
        }
        assert json.loads(parse_json_reply(msg)) == {"status": "completed", "stdout": "ok"}

    def test_inline_think_with_an_empty_answer_falls_back_to_it(self) -> None:
        msg = {"content": '<think>the answer is {"status": "success", "content": "x"}</think>  '}
        assert json.loads(parse_json_reply(msg)) == {"status": "success", "content": "x"}

    def test_inline_think_with_a_non_json_answer_is_unusable(self) -> None:
        assert parse_json_reply({"content": '<think>{"status": "success"}</think>I refuse'}) is None

    @pytest.mark.parametrize(
        "content",
        [
            '<think>Let me draft: {"status": "success", "stdout": "a.py", "exit_code": 0} then check',
            '{"status": "success", "results": [{"title": "a", "content": "b"}, {"title": "c", "con',
        ],
        ids=["draft-in-reasoning", "inner-fragment"],
    )
    def test_a_cut_off_reply_is_unusable(self, content: str) -> None:
        # Either would otherwise parse: a draft from the reasoning, or an inner object as the whole result.
        assert parse_json_reply({"content": content}) is not None
        assert parse_json_reply({"content": content}, "length") is None
        assert SANDBOX.parse_reply({"content": content}, "length") is None


class TestBashFailures:
    def test_counts_timeouts_and_non_zero_exits_of_bash_only(self) -> None:
        items = [
            raw_call("bash", call_id="ok"),
            raw_output("ok", json.dumps({"status": "completed", "exit_code": 0})),
            raw_call("bash", call_id="fail"),
            raw_output("fail", json.dumps({"status": "completed", "exit_code": 2})),
            raw_call("bash", call_id="slow"),
            raw_output("slow", json.dumps({"status": "timeout", "exit_code": None})),
            raw_call("bash", call_id="null"),
            raw_output("null", json.dumps({"status": "completed", "exit_code": None})),
            raw_call("bash", call_id="text"),
            raw_output("text", "Something went wrong"),
            raw_call("bash", call_id="list"),
            raw_output("list", "[1]"),
            raw_call("bash", call_id="none"),
            raw_output("none", None),
            raw_call("read_file", call_id="rf"),
            raw_output("rf", json.dumps({"status": "error", "exit_code": 1})),
            raw_output("orphan", json.dumps({"status": "timeout"})),
        ]
        assert sandbox_shell_failures(items) == 2

    def test_no_items(self) -> None:
        assert sandbox_shell_failures([]) == 0


class TestBashReadBacks:
    def _ws(self) -> SandboxWorkspace:
        ws = SandboxWorkspace()
        ws.replay(
            [
                _call("write_file", "w", path="/home/user/site/notes.md", content="héllo\nworld\n"),
                _out("w", {"hash": "x", "status": "success"}),
            ]
        )
        return ws

    @pytest.mark.parametrize(
        "args, stdout",
        [
            ({"command": "cat /home/user/site/notes.md"}, "héllo\nworld\n"),
            ({"command": "cat site/notes.md"}, "héllo\nworld\n"),
            ({"command": "head -n 1 ~/site/notes.md", "timeout": 30}, "héllo\n"),
            ({"command": "tail -n 1 notes.md", "cwd": "/home/user/site"}, "world\n"),
            ({"command": "sed -n '2p' notes.md", "cwd": "~/site"}, "world\n"),
            ({"command": "wc -l site/notes.md", "cwd": "/home/user"}, "2 site/notes.md\n"),
            ({"command": "wc -c /home/user/site/notes.md"}, "13 /home/user/site/notes.md\n"),
            ({"command": "wc -l ~/site/notes.md"}, "2 /home/user/site/notes.md\n"),  # bash expands ~
        ],
    )
    def test_a_known_file_is_read_back_in_code(self, args: dict[str, Any], stdout: str) -> None:
        out = self._ws().answer(_call("bash", **args), KNOWN_TOOLS)
        result = json.loads(out)
        assert list(result) == [
            "duration_ms",
            "exit_code",
            "status",
            "stderr",
            "stderr_truncated",
            "stdout",
            "stdout_truncated",
        ]
        assert 75 <= result.pop("duration_ms") < 100
        assert result == {
            "exit_code": 0,
            "status": "completed",
            "stderr": "",
            "stderr_truncated": False,
            "stdout": stdout,
            "stdout_truncated": False,
        }
        assert "\\u" not in out  # not ASCII-escaped
        assert out == self._ws().answer(_call("bash", **args), KNOWN_TOOLS)  # stable

    @pytest.mark.parametrize(
        "args",
        [
            {"command": "cat notes.md"},  # relative to the home directory: another file
            {"command": "cat notes.md", "cwd": "/tmp"},
            {"command": "cat notes.md", "cwd": ""},
            {"command": "cat ~bob/site/notes.md"},
            {"command": "cat site/notes.md | head -n 1"},
            {"command": "grep world site/notes.md"},
            {"command": "cat site/notes.md", "run_in_background": True},
            {"command": ["cat", "site/notes.md"]},
        ],
    )
    def test_anything_else_goes_to_the_llm(self, args: dict[str, Any]) -> None:
        assert self._ws().answer(_call("bash", **args), KNOWN_TOOLS) is None

    def test_output_over_the_stream_cap_goes_to_the_llm(self) -> None:
        ws = SandboxWorkspace()
        ws.files["/home/user/big.txt"] = "x" * BASH_STREAM_CAP + "\n"
        assert ws.answer(_call("bash", command="cat big.txt"), KNOWN_TOOLS) is None
        assert (
            json.loads(ws.answer(_call("bash", command="wc -c big.txt"), KNOWN_TOOLS))["stdout"] == "20001 big.txt\n"
        )

    def test_a_file_a_command_changed_is_not_read_back(self) -> None:
        ws = self._ws()
        ws.observe(_call("bash", command="sed -i 's/world/there/' site/notes.md"), json.dumps({"exit_code": 0}))
        assert ws.answer(_call("bash", command="cat site/notes.md"), KNOWN_TOOLS) is None

    def test_bash_outside_the_rows_tools_gets_the_harness_error(self) -> None:
        call = _call("bash", command="cat site/notes.md")
        assert json.loads(self._ws().answer(call, frozenset({"read_file"}))) == HARNESS_ERROR
