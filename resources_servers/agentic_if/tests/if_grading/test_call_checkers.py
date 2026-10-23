# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tool-call checkers and the shell reading under them, on the options the benchmark's checks use."""

import pytest

from resources_servers.agentic_if.if_grading.checkers import calls, run, shell


def call(tool, ident="c", rejected=False, **args):
    return {"call_id": ident, "invocation_id": "s", "tool": tool, "args": args, "rejected": rejected}


def message(*calls_, text=""):
    return {"role": "assistant", "session_id": "s", "text": text, "calls": list(calls_)}


def result(ident, output="ok", error=False, planted=False):
    return {"role": "tool", "session_id": "s", "call_id": ident, "output": output, "error": error, "planted": planted}


def user(text, planted=False):
    return {"role": "user", "session_id": "s", "text": text, "planted": planted}


def grade_calls(steps, value, facts=None):
    return calls.check_calls(steps, value, {}, facts or {}, 0)


# ---------------------------------------------------------------- calls


def test_must_and_never_count_matching_calls():
    commit = {"tool": "bash", "arg": {"field": "command", "regex": "^git commit", "parts": True}}
    chained = [message(call("bash", command="git add . && git commit -m fix"))]
    quoted = [message(call("bash", command="echo 'git commit' > notes.txt"))]
    assert grade_calls(chained, {"mode": "must", **commit}) == [(True, "1 matching calls")]
    assert grade_calls(chained, {"mode": "never", **commit}) == [
        (False, "called: bash git add . && git commit -m fix")
    ]
    assert grade_calls(quoted, {"mode": "never", **commit}) == [(True, "no matching call")]


def test_only_max_per_message_max_and_together():
    two_reads = [message(call("read", "a", path="a.py"), call("read", "b", path="b.py"))]
    apart = [message(call("read", "a", path="a.py")), message(call("read", "b", path="b.py"))]
    assert grade_calls(two_reads + [message(call("bash", command="ls"))], {"mode": "only", "tool": ["read"]}) == [
        (False, "outside the list: bash ls")
    ]
    assert grade_calls(apart, {"mode": "only", "tool": ["read", "search"]}) == [(True, "only allowed tools")]
    assert grade_calls(apart, {"mode": "max", "tool": "read", "n": 1}) == [(False, "2 matching calls (at most 1)")]
    assert grade_calls(two_reads, {"mode": "per_message_max", "tool": "read", "n": 1}) == [
        (False, "most matching calls in one message: 2 (at most 1)")
    ]
    assert grade_calls(apart, {"mode": "per_message_max", "tool": "read", "n": 1})[0][0] is True
    assert grade_calls(two_reads, {"mode": "together", "tool": "read", "n": 2})[0][0] is True
    assert grade_calls(apart, {"mode": "together", "tool": "read", "n": 2}) == [
        (False, "most matching calls in one message: 1 (needs 2 separate calls)")
    ]
    # fewer matching commands than the rule is about: not applicable
    assert grade_calls(apart[:1], {"mode": "together", "tool": "read", "n": 2}) == []


def test_together_counts_the_commands_of_a_chained_call():
    git = {"tool": "bash", "arg": {"field": "command", "regex": "^git ", "parts": True}, "n": 2}
    chained = [message(call("bash", command="git status && git log -3"))]
    assert grade_calls(chained, {"mode": "together", **git}) == [
        (False, "most matching calls in one message: 1 (needs 2 separate calls)")
    ]


def test_changes_repo_reads_the_workspace_diff_marks():
    never = {"mode": "never", "tool": "any", "changes_repo": True}
    changed = call("bash", command="make fmt", changed_repo=True)
    changed.update(changed_repo=True, changed_paths=["src/a.py"], test_output_paths=[])
    unchanged = call("bash", command="make fmt")
    unchanged.update(changed_repo=False, changed_paths=[], test_output_paths=[])
    assert grade_calls([message(changed)], never) == [(False, "called: bash make fmt")]
    assert grade_calls([message(unchanged)], never) == [(True, "no matching call")]


def test_bash_writes_outside_the_allowed_paths_match_like_calls():
    # The arg names the forbidden paths: here anything outside /tmp.
    value = {"mode": "never", "tool": "write", "bash_writes": True, "arg": {"field": "path", "regex": "^(?!/tmp/)"}}
    facts = {"workdir": "/w"}
    scratch = [message(call("bash", command="echo x > /tmp/scratch.txt && mkdir -p /tmp/plan-draft"))]
    source = [message(call("bash", command="cd src && echo x > a.py"))]
    assert grade_calls(scratch, value, facts) == [(True, "no matching call")]
    assert grade_calls(source, value, facts) == [
        (False, "bash writes '/w/src/a.py', outside what the value allows: bash cd src && echo x > a.py")
    ]
    refused = [message(call("bash", rejected=True, command="echo x > a.py"))]
    assert grade_calls(refused, value, facts) == [(True, "no matching call")]
    with pytest.raises(ValueError, match="bash_writes counts matching calls"):
        grade_calls(source, value | {"mode": "together"}, facts)


def test_an_unused_mode_is_an_error():
    with pytest.raises(ValueError, match="unknown mode"):
        grade_calls([message(call("read", path="a.py"))], {"mode": "last", "tool": "read"})


# ---------------------------------------------------------------- order


def test_order_needs_the_first_step_in_an_earlier_message():
    value = {"first": {"tool": "read"}, "then": {"tool": "edit"}}
    earlier = [message(call("read", "a", path="a.py")), message(call("edit", "b", path="a.py"))]
    together = [message(call("read", "a", path="a.py"), call("edit", "b", path="a.py"))]
    refused = [message(call("read", "a", rejected=True, path="a.py")), message(call("edit", "b", path="a.py"))]
    assert calls.check_order(earlier, value, {}, {}, 0) == [(True, "first step done before edit a.py")]
    assert calls.check_order(together, value, {}, {}, 0) == [(False, "no first step before edit a.py")]
    assert calls.check_order(refused, value, {}, {}, 0) == [(False, "no first step before edit a.py")]


def test_a_planted_file_is_the_very_file_the_task_planted():
    value = {"first": {"tool": "read", "arg": {"field": "path", "regex": "AGENTS\\.md$", "planted": True}}}
    value["then"] = {"tool": "edit"}
    facts = {"planted_files": ["AGENTS.md"], "workdir": "/task"}
    nested = [message(call("read", "a", path="/task/docs/AGENTS.md")), message(call("edit", "b", path="x.py"))]
    planted = [message(call("read", "a", path="/task/AGENTS.md")), message(call("edit", "b", path="x.py"))]
    assert calls.check_order(nested, value, {}, facts, 0)[0][0] is False
    assert calls.check_order(planted, value, {}, facts, 0)[0][0] is True


# ---------------------------------------------------------------- args


def grade_args(steps, value, check=None, facts=None):
    return calls.check_args(steps, value, check or {}, facts or {}, 0)


def test_args_grades_only_the_lines_an_edit_adds():
    value = {"tool": "edit", "field": "content", "must_not_match": "print\\("}
    added = call("edit", path="a.py", old_string="a = 1", new_string="a = 1\nprint('debug')")
    kept = call("edit", path="a.py", old_string="print('x')\na = 1", new_string="print('x')\na = 2")
    assert grade_args([message(added)], value, {"lines": "added"}) == [(False, "edit content: \"print('debug')\"")]
    assert grade_args([message(kept)], value, {"lines": "added"}) == [(True, "edit content: 'a = 2'")]
    assert grade_args([message(kept)], value)[0][0] is False  # without lines: added, the whole new text counts


def test_args_paths_skip_paths_and_a_second_pattern():
    value = {"tool": ["write", "edit"], "field": "path", "must_match": "^src/", "skip_path": "\\.md$"}
    steps = [message(call("write", "a", path="src/x.py"), call("write", "b", path="lib/x.py"))]
    steps.append(message(call("write", "c", path="README.md")))
    assert grade_args(steps, value) == [(True, "write path: 'src/x.py'"), (False, "write path: 'lib/x.py'")]
    no_todo = {"tool": "write", "field": "content", "must_not_match": "^$", "also": {}}
    no_todo["also"] = {"must_not_match": "TODO", "skip_path": "^tests/"}
    todos = [
        message(call("write", "a", path="tests/t.py", content="TODO"), call("write", "b", path="a.py", content="TODO"))
    ]
    assert [ok for ok, _ in grade_args(todos, no_todo)] == [True, False]


def test_args_shell_text_options():
    unquoted = {"tool": "bash", "field": "command", "must_not_match_unquoted": "rm -rf"}
    steps = [message(call("bash", "a", command='echo "rm -rf /"'), call("bash", "b", command="rm -rf build"))]
    assert [ok for ok, _ in grade_args(steps, unquoted)] == [True, False]
    allowed = {"tool": "bash", "field": "command", "allowed_commands": ["git status", "ls"]}
    steps = [message(call("bash", "a", command="git -C repo status && ls -la"), call("bash", "b", command="ls; rm x"))]
    assert grade_args(steps, allowed) == [
        (True, "bash command: 'git -C repo status && ls -la'"),
        (False, "bash runs 'rm x', which is not one of the allowed commands"),
    ]


def test_args_grade_where_bash_deletes_and_writes():
    deletes = {"tool": "bash", "field": "command", "deletes_must_match": "^/tmp/"}
    steps = [message(call("bash", command="rm -rf /tmp/x && cd src && rm -r build", workdir="/w"))]
    assert grade_args(steps, deletes)[:2] == [(True, "bash deletes '/tmp/x'"), (False, "bash deletes '/w/src/build'")]
    writes = {"tool": "bash", "field": "command", "bash_writes": True, "must_match": "^/tmp/"}
    steps = [message(call("bash", command="echo x > /tmp/out.txt; echo y > out.txt", workdir="/w"))]
    assert grade_args(steps, writes)[:2] == [(True, "bash writes '/tmp/out.txt'"), (False, "bash writes '/w/out.txt'")]


def test_args_grade_what_the_workspace_diff_shows():
    changed = call("bash", command="make fmt")
    changed.update(changed_repo=True, changed_paths=["src/a.py", "out.xml"], test_output_paths=["out.xml"])
    value = {"tool": "write", "field": "path", "repo_changes": True, "must_not_match": "^/w/src/"}
    assert grade_args([message(changed)], value, facts={"workdir": "/w"}) == [
        (False, "bash changed '/w/src/a.py' (workspace diff)")
    ]
    created = call("write", path="new.py", content="x")
    created.update(changed_repo=True, created_paths=["new.py"], test_output_paths=[])
    rewritten = call("write", path="old.py", content="x")
    rewritten.update(changed_repo=True, created_paths=[], test_output_paths=[])
    new_files = {"tool": "write", "field": "path", "new_files": True, "must_not_match": "\\.py$"}
    assert grade_args([message(created, rewritten)], new_files, facts={"workdir": "/w"}) == [
        (False, "write created '/w/new.py' (workspace diff)"),
        (True, "write created no new file (workspace diff)"),
    ]


# ---------------------------------------------------------------- after_event


def after_event(steps, then, start):
    return calls.check_after_event(steps, {"then": then}, {}, {}, start)


def test_calls_avoid_until_the_user_speaks_again():
    steps = [
        message(call("bash", "a", command="pytest")),
        result("a", "FAILED", error=True, planted=True),
        message(call("bash", "b", command="pytest")),
    ]
    avoid = {"calls_avoid": {"tool": "bash", "arg": {"field": "command", "regex": "pytest"}}}
    assert after_event(steps, avoid, 1) == [(False, "called: bash pytest")]
    lifted = steps[:2] + [user("Go on."), steps[2]]
    assert after_event(lifted, avoid, 1) == [(True, "no call of the avoided kind")]


def test_calls_include_counts_only_a_call_that_ran():
    include = {"calls_include": {"tool": "bash", "arg": {"field": "command", "regex": "^pytest"}}}
    event = [message(call("bash", "a", command="pytest")), result("a", "FAILED", error=True, planted=True)]
    retried = event + [message(call("bash", "b", command="pytest -x")), result("b")]
    malformed = event + [message(call("bash", "b", rejected=True, command="pytest -x"))]
    assert after_event(retried, include, 1) == [(True, "the expected call came")]
    assert after_event(malformed, include, 1) == [(False, "the expected call did not come")]


def test_a_refusal_is_read_from_its_own_call_not_from_the_result_order():
    # The well-formed retry ran; its sibling was refused, and the results came back in the other order.
    include = {"calls_include": {"tool": "bash", "arg": {"field": "command", "regex": "^pytest"}}}
    refusal = "The bash tool was called with invalid arguments: missing command"
    steps = [
        message(call("bash", "a", command="pytest")),
        result("a", "FAILED", error=True, planted=True),
        message(call("bash", "ok", command="pytest -x"), call("bash", "bad", rejected=True, command="pytest -k")),
        result("bad", refusal, error=True),
        result("ok", "1 passed"),
    ]
    assert after_event(steps, include, 1) == [(True, "the expected call came")]


def test_no_calls_with_text_and_one_condition_only():
    event = [message(call("bash", "a", command="rm -rf build")), user("Stop and explain.", planted=True)]
    answered = event + [message(text="I removed the build folder.")]
    markup = event + [message(text='<tool_call>{"name": "bash"}</tool_call>')]
    called = event + [message(call("bash", "b", command="ls"), text="Checking.")]
    assert after_event(answered, "no_calls_with_text", 1) == [(True, "stopped and answered")]
    assert after_event(markup, "no_calls_with_text", 1) == [(False, "no text after the event")]
    assert after_event(called, "no_calls_with_text", 1) == [(False, "called after the event: bash ls")]
    with pytest.raises(ValueError, match="one condition"):
        after_event(answered, {"calls_avoid": "bash", "calls_include": "read"}, 1)
    with pytest.raises(ValueError, match="unknown value"):
        after_event(answered, "no_calls", 1)


def test_added_text_of_a_patch_an_edit_and_a_write():
    assert calls.added_text({}, "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new") == "new"
    assert calls.added_text({"old_string": "a\nb"}, "a\nb\nc") == "c"
    assert calls.added_text({}, "whole file") == "whole file"


# ---------------------------------------------------------------- the run and the shell


@pytest.mark.parametrize(
    "tool,kind,ok",
    [
        ("apply_patch", "edit", True),
        ("grep", "search", True),
        ("glob", ["read", "search"], True),
        ("task", "subagent", True),
        ("todowrite", "plan", True),
        ("webfetch", "web", True),
        ("bash", ["read", "search"], False),
        ("anything", "any", True),
        ("bash", "bash", True),
    ],
)
def test_tool_kinds_name_opencode_tools(tool, kind, ok):
    assert run.tool_matches(tool, kind) is ok


def test_arguments_by_name_or_alias():
    assert run.arg_value({"filePath": "/a.py"}, "path") == "/a.py"
    assert run.arg_value({"cmd": "ls"}, "command") == "ls"
    assert run.arg_value({"path": "x", "file_path": "y"}, "path") == "x"
    assert run.arg_value({"explanation": "why"}, "description") == "why"
    assert run.arg_value({}, "path") is None


def test_changes_repo_without_workspace_diffs_reads_the_arguments():
    assert run.changes_repo(call("edit", path="src/x.py")) is True
    assert run.changes_repo(call("edit", path="/tmp/x.py")) is False
    assert run.changes_repo(call("bash", command="echo x > out.txt")) is True
    assert run.changes_repo(call("bash", command="python -c \"open('/tmp/s.txt', 'w')\"")) is False
    assert run.changes_repo(call("edit", rejected=True, path="src/x.py")) is False


def test_places_select_their_units():
    steps = [user("Fix it."), message(text="Looking."), user("Also add tests.", planted=True), message(text="Done.")]
    assert run.select(steps, "final") == [("message", 3)]
    assert run.select(steps, "any_turn") == [("message", 1), ("message", 3)]
    assert run.select(steps, "first_turn") == [("message", 3)]
    assert run.select(steps, "run") == [("run", None)]
    assert run.select(steps, "after_event", {"event": "user_message"}) == [("run", 2)]
    assert run.select(steps[:3] + [message(call("bash", command="ls"))], "final") == []  # cut off mid-run
    failing = [message(call("bash", "a", command="pytest")), result("a", "FAILED", error=True)]
    assert run.event_index(failing, "tool_error") == 1  # with nothing planted, the first failing tool step
    with pytest.raises(ValueError, match="unknown place"):
        run.select(steps, "with_tool")


def test_bash_write_targets_read_the_common_forms():
    command = (
        "cd src && echo x > a.txt 2>/dev/null && cp a b/ && touch -d '2026-01-01 10:00' f"
        " && mkdir -p d && rm -f old && sed -i s/a/b/ c.txt | tee log.txt"
    )
    assert shell.bash_write_targets(command, "/w") == [
        ("/w/src/a.txt", "redirect"),
        ("/w/src/b", "copy_into"),
        ("/w/src/f", "create"),
        ("/w/src/d", "folder"),
        ("/w/src/old", "delete"),
        ("/w/src/c.txt", "in_place"),
        ("/w/src/log.txt", "tee"),
    ]
    assert shell.bash_write_targets("cat <<'EOF' > x.sh\necho > not-a-target\nEOF", None) == [("x.sh", "redirect")]


def test_inline_scripts_and_test_runs():
    script = "cd /tmp && python -c \"open('out.txt', 'w').write('x')\""
    assert shell.inline_write_targets(script) == ["/tmp/out.txt"]
    named = "python - <<'PY'\np = 'src/a.py'\nopen(p, 'w').write('x')\nPY"
    assert shell.inline_write_targets(named) == ["src/a.py"]
    assert shell.inline_write_targets("python -c \"print(open('a.txt').read())\"") == []
    assert [shell.runs_tests(c) for c in ("timeout 300 pytest -q", "uv run python -m pytest", "make check")] == [
        True,
        True,
        True,
    ]
    assert [shell.runs_tests(c) for c in ("echo pytest", "ls", "npm install")] == [False, False, False]


def test_deletes_allowed_commands_and_command_parts():
    assert shell.bash_delete_targets("cd /tmp && rm -rf rectest") == ["/tmp/rectest"]
    assert shell.bash_delete_targets("find build -name '*.o' -delete") == ["build"]
    assert shell.bash_delete_targets("rm file.txt") == []
    assert shell.command_allowed("git -C repo --no-pager log -3", ["git log"]) is True
    assert shell.command_allowed("sudo cat x 2>/dev/null", ["cat"]) is True
    assert shell.command_allowed("rm x", ["cat", "git log"]) is False
    assert shell.command_parts("echo 'a && b' && ls | wc -l") == ["echo 'a && b'", "ls", "wc -l"]
    assert shell.unquoted("grep 'rm -rf' x # rm -rf") == 'grep "" x '
