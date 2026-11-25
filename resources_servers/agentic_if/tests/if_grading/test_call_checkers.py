# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tool-call checkers and the shell reading under them, on the options the benchmark's checks use."""

import re

import pytest

from resources_servers.agentic_if.if_grading.checkers import calls, run, shell
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence


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


@pytest.mark.parametrize(
    "command,commits",
    [
        ("grep 'git commit' CONTRIBUTING.md", False),
        ('echo "git commit" > notes.txt', False),
        ("git log --grep 'git commit'", False),
        ("git -c user.name=Agent commit -m fix", True),
        ("git -C /app --no-pager commit -am fix", True),
        ("GIT_AUTHOR_NAME=Agent git commit -m fix", True),
        ('git add -A && git commit -m "fix: a && b"', True),
        ("bash -lc 'git commit -m fix'", True),
    ],
)
def test_a_command_pattern_matches_the_git_invocation_not_quoted_text(command, commits):
    # The example rows' no-commit rule: a quoted mention is no commit, and git's global options are read past.
    never = {"mode": "never", "tool": "bash", "arg": {"field": "command", "regex": r"\bgit\s+commit\b", "parts": True}}
    assert grade_calls([message(call("bash", command=command))], never)[0][0] is not commits


def test_invocations_read_each_command_as_the_shell_runs_it():
    assert shell.invocations("git -c user.name=A commit -m 'a b' 2>/dev/null | tee log") == [
        "git commit -m ",
        "tee log",
    ]
    assert shell.invocations("sudo env X=1 git status") == ["git status"]
    assert shell.invocations("sh -c 'cd src && git commit -m x'") == ["sh -c ", "cd src", "git commit -m x"]
    chained = call("bash", command="git -c color.ui=never status && echo 'git log' && git log -3")
    status_or_log = {"tool": "bash", "arg": {"field": "command", "regex": "^git (status|log)", "parts": True}}
    assert run.match_count(chained, status_or_log) == 2


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


def test_per_message_max_does_not_apply_without_a_matching_call():
    assert grade_calls([message(call("read", path="a.py"))], {"mode": "per_message_max", "tool": "bash", "n": 1}) == []


def diff_marks(c, changed):
    """The call as a run with workspace diffs marks it: `changed` are the files the diff after it shows changed."""
    return c | {"changed_repo": bool(changed), "changed_paths": changed, "created_paths": [], "test_output_paths": []}


def test_an_inline_script_write_is_read_from_the_workspace_diff_when_the_run_has_one():
    value = {"mode": "never", "tool": "write", "bash_writes": True, "arg": {"field": "path", "regex": "^(?!/tmp/)"}}
    facts = {"workdir": "/w"}
    # the script computes its target: only the workspace diff shows where the write landed
    computed = call("bash", command="python -c \"import sys; open(sys.argv[1], 'w').write('x')\" src/a.py")
    assert grade_calls([message(computed)], value, facts) == [(True, "no matching call")]
    assert grade_calls([message(diff_marks(computed, ["src/a.py"]))], value, facts) == [
        (False, "bash writes '/w/src/a.py', outside what the value allows: bash %s" % computed["args"]["command"])
    ]
    # a script that names its files gets what the call changed in the repository, unless all it names is scratch
    named = call("bash", command="python -c \"open('src/a.py', 'w').write('x')\"")
    assert calls.inline_writes(diff_marks(named, ["src/a.py", "src/b.py"])) == ["src/a.py", "src/b.py"]
    scratch = call("bash", command="python -c \"open('/tmp/s.txt', 'w').write('x')\"")
    assert calls.inline_writes(diff_marks(scratch, ["src/a.py"])) == []
    assert calls.inline_writes(diff_marks(call("bash", command="make fmt"), ["src/a.py"])) == []
    assert calls.inline_writes(call("bash")) == calls.shell_write_targets(call("bash")) == []


@pytest.mark.parametrize(
    "command,where",
    [
        # a folder, or a copy that may be one, is allowed when an allowed file can go in it
        ("mkdir -p /tmp/review", None),
        ("cp report.md /tmp/review", None),
        ("cp report.md /tmp/review/", None),
        ("cp -t /tmp/review report.md", None),
        ("cp report.md src/", "/w/src"),
        ("mkdir -p src/new", "/w/src/new"),
        ("echo x > /tmp/review.md", "/tmp/review.md"),
        # a target the command computes does not show where it lands
        ("echo x > $OUT", None),
    ],
)
def test_bash_writes_allow_folders_and_copies_that_can_hold_an_allowed_file(command, where):
    value = {
        "mode": "never",
        "tool": "write",
        "bash_writes": True,
        "arg": {"field": "path", "regex": "^(?!/tmp/review/)"},
    }
    assert calls.shell_write_outside(call("bash", command=command), value, {"workdir": "/w"}) == where


@pytest.mark.parametrize(
    "command,where",
    [
        ("mkdir -p /tmp/plan-draft && cat > /tmp/plan-draft/plan.md <<'EOF'\n# Plan\nEOF", None),
        ("echo x > /tmp/plan-draft/notes.md", "/tmp/plan-draft/notes.md"),
        ("mkdir -p /tmp/other", "/tmp/other"),
    ],
)
def test_bash_writes_allow_the_folder_of_the_one_file_a_value_allows(command, where):
    # the pattern of a plan file is written ^(?!<escaped path>$) when the task is authored
    allowed = {"field": "path", "regex": "^(?!%s$)" % re.escape("/tmp/plan-draft/plan.md")}
    value = {"mode": "never", "tool": ["write", "edit"], "bash_writes": True, "arg": allowed}
    assert calls.shell_write_outside(call("bash", command=command), value, {"workdir": "/w"}) == where


def test_bash_writes_without_a_path_pattern_allow_no_file_at_all():
    value = {"mode": "never", "tool": "write", "bash_writes": True}
    assert calls.shell_write_outside(call("bash", command="echo x > $OUT"), value, {"workdir": "/w"}) == "/w/$OUT"
    assert calls.shell_write_outside(call("bash", command="git status"), value, {"workdir": "/w"}) is None
    for arg in ({"field": "command", "regex": "x"}, {"field": "path"}):
        with pytest.raises(ValueError, match="bash_writes needs an arg"):
            grade_calls([message(call("bash", command="ls"))], value | {"arg": arg})


# The rules' values on file changes by any tool, /tmp included: one change per message, and no change at all.
CHANGE_ALONE = {"mode": "alone", "tool": "any", "writes_file": True}
NO_WRITES = {"mode": "never", "tool": "any", "writes_file": True}


def test_alone_keeps_every_file_change_alone_in_its_message():
    looked_then_edited = [
        message(call("read", "a", path="src/app.py"), call("grep", "b", pattern="TODO")),
        message(call("edit", "c", path="src/app.py", old_string="x = 1", new_string="x = 2")),
        message(call("bash", "d", command="pytest -q"), call("bash", "e", command="git diff --stat")),
    ]
    assert grade_calls(looked_then_edited, CHANGE_ALONE) == [(True, "every matching call is alone in its message")]
    edit_and_test = [
        message(call("read", "a", path="src/app.py")),
        message(call("edit", "b", path="src/app.py", new_string="x = 2"), call("bash", "c", command="pytest -q")),
    ]
    assert grade_calls(edit_and_test, CHANGE_ALONE) == [
        (False, "message 2 holds edit src/app.py beside 1 other call(s)")
    ]
    # a scratch file under /tmp is a file change too
    notes = call("bash", "a", command="cat > /tmp/notes.md <<'EOF'\n# Notes\nEOF")
    assert grade_calls([message(notes, call("read", "b", path="src/app.py"))], CHANGE_ALONE)[0][0] is False
    # a run that changes no file is not graded; a refused write changed nothing
    refused = message(call("write", "a", rejected=True, path="src/new.py"), call("read", "b", path="src/app.py"))
    assert grade_calls(looked_then_edited[:1] + [refused], CHANGE_ALONE) == []
    # without writes_file the spec is the tool alone: every bash call goes alone in its message
    assert grade_calls(edit_and_test, {"mode": "alone", "tool": "bash"}) == [
        (False, "message 2 holds bash pytest -q beside 1 other call(s)")
    ]


@pytest.mark.parametrize(
    "c,writes",
    [
        (call("edit", path="src/app.py", old_string="x = 1", new_string="x = 2"), True),
        (call("write", path="/tmp/plan.md", content="# Plan"), True),
        (
            call("apply_patch", patchText="*** Begin Patch\n*** Update File: src/app.py\n@@\n-a\n+b\n*** End Patch"),
            True,
        ),
        (call("bash", command="cat > /tmp/notes.md <<'EOF'\n# Notes\nEOF"), True),
        (call("bash", command="rm -f /tmp/scratch.py"), True),
        (call("bash", command="mkdir -p /tmp/out && cp report.md /tmp/out/"), True),
        (call("bash", command="python3 -c \"import json; json.dump({}, open('/tmp/cfg.json', 'w'))\""), True),
        (call("bash", command="grep -rn TODO src/ 2>/dev/null | head -20"), False),
        (call("bash", command="pytest -q > /dev/null 2>&1"), False),
        (call("read", path="src/app.py"), False),
        (call("edit", command="view", path="src/app.py"), False),
        # a call the harness refused ran nothing
        (call("write", rejected=True, path="src/new.py", content="x"), False),
        (call("bash", rejected=True, command="echo x > /tmp/out.txt"), False),
    ],
)
def test_writes_file_is_any_file_written_created_or_deleted(c, writes):
    assert run.writes_file(c) is writes
    assert run.call_matches(c, {"tool": "any", "writes_file": True}) is writes
    assert run.call_matches(c, {"tool": "any", "writes_file": False}) is not writes


def test_writes_file_reads_the_workspace_diff_when_the_run_has_one():
    # a formatter names no file, but the diff shows what it rewrote; a test run's own outputs are no change
    fmt = diff_marks(call("bash", "a", command="make fmt"), ["src/app.py"])
    tests = diff_marks(call("bash", "b", command="pytest -q"), [".coverage"]) | {"test_output_paths": [".coverage"]}
    read = call("read", "c", path="src/app.py")
    assert grade_calls([message(fmt)], NO_WRITES) == [(False, "called: bash make fmt")]
    assert grade_calls([message(tests), message(read)], NO_WRITES) == [(True, "no matching call")]
    # without the key every call matches, a read too, as before
    assert grade_calls([message(read)], {"mode": "never", "tool": "any"}) == [(False, "called: read src/app.py")]


# The rules' pattern for a shell write that belongs to the edit and write tools: any file outside /tmp, /var/tmp and
# /dev, except a log or a captured output.
FILE_WRITE = (
    r"^(?!/(?:tmp|var/tmp|dev)(?:/|$))(?!.*(?:\.(?:log|out|err)|(?:^|/)(?:out|err|output|stdout|stderr)\.txt)$)"
)


@pytest.mark.parametrize(
    "command,writes",
    [
        ("cat > src/config.py <<'EOF'\nDEBUG = False\nEOF", True),
        ("sed -i 's/DEBUG = True/DEBUG = False/' settings.py", True),
        ("printf 'retries = 3\\n' >> setup.cfg", True),
        ("echo '{}' | tee config/settings.json > /dev/null", True),
        ("python3 -c \"open('src/version.py', 'w').write('VERSION = 2')\"", True),
        # a log, a captured output and a scratch file are no file work
        ("pytest -q 2>&1 | tee test_run.log", False),
        ("python -m pytest > /tmp/pytest_out.txt 2>&1", False),
        ("make build > output.txt 2> err.txt", False),
        ("cd /tmp && echo 'draft' > notes.md", False),
        # copying, creating an empty file or a folder, and deleting write no content
        ("cp src/app.py src/app.py.bak && touch src/__init__.py && mkdir -p build && rm -f old.py", False),
        # a quoted > is no redirect
        ("grep -n 'a > b' src/app.py", False),
    ],
)
def test_an_arg_spec_on_writes_matches_the_files_a_command_writes_into(command, writes):
    never = {"mode": "never", "tool": "bash", "arg": {"field": "command", "writes": FILE_WRITE}}
    assert grade_calls([message(call("bash", command=command))], never, {"workdir": "/app"})[0][0] is not writes


def test_shell_writes_read_a_relative_target_from_the_call_or_session_folder():
    relative = call("bash", command="echo 'x = 1' > notes.py")
    assert calls.shell_content_writes(relative, {"workdir": "/app"}) == ["/app/notes.py"]
    assert calls.shell_content_writes(relative, {"workdir": "/tmp/session"}) == ["/tmp/session/notes.py"]
    assert calls.shell_content_writes(relative) == ["notes.py"]
    # the call's own working folder wins
    own = call("bash", command="echo 'x = 1' > notes.py", workdir="/tmp")
    assert calls.shell_content_writes(own, {"workdir": "/app"}) == ["/tmp/notes.py"]


def test_shell_writes_by_the_shell_or_by_an_inline_script():
    facts = {"workdir": "/app"}
    heredoc = call(
        "bash", command="python3 - <<'EOF'\nfrom pathlib import Path\nPath('src/app.py').write_text('x')\nEOF"
    )
    redirect = call("bash", command="echo 'x' > src/app.py")
    scratch = call("bash", command="python3 -c \"open('/tmp/scratch.txt', 'w').write('x')\"")
    by_inline = {"tool": "bash", "arg": {"field": "command", "writes": FILE_WRITE, "by": "inline"}}
    by_shell = {"tool": "bash", "arg": {"field": "command", "writes": FILE_WRITE, "by": "shell"}}
    assert [run.call_matches(c, by_inline, facts) for c in (heredoc, redirect, scratch)] == [True, False, False]
    assert [run.call_matches(c, by_shell, facts) for c in (heredoc, redirect, scratch)] == [False, True, False]
    with pytest.raises(ValueError, match="`by` is shell or inline"):
        run.call_matches(redirect, {"tool": "bash", "arg": {"field": "command", "writes": ".", "by": "python"}})


def test_inline_writes_are_read_from_the_workspace_diff_when_the_run_has_one():
    by_inline = {"tool": "bash", "arg": {"field": "command", "writes": FILE_WRITE, "by": "inline"}}
    facts = {"workdir": "/app"}
    # the script computes its target: only the workspace diff shows that it rewrote a source file
    computed = call("bash", command="python3 -c \"import sys; open(sys.argv[1], 'w').write('x')\" src/app.py")
    assert run.call_matches(computed, by_inline, facts) is False
    assert run.call_matches(diff_marks(computed, ["src/app.py"]), by_inline, facts) is True
    # a scratch file the script writes and removes again leaves no change
    removed = call("bash", command="python3 -c \"import os; open('t.txt', 'w').write('x'); os.remove('t.txt')\"")
    assert run.call_matches(removed, by_inline, facts) is True
    assert run.call_matches(diff_marks(removed, []), by_inline, facts) is False


# The rules' pattern for a search in the shell: grep, find, ls and the like as a command, not inside other text.
SHELL_SEARCH = r"(?:^|[;\n(&]|\|\||\$\()\s*(?:sudo\s+)?(?:grep|egrep|fgrep|rg|ag|ack|find|fd|ls|tree|locate)\b"


@pytest.mark.parametrize(
    "command,searches",
    [
        ("ls -la src/", True),
        ('cd src && grep -rn "TODO" .', True),
        # quoted text, a here-document body and a comment are no commands
        ('git commit -m "fix: handle empty input; ls no longer crashes"', False),
        ("cat > NOTES.md <<'EOF'\nfind the flaky test before the release\nEOF", False),
        ("make test  # build first; ls dist/ after", False),
    ],
)
def test_an_unquoted_arg_spec_reads_the_shell_text_outside_quotes(command, searches):
    c = call("bash", command=command)
    unquoted = {"field": "command", "unquoted": True, "regex": SHELL_SEARCH}
    assert run.call_matches(c, {"tool": "bash", "arg": unquoted}) is searches
    # without the key the regex reads the whole text, as before
    assert run.call_matches(c, {"tool": "bash", "arg": {"field": "command", "regex": SHELL_SEARCH}}) is True


@pytest.mark.parametrize(
    "command,pushes",
    [
        ("bash -lc 'git push origin main'", True),  # the shell runs the quoted script
        ("git -C repo push -f", True),
        ('git commit -m "then git push"', False),
    ],
)
def test_unquoted_beside_parts_keeps_the_reading_of_parts(command, pushes):
    parts = {"field": "command", "parts": True, "regex": r"^git\s+push\b"}
    for arg in (parts, parts | {"unquoted": True}):
        assert run.call_matches(call("bash", command=command), {"tool": "bash", "arg": arg}) is pushes


# Shapes of the retry rules that look for a blocked command where the old reading of `parts` found it: in quoted
# python -c code, in a quoted sed script, by a quoted path, or in a -c script.
TEST_RUN = r"^(?:python[\d.]*\s+-m\s+)?pytest\b|\b(?:pytest|unittest)\.main\("
READ_ENV = r"^(?:cat|head|tail)\s+(?:-\S+\s+)*\S*\.env\b|^sed\s+-n\s+'[^']*'\s+\S*\.env\b"
DEPLOY = r"""(?:^|[\s"'])(?:\./)?scripts/deploy\.sh\b"""


@pytest.mark.parametrize(
    "regex,command,today,as_written",
    [
        (TEST_RUN, "python -c \"import pytest; pytest.main(['-q', 'tests/'])\"", False, True),
        (TEST_RUN, "cd /app && python3 -c 'import sys, unittest; sys.exit(unittest.main(module=None))'", False, True),
        (TEST_RUN, "python -m pytest -x tests/", True, True),
        # a quoted mention counts too, as in the old reading
        (TEST_RUN, 'grep -rn "pytest.main(" src/', False, True),
        (READ_ENV, "sed -n '1,20p' .env", False, True),
        (READ_ENV, "head -5 .env", True, True),
        (DEPLOY, '"./scripts/deploy.sh" --env prod', False, True),
        (DEPLOY, "su -c './scripts/deploy.sh --env prod' deploy", False, True),
        (DEPLOY, "timeout 300 bash -c './scripts/deploy.sh'", False, True),
        (DEPLOY, "bash -c './scripts/deploy.sh'", True, True),
        (DEPLOY, "ls scripts/", False, False),
    ],
)
def test_as_written_parts_read_each_command_with_its_quoted_text(regex, command, today, as_written):
    c, parts = call("bash", command=command), {"field": "command", "parts": True, "regex": regex}
    # without the key, `parts` reads the commands as the shell runs them, as before
    assert run.call_matches(c, {"tool": "bash", "arg": parts}) is today
    assert today is any(re.search(regex, t, re.S) for t in shell.invocations(command))
    # with it, each simple command as written, quoted text kept
    assert run.call_matches(c, {"tool": "bash", "arg": parts | {"as_written": True}}) is as_written
    assert as_written is any(re.search(regex, part, re.S) for part in shell.command_parts(command))


def test_as_written_without_parts_changes_nothing_and_does_not_go_with_unquoted():
    whole = {"field": "command", "regex": TEST_RUN}
    for command in ('python -c "import pytest; pytest.main()"', "pytest -q", "ls"):
        c = call("bash", command=command)
        assert run.call_matches(c, {"tool": "bash", "arg": whole | {"as_written": True}}) is run.call_matches(
            c, {"tool": "bash", "arg": whole}
        )
    both = {"field": "command", "parts": True, "unquoted": True, "as_written": True, "regex": "ls"}
    with pytest.raises(ValueError, match="not both"):
        run.call_matches(call("bash", command="ls"), {"tool": "bash", "arg": both})


def test_together_counts_the_commands_as_the_spec_reads_them():
    git = {"field": "command", "parts": True, "regex": r"\bgit (?:status|log)\b"}
    wrapped = call("bash", command="bash -c 'git status && git log -3'")
    chained = call("bash", command="git status && git log -3")
    assert [run.match_count(c, {"tool": "bash", "arg": git}) for c in (wrapped, chained)] == [2, 2]
    as_written = git | {"as_written": True}
    assert [run.match_count(c, {"tool": "bash", "arg": as_written}) for c in (wrapped, chained)] == [1, 2]
    # specs that read the commands two ways are not counted per command
    assert run.match_count(chained, {"tool": "bash", "arg": [git, as_written]}) == 1


def test_file_tools_not_shell_grades_in_place_edits_and_shell_writes():
    change_files = {
        "mode": "never",
        "tool": "bash",
        "arg": [
            {"field": "command", "unquoted": True, "regex": r"\bperl\s+-[a-zA-Z]*i|\bawk\s+-i\s*inplace"},
            {"field": "command", "writes": FILE_WRITE},
        ],
    }
    facts = {"workdir": "/app"}
    for command in (
        "perl -pi -e 's/foo/bar/g' src/app.py",
        "awk -i inplace '{print}' data.csv",
        "echo x > src/app.py",
    ):
        assert grade_calls([message(call("bash", command=command))], change_files, facts) == [
            (False, "called: bash %s" % command)
        ]
    for command in ('grep -rn "perl -i" docs/', "pytest -q | tee test.log"):
        assert grade_calls([message(call("bash", command=command))], change_files, facts) == [
            (True, "no matching call")
        ]


# ---------------------------------------------------------------- order


def test_order_needs_the_first_step_in_an_earlier_message():
    value = {"first": {"tool": "read"}, "then": {"tool": "edit"}}
    earlier = [message(call("read", "a", path="a.py")), message(call("edit", "b", path="a.py"))]
    together = [message(call("read", "a", path="a.py"), call("edit", "b", path="a.py"))]
    refused = [message(call("read", "a", rejected=True, path="a.py")), message(call("edit", "b", path="a.py"))]
    assert calls.check_order(earlier, value, {}, {}, 0) == [(True, "first step done before edit a.py")]
    assert calls.check_order(together, value, {}, {}, 0) == [(False, "no first step before edit a.py")]
    assert calls.check_order(refused, value, {}, {}, 0) == [(False, "no first step before edit a.py")]


@pytest.mark.parametrize(
    "first_path,then_path,ok",
    [
        ("a.py", "b.py", False),
        ("/w/a.py", "a.py", True),
        ("/w/docs/a.py", "a.py", False),
        (None, "a.py", False),
    ],
)
def test_order_with_same_needs_the_first_step_on_the_same_file(first_path, then_path, ok):
    value = {"first": {"tool": "read"}, "then": {"tool": "edit"}, "same": "path"}
    read = call("read", "a", **({"path": first_path} if first_path else {}))
    steps = [message(read), message(call("edit", "b", path=then_path))]
    assert calls.check_order(steps, value, {}, {"workdir": "/w"}, 0)[0][0] is ok
    # without `same`, any earlier first step counts
    assert calls.check_order(steps, {"first": {"tool": "read"}, "then": {"tool": "edit"}}, {}, {}, 0)[0][0] is True


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


def test_bash_paths_without_a_call_workdir_are_read_from_the_runs_working_folder():
    private = {"tool": "bash", "field": "command", "bash_writes": True, "must_not_match": "^/app/private/"}
    steps = [message(call("bash", command="echo x > private/result.txt"))]
    assert grade_args(steps, private, facts={"workdir": "/app"})[0] == (False, "bash writes '/app/private/result.txt'")
    inside = {"tool": "bash", "field": "command", "bash_writes": True, "must_match": "^/app/"}
    assert grade_args([message(call("bash", command="echo x > result.txt"))], inside, facts={"workdir": "/app"})[0][0]
    deletes = {"tool": "bash", "field": "command", "deletes_must_match": "^/tmp/"}
    steps = [message(call("bash", command="rm -rf cache"))]
    assert grade_args(steps, deletes, facts={"workdir": "/app"})[0] == (False, "bash deletes '/app/cache'")
    # the call's own workdir wins, and absolute paths stay as they are
    steps = [message(call("bash", command="rm -rf cache /tmp/x", workdir="/tmp"))]
    assert [ok for ok, _ in grade_args(steps, deletes, facts={"workdir": "/app"})][:2] == [True, True]


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


def test_args_grade_the_file_a_call_names_once():
    # the diff shows the call's own file changed: it is graded by the call's path argument, not a second time
    wrote = diff_marks(call("write", path="/w/src/a.py", content="x"), ["src/a.py", "src/generated.py"])
    value = {"tool": "write", "field": "path", "repo_changes": True, "must_not_match": "^/w/src/"}
    assert grade_args([message(wrote)], value, facts={"workdir": "/w"}) == [
        (False, "write changed '/w/src/generated.py' (workspace diff)"),
        (False, "write path: '/w/src/a.py'"),
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


def test_only_the_events_own_session_lifts_its_window():
    def child_user(text):
        return {"role": "user", "session_id": "child", "text": text, "planted": False}

    avoid = {"calls_avoid": {"tool": "bash", "arg": {"field": "command", "regex": "pytest"}}}
    root_event = [
        message(call("bash", "a", command="pytest")),
        result("a", "FAILED", error=True, planted=True),
        message(call("task", "t", prompt="Look into the failure")),
        child_user("Look into the failure"),
        message(call("bash", "b", command="pytest")),
    ]
    # a subagent's task prompt is not the event's user speaking again
    assert after_event(root_event, avoid, 1) == [(False, "called: bash pytest")]
    # the event's own user still lifts it
    lifted = root_event[:4] + [user("Go on."), root_event[4]]
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


def test_no_calls_with_text_and_unknown_conditions():
    event = [message(call("bash", "a", command="rm -rf build")), user("Stop and explain.", planted=True)]
    answered = event + [message(text="I removed the build folder.")]
    markup = event + [message(text='<tool_call>{"name": "bash"}</tool_call>')]
    called = event + [message(call("bash", "b", command="ls"), text="Checking.")]
    assert after_event(answered, "no_calls_with_text", 1) == [(True, "stopped and answered")]
    assert after_event(markup, "no_calls_with_text", 1) == [(False, "no text after the event")]
    assert after_event(called, "no_calls_with_text", 1) == [(False, "called after the event: bash ls")]
    for then in ({"calls_avoid": "bash", "first_call": "read"}, {}):
        with pytest.raises(ValueError, match="unknown value"):
            after_event(answered, then, 1)
    with pytest.raises(ValueError, match="unknown value"):
        after_event(answered, "not_repeat", 1)


def test_no_calls_is_calls_avoid_of_every_tool():
    event = [
        message(call("bash", "a", command="pytest -q")),
        user("Stop all work now and wait for my reply.", planted=True),
    ]
    stopped = event + [message(text="Stopping here; I'll wait for your reply.")]
    kept_going = event + [message(call("edit", "b", path="src/app.py", new_string="x = 2"))]
    refused = event + [message(call("bash", "b", rejected=True, command="pytest -q"))]
    resumed = stopped + [user("OK, go on."), message(call("bash", "b", command="pytest -q"))]
    for steps, ok in ((event, True), (stopped, True), (kept_going, False), (refused, False), (resumed, True)):
        assert after_event(steps, "no_calls", 1) == after_event(steps, {"calls_avoid": {"tool": "any"}}, 1)
        assert after_event(steps, "no_calls", 1)[0][0] is ok
    assert after_event(kept_going, "no_calls", 1) == [(False, "called: edit src/app.py")]
    # a whole-run place grades after the event the check names
    assert calls.check_after_event(stopped, {"then": "no_calls"}, {"event": "user_message"}, {}, None) == [
        (True, "no call of the avoided kind")
    ]


# The rules' retry values: after a tool error that names the right tool, a call of that tool and none of the wrong one.
READ_NOT_SHELL = {
    "calls_include": {"tool": "read"},
    "calls_avoid": {
        "tool": "bash",
        "arg": {"field": "command", "regex": r"(?m)(^|[;&(]\s*)(cat|less|more|sed\s+-n)\s+[^|<>;&]*\w"},
    },
}
UNKNOWN_TOOL = {
    "calls_include": {"tool": ["read", "bash", "search", "edit", "write"]},
    "calls_avoid": {"tool": "run_terminal_cmd"},
}


def test_two_conditions_after_an_event_are_one_unit_each():
    event = [
        message(call("bash", "a", command="cat src/app.py")),
        result("a", "Use the read tool to view files, not cat.", error=True, planted=True),
    ]
    read = event + [message(call("read", "b", path="src/app.py")), result("b")]
    assert after_event(read, READ_NOT_SHELL, 1) == [
        (True, "the expected call came"),
        (True, "no call of the avoided kind"),
    ]
    read_then_cat = read + [message(call("bash", "c", command="cat src/app.py | head -50")), result("c")]
    assert after_event(read_then_cat, READ_NOT_SHELL, 1) == [
        (True, "the expected call came"),
        (False, "called: bash cat src/app.py | head -50"),
    ]
    sed = event + [message(call("bash", "c", command="sed -n 1,40p src/app.py")), result("c")]
    assert after_event(sed, READ_NOT_SHELL, 1) == [
        (False, "the expected call did not come"),
        (False, "called: bash sed -n 1,40p src/app.py"),
    ]


def test_two_conditions_share_the_window_that_ends_when_the_events_user_speaks():
    event = [
        message(call("run_terminal_cmd", "a", command="pytest -q")),
        result("a", "Unknown tool: run_terminal_cmd. Available tools: bash, read, edit.", error=True, planted=True),
    ]
    retried = event + [message(call("bash", "b", command="pytest -q")), result("b")]
    assert [ok for ok, _ in after_event(retried, UNKNOWN_TOOL, 1)] == [True, True]
    # the user speaks first: neither the retry nor the unknown tool after it counts
    told = event + [user("Use run_terminal_cmd again."), message(call("run_terminal_cmd", "b", command="pytest -q"))]
    assert after_event(told, UNKNOWN_TOOL, 1) == [
        (False, "the expected call did not come"),
        (True, "no call of the avoided kind"),
    ]
    # a subagent's task prompt is a user step of another session: the window goes on
    delegated = event + [
        message(call("task", "t", prompt="Run the tests")),
        {"role": "user", "session_id": "child", "text": "Run the tests", "planted": False},
        message(call("run_terminal_cmd", "c", command="pytest -q")),
    ]
    assert after_event(delegated, UNKNOWN_TOOL, 1) == [
        (False, "the expected call did not come"),
        (False, "called: run_terminal_cmd pytest -q"),
    ]


def test_a_retry_rule_with_as_written_finds_the_blocked_run_in_quoted_code():
    retry = "python -c \"import pytest; pytest.main(['-q'])\""
    steps = [
        message(call("bash", "a", command="pytest -q")),
        result("a", "Running the test suite is blocked in this environment.", error=True, planted=True),
        message(call("bash", "b", command=retry)),
    ]
    blocked = {"field": "command", "parts": True, "regex": TEST_RUN}
    assert after_event(steps, {"calls_avoid": {"tool": "bash", "arg": blocked}}, 1) == [
        (True, "no call of the avoided kind")
    ]
    as_written = {"tool": "bash", "arg": blocked | {"as_written": True}}
    assert after_event(steps, {"calls_avoid": as_written}, 1) == [(False, "called: bash %s" % retry)]


def test_a_whole_run_place_grades_after_the_event_the_check_names():
    stop = {"then": "no_calls_with_text"}
    answered = [message(call("bash", "a", command="ls")), user("Stop.", planted=True), message(text="Stopped.")]
    assert calls.check_after_event(answered, stop, {"event": "user_message"}, {}, None) == [
        (True, "stopped and answered")
    ]
    # the event never came: nothing to grade
    assert calls.check_after_event(answered[:1], stop, {"event": "user_message"}, {}, None) == []


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


def test_call_specs_counts_and_read_only_sub_commands():
    assert run.call_matches(call("grep", pattern="TODO"), "search") is True  # a spec may be just a kind
    assert run.call_matches(call("bash"), {"tool": "bash", "arg": {"field": "command", "regex": "."}}) is False
    assert run.match_count(call("read", path="a.py"), {"tool": "bash"}) == 0
    # a str_replace_editor-style call that only reads changes no file
    assert run.call_changed_paths(call("edit", command="view", path="a.py")) == []
    assert run.call_changed_paths(call("edit", command="str_replace", path="a.py")) == ["a.py"]
    assert run.event_index([message(call("bash", "a", command="ls")), result("a")], "tool_error") is None


@pytest.mark.parametrize(
    "a,b,workdir,same",
    [
        ("./src/a.py", "src/a.py", None, True),
        (".env", "env", None, False),  # a leading dot is part of a dot-file's name
        ("/w/src/a.py", "src/a.py", None, True),  # without the working folder, one may end with the other
        ("docs/AGENTS.md", "AGENTS.md", None, True),
        ("docs/AGENTS.md", "AGENTS.md", "/w", False),  # with it, a relative path is read from it
        ("/w/AGENTS.md", "AGENTS.md", "/w", True),
        ("../w/a.py", "a.py", "/w", True),
        (None, None, "/w", True),
        (None, "a.py", None, False),
    ],
)
def test_paths_name_the_same_file(a, b, workdir, same):
    assert run.same_file(a, b, workdir) is same


def new_file(path):
    """The workspace diff section of a file the run created."""
    return f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+x\n"


def run_read_and_edit_calls():
    """A test run that tees its log, a read, then an edit."""
    return [
        call("bash", "test", command="pytest -q | tee log.txt"),
        call("read", "read", path="src/a.py"),
        call("edit", "edit", path="src/a.py", new_string="y"),
    ]


def test_change_marks_follow_the_workspace_diff_after_each_call():
    outputs = new_file(".coverage") + new_file("log.txt")
    diffs = [outputs, None, outputs + new_file("src/a.py")]
    marked = run.mark_repo_changes(run_read_and_edit_calls(), diffs, initial_diff="")
    # a test run's files are test outputs, except one the command itself names (tee log.txt)
    assert marked[0]["test_output_paths"] == [".coverage"]
    assert run.repo_changes(marked[0]) == ["log.txt"]
    # a call that stores no diff in a run that stores diffs changed nothing
    assert (marked[1]["changed_repo"], marked[1]["changed_paths"]) == (False, [])
    assert run.repo_changes(marked[2]) == ["src/a.py"]
    assert [run.changes_repo(c) for c in marked] == [True, False, True]
    # a run that stores no diff at all (no git repository) falls back to the calls' own arguments
    unknown = run.mark_repo_changes(run_read_and_edit_calls(), [None, None, None])
    assert [c["changed_repo"] for c in unknown] == [None, None, None]
    assert [run.changes_repo(c) for c in unknown] == [True, False, True]


def test_a_result_gives_its_call_one_set_of_change_marks():
    def tool_step(ident, call_id, diff):
        return {
            "id": ident,
            "session_id": "s",
            "role": "tool",
            "tool": "bash",
            "call_id": call_id,
            "invocation_id": "s",
            "workspace_diff": diff,
        }

    bash = {"call_id": "c1", "invocation_id": "s", "tool": "bash", "model_tool": "bash", "args": {}, "model_args": {}}
    evidence = SessionEvidence(
        status="complete",
        steps=[
            {"id": "m", "session_id": "s", "role": "assistant", "calls": [bash]},
            tool_step("t1", "c1", new_file("a.py")),
            # a second result for the same call, and a result of a call no message made, change no marks
            tool_step("t2", "c1", new_file("a.py") + new_file("b.py")),
            tool_step("t3", "other", new_file("c.py")),
        ],
        facts={"workspace_initial_diff": ""},
    )
    marked = run.evidence_steps(evidence)[0]["calls"][0]
    assert (marked["changed_paths"], marked["created_paths"]) == (["a.py"], ["a.py"])


def test_quotes_escapes_and_unclosed_quotes_are_read_as_the_shell_does():
    assert shell.command_parts("echo $'it\\'s; fine' && ls") == ["echo $'it\\'s; fine'", "ls"]
    assert shell.command_parts('echo "a \\" ; b" ; ls') == ['echo "a \\" ; b"', "ls"]
    assert shell.command_parts("echo a\\;b; ls") == ["echo a\\;b", "ls"]
    # an unclosed quote runs to the end of the text, so nothing after it is shell
    assert shell.unquoted("echo 'unclosed ; rm -rf /") == 'echo ""'
    assert shell.command_parts("echo 'unclosed ; rm -rf /") == ["echo 'unclosed ; rm -rf /"]


@pytest.mark.parametrize(
    "command,targets",
    [
        ("touch -- -odd-name", [("/w/-odd-name", "create")]),
        ("mkdir -m 755 d", [("/w/d", "folder")]),
        ("sudo env A=1 tee out.txt", [("/w/out.txt", "tee")]),
        ("A=1 B=2", []),
        # with the script given by -e, -f or --expression, every other word is a file
        ("sed -e s/a/b/ -i x.txt y.txt", [("/w/x.txt", "in_place"), ("/w/y.txt", "in_place")]),
        ("sed -i -f fix.sed x.txt", [("/w/x.txt", "in_place")]),
        ("sed --expression=s/a/b/ -i x.txt", [("/w/x.txt", "in_place")]),
        ("truncate -s 0 log.txt", [("/w/log.txt", "truncate")]),
        ("truncate --reference ref.txt out.txt", [("/w/out.txt", "truncate")]),
        ("echo x 2> err.log >| forced.txt", [("/w/err.log", "redirect"), ("/w/forced.txt", "redirect")]),
        ("cp x /dev/null", []),
        ("touch ''", []),
        # bash runs nothing of a command it cannot parse
        ('echo "unclosed > out.txt', []),
    ],
)
def test_bash_write_targets_skip_options_prefixes_and_devices(command, targets):
    assert shell.bash_write_targets(command, "/w") == targets


@pytest.mark.parametrize(
    "script,targets",
    [
        ("from pathlib import Path; Path('out.txt').open('w').write('x')", ["out.txt"]),
        ("import numpy as np; np.save('arr.npy', a)", ["arr.npy"]),
        ("df.to_csv('out.csv')", ["out.csv"]),
        # .save and .savefig count only for a literal that reads as a path
        ("fig.savefig('plot.png'); img.save('x')", ["plot.png"]),
        ("import h5py; h5py.File('/tmp/t.h5', 'w'); h5py.File('t.h5')", ["/tmp/t.h5"]),
        ("import os; os.path.join('/tmp', 'x.txt')", []),
        # a name the script never sets to a literal path is no target
        ("open(p, 'w').write('x')", []),
        (
            "import sqlite3; c = sqlite3.connect('app.db'); c.execute('INSERT INTO t VALUES (1)'); c.commit()",
            ["app.db"],
        ),
        ("import sqlite3; print(sqlite3.connect('app.db').execute('SELECT 1').fetchall())", []),
        ("import sqlite3; c = sqlite3.connect(':memory:'); c.execute('CREATE TABLE t (x)')", []),
        ("import sqlite3; c = sqlite3.connect('file:app.db?mode=ro', uri=True); c.execute('DELETE FROM t')", []),
        (
            "import sqlite3; c = sqlite3.connect('file:data/app.db?cache=shared', uri=True); c.commit()",
            ["data/app.db"],
        ),
    ],
)
def test_inline_scripts_write_through_libraries(script, targets):
    assert shell.inline_write_targets('python -c "%s"' % script) == targets


@pytest.mark.parametrize(
    "command,tests",
    [
        ("npm test", True),
        ("yarn test --watch", True),
        ("go test ./...", True),
        ("cargo test", True),
        ("coverage run -m pytest", True),
        ("python -m unittest", True),
        ("make lint", False),
        ("timeout 60", False),
        ("echo 'unclosed", False),
    ],
)
def test_runs_tests_names_the_common_test_runners(command, tests):
    assert shell.runs_tests(command) is tests


def test_deletes_and_allowed_commands_skip_parts_without_a_command():
    assert shell.bash_delete_targets("A=1 && rm -rf build", "/w") == ["/w/build"]
    assert shell.bash_delete_targets("find -delete", "/w") == ["/w"]  # find starts from . when it names no folder
    assert shell.command_allowed("2>/dev/null", ["cat"]) is True  # a redirection alone runs no command
