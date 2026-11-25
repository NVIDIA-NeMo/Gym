# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A restored prefix recorded by another harness is not a change of the continued session's first call.

opencode_sandboxed_agent stamped each file-changing call with a plain `git diff` against the Git index, which
abbreviates object IDs; this harness stamps `git diff --binary --full-index` against the episode's base tree. When the
prefix staged nothing, both say the same about the files.
"""

import os
import shutil
import subprocess

from resources_servers.agentic_if.episode.workspace import workspace_diff_command
from resources_servers.agentic_if.if_grading import engine
from resources_servers.agentic_if.if_grading.checkers.run import changes_repo, evidence_steps, writes_file
from resources_servers.agentic_if.if_grading.configuration import load_config
from resources_servers.agentic_if.if_grading.contracts import ResolvedCheck, SessionEvidence


class Repo:
    """A task repository, with its workspace diff as each harness takes it."""

    def __init__(self, path):
        self.path = path
        path.mkdir(parents=True)
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        self.write("a.py", "def a():\n    return 1\n")
        self.write("b.py", "def b():\n    return 2\n")
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD^{tree}").strip()

    def git(self, *args, env=None):
        return subprocess.run(
            ["git", "-C", str(self.path), *args], check=True, capture_output=True, text=True, env=env
        ).stdout

    def write(self, name, content):
        (self.path / name).write_text(content)

    def recorded_diff(self):
        """opencode_sandboxed_agent's stamp (if-profile.js `workspaceDiff`): untracked files added to a copy of the
        index as intent-to-add, then a plain `git diff`, against that index."""
        index = self.path.parent / "recorded-index"
        shutil.copy2(self.path / ".git" / "index", index)
        env = os.environ | {"GIT_INDEX_FILE": str(index)}
        try:
            self.git("add", "-N", "--ignore-removal", ".", env=env)
            return self.git("diff", env=env)
        finally:
            index.unlink()

    def live_diff(self):
        """This harness's stamp, which seed's snapshots also take: against the base tree, with full object IDs."""
        command = ["/bin/sh", "-c", workspace_diff_command(self.base)]
        return subprocess.run(command, cwd=self.path, check=True, capture_output=True, text=True).stdout


class Judge:
    """No check here asks a judge."""

    config = load_config()
    version, hash = "test-judge", "test-hash"

    def evaluate(self, **kwargs):
        raise AssertionError("a deterministic check asked the judge")


def tool_call(ident, tool, **args):
    return {
        "call_id": ident,
        "invocation_id": "root",
        "tool": tool,
        "args": args,
        "model_tool": tool,
        "model_args": args,
    }


def run_evidence(calls, *, initial, continuation_start, answer="Done."):
    """The evidence of a root session: the user's task, each (call, diff) as an assistant step and its result, the
    continuation's user message before call `continuation_start`, and a final answer."""
    steps = [{"role": "user", "text": "Fix the bug."}]
    for number, (call, diff) in enumerate(calls):
        if number == continuation_start:
            start = len(steps)
            steps.append({"role": "user", "text": "Which of the two files would you change next? Change nothing yet."})
        steps.append({"role": "assistant", "calls": [call]})
        steps.append(
            {
                "role": "tool",
                "tool": call["tool"],
                "call_id": call["call_id"],
                "invocation_id": "root",
                "output": "ok",
                "workspace_diff": diff,
            }
        )
    steps.append({"role": "assistant", "text": answer})
    return SessionEvidence(
        status="complete",
        steps=[{"id": f"s{index}", "session_id": "root", **step} for index, step in enumerate(steps)],
        continuation_start=start,
        facts={"workspace_initial_diff": initial, "workdir": "/app"},
    )


def prefix_then_live(tmp_path, *, edit_after=True):
    """A prefix recorded by opencode_sandboxed_agent edits a.py and writes notes.md; the continued session runs a
    command that changes nothing, then (with `edit_after`) edits b.py. Returns the evidence and the diffs."""
    repo = Repo(tmp_path / "repo")
    initial = repo.live_diff()
    repo.write("a.py", "def a():\n    return 10\n")
    first = repo.recorded_diff()
    repo.write("notes.md", "plan\n")
    second = repo.recorded_diff()
    calls = [
        (tool_call("p1", "edit", path="a.py", old_string="1", new_string="10"), first),
        (tool_call("p2", "write", path="notes.md", content="plan\n"), second),
        (tool_call("l1", "bash", command="git status --short && cat a.py"), repo.live_diff()),
    ]
    if edit_after:
        repo.write("b.py", "def b():\n    return 20\n")
        calls.append((tool_call("l2", "edit", path="b.py", old_string="2", new_string="20"), repo.live_diff()))
    return run_evidence(calls, initial=initial, continuation_start=2), second, calls[2][1]


def marks(evidence):
    return {call["call_id"]: call for step in evidence_steps(evidence) for call in step.get("calls", [])}


def object_ids(diff):
    """The pre- and post-image object IDs of each `index` line of a diff."""
    return [i for line in diff.splitlines() if line.startswith("index ") for i in line.split()[1].split("..")]


def test_the_first_call_after_a_restored_prefix_changes_only_what_it_changed(tmp_path):
    evidence, recorded, live = prefix_then_live(tmp_path)
    # The same files in two forms: the prefix's last diff abbreviates object IDs, the continued session's spells them.
    assert recorded != live and len(object_ids(recorded)) == len(object_ids(live)) == 4
    assert all(len(i) < 40 for i in object_ids(recorded)) and all(len(i) >= 40 for i in object_ids(live))
    marked = marks(evidence)
    assert (marked["p1"]["changed_paths"], marked["p2"]["created_paths"]) == (["a.py"], ["notes.md"])
    assert (marked["l1"]["changed_repo"], marked["l1"]["changed_paths"]) == (False, [])
    assert not changes_repo(marked["l1"]) and not writes_file(marked["l1"])
    # A call that really changes a file is still marked, with that file alone.
    assert (marked["l2"]["changed_repo"], marked["l2"]["changed_paths"]) == (True, ["b.py"])
    assert changes_repo(marked["l2"]) and writes_file(marked["l2"])


def test_a_planted_file_in_the_initial_diff_is_no_change_of_the_prefix(tmp_path):
    repo = Repo(tmp_path / "repo")
    repo.write("TODO.md", "planted\n")
    initial = repo.live_diff()
    repo.write("a.py", "def a():\n    return 10\n")
    calls = [
        (tool_call("p1", "edit", path="a.py", old_string="1", new_string="10"), repo.recorded_diff()),
        (tool_call("l1", "bash", command="ls"), repo.live_diff()),
    ]
    marked = marks(run_evidence(calls, initial=initial, continuation_start=1))
    assert (marked["p1"]["changed_paths"], marked["p1"]["created_paths"]) == (["a.py"], [])
    assert (marked["l1"]["changed_repo"], marked["l1"]["changed_paths"]) == (False, [])


def test_a_prefix_recorded_by_this_harness_keeps_its_marks(tmp_path):
    """Every diff in one form: comparing them by meaning marks what comparing their text marked."""
    repo = Repo(tmp_path / "repo")
    initial = repo.live_diff()
    repo.write("a.py", "def a():\n    return 10\n")
    first = repo.live_diff()
    calls = [
        (tool_call("p1", "edit", path="a.py", old_string="1", new_string="10"), first),
        (tool_call("l1", "bash", command="ls"), repo.live_diff()),
    ]
    repo.write("a.py", "def a():\n    return 11\n")
    calls.append((tool_call("l2", "edit", path="a.py", old_string="10", new_string="11"), repo.live_diff()))
    marked = marks(run_evidence(calls, initial=initial, continuation_start=1))
    assert [(marked[c]["changed_repo"], marked[c]["changed_paths"]) for c in ("p1", "l1", "l2")] == [
        (True, ["a.py"]),
        (False, []),
        (True, ["a.py"]),
    ]


def grade(check, evidence):
    return engine.grade([ResolvedCheck.model_validate(check)], evidence, judge=Judge()).checks[0]


NO_CHANGE_AFTER_QUESTION = {
    # exploratory_question_no_change: after the mid-run question, change no file until the user agrees
    "id": "no_change",
    "kind": "calls",
    "parameters": {"value": {"mode": "never", "tool": "any", "changes_repo": True}},
    "scope": {"activation": "continuation_start", "places": ["after_event"]},
}
ONLY_B_CHANGES = {
    # change_only_allowed_paths: what the workspace diffs show changed after the message must be b.py
    "id": "only_b",
    "kind": "args",
    "parameters": {"value": {"tool": "edit", "field": "path", "must_match": "b\\.py$", "repo_changes": True}},
    "scope": {"activation": "continuation_start", "places": ["run"]},
}


def test_mid_run_rules_on_repository_changes_grade_the_continued_session_alone(tmp_path):
    asked, _, _ = prefix_then_live(tmp_path / "asked", edit_after=False)
    assert grade(NO_CHANGE_AFTER_QUESTION, asked).status == "pass"
    assert grade(ONLY_B_CHANGES, asked).status == "not_applicable"
    edited, _, _ = prefix_then_live(tmp_path / "edited")
    result = grade(NO_CHANGE_AFTER_QUESTION, edited)
    assert result.status == "fail" and result.explanation == "called: edit b.py"
    assert grade(ONLY_B_CHANGES, edited).status == "pass"
