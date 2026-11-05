# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the judge is shown of a session: changes as marked lines, the files changed so far, the run after an event."""

from resources_servers.agentic_if.if_grading.configuration import load_config
from resources_servers.agentic_if.if_grading.rendering import render, render_options


CONFIG = load_config()


def call(tool, ident, *, model_tool=None, **args):
    return {"call_id": ident, "invocation_id": "s", "tool": tool, "model_tool": model_tool or tool, "args": args}


def step(role, text="", *calls):
    return {"role": role, "session_id": "s", "text": text, "calls": list(calls)}


def result(ident, tool, output):
    return {"role": "tool", "session_id": "s", "call_id": ident, "invocation_id": "s", "tool": tool, "output": output}


def test_a_rule_asks_for_its_render_options_in_its_parameters():
    assert render_options({"show_calls": False, "edits": "diff"}) == {"calls": False, "edits": True}
    assert render_options(None) == {"calls": True, "edits": False}


def test_edits_diff_shows_each_write_and_edit_call_as_its_change():
    old = "\n".join(f"line {n}" for n in range(1, 11))
    new = old.replace("line 2", "line two").replace("line 9", "line nine")
    steps = [
        step("user", "Fix the parser."),
        step(
            "assistant",
            "Editing.",
            call("edit", "e1", model_tool="EditFile", path="src/a.py", old_string=old, new_string=new),
            call("write", "w1", path="src/new.py", content="x = 1\ny = 2"),
            call("apply_patch", "p1", patch="*** Begin Patch\n*** Update File: src/b.py\n-old\n+new\n*** End Patch"),
            call("edit", "e2", path="src/c.py", old_string="same", new_string="same"),
            call("bash", "b1", command="pytest -q"),
        ),
        step("assistant", "Done."),
    ]
    shown = render(steps, "run", None, CONFIG, edits=True)
    assert (
        "  CALL EditFile src/a.py (its change, line by line):\n"
        "    kept:    line 1\n"
        "    removed: line 2\n"
        "    added:   line two\n"
        "    kept:    line 3\n"
        "    kept:    line 4\n"
        "    ...\n"
        "    kept:    line 7\n"
        "    kept:    line 8\n"
        "    removed: line 9\n"
        "    added:   line nine\n"
        "    kept:    line 10\n"
    ) in shown
    assert "  CALL write src/new.py (the whole new content):\n    added:   x = 1\n    added:   y = 2\n" in shown
    assert "  CALL apply_patch  (a patch):\n    *** Begin Patch\n    *** Update File: src/b.py\n    -old\n" in shown
    assert "  CALL edit src/c.py (its change, line by line):\n    (no change: the new text is the old text)\n" in shown
    assert '  CALL bash {"command": "pytest -q"}\n' in shown  # any other call shows its arguments
    # without the option, an edit shows its arguments as the model wrote them
    assert '  CALL EditFile {"path": "src/a.py", "old_string": "line 1\\nline 2' in render(steps, "run", None, CONFIG)


def test_message_with_changed_files_lists_only_the_files_changed_before_the_message():
    writes = [call("write", f"w{n}", path=f"src/f{n}.py", content="x") for n in range(52)]
    steps = [
        step("assistant", "", *writes),
        step("assistant", "Wrote the files.", call("write", "own", path="src/own.py", content="x")),
    ]
    shown = render(steps, "message_with_changed_files", 1, CONFIG)
    files, message = shown.split("\n\n--- the message ---\n")
    assert files.splitlines()[0] == CONFIG["contexts"]["message_with_changed_files"]["files_label"]
    assert files.splitlines()[1:] == [f"- src/f{n}.py" for n in range(50)] + ["- [... 2 more files]"]
    # the message's own call comes after its text: the file it writes is not changed yet
    assert message == 'Wrote the files.\n[call] write {"path": "src/own.py", "content": "x"}'


def test_after_event_shows_the_task_the_message_before_the_event_and_everything_after():
    steps = [
        step("user", "Fix the parser."),
        step(
            "assistant",
            "Running the tests.",
            call("bash", "b1", model_tool="RunCommand", command="pytest -q"),
            call("read", "r1", path="a.py"),
        ),
        result("r1", "read", "contents"),
        result("b1", "bash", "Command timed out.") | {"error": True, "planted": True},
        step("assistant", "", call("bash", "b2", model_tool="RunCommand", command="pytest -x")),
        result("b2", "bash", "1 passed"),
        step("assistant", "The tests pass."),
    ]
    assert render(steps, "after_event", 3, CONFIG) == (
        "USER (the task): Fix the parser.\n"
        "--- the assistant message just before the event ---\n"
        "ASSISTANT: Running the tests.\n"
        '  CALL RunCommand {"command": "pytest -q"}\n'
        '  CALL read {"path": "a.py"}\n'
        "--- the event, and everything after it ---\n"
        # the planted result answers a call made before the event, matched by its identity
        "  RESULT (RunCommand, error, planted): Command timed out.\n"
        "ASSISTANT: (no text)\n"
        '  CALL RunCommand {"command": "pytest -x"}\n'
        "  RESULT (RunCommand): 1 passed\n"
        "ASSISTANT: The tests pass."
    )
    # an event at the start of the run has no task or message before it to show
    assert render(steps[3:], "after_event", 0, CONFIG).startswith(
        "--- the event, and everything after it ---\n  RESULT (bash, error, planted): Command timed out.\n"
    )


def test_a_result_is_never_matched_to_a_call_by_arrival_order():
    steps = [
        step(
            "assistant",
            "Two reads.",
            {"tool": "read", "args": {"path": "a"}, "model_tool": "ReadSource"},
            {"tool": "read", "args": {"path": "b"}, "model_tool": "ReadSource"},
        ),
        {"role": "tool", "session_id": "s", "tool": "read", "output": "A"},
        step("assistant", "One search.", {"tool": "grep", "args": {"pattern": "x"}, "model_tool": "Search"}),
        {"role": "tool", "session_id": "s", "tool": "grep", "output": "x found"},
    ]
    shown = render(steps, "run", None, CONFIG)
    # without call identities, a result two calls could answer is shown by its own tool name only
    assert "  RESULT (read): A\n" in shown
    assert shown.endswith('  CALL Search {"pattern": "x"}\n  RESULT (Search): x found')


def test_a_long_run_is_redrawn_with_tighter_caps_before_any_message_is_cut_out():
    steps = [step("user", "Fix the parser.")]
    steps += [step("assistant", f"M{n:02d} " + "x" * 2400) for n in range(20)]
    steps.append(step("assistant", "Final report."))
    shown = render(steps, "run", None, CONFIG)
    assert all(f"M{n:02d} " in shown for n in range(20))
    assert shown.count("[... 1804 characters cut ...]") == 20  # each message cut to the over_max cap of 600
    assert shown.endswith("\nASSISTANT: Final report.")
    without = load_config()
    del without["contexts"]["run"]["over_max"]
    cut = render(steps, "run", None, without)
    # without over_max, the middle of the run is cut out, whole messages with it
    assert [n for n in range(20) if f"M{n:02d} " not in cut] == [12, 13, 14, 15]
    assert cut.endswith("\nASSISTANT: Final report.")
