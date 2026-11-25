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


EVIDENCE_HEADER = (
    "\n\n--- lines of the session cut out of the view above that hold a link, date, version or number the assistant "
    "wrote ---\n"
)
RULE = "-" * 70  # a filler line long enough that a tool result of a few dozen of them is cut in the run view


def test_evidence_is_a_render_option_only_when_a_rule_turns_it_on():
    assert render_options({"evidence": True}) == {"calls": True, "edits": False, "evidence": True}
    assert render_options({"evidence": False}) == {"calls": True, "edits": False}


def test_evidence_adds_the_cut_out_lines_that_hold_a_token_the_assistant_wrote():
    notes = "\n".join(
        ["    1\trequires 2.31.4 or later"]
        + [RULE] * 12
        + [
            "   57\tversion = 2.31.4",
            "   58\treleased 2026-03-14 by the release bot",
            "   59\tpinned 2.31.40 for the old API",
            "   60\tdownloads 9999",
            "   61\tnotes: https://example.com/notes",
        ]
        + [RULE] * 12
    )
    steps = [
        step("user", "Upgrade the client."),
        step("assistant", "Reading the notes.", call("read", "r1", model_tool="ReadSource", path="NOTES.md")),
        result("r1", "read", notes),
        step("assistant", "Upgraded to 2.31.4, released 2026-03-14; see https://example.com/notes."),
    ]
    plain = render(steps, "run", None, CONFIG)
    assert "version = 2.31.4" not in plain  # cut out of the long result
    # a line-number prefix is left out; a line in view ("requires ...") is not repeated, nor one whose number the
    # assistant did not write ("downloads 9999") or that holds a longer version ("2.31.40")
    assert render(steps, "run", None, CONFIG, evidence=True) == plain + EVIDENCE_HEADER + (
        "  RESULT (ReadSource): version = 2.31.4\n"
        "  RESULT (ReadSource): released 2026-03-14 by the release bot\n"
        "  RESULT (ReadSource): notes: https://example.com/notes"
    )
    # nothing to add: the result is shown whole, or the assistant wrote no token
    short = steps[:2] + [result("r1", "read", "version = 2.31.4")] + steps[3:]
    assert render(short, "run", None, CONFIG, evidence=True) == render(short, "run", None, CONFIG)
    quiet = steps[:3] + [step("assistant", "Upgraded the client.")]
    assert render(quiet, "run", None, CONFIG, evidence=True) == render(quiet, "run", None, CONFIG)
    # message contexts are not cut-out run views: the option leaves them as they are
    assert render(steps, "message", 3, CONFIG, evidence=True) == render(steps, "message", 3, CONFIG)


def test_evidence_keeps_two_lines_per_token_each_cut_around_its_token_within_evidence_chars():
    config = load_config()
    config["contexts"]["run"].update(evidence_line_chars=40, evidence_chars=130)
    log = "\n".join(
        ["start"]
        + [RULE] * 12
        + [
            "build 4821 started",
            "build 4821 queued",
            "build 4821 finished",  # a third line for the same token
            "x" * 100 + " artifact 7731 " + "y" * 100,
            "artifact 7731 uploaded",  # no room left
        ]
        + [RULE] * 12
    )
    steps = [
        step("assistant", "Building.", call("bash", "b1", command="make")),
        result("b1", "bash", log),
        step("assistant", "Build 4821 produced artifact 7731."),
    ]
    added = render(steps, "run", None, config, evidence=True)[len(render(steps, "run", None, config)) :]
    assert added == EVIDENCE_HEADER + (
        "  RESULT (bash): build 4821 started\n"
        "  RESULT (bash): build 4821 queued\n"
        "  RESULT (bash): xxxxxxxxxx artifact 7731 yyyyyyyyyyyyyyy"
    )
    assert len(added) - len(EVIDENCE_HEADER) <= 130


def test_evidence_after_an_event_searches_the_steps_before_it_for_the_tokens_written_after_it():
    steps = [
        step("user", "Upgrade the client. Release notes: https://example.com/notes"),
        step("assistant", "Checking the version.", call("read", "r1", model_tool="ReadSource", path="setup.cfg")),
        result("r1", "read", "version = 2.31.4\nbuild 7731"),
        step("assistant", "Upgrading build 7731.", call("bash", "b1", command="make upgrade")),
        result("b1", "bash", "Notice: the registry is read-only.") | {"planted": True},
        step("assistant", "The client is at 2.31.4; see https://example.com/notes."),
    ]
    plain = render(steps, "after_event", 4, CONFIG)
    # the task message is in view, so its link is not repeated; build 7731 was written before the event only
    assert render(steps, "after_event", 4, CONFIG, evidence=True) == plain + EVIDENCE_HEADER + (
        "  RESULT (ReadSource) before the part shown: version = 2.31.4"
    )


def test_evidence_lists_lines_in_view_too_when_the_run_itself_was_cut_to_fit():
    steps = [
        step("assistant", "Building.", call("bash", "b1", command="make")),
        result("b1", "bash", "build 4821 passed"),
    ]
    final = [step("assistant", "Build 4821 passed.")]
    # shown whole, the result is in view
    assert render(steps + final, "run", None, CONFIG, evidence=True) == render(steps + final, "run", None, CONFIG)
    long = steps + [step("assistant", f"M{n:02d} " + "x" * 2400) for n in range(80)] + final
    view = render(long, "run", None, CONFIG)
    # even with over_max's caps the run is too long, so its middle is cut out, whole messages with it
    assert not all(f"M{n:02d} " in view for n in range(80))
    assert render(long, "run", None, CONFIG, evidence=True) == view + EVIDENCE_HEADER + (
        "  RESULT (bash): build 4821 passed"
    )


ARRIVAL = (
    "--- the instruction arrives here (the planted step below): what came before is part of the session; only what "
    "the assistant does from here on is graded ---"
)


def test_a_whole_run_view_puts_a_line_before_the_step_where_the_instruction_arrived():
    steps = [
        step("user", "Fix the parser."),
        step("assistant", "Running the tests.", call("bash", "b1", command="pytest -q")),
        result("b1", "bash", "1 failed"),
        step("user", "Do not touch the tests.") | {"planted": True},
        step("assistant", "Fixed the parser; the tests pass."),
    ]
    assert render(steps, "run", None, CONFIG, instruction_step=3) == (
        "USER: Fix the parser.\n"
        "ASSISTANT: Running the tests.\n"
        '  CALL bash {"command": "pytest -q"}\n'
        "  RESULT (bash): 1 failed\n"
        f"{ARRIVAL}\n"
        "USER (planted): Do not touch the tests.\n"
        "ASSISTANT: Fixed the parser; the tests pass."
    )
    assert ARRIVAL not in render(steps, "run", None, CONFIG)
    # a run redrawn with the tighter caps of over_max keeps the line
    long = steps[:3] + [step("assistant", f"M{n:02d} " + "x" * 2400) for n in range(20)] + steps[3:]
    assert render(long, "run", None, CONFIG, instruction_step=23).count(f"\n{ARRIVAL}\nUSER (planted): ") == 1
