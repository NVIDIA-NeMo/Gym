# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The data-build scripts' rules on synthetic rows: the normalisation, the drop rules, the session cap, and the
command-line checks."""

import json
from pathlib import Path
from typing import Any

import pytest

from resources_servers.agentic_turn_simulation.scripts import mix_rows
from resources_servers.agentic_turn_simulation.scripts.clean_rows import (
    REDACTED_SANDBOX_HOME,
    SANDBOX_HOME,
    Config,
    TextPatterns,
    absent_input,
    duplicates,
    effective_sessions,
    flattened_tool_history,
    incomplete_tool_list,
    load_patterns,
    main,
    openclaw_tooling,
    parse_args,
    placeholder_pattern,
    prepare_row,
    restore_sandbox_home,
    row_features,
    same_state,
    session_cap,
    typed_text,
)


VOCABULARY = {"tags": ["EMAIL", "URL"], "numbered_tags": ["NAME"]}
# The placeholder inside the redacted home path, e.g. "[TAG]" of "/home/[TAG]".
HOME_PLACEHOLDER = REDACTED_SANDBOX_HOME.removeprefix("/home/")
HOME_VOCABULARY = {"tags": [HOME_PLACEHOLDER.strip("[]")]}


def _row(
    row_id: str,
    session: str | None,
    *turns: tuple[str, str],
    harness: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    items = [{"type": "message", "role": role, "content": text} for role, text in turns]
    vm = {"row_id": row_id, "system_prompt": "sp", **metadata}
    if session is not None:
        vm["session_id"] = session
    if harness is not None:
        vm["harness"] = harness
    return {"responses_create_params": {"input": items, "tools": tools or []}, "verifier_metadata": vm}


def _line(row: dict[str, Any]) -> bytes:
    return (json.dumps(row) + "\n").encode()


# ---------------------------------------------------------------------------------------------------- normalise
def _home_row(harness: str) -> dict[str, Any]:
    row = _row(
        "a",
        "s",
        ("user", f"list {REDACTED_SANDBOX_HOME}/demo please"),
        harness=harness,
        tools=[{"type": "function", "name": "bash", "description": f"runs in {REDACTED_SANDBOX_HOME}"}],
        reference_turn=[
            {"type": "message", "role": "assistant", "content": f"saved {REDACTED_SANDBOX_HOME}/demo/a.txt"}
        ],
    )
    row["responses_create_params"]["instructions"] = f"home is {REDACTED_SANDBOX_HOME}"
    return row


def test_restore_sandbox_home_rewrites_every_location_of_a_sandbox_row() -> None:
    row = _home_row("sandbox")
    line = _line(row)
    out, restored, counts = restore_sandbox_home(line, row)
    assert REDACTED_SANDBOX_HOME.encode() not in out and HOME_PLACEHOLDER.encode() not in out
    assert json.loads(out) == restored
    assert restored["responses_create_params"]["input"][0]["content"] == "list /home/user/demo please"
    assert restored["verifier_metadata"]["reference_turn"][0]["content"] == "saved /home/user/demo/a.txt"
    assert counts == {"input": 1, "tools": 1, "instructions": 1, "reference_turn": 1}
    # Only the path changes: the line is otherwise byte-identical.
    assert SANDBOX_HOME == "/home/user"
    assert out == line.replace(REDACTED_SANDBOX_HOME.encode(), b"/home/user")


def test_restore_sandbox_home_leaves_other_harnesses_and_bare_tags_alone() -> None:
    row = _home_row("claude_code")
    line = _line(row)
    assert restore_sandbox_home(line, row) == (line, row, {})
    bare = _row("b", "s", ("user", f"ask {HOME_PLACEHOLDER} about /home/user"), harness="sandbox")
    assert restore_sandbox_home(_line(bare), bare)[2] == {}
    # The placeholder outside the home path is not the artifact and stays.
    assert restore_sandbox_home(_line(bare), bare)[0] == _line(bare)


def test_restore_sandbox_home_refuses_a_line_whose_raw_and_decoded_forms_differ() -> None:
    row = _row("a", "s", ("user", f"cd {REDACTED_SANDBOX_HOME}"), harness="sandbox")
    escaped = _line(row).replace(b"/home/", b"\\/home\\/")
    with pytest.raises(ValueError, match="differs from the structural one"):
        restore_sandbox_home(escaped, json.loads(escaped))


# ---------------------------------------------------------------------------------------------------- drops
def test_placeholder_pattern_matches_exact_and_numbered_tags_only() -> None:
    pattern = placeholder_pattern(VOCABULARY)
    text = "[EMAIL] [NAME_12] [NAME] [TOKEN] [INFO] [URL_2]"
    assert pattern.findall(text) == ["EMAIL", "NAME_12"]
    assert placeholder_pattern({"tags": []}) is None
    with pytest.raises(ValueError, match="unknown vocabulary keys"):
        placeholder_pattern({"tag": ["NAME"]})


def test_last_user_placeholder_reads_only_the_last_user_message() -> None:
    config = Config(placeholders=placeholder_pattern(VOCABULARY))

    def tags(*turns: tuple[str, str], harness: str = "claude_code") -> tuple[str, ...]:
        return prepare_row(_line(_row("a", "s", *turns, harness=harness)), config)[1].placeholder_tags

    assert tags(("user", "send it to [EMAIL] and [NAME_3]")) == ("EMAIL", "NAME_N")
    assert tags(("user", "mail [EMAIL]"), ("assistant", "done"), ("user", "thanks, now the report")) == ()
    assert tags(("user", "the key was [TOKEN], rotate it")) == ()


def test_last_user_placeholder_runs_after_restore_sandbox_home() -> None:
    config = Config(placeholders=placeholder_pattern(HOME_VOCABULARY))
    path = ("user", f"open {REDACTED_SANDBOX_HOME}/notes.md")
    line, row = prepare_row(_line(_row("a", "s", path, harness="sandbox")), config)
    assert row.placeholder_tags == () and b"/home/user/notes.md" in line
    # Without the normalise level, or in another harness, the same text is a placeholder.
    tag = tuple(HOME_VOCABULARY["tags"])
    assert prepare_row(_line(_row("a", "s", path, harness="openclaw")), config)[1].placeholder_tags == tag
    validity_only = Config(levels=frozenset({"validity"}), placeholders=config.placeholders)
    assert prepare_row(_line(_row("a", "s", path, harness="sandbox")), validity_only)[1].placeholder_tags == tag


def test_typed_text_drops_harness_blocks_and_normalises() -> None:
    assert typed_text("<system-reminder>x</system-reminder>  Fix   THE bug\n") == "fix the bug"


def test_typed_text_takes_dataset_patterns_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "patterns.json"
    path.write_text(json.dumps({"strip": ["<note>.*?</note>"], "cut": ["(?i)-- sent from my phone"]}))
    patterns = load_patterns(path)
    assert isinstance(patterns, TextPatterns) and len(patterns.strip) == 5 and len(patterns.cut) == 3
    assert typed_text("Fix it <note>a\nb</note> now -- Sent from my phone", patterns) == "fix it now"
    assert typed_text("Fix it <note>a</note> now") == "fix it <note>a</note> now"


def test_flattened_tool_history_flags_text_renderings_only() -> None:
    reply = {"type": "message", "role": "assistant", "content": "[Tool result: ok]"}
    assert flattened_tool_history([reply])
    assert not flattened_tool_history([{**reply, "content": "<tool_call>x</tool_call>"}])


def test_absent_input_drops_dropped_media_and_keeps_declared_unreadable_input() -> None:
    unreadable = 'Cannot read "x.png" (this model does not support image input) [Image #1]'
    rows = [
        row_features(_row("a", "s1", ("user", "what is in [Image #1]"))),
        row_features(_row("b", "s2", ("user", unreadable))),
        row_features(_row("c", "s3", ("user", "please use the archive I sent you yesterday evening"))),
        row_features(_row("d", "s4", ("user", "please use the archive I sent you yesterday evening"))),
        row_features(_row("e", "s5", ("user", unreadable))),
    ]
    # A reviewed id is dropped even when the harness also said it could not read the input.
    assert absent_input(rows, {"c": "refers_to_archive", "e": "needs_attachment"}) == {
        "a": "inline_media_placeholder_media_not_in_input",
        "c": "reviewed:refers_to_archive",
        "d": "same_text_as_reviewed:c",
        "e": "reviewed:needs_attachment",
    }


def test_incomplete_tool_list_cases() -> None:
    system = "## Tooling\n- read: read a file\n- browser_mcp: browse\n\n## Next"
    assert openclaw_tooling(system) == ["read", "browser_mcp"]
    gap = _row("a", "s", ("system", system), ("user", "go"), harness="openclaw", tools=[{"name": "read"}])
    assert incomplete_tool_list(gap) == "openclaw_tooling_gap"
    assert (
        incomplete_tool_list(_row("b", "s", ("user", "go"), harness="hermes", record_kind="subagent"))
        == "hermes_subagent"
    )
    mcp = _row("c", "s", ("system", "# MCP Server Instructions\n..."), ("user", "go"), harness="claude_code")
    assert incomplete_tool_list(mcp) == "claude_code_mcp_instructions"
    assert incomplete_tool_list(_row("d", "s", ("user", "go"), harness="sandbox")) is None


def test_openclaw_tooling_needs_the_exact_heading() -> None:
    assert openclaw_tooling("### Tooling\n- mcp_tool: x\n## Tooling notes\n- other: y") is None
    assert openclaw_tooling("intro\n### Tooling\n- mcp_tool: x\n\n## Tooling\n- read: r\n") == ["read"]


# ---------------------------------------------------------------------------------------------------- row pairs
def test_same_state_first_turn_twins_need_thirty_characters() -> None:
    long = "please write a function that parses the config file"
    assert same_state(row_features(_row("a", "s1", ("user", long))), row_features(_row("b", "s2", ("user", long))))
    assert not same_state(row_features(_row("a", "s1", ("user", "hi"))), row_features(_row("b", "s2", ("user", "hi"))))


ASK = "please write a function that parses the config file"


def test_duplicates_drop_copies_and_keep_a_later_repeat_in_the_same_session() -> None:
    rows = [
        row_features(_row("a", "s1", ("user", ASK))),
        row_features(_row("b", "s2", ("user", ASK))),  # same state, another session
        row_features(_row("c", "s1", ("user", "x"), ("assistant", "done, the parser is in place now"), ("user", ASK))),
    ]
    effective_sessions(rows)
    assert duplicates(rows, set()) == ({"b"}, set())


def test_duplicate_repeat_compares_effective_sessions() -> None:
    # s2 is a clone of s1 (its first turn is a same-state copy), so a later re-ask in s2 is in the kept row's own
    # effective session: a new state, left to the session cap, even though its session id differs.
    clone_reask = ("user", "x"), ("assistant", "done, the parser is in place now"), ("user", ASK)
    rows = [
        row_features(_row("a", "s1", ("user", ASK))),
        row_features(_row("b", "s2", ("user", ASK))),
        row_features(_row("c", "s2", *clone_reask)),
    ]
    effective_sessions(rows)
    assert rows[0].effective_session == rows[1].effective_session == rows[2].effective_session
    assert duplicates(rows, set()) == ({"b"}, set())
    # The same re-ask from an unrelated session is a repeat.
    other_reask = ("user", "y"), ("assistant", "the tests pass and the build is green again"), ("user", ASK)
    rows = [row_features(_row("a", "s1", ("user", ASK))), row_features(_row("f", "s4", *other_reask))]
    effective_sessions(rows)
    assert duplicates(rows, set()) == (set(), {"f"})


def test_duplicate_same_state_keeps_one_per_state_inside_the_kept_session() -> None:
    # s2 is a clone of s1. c and d are the same re-ask after the same reply: a new state inside the kept row's
    # effective session, so one of them stays for the session cap and the other is a same-state copy.
    reask = ("user", "x"), ("assistant", "done, the parser is in place now"), ("user", ASK)
    rows = [
        row_features(_row("a", "s1", ("user", ASK))),
        row_features(_row("b", "s2", ("user", ASK))),
        row_features(_row("c", "s1", *reask)),
        row_features(_row("d", "s2", *reask)),
    ]
    effective_sessions(rows)
    assert duplicates(rows, set()) == ({"b", "d"}, set())
    # A row an earlier rule dropped is never the one kept: with c dropped, d stays.
    assert duplicates(rows, {"c"}) == ({"b"}, set())


def test_rows_without_session_id_or_system_prompt_fall_back() -> None:
    row = _row("a", None, ("system", "be brief"), ("user", ASK))
    del row["verifier_metadata"]["system_prompt"]
    features = row_features(row)
    assert features.session_id == "a" and len(features.system_prompt) == 64
    rows = [features, row_features(_row("b", None, ("user", ASK)))]
    effective_sessions(rows)
    assert duplicates(rows, set()) == (set(), set())  # different system prompts never group


def test_session_cap_spreads_kept_rows_over_the_session() -> None:
    rows = []
    for n in range(1, 11):
        turns = [("user", f"step {i} of the task, continue with part {i}") for i in range(n)]
        rows.append(row_features(_row(f"r{n:02d}", "s", *turns)))
    effective_sessions(rows)
    dropped = session_cap(rows, set(), 5)
    kept = sorted(r.n_user for r in rows if r.row_id not in dropped)
    assert len(dropped) == 5 and kept[0] == 1 and kept[-1] == 10
    assert sorted(r.row_id for r in rows if r.row_id not in session_cap(rows, set(), 1)) == ["r01"]
    with pytest.raises(ValueError, match="cap must be >= 1"):
        session_cap(rows, set(), 0)


# ---------------------------------------------------------------------------------------------------- command line
def test_parse_args_checks_cap_and_levels(tmp_path: Path) -> None:
    rows = str(tmp_path / "rows.jsonl")
    for bad in (["--cap", "0"], ["--levels", "normalise,everything"], []):
        with pytest.raises(SystemExit):
            parse_args([rows, *bad] if bad else [rows])
    assert parse_args([rows, str(tmp_path / "out.jsonl"), "--levels", "normalise"]).levels == {"normalise"}


def test_main_normalises_drops_and_plans(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [
        _row("a", "s1", ("user", f"open {REDACTED_SANDBOX_HOME}/notes.md and fix the heading"), harness="sandbox"),
        _row("b", "s2", ("user", "write to [NAME_1] about the meeting tomorrow"), harness="sandbox"),
        _row("c", "s3", ("user", "summarise the log file in the current directory"), harness="sandbox"),
    ]
    source, out, plan = tmp_path / "rows.jsonl", tmp_path / "out.jsonl", tmp_path / "plan.json"
    vocabulary = tmp_path / "tags.json"
    source.write_bytes(b"".join(_line(r) for r in rows))
    vocabulary.write_text(json.dumps(VOCABULARY))
    main([str(source), str(out), "--placeholders", str(vocabulary), "--plan", str(plan)])
    kept = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["verifier_metadata"]["row_id"] for r in kept] == ["a", "c"]
    assert "/home/user/notes.md" in kept[0]["responses_create_params"]["input"][0]["content"]
    rules = {r["rule"]: r for r in json.loads(plan.read_text())["rules"]}
    assert rules["restore_sandbox_home"]["ids"] == ["a"] and rules["restore_sandbox_home"]["level"] == "normalise"
    assert rules["last_user_placeholder"]["ids"] == ["b"] and rules["last_user_placeholder"]["tag_rows"] == {
        "NAME_N": 1
    }
    assert json.loads(capsys.readouterr().out)["rows_out"] == 2
    # A source file run without the cap, and a plan-only run: no rows file is written.
    plan_only = tmp_path / "plan_only.json"
    main([str(source), "--levels", "normalise,validity", "--placeholders", str(vocabulary), "--plan", str(plan_only)])
    entries = {r["rule"]: r for r in json.loads(plan_only.read_text())["rules"]}
    assert entries["session_cap"]["enabled"] is False and entries["last_user_placeholder"]["ids"] == ["b"]


def _fallback_key_row(row_id: str, session: str | None, *turns: tuple[str, str]) -> dict[str, Any]:
    """A row that names its id with the fallback key instead of `row_id`."""
    row = _row(row_id, session, *turns, harness="sandbox")
    row["verifier_metadata"][mix_rows.ROW_ID_FALLBACK_KEY] = row["verifier_metadata"].pop("row_id")
    return row


def test_rows_with_the_fallback_id_key_still_work(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fallback = _fallback_key_row("a", None, ("user", "summarise the log file in the current directory"))
    assert "row_id" not in fallback["verifier_metadata"]
    features = row_features(fallback)
    assert features.row_id == "a" and features.session_id == "a"
    # row_id wins when a row carries both keys.
    both = _row("b", None, ("user", ASK))
    both["verifier_metadata"][mix_rows.ROW_ID_FALLBACK_KEY] = "other"
    assert mix_rows.row_id(both["verifier_metadata"]) == "b"

    # clean_rows passes the row through byte for byte and plans by its id.
    source, out, plan = tmp_path / "rows.jsonl", tmp_path / "out.jsonl", tmp_path / "plan.json"
    copy = _fallback_key_row("c", "s", ("user", "summarise the log file in the current directory"))
    source.write_bytes(_line(fallback) + _line(copy))
    main([str(source), str(out), "--plan", str(plan)])
    assert out.read_bytes() == _line(fallback)
    rules = {r["rule"]: r for r in json.loads(plan.read_text())["rules"]}
    assert rules["duplicate_same_state"]["ids"] == ["c"]
    capsys.readouterr()

    # mix_rows reads the id from either key: the lines come out unchanged, and an id repeated across keys fails.
    current = tmp_path / "current.jsonl"
    current.write_bytes(_line(_row("b", "s", ("user", ASK))))
    mixed = tmp_path / "mixed.jsonl"
    mix_rows.main([str(mixed), str(out), str(current)])
    assert sorted(mixed.read_bytes().splitlines(keepends=True)) == sorted([_line(fallback), current.read_bytes()])
    assert json.loads(capsys.readouterr().out)["rows"] == 2
    clash = tmp_path / "clash.jsonl"
    clash.write_bytes(_line(_row("a", "s", ("user", ASK))))
    with pytest.raises(SystemExit, match="1 row ids repeat"):
        mix_rows.main([str(tmp_path / "clash_mix.jsonl"), str(out), str(clash)])
