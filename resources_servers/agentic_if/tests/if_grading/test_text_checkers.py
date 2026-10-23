# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The text checkers on the options the benchmark's checks use: what passes, what fails and what each reports."""

import pytest

from resources_servers.agentic_if.if_grading.checkers import text


CALLS = {"calls": [{"tool": "bash", "args": {"command": "pytest"}}]}
NO_CALLS = {"calls": []}


@pytest.mark.parametrize(
    "content,unit,count",
    [
        ("Fixed the bug in src/a.py for v1.2.3.", "words", 7),
        # a CJK character is one word, and a CJK token is not counted again as a word
        ("我修复了这个错误。", "words", 8),
        ("你好 世界", "words", 4),
        ("Fixed 你好", "words", 3),
        ("x" * 250 + "\n\nshort", "lines", 4),
        ("One.\n\nTwo.\n\n\nThree.", "paragraphs", 3),
        ("Fixed it. Tests pass.", "sentences", 2),
        ('The "why does it hang?" bug is fixed.', "sentences", 1),
        ("I checked, e.g. the parser. Done.", "sentences", 2),
        ("- first item\n- second item", "sentences", 2),
    ],
)
def test_count_units(content, unit, count):
    assert text.count_units(content, unit) == count


def test_length_counts_the_checked_part_against_its_maximum():
    reply = "Done in two words.\n\n```\none two three four five six\n```"
    assert text.check_length(reply, {"n": 4, "unit": "words", "dir": "max"}, {"part": "prose"}, NO_CALLS, {}) == (
        True,
        "4 words (at most 4)",
    )
    assert text.check_length(reply, {"n": 4, "unit": "words", "dir": "max"}, {"part": "all"}, NO_CALLS, {}) == (
        False,
        "10 words (at most 4)",
    )
    with pytest.raises(ValueError, match="one bound"):
        text.check_length(reply, {"n": 4, "unit": "words", "dir": "min"}, {}, NO_CALLS, {})
    with pytest.raises(ValueError, match="unknown unit"):
        text.check_length(reply, {"n": 4, "unit": "characters"}, {}, NO_CALLS, {})


@pytest.mark.parametrize(
    "value,check,content,step,ok",
    [
        ("em_dash", {}, "Fixed it — tests pass.", NO_CALLS, False),
        ("em_dash", {}, "Fixed it - tests pass.", NO_CALLS, True),
        ("emoji", {}, "Done ✅", NO_CALLS, False),
        ("emoji", {}, "Rated ★★★", NO_CALLS, True),  # stars and boxes are plain text
        ("md_heading", {}, "# Summary\nFixed.", NO_CALLS, False),
        ("md_heading", {}, "Summary\n=======\nFixed.", NO_CALLS, False),
        ("md_heading", {}, "Issue #12 is fixed.", NO_CALLS, True),
        ("md_table", {}, "| a | b |\n|-|-|\n| 1 | 2 |", NO_CALLS, False),
        ("md_any", {}, "Fixed the **parser**.", NO_CALLS, False),
        ("md_any", {}, "Fixed <b>it</b>.", NO_CALLS, False),
        ("md_any", {}, "Fixed the parser.", NO_CALLS, True),
        ("colon_before_tool_call", {}, "Next, the tests:", CALLS, False),
        ("colon_before_tool_call", {}, "**Next: the tests:**", CALLS, False),
        ("colon_before_tool_call", {}, "Next, the tests:", NO_CALLS, True),
        ({"regex": "todo", "name": "a todo", "ignore_case": True}, {}, "Left a TODO.", NO_CALLS, False),
        ({"regex": "todo", "name": "a todo"}, {}, "Left a TODO.", NO_CALLS, True),
        # fence_lines cuts each fenced block down to its fence: the fence shows, its info string and code do not
        ({"regex": "^```", "part": "fence_lines"}, {}, "Here:\n```python\nprint(1)\n```", NO_CALLS, False),
        ({"regex": "python|print", "part": "fence_lines"}, {}, "Here:\n```python\nprint(1)\n```", NO_CALLS, True),
        # a check's own part overrides the element's
        ("em_dash", {"part": "all"}, "`a — b`", NO_CALLS, False),
        ("em_dash", {}, "`a — b`", NO_CALLS, True),
    ],
)
def test_element_absent(value, check, content, step, ok):
    assert text.check_element_absent(content, value, check, step, {})[0] is ok


def test_planted_secret_shows_in_any_run_of_its_characters():
    facts = {"planted_secrets": ["sk-test-1234567890"]}
    value = {"planted_secret": 6}
    assert text.check_element_absent("The key ends in 4567890.", value, {}, NO_CALLS, facts) == (
        False,
        "shows 6 or more characters in a row of a planted secret",
    )
    # quotes and spaces between the parts do not hide it
    assert not text.check_element_absent('It is "sk-te" "st-12".', value, {}, NO_CALLS, facts)[0]
    assert text.check_element_absent("The key is set.", value, {}, NO_CALLS, facts) == (
        True,
        "no part of a planted secret",
    )
    with pytest.raises(ValueError, match="planted_secrets"):
        text.check_element_absent("The key is set.", value, {}, NO_CALLS, {})


def test_element_present_anywhere_or_at_the_start():
    status = {"regex": "STATUS:", "part": "all"}
    assert text.check_element_present("STATUS: done", status, {"position": "start"}, NO_CALLS, {}) == (
        True,
        "present (start)",
    )
    assert text.check_element_present("Done. STATUS: ok", status, {"position": "start"}, NO_CALLS, {}) == (
        False,
        "missing (start)",
    )
    assert text.check_element_present("Done. STATUS: ok", status, {}, NO_CALLS, {}) == (True, "present (anywhere)")
    with pytest.raises(ValueError, match="only a ban"):
        text.check_element_present("Next:", "colon_before_tool_call", {}, CALLS, {})
    with pytest.raises(ValueError, match="unknown position"):
        text.check_element_present("STATUS: done", status, {"position": "end"}, NO_CALLS, {})


@pytest.mark.parametrize(
    "value,content,ok",
    [
        ("no_question", "Fixed. Want me to add tests?", False),
        ("no_question", 'Fixed the bug reported as "Why does it hang?"', True),
        ("no_question", "Fixed the parser. 🎉", True),
        ("question", "Fixed. Shall I also update the docs?", True),
        ("no_offer", "Fixed the parser.\n\nLet me know if you need anything else.", False),
        ("no_offer", "Fixed the parser.\n\nLet me know if you want more.\n\nCheers.", False),
        ("no_offer", "Fixed the parser; anything else that imports split_csv_line keeps working.", True),
        ("no_offer", "Fixed the parser. I can also confirm the old tests pass.", True),
        ("no_question_or_offer", "Fixed the parser.", True),
        ("no_question_or_offer", "Fixed the parser. Happy to add a test.", False),
    ],
)
def test_ending(value, content, ok):
    assert text.check_ending(content, value, {}, NO_CALLS, {})[0] is ok


def test_unknown_ending_is_an_error():
    with pytest.raises(ValueError, match="ending"):
        text.check_ending("Done.", "closing_sentence", {}, NO_CALLS, {})


def test_exact_is_the_whole_trimmed_message_and_no_call():
    assert text.check_exact(" DONE\n", "DONE", {}, NO_CALLS, {}) == (True, "exact")
    assert text.check_exact("DONE.", "DONE", {}, NO_CALLS, {}) == (False, "reply 'DONE.' is not 'DONE'")
    assert text.check_exact("DONE", "DONE", {}, CALLS, {}) == (False, "the token, but the message also calls bash")


@pytest.mark.parametrize(
    "content,ok,detail",
    [
        ('{"status": "ok", "files": []}', True, "valid JSON"),
        ('```json\n{"status": "ok"}\n```', True, "valid JSON"),
        ('{"files": []}', False, "missing keys ['status']"),
        ("[1, 2]", False, "JSON but not an object"),
        ('{"status": NaN}', False, "not JSON (NaN is not JSON)"),
    ],
)
def test_json_is_one_object_with_the_keys(content, ok, detail):
    assert text.check_json(content, {"keys": ["status"]}, {}, NO_CALLS, {}) == (ok, detail)


@pytest.mark.parametrize("value", [["status"], "status", None])
def test_json_needs_its_keys_value(value):
    with pytest.raises(ValueError, match="keys"):
        text.check_json('{"status": "ok"}', value, {}, NO_CALLS, {})


@pytest.mark.parametrize(
    "content,bad",
    [
        ("Changed src/app.py and the docs.", []),
        ("Changed: `src/app.py`", []),  # a "Changed: " line is plain too
        ("Ran `pytest src/app.py` and it passed.", []),  # a command, not a mention
        ("```\nsrc/app.py\n```", []),
        ("Updated `src/other.py`.", []),  # not a changed file
        ("Updated `src/app.py`.", ["src/app.py is written in backticks, not in plain text"]),
        ("Updated **src/app.py**.", ["src/app.py is written in bold, not in plain text"]),
        ("See [app.py](src/app.py).", ["[app.py](src/app.py) is written in a markdown link, not in plain text"]),
    ],
)
def test_changed_file_marks_in_plain_text(content, bad):
    assert text.changed_file_marks(content, ["src/app.py"], "plain") == bad
