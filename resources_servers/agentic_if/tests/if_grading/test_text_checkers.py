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
        ("1) fix the parser 2) add tests", "sentences", 2),  # an inline numbered list
        # the spaces and line breaks inside the part count, the spaces around it do not
        ("  Fixed it.\n\nTests pass.  ", "characters", 22),
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
        text.check_length(reply, {"n": 4, "unit": "pages"}, {}, NO_CALLS, {})


SENTENCES_AND_WORDS = [{"n": 2, "unit": "sentences", "dir": "max"}, {"n": 50, "unit": "words", "dir": "max"}]
TWO_LONG_SENTENCES = " ".join(["word"] * 30) + ". " + " ".join(["word"] * 30) + "."


@pytest.mark.parametrize(
    "content,value,part,ok,detail",
    [
        # the recipe's max250_characters and max500_characters: the part's characters, code left out of the prose
        ("x" * 251, {"n": 250, "unit": "characters"}, "all", False, "251 characters (at most 250)"),
        (
            "Fixed the parser.\n```\n" + "x" * 600 + "\n```",
            {"n": 500, "unit": "characters", "dir": "max"},
            "prose",
            True,
            "17 characters (at most 500)",
        ),
        # max2_sentences_50_words: a list of bounds, each counted and all held
        (
            "Fixed the parser. All tests pass.",
            SENTENCES_AND_WORDS,
            "all",
            True,
            "2 sentences (at most 2); 6 words (at most 50)",
        ),
        (
            "Fixed. Tests pass. Changelog next.",
            SENTENCES_AND_WORDS,
            "all",
            False,
            "3 sentences (at most 2); 5 words (at most 50)",
        ),
        (TWO_LONG_SENTENCES, SENTENCES_AND_WORDS, "all", False, "2 sentences (at most 2); 60 words (at most 50)"),
    ],
)
def test_length_in_characters_and_with_a_list_of_bounds(content, value, part, ok, detail):
    assert text.check_length(content, value, {"part": part}, NO_CALLS, {}) == (ok, detail)


@pytest.mark.parametrize(
    "value",
    [[], [{"n": 2, "unit": "sentences", "dir": "min"}], [{"n": 2, "unit": "sentences", "dir": "max"}, "words"]],
)
def test_a_list_of_bounds_holds_only_upper_bounds(value):
    with pytest.raises(ValueError, match="one bound"):
        text.check_length("Fixed it.", value, {}, NO_CALLS, {})


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


# text_no_colon_before_tool_call's values end_with_full_stop and question_before_tool_call
NOT_A_FULL_STOP = {"end_before_call": "(?:[^.。\"'”’)\\]]|\\.\\.)[\"'”’)\\]]*"}
A_QUESTION_MARK = {"end_before_call": "[?？][\"'”’)\\]]*"}


@pytest.mark.parametrize(
    "value,content,ok",
    [
        (NOT_A_FULL_STOP, "Let me read the date parser.", True),
        (NOT_A_FULL_STOP, "**Plan: read the parser.**", True),  # closing emphasis marks are set aside
        (NOT_A_FULL_STOP, "Let me read the date parser:", False),
        (NOT_A_FULL_STOP, "Checking the logs...", False),  # an ellipsis is not a full stop
        (NOT_A_FULL_STOP, "Running `pytest -q`", False),
        (A_QUESTION_MARK, "Does the README cover it? Let me check.", True),  # an earlier question is fine
        (A_QUESTION_MARK, "Does the README cover it?", False),
        (A_QUESTION_MARK, "**Which parser is used?**", False),
    ],
)
def test_an_ending_banned_before_a_tool_call_is_read_at_the_end_of_a_message_that_calls_one(value, content, ok):
    assert text.check_element_absent(content, value, {}, CALLS, {})[0] is ok
    assert text.check_element_absent(content, value, {}, NO_CALLS, {}) == (True, "absent")


# code_only_when_asked's value indented_block
ANY_CODE_BLOCK = {"element": "any_code_block", "name": "a code block of any kind, fenced or indented"}


def test_end_before_call_reports_the_end_and_both_new_elements_are_only_bans():
    assert text.check_element_absent("Does the README cover it?", A_QUESTION_MARK, {}, CALLS, {}) == (
        False,
        "found ['Does the README cover it?']",
    )
    for value in (A_QUESTION_MARK, ANY_CODE_BLOCK):
        with pytest.raises(ValueError, match="only a ban"):
            text.check_element_present("Does the README cover it?\n```\nx = 1\n```", value, {}, CALLS, {})


@pytest.mark.parametrize(
    "content,found",
    [
        ("Here is the new check:\n\n    if year < 69:\n        year += 2000", "an indented code block"),
        ("The new check:\n```python\nyear += 2000\n```", "```"),
        ("The new check:\n> ```python\n> year += 2000\n> ```", "> ```"),  # a fence inside a quote
        ("The fix is ```year += 2000``` in parse_date().", "```year += 2000```"),  # triple backticks within a line
        ("The new check:\n<pre>\nyear += 2000\n</pre>", "<pre>"),
        # a list item's indented continuation is prose, and a stray fence mark within a line is no code
        ("- parse_date() changed\n\n    It now maps 00-68 to 20xx.", None),
        ("Changed:\n- src/utils/dates.py\n    - the century cutoff in parse_date()", None),
        ("I removed the stray ``` at the end of the README.", None),
    ],
)
def test_any_code_block_finds_fenced_indented_and_html_code(content, found):
    assert text.check_element_absent(content, ANY_CODE_BLOCK, {"part": "all"}, NO_CALLS, {}) == (
        (False, "found [%r]" % found) if found else (True, "absent")
    )


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
        # a message that is all code has no last sentence: it asks nothing
        ("no_question", "```\nprint(1)\n```", True),
        ("question", "```\nprint(1)\n```", False),
        # a sentence that is all quote is the question itself
        ("question", 'Fixed.\n\n"Why does it hang?"', True),
    ],
)
def test_ending(value, content, ok):
    assert text.check_ending(content, value, {}, NO_CALLS, {})[0] is ok


def test_unknown_ending_is_an_error():
    for value in ("closing", {"line": "TASK_COMPLETE"}):
        with pytest.raises(ValueError, match="ending"):
            text.check_ending("Done.", value, {}, NO_CALLS, {})


@pytest.mark.parametrize(
    "content,ok",
    [
        ("All 14 tests pass.", True),
        ("Fixed it.\n\n- a\n- b\n\nAll tests pass.", True),  # a list before the closing sentence
        ("Fixed it; all tests pass. ✅", True),  # emoji after the full stop
        ("**All tests pass.**", True),  # closing Markdown marks
        ("完成。", True),
        ("All tests pass", False),  # no closing mark
        ("", False),
    ],
)
def test_closing_sentence_is_a_last_line_of_prose_with_a_closing_mark(content, ok):
    assert text.check_ending(content, "closing_sentence", {}, NO_CALLS, {})[0] is ok


@pytest.mark.parametrize(
    "content,detail",
    [
        ("Fixed it.\n\n1. a.\n2. b.", "the last line is a list item: '2. b.'"),
        ("Fixed it:\n\n```bash\nmake test\n```", "the last line is code: '```'"),
        ("Fixed it.\n\n    make test", "the last line is code: 'make test'"),
        ("| a | b |\n|---|---|\n| 1 | 2 |", "the last line is a table row: '| 1 | 2 |'"),
        ("Fixed it.\n\n## Next steps", "the last line is a heading or a rule: '## Next steps'"),
        ("Fixed.\n\n---", "the last line is a heading or a rule: '---'"),
        ("> All tests pass.", "the last line is a quote: '> All tests pass.'"),
        ("- item\n  continued here.", "the last line is an indented line: 'continued here.'"),
    ],
)
def test_closing_sentence_fails_a_last_line_that_is_not_prose(content, detail):
    assert text.check_ending(content, "closing_sentence", {}, NO_CALLS, {}) == (False, detail)


@pytest.mark.parametrize(
    "content,ok",
    [
        ("Report.\nTASK_COMPLETE", True),
        ("Report.\n\nTASK_COMPLETE\n", True),
        ("Report. TASK_COMPLETE", False),  # not a line of its own
        ("Report.\n**TASK_COMPLETE**", False),  # marks around it
        ("Report.\nTASK_COMPLETE.", False),
        ("Report.\nTASK_COMPLETE\nThanks.", False),  # a line after it
        ("", False),
    ],
)
def test_last_line_is_the_set_line_written_exactly(content, ok):
    assert text.check_ending(content, {"last_line": "TASK_COMPLETE"}, {}, NO_CALLS, {})[0] is ok


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


def test_indented_code_is_code_unless_it_continues_a_list_item():
    rm = {"regex": "rm -rf"}
    assert text.check_element_absent("Run:\n\n    rm -rf build\n\nDone.", rm, {}, NO_CALLS, {}) == (True, "absent")
    assert text.check_element_absent("Run:\n\n    rm -rf build", rm | {"part": "all"}, {}, NO_CALLS, {})[0] is False
    assert text.check_element_absent("- Clean the build:\n\n    rm -rf build", rm, {}, NO_CALLS, {}) == (
        False,
        "found ['rm -rf']",
    )


def test_unknown_parts_and_elements_are_errors_not_verdicts():
    with pytest.raises(ValueError, match="unknown part 'code'"):
        text.check_length("Done.", {"n": 4, "unit": "words"}, {"part": "code"}, NO_CALLS, {})
    with pytest.raises(ValueError, match="unknown element 'md_fancy'"):
        text.check_element_absent("Done.", "md_fancy", {}, NO_CALLS, {})
    for value in (5, {"name": "no regex"}):
        with pytest.raises(ValueError, match="an element is a name or"):
            text.check_element_present("Done.", value, {}, NO_CALLS, {})
    # colon_before_tool_call reads the whole message and has no pattern of its own
    assert text.element_regex("colon_before_tool_call") == (None, "all", "colon_before_tool_call")


@pytest.mark.parametrize(
    "content,changed,mark,bad",
    [
        ("Changed app.py.", ["src/app.py"], "backticks", ["app.py is written in plain text, not in backticks"]),
        (
            "Changed <code>src/app.py</code>.",
            ["src/app.py"],
            "plain",
            ["src/app.py is written in <code> tags, not in plain text"],
        ),
        ("See file:///w/src/app.py for the change.", ["src/app.py"], "file_uri", []),
        (
            "See file:///w/src/app.py for the change.",
            ["src/app.py"],
            "plain",
            ["file:///w/src/app.py is written as a file:// address, not in plain text"],
        ),
        ("See [the file](file:///w/src/app.py).", ["src/app.py"], "file_uri", []),
        ("See [the file](file:///w/src/app.py).", ["src/app.py"], "link", []),
        ("- Changed: src/app.py", ["src/app.py"], "changed_line", []),
        (
            "- Changed: src/app.py",
            ["src/app.py"],
            "backticks",
            ['src/app.py is written on a line of its own that starts with "Changed: ", not in backticks'],
        ),
        # a web address, a markup tag, a route, a date, and/or and a library name are no mention of a changed file
        ("See https://github.com/org/repo/blob/main/src/app.py for the change.", ["src/app.py"], "backticks", []),
        ("See www.example.com/src/app.py for the change.", ["src/app.py"], "backticks", []),
        ('<code_snippet path="src/app.py" mode="EXCERPT">', ["src/app.py"], "backticks", []),
        ("Changed `src/app.py`; GET /api/users now returns 404.", ["src/app.py", "api/users"], "backticks", []),
        ("Changed `src/app.py`; the docs and/or tests on 2026/10/23.", ["src/app.py", "or", "10/23"], "backticks", []),
        ("Built on Three.js; changed `src/app.py`.", ["src/app.py", "vendor/Three.js"], "backticks", []),
    ],
)
def test_changed_file_marks_for_each_mark(content, changed, mark, bad):
    assert text.changed_file_marks(content, changed, mark) == bad


@pytest.mark.parametrize(
    "content,name_mark,location_mark,bad",
    [
        # the mark around the code name written right before a location
        (
            "The bug was in parse_date() src/utils/dates.py:42.",
            "backticks",
            None,
            ["the name parse_date() before src/utils/dates.py:42 is written in plain text, not in backticks"],
        ),
        (
            "Changed **f()** a.py:1, <code>g()</code> b.py:2, ``h()`` c.py:3, *k()* d.py:4 and `m()` at e.py#L5.",
            "plain",
            None,
            [
                "the name f() before a.py:1 is written in bold, not in plain text",
                "the name g() before b.py:2 is written in <code> tags, not in plain text",
                "the name h() before c.py:3 is written in double backticks, not in plain text",
                "the name k() before d.py:4 is written in italics, not in plain text",
                "the name m() before e.py#L5 is written in backticks, not in plain text",
            ],
        ),
        ("The bug was in ``parse_date()`` [src/utils/dates.py:42].", None, "brackets", []),
        # the name inside the parentheses is the one before the location when they do not open at it
        (
            "Fixed (see parse_date() src/utils/dates.py:42).",
            "backticks",
            None,
            ["the name parse_date() before src/utils/dates.py:42 is written in plain text, not in backticks"],
        ),
        # the mark around the location
        ("`parse_date()` (`src/utils/dates.py:42`) maps 00-68 to 20xx.", None, "parens_backticks", []),
        (
            "`parse_date()` (src/utils/dates.py:42) maps 00-68 to 20xx.",
            None,
            "parens_backticks",
            ["the location src/utils/dates.py:42 is written in parentheses, not in backticks inside parentheses"],
        ),
        (
            "The bug was in parse_date() src/utils/dates.py:42.",
            None,
            "bold",
            ["the location src/utils/dates.py:42 is written in plain text, not in bold"],
        ),
        # parentheses that enclose the location with more words inside
        ("`IPv4` declares `host` in `_fields` (src/objects.py, lines 612 and 618).", None, "parens", []),
        (
            "documented at src/objects.py, line 109.",
            None,
            "parens",
            ["the location src/objects.py, line 109 is written in plain text, not in parentheses"],
        ),
        ("`parse_date()` ([dates.py:42](src/utils/dates.py#L42)) maps 00-68 to 20xx.", None, "parens_link", []),
        # a link location is the target of a link whose text is the name in backticks
        ("[`parse_date()`](src/utils/dates.py:42) maps 00-68 to 20xx.", None, "link", []),
        (
            "`parse_date()` at src/utils/dates.py:42 maps 00-68 to 20xx.",
            None,
            "link",
            [
                "the location src/utils/dates.py:42 is written in plain text, not as the target of a link whose text"
                " is in backticks"
            ],
        ),
        (
            "[parse_date()](src/utils/dates.py:42) maps 00-68 to 20xx.",
            None,
            "link",
            ["the link [parse_date()](src/utils/dates.py:42) does not have its text in backticks"],
        ),
        # left aside: a command in backticks, a label in bold, a fenced block, a web address
        ("Ran `pytest tests/test_dates.py:12`; **Fix:** src/utils/dates.py:42.", "backticks", None, []),
        ("```\nparse_date() src/utils/dates.py:42\n```\nSee https://x.org/a/b/dates.py#L4.", "backticks", "bold", []),
        ("Ran `python -m pytest tests/test_dates.py::test_two_digit_years -q`.", "plain", "link", []),
    ],
)
def test_code_reference_marks_for_each_mark(content, name_mark, location_mark, bad):
    assert text.code_reference_marks(content, name_mark=name_mark, location_mark=location_mark) == bad


@pytest.mark.parametrize(
    "marks,error",
    [({"name_mark": "underline"}, "unknown name mark 'underline'"), ({"location_mark": "curly"}, "unknown location")],
)
def test_an_unknown_code_reference_mark_is_an_error(marks, error):
    with pytest.raises(ValueError, match=error):
        text.code_reference_marks("Fixed it.", **marks)
