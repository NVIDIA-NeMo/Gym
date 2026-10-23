# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The language checker: the user's language, the verdict code can reach, and when the judge decides instead."""

import pytest

from resources_servers.agentic_if.if_grading.checkers import language


PASTED = (
    "```\n$ pytest -q\n```\n"
    "Traceback (most recent call last):\n"
    '  File "app.py", line 3, in <module>\n'
    "KeyError: 'name'\n"
    "    an indented line\n"
    "> a quoted line\n"
    "2026-01-01 10:00:00 INFO started\n"
    "$ make test\n"
    "src/app.py:12: error: bad type\n"
    "tests/test_app.py::test_name FAILED\n"
    "diff --git a/x b/x\n"
)


def test_own_paragraphs_set_pasted_output_aside():
    message = "Me sale este error cuando corro las pruebas:\n" + PASTED + "¿Puedes arreglarlo?"
    assert language.own_paragraphs(message) == ["Me sale este error cuando corro las pruebas:", "¿Puedes arreglarlo?"]
    assert language.own_paragraphs(PASTED) == []
    assert language.own_paragraphs(None) == []


@pytest.mark.parametrize(
    "message,expected,detail",
    [
        # a Spanish user who pastes an English traceback writes Spanish
        (
            "Me sale este error cuando corro las pruebas:\n" + PASTED + "¿Puedes arreglarlo?",
            "spanish",
            "(paragraph 1 of 2 of the user's own words)",
        ),
        ("Fix it.\n\nAnd the tests.", "english", "(paragraph 2 of 2 of the user's own words)"),
        # no paragraph alone tells the language; read together they do
        ("Fix the parser.\n\nRun it.", "english", "(all of the user's own words)"),
        (PASTED, None, "no words of the user's own: only pasted text (code, output)"),
        ("ok, sigue", None, "Latin script, too few function words to tell the language"),
        ("Hola.\n\nGracias, el test.", None, "Latin script, too few function words to tell the language"),
    ],
    ids=[
        "spanish_with_traceback",
        "second_paragraph",
        "all_paragraphs",
        "only_pasted",
        "too_short",
        "too_few_together",
    ],
)
def test_the_user_language_is_read_from_their_own_words(message, expected, detail):
    found, why = language.detect(message)
    assert found == expected
    assert why.endswith(detail)


@pytest.mark.parametrize(
    "text,expected,detail",
    [
        ("Привет, как дела? Это тест.", "russian", "Cyrillic script"),
        ("Привіт, як справи? Це їжак.", "ukrainian", "Cyrillic script"),
        ("これはテストです。", "japanese", "kana"),
        ("这是一个测试。", "chinese", "han script"),
        ("이것은 테스트입니다.", "korean", "hangul script"),
        ("هذا اختبار", "arabic", "arabic script"),
        ("यह एक परीक्षण है", "hindi", "devanagari script"),
        ("นี่คือการทดสอบ", "thai", "thai script"),
        ("זה מבחן", "hebrew", "hebrew script"),
        ("Αυτό είναι ένα τεστ", "greek", "greek script"),
        ("The bug is fixed and the tests pass in the repo.", "english", "function words {'en': 6, 'nl': 2}"),
        ("ეს არის ტესტი", None, "script other"),  # Georgian: a script no language here is written in
        ("12345 !!!", None, "no letters"),
        # a tie is no reading: max() would pick whichever language is listed first
        ("Le pipeline CI échoue pendant l'analyse de sécurité", None, "Latin script, function words tie"),
    ],
)
def test_read_language_tells_scripts_and_latin_languages_apart(text, expected, detail):
    found, why = language.read_language(text)
    assert found == expected
    assert why.startswith(detail)


@pytest.mark.parametrize(
    "value,text,ok,detail",
    [
        ("Chinese", "`src/app.py` ```\nprint(1)\n```", None, "no prose (only code, paths or symbols)"),
        ("Chinese", "/usr/bin/python3 --version", None, "no prose (only code, paths or symbols)"),
        ("Chinese", "修复了 `修复`", True, "ok (5 of 5 words in chinese)"),  # the language's words in code count
        ("Chinese", "已修复。Это тест.", False, "the prose mixes in cyrillic"),
        ("Japanese", "已修复问题。", False, "no kana: not Japanese"),
        ("Japanese", "修正しました。", True, "ok (6 of 6 words in japanese)"),
        ("Ukrainian", "Привет, как дела? Это тест.", False, "Cyrillic without the letters only Ukrainian uses"),
        ("Russian", "Привіт, як справи? Це їжак.", False, "Ukrainian letters in the text"),
        (
            "Chinese",
            "已修复。The tests pass now and the build is green.",
            False,
            "a sentence in a Latin-script language: 'The tests pass now and the build is green.'",
        ),
        ("Chinese", "Fixed the bug 已修复", False, "3 of 6 words in the script of chinese (prose)"),
        (
            "English",
            "I fixed the bug. Corregí el error en el analizador y las pruebas pasan.",
            False,
            "a sentence in spanish: 'Corregí el error en el analizador y las pruebas pasan.'",
        ),
        ("English", "Fixed el bug in la parser de la app.", False, "english function words 1, another language 4"),
        ("English", "Fixed it.", True, "too few function words to tell the language (1)"),
        ("Spanish", "Corregí el error y las pruebas pasan ahora.", True, "ok (spanish function words 3 of 3)"),
    ],
)
def test_check_verdicts_by_script_and_function_words(value, text, ok, detail):
    verdict, why, _share = language.check(value, text)
    assert verdict is ok
    assert why.startswith(detail)


def test_an_unknown_language_is_an_error_not_a_verdict():
    with pytest.raises(ValueError, match="unknown language 'klingon'"):
        language.check("Klingon", "Done.")


@pytest.mark.parametrize(
    "value,text,ok,clear,detail",
    [
        ("Chinese", "```\nprint(1)\n```", None, True, "no prose (only code, paths or symbols): nothing to grade"),
        ("Chinese", "已修复。", True, True, "ok (3 of 3 words in chinese)"),
        ("Spanish", "Corregí el error y las pruebas pasan ahora.", True, True, "ok (spanish function words 3 of 3)"),
        # a fail with no word of the language is clear; one with some of its words goes to the judge
        ("Ukrainian", "I fixed the bug in the parser.", False, True, "a sentence in a Latin-script language"),
        ("Chinese", "Fixed the bug 已修复", False, False, "3 of 6 words in the script of chinese"),
        ("English", "Fixed it.", True, False, "too few function words to tell the language (1)"),
        # Thai has no full stops: a run of Latin words can be a sentence in another language
        (
            "Thai",
            "นี่คือการทดสอบที่ผ่านแล้วและทุกอย่างเรียบร้อยดีครับ see the logs",
            True,
            False,
            "ok (38 of 41 words in thai); a run of Latin words",
        ),
        # code sets inline code aside, so a sentence in backticks is for the judge to read
        (
            "Chinese",
            "已修复。`All of the split tests pass now.`",
            True,
            False,
            "ok (3 of 3 words in chinese); a sentence of words in backticks",
        ),
    ],
)
def test_assess_sends_every_uncertain_text_to_the_judge(value, text, ok, clear, detail):
    assessment = language.assess(value, text)
    assert (assessment["ok"], assessment["clear"]) == (ok, clear)
    assert assessment["detail"].startswith(detail)


def test_sentence_in_code_ignores_commands_and_code_blocks():
    assert language.sentence_in_code("Done. `All of the tests pass now.`") == "`All of the tests pass now.`"
    assert language.sentence_in_code("Run `make test` then `git status`.") is None
    assert language.sentence_in_code("```\n`All of the tests pass now.`\n```") is None
