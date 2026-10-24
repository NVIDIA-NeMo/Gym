# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foreign-script letters: scripts from Unicode names, the counting rule and its exclusions, what the final reply is,
and what counts as the turn's input."""

import json
from typing import Any

import pytest

from resources_servers.agentic_turn_simulation.foreign_script import final_reply, foreign_script_letters, script_of


PERSIAN = "سلام، لطفاً گزارش را بررسی کن و نتیجه را بگو."
ARABIC = "مرحبا، من فضلك راجع التقرير وأخبرني بالنتيجة."
ENGLISH = "Please check the report and tell me the result."


def _user(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": text}


def _assistant(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _call(name: str, call_id: str = "c1", **args: Any) -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args)}


def _out(call_id: str, output: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def letters(reply: str, *history_texts: str, turn: list[dict[str, Any]] | None = None, **kw: Any) -> int:
    """The count for a turn that ends on `reply`, after `turn`, answering user messages `history_texts`."""
    history = [_user(text) for text in history_texts]
    return foreign_script_letters([*(turn or []), _assistant(reply)], history=history, stop_tools=("ask_user",), **kw)


class TestScripts:
    @pytest.mark.parametrize(
        "char, script",
        [
            ("a", "Latin"),
            ("é", "Latin"),
            ("Ａ", "Latin"),  # fullwidth
            ("ж", "Cyrillic"),
            ("ب", "Arabic"),
            ("ﺏ", "Arabic"),  # a presentation form
            ("ی", "Arabic"),  # Persian yeh
            ("中", "Han"),
            ("の", "Kana"),
            ("ー", "Kana"),  # the prolonged sound mark
            ("ｱ", "Kana"),  # halfwidth
            ("한", "Hangul"),
            ("ह", "Devanagari"),
            ("α", "Greek"),
            ("\u00b5", None),  # micro sign
            ("\u2126", None),  # ohm sign
            ("𝑥", None),  # math italic
            ("ʼ", None),  # modifier letter
            ("々", None),  # ideographic iteration mark
            ("1", None),
            ("،", None),
        ],
    )
    def test_a_letters_script_is_the_first_word_of_its_unicode_name(self, char: str, script: str | None) -> None:
        assert script_of(char) == script


class TestCounting:
    def test_cyrillic_spliced_into_an_arabic_reply_counts(self) -> None:
        reply = "راجعت التقرير وكل шесть الاختبارات نجحت بدون أخطاء."
        assert letters(reply, ARABIC) == 5

    def test_chinese_inside_a_persian_reply_counts(self) -> None:
        reply = "گزارش را بررسی کردم و 检查结果 مشکلی ندارد؛ همه چیز درست است."
        assert letters(reply, PERSIAN) == 4

    def test_every_foreign_script_adds_its_letters(self) -> None:
        reply = "گزارش را بررسی کردم و 检查结果 مشکلی ندارد، и все نتیجه‌ها خوب است."
        assert letters(reply, PERSIAN) == 4 + 4

    def test_a_script_with_fewer_than_three_letters_does_not_count(self) -> None:
        assert letters("گزارش را بررسی کردم و 文件 مشکلی ندارد.", PERSIAN) == 0

    def test_a_clean_reply_and_an_empty_one_count_nothing(self) -> None:
        assert letters("گزارش را بررسی کردم و مشکلی ندارد.", PERSIAN) == 0
        assert letters("", PERSIAN) == 0

    def test_a_script_the_input_shows_does_not_count(self) -> None:
        # The user wrote one Cyrillic word, so Cyrillic in the reply is not foreign.
        assert letters("راجعت التقرير وكل шесть الاختبارات نجحت.", ARABIC + " (шесть)") == 0

    def test_latin_never_counts(self) -> None:
        assert letters("Готово: запустил npm install, build passed, все тесты зелёные.", "Собери проект.") == 0

    def test_greek_counts_only_with_a_word_of_three_letters(self) -> None:
        assert letters("The ratio α/β times γ is below one.", ENGLISH) == 0
        assert letters("I checked it and το αρχείο is ready to send.", ENGLISH) == 2 + 6


class TestLegitimateScripts:
    @pytest.mark.parametrize(
        "reply",
        [
            "عبارت «你好世界» یعنی «سلام دنیا».",
            'عبارت "你好世界" یعنی سلام دنیا.',
            "عبارت 「你好世界」 یعنی سلام دنیا.",
            "متن اصلی:\n> 你好世界，这是测试\nیعنی سلام دنیا.",
        ],
        ids=["guillemets", "straight-quotes", "corner-brackets", "block-quote"],
    )
    def test_a_quoted_phrase_in_another_script_does_not_count(self, reply: str) -> None:
        assert letters(reply, PERSIAN) == 0

    def test_the_same_phrase_unquoted_counts(self) -> None:
        assert letters("عبارت 你好世界 یعنی سلام دنیا.", PERSIAN) == 4

    @pytest.mark.parametrize(
        "reply",
        [
            "این دستور را اجرا کنید:\n```python\nprint('Привет мир')\n```\nخروجی را ببینید.",
            "متغیر `名前の値` را تغییر دادم.",
            "این دستور را اجرا کنید:\n```\nprint('Привет мир')\nبقیه پاسخ هم داخل کد ماند",
            "صفحه https://例え.テスト/パス را ببینید.",
            "به 用户@例子.中国 ایمیل بزنید.",
            "برچسب <span title='标签'> را اضافه کردم.",
        ],
        ids=["fenced-block", "inline-code", "open-fence", "url", "email", "tag"],
    )
    def test_code_urls_and_markup_do_not_count(self, reply: str) -> None:
        assert letters(reply, PERSIAN) == 0

    def test_a_translation_into_a_script_the_input_shows_does_not_count(self) -> None:
        request = "این جمله را به روسی ترجمه کن، مثل نمونه: Добрый день = روز بخیر"
        assert letters("ترجمه: Спасибо за помощь یعنی ممنون از کمکت.", request) == 0

    def test_a_translation_on_lines_of_its_own_does_not_count(self) -> None:
        # The target script appears nowhere in the input, but the reply writes it as whole lines.
        reply = "ترجمه به ژاپنی:\n報告書を確認しました。\n問題はありません。"
        assert letters(reply, "این را به ژاپنی ترجمه کن: گزارش را بررسی کردم.") == 0

    def test_a_reply_wholly_in_another_language_does_not_count(self) -> None:
        assert letters("Отчёт проверен, всё в порядке.", ARABIC) == 0

    def test_han_is_japanese_when_the_input_has_kana(self) -> None:
        reply = "Updated config.yaml and settings.json: 設定を更新しました"
        assert letters(reply, "せっていを こうしんして ください") == 0
        assert letters(reply, "Please update the settings.") == 4 + 5  # Han and kana, both foreign


class TestFinalReply:
    def test_only_the_last_model_reply_counts(self) -> None:
        turn = [_assistant("پیش‌نویس: 检查结果 را می‌سازم."), _call("bash", command="ls"), _out("c1", "ok")]
        assert letters("گزارش آماده است.", PERSIAN, turn=turn) == 0
        assert letters("گزارش 检查结果 آماده است.", PERSIAN, turn=turn) == 4

    def test_a_stop_tools_question_is_part_of_the_reply_one_line_per_object(self) -> None:
        options = [{"label": "اول", "description": "گزارش 检查结果 کامل"}, {"label": "دوم", "description": "خلاصه"}]
        question = _call("ask_user", questions=[{"id": "q1", "question": "کدام را می‌خواهید؟", "options": options}])
        turn = [_call("bash", "c0", command="ls"), question, _out("c0", "ok")]  # a call made beside it ran
        assert final_reply(turn, ("ask_user",)) == "q1 - کدام را می‌خواهید؟\nاول - گزارش 检查结果 کامل\nدوم - خلاصه"
        assert foreign_script_letters(turn, history=[_user(PERSIAN)], stop_tools=("ask_user",)) == 4

    def test_the_arguments_of_other_calls_are_not_the_reply(self) -> None:
        turn = [_call("write_file", path="a.txt", content="گزارش 检查结果 آماده است")]
        assert final_reply(turn, ("ask_user",)) == ""
        assert foreign_script_letters(turn, history=[_user(PERSIAN)], stop_tools=("ask_user",)) == 0


class TestInput:
    REPLY = "نتیجه: 检查结果 درست است و مشکلی ندارد."

    def test_the_turns_tool_results_are_input(self) -> None:
        turn = [_call("bash", command="cat report.txt"), _out("c1", "检查结果: 通过")]
        assert letters(self.REPLY, PERSIAN, turn=turn) == 0

    def test_a_json_escape_is_not_a_letter(self) -> None:
        # The policy wrote the garbled text into a file and read it back as escaped JSON: no excuse for the reply.
        written = "نتیجه: 检查结果 درست است"
        turn = [
            _call("write_file", "c1", path="report.md", content=written),
            _out("c1", "ok"),
            _call("read_file", "c2", path="report.md"),
            _out("c2", json.dumps({"content": written})),
        ]
        assert "\\u68c0" in turn[3]["output"] and letters(self.REPLY, PERSIAN, turn=turn) == 4

    def test_the_instructions_and_every_history_item_are_input(self) -> None:
        assert letters(self.REPLY, PERSIAN, instructions="Reply in the user's language. 检查") == 0
        call = {"type": "function_call", "call_id": "h1", "name": "read_file", "arguments": '{"path": "检查.txt"}'}
        history = [_user(PERSIAN), call, _out("h1", "ok")]
        assert foreign_script_letters([_assistant(self.REPLY)], history=history, stop_tools=()) == 0

    def test_reasoning_is_not_input(self) -> None:
        reasoning = {"type": "reasoning", "summary": [{"type": "summary_text", "text": "检查结果"}]}
        turn = [_assistant(self.REPLY)]
        assert foreign_script_letters(turn, history=[_user(PERSIAN), reasoning], stop_tools=()) == 4
