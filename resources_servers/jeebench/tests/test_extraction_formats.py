# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.jeebench.scoring import extract_answer


@pytest.mark.parametrize(
    "text,kind,expected",
    [
        ("अंतिम उत्तर: 1.78", "Numeric", "1.78"),
        ("**अंतिम उत्तर : [C] Zn धातु और NaOH(aq)**", "MCQ", "C"),
        ("Final Answer: (A) और (C)", "MCQ(multiple)", "AC"),
        ("চূড়ান্ত উত্তর: (B)", "MCQ", "B"),
        ("চূড়ান্ত উত্তৰ: (B)", "MCQ", "B"),
        ("અંતિમ જવાબ: (B)", "MCQ", "B"),
        ("ಅಂತಿಮ ಉತ್ತರ: (B)", "MCQ", "B"),
        ("അന്തിമ ഉത്തരം: (B)", "MCQ", "B"),
        ("अन्तिम उत्तर: (B)", "MCQ", "B"),
        ("अंतिम उत्तरम्: (B)", "MCQ", "B"),
        ("ଅନ୍ତିମ ଉତ୍ତର: (B)", "MCQ", "B"),
        ("ਅੰਤਿਮ ਉੱਤਰ: (B)", "MCQ", "B"),
        ("இறுதி விடை: (B)", "MCQ", "B"),
        ("తుది సమాధానం: (B)", "MCQ", "B"),
        ("حتمی جواب: (B)", "MCQ", "B"),
        ("The correct option is (A).", "MCQ", "A"),
        (r"\boxed{0.20\ \text{bar}}", "Numeric", "0.2"),
        (r"\boxed{935.00\ \text{K}}", "Numeric", "935.0"),
        (r"\boxed{(B)\ 103}", "MCQ", "B"),
        (r"\boxed{\text{C) } \mathrm{Zn}\ \text{धातु और}\ \mathrm{NaOH}\ (aq)}", "MCQ", "C"),
        (r"\boxed{\text{Correct options: }A,\;B,\;D}", "MCQ(multiple)", "ABD"),
        (r"\boxed{\frac{16}{9}}", "Numeric", str(16 / 9)),
        (r"\boxed{6.75\times 10^{-5}}", "Numeric", "6.75e-05"),
        ("अंतिम उत्तर: १२", "Integer", "12"),
        (r"The minimum value of the function f(x) is $0$.", "Integer", "0"),
        (r"\[Y = 6.75\]", "Numeric", "6.75"),
        (r"$P_{B}^{0}\approx0.20\;{\rm bar}$", "Numeric", "0.2"),
        (r"Thus, $r^2 = 2$.", "Integer", "2"),
        ("Final Answer: (A) and (C) are correct.", "MCQ(multiple)", "AC"),
        ("Final Answer:\nExplanation.\n1.01", "Numeric", "1.01"),
        (r"\boxed{4}. Thus the answer is consistent.", "Integer", "4"),
        (r"\boxed{2³}", "Integer", "8"),
        (r"\boxed{2**3}", "Integer", "8"),
        (r"\boxed{C और D सही हैं; A और B गलत हैं।}", "MCQ(multiple)", "CD"),
        ("(A) सही है\n(B) गलत है\n(C) सही है\n(D) गलत है", "MCQ(multiple)", "AC"),
        ("সুতৰাং, সঠিক উত্তৰ হ'ল (B) ডায়ামেগনেটিক।", "MCQ(multiple)", "B"),
        ("সেয়ে, সঠিক উক্তিবোৰ (B) আৰু (C)।", "MCQ(multiple)", "BC"),
        ("সেয়ে, Y-ৰ মান হৈছে 4।", "Integer", "4"),
        ("সুতৰাং, চূড়ান্ত উত্তৰ হৈছে 2.14।", "Numeric", "2.14"),
        ("**Final answer (মোটা ফলাৰ আখৰত):** 2.00", "Numeric", "2.00"),
        (r"\boxed{\text{শুদ্ধ উক্তিশক্তিকেইটা হ’ল \textbf{(B)} আৰু \textbf{(C)}.}}", "MCQ(multiple)", "BC"),
        ("एक\u00adमात्र सत्य कथन (ii), (iii) और (iv) हैं।", "MCQ(multiple)", "BCD"),
        ("Hence A, B and C are correct, D is not.", "MCQ(multiple)", "ABC"),
        ("**Final answer:** 0. (0 is a non-negative integer.)", "Integer", "0"),
        ("**উত্তৰ:** ... সৰ্বনিম্ন সংখ্যা হ’ল **6**। (N=6.)", "Integer", "6"),
        (r"\boxed{20\%}", "Numeric", "20"),
        ("अंतिम उत्तर: → (A) और (B)", "MCQ(multiple)", "AB"),
        ("सही विकल्प हैं: [A], [B] और [C]; [D] गलत है।", "MCQ(multiple)", "ABC"),
        ("Final answer (two decimal places): –0.24 mm", "Numeric", "-0.24"),
    ],
)
def test_explicit_answer_formats(text, kind, expected):
    assert extract_answer(text, kind)[0] == expected


@pytest.mark.parametrize(
    "text,kind",
    [
        ("Answer: A\nFinal answer: (B) or (C)", "MCQ"),
        ("Final answer: A या B", "MCQ"),
        ("Final answer: A and C are incorrect", "MCQ(multiple)"),
        (r"\boxed{4}\nFinal answer: uncertain", "Integer"),
        (r"\boxed{0.20 or 0.30 bar}", "Numeric"),
        (r"\boxed{2 < x < 3}", "Numeric"),
        (r"\boxed{3/0}", "Numeric"),
        (r"\boxed{1e999}", "Numeric"),
        (r"\boxed{2^{9999}}", "Numeric"),
        (r"\boxed{0.9315\ \text{MeV}}", "Integer"),
        ("Somewhere A is mentioned, but I cannot determine the answer.", "MCQ"),
        ("(A) सही है\n(B) गलत है\n(C) सही है", "MCQ(multiple)"),
        ("(A) सही है\n(B) गलत है\n(C) सही है\n(D) गलत है\n(A) गलत है", "MCQ(multiple)"),
        (r"\boxed{A सही है; A गलत है।}", "MCQ(multiple)"),
        ("The largest determinant is 4.00. This is the maximum for\n$n=3$.", "Numeric"),
    ],
)
def test_unsupported_or_ambiguous_answers_remain_unparsed(text, kind):
    assert extract_answer(text, kind) == ("None", "unparsed")


def test_unique_exact_option_text_without_reference_answer():
    question = r"Value?" + "\n[A] " + r"$\frac{5}{21}$" + "\n[B] 3\n[C] 4\n[D] 5"
    assert extract_answer(r"\boxed{\dfrac{5}{21}}", "MCQ", question=question) == ("A", "option_text_match")
    assert extract_answer(r"\boxed{4}", "MCQ", question=question) == ("C", "option_text_match")
    assert extract_answer(r"\boxed{6}", "MCQ", question=question) == ("None", "unparsed")
    assert extract_answer(r"\boxed{3}", "MCQ", question=question.replace("[C] 4", "[C] 3")) == ("None", "unparsed")


def test_looping_prose_does_not_become_an_answer_prefix():
    text = "अतः, विकल्प (A) आयामरहितम् अस्ति। " * 3000
    assert extract_answer(text, "MCQ(multiple)") == ("None", "unparsed")
    assert extract_answer(text + "\nFinal answer: A and C", "MCQ(multiple)") == ("AC", "answer_marker")


def test_later_markers_are_considered_across_languages():
    assert extract_answer("अंतिम उत्तर: A\nFinal answer: B", "MCQ")[0] == "B"
    assert extract_answer("Final answer: B\nअंतिम उत्तर: A", "MCQ")[0] == "A"


@pytest.mark.parametrize(
    "text,kind,expected",
    [
        ("Final Answer: (C), as it yields a valid configuration.", "MCQ", "C"),
        ("Final Answer: (A), (B), (D) are correct; (C) is not correct.", "MCQ(multiple)", "ABD"),
        ("A is correct and B is correct", "MCQ(multiple)", "AB"),
        ("Final answer: A is correct and B is correct", "MCQ(multiple)", "AB"),
        ("either A or B is correct", "MCQ", "None"),
        ("Final answer: either A or B is correct", "MCQ(multiple)", "None"),
        ("Final answer: A is correct; A is not correct", "MCQ", "None"),
        ("Final answer: A is correct and B is correct", "MCQ", "None"),
        ("Final answer: (B) is not correct", "MCQ", "None"),
        ("Final answer: 1,000", "Integer", "None"),
        ("Final answer: 2, 3", "Integer", "None"),
        ("Final answer: The value is 2 + 3", "Integer", "5"),
        ("the value of m is 2 and the value of n is 5", "Integer", "None"),
        ("Final answer: The value is 2 + x", "Numeric", "None"),
        ("Final answer: 2; 3", "Numeric", "None"),
        ("Final answer: 2 widgets", "Numeric", "None"),
        ("Final answer: (2 + 3)", "Integer", "5"),
        ("Final answer: m = 2 and n = 5", "Integer", "None"),
        ("Final answer: m = 2 n = 5", "Integer", "None"),
        ("Final answer: The value of m is 2 and the value of n is 5", "Integer", "None"),
        ("Final answer: The value is 2 / x", "Numeric", "None"),
        ("Final answer: 2 kg and 3 m", "Numeric", "None"),
        ("Final answer: 2, the value is 5", "Integer", "None"),
        ("Final answer: 2; n = 5", "Integer", "None"),
        ("Final answer: (B) wrong because of symmetry", "MCQ", "None"),
        ("अंतिम उत्तर: (B) अभिक्रिया ऊष्माक्षेपी है", "MCQ", "B"),
        ("अंतिम उत्तर: (B) किया गया", "MCQ", "B"),
        ("حتمی جواب: (B) کیا گیا", "MCQ", "B"),
        ("अंतिम उत्तर: (A) या (B)", "MCQ", "None"),
        ("حتمی جواب: (A) یا (B)", "MCQ", "None"),
        ("अंतिम उत्तर: (B) सही नहीं है", "MCQ", "None"),
    ],
)
def test_review_regressions(text, kind, expected):
    answer, method = extract_answer(text, kind)
    assert answer == expected
    assert (method == "unparsed") == (expected == "None")


@pytest.mark.parametrize("word", ["अभिक्रिया", "किया", "दिया", "کیا", "دیا", "या\u0301", "\u0301یا", "या\u200d"])
def test_uncertainty_substrings_with_combining_marks_are_not_words(word):
    from resources_servers.jeebench.extraction import uncertain

    assert not uncertain(word)


@pytest.mark.parametrize("word", ["या", "अथवा", "नहीं", "یا", "either", "not"])
def test_standalone_uncertainty_words_with_punctuation(word):
    from resources_servers.jeebench.extraction import uncertain

    assert uncertain(f"({word})")
