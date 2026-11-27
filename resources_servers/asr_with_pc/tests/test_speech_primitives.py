# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the text normalizers, edit-count primitives and the generation-loop guardrail."""

import pytest

from resources_servers.asr_with_pc import speech_metrics as sm
from resources_servers.asr_with_pc.open_asr_normalizer import OpenASREnglishTextNormalizer


class TestCleanupGenerationLoops:
    def test_collapses_a_character_run_of_21_but_not_19(self) -> None:
        assert sm.cleanup_generation_loops("x" * 21 + " ok " + "y" * 20) == "x ok y"
        assert sm.cleanup_generation_loops("y" * 19) == "y" * 19

    def test_collapses_a_block_repeated_20_times_but_not_19(self) -> None:
        assert sm.cleanup_generation_loops("abc " * 20 + "end") == "abc end"
        assert sm.cleanup_generation_loops("abc " * 19 + "end") == "abc " * 19 + "end"

    def test_long_blocks_are_collapsed_up_to_the_4096_character_cap(self) -> None:
        # A 50-character block is far beyond the 20 characters a smaller cap would cover.
        block = "".join(chr(ord("a") + index % 26) for index in range(50)) + " "
        assert sm.cleanup_generation_loops(block * 25 + "end") == block + "end"

    def test_blocks_may_span_lines(self) -> None:
        assert sm.cleanup_generation_loops("line\n" * 25 + "z") == "line\nz"

    def test_leaves_ordinary_text_alone(self) -> None:
        assert sm.cleanup_generation_loops("hello world, hello again") == "hello world, hello again"


class TestOpenASRNormalizer:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("b b c is great", "bbc is great"),
            ("5 g network", "5g network"),
            # "a" / "i" runs need three letters to be joined.
            ("a b", "a b"),
            ("i a m here", "i am here"),
            ("Hmm, wi fi and e mail", "wifi and email"),
            ("she's here, ok", "she is here okay"),
        ],
    )
    def test_leaderboard_replacements(self, text: str, expected: str) -> None:
        assert OpenASREnglishTextNormalizer()(text) == expected

    def test_empty_reference_is_dropped(self) -> None:
        assert sm.evaluate_open_asr("", "hello") == {"wer": None, "is_correct": None, "text": "", "pred_text": "hello"}


class TestMultilingualNormalization:
    def test_digits_are_kept_for_locales_without_number_words(self) -> None:
        assert sm.normalize_multilingual_text("room 12 (noise) [x]", language="xx") == "room 12"

    def test_spacing_marks_become_separators(self) -> None:
        assert sm.normalize_multilingual_text("काम", language="hi") == "क म"
        assert sm.normalize_multilingual_text("काम", language="hi", remove_diacritics=True) == "क म"

    def test_ligatures_and_stroked_letters_fold_when_diacritics_are_removed(self) -> None:
        folded = sm.normalize_multilingual_text("Œuvre Ørsted Łódź", language="fr", remove_diacritics=True)
        assert folded == "oeuvre orsted lodz"

    def test_english_uses_the_whisper_normalizer(self) -> None:
        assert sm.normalize_multilingual_text("It's 5, Dr. X", language="en_us") == "it is 5 doctor x"

    def test_whitespace_modes(self) -> None:
        assert sm.normalize_cer_whitespace("a  b", "all") == "ab"
        assert sm.normalize_cer_whitespace("a  b", "keep") == "a  b"
        assert sm.normalize_cer_whitespace("  今  天 x y 你", "han_adjacent") == "今天x y你"

    def test_empty_reference_is_dropped(self) -> None:
        result = sm.evaluate_multilingual_asr("", "x", language="de")
        assert (result["wer"], result["cer"], result["is_correct"], result["pred_text"]) == (None, None, None, "x")

    def test_compound_pairs_are_joined_only_when_they_match(self) -> None:
        assert sm.normalize_compound_pairs("ice cream is cold", "icecream is cold") == ("icecream is cold",) * 2
        assert sm.normalize_compound_pairs("a b", "c d") == ("a b", "c d")


class TestCompoundHandling:
    def test_open_asr_scores_with_compound_merging(self) -> None:
        # The vendored normalizer does not know "foot ball"; only merge_compounds=True makes it equal "football".
        result = sm.evaluate_open_asr("a foot ball game", "a football game")
        assert result["wer_errors"] == 0

    def test_han_adjacent_mode_does_not_also_join_latin_compounds(self) -> None:
        # Compound normalization would join "ab cd" with "abcd"; the whitespace modes switch it off.
        kwargs = dict(language="de", cer_whitespace_mode="han_adjacent", normalize_compound=True)
        assert sm.evaluate_multilingual_asr("ab cd", "abcd", **kwargs)["wer_errors"] == 2
        assert sm.evaluate_multilingual_asr("ab cd", "abcd", language="de", normalize_compound=True)["wer_errors"] == 0

    def test_language_codes_use_their_language_part(self) -> None:
        assert [sm.base_language(code) for code in ("de_de", "de-DE", "EN-us", "", None)] == ["de", "de", "en", "", ""]


class TestMultilingualNormalizerParity:
    def test_symbol_and_diacritic_folding_follows_whisper_normalizer(self) -> None:
        # The helpers come from whisper-normalizer; with and without diacritic folding.
        assert (
            sm.normalize_multilingual_text("Fußball, Œuvre!", language="de", remove_diacritics=True)
            == "fussball oeuvre"
        )
        assert sm.normalize_multilingual_text("Fußball, Œuvre!", language="de") == "fußball œuvre"
