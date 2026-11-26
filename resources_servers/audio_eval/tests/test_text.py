# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared text helpers: the generation-loop guardrail, compound pairs and CER whitespace modes."""

import pytest

from resources_servers.audio_eval._scoring.text import (
    cleanup_generation_loops,
    normalize_cer_whitespace,
    normalize_compound_pairs,
    normalize_whitespace,
)


class TestCleanupGenerationLoops:
    def test_collapses_a_character_run_of_20_but_not_19(self) -> None:
        assert cleanup_generation_loops("x" * 21 + " ok " + "y" * 20) == "x ok y"
        assert cleanup_generation_loops("y" * 19) == "y" * 19

    def test_collapses_a_block_repeated_20_times_but_not_19(self) -> None:
        assert cleanup_generation_loops("abc " * 20 + "end") == "abc end"
        assert cleanup_generation_loops("abc " * 19 + "end") == "abc " * 19 + "end"

    def test_blocks_may_span_lines(self) -> None:
        assert cleanup_generation_loops("line\n" * 25 + "z") == "line\nz"

    def test_leaves_ordinary_text_alone(self) -> None:
        assert cleanup_generation_loops("hello world, hello again") == "hello world, hello again"


def test_whitespace_is_collapsed_and_none_is_empty() -> None:
    assert normalize_whitespace("  a \n\t b  ") == "a b"
    assert normalize_whitespace(None) == ""


def test_compound_pairs_are_joined_only_when_they_match() -> None:
    assert normalize_compound_pairs("ice cream is cold", "icecream is cold") == ("icecream is cold",) * 2
    assert normalize_compound_pairs("a b", "c d") == ("a b", "c d")


def test_cer_whitespace_modes() -> None:
    assert normalize_cer_whitespace("a  b", "keep") == "a  b"
    assert normalize_cer_whitespace("a  b", "all") == "ab"
    assert normalize_cer_whitespace("  今  天 x y 你", "han_adjacent") == "今天x y你"
    assert normalize_cer_whitespace("今 天 use AI", "han_adjacent") == "今天use AI"
    with pytest.raises(ValueError, match="Invalid CER whitespace mode"):
        normalize_cer_whitespace("x", "bogus")
