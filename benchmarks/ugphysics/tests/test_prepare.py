# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``benchmarks/ugphysics/prepare.py``.

Covers the pure helper functions (no HuggingFace download). These
mirror Skills' ``nemo_skills/dataset/ugphysics/prepare.py`` exactly,
so any divergence here breaks parity.
"""

from benchmarks.ugphysics.prepare import (
    OB_ANS_TYPE_ID2EN,
    SUBSETS,
    format_entry,
    get_boxed_answer_example,
    get_prompt_sentence,
)


class TestPromptSentence:
    def test_single_answer_basic_types(self) -> None:
        # All seven Skills answer-type tags map to the canonical English label.
        for tag, en in OB_ANS_TYPE_ID2EN.items():
            assert get_prompt_sentence(tag, is_multiple_answer=False) == f"The answer of the problem should be {en}."

    def test_multi_answer_homogeneous(self) -> None:
        # All-same answer types: "each of them should be ...".
        s = get_prompt_sentence("NV,NV", is_multiple_answer=True)
        assert s == "The problem has multiple answers, each of them should be a numerical value without units."

    def test_multi_answer_heterogeneous(self) -> None:
        # Mixed answer types: "with the answers in order being ..., ...".
        s = get_prompt_sentence("NV,EX", is_multiple_answer=True)
        assert s == (
            "The problem has multiple answers, with the answers in order being "
            "a numerical value without units, an expression."
        )

    def test_unknown_tag_passes_through(self) -> None:
        # Skills' implementation falls through to the raw tag if unknown.
        s = get_prompt_sentence("UNKNOWN", is_multiple_answer=False)
        assert s == "The answer of the problem should be UNKNOWN."


class TestBoxedExample:
    def test_single(self) -> None:
        assert get_boxed_answer_example(False) == r"\boxed{answer}(unit)"

    def test_multiple(self) -> None:
        assert get_boxed_answer_example(True) == r"\boxed{multiple answers connected with commas}"


class TestFormatEntry:
    def test_renames_and_keeps_metadata(self) -> None:
        # Pretend HF row, structurally equivalent to a real ugphysics
        # entry. Keys mirror the upstream UGPhysics dataset schema.
        entry = {
            "index": 42,
            "problem": "A ball...",
            "answers": "10",
            "solution": "v = sqrt(2gh)",
            "answer_type": "NV",
            "subject": "ClassicalMechanics",
            "language": "EN",  # Skills lowercases this
            "is_multiple_answer": False,
        }
        out = format_entry(entry)
        # Skills→Gym renames.
        assert out["question"] == "A ball..."
        assert out["expected_answer"] == "10"
        assert out["subject"] == "ClassicalMechanics"
        # Forwarded fields.
        assert out["index"] == 42
        assert out["solution"] == "v = sqrt(2gh)"
        assert out["answer_type"] == "NV"
        assert out["language"] == "en"
        assert out["is_multiple_answer"] is False
        # Derived prompt fields.
        assert out["prompt_sentence"] == "The answer of the problem should be a numerical value without units."
        assert out["boxed_answer_example"] == r"\boxed{answer}(unit)"

    def test_multi_answer(self) -> None:
        entry = {
            "index": 7,
            "problem": "Two-part problem.",
            "answers": "a; b",
            "solution": "...",
            "answer_type": "EX,NV",
            "subject": "QuantumMechanics",
            "language": "en",
            "is_multiple_answer": True,
        }
        out = format_entry(entry)
        assert out["boxed_answer_example"] == r"\boxed{multiple answers connected with commas}"
        assert "in order being" in out["prompt_sentence"]


class TestSubsets:
    def test_thirteen_subjects(self) -> None:
        # The UGPhysics dataset has 13 subjects; Skills uses these for
        # `subset_for_metrics`. Order matters — keep stable for stable
        # Tier-2 metric ordering across runs.
        assert len(SUBSETS) == 13
        # Spot check a representative sample.
        for s in (
            "AtomicPhysics",
            "ClassicalElectromagnetism",
            "QuantumMechanics",
            "Thermodynamics",
            "WaveOptics",
        ):
            assert s in SUBSETS
