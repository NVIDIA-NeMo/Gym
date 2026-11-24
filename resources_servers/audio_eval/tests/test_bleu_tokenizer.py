# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BLEU tokenizer selection must resolve every target-language spelling the benchmarks emit."""

import pytest

from resources_servers.audio_eval._scoring.evaluation.evaluator.audio import resolve_bleu_tokenize


@pytest.mark.parametrize(
    ("tgt_lang", "expected"),
    [
        ("zh-CN", "zh"),  # CoVoST2
        ("cmn_hans_cn", "zh"),  # FLEURS
        ("ja_jp", "ja-mecab"),
        ("ko", "ko-mecab"),
        ("de_de", "13a"),
        (None, "13a"),
    ],
)
def test_resolve_bleu_tokenize(tgt_lang: str | None, expected: str) -> None:
    assert resolve_bleu_tokenize(tgt_lang) == expected
