# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the translation components' COMET encoder APIs without downloading weights."""

import math
from pathlib import Path

import pytest


comet_models = pytest.importorskip("comet.models")


@pytest.fixture
def comet_encoder(tmp_path: Path) -> Path:
    """Save a small local XLM-R tokenizer and config for actual CPU inference."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import XLMRobertaConfig, XLMRobertaTokenizerFast

    vocabulary = ["<s>", "<pad>", "</s>", "<unk>", "hello", "world", "hallo", "welt"]
    pieces = [(word, -1.0) for word in vocabulary]
    tokenizer = Tokenizer(models.Unigram(pieces, unk_id=3))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.RobertaProcessing(("</s>", 2), ("<s>", 0))
    XLMRobertaTokenizerFast(vocab=pieces, tokenizer_object=tokenizer, model_max_length=64).save_pretrained(tmp_path)
    XLMRobertaConfig(
        vocab_size=len(vocabulary),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=66,
    ).save_pretrained(tmp_path)
    return tmp_path


@pytest.mark.parametrize("model_name", ["ReferencelessRegression", "XCOMETMetric"])
def test_comet_predicts_with_component_transformers(comet_encoder: Path, model_name: str) -> None:
    """Run both COMET scoring paths through their real encoder and prediction loop."""
    import torch

    options = {"word_layer": 1} if model_name == "XCOMETMetric" else {}
    model = getattr(comet_models, model_name)(
        pretrained_model=str(comet_encoder),
        hidden_sizes=[8],
        load_pretrained_weights=False,
        local_files_only=True,
        **options,
    )
    model.eval()
    samples = [
        {"src": "hello world", "mt": "hallo welt", "ref": "hallo welt"},
        {"src": "hello", "mt": "hallo", "ref": "hallo"},
    ]
    with torch.inference_mode():
        prediction = model.predict(samples, gpus=0, accelerator="cpu", progress_bar=False, num_workers=0)
    assert len(prediction.scores) == len(samples)
    assert all(math.isfinite(score) for score in prediction.scores)
