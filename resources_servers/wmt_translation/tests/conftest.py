# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from .flores200_model import prepare_flores200_model, verify_flores200_model


@pytest.fixture(scope="session")
def flores200_tokenizer_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Keep real tokenizer/scoring coverage without SacreBLEU's expired download endpoint."""
    from sacrebleu.tokenizers import tokenizer_spm

    supplied_cache = os.environ.get("SACREBLEU")
    if supplied_cache is not None:
        if not supplied_cache.strip():
            raise ValueError("SACREBLEU must name a nonempty tokenizer cache directory")
        cache_root = Path(supplied_cache)
        model = verify_flores200_model(cache_root)
    else:
        cache_root = tmp_path_factory.mktemp("wmt-sacrebleu")
        model = prepare_flores200_model(cache_root)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(tokenizer_spm, "SACREBLEU_DIR", str(cache_root))
        yield model
