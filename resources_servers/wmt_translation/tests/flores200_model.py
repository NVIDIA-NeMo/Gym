# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare WMT's test tokenizer from a pinned Meta NLLB SentencePiece artifact.

Its digest matches independently hosted FLORES-200-named copies; the canonical endpoint's
expired certificate prevents a direct byte comparison. Meta's model card declares the
remote artifact CC-BY-NC-4.0. The artifact is downloaded for tests, never vendored here.
"""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


MODEL_URL = (
    "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/"
    "f8d333a098d19b4fd9a8b18f94170487ad3f821d/sentencepiece.bpe.model"
)
# Public artifact checksum, verified against Meta's LFS metadata.
MODEL_SHA256 = "14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a"  # pragma: allowlist secret
MODEL_SIZE = 4_852_054
MODEL_FILENAME = "flores200sacrebleuspm"


def _validate_model(data: bytes) -> None:
    if len(data) != MODEL_SIZE or hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("FLORES-200 tokenizer size or SHA-256 mismatch")


def verify_flores200_model(cache_root: Path) -> Path:
    """Read and verify an existing model without changing the supplied cache."""
    model = cache_root / "models" / MODEL_FILENAME
    if model.stat().st_size != MODEL_SIZE:
        raise ValueError("FLORES-200 tokenizer size or SHA-256 mismatch")
    _validate_model(model.read_bytes())
    return model


def prepare_flores200_model(cache_root: Path) -> Path:
    """Verify or atomically populate SacreBLEU's FLORES-200 model cache for tests."""
    model = cache_root / "models" / MODEL_FILENAME
    if model.exists():
        return verify_flores200_model(cache_root)

    # SacreBLEU's TinyURL redirects to a Meta endpoint whose certificate expired.
    with urlopen(MODEL_URL, timeout=30) as response:
        data = response.read(MODEL_SIZE + 1)
    _validate_model(data)

    model.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(prefix=".flores200-", dir=model.parent, delete=False)
    staged = Path(temporary.name)
    try:
        with temporary:
            temporary.write(data)
        staged.replace(model)
    finally:
        staged.unlink(missing_ok=True)
    return model
