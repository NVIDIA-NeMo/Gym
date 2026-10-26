# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IN22-Conv entry point, so composing Gen and Conv needs no global subset override."""

from collections.abc import Sequence
from pathlib import Path

from benchmarks.indic.in22.prepare import prepare as prepare_in22


def prepare(
    output_fpath: str | Path | None = None,
    *,
    languages: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    question_ids: Sequence[str] | None = None,
) -> Path:
    """Prepare the pinned conversational subset using the shared IN22 adapter."""
    return prepare_in22(
        output_fpath,
        subset="conv",
        languages=languages,
        directions=directions,
        question_ids=question_ids,
    )
